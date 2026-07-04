# Distributed / multi-GPU training — design note

Design note for epic [#265](https://github.com/rjwalters/thrust/issues/265)
("Distributed / multi-GPU training"). This is the **design/ADR deliverable**,
not an implementation: it answers the epic's three open design questions with
citations to Burn 0.21 source, issues a go/no-go recommendation, and decomposes
the work into phased child issues.

- **Burn version:** 0.21.0 (`Cargo.toml`).
- **Method:** the answers below are grounded in (a) Thrust's own trainer code,
  (b) the Burn 0.21 crates as published on crates.io, inspected from the local
  registry and, where a crate was not cached locally, downloaded and read from
  `static.crates.io`. Version and publication facts were confirmed against the
  crates.io sparse index (`index.crates.io`).
- **Template:** mirrors [`FP16_FEASIBILITY.md`](./FP16_FEASIBILITY.md) — verdict
  first, then one section per open question, then the defer trigger and child
  issues.

---

## Verdict: implement the async actor-learner path (single-host first); defer synchronous multi-GPU DDP

**Recommendation: build a single-host, asynchronous actor-learner architecture
as the near-term distributed path, and defer synchronous data-parallel (DDP)
multi-GPU training.** The deferral is a *payoff* decision, not a *capability*
one — see the correction to Q3 below.

- The actor-learner path needs **no gradient synchronization at all**: actors
  are pure inference (no `AutodiffBackend`), only the single learner trains. It
  is coordinated with in-process channels (`crossbeam-channel`, already a
  `training`-feature dependency — `Cargo.toml:99,144`) and pairs directly with
  the V-trace off-policy correction tracked in
  [#263](https://github.com/rjwalters/thrust/issues/263).
- **Correction to the epic's starting premise:** synchronous DDP is *not*
  blocked by a missing Burn primitive. Burn 0.21 **does** ship a native
  collective library — `burn-collective 0.21.0`, published on crates.io — with
  `all_reduce` / `reduce` / `broadcast` over backend tensors, in both a
  single-host (`local`) and a multi-host (`global`) flavour. The reason to defer
  DDP is the **same workload-size crossover** that governs GPU and FP16 adoption
  in this repo, not an absent allreduce.
- Per [`BURN_BACKENDS.md`](./BURN_BACKENDS.md), at today's net sizes (4-input,
  64-unit MLPs) **CPU NdArray beats every GPU backend 4.4–9.5×** because kernel
  launch + host↔device transfer dominate. Multi-GPU DDP multiplies GPUs that are
  *already the slower option* at these sizes; it cannot pay off until a workload
  is large enough to want even one GPU. Actor-learner, by contrast, scales
  *environment throughput* (rollout collection), which is useful on CPU today.

**Re-evaluation trigger for DDP:** revisit synchronous multi-GPU data
parallelism when a large-net / CNN / high-parallelism workload lands (image-obs
envs, Snake CNN — the same trigger `BURN_BACKENDS.md` names for GPU adoption)
*and* per-device batches are large enough that a single GPU is already the
right backend. At that point `burn-collective`'s device-side allreduce (not a
host round-trip) is the mechanism to reach for.

---

## Q1 — Actor-learner (IMPALA-style) vs. synchronous data-parallel (DDP)?

**Recommendation: actor-learner first.** Both are now technically buildable on
Burn 0.21; they differ in coordination cost and in what they scale.

| Dimension | DDP-style (synchronous allreduce) | Actor-learner (async, IMPALA-style) |
| --- | --- | --- |
| Burn 0.21 support | **Native**: `burn-collective` `all_reduce` (see Q3) | No multi-device primitive required |
| Gradient sync mechanism | Device-side allreduce per update step | None — only the learner trains |
| Off-policy correction | Not needed (synchronous, on-policy) | **V-trace ([#263](https://github.com/rjwalters/thrust/issues/263)) required** for staleness |
| What it scales | Gradient compute across GPUs | Environment/rollout throughput across workers |
| Trainer-loop impact | Restructure `train_step` to allreduce grads before `optimizer.step` | Actors added *around* the existing trainer; learner loop mostly unchanged |
| Rust infra in-tree | — | `crossbeam-channel` (`Cargo.toml:99,144`) |
| Pays off at current sizes? | **No** — multiplies GPUs that lose to CPU 4.4–9.5× (`BURN_BACKENDS.md`) | **Yes** — more parallel envs help even on CPU |

**Why actor-learner is the near-term choice.** The PPO trainer holds its policy
in `Option<P>` and swaps it through Burn's move-through optimizer on every step
(`src/train/ppo/trainer.rs:79`, and the backward → `GradientsParams::from_grads`
→ `optimizer.step` sequence at `trainer.rs:240-245`). A DDP variant must splice
a collective allreduce of the `GradientsParams` in between backward and step on
every minibatch — and the minibatch path is *already* host-round-trip-heavy
(`select_rows_2d` / `select_rows_int` read tensors to CPU `Vec<f32>`/`Vec<i64>`
and re-upload — `trainer.rs:299-343`). Actor-learner leaves that hot loop
untouched: actors run inference-only (no autodiff), stream trajectories to the
learner over channels, and the learner applies its existing update step with a
V-trace correction for the staleness introduced by asynchrony. Lower blast
radius, and it scales the axis (env throughput) that actually helps in Thrust's
current CPU-favoured regime.

DDP remains the right tool once a single large net is GPU-bound; it is a Phase 4
option (see child issues), gated behind the workload trigger.

## Q2 — Single-host multi-GPU first, or multi-host from the start?

**Recommendation: single-host first, unconditionally — for either architecture.**

- Both Burn device backends enumerate multiple local devices, so single-host
  multi-GPU needs no network layer: `WgpuDevice`/`CudaDevice` select among local
  adapters, and Thrust's `default_burn_device::<B>()` (`src/utils/cuda.rs:27-31`)
  currently hard-codes the singleton default device — the single point that a
  multi-device build generalizes.
- `burn-collective`'s config encodes exactly this split: `CollectiveConfig`
  carries `num_devices` for the single-host (`local`) case, and the multi-host
  (`global`) case is *opt-in* via `num_nodes` / `global_address` / `node_address`
  (`burn-collective-0.21.0/src/config.rs:11-24`); its `Default` is single-node
  (`num_nodes: None`, `config.rs:77`). Multi-host is a strict superset that adds
  a websocket transport (`burn-communication`, see Q3), node addressing, and
  failure handling.
- The same holds for actor-learner: single-host uses in-process
  `crossbeam-channel`; multi-host would swap that for a network transport
  (TCP/gRPC/websocket) — a separate complexity layer with its own reliability
  and serialization concerns.

So Phase 1–3 are single-host; multi-host is Phase 4 for whichever architecture
reaches it first.

## Q3 — What does Burn 0.21 actually support for multi-device gradient reduction?

**Corrected answer: Burn 0.21 ships a native, published collective library with
`all_reduce`.** This note deliberately diverges from the epic's (and the
curator comment's) premise that `burn-collective` is "unpublished" / "a stub" —
that is not what the registry shows. The facts, each verified against source:

### `burn-collective 0.21.0` is published and provides allreduce

- **Published on crates.io.** The sparse index lists `burn-collective` versions
  `0.19.0 … 0.20.1, 0.21.0-pre.1 … 0.21.0-pre.5, 0.21.0` (first published
  2025-10-28; `0.21.0` is present and **not yanked**). It was simply absent from
  the local cache because Thrust never enables the feature that pulls it in, so
  cargo never downloaded it — absence-from-cache is not absence-from-crates.io.
- **Wired into `burn` behind the `collective` feature.** `burn 0.21.0`'s
  `Cargo.toml` declares `collective = ["burn-collective", …]` (lines 74–76) with
  `[dependencies.burn-collective] version = "0.21.0"`, `optional = true`
  (line 278); `burn/src/lib.rs:139-141` re-exports it
  (`#[cfg(feature = "collective")] pub mod collective;`), and
  `burn/src/collective.rs` is just `pub use burn_collective::*;`. Enabling it in
  Thrust is a one-line feature add: `burn = { features = ["collective"] }`.
- **Real collective ops over backend tensors.** `burn-collective`'s public API
  (`burn-collective-0.21.0/src/api.rs`) is:
  - `register::<B>(id: PeerId, device, config: CollectiveConfig)` (api.rs:54)
  - `all_reduce::<B>(id, tensor: B::FloatTensorPrimitive, op: ReduceOperation)` (api.rs:70)
  - `broadcast::<B>(id, tensor)` (api.rs:86) and `reduce::<B>(…, root)` (api.rs:101)
  - `ReduceOperation::{Sum, Mean, …}` (config.rs:262-264) and
    `AllReduceStrategy::{Tree(n), Ring, Centralized}` (config.rs:269-272).

  Crucially these operate on `B::FloatTensorPrimitive` under a plain
  `B: Backend` bound — i.e. exactly the gradient tensors a DDP step produces —
  **not** a host `Vec<f32>`. So the "manual host round-trip per update" the epic
  described as the only DDP option is a *fallback*, not the primitive Burn
  offers; the native path keeps the reduction on-device.
- **Single-host and multi-host, both provided.** The `local` module runs a
  coordinating thread with a tokio runtime for in-process cross-device reduction
  (`burn-collective-0.21.0/src/local/server.rs:21,31`); the `global` module does
  multi-node reduction over `burn-communication` websockets with `Address` /
  `NodeId` node identity (`src/global/shared.rs:4,61,80`). This directly answers
  Q2's transport question from the library's own structure.

### `burn-router` is *not* a gradient-sync primitive

For completeness, the other multi-device-looking crate, `burn-router 0.21.0`, is
a **heterogeneous backend router**, not a collective:
`Router<Backends> = BackendRouter<DirectByteChannel<Backends>>` and
`DirectByteChannel = DirectChannel<Backends, ByteBridge<Backends>>`
(`burn-router-0.21.0/src/lib.rs:27,36`). It routes tensor ops to *different*
backends by serialising tensor data across a byte bridge — designed for
heterogeneous pipeline parallelism (different layers on different backends), not
for averaging gradients across replicas. Its byte-bridge transfer cost makes it
the wrong tool for allreduce. Use `burn-collective` for gradient sync;
`burn-router` is unrelated to this epic.

### So why still defer DDP?

Because the *capability* being present does not make it *pay off*. The DDP win
requires per-device work large enough that a GPU beats the CPU in the first
place, and `BURN_BACKENDS.md` shows Thrust's current 64-unit MLPs sit far below
that crossover (CPU wins 4.4–9.5×; a 4090 sat at 38–41 % utilization). Sharding
those tiny nets across multiple GPUs and adding an allreduce barrier per update
would be strictly slower. The `burn-collective` allreduce is the right mechanism
to adopt **when the workload grows into the GPU-favoured regime** — filed as the
Phase 4 child issue, gated behind the same trigger as GPU/FP16 adoption.

---

## Reproducing the Burn-source claims

The publication and API facts above are checkable without a GPU host:

```bash
# 1. Confirm burn-collective 0.21.0 is published (not yanked):
curl -s https://index.crates.io/bu/rn/burn-collective | grep -o '"vers":"0.21.0"'

# 2. Confirm burn wires it behind the `collective` feature:
grep -n 'burn-collective' ~/.cargo/registry/src/index.crates.io-*/burn-0.21.0/Cargo.toml

# 3. Read the collective API (download the crate; it is optional so not cached):
curl -sL https://static.crates.io/crates/burn-collective/burn-collective-0.21.0.crate \
  | tar xz && sed -n '48,111p' burn-collective-0.21.0/src/api.rs
```

A first empirical exercise of the `local` allreduce is the Phase 1 spike below.

---

## Child issues

Following the epic's candidate phasing (refined to target the corrected Q3):

| Phase | Issue | Scope | Depends on |
| --- | --- | --- | --- |
| 1 | Feasibility spike: `burn-collective` local allreduce + cross-device gradient aggregation | Enable burn's `collective` feature in a throwaway example; `register` N devices and `all_reduce` gradient-shaped tensors on `local`; time it vs single-device and vs the host-round-trip fallback; record which backends work | None |
| 2 | Single-host async actor-learner: multi-actor PPO over `crossbeam-channel` | N inference-only actors collect rollouts; one learner trains; channel-based trajectory hand-off; no gradient sync | Phase 1 verdict |
| 3 | V-trace integration for the actor-learner learner step | Wire V-trace ([#263](https://github.com/rjwalters/thrust/issues/263)) into the learner's update to correct actor-trajectory staleness | Phase 2 + #263 |
| 4 | Multi-host coordination + benchmarks | Network transport (`burn-collective` `global` websocket for DDP, or a networked actor-learner); throughput benchmarks vs single-host baseline; revisit DDP once the GPU-workload trigger is met | Phase 3 |

The child issues are filed against this epic with `loom:triage` and are
sequenced behind the re-evaluation trigger where GPU-bound (Phase 4 DDP in
particular). They mirror — deliberately — the FP16 epic's approach of filing
implementation phases behind a workload trigger (cf. [#267](https://github.com/rjwalters/thrust/issues/267)
children [#270](https://github.com/rjwalters/thrust/issues/270) /
[#272](https://github.com/rjwalters/thrust/issues/272)).
