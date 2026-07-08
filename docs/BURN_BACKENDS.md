# Burn Backends in Thrust

After phase 5 of the Burn migration (#82), Thrust uses
[Burn](https://burn.dev) as its only tensor backend. Burn is a pure-Rust
deep-learning framework with a multi-backend GPU story — the actual
hardware target is chosen at compile time via Cargo features.

## Default: NdArray (CPU, pure Rust)

The default `training` feature pulls in Burn's `ndarray` backend, the
`autodiff` decorator, and nothing else. This is what `cargo build` /
`cargo test` use, what CI runs, and what the
`train_simple_bandit` example targets:

```bash
cargo run --release --example train_simple_bandit
```

The NdArray backend is the right choice for:

- Headless CI runs (no GPU required).
- Reproducible numerical experiments.
- Onboarding / demos where you don't want to install a GPU runtime.

## Opt-in: wgpu (cross-platform GPU)

```toml
# Cargo.toml feature combo
thrust-rl = { version = "0.1", features = ["training", "wgpu"] }
```

```bash
cargo run --release --features "training,wgpu" --example train_simple_bandit
```

wgpu targets every desktop/mobile GPU API: Vulkan (Linux), Metal
(macOS / iOS), DX12 (Windows), and WebGPU (browser). This is the
recommended GPU backend for cross-platform development.

## Opt-in: CUDA (Linux + NVIDIA)

```bash
cargo run --release --features "training,cuda" --example train_simple_bandit
```

The CUDA backend talks directly to the NVIDIA driver without going
through libtorch. Linux + an NVIDIA GPU + CUDA toolkit required.

## Picking the right backend

| Use case | Backend |
| --- | --- |
| CI / headless tests / numerical reproducibility | NdArray (default) |
| Local laptop GPU (any vendor) | wgpu |
| Linux box with an NVIDIA GPU | cuda or wgpu |
| Browser-side inference / training | wgpu (compiles to WebGPU) |

## End-to-end validation runs

Burn's compile-time backend swap is verified by the runs below. Both
`train_simple_bandit` and `train_cartpole_modern` parameterize their
`Backend` type alias with a `#[cfg(feature = "wgpu")]` so the same
trainer source code drives either backend; only the compile-time
feature flag changes.

### `train_simple_bandit` (#102)

**Host setup**

- macOS (Darwin 25.5).
- Apple M3 Ultra (Metal backend selected by wgpu).
- Rust 1.x stable, Burn 0.21, `cubecl_wgpu` 0.21.

**Command**

```bash
# wgpu (Metal)
cargo run --release --features "training,wgpu" --example train_simple_bandit
# NdArray (CPU) baseline
cargo run --release --example train_simple_bandit
```

**Result**

| Backend | Final success | Random baseline | Wall clock (50k steps) | env-steps/sec |
| ------- | ------------- | ---------------- | ----------------------- | -------------- |
| NdArray (CPU) | **98.3%** | 50% | 6.4s | ~7,800 |
| wgpu / Metal | **98.3%** | 50% | 85.2s | ~590 |

Both backends learn the same near-optimal policy (action probability
~1.0 on the correct lever in each of the two contexts). wgpu is roughly
13x slower wall-clock for this trivial 1-input/2-action MLP because GPU
kernel launch overhead dominates — see the "performance note" below.

### `train_cartpole_modern` (#102)

**Host setup**: same as above (M3 Ultra / Metal via wgpu).

**Command**

```bash
# wgpu (Metal)
TOTAL_TIMESTEPS=40000 cargo run --release \
    --features "training,wgpu" \
    --example train_cartpole_modern
# NdArray (CPU) baseline
TOTAL_TIMESTEPS=40000 cargo run --release --example train_cartpole_modern
```

**Result**

| Backend | Final avg episode length (last 100) | Random baseline | Wall clock (~37k steps) | env-steps/sec |
| ------- | ------------------------------------ | --------------- | ----------------------- | -------------- |
| NdArray (CPU) | **161.1** | ~22 | 4.1s | ~8,970 |
| wgpu / Metal | **175.0** | ~22 | 97.5s | ~378 |

Both backends pass the "better than random" bar by an order of
magnitude on the same 40k-step budget. The small difference between
runs (161 vs 175) is consistent with run-to-run RNG variance for a
~9-update PPO smoke run; no correctness regression is implied.

### Bug surfaced during validation

The wgpu run on `train_cartpole_modern` initially panicked with
`index out of bounds: the len is 0` inside `train::ppo::trainer::select_rows_int`.
Root cause: `TensorData::to_vec::<E>()` is dtype-strict, and Burn's
default integer dtype is backend-dependent — `NdArray` stores int
tensors as `i64`, but `Wgpu<f32, i32>` stores them as `i32`. The old
code unwrapped the error to a default (empty) vector, which only worked
by accident on NdArray. Fixed by switching to `TensorData::iter::<i64>()`,
which performs the cross-dtype cast per-element. See PR resolving #102.

### Performance note

These numbers are validation, not throughput. On a small (1-input or
4-input, 64-unit) MLP, GPU kernel-launch overhead dominates per-op
latency, so wgpu/Metal looks slower than the SIMD-friendly NdArray
path. Burn's wgpu backend autotunes a fusion runtime on first use
(visible as the ~50s "Created wgpu compute server" log line), which
also inflates the wall-clock numbers above. Real wins on wgpu show up
for larger nets, bigger batches, or convolution-heavy workloads
(Snake CNN, image envs) — see issue #65 follow-ups.

## Throughput benchmarking on GPU backends

The `trainer_throughput` criterion harness (`benches/trainer_throughput.rs`) is
generic over the Burn backend, so the exact same bench bodies run on CPU and
GPU. The CPU `ndarray` baseline is **always** registered; the wgpu and cuda
variants are added behind Cargo-feature gates. A single run therefore produces a
paired CPU-vs-GPU comparison.

### Running

```bash
# CPU only (default — what CI runs). Emits the `/ndarray` groups.
cargo bench --features training --bench trainer_throughput

# wgpu (cross-platform GPU: Vulkan/Metal/DX12/WebGPU).
# Emits BOTH the `/ndarray` baseline AND the `/wgpu` groups in one run.
cargo bench --features "training,wgpu" --bench trainer_throughput

# cuda (Linux + NVIDIA + CUDA toolkit).
# Emits BOTH the `/ndarray` baseline AND the `/cuda` groups in one run.
cargo bench --features "training,cuda" --bench trainer_throughput
```

A quick smoke run (short warm-up / measurement windows) is:

```bash
cargo bench --features "training,wgpu" --bench trainer_throughput -- \
    --warm-up-time 1 --measurement-time 3
```

### Reading the results

Every benchmark group is suffixed with its backend tag, so the eight logical
groups appear once per compiled backend, side by side:

| Logical group | CPU group id | wgpu group id | cuda group id |
| --- | --- | --- | --- |
| A2C per-update | `a2c_train_step/ndarray` | `a2c_train_step/wgpu` | `a2c_train_step/cuda` |
| PPO per-update | `ppo_train_step/ndarray` | `ppo_train_step/wgpu` | `ppo_train_step/cuda` |
| DQN per-update | `dqn_train_step/ndarray` | `dqn_train_step/wgpu` | `dqn_train_step/cuda` |
| SAC per-update | `sac_train_step/ndarray` | `sac_train_step/wgpu` | `sac_train_step/cuda` |
| A2C full loop | `a2c_cartpole_steps_per_sec/ndarray` | …`/wgpu` | …`/cuda` |
| PPO full loop | `ppo_cartpole_steps_per_sec/ndarray` | …`/wgpu` | …`/cuda` |
| DQN full loop | `dqn_cartpole_steps_per_sec/ndarray` | …`/wgpu` | …`/cuda` |
| SAC full loop | `sac_pendulum_steps_per_sec/ndarray` | …`/wgpu` | …`/cuda` |
| Nature-DQN policy forward | `nature_dqn_policy_forward/ndarray` | …`/wgpu` | …`/cuda` |
| Nature-DQN Q-net forward | `nature_dqn_qnet_forward/ndarray` | …`/wgpu` | …`/cuda` |
| Nature-DQN policy train step | `nature_dqn_policy_train_step/ndarray` | …`/wgpu` | …`/cuda` |
| Nature-DQN Q-net train step | `nature_dqn_qnet_train_step/ndarray` | …`/wgpu` | …`/cuda` |

The four `nature_dqn_*` groups are the large-net crossover benchmarks
(issue #328). Unlike the small-net groups, each runs at **two** batch sizes —
`b32` (Nature-DQN paper batch) and `b256` (Atari PPO minibatch upper range) —
so their full benchmark ids are e.g. `nature_dqn_policy_forward/wgpu/b256`. The
`*_forward` pair measures pure inference (no autograd); the `*_train_step` pair
measures forward + backward + one Adam step. They feed synthetic NCHW
`[B, 4, 84, 84]` observations straight to the Nature-DQN conv stack, decoupled
from env stepping (per the Epic #306 Phase 1 spike) so the numbers reflect the
*network*, not subprocess IPC.

To compare a backend against the CPU baseline, read the matching `/ndarray` and
`/wgpu` (or `/cuda`) groups from the same run. (The same fairness caveats as the
CPU benches apply: `*_train_step` groups are cross-algorithm comparable;
`*_steps_per_sec` groups are comparable only within an algorithm class, and the
SAC Pendulum loop is not comparable across environments — see the module header
of `benches/trainer_throughput.rs`.)

### Caveats

- **GPU toolchain required to compile.** The GPU registration code only compiles
  when its feature is on. `wgpu` uses Metal on macOS and generally builds on a
  developer laptop; `cuda` requires Linux + an NVIDIA GPU + the CUDA toolkit and
  will not build elsewhere. CI is CPU-host only and never sets these features, so
  the default `cargo bench --features training` path is unaffected.
- **No graceful skip.** Criterion has no skip primitive, and there is no runtime
  adapter probe. If a GPU feature is compiled on a host with no adapter, Burn
  panics on first device use. This is acceptable because the feature is opt-in
  and only enabled on GPU hosts.
- **Small nets favour the CPU.** As with the validation runs above, autotune and
  kernel-launch overhead dominate per-op latency on the small (4-input,
  64-unit) MLPs these benches use, so wgpu/Metal can look slower than the
  SIMD-friendly NdArray path. See the [Performance note](#performance-note)
  above — real GPU wins show up for larger nets, bigger batches, or
  convolution-heavy workloads. Treat these benches as a relative harness, not an
  absolute GPU endorsement.

### Measured CPU-vs-GPU throughput (issue #184)

Run on an operator GPU host (the alc-2 workstation):

- **CPU**: Intel Core i9-14900K (32 threads)
- **GPU**: NVIDIA GeForce RTX 4090 (24 GB), driver 580.126.09
- **OS**: Ubuntu 24.04.1 LTS, kernel 6.14
- **Stack**: Burn 0.21, `ndarray` (CPU) vs `wgpu` → **Vulkan** (GPU)
- **Command**: `cargo bench --features "training,wgpu" --bench trainer_throughput -- --warm-up-time 2 --measurement-time 10`

Median per-iteration wall-clock, lower is better. "CPU faster ×" is `wgpu / ndarray`:

| Benchmark group | NdArray (CPU) | wgpu (RTX 4090) | CPU faster × |
| --- | --- | --- | --- |
| `a2c_train_step` (per update) | 261 µs | 1.265 ms | 4.8× |
| `ppo_train_step` (per update) | 296 µs | 1.890 ms | 6.4× |
| `dqn_train_step` (per update) | 37.9 ms | 172.9 ms | 4.6× |
| `sac_train_step` (per update) | 653 µs | 3.666 ms | 5.6× |
| `a2c_cartpole_steps_per_sec` (full loop) | 477 µs | 2.973 ms | 6.2× |
| `ppo_cartpole_steps_per_sec` (full loop) | 620 µs | 5.883 ms | 9.5× |
| `dqn_cartpole_steps_per_sec` (full loop) | 43.4 ms | 188.7 ms | 4.4× |
| `sac_pendulum_steps_per_sec` (full loop) | 10.7 ms | 65.8 ms | 6.2× |

#### cuda column (issue #219, measured 2026-06-28)

The original #184 run left `cuda` blank — alc-2 had the NVIDIA driver but no CUDA
toolkit. After installing `nvidia-cuda-toolkit` (nvcc 12.0.140) on the same host,
the cuda backend was benched with
`cargo bench --features "training,cuda" --bench trainer_throughput -- --warm-up-time 2 --measurement-time 10`.
Because this is a **separate run**, its own `/ndarray` baseline (measured in the
same invocation, slightly higher than #184's — normal run-to-run variance under
host load) is shown alongside so the cuda/CPU ratio is apples-to-apples:

| Benchmark group | NdArray (CPU, same run) | cuda (RTX 4090) | CPU faster × |
| --- | --- | --- | --- |
| `a2c_train_step` (per update) | 341 µs | 11.96 ms † | ~35× † |
| `ppo_train_step` (per update) | 384 µs | 1.024 ms | 2.7× |
| `dqn_train_step` (per update) | 46.0 ms | 150.2 ms | 3.3× |
| `sac_train_step` (per update) | 840 µs | 2.633 ms | 3.1× |
| `a2c_cartpole_steps_per_sec` (full loop) | 618 µs | 1.641 ms | 2.7× |
| `ppo_cartpole_steps_per_sec` (full loop) | 805 µs | 2.930 ms | 3.6× |
| `dqn_cartpole_steps_per_sec` (full loop) | 53.7 ms | 160.5 ms | 3.0× |
| `sac_pendulum_steps_per_sec` (full loop) | 14.1 ms | 44.76 ms | 3.2× |

† `a2c_train_step/cuda` did not stabilize — criterion reported a [0.77 ms, 11.96 ms,
34.3 ms] spread, the signature of JIT-kernel / autotune contamination on the
smallest workload. The stable lower bound (~0.77 ms ≈ 2.3× CPU) is in line with the
other groups; treat the 35× median as an artifact, not a real regression.

**Two findings.** (1) **cuda also trails the CPU NdArray backend** at these
small-MLP sizes — by ~2.7–3.6×, exactly the #184 prediction. The dispatch-overhead
story is identical: there isn't enough arithmetic per launch to amortize host↔device
cost. (2) **cuda is consistently faster than wgpu** on the same 4090 (e.g. PPO full
loop 2.93 ms vs wgpu 5.88 ms; SAC Pendulum 44.8 ms vs 65.8 ms; DQN per-update 150 ms
vs 173 ms) — the native CUDA path has lower per-kernel launch overhead than wgpu →
Vulkan, so it closes some of the gap to CPU but does not overturn the conclusion.
**NdArray (CPU) remains the right default**; cuda is the better GPU choice over wgpu
on Linux+NVIDIA when a workload is large enough to want a GPU at all.

### Measured large-net (Nature-DQN CNN) throughput (issue #328)

The groups above are all small-net (4-input, 64-unit MLP) workloads. Issue #328
adds the designated **large-net re-evaluation trigger**: the Nature-DQN
convolutional stack (Mnih et al. 2015 — three conv layers + a 512-wide FC trunk,
~1.69 M parameters), benchmarked in isolation on synthetic `[B, 4, 84, 84]`
frames. This is the workload the [Interpretation](#interpretation) section named
as "what would flip the conclusion."

**Host setup**

- macOS arm64 (Darwin 25.5).
- Apple M3 Ultra (Metal backend selected by wgpu).
- Rust stable, Burn 0.21.
- CPU and wgpu columns measured **on this host**; cuda column pending (see below).

**Command**

```bash
# CPU baseline (ndarray) and wgpu/Metal measured in separate runs on this host,
# both with 10-sample criterion windows to bound wall-clock:
cargo bench --features training --bench trainer_throughput \
    -- --sample-size 10 --warm-up-time 2 --measurement-time 10 nature_dqn
cargo bench --features "training,wgpu" --bench trainer_throughput \
    -- --sample-size 10 --warm-up-time 2 --measurement-time 10 'nature_dqn.*wgpu'
```

> **Methodology note (issue #335).** The four `nature_dqn_*` timed closures now
> include an explicit `B::sync(device)` call to drain the GPU command queue
> *before* criterion stops the clock. Previously they returned device-resident
> tensors/models without any host read-back, so on async backends (wgpu/Metal,
> cuda) some GPU work could be deferred past the end of the timed region,
> under-reporting true compute time. `B::sync` is a no-op on NdArray (CPU numbers
> unaffected). The wgpu/Metal numbers below were **re-measured with the sync in
> place** and are materially higher for the forward groups than the original
> #328 run — the earlier numbers under-counted deferred kernel work. The
> pending cuda run on alc-2 will use this corrected bench code.

Median per-iteration wall-clock, lower is better. "wgpu faster ×" is
`ndarray / wgpu` — note the direction is **inverted** from the small-net tables
above (there CPU won; here the GPU wins, often by two orders of magnitude):

| Benchmark group | NdArray (CPU) | wgpu / Metal | cuda | wgpu faster × |
| --- | --- | --- | --- | --- |
| `nature_dqn_policy_forward` (b32) | 114.4 ms | 13.73 ms | **cuda: pending — run on alc-2** | ~8.3× |
| `nature_dqn_policy_forward` (b256) | 913.9 ms | 35.74 ms | **cuda: pending — run on alc-2** | ~26× |
| `nature_dqn_qnet_forward` (b32) | 115.6 ms | 15.48 ms | **cuda: pending — run on alc-2** | ~7.5× |
| `nature_dqn_qnet_forward` (b256) | 922.2 ms | 113.7 ms ‡ | **cuda: pending — run on alc-2** | ~8.1× ‡ |
| `nature_dqn_policy_train_step` (b32) | 1.224 s | 29.96 ms | **cuda: pending — run on alc-2** | ~41× |
| `nature_dqn_policy_train_step` (b256) | 9.873 s | 169.5 ms | **cuda: pending — run on alc-2** | ~58× |
| `nature_dqn_qnet_train_step` (b32) | 1.238 s | 27.85 ms | **cuda: pending — run on alc-2** | ~45× |
| `nature_dqn_qnet_train_step` (b256) | 10.63 s | 167.1 ms | **cuda: pending — run on alc-2** | ~64× |

All eight rows above were **re-measured after the in-region `B::sync(device)`
fix (issue #335)**, replacing the original #328 numbers. The most visible change
is the forward groups: forcing a command-queue drain inside the timed region
pushed `policy_forward/b32` from a reported 1.88 ms up to 13.73 ms and
`qnet_forward/b256` from 19.5 ms up to 113.7 ms — the earlier figures were
under-counting deferred kernel work. The `policy_train_step/wgpu` rows, which
previously showed autotune-contaminated wide intervals (medians of 437 ms /
737 ms), now stabilize cleanly at ~30 ms / ~170 ms once the sync bounds the
per-iteration work — confirming the earlier medians were autotune-inflated
floors rather than real steady-state cost. The GPU still wins every row
decisively; the ratios are simply more honest.

‡ `nature_dqn_qnet_forward/wgpu/b256` showed a wider-than-usual interval
(`[35.85 ms, 113.7 ms, 159.2 ms]`) — first-use autotune contamination on this
group. Treat the ~8× ratio as a conservative floor; the tight b32 sibling
(15.48 ms) and the sibling forward groups suggest the steady-state ratio is
higher. A longer run (`--sample-size 50 --measurement-time 30`) would tighten it.

**cuda: pending operator run on alc-2.** Command:
`cargo bench --features "training,cuda" --bench trainer_throughput --warm-up-time 2 --measurement-time 10 -- nature_dqn`.
The bench code now includes the in-region `B::sync(device)` fix (issue #335), so
this run will time full GPU work without the operator needing any change. Host:
`sphere@alc-2` over Tailscale (Ubuntu 24.04, RTX 4090, CUDA 12.x). See the
[cuda column (issue #219)](#cuda-column-issue-219-measured-2026-06-28) above for
environment-setup reference. The operator fills in the cuda column, removes the
placeholders, and updates the verdict below if the cuda result changes the
conclusion.

### Interpretation

**The CPU NdArray backend wins every group by 4.4–9.5×.** This is the expected
result at these sizes, not a misconfiguration — `nvidia-smi` sampled the 4090 at
only **38–41 % utilization / ~150 MB** during the wgpu run, confirming Vulkan
genuinely drove the discrete GPU (not a software/`lavapipe` fallback) but left it
almost entirely idle. The benches use 64-unit MLPs with ≤256-row batches, so each
op is dominated by GPU kernel-launch + host↔device transfer overhead rather than
the matmul itself — there simply isn't enough arithmetic per launch to amortize
the dispatch cost. The off-policy `*_train_step` groups (DQN/SAC) close the gap
slightly (4.6× / 5.6×) because their larger replay minibatches give the GPU a
bit more to chew on, which is the trend you'd expect.

**What would flip the conclusion:** materially wider/deeper networks (e.g. CNN
policies for image envs), much larger batch sizes, or many parallel environments
batched into a single device dispatch — anything that raises arithmetic-per-launch
above the dispatch-overhead floor. Until Thrust's default workloads grow in that
direction, **NdArray (CPU) remains the right default**, and the GPU backends are
best reserved for large-model / high-parallelism configurations. This is the
quantified version of the [Performance note](#performance-note) and the
validation-run observations above.

#### Large-net verdict (Nature-DQN CNN)

**The CNN workload flips the small-net conclusion — decisively, at both B=32 and
B=256.** Where the 4–64-unit MLP groups had CPU winning by 4.4–9.5×, the
Nature-DQN conv stack has wgpu/Metal winning by **~7–64×** on every group
(both forward passes, both Q-net rows, and — now that the in-region sync bounds
its per-iteration cost — the policy train step too). These ratios are measured
with the `B::sync(device)` fix (issue #335) in the timed region, so they reflect
full GPU compute; they are lower than the original #328 forward-group figures
precisely because the earlier run under-counted deferred kernel work. The
crossover is not marginal and does not require B=256 to appear: even at the
Nature-DQN paper batch of 32, `nature_dqn_policy_forward` is ~8× faster on the
GPU and `nature_dqn_qnet_train_step` ~45× faster. Larger batch widens the gap
(CPU scales roughly linearly in batch — 114 ms → 914 ms forward — while the GPU
grows far more slowly, 13.7 ms → 35.7 ms), and it is the train-step groups
(forward + backward + optimizer, the most arithmetic per launch) that show the
largest advantage. This is exactly the prediction the small-net Interpretation
made: raise arithmetic-per-launch above the dispatch-overhead floor — here via a
convolutional trunk with ~1.69 M parameters over 84×84 frames — and the GPU's
throughput advantage dominates.

**Recommendation update — image-env training should default to a GPU backend.**
For the Nature-DQN / Atari-scale image workloads that motivated Epic #306, the
[Picking the right backend](#picking-the-right-backend) guidance inverts: prefer
**wgpu** (Metal/Vulkan) — or **cuda** on Linux+NVIDIA — over NdArray. NdArray
(CPU) remains the right default only for the small-MLP control tasks (CartPole,
Pendulum) that dominate the rest of the suite; it is not competitive on the CNN
policy (a 9.9 s CPU train step at B=256 versus a sub-second, ~170 ms GPU step). The
small-net default is unchanged; the large-net default is now GPU.

**cuda status.** The large-net wgpu/Metal numbers above are measured and
available; the **cuda crossover verdict is pending the alc-2 operator run**. The
#219 finding that cuda beats wgpu on the same 4090 suggests cuda will land at
least as fast as these wgpu numbers (so the GPU-wins conclusion is very unlikely
to reverse), but that is a prediction, not a measurement — the cuda column stays
a placeholder until the operator fills it in.

> **Mixed precision (FP16/BF16):** f16/bf16 autodiff is supported on the cubecl
> GPU backends (wgpu/cuda) but **not** on NdArray (CPU), and pays off only in
> the same large-net regime that justifies a GPU at all. See
> [`FP16_FEASIBILITY.md`](./FP16_FEASIBILITY.md) for the spike (issue #267).

## Why Burn instead of libtorch?

The pre-v0.1.0 trainer stack used `tch` (Rust bindings to libtorch).
Phase 5 dropped that path in favour of Burn for these reasons:

- **No C++ FFI**. `cargo build` works on any platform that has Rust;
  no separate libtorch download, no `LIBTORCH_USE_PYTORCH=1` env var,
  no `dyld: Library not loaded` errors.
- **Multi-vendor GPU**. libtorch's CUDA path works, but ROCm / Metal /
  Vulkan / WebGPU are not first-class concerns of the Rust bindings.
  Burn supports all of them through a single tensor API.
- **Tensor IR usable from WASM**. The browser-side inference path can
  share kernels with the training-side stack.
- **Active open-source development**. Burn is under active development
  and ships small, well-scoped releases.

See issue #65 for the full migration write-up.
