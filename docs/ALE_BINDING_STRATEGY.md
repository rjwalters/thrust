# ALE binding strategy — spike

Design note for epic [#306](https://github.com/rjwalters/thrust/issues/306)
("Atari (ALE) environment + CNN policy"), Phase 1 of 4
([#324](https://github.com/rjwalters/thrust/issues/324)). This is the **spike
deliverable**, not an implementation: it picks the Rust path to the Arcade
Learning Environment (ALE), settles the ROM story, selects the reference game,
and confirms-or-amends the Phase 2–4 scope so the downstream issues
(#325–#330) can be built against a fixed decision.

- **Burn version:** 0.21.0 (`Cargo.toml`).
- **Spike host / date:** macOS (Darwin 25.5, Apple M-series), 2026-07-07.
- **Method:** the candidate landscape was surveyed from crates.io and the
  linked upstream repositories; the license, build-cost, and maintenance
  facts in Q1 are grounded in that survey (see the verified-facts note at the
  end of Q1). Claims that could not be verified from the spike host are
  **explicitly flagged** as such. No ALE, libale, or ROM was installed,
  vendored, or committed — per the epic's Phase-1 scope this note *prescribes*
  the integration, it does not build it.
- **Template:** mirrors [`FP16_FEASIBILITY.md`](./FP16_FEASIBILITY.md) and
  [`DISTRIBUTED_TRAINING_DESIGN.md`](./DISTRIBUTED_TRAINING_DESIGN.md) — verdict
  first, then one section per open question, then the Phase 2–4 prescription and
  scope reconciliation.

---

## Verdict: **Option D — subprocess via `ale-py`** — zero Rust build-cost, cleanest license boundary, rides Farama's upstream maintenance

**Recommendation: integrate ALE through a subprocess frame protocol against
Farama's `ale-py`, not through a compiled-in FFI binding.** The `AtariEnv`
adapter (Phase 2, #325) spawns `ale-py` as a child process and exchanges
seeds/actions/observations over a simple length-prefixed binary protocol on
the child's stdin/stdout. The Rust crate gains **no** compile-time native
dependency; `env-atari` becomes a pure compile-time flag (`env-atari = []`)
with no entry in `[dependencies]`.

This decision is driven by three conclusions, one per axis the epic asked
about:

- **Build-dependency cost (the axis that gates a CI job): zero.** Unlike the
  FFI path (Option C), Option D adds no cmake, no C++17 toolchain, no zlib, and
  no `build.rs` to the Rust build. The default build and `cargo publish` are
  completely unaffected, and — because there is no path/native dependency —
  Phase 2 needs **none** of the `sed` publish-workaround that `env-bucket-brigade`
  requires (`docs/RELEASING.md` Step 2.5). This is a strictly cleaner story than
  the precedent it mirrors.

- **License posture: the subprocess boundary keeps thrust MIT OR Apache-2.0.**
  ALE is GPL-2.0-or-later. Options A and C *link* it into the thrust binary,
  which (engineering assessment, **flagged for operator/legal confirmation** —
  see Q2) makes any *distributed binary* built with `env-atari` a combined work
  subject to GPL-2.0. Option D invokes `ale-py` as a separate program at arm's
  length — the classic subprocess boundary — so thrust's own artifact stays
  MIT/Apache and no GPL text enters the repo or the crate tarball at all. Option
  B (`atari-env`) is GPL-2.0 *on the crate itself* and is ruled out outright.

- **CI impact: safe to add an `env-atari` clippy job unconditionally.** Because
  the feature pulls no native dep, `cargo clippy --features "training,env-atari"`
  compiles on a bare runner with no ALE present (the subprocess is only needed at
  *runtime*, for tests that actually step the env). Live-subprocess integration
  tests are gated behind a runtime "is `ale-py` present?" skip, so the default
  test matrix stays green without Python.

The price paid is real and acknowledged: **(1)** a runtime Python dependency
(`pip install ale-py`) on any host that actually steps the env, including the
alc-2 training host in Phase 4; and **(2)** per-step IPC + serialization
overhead versus an in-process FFI call. Neither is disqualifying for this epic
— the workload's purpose is to exercise the **CNN** on a large-net GPU crossover
(the env is a data source, not the thing under test), and IPC cost is amortized
by frameskip (4) and by running vectorized `ale-py` workers in parallel. If
Phase 4 shows env throughput to be the binding constraint, Option C is
documented below as a pre-analyzed escalation path — but we do not pay its build
and license cost speculatively.

---

## Q1 — Which Rust binding path to ALE?

### Decision table

All four candidate paths, scored on the five axes the epic named. The chosen
row is **Option D**.

| # | Approach | Source | License posture | Rust build cost | Determinism | Performance | Maintenance | CI impact |
|---|----------|--------|-----------------|-----------------|-------------|-------------|-------------|-----------|
| A | High-level Rust wrapper | `ale` v0.1.3 + `ale-sys` v0.1.2 ([trolleyman/ale-rs](https://github.com/trolleyman/ale-rs)) | MIT on wrapper, **but links GPL-2.0 libale** | Still links system libale: cmake + C++17 + zlib | In-process, deterministic | Fast (in-process FFI) | **Abandoned since 2020**; written against pre-Farama ALE ~0.6, almost certainly API-incompatible with 0.8+ | Would need an ALE setup action before any `--features env-atari` step |
| B | High-level Rust wrapper | `atari-env` v0.1.1 ([elbaro/gym-rs](https://github.com/elbaro/gym-rs)) | **GPL-2.0-or-later on the crate itself** | Links libale + optional SDL2 | In-process, deterministic | Fast (in-process FFI) | **Abandoned since 2021** | **Ruled out** — copyleft conflict with `MIT OR Apache-2.0` |
| C | Direct FFI via custom `build.rs` | Fresh `bindgen` bindings against ALE 0.8 `ale_c.h` (no crate; `ale-sys` is the prior-art template) | ALE core is GPL-2.0-or-later; **linked** into the thrust binary → GPL propagation question (see Q2) | cmake ≥ 3.14, C++17 compiler, zlib; ~2–5 min added CI build | In-process, fully deterministic | **Fastest** — in-process, no IPC | ALE 0.8 actively maintained by Farama, **but we own and maintain the bindings** | Needs libale pre-installed on the runner for the clippy/test step; heaviest CI change |
| **D** | **Process-level frame protocol** | **Spawn Farama `ale-py` as a subprocess; length-prefixed binary (or msgpack) over stdin/stdout** | **Subprocess boundary — thrust crate stays MIT/Apache; GPL ALE is a separate runtime program** | **Zero** Rust compile-time native deps | Deterministic given seed forwarded over the protocol (Machado sticky actions p=0.25) | IPC + serialization per step; mitigated by frameskip 4 + vectorized workers | `ale-py` actively maintained by Farama (canonical impl); **we ride upstream** | Clippy step safe with no native dep; live tests gated behind a runtime `ale-py`-present skip |

### Verdict and rationale

**Option D wins on the two axes that matter most for this repository: build-cost
and license.** thrust's entire dependency philosophy — visible throughout
`Cargo.toml` — is "a fresh checkout builds with no external system libraries"
(pure-Rust NdArray default, pure-Rust threaded matmul instead of system BLAS,
wasm-gated deps kept out of the native graph). Options A and C break that stance
by dragging cmake + a C++17 toolchain + zlib into the build and a GPL-2.0 library
into the link. Option D preserves it completely: `env-atari` becomes a
compile-time flag with no dependency line, so the default build, the docs job,
and the publish dry-run are all untouched.

On the remaining axes Option D is competitive rather than dominant, and the
trade is acceptable:

- **Determinism** is preserved. `ale-py` exposes seedable RNG and the Machado et
  al. (2018) sticky-action protocol (p = 0.25); the frame protocol forwards the
  seed on reset and returns the emulator's frames verbatim, so reproducibility is
  a property of what we send, not of the transport. (Rust-side preprocessing —
  see the Phase 2 note on #326 — further removes any nondeterminism from the
  observation pipeline.)
- **Performance** is the one axis where Option C is strictly better (in-process
  FFI has no IPC cost). But the epic's stated purpose is the **CNN large-net GPU
  crossover** (per `docs/BURN_BACKENDS.md`), where the network — not the emulator
  — is the object under test. Frameskip 4 and parallel `ale-py` workers amortize
  the pipe cost, and Phase 3's throughput benchmark (#328) is prescribed to
  measure the policy in isolation on synthetic 84×84×4 batches, decoupled from
  env IPC, precisely so the backend comparison is not contaminated by transport
  overhead.
- **Maintenance**: Option D rides Farama's actively maintained `ale-py` and reuses
  the exact ROM/AutoROM UX users already know. Option C would make us the
  long-term maintainer of hand-written `bindgen` bindings against a C++ ABI —
  the same burden that left Options A and B abandoned.

**Options A and B are rejected.** B is a hard license conflict (GPL-2.0 crate).
A is six years abandoned, pre-Farama, and still incurs the full native-build cost
while offering an almost-certainly-incompatible API; its only residual value is
as a `bindgen` reference *if* Option C is ever pursued.

**Option C is the documented escalation path, not the choice.** If Phase 4
demonstrates that subprocess env throughput caps PPO/DQN rollout rate below what
the GPU policy can consume, revisit C: accept the cmake/C++17/zlib build cost and
the GPL-linked-binary posture (with the Q2 legal question resolved first), using
`ale-sys` as the bindgen template and vendoring the bindings under
`envs/atari/` (which already falls inside the `"envs/**"` tarball exclude). That
is a deliberate, evidence-gated future decision — not something to pay for now.

> **Verified-facts note.** The crate names, versions, last-publish dates,
> licenses, and upstream-repo attributions in rows A/B are drawn from the
> curator's crates.io survey on this issue and are treated as verified. The
> claim that ale-py is "actively maintained by Farama as of 2025", that ALE 0.8
> ships `ale_c.h`, and the specific build-tool versions (cmake ≥ 3.14, C++17)
> are from that same survey. Anything beyond those facts — e.g. exact `ale-py`
> API surface for the frame protocol — is design intent for Phase 2, not a
> verified claim, and is marked as such where it appears.

---

## Q2 — ROM licensing and acquisition story

### License posture of the chosen option (GPL-2.0 — engineering analysis, flagged for legal)

> **This subsection is engineering analysis to inform an operator/legal
> decision. It is not legal advice.** The GPL-2.0 boundary question should be
> confirmed by the operator (and counsel if the crate is ever distributed as a
> binary with `env-atari` enabled) before Phase 2 ships.

ALE is licensed **GPL-2.0-or-later**. The four options sit on different sides of
the copyleft boundary:

- **Option B** puts GPL-2.0 on the Rust crate itself → direct incompatibility
  with thrust's `MIT OR Apache-2.0`. Ruled out.
- **Options A and C** *link* libale into the thrust binary. Under the FSF's
  reading, a binary that links a GPL-2.0 library is a combined/derivative work,
  so any **distributed binary** built with `env-atari` would be encumbered by
  GPL-2.0 as a whole. thrust's *source* and its *crates.io tarball* would remain
  MIT/Apache (ALE source is neither vendored nor shipped), but a user who builds
  with the feature and then redistributes the binary inherits the obligation.
  This is analogous to linking a GPL library like `readline`: shippable with a
  clear NOTICE, but a genuine departure from thrust's permissive posture that
  must be a conscious, documented choice.
- **Option D** invokes `ale-py` as a **separate program over a pipe** — arm's
  length runtime aggregation, the same category as calling any GPL CLI tool from
  a permissively licensed program. The thrust binary is not a derivative work of
  ALE; no GPL text enters thrust's repo, tarball, or a user's thrust binary. This
  is the cleanest boundary of the four and the primary reason it is chosen.

**Conclusion (flagged for confirmation):** Option D's subprocess boundary keeps
thrust unambiguously `MIT OR Apache-2.0` with no GPL propagation into any thrust
artifact. Phase 2 should still add a short NOTICE in the `env-atari` module docs
stating that enabling the feature requires the user to separately install
`ale-py` (GPL-2.0-or-later) and that ALE is invoked as an external program.

### No ROMs in the repo or the tarball

Atari ROMs are copyrighted and are **never** committed to this repository and
**never** included in the crate tarball. This is stated here as policy and
enforced by the audit step below. Under Option D the point is nearly automatic:
ROMs live inside the user's Python environment (resolved by `ale-py`), so no ROM
file is ever a candidate for the repo tree in the first place. The exclude list
is belt-and-suspenders, not the primary control — the primary control is "never
commit a ROM."

### User ROM acquisition pattern

Mirror the established Farama UX. The user, once, runs:

```bash
pip install ale-py           # the emulator + Python API (Option D runtime dep)
# ROM acquisition (either path, depending on ale-py version):
pip install "autorom[accept-rom-license]"   # AutoROM bundled with license acceptance
AutoROM --accept-license                     # downloads ROMs to ale-py's ROM dir
```

> **Should-verify:** recent `ale-py` (0.8+) is understood to bundle
> license-accepted ROMs directly, making the separate `AutoROM` step optional.
> Phase 2 must confirm which applies to the pinned `ale-py` version and document
> the single canonical command. Both are recorded here so Phase 2 does not
> rediscover them.

### Env var and fallback behavior

- **Env var name: `ALE_ROM_PATH`.** Optional override pointing at a directory of
  ROM `.bin` files (or a specific ROM file). When set, the `AtariEnv` subprocess
  client instructs `ale-py` to load the ROM from that path.
- **Fallback:** when `ALE_ROM_PATH` is unset, defer to `ale-py`'s own ROM
  resolution (its bundled/AutoROM-populated ROM directory). If the requested ROM
  cannot be resolved by either mechanism, the env must **panic with a helpful,
  actionable error** naming both the missing ROM and the two remedies (set
  `ALE_ROM_PATH`, or run the AutoROM command above) — never a bare "file not
  found."
- **Compile-time vs runtime:** the ROM path is **runtime-only** configuration.
  No ROM path is baked in at compile time; nothing about ROM location is a build
  input. This is trivially true for Option D (the Rust build never touches a
  ROM).

### Tarball exclude audit (prescription for Phase 2)

Phase 2's PR (#325) must include, and CI should enforce, a ROM-extension audit:

```bash
# Must return empty — no ROM binaries anywhere in the packaged tarball.
cargo package --list --no-default-features --allow-dirty | grep -iE '\.(bin|rom|a26)$'
```

Recommended concrete change in Phase 2: **extend the existing `publish_dry_run`
grep** (`.github/workflows/ci.yml`, the "Assert tarball excludes non-library
assets" step) to also fail on `\.(bin|rom|a26)$`, so a stray ROM is caught at PR
time exactly the way stray `.safetensors`/`web/`/`envs/` assets already are.
Because Option D vendors nothing under `envs/atari/`, no *new* `exclude` glob is
required — but if Phase 2 (or a future Option-C escalation) ever vendors bindings
there, they already fall under the existing `"envs/**"` exclude. The design doc
itself (`docs/ALE_BINDING_STRATEGY.md`) is intentionally **not** excluded: like
the other design notes it carries documentation value on crates.io/docs.rs.

---

## Q3 — Reference game selection: **Pong**

**Choose Pong** as the Phase 4 reference game.

> **Disambiguation:** this is Atari 2600 **Pong via ALE**, distinct from thrust's
> existing in-tree toy `pong` self-play env (`src/env/games/pong`,
> `examples/games/pong/`). The ALE Pong is a pixel-observation ROM environment;
> the in-tree one is a hand-coded low-dimensional self-play env. They share a
> name only.

Justification on the three axes the epic named:

- **Action space (small):** Pong's minimal action set is 6 (`NOOP`, `FIRE`,
  `RIGHT`, `LEFT`, `RIGHTFIRE`, `LEFTFIRE`), of which only up/down are
  behaviorally meaningful — among the smallest in the ALE suite. Breakout is 4
  actions, comparably small, so action-space size alone does not separate them.

- **Fast, decisive learning signal:** Pong scores ±1 per point to 21, giving a
  frequent reward signal and — critically — a **binary success criterion**: the
  random-policy floor is roughly −21 (lose every point), so *any positive score
  is decisive evidence the agent learned to beat the hard-coded opponent.* This
  matches the epic's own framing ("Pong: positive score is decisive"). Breakout's
  reward is sparser (brick hits) and requires a longer credit-assignment horizon,
  so time-to-first-positive-signal is longer and the success threshold is less
  crisp.

- **Published baselines (cited):** Nature DQN (Mnih et al. 2015, *Human-level
  control through deep reinforcement learning*, Extended Data Table 2) reports
  **Pong ≈ 18.9** (human ≈ 9.3, random ≈ −20.7); PPO (Schulman et al. 2017,
  *Proximal Policy Optimization Algorithms*, Atari results) reaches **≈ 20.7 on
  Pong** after ~40M frames. For contrast, Nature DQN scores **Breakout ≈ 401**.
  Both games have well-published DQN/PPO baselines; Pong's near-ceiling positive
  score is the cleaner Phase-4 target.

> **Citation confidence:** the ~18.9 (DQN) / ~20.7 (PPO) Pong figures and the
> ~401 Breakout figure are from the cited papers and are widely reproduced;
> treat the exact decimals as approximate. Phase 4 (#329) should cite the score
> it actually reproduces, not this note's numbers.

**Expected time-to-positive-signal on alc-2:** a Nature-scale DQN typically
crosses zero on Pong within the low-single-digit millions of frames (well before
the tens of millions needed to approach ceiling), making it a practical target
for a single operator-gated GPU run (the #329 budget, sibling to #134). This is
an order-of-magnitude expectation from the literature, not a measured thrust
result — Phase 4 owns the honest learning-curve report.

---

## Phase 2–4 Cargo.toml and CI prescription

This is the section #325–#330 builders reference. It mirrors the
`env-bucket-brigade` precedent, **simplified** because Option D adds no
dependency.

### Cargo.toml — feature declaration

Option D needs **no** `[dependencies]` entry. The feature is a pure compile-time
flag:

```toml
# [features]
# Atari (ALE) env adapter. Option D (subprocess via ale-py) — see
# docs/ALE_BINDING_STRATEGY.md. No native/path dependency: the emulator runs as
# an external `ale-py` process invoked at runtime. Compose with `training`,
# e.g. `--features "training,env-atari"`.
env-atari = []
```

Contrast with `env-bucket-brigade = ["bucket-brigade-core"]`, which pulls a
path-dep and therefore requires the publish `sed` workaround. `env-atari` needs
none of that.

### Source gating pattern (exact, mirrors `env-bucket-brigade`)

Follow `src/env/games/mod.rs`'s existing pattern — `#[cfg(feature = "…")]` on
both the module declaration and the re-export:

```rust
#[cfg(feature = "env-atari")]
pub mod atari;

#[cfg(feature = "env-atari")]
pub use atari::AtariEnv;
```

### CI treatment

| Job | `env-bucket-brigade` today | Prescribed `env-atari` treatment |
|-----|----------------------------|----------------------------------|
| `clippy` | Extra step: `--features "training,env-bucket-brigade"` | **Add** an equivalent step: `cargo clippy --all-targets --features "training,env-atari" -- -D warnings`. **Unconditionally safe** — Option D pulls no native dep, so the Rust side compiles on a bare runner with no ALE present. |
| `test_linux` / `test_macos` | Not included (only `--features training`) | Unit tests that need no live emulator (protocol serialization, obs-shape/stacking math) run in the normal `--features training,env-atari` matrix. Integration tests that spawn `ale-py` are **gated behind a runtime "is `ale-py` present?" skip** so the default matrix stays green without Python. Optionally add a *separate* opt-in job with a `pip install ale-py` setup step for the live path. |
| `check_no_training` | Passes (feature off) | Unchanged — `env-atari` is not in the default set. |
| `docs` | Automatic via `cargo doc --no-deps --all-features` | Automatic and safe: `--all-features` enables `env-atari`, but with no native dep the module docs build anywhere. No change needed. |
| `publish_dry_run` | Requires `sed` to comment out the path dep + feature (`RELEASING.md` Step 2.5) | **No `sed` workaround needed** — `env-atari` has no path/native dep. Extend only the tarball-audit `grep` to also reject `\.(bin|rom|a26)$` (ROM belt-and-suspenders). |
| `security_audit` | Unchanged | Unchanged — no new Rust dep to audit. |

### Tarball `exclude`

No new entry required. Option D vendors nothing; the design note is intentionally
kept in the tarball for documentation value; ROMs are excluded-by-audit (they are
never in the tree to begin with).

---

## Phase 2–4 scope: confirmed or amended

The verdict (Option D, subprocess) changes the *binding mechanism* but not the
overall shape of the epic. One paragraph per downstream issue. Amendment comments
have been posted to the affected issues and are linked in the final section.

- **#325 — AtariEnv adapter — AMENDED (binding mechanism).** The adapter is a
  **subprocess client** (spawn `ale-py`, framed length-prefixed binary protocol
  over stdin/stdout) implementing the `Environment` trait (`src/env/mod.rs:64`),
  **not** an FFI binding or a wrapper crate. `env-atari = []` (no `[dependencies]`
  line, no path dep, no publish `sed` workaround). The seeding/determinism
  contract is unchanged: seeded reset forwarded over the protocol, reproducible
  episodes, unit tests for obs shape, stacking semantics, and determinism under
  seed. Everything except "how the bytes get to ALE" is as originally scoped.

- **#326 — Machado et al. preprocessing — CONFIRMED (clarification).** The
  grayscale + 84×84 downsample + max-over-consecutive-frames + frame-stack-4 +
  sticky-actions (p=0.25) pipeline stands as specified. Clarification: prefer
  doing the preprocessing **Rust-side** (receive raw 210×160 frames over the
  protocol, do grayscale/resize/max/stack in Rust) rather than delegating to
  `ale-py`'s built-in wrappers, to keep determinism under thrust's control and
  keep the protocol thin. No scope change.

- **#327 — Nature-DQN-scale CNN policy — CONFIRMED (unchanged).** Binding-agnostic.
  The observation is 84×84×4 regardless of transport; the Conv
  32×8×8/4 → 64×4×4/2 → 64×3×3/1 → FC 512 architecture (actor-critic + Q
  variants) is unaffected by the Option D decision. Parallelizable with Phase 2
  as before.

- **#328 — Large-net throughput benches + BURN_BACKENDS.md — CONFIRMED
  (clarification).** Unchanged in scope. Clarification: benchmark the **policy**
  (forward/backward on synthetic 84×84×4 batches) **in isolation** from env
  stepping, so the CPU/wgpu/cuda crossover measurement is not contaminated by
  subprocess IPC. This keeps the `docs/BURN_BACKENDS.md` re-measurement an honest
  test of the *network*, which is the crossover trigger's actual subject.

- **#329 — Reference-game training run — CONFIRMED (runbook addition).**
  Reference game is **Pong** (Q3). Runtime addition: the alc-2 runbook must
  include `pip install ale-py` (+ ROM acquisition) as a prerequisite, since the
  env is a subprocess. Success criterion unchanged and now crisp: a positive Pong
  score. Operator-gated long GPU run as before.

- **#330 — Fire downstream triggers — CONFIRMED (unchanged).** Re-triage
  #270/#272/#281 and update #305. Nothing about the Option D decision changes
  this closeout phase.

---

## Doc build verification

- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` — expected to pass unchanged:
  this note is a standalone markdown file, adds no Rust doc comments, and is not
  referenced from any `//!`/`///` doc comment, so it cannot introduce a broken
  intra-doc link. (Markdown itself is not doc-tested.)
- Markdown links in this file are relative to `docs/` (`./FP16_FEASIBILITY.md`,
  `./DISTRIBUTED_TRAINING_DESIGN.md`, `./BURN_BACKENDS.md`) or absolute GitHub
  issue URLs; all sibling docs exist in the tree.
- `git diff --stat` for the Phase-1 PR shows **exactly one new file**:
  `docs/ALE_BINDING_STRATEGY.md`. No `Cargo.toml`, `src/`, `ci.yml`, or
  `Cargo.lock` changes.

---

## Posted amendment comments

Per the epic protocol, scope amendments were posted to the affected issues and
are linked here on completion of the Phase-1 PR:

- **#325** — binding mechanism amended to subprocess (Option D); see the posted
  comment.
- **#326** — preprocessing clarified to Rust-side; see the posted comment.
- **#328** — bench-the-policy-in-isolation clarification; see the posted comment.
- **#329** — Pong confirmed + `ale-py` runbook prerequisite; see the posted
  comment.

(#327 and #330 are confirmed unchanged and need no amendment.)
