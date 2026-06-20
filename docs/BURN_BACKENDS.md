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

### Committing the numbers

Capturing and committing the actual CPU-vs-GPU throughput table requires running
the harness on operator GPU hardware, which is **operator-gated** and tracked
separately (issue #184). This document intentionally ships only the run
instructions; the measured numbers land via that issue.

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
