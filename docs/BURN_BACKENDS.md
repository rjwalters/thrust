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
