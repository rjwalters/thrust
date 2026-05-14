# GPU Training Setup Guide

This guide explains how to set up GPU-accelerated training with CUDA for Thrust RL.

## Quick Start

```bash
# 1. Install PyTorch with CUDA support
./scripts/setup-libtorch.sh

# 2. Run GPU training
./scripts/train-gpu.sh train_cartpole_best
```

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA 12.x installed
- Python 3.8+ with pip
- Rust nightly toolchain

## CUDA Linking: How It Works (and How It Used to Break)

When using `tch-rs` (Rust bindings for PyTorch) on Linux, the binary needs to
link against `libtorch_cuda.so` and `libc10_cuda.so` even though no Rust code
directly references symbols in those libraries. This is because libtorch
performs lazy CUDA backend registration that only initializes when those
shared objects are present in the process's address space at startup.

### The historical problem (fixed by issue #9)

Modern Linux distros ship `ld` with `--as-needed` on by default. Under that
flag, any `DT_NEEDED` entry for a shared library from which no symbol is
referenced is dropped at link time. Because thrust-rl only calls `tch`
high-level APIs and `tch` itself only references `libtorch_cpu` / `libc10`
symbols, `-ltorch_cuda` was silently stripped from the final binary. The
runtime symptom was a `tch::Cuda::is_available() == false` on a CUDA-
equipped box, with no error message --- the training run then proceeded on
CPU at roughly 1/5 of the expected throughput.

Workarounds that *did not* fully solve the problem:

- `RUSTFLAGS="-C link-arg=-Wl,--no-as-needed"` was positionally ineffective:
  the flag was emitted at a point in the linker command line that did not
  cover the `-ltorch_cuda` arg emitted later by `torch-sys`.
- `LD_PRELOAD=libtorch_cuda.so:libtorch.so` did work at runtime but was a
  hidden foot-gun: anyone who ran the binary outside the wrapper script
  silently got a CPU run.

### The fix (current behavior)

A crate-root `build.rs` now emits positional link args that bracket the
CUDA libraries with `-Wl,--no-as-needed` / `-Wl,--as-needed`, plus an
`-rpath` entry pointing at the libtorch lib dir. From the build script's
perspective:

```rust
println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
println!("cargo:rustc-link-lib=torch_cuda");
println!("cargo:rustc-link-lib=c10_cuda");
println!("cargo:rustc-link-arg=-Wl,--as-needed");
println!("cargo:rustc-link-arg=-Wl,-rpath,<libtorch_lib_dir>");
```

This is emitted only when:

1. The target OS is Linux (macOS / Windows builds get nothing).
2. The `training` feature is enabled.
3. A CUDA-enabled libtorch is detected on the build host (either via
   `LIBTORCH_USE_PYTORCH=1` + a PyTorch with CUDA, or `LIBTORCH=...` pointing
   at a libtorch directory that contains `libtorch_cuda.so` and
   `libc10_cuda.so`).

CPU-only builds and non-Linux builds are no-ops.

### Verifying the fix

```bash
cargo build --release --example test_cuda
ldd target/release/examples/test_cuda | grep -E '(torch_cuda|c10_cuda)'
# Expected: two lines, one for each CUDA library, neither marked "not found".

readelf -d target/release/examples/test_cuda | grep NEEDED
# Expected: includes libtorch_cuda.so and libc10_cuda.so.

./target/release/examples/test_cuda
# Expected: Device::cuda_if_available() = Cuda(0)
#           tch::Cuda::is_available() = true
```

No `LD_PRELOAD` is required.

### Opting out of the fix

If your environment is unusual and the link-arg emission causes problems,
set `THRUST_DISABLE_CUDA_LINK_FIX=1` before building. The `build.rs` will
become a no-op and you can manage CUDA linking yourself.

### The Option D guard: `THRUST_EXPECT_CUDA`

In addition to the link-time fix, examples can use a runtime guard that
hard-fails if `tch::Cuda::is_available()` returns false but the user
expected CUDA:

```rust
use thrust_rl::utils::cuda::ensure_cuda_if_expected;

let device = Device::cuda_if_available();
ensure_cuda_if_expected(device); // exits(2) if THRUST_EXPECT_CUDA=1 and device == Cpu
```

The wrapper scripts (`scripts/train-gpu.sh`, `scripts/gpu-train.sh`,
`scripts/train-snake-remote.sh`) all `export THRUST_EXPECT_CUDA=1` so that
silent CPU fallback becomes a loud, fast failure with exit code 2.

## Environment Setup

The `train-gpu.sh` script handles all environment setup automatically:

```bash
#!/bin/bash
# Set environment to use PyTorch from pip
export LIBTORCH_USE_PYTORCH=1

# Set LD_LIBRARY_PATH to include PyTorch lib directory.
# (build.rs also embeds an rpath, so this is largely belt-and-suspenders.)
TORCH_LIB=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH}"

# Make silent CPU fallback a hard error.
export THRUST_EXPECT_CUDA=1

# Build and run
cargo build --example "$EXAMPLE_NAME" --release
cargo run --example "$EXAMPLE_NAME" --release
```

## Verifying CUDA Support

Test CUDA detection with the diagnostic example:

```bash
# With GPU script (handles environment automatically)
./scripts/train-gpu.sh test_cuda

# Or manually
source venv/bin/activate
export LIBTORCH_USE_PYTORCH=1
TORCH_LIB=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH}"
cargo run --example test_cuda --release
```

Expected output:
```
Testing CUDA availability in tch-rs
Device::cuda_if_available() = Cuda(0)
tch::Cuda::is_available() = true
tch::Cuda::device_count() = 1
Successfully created tensor on CUDA
```

## Troubleshooting

### Error: "CUDA not available in PyTorch"

**Problem**: PyTorch is installed without CUDA support.

**Solution**: Reinstall PyTorch with CUDA:
```bash
./scripts/setup-libtorch.sh
```

This script automatically detects your GPU and installs the correct PyTorch version with CUDA support.

### `ldd ... | grep torch_cuda` is empty after a clean rebuild

**Problem**: `build.rs` did not detect CUDA at build time, so the link-arg
emission was skipped.

**Check**:
1. `python3 -c 'import torch; print(torch.cuda.is_available())'` returns `True`.
2. `LIBTORCH_USE_PYTORCH=1` is set in the build environment.
3. `cargo clean && cargo build --example test_cuda --release` (build.rs is
   cached, so you need a clean rebuild after changing the env).
4. Inspect the build.rs log via `CARGO_LOG=cargo::core::compiler::custom_build=debug cargo build ...`,
   or check `target/.../build/thrust-rl-*/output` for the link-arg directives.

**Fallback**: As a last resort, set the legacy `LD_PRELOAD`:
```bash
TORCH_LIB=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_PRELOAD="${TORCH_LIB}/libtorch_cuda.so:${TORCH_LIB}/libtorch.so"
```

### Error: "FATAL: THRUST_EXPECT_CUDA=1 but tch fell back to Device::Cpu"

This is the Option D guard firing. Follow the diagnostic steps printed by
the binary (mostly: `ldd | grep torch_cuda`, then verify the build env).
Exit code is 2.

### Error: "libtorch_cuda.so: undefined symbol"

**Problem**: PyTorch version mismatch with `tch-rs` requirements.

**Solution**: Ensure you have PyTorch 2.9.0+ with CUDA 12.x:
```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
```

Should show: `PyTorch 2.9.0+cu128, CUDA 12.8`

### CUDA Available but Training Slow

**Problem**: Training is running on CPU despite CUDA being detected.

**Check**: Verify your training code actually uses the GPU device:
```rust
let device = Device::cuda_if_available();
println!("Training on: {:?}", device);

// Ensure tensors are created on the correct device
let tensor = Tensor::randn([2, 2], (Kind::Float, device));
```

## Platform-Specific Notes

### Linux (Ubuntu/Debian)

Works out of the box --- `build.rs` handles the linker behavior automatically.
No `LD_PRELOAD` needed.

### macOS

CUDA is not supported on macOS. The `build.rs` is a no-op on macOS so
nothing changes from a build-system perspective. Use Metal Performance
Shaders (MPS) instead:
```rust
let device = Device::Mps;  // For M1/M2/M3 Macs
```

Or train on CPU:
```rust
let device = Device::Cpu;
```

### Windows

Windows uses different environment variables:
```powershell
$env:LIBTORCH_USE_PYTORCH = "1"
$env:Path = "C:\path\to\python\Lib\site-packages\torch\lib;$env:Path"
```

Note: the `--as-needed` linker behavior is Linux-specific. The `build.rs`
is a no-op on Windows. Ensure the PyTorch DLLs are in your PATH.

## Performance Tips

1. **Batch Size**: Increase batch size to maximize GPU utilization
   ```rust
   let batch_size = 256;  // Larger batches for GPU
   ```

2. **Multiple Environments**: Run many parallel environments to keep GPU busy
   ```rust
   let num_envs = 64;  // More environments = better GPU utilization
   ```

3. **Mixed Precision**: Use FP16 for faster training (when supported)
   ```rust
   let tensor = Tensor::randn([2, 2], (Kind::Half, device));
   ```

4. **Profile GPU Usage**: Monitor with `nvidia-smi -l 1` during training

## References

- [tch-rs Documentation](https://docs.rs/tch/)
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
- [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- [ld(1) --as-needed semantics](https://man7.org/linux/man-pages/man1/ld.1.html)
- Issue [#9](https://github.com/rjwalters/thrust/issues/9) --- root cause + fix history

## Key Discoveries (Historical)

The build-script fix was the result of systematic debugging documented in
issue #9:

1. Verified `torch-sys` emits correct `cargo:rustc-link-lib=torch_cuda` directives.
2. Found binary only links `libtorch_cpu.so`, not `libtorch_cuda.so`.
3. Discovered the linker drops the CUDA libs as "unused" even with
   `RUSTFLAGS="-C link-arg=-Wl,--no-as-needed"` --- because the flag was at
   the wrong position on the link line.
4. Initial workaround: `LD_PRELOAD` at runtime. Worked but was fragile.
5. Proper fix (issue #9): re-emit `-Wl,--no-as-needed -ltorch_cuda
   -lc10_cuda -Wl,--as-needed` from a leaf-crate `build.rs`, where Cargo
   appends the link args at a position that *does* cover the explicit
   `-ltorch_cuda` arg.

The `train-gpu.sh` / `gpu-train.sh` / `train-snake-remote.sh` scripts still
exist and add a `THRUST_EXPECT_CUDA=1` belt-and-suspenders guard so any
future regression is loud rather than silent.
