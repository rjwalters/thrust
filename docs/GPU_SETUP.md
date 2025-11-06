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

## The CUDA Detection Problem

When using `tch-rs` (Rust bindings for PyTorch), you may encounter a situation where:

- ✅ `nvidia-smi` shows your GPU is available
- ✅ Python PyTorch detects CUDA: `torch.cuda.is_available()` returns `True`
- ✅ `torch-sys` correctly emits `cargo:rustc-link-lib=torch_cuda` during build
- ❌ But `tch::Cuda::is_available()` returns `false` in your Rust binary

### Root Cause

The Rust linker drops `libtorch_cuda.so` as "unused" even though it's needed at runtime:

1. `torch-sys` correctly detects CUDA at build time and emits link directives
2. However, your binary doesn't directly reference symbols from `libtorch_cuda.so`
3. The linker removes it as an optimization, even with `--no-as-needed`
4. At runtime, `libtorch.so` tries to load CUDA support but can't find the library
5. Result: `torch::cuda::is_available()` returns false

### The Solution: LD_PRELOAD

The only reliable solution is to **force-load the CUDA libraries at runtime** using `LD_PRELOAD`:

```bash
# Get PyTorch lib directory
TORCH_LIB=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")

# Set library search path
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH}"

# CRITICAL: Preload CUDA libraries before running
export LD_PRELOAD="${TORCH_LIB}/libtorch_cuda.so:${TORCH_LIB}/libtorch.so"

# Now CUDA will be available
cargo run --example your_example --release
```

This works because:
- `LD_PRELOAD` loads the libraries into memory before your program starts
- When `libtorch.so` initializes, it finds the CUDA implementation already loaded
- The CUDA detection logic in `torch::cuda::is_available()` succeeds

## Environment Setup

The `train-gpu.sh` script handles all environment setup automatically:

```bash
#!/bin/bash
# Set environment to use PyTorch from pip
export LIBTORCH_USE_PYTORCH=1

# Set LD_LIBRARY_PATH to include PyTorch lib directory
TORCH_LIB=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH}"

# CRITICAL: Preload CUDA libraries to ensure they're loaded at runtime
export LD_PRELOAD="${TORCH_LIB}/libtorch_cuda.so:${TORCH_LIB}/libtorch.so"

# CRITICAL: Force linker to keep CUDA library dependency
export RUSTFLAGS="-C link-arg=-Wl,--no-as-needed"

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
export LD_PRELOAD="${TORCH_LIB}/libtorch_cuda.so:${TORCH_LIB}/libtorch.so"
cargo run --example test_cuda --release
```

Expected output:
```
🔍 Testing CUDA availability in tch-rs
Device::cuda_if_available() = Cuda(0)
tch::Cuda::is_available() = true
tch::Cuda::device_count() = 1
✅ Successfully created tensor on CUDA
```

## Troubleshooting

### Error: "CUDA not available in PyTorch"

**Problem**: PyTorch is installed without CUDA support.

**Solution**: Reinstall PyTorch with CUDA:
```bash
./scripts/setup-libtorch.sh
```

This script automatically detects your GPU and installs the correct PyTorch version with CUDA support.

### Error: "cannot open shared object file: libtorch_cuda.so"

**Problem**: `LD_PRELOAD` path is incorrect or PyTorch is not installed.

**Solution**:
1. Verify PyTorch installation: `python3 -c "import torch; print(torch.__file__)"`
2. Check library exists: `ls $(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")/libtorch_cuda.so`
3. Reinstall if needed: `./scripts/setup-libtorch.sh`

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

Works out of the box with the `LD_PRELOAD` solution.

### macOS

CUDA is not supported on macOS. Use Metal Performance Shaders (MPS) instead:
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

Note: `LD_PRELOAD` is Linux-specific. On Windows, ensure the PyTorch DLLs are in your PATH.

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
- [LD_PRELOAD Explanation](https://man7.org/linux/man-pages/man8/ld.so.8.html)

## Key Discoveries

This solution was discovered through systematic debugging:

1. Verified `torch-sys` emits correct link directives ✅
2. Found binary only links `libtorch_cpu.so`, not `libtorch_cuda.so` ❌
3. Discovered linker drops CUDA library as "unused" even with `--no-as-needed`
4. Solution: Use `LD_PRELOAD` to force-load libraries at runtime ✅

The `train-gpu.sh` script encapsulates this hard-won knowledge so you don't have to rediscover it.
