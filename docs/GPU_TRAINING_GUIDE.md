# GPU Training Guide

## Quick Start

**TL;DR**: Use `./scripts/train-snake-remote.sh` which handles everything automatically.

## Understanding the GPU Training Stack

### The Problem
Training with CUDA requires careful coordination between:
1. PyTorch C++ libraries (libtorch)
2. Rust bindings (tch-rs)
3. Your training code
4. The linker

### Version Requirements

**CRITICAL**: tch-rs 0.22.0 requires PyTorch 2.9.0 exactly. Mismatches cause compilation errors.

Check versions on remote:
```bash
ssh rwalters-sandbox-2 'python3 -c "import torch; print(torch.__version__)"'
# Should show: 2.9.0+cu128
```

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│  Your Rust Code (examples/train_*.rs)      │
├─────────────────────────────────────────────┤
│  tch-rs (Rust bindings) v0.22.0            │
├─────────────────────────────────────────────┤
│  PyTorch C++ (libtorch) v2.9.0             │
├─────────────────────────────────────────────┤
│  CUDA Libraries (cu128)                     │
├─────────────────────────────────────────────┤
│  NVIDIA GPU (L4)                            │
└─────────────────────────────────────────────┘
```

## Required Environment Variables

### 1. LIBTORCH_USE_PYTORCH=1
Tells tch-rs to use Python's PyTorch instead of downloading standalone libtorch.

### 2. LD_LIBRARY_PATH
Points to PyTorch's lib directory so libraries can be found at runtime:
```bash
export LD_LIBRARY_PATH=$(python3 -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
```

### 3. LD_PRELOAD (no longer required as of issue #9)
The crate-root `build.rs` now keeps `libtorch_cuda.so` and
`libc10_cuda.so` in the binary's `DT_NEEDED` list automatically, so
setting `LD_PRELOAD` is no longer necessary. If you ever need to fall
back (e.g. a custom libtorch that the build.rs can't auto-detect):
```bash
export LD_PRELOAD="$PYTORCH_LIB/libtorch_cuda.so:$PYTORCH_LIB/libtorch.so"
```

Historical symptom that this used to fix:
```
undefined symbol: _ZNK3c107SymBool14guard_or_falseEPKcl
```

### 4. RUSTFLAGS (no longer required as of issue #9)
The `build.rs` re-emits `-Wl,--no-as-needed -ltorch_cuda -lc10_cuda` at
the correct position on the linker command line, so the historical
`RUSTFLAGS="-C link-arg=-Wl,--no-as-needed"` workaround is unnecessary.
Kept here for historical reference --- still safe to set, just redundant.

### 5. THRUST_EXPECT_CUDA=1 (CRITICAL for GPU runs)
Turns silent CPU fallback into a hard error (exit 2) if for any reason
`tch::Cuda::is_available()` ends up false. All three GPU wrapper scripts
set this automatically.

### 6. LIBTORCH_BYPASS_VERSION_CHECK=1
Allows minor version mismatches (use carefully).

## Why Use Nightly Rust?

Some of our code uses nightly-only features. Always use:
```bash
cargo +nightly build --release
cargo +nightly run --release
```

## Training Workflows

### Option 1: Automated (Recommended)
Use the all-in-one script:
```bash
./scripts/train-snake-remote.sh
```

This script:
1. SSHs into rwalters-sandbox-2
2. Pulls latest code
3. Sets up all environment variables correctly
4. Builds with `cargo +nightly`
5. Runs training with CUDA
6. Exports the model
7. Downloads it to `web/public/`

### Option 2: Manual on Remote
```bash
# 1. SSH to remote
ssh rwalters-sandbox-2

# 2. Navigate to project
cd ~/thrust

# 3. Pull latest code
git pull

# 4. Set up environment (build.rs handles the linker; just point at libtorch)
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$(python3 -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
export THRUST_EXPECT_CUDA=1   # hard-fail if tch falls back to Cpu

# 5. Build with nightly
cargo +nightly build --example train_snake_multi_v2 --release

# 6. Run training
cargo +nightly run --example train_snake_multi_v2 --release -- --mode shared --epochs 1000 --cuda
```

### Option 3: Using gpu-train.sh on Remote
```bash
ssh rwalters-sandbox-2
cd ~/thrust
./scripts/gpu-train.sh train_snake_multi_v2
```

## Verifying GPU Usage

### On the remote machine:
```bash
# Check if training process exists
ps aux | grep train_snake

# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# Check CUDA is available
python3 -c 'import torch; print(torch.cuda.is_available())'
```

### Expected nvidia-smi output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA L4           Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   45C    P0    35W /  72W |   1234MiB / 23034MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
```

Look for:
- **GPU-Util**: Should be 80-100% during training
- **Memory-Usage**: Should be several GB
- **Pwr:Usage**: Should be near cap (72W for L4)

## Common Issues

### Issue 1: "Using CPU" instead of GPU
**Cause**: `build.rs` did not detect a CUDA-enabled libtorch at build time,
or `LIBTORCH_USE_PYTORCH=1` was not set. With `THRUST_EXPECT_CUDA=1`, this
now hard-fails with exit code 2 and a diagnostic message.
**Fix**:
1. Verify Python has CUDA: `python3 -c 'import torch; print(torch.cuda.is_available())'`.
2. `cargo clean` (build.rs is cached) and rebuild with `LIBTORCH_USE_PYTORCH=1`.
3. Verify with `ldd target/release/examples/<bin> | grep torch_cuda`.

### Issue 2: Symbol lookup errors
**Symptoms**:
```
symbol lookup error: undefined symbol: _ZNK3c107SymBool...
```
**Cause**: The CUDA libraries were not linked. This used to be the
"`--as-needed` strips libtorch_cuda" bug; `build.rs` should now prevent it.
**Fix**: Confirm `ldd | grep torch_cuda` shows both libtorch_cuda and
c10_cuda. If not, see Issue 1. As a last resort:
```bash
export LD_PRELOAD="$PYTORCH_LIB/libtorch_cuda.so:$PYTORCH_LIB/libtorch.so"
```

### Issue 3: PyTorch version mismatch
**Symptoms**: Compilation errors in torch-sys about missing/changed APIs
**Cause**: tch-rs 0.22 requires PyTorch 2.9.0 exactly
**Fix**:
```bash
ssh rwalters-sandbox-2
pip3 install --break-system-packages torch==2.9.0
```

### Issue 4: Build succeeds but runtime crash / CPU fallback
**Cause**: Binary was built against a libtorch that lacks CUDA, or the
build.rs failed to detect CUDA so the link args were not emitted.
**Fix**: Clean and rebuild with `LIBTORCH_USE_PYTORCH=1` set:
```bash
cargo clean
export LIBTORCH_USE_PYTORCH=1
cargo build --example <bin> --release
```

### Issue 5: "Cannot find libtorch_cpu.so"
**Cause**: LD_LIBRARY_PATH not set at runtime (the `build.rs` adds an
`-rpath`, but only for the CUDA lib dir).
**Fix**: Set LD_LIBRARY_PATH before running (not just building).

## Remote Machine Setup

The remote machine (rwalters-sandbox-2) should have:
- Ubuntu 24.04
- Python 3.12 with PyTorch 2.9.0+cu128
- CUDA 12.8
- Rust nightly toolchain
- NVIDIA L4 GPU

### Verifying Setup
```bash
ssh rwalters-sandbox-2 << 'EOF'
echo "Python version:"
python3 --version

echo -e "\nPyTorch version:"
python3 -c 'import torch; print(torch.__version__)'

echo -e "\nCUDA available:"
python3 -c 'import torch; print(torch.cuda.is_available())'

echo -e "\nGPU name:"
python3 -c 'import torch; print(torch.cuda.get_device_name(0))'

echo -e "\nRust nightly:"
cargo +nightly --version

echo -e "\nNVIDIA driver:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader
EOF
```

## Script Reference

### train-snake-remote.sh
**Purpose**: Complete automated training pipeline
**Use**: `./scripts/train-snake-remote.sh`
**GPU**: ✅ Yes (auto-configured)
**Nightly**: ✅ Yes

### gpu-train.sh
**Purpose**: Run training on remote machine
**Use**: `ssh rwalters-sandbox-2 'cd ~/thrust && ./scripts/gpu-train.sh train_snake_multi_v2'`
**GPU**: ✅ Yes (with recent fixes)
**Nightly**: ⚠️ **NO - needs to be updated**

### train-gpu.sh
**Purpose**: Local GPU training (if you have CUDA locally)
**Use**: `./scripts/train-gpu.sh train_snake_multi_v2`
**GPU**: ✅ Yes
**Nightly**: ❌ No (uses stable)

### remote-optimize.sh
**Purpose**: Hyperparameter optimization on remote GPU
**Use**: `./scripts/remote-optimize.sh --trials 50`
**GPU**: ✅ Yes
**Nightly**: ✅ Yes

## Troubleshooting Checklist

If GPU training isn't working, verify each step:

- [ ] PyTorch version is 2.9.0: `python3 -c 'import torch; print(torch.__version__)'`
- [ ] CUDA is available: `python3 -c 'import torch; print(torch.cuda.is_available())'`
- [ ] Using nightly Rust: `cargo +nightly --version`
- [ ] LIBTORCH_USE_PYTORCH=1 is set (build.rs needs this to detect CUDA)
- [ ] LD_LIBRARY_PATH points to PyTorch lib directory
- [ ] `ldd target/release/examples/<bin> | grep torch_cuda` is non-empty
- [ ] Ran `cargo clean` after changing build environment
- [ ] Built with `cargo +nightly build --release`
- [ ] Training code passes `--cuda` flag
- [ ] THRUST_EXPECT_CUDA=1 set (fails loudly if anything regresses)
- [ ] nvidia-smi shows GPU utilization during training

LD_PRELOAD and RUSTFLAGS=`--no-as-needed` are no longer required as of
issue #9. The crate-root `build.rs` handles linker behavior.

## Performance Expectations

### L4 GPU (on rwalters-sandbox-2)
- **Steps/second**: 3-6M (CartPole with 256 envs)
- **Training time**: ~10-30 minutes for 1000 epochs (Snake)
- **GPU utilization**: 80-100% during training
- **Memory usage**: 1-4GB depending on model size

### CPU (if GPU disabled)
- **Steps/second**: 100-500K
- **Training time**: 2-10 hours for same workload
- **Much slower**: 10-60x slowdown vs GPU

## See Also

- [REMOTE_TRAINING.md](REMOTE_TRAINING.md) - General remote training workflow
- [GPU_SETUP.md](GPU_SETUP.md) - Local GPU setup
- [tch-rs documentation](https://github.com/LaurentMazare/tch-rs)
