# Version Configuration

This document specifies the working version combinations for local development and GPU training.

## Local Development (macOS)

- **Operating System**: macOS 14+ (Darwin)
- **Rust**: nightly-2025-xx-xx (edition 2024 support required)
- **PyTorch**: 2.9.0 (via Homebrew: `brew install pytorch`)
- **tch-rs**: 0.22.0
- **CUDA**: N/A (CPU/MPS only)

### Setup
```bash
# Install PyTorch via Homebrew
brew install pytorch

# Set environment variables
export LIBTORCH="$(brew --prefix pytorch)/lib"
export DYLD_LIBRARY_PATH="$LIBTORCH:$DYLD_LIBRARY_PATH"
export LIBTORCH_USE_PYTORCH=1

# Build and run
cargo +nightly build --release
cargo +nightly run --example train_cartpole
```

## GPU Training (Linux - rwalters-sandbox-2)

- **Operating System**: Linux (Ubuntu/Debian)
- **GPU**: NVIDIA L4 (23GB VRAM)
- **CUDA**: 12.1 (CUDA 12.8 driver compatible)
- **Rust**: nightly-2025-xx-xx
- **PyTorch**: 2.2.0+cu121
- **tch-rs**: 0.15.0

### Setup on GPU Machine
```bash
# Create Python venv
python3 -m venv venv
source venv/bin/activate

# Install PyTorch 2.2.0 with CUDA 12.1 support
pip install torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install Rust nightly
rustup install nightly
rustup default nightly

# Edit Cargo.toml to use tch = "0.15"
# (GPU machine copy can differ from local)

# Build with proper environment
source venv/bin/activate
export LIBTORCH="$(pwd)/libtorch"
export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1

cargo build --release
```

### Training Scripts
Use the provided helper scripts on the GPU machine:
```bash
# Run training (handles all environment setup)
./scripts/gpu-train.sh train_cartpole_long

# Check status
./scripts/gpu-status.sh

# Build without running
./scripts/gpu-build.sh --release
```

## Why Different Versions?

### Local (tch 0.22 + PyTorch 2.9)
- **Latest stable**: Uses the newest tch-rs with latest PyTorch
- **Edition 2024 support**: Required for nightly Rust features
- **Development speed**: Newer versions often have better compilation times

### GPU (tch 0.15 + PyTorch 2.2.0)
- **CUDA availability**: PyTorch 2.2.0 is widely available with CUDA 12.1
- **Stability**: tch 0.15 is battle-tested with PyTorch 2.2.0
- **Compatibility**: Proven working combination for GPU training

## Version Compatibility Matrix

| tch-rs | PyTorch | Notes |
|--------|---------|-------|
| 0.15.0 | 2.2.0   | ✅ Stable, used on GPU |
| 0.16.0 | 2.3.0   | ⚠️ Untested |
| 0.17.0 | 2.4.0   | ⚠️ Untested |
| 0.22.0 | 2.9.0   | ✅ Latest, used locally |

## Troubleshooting

### Compilation Errors
1. **Version mismatch**: Ensure PyTorch and tch versions match per this document
2. **Clean build**: `rm -rf target/` and rebuild
3. **Environment**: Verify `LIBTORCH_USE_PYTORCH=1` is set

### GPU Not Detected
1. **Check CUDA**: `nvidia-smi` should show GPU
2. **LD_LIBRARY_PATH**: Must include PyTorch lib directory
3. **Device check**: Code should show `Device: Cuda(0)` in logs

### Training Issues
1. **NaN/Inf errors**: Fixed by numerical stability patches in `MlpPolicy`
2. **Episode reset**: Ensure environments reset after episode end
3. **Performance**: Should see ~1000 steps/second on GPU

## Future Updates

When updating versions:
1. Update both local and GPU to matching versions if possible
2. Test locally first
3. Document any compatibility issues here
4. Update scripts if environment variables change
