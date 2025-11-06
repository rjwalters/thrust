# LibTorch Setup Guide

This guide covers installing LibTorch (PyTorch C++ backend) for use with tch-rs.

## Quick Install (macOS)

### Option 1: Homebrew (Recommended for macOS)

```bash
# Install PyTorch via Homebrew (includes libtorch)
brew install pytorch

# Set environment variable to use the Homebrew PyTorch
export LIBTORCH_USE_PYTORCH=1

# Add to your shell profile (~/.zshrc or ~/.bashrc)
echo 'export LIBTORCH_USE_PYTORCH=1' >> ~/.zshrc

# Reload shell
source ~/.zshrc
```

That's it! Now `tch-rs` will automatically use the Homebrew-installed PyTorch.

### Option 2: Download Pre-built LibTorch

```bash
# Download LibTorch (CPU version for macOS)
cd /tmp
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.1.2.zip
# Or for x86:
# wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-2.1.2.zip

# Extract
unzip libtorch-macos-*.zip

# Move to a permanent location
sudo mv libtorch /usr/local/

# Set environment variable
export LIBTORCH=/usr/local/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=${LIBTORCH}/lib:$DYLD_LIBRARY_PATH

# Add to your shell profile (~/.zshrc or ~/.bashrc)
echo 'export LIBTORCH=/usr/local/libtorch' >> ~/.zshrc
echo 'export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH' >> ~/.zshrc
echo 'export DYLD_LIBRARY_PATH=${LIBTORCH}/lib:$DYLD_LIBRARY_PATH' >> ~/.zshrc
```

### Option 3: Use Existing PyTorch Installation

If you already have PyTorch installed via pip/conda:

```bash
# Find your PyTorch installation
python3 -c "import torch; print(torch.__path__[0])"

# Set LIBTORCH_USE_PYTORCH
export LIBTORCH_USE_PYTORCH=1

# Add to shell profile
echo 'export LIBTORCH_USE_PYTORCH=1' >> ~/.zshrc
```

## Quick Install (Linux)

```bash
# Download LibTorch (choose your CUDA version or CPU)
cd /tmp

# CPU version:
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcpu.zip

# CUDA 11.8:
# wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcu118.zip

# CUDA 12.1:
# wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcu121.zip

# Extract
unzip libtorch-*.zip

# Move to permanent location
sudo mv libtorch /usr/local/

# Set environment variables
export LIBTORCH=/usr/local/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

# Add to shell profile
echo 'export LIBTORCH=/usr/local/libtorch' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

## Verify Installation

```bash
# Check that libtorch is accessible
ls $LIBTORCH/lib/

# Try building thrust
cd /path/to/thrust
cargo check

# Should see: "Compiling tch..." and "Compiling torch-sys..."
```

## Troubleshooting

### Issue: "Cannot find libtorch"

**Solution**: Make sure `LIBTORCH` environment variable is set:
```bash
echo $LIBTORCH
# Should print: /usr/local/libtorch (or your path)
```

### Issue: "dyld: Library not loaded"  (macOS)

**Solution**: Add libtorch to dynamic library path:
```bash
export DYLD_LIBRARY_PATH=${LIBTORCH}/lib:$DYLD_LIBRARY_PATH
```

### Issue: "error while loading shared libraries" (Linux)

**Solution**: Update LD_LIBRARY_PATH:
```bash
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
# Or add to /etc/ld.so.conf.d/libtorch.conf and run ldconfig
```

### Issue: CUDA version mismatch

**Solution**: Download the correct LibTorch version matching your CUDA installation:
```bash
# Check your CUDA version
nvcc --version

# Download matching LibTorch (11.8, 12.1, etc.)
```

## Version Compatibility

| Rust tch-rs | LibTorch | PyTorch |
|-------------|----------|---------|
| 0.16.x      | 2.1.x    | 2.1.x   |
| 0.17.x      | 2.2.x    | 2.2.x   |

**Recommendation**: Use LibTorch 2.1.2 with tch-rs 0.16 for maximum compatibility.

## Alternative: Docker

If you prefer a containerized setup:

```dockerfile
FROM rust:latest

# Install LibTorch
RUN cd /tmp && \
    wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcpu.zip && \
    unzip libtorch-*.zip && \
    mv libtorch /usr/local/

ENV LIBTORCH=/usr/local/libtorch
ENV LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

# Build thrust
WORKDIR /workspace
COPY . .
RUN cargo build --release
```

## Next Steps

Once LibTorch is installed:

1. Uncomment `tch = "0.16"` in `Cargo.toml`
2. Uncomment the full MLP implementation in `src/policy/mlp.rs`
3. Run `cargo check` to verify
4. Run `cargo test` to test the policy

## References

- [tch-rs Repository](https://github.com/LaurentMazare/tch-rs)
- [PyTorch Downloads](https://pytorch.org/get-started/locally/)
- [LibTorch C++ Documentation](https://pytorch.org/cppdocs/)
