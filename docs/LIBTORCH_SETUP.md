# LibTorch Setup Guide

This guide covers installing LibTorch (PyTorch C++ backend) for use with
`tch-rs` in Thrust. The current pinned versions (per `Cargo.toml` and
`VERSIONS.md`) are:

- **`tch-rs`**: `0.22.x`
- **LibTorch / PyTorch**: `2.9.x`

> **Heads up (macOS)**: `brew install pytorch` is **no longer the recommended
> path**. The Homebrew bottle links libtorch against specific minor versions
> of `protobuf` / `abseil` that drift out of sync with current Homebrew the
> moment any of them updates, producing `dyld: Library not loaded:
> libprotobuf.33.0.0.dylib` and similar errors at runtime. The two
> recommended paths below both sidestep Homebrew entirely.

## Quick Install (Recommended): pip torch in a venv

This pattern is identical to what the Linux GPU box uses and is immune to
Homebrew bottle drift on macOS because the pip wheel ships its own bundled
deps.

```bash
# One-shot: creates ./venv, pip installs torch>=2.9, writes .envrc.libtorch
./scripts/setup-libtorch.sh

# Load the env into your current shell (the script prints the exact exports
# it wrote; you can either source the file or copy the block into ~/.zshrc):
source .envrc.libtorch

# Verify cargo can find libtorch:
cargo check --features training
```

The wrapper scripts auto-detect `./venv` so you typically don't need to
re-source on each invocation:

```bash
./scripts/test.sh                    # cargo test, with libtorch env
./scripts/check.sh                   # fmt + clippy + test
./scripts/train-cpu.sh train_cartpole_modern
./scripts/train-gpu.sh train_cartpole_modern   # Linux + NVIDIA only
```

### What the venv path sets under the hood

```bash
source venv/bin/activate
export LIBTORCH_USE_PYTORCH=1                 # tch-rs reads torch from python pkg
export LIBTORCH_BYPASS_VERSION_CHECK=1        # tch is overly strict on patch versions
# macOS:
export DYLD_LIBRARY_PATH="$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${DYLD_LIBRARY_PATH:-}"
# Linux:
export LD_LIBRARY_PATH="$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"
```

> **macOS note**: macOS uses `DYLD_LIBRARY_PATH`, NOT `LD_LIBRARY_PATH`.
> On SIP-enabled systems, `DYLD_LIBRARY_PATH` is stripped from child
> processes of system binaries, but `cargo` / `rustc` / examples spawned
> by them are unaffected in practice.

## Alternative: Download self-contained libtorch (no Python)

If you want to avoid Python entirely, `pytorch.org` publishes a
self-contained libtorch zip that bundles its own protobuf / abseil:

```bash
./scripts/download-libtorch.sh           # default: 2.9.0 CPU, current arch

# Or pick a different version / CUDA tag:
PYTORCH_VERSION=2.9.0 CUDA_TAG=cu121 ./scripts/download-libtorch.sh
```

The script writes `./libtorch/` (gitignored) and prints the export block.
The wrapper scripts auto-detect `./libtorch` as a fallback when `./venv`
is absent.

Manual install:

```bash
# macOS arm64:
curl -L https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.9.0.zip -o /tmp/libtorch.zip
# macOS x86_64:
# curl -L https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-2.9.0.zip -o /tmp/libtorch.zip
# Linux x86_64 CPU:
# curl -L 'https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.9.0%2Bcpu.zip' -o /tmp/libtorch.zip

unzip /tmp/libtorch.zip -d "$(pwd)"
export LIBTORCH="$(pwd)/libtorch"
export DYLD_LIBRARY_PATH="$LIBTORCH/lib:${DYLD_LIBRARY_PATH:-}"   # macOS
# export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"      # Linux
```

## Verify Installation

```bash
# Cargo should compile and start linking torch-sys:
cargo check --features training

# Full sanity check (runs a minimal training):
./scripts/train-cpu.sh train_cartpole_modern
```

## Troubleshooting

### `dyld: Library not loaded: .../libprotobuf.33.0.0.dylib` (macOS)

You almost certainly have `./libtorch/` populated from `brew install pytorch`
(directly or by a previous version of this repo's setup). Fix:

```bash
rm -rf libtorch venv
./scripts/setup-libtorch.sh
source .envrc.libtorch
```

See [issue #8](https://github.com/rjwalters/thrust/issues/8) for the gory
detail. Short version: Homebrew's pytorch bottle hardcodes the exact SONAME
of the protobuf / abseil it built against (e.g. `libprotobuf.33.0.0.dylib`),
and macOS dyld will not relax that to whatever current Homebrew installed.

### `dyld: Library not loaded: .../libabsl_log_internal_check_op.2508.0.0.dylib`

Same root cause as above. Same fix.

### `error: linking with `cc` failed: ... -ltorch ...`

`LIBTORCH` is set but the path is wrong, or `LIBTORCH_USE_PYTORCH` is not
exported. Re-source the env:

```bash
source .envrc.libtorch     # if you used the venv path
# OR
export LIBTORCH="$(pwd)/libtorch"   # if you used download-libtorch.sh
```

### `error while loading shared libraries` (Linux)

Update `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"
# Or persistently: add a file under /etc/ld.so.conf.d/ and run ldconfig
```

### CUDA version mismatch (Linux GPU)

The pip wheel installed by `scripts/setup-libtorch.sh` matches the system
CUDA in most cases. If not, override the index URL:

```bash
source venv/bin/activate
pip install torch==2.9.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### "tch version doesn't match" panic

Set `LIBTORCH_BYPASS_VERSION_CHECK=1` (the wrapper scripts already do this).
`tch-rs` is overly strict on patch versions; the actual ABI is stable
across patch releases of the same minor PyTorch version.

## Version Compatibility

| Rust `tch-rs` | LibTorch / PyTorch | Notes |
|---------------|--------------------|-------|
| `0.15.x`      | `2.2.x`            | Older GPU recipe (see `VERSIONS.md`) |
| `0.22.x`      | `2.9.x`            | Current pin (`Cargo.toml`) |

## Docker (optional)

```dockerfile
FROM rust:latest
RUN apt-get update && apt-get install -y python3 python3-venv python3-pip unzip curl
WORKDIR /workspace
COPY . .
RUN ./scripts/setup-libtorch.sh
# `setup-libtorch.sh` wrote .envrc.libtorch -- source it before cargo:
RUN bash -c 'source .envrc.libtorch && cargo build --release --features training'
```

## References

- [tch-rs repository](https://github.com/LaurentMazare/tch-rs)
- [PyTorch downloads](https://pytorch.org/get-started/locally/)
- [LibTorch C++ docs](https://pytorch.org/cppdocs/)
- [Issue #8 — Homebrew bottle drift root cause](https://github.com/rjwalters/thrust/issues/8)
