#!/bin/bash
# Setup script for libtorch
#
# This script installs PyTorch into a local Python venv and prints the exact
# export lines you need to make `tch-rs` pick up the bundled libtorch from
# that venv. Works on both Linux (CPU + GPU) and macOS (arm64 + x86_64).
#
# Why a venv (and NOT `brew install pytorch`)?
#   The Homebrew `pytorch` formula on macOS produces libtorch dylibs that are
#   linked against specific minor versions of `protobuf` and `abseil` from the
#   Homebrew bottle at build time. When Homebrew bumps those formulae (which
#   it does often), the libtorch dylibs cannot find their deps any more and
#   you get `dyld: Library not loaded: ...libprotobuf.33.0.0.dylib` errors.
#   The pip torch wheel bundles its own protobuf/abseil so it is immune to
#   Homebrew bottle drift. See docs/LIBTORCH_SETUP.md for the full story.
#
# Usage:
#   ./scripts/setup-libtorch.sh
#
# After running, eval the export block it prints (or copy it into ~/.zshrc).

set -e

echo "Thrust libtorch Setup Script"
echo

# Detect platform
UNAME_S="$(uname -s)"
UNAME_M="$(uname -m)"
case "$UNAME_S" in
    Darwin)
        PLATFORM="macos"
        LIB_PATH_VAR="DYLD_LIBRARY_PATH"
        ;;
    Linux)
        PLATFORM="linux"
        LIB_PATH_VAR="LD_LIBRARY_PATH"
        ;;
    *)
        echo "ERROR: Unsupported platform: $UNAME_S"
        echo "       This script supports macOS and Linux only."
        exit 1
        ;;
esac
echo "Platform: $PLATFORM ($UNAME_M)"

# Detect GPU (Linux only; macOS Metal is auto-detected by PyTorch)
HAS_GPU=false
if [ "$PLATFORM" = "linux" ] && command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/^/   /'
    HAS_GPU=true
elif [ "$PLATFORM" = "macos" ]; then
    echo "macOS: CPU + MPS (Apple Silicon GPU) will be auto-detected by PyTorch"
else
    echo "No NVIDIA GPU detected - will use CPU"
fi

echo

# Warn about a stale ./libtorch directory (common leftover from Homebrew path)
if [ -d "libtorch" ]; then
    echo "WARNING: Found ./libtorch/ directory."
    echo "         If this was populated from 'brew install pytorch', it is likely"
    echo "         the source of dyld 'Library not loaded' errors. Consider:"
    echo "             rm -rf libtorch/"
    echo "         and let this script set up a pip-based venv instead."
    echo
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found"
    if [ "$PLATFORM" = "macos" ]; then
        echo "       Install with: brew install python@3.12"
    else
        echo "       Install with: sudo apt-get install python3 python3-venv"
    fi
    exit 1
fi
PYTHON_VERSION="$(python3 --version 2>&1 | awk '{print $2}')"
echo "Python: $PYTHON_VERSION"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment at ./venv ..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source venv/bin/activate

# Install PyTorch
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null

if [ "$HAS_GPU" = true ]; then
    echo "Installing PyTorch with CUDA support (this may take several minutes)..."
    pip install 'torch>=2.9' 'numpy<2' > /dev/null 2>&1
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "false")
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
        GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
        echo "PyTorch $TORCH_VERSION with CUDA $CUDA_VERSION installed (GPU count: $GPU_COUNT)"
    else
        echo "WARNING: PyTorch $TORCH_VERSION installed but CUDA is NOT available."
        echo "         CUDA drivers may not be installed. Check 'nvidia-smi'."
    fi
else
    echo "Installing PyTorch (CPU only)..."
    pip install 'torch>=2.9' 'numpy<2' > /dev/null 2>&1
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo "PyTorch $TORCH_VERSION installed (CPU)"
fi

# Compute the lib directory that tch-rs needs at runtime
TORCH_LIB="$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")"
echo "PyTorch lib dir: $TORCH_LIB"

echo
echo "===================================================================="
echo "Setup complete. To use this venv with cargo, run:"
echo
echo "    source venv/bin/activate"
echo "    export LIBTORCH_USE_PYTORCH=1"
echo "    export LIBTORCH_BYPASS_VERSION_CHECK=1"
echo "    export $LIB_PATH_VAR=\"$TORCH_LIB:\$$LIB_PATH_VAR\""
echo
echo "Or just use the wrapper scripts which set these for you:"
echo "    ./scripts/train-cpu.sh <example_name>     # CPU training"
if [ "$HAS_GPU" = true ]; then
    echo "    ./scripts/train-gpu.sh <example_name>     # GPU training (this machine)"
fi
echo "    ./scripts/test.sh                          # cargo test"
echo "    ./scripts/check.sh                         # cargo fmt + clippy + test"
echo "===================================================================="

# Persist the export block to .envrc.libtorch for editor / direnv use
ENV_FILE=".envrc.libtorch"
cat > "$ENV_FILE" <<EOF
# Generated by scripts/setup-libtorch.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)
# Source this file (or have direnv pick it up) to make 'cargo' use the venv's libtorch.
# shellcheck disable=SC1091
source "\$(dirname "\${BASH_SOURCE[0]:-\$0}")/venv/bin/activate"
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export $LIB_PATH_VAR="$TORCH_LIB:\${$LIB_PATH_VAR:-}"
EOF
echo
echo "Wrote $ENV_FILE -- 'source $ENV_FILE' to load the env into your shell."
