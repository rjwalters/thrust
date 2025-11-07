#!/usr/bin/env bash
# Run hyperparameter optimization on GPU machine (to be executed ON the remote)
#
# Usage (on GPU machine):
#   ./scripts/gpu-optimize.sh [--trials N]
#
# Example:
#   ./scripts/gpu-optimize.sh --trials 50

set -euo pipefail

TRIALS=30

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--trials N]"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Activate Python venv and Rust
source venv/bin/activate
source ~/.cargo/env

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: nvidia-smi not found - GPU may not be available"
fi

# Verify CUDA is available in PyTorch
CUDA_CHECK=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "false")
if [ "$CUDA_CHECK" != "True" ]; then
    echo "❌ Error: CUDA not available in PyTorch"
    echo "Run ./scripts/setup-libtorch.sh to set up PyTorch with CUDA"
    exit 1
fi

echo "✅ GPU detected: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "   CUDA version: $(python3 -c 'import torch; print(torch.version.cuda)')"
echo ""

# Set environment to use PyTorch from pip
export LIBTORCH_USE_PYTORCH=1

# Set LD_LIBRARY_PATH to include PyTorch lib directory
TORCH_LIB=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

# CRITICAL: Preload CUDA libraries to ensure they're loaded at runtime
# The linker drops these as "unused" even with --no-as-needed, so we force-load them
export LD_PRELOAD="${TORCH_LIB}/libtorch_cuda.so:${TORCH_LIB}/libtorch.so"

# CRITICAL: Force linker to keep CUDA library dependency
# Without this, the linker removes libtorch_cuda.so as "unused" even though it's needed at runtime
export RUSTFLAGS="-C link-arg=-Wl,--no-as-needed"

# IMPORTANT: Build must happen on this machine to detect CUDA at compile time
echo "Building with CUDA support..."
cargo +nightly build --example optimize_cartpole --release

# Create log file with timestamp
LOG_FILE="optimization_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "🚀 Starting hyperparameter optimization"
echo "📊 Trials: $TRIALS"
echo "📝 Logging to: $LOG_FILE"
echo "🖥️  Check GPU usage: nvidia-smi"
echo ""

# Run optimization with GPU
cargo +nightly run --example optimize_cartpole --release -- --trials "$TRIALS" 2>&1 | tee "$LOG_FILE"
