#!/usr/bin/env bash
# Run training on GPU machine (to be executed ON the remote)
#
# Usage (on GPU machine):
#   ./scripts/gpu-train.sh <example_name>
#
# Example:
#   ./scripts/gpu-train.sh train_cartpole_long

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <example_name>"
    echo "Example: $0 train_cartpole_long"
    exit 1
fi

EXAMPLE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Set up Rust
source ~/.cargo/env

# Set up libtorch paths - use system Python's PyTorch (installed globally)
# Don't activate venv - use system python3 which has PyTorch installed
export LIBTORCH_USE_PYTORCH=1
export PYTORCH_LIB=$(/usr/bin/python3 -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
export LD_LIBRARY_PATH="$PYTORCH_LIB:${LD_LIBRARY_PATH:-}"
export LIBTORCH_BYPASS_VERSION_CHECK=1

# CRITICAL: Preload CUDA libraries to ensure they're loaded at runtime
export LD_PRELOAD="$PYTORCH_LIB/libtorch_cuda.so:$PYTORCH_LIB/libtorch.so"

# CRITICAL: Force linker to keep CUDA library dependency
export RUSTFLAGS="-C link-arg=-Wl,--no-as-needed"

echo "Debug: PYTORCH_LIB=$PYTORCH_LIB"
echo "Debug: LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "Debug: LD_PRELOAD=$LD_PRELOAD"
echo "Debug: RUSTFLAGS=$RUSTFLAGS"
echo ""

# Create log file with timestamp
LOG_FILE="training_${EXAMPLE}_$(date +%Y%m%d_%H%M%S).log"

echo "🚀 Starting training: $EXAMPLE"
echo "📊 Logging to: $LOG_FILE"
echo "🖥️  Check GPU usage: nvidia-smi"
echo ""

# IMPORTANT: Build on this machine to detect CUDA at compile time
echo "🔨 Building with CUDA support..."
cargo build --example "$EXAMPLE" --release

echo ""
echo "▶️  Running training..."
# Run training with CUDA libraries preloaded
cargo run --example "$EXAMPLE" --release 2>&1 | tee "$LOG_FILE"
