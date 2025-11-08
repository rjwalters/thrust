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

# Activate Python venv and Rust
source venv/bin/activate
source ~/.cargo/env

# Set up libtorch paths - use Python PyTorch installation
export LIBTORCH_USE_PYTORCH=1
export PYTORCH_LIB=$(python3 -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
export LD_LIBRARY_PATH="$PYTORCH_LIB:${LD_LIBRARY_PATH:-}"
export LIBTORCH_BYPASS_VERSION_CHECK=1

# Create log file with timestamp
LOG_FILE="training_${EXAMPLE}_$(date +%Y%m%d_%H%M%S).log"

echo "🚀 Starting training: $EXAMPLE"
echo "📊 Logging to: $LOG_FILE"
echo "🖥️  Check GPU usage: nvidia-smi"
echo ""

# Run training
cargo run --example "$EXAMPLE" --release 2>&1 | tee "$LOG_FILE"
