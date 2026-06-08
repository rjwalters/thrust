#!/usr/bin/env bash
# Run training on GPU machine (to be executed ON the remote)
#
# Usage (on GPU machine):
#   ./scripts/gpu-train.sh <example_name>
#
# Example:
#   ./scripts/gpu-train.sh train_cartpole_modern

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <example_name>"
    echo "Example: $0 train_cartpole_modern"
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

# Make silent CPU fallback a hard error: the crate-root build.rs is supposed
# to keep libtorch_cuda in NEEDED, but if it ever fails (e.g. a custom
# libtorch path that doesn't ship libtorch_cuda.so), we'd rather know
# loudly than benchmark a misleading run.
export THRUST_EXPECT_CUDA=1

# NOTE: As of issue #9 the crate-root build.rs now emits positional
# `-Wl,--no-as-needed` + `-ltorch_cuda` + `-lc10_cuda` so the linker does
# NOT strip the CUDA libs from DT_NEEDED. The historical LD_PRELOAD +
# RUSTFLAGS workaround is no longer required. If you ever need to fall
# back, uncomment the two lines below.
# export LD_PRELOAD="$PYTORCH_LIB/libtorch_cuda.so:$PYTORCH_LIB/libtorch.so"
# export RUSTFLAGS="-C link-arg=-Wl,--no-as-needed"

echo "Debug: PYTORCH_LIB=$PYTORCH_LIB"
echo "Debug: LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "Debug: THRUST_EXPECT_CUDA=$THRUST_EXPECT_CUDA"
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
