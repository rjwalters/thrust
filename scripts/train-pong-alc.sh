#!/usr/bin/env bash
# Train Pong on alcubierre cluster (alc-2, RTX 4090 24GB)
#
# Usage:
#   ./scripts/train-pong-alc.sh [--no-sync]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

REMOTE="alc-2"
REMOTE_DIR="thrust"
EXAMPLE="train_pong"
SYNC=true

for arg in "$@"; do
    case $arg in
        --no-sync) SYNC=false ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# --- 1. Sync code -----------------------------------------------------------
if $SYNC; then
    echo "📡 Syncing code to $REMOTE..."
    rsync -az --progress \
        --exclude 'target' \
        --exclude '.git' \
        --exclude '*.safetensors' \
        --exclude '*.pt' \
        --filter=':- .gitignore' \
        "$PROJECT_ROOT/" $REMOTE:~/$REMOTE_DIR/
    echo "✅ Sync complete"
fi

# --- 2. Build + launch training ---------------------------------------------
echo ""
echo "🚀 Launching Pong training on $REMOTE (RTX 4090)..."
ssh $REMOTE << 'ENDSSH'
set -euo pipefail
cd ~/thrust

EXAMPLE="train_pong"

if ! command -v cargo &>/dev/null; then
    echo "📦 Installing Rust stable..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path
    export PATH="$HOME/.cargo/bin:$PATH"
fi
source ~/.cargo/env
echo "🦀 $(rustc --version) / $(cargo --version)"

export LIBTORCH_USE_PYTORCH=1
PYTORCH_LIB=$(/usr/bin/python3 -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))')
export LD_LIBRARY_PATH="$PYTORCH_LIB:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$PYTORCH_LIB/libtorch_cuda.so:$PYTORCH_LIB/libtorch.so"
export RUSTFLAGS="-C link-arg=-Wl,--no-as-needed"
export LIBTORCH_BYPASS_VERSION_CHECK=1

echo "🖥️  GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
echo "🐍 PyTorch: $(/usr/bin/python3 -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"
echo "📚 libtorch: $PYTORCH_LIB"
echo ""

echo "🔨 Building $EXAMPLE (release, CUDA)..."
cargo build --example "$EXAMPLE" --release --features training 2>&1
echo "✅ Build successful"
echo ""

# Kill any existing training using the same pid file
if [ -f training.pid ]; then
    OLD_PID=$(cat training.pid)
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "⏹️  Stopping previous training (PID $OLD_PID)..."
        kill "$OLD_PID"
        sleep 2
    fi
fi

LOG="training_${EXAMPLE}_$(date +%Y%m%d_%H%M%S).log"
nohup env \
    LIBTORCH_USE_PYTORCH=1 \
    LD_LIBRARY_PATH="$PYTORCH_LIB:${LD_LIBRARY_PATH:-}" \
    LD_PRELOAD="$PYTORCH_LIB/libtorch_cuda.so:$PYTORCH_LIB/libtorch.so" \
    RUSTFLAGS="-C link-arg=-Wl,--no-as-needed" \
    LIBTORCH_BYPASS_VERSION_CHECK=1 \
    cargo run --example "$EXAMPLE" --release --features training \
    > "$LOG" 2>&1 &

PID=$!
echo $PID > training.pid
echo "▶️  Training started (PID $PID)"
echo "📝 Log: ~/thrust/$LOG"
ENDSSH

echo ""
echo "✅ Training is running on $REMOTE"
echo ""
echo "Monitor:"
echo "  ssh alc-2 'tail -f ~/thrust/training_train_pong_*.log'"
echo "  ./scripts/alc-status.sh"
