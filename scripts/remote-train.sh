#!/usr/bin/env bash
# Run training on remote GPU machine
#
# Usage:
#   ./scripts/remote-train.sh <example_name> [--sync-first]
#
# Examples:
#   ./scripts/remote-train.sh train_cartpole_long
#   ./scripts/remote-train.sh train_cartpole_long --sync-first

set -euo pipefail

REMOTE="rwalters-sandbox-2"
REMOTE_DIR="thrust"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <example_name> [--sync-first]"
    echo "Examples:"
    echo "  $0 train_cartpole_long"
    echo "  $0 train_cartpole_long --sync-first"
    exit 1
fi

EXAMPLE="$1"
SYNC_FIRST="${2:-}"

if [[ "$SYNC_FIRST" == "--sync-first" ]]; then
    echo "📡 Syncing code to $REMOTE..."
    rsync -avz --exclude 'target' --exclude '.git' --exclude 'libtorch' --exclude 'venv' \
        ./ $REMOTE:~/$REMOTE_DIR/
fi

echo "🚀 Starting training: $EXAMPLE"
echo "📊 Training will run in background on GPU machine"
echo ""

# Start training in background and get the log file name
LOG_FILE="training_${EXAMPLE}_$(date +%Y%m%d_%H%M%S).log"

ssh $REMOTE "cd $REMOTE_DIR && \
    source venv/bin/activate && \
    source ~/.cargo/env && \
    export LIBTORCH=\$(pwd)/libtorch && \
    export LD_LIBRARY_PATH=\$LIBTORCH/lib:\$LD_LIBRARY_PATH && \
    export LIBTORCH_USE_PYTORCH=1 && \
    nohup cargo run --example $EXAMPLE --release > $LOG_FILE 2>&1 & \
    echo \$! > training.pid && \
    echo 'Training started with PID:' \$(cat training.pid) && \
    echo 'Log file: $LOG_FILE'"

echo ""
echo "✅ Training started on $REMOTE"
echo ""
echo "📝 To monitor progress:"
echo "   ./scripts/remote-status.sh"
echo ""
echo "🛑 To stop training:"
echo "   ./scripts/remote-stop.sh"
