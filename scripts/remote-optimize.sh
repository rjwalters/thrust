#!/usr/bin/env bash
# Run hyperparameter optimization on remote GPU machine
#
# Usage:
#   ./scripts/remote-optimize.sh [--sync-first] [--trials N]
#
# Examples:
#   ./scripts/remote-optimize.sh --trials 50
#   ./scripts/remote-optimize.sh --sync-first --trials 30

set -euo pipefail

REMOTE="rwalters-sandbox-2"
REMOTE_DIR="thrust"
TRIALS=30
SYNC_FIRST=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sync-first)
            SYNC_FIRST=true
            shift
            ;;
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--sync-first] [--trials N]"
            exit 1
            ;;
    esac
done

if [[ "$SYNC_FIRST" == "true" ]]; then
    echo "📡 Syncing code to $REMOTE..."
    rsync -avz --exclude 'target' --exclude '.git' --exclude 'libtorch' --exclude 'venv' \
        ./ $REMOTE:~/$REMOTE_DIR/
fi

echo "🚀 Starting hyperparameter optimization"
echo "📊 Trials: $TRIALS"
echo "🖥️  Target: $REMOTE"
echo ""

# Create timestamped log file
LOG_FILE="optimization_$(date +%Y%m%d_%H%M%S).log"

# Start optimization on remote with proper GPU setup
ssh $REMOTE "cd $REMOTE_DIR && \
    source venv/bin/activate && \
    source ~/.cargo/env && \
    \
    # Verify CUDA availability
    if ! python3 -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then \
        echo '❌ Error: CUDA not available in PyTorch'; \
        exit 1; \
    fi && \
    \
    echo '✅ GPU detected:' \$(python3 -c 'import torch; print(torch.cuda.get_device_name(0))') && \
    echo '   CUDA version:' \$(python3 -c 'import torch; print(torch.version.cuda)') && \
    echo '' && \
    \
    # Set environment for GPU usage
    export LIBTORCH_USE_PYTORCH=1 && \
    TORCH_LIB=\$(python3 -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))') && \
    export LD_LIBRARY_PATH=\"\$TORCH_LIB:\$LD_LIBRARY_PATH\" && \
    export LD_PRELOAD=\"\$TORCH_LIB/libtorch_cuda.so:\$TORCH_LIB/libtorch.so\" && \
    export RUSTFLAGS='-C link-arg=-Wl,--no-as-needed' && \
    \
    # Build on GPU machine for CUDA detection
    echo 'Building with CUDA support...' && \
    cargo +nightly build --example optimize_cartpole --release && \
    \
    # Run optimization in background with GPU environment preserved
    nohup env LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH\" LD_PRELOAD=\"\$LD_PRELOAD\" LIBTORCH_USE_PYTORCH=1 \
        cargo +nightly run --example optimize_cartpole --release -- --trials $TRIALS > $LOG_FILE 2>&1 & \
    echo \$! > optimization.pid && \
    echo 'Optimization started with PID:' \$(cat optimization.pid) && \
    echo 'Log file: $LOG_FILE'"

echo ""
echo "✅ Optimization started on $REMOTE"
echo ""
echo "📝 To monitor progress:"
echo "   ./scripts/remote-logs.sh"
echo ""
echo "📊 To check status:"
echo "   ssh $REMOTE 'cd $REMOTE_DIR && tail -20 $LOG_FILE'"
echo ""
echo "🛑 To stop optimization:"
echo "   ./scripts/remote-stop.sh"
