#!/usr/bin/env bash
# Check GPU and training status (to be executed ON the remote)

set -euo pipefail

echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "=== Training Processes ==="
ps aux | grep -E 'cargo.*train|train_cartpole' | grep -v grep || echo "No training processes running"

echo ""
echo "=== Recent Training Logs ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_FILE=$(ls -t training_*.log 2>/dev/null | head -1)
if [ -n "$LOG_FILE" ]; then
    echo "Latest log: $LOG_FILE"
    echo "--- Last 20 lines ---"
    tail -20 "$LOG_FILE"
else
    echo "No training logs found"
fi
