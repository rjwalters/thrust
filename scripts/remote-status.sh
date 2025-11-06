#!/usr/bin/env bash
# Check training status on remote GPU machine

set -euo pipefail

REMOTE="rwalters-sandbox-2"
REMOTE_DIR="thrust"

echo "🔍 Checking training status on $REMOTE..."
echo ""

ssh $REMOTE "cd $REMOTE_DIR && \
    echo '=== GPU Status ===' && \
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits && \
    echo '' && \
    echo '=== Training Processes ===' && \
    (ps aux | grep -E 'cargo.*train|train_cartpole' | grep -v grep || echo 'No training processes running') && \
    echo '' && \
    echo '=== Latest Training Logs ===' && \
    if [ -f training.pid ]; then \
        PID=\$(cat training.pid); \
        if ps -p \$PID > /dev/null 2>&1; then \
            echo \"Training process \$PID is running\"; \
        else \
            echo \"Training process \$PID is not running\"; \
        fi; \
    fi && \
    echo '' && \
    LOG_FILE=\$(ls -t training_*.log 2>/dev/null | head -1); \
    if [ -n \"\$LOG_FILE\" ]; then \
        echo \"Latest log file: \$LOG_FILE\"; \
        echo '--- Last 30 lines ---'; \
        tail -30 \$LOG_FILE; \
    else \
        echo 'No training log files found'; \
    fi"
