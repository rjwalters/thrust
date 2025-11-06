#!/usr/bin/env bash
# Stop training on remote GPU machine

set -euo pipefail

REMOTE="rwalters-sandbox-2"
REMOTE_DIR="thrust"

echo "🛑 Stopping training on $REMOTE..."

ssh $REMOTE "cd $REMOTE_DIR && \
    if [ -f training.pid ]; then \
        PID=\$(cat training.pid); \
        if ps -p \$PID > /dev/null 2>&1; then \
            echo \"Killing process \$PID...\"; \
            kill \$PID; \
            sleep 2; \
            if ps -p \$PID > /dev/null 2>&1; then \
                echo \"Process still running, force killing...\"; \
                kill -9 \$PID; \
            fi; \
            echo \"✅ Training stopped\"; \
        else \
            echo \"Process \$PID is not running\"; \
        fi; \
        rm -f training.pid; \
    else \
        echo \"No training.pid file found\"; \
        echo \"Attempting to kill any cargo/training processes...\"; \
        pkill -f 'cargo.*train' || echo \"No processes found\"; \
    fi"
