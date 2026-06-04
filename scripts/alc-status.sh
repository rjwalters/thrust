#!/usr/bin/env bash
# Check training status on alc-2

set -euo pipefail

REMOTE="alc-2"

echo "🔍 Status on $REMOTE..."
echo ""
ssh $REMOTE << 'ENDSSH'
echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader,nounits

echo ""
echo "=== Training process ==="
if [ -f ~/thrust/training.pid ]; then
    PID=$(cat ~/thrust/training.pid)
    if kill -0 "$PID" 2>/dev/null; then
        echo "Running (PID $PID)"
        ps -p "$PID" -o pid,etime,pcpu,pmem,cmd --no-headers
    else
        echo "Not running (PID $PID exited)"
    fi
else
    echo "No training.pid found"
fi

echo ""
echo "=== Latest log (last 20 lines) ==="
cd ~/thrust
LOG=$(ls -t training_snake_*.log 2>/dev/null | head -1 || echo "")
if [ -n "$LOG" ]; then
    echo "Log: $LOG"
    tail -20 "$LOG"
else
    echo "No log files found"
fi

echo ""
echo "=== Models ==="
ls -lh ~/thrust/models/*.safetensors 2>/dev/null || echo "No .safetensors files yet"
ENDSSH
