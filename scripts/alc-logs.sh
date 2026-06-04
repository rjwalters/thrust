#!/usr/bin/env bash
# Tail training logs from alc-2
#
# Usage:
#   ./scripts/alc-logs.sh            # last 50 lines
#   ./scripts/alc-logs.sh --follow   # live tail (Ctrl-C to stop)

set -euo pipefail

REMOTE="alc-2"
FOLLOW=false

for arg in "$@"; do
    case $arg in
        --follow|-f) FOLLOW=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

if $FOLLOW; then
    echo "📊 Following training log on $REMOTE (Ctrl-C to stop)..."
    ssh $REMOTE 'cd ~/thrust && LOG=$(ls -t training_snake_*.log 2>/dev/null | head -1); [ -n "$LOG" ] && tail -f "$LOG" || echo "No log files found"'
else
    echo "📊 Latest training log on $REMOTE:"
    ssh $REMOTE 'cd ~/thrust && LOG=$(ls -t training_snake_*.log 2>/dev/null | head -1); [ -n "$LOG" ] && (echo "File: $LOG"; tail -50 "$LOG") || echo "No log files found"'
fi
