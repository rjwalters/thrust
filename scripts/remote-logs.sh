#!/usr/bin/env bash
# Fetch and display training logs from remote GPU machine
#
# Usage:
#   ./scripts/remote-logs.sh [--follow] [--lines N]

set -euo pipefail

REMOTE="rwalters-sandbox-2"
REMOTE_DIR="thrust"

FOLLOW=""
LINES="50"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --follow|-f)
            FOLLOW="yes"
            shift
            ;;
        --lines|-n)
            LINES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--follow] [--lines N]"
            exit 1
            ;;
    esac
done

if [[ -n "$FOLLOW" ]]; then
    echo "📊 Following training logs on $REMOTE (Ctrl+C to stop)..."
    ssh $REMOTE "cd $REMOTE_DIR && \
        LOG_FILE=\$(ls -t training_*.log 2>/dev/null | head -1); \
        if [ -n \"\$LOG_FILE\" ]; then \
            tail -f \$LOG_FILE; \
        else \
            echo 'No training log files found'; \
        fi"
else
    echo "📊 Fetching last $LINES lines from training logs..."
    ssh $REMOTE "cd $REMOTE_DIR && \
        LOG_FILE=\$(ls -t training_*.log 2>/dev/null | head -1); \
        if [ -n \"\$LOG_FILE\" ]; then \
            echo \"Log file: \$LOG_FILE\"; \
            tail -$LINES \$LOG_FILE; \
        else \
            echo 'No training log files found'; \
        fi"
fi
