#!/usr/bin/env bash
# Check optimization status on remote GPU machine

set -euo pipefail

REMOTE="rwalters-sandbox-2"
REMOTE_DIR="thrust"

echo "🔍 Checking optimization status on $REMOTE..."
echo ""

ssh $REMOTE "cd $REMOTE_DIR && \
    echo '=== GPU Status ===' && \
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits && \
    echo '' && \
    echo '=== Optimization Processes ===' && \
    (ps aux | grep -E 'optimize_cartpole' | grep -v grep || echo 'No optimization processes running') && \
    echo '' && \
    if [ -f optimization.pid ]; then \
        PID=\$(cat optimization.pid); \
        if ps -p \$PID > /dev/null 2>&1; then \
            echo \"✅ Optimization process \$PID is running\"; \
        else \
            echo \"❌ Optimization process \$PID is not running\"; \
        fi; \
    fi && \
    echo '' && \
    echo '=== Results ===' && \
    if [ -f cartpole_optimization_results.json ]; then \
        TRIAL_COUNT=\$(jq '.trials | length' cartpole_optimization_results.json 2>/dev/null || echo '0'); \
        BEST_PERF=\$(jq -r '.best_performance' cartpole_optimization_results.json 2>/dev/null || echo 'N/A'); \
        echo \"📊 Progress: \$TRIAL_COUNT trials completed\"; \
        echo \"🏆 Best performance: \$BEST_PERF steps/episode\"; \
    else \
        echo 'No results file found'; \
    fi && \
    echo '' && \
    echo '=== Latest Logs ===' && \
    LOG_FILE=\$(ls -t optimization_*.log 2>/dev/null | head -1); \
    if [ -n \"\$LOG_FILE\" ]; then \
        echo \"Latest log file: \$LOG_FILE\"; \
        echo '--- Last 20 lines ---'; \
        tail -20 \$LOG_FILE; \
    else \
        echo 'No optimization log files found'; \
    fi"
