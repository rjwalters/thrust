#!/bin/bash
#
# Launch multiple CartPole training runs in parallel with different vf_coef values
# to determine optimal configuration.
#
# Usage:
#   ./scripts/parallel_cartpole_training.sh
#
# This script launches 3 training runs with:
#   - vf_coef = 0.5 (standard, recommended by research)
#   - vf_coef = 1.0 (middle ground)
#   - vf_coef = 2.0 (current, known to have instability)

set -e

# Configuration
THRUST_DIR="/root/thrust"
VENV_PATH="$THRUST_DIR/venv"
LOG_DIR="$THRUST_DIR/logs/parallel_training_$(date +%Y%m%d_%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🚀 Starting parallel CartPole training runs"
echo "Log directory: $LOG_DIR"

# Create log directory
mkdir -p "$LOG_DIR"

# Setup environment function
setup_env() {
    cd "$THRUST_DIR"
    git pull origin master
    source "$VENV_PATH/bin/activate"
    source ~/.cargo/env
    export LIBTORCH_USE_PYTORCH=1
    TORCH_LIB=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
    export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
    export LD_PRELOAD="$TORCH_LIB/libtorch_cuda.so:$TORCH_LIB/libtorch.so"
    export RUSTFLAGS="-C link-arg=-Wl,--no-as-needed"
}

# Launch training run with specific vf_coef
launch_training() {
    local vf_coef=$1
    local run_name="vfcoef_${vf_coef}"
    local log_file="$LOG_DIR/training_${run_name}.log"
    local pid_file="$LOG_DIR/training_${run_name}.pid"

    echo -e "${YELLOW}Launching training run: vf_coef=$vf_coef${NC}"

    # Create a temporary modified version of the training script
    local temp_script="$LOG_DIR/train_${run_name}.sh"

    cat > "$temp_script" <<EOF
#!/bin/bash
cd "$THRUST_DIR"
source "$VENV_PATH/bin/activate"
source ~/.cargo/env
export LIBTORCH_USE_PYTORCH=1
TORCH_LIB=\$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="\$TORCH_LIB:\$LD_LIBRARY_PATH"
export LD_PRELOAD="\$TORCH_LIB/libtorch_cuda.so:\$TORCH_LIB/libtorch.so"
export RUSTFLAGS="-C link-arg=-Wl,--no-as-needed"

# Modify the source file with the desired vf_coef
sed -i.bak 's/.vf_coef([0-9.]*) /.vf_coef($vf_coef) /' examples/train_cartpole_best.rs

# Run training
cargo +nightly run --example train_cartpole_best --release 2>&1

# Restore original file
mv examples/train_cartpole_best.rs.bak examples/train_cartpole_best.rs
EOF

    chmod +x "$temp_script"

    # Launch in background
    nohup bash "$temp_script" > "$log_file" 2>&1 &
    local pid=$!
    echo $pid > "$pid_file"

    echo -e "${GREEN}✓ Started: PID=$pid, Log=$log_file${NC}"

    # Wait a moment and check if process is still running
    sleep 2
    if ps -p $pid > /dev/null; then
        echo -e "${GREEN}✓ Process confirmed running${NC}"
    else
        echo -e "${RED}✗ Process failed to start!${NC}"
        tail -20 "$log_file"
    fi

    echo ""
}

# Main execution
echo "Setting up environment..."
setup_env

echo ""
echo "Launching 3 parallel training runs..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Launch runs in parallel
launch_training "0.5"  # Standard (Stable-Baselines3, CleanRL)
launch_training "1.0"  # Middle ground
launch_training "2.0"  # Current (high, known instability)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "All training runs launched!"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_DIR/training_vfcoef_0.5.log"
echo "  tail -f $LOG_DIR/training_vfcoef_1.0.log"
echo "  tail -f $LOG_DIR/training_vfcoef_2.0.log"
echo ""
echo "Check PIDs:"
echo "  cat $LOG_DIR/*.pid"
echo ""
echo "Kill all runs:"
echo "  for pid in \$(cat $LOG_DIR/*.pid); do kill \$pid; done"
echo ""

# Create a monitoring script
cat > "$LOG_DIR/monitor.sh" <<'MONITOR_EOF'
#!/bin/bash
# Monitor all training runs

LOG_DIR="$(dirname "$0")"

echo "Monitoring parallel training runs..."
echo "Press Ctrl+C to exit"
echo ""

while true; do
    clear
    echo "═══════════════════════════════════════════════════════════════"
    echo "Parallel CartPole Training Monitor"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    for vf in 0.5 1.0 2.0; do
        log_file="$LOG_DIR/training_vfcoef_${vf}.log"
        pid_file="$LOG_DIR/training_vfcoef_${vf}.pid"

        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if ps -p $pid > /dev/null 2>&1; then
                status="🟢 RUNNING"
            else
                status="🔴 STOPPED"
            fi
        else
            status="⚪ NO PID"
        fi

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "vf_coef = $vf [$status]"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        if [ -f "$log_file" ]; then
            # Extract latest metrics
            tail -1 "$log_file" | grep -oP "Update \K[0-9]+/[0-9]+|Steps: \K[0-9]+|Avg Steps/Ep: \K[0-9.]+|ExpVar: \K[-0-9.]+" | paste - - - - | \
            awk '{printf "  Progress: %s | Steps: %s | Avg Ep Len: %s | ExpVar: %s\n", $1, $2, $3, $4}'
        else
            echo "  [No log data yet]"
        fi
        echo ""
    done

    echo "Last updated: $(date)"
    sleep 5
done
MONITOR_EOF

chmod +x "$LOG_DIR/monitor.sh"

echo "Run monitor:"
echo "  $LOG_DIR/monitor.sh"
echo ""
echo "✨ All set! Training runs are executing in parallel."
