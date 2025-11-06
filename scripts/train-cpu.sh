#!/bin/bash
# CPU training script

if [ -z "$1" ]; then
    echo "Usage: $0 <example_name>"
    echo "Example: $0 train_cartpole"
    exit 1
fi

EXAMPLE_NAME=$1

echo "🚀 Starting CPU training: $EXAMPLE_NAME"
echo

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Source cargo environment if available
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# Set environment to use PyTorch from pip
export LIBTORCH_USE_PYTORCH=1

# Set LD_LIBRARY_PATH to include PyTorch lib directory
TORCH_LIB=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH}"

# Run the example
cargo run --example "$EXAMPLE_NAME" --release
