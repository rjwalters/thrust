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

# Set environment to use PyTorch from pip
export LIBTORCH_USE_PYTORCH=1

# Run the example
cargo run --example "$EXAMPLE_NAME" --release
