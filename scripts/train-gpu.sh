#!/bin/bash
# GPU training script

if [ -z "$1" ]; then
    echo "Usage: $0 <example_name>"
    echo "Example: $0 train_cartpole_best"
    exit 1
fi

EXAMPLE_NAME=$1

echo "🚀 Starting GPU training: $EXAMPLE_NAME"
echo

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: nvidia-smi not found - GPU may not be available"
fi

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "⚠️  Warning: venv not found - run ./scripts/setup-libtorch.sh first"
fi

# Verify CUDA is available in PyTorch
CUDA_CHECK=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "false")
if [ "$CUDA_CHECK" != "True" ]; then
    echo "❌ Error: CUDA not available in PyTorch"
    echo "Run ./scripts/setup-libtorch.sh to set up PyTorch with CUDA"
    exit 1
fi

echo "✅ GPU detected: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "   CUDA version: $(python3 -c 'import torch; print(torch.version.cuda)')"
echo

# Set environment to use PyTorch from pip
export LIBTORCH_USE_PYTORCH=1

# Run the example
cargo run --example "$EXAMPLE_NAME" --release
