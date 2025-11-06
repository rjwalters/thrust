#!/bin/bash
# Setup script for libtorch
# Detects environment and sets up PyTorch for tch-rs

set -e

echo "🔧 Thrust libtorch Setup Script"
echo

# Detect if we're on a GPU machine
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    HAS_GPU=true
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
else
    echo "ℹ️  No NVIDIA GPU detected - will use CPU"
    HAS_GPU=false
fi

echo

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    echo "Please install Python 3.8+ first"
    exit 1
fi

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install PyTorch based on GPU availability
if [ "$HAS_GPU" = true ]; then
    echo "🚀 Installing PyTorch with CUDA support..."
    echo "   This may take a few minutes..."
    pip install --upgrade pip > /dev/null
    pip install 'torch>=2.9' 'numpy<2' > /dev/null 2>&1

    # Verify CUDA is available
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "false")
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
        echo "✅ PyTorch $TORCH_VERSION with CUDA $CUDA_VERSION installed"
        echo "   GPU Count: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
    else
        echo "⚠️  PyTorch installed but CUDA not available"
        echo "   This may happen if CUDA drivers are not installed"
    fi
else
    echo "📦 Installing PyTorch (CPU only)..."
    pip install --upgrade pip > /dev/null
    pip install 'torch>=2.9' 'numpy<2' > /dev/null 2>&1
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo "✅ PyTorch $TORCH_VERSION (CPU) installed"
fi

echo
echo "✅ Setup complete!"
echo
echo "To use with cargo:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Set environment variable: export LIBTORCH_USE_PYTORCH=1"
echo "  3. Build/run your project: cargo run --example train_cartpole"
echo
echo "Or use the provided training scripts:"
echo "  ./scripts/train-cpu.sh <example_name>  # For CPU training"
echo "  ./scripts/train-gpu.sh <example_name>  # For GPU training"
