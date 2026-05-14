#!/bin/bash
# Train Snake on GPU sandbox with improved rewards

set -e

echo "🚀 Training ULTRA-COMPACT Snake agent on GPU sandbox..."
echo "📊 Configuration:"
echo "  - Model: 16x SMALLER (8→16→16 channels, 64 hidden) for fast WASM"
echo "  - Food reward: +100.0 (10x baseline - aggressive eating!)"
echo "  - Length bonus: +1.0 per extra segment (10x increase)"
echo "  - Death penalty: -0.1 (minimal, encourages risk-taking)"
echo "  - Training: 1000 epochs, shared policy mode"
echo ""

# SSH to GPU sandbox and train
ssh rwalters-sandbox-2 << 'EOF'
cd ~/thrust

# Pull latest changes
git pull

# Set up PyTorch environment for GPU
export LIBTORCH_USE_PYTORCH=1
PYTORCH_LIB=$(python3 -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
export LD_LIBRARY_PATH="$PYTORCH_LIB:${LD_LIBRARY_PATH:-}"

# Make silent CPU fallback a hard error. The crate-root build.rs (added in
# issue #9) is responsible for keeping libtorch_cuda.so in NEEDED, but
# this guard hard-fails (exit 2) if for any reason tch falls back to Cpu.
export THRUST_EXPECT_CUDA=1

# NOTE: As of issue #9 the crate-root build.rs now emits positional link
# args that defeat the linker's --as-needed stripping of libtorch_cuda,
# so the historical LD_PRELOAD + RUSTFLAGS workaround is no longer
# required. Uncomment if you hit a regression and need a fallback.
# export LD_PRELOAD="$PYTORCH_LIB/libtorch_cuda.so:$PYTORCH_LIB/libtorch.so"
# export RUSTFLAGS="-C link-arg=-Wl,--no-as-needed"

echo "🔍 CUDA Environment:"
echo "   LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "   THRUST_EXPECT_CUDA=$THRUST_EXPECT_CUDA"
echo ""

# Clean and rebuild to ensure CUDA linking
cargo +nightly clean
cargo +nightly build --example train_snake_multi_v2 --release

# Run training with shared policy mode
cargo +nightly run --example train_snake_multi_v2 --release -- \
  --mode shared \
  --epochs 1000 \
  --cuda

echo "✅ Training complete!"
echo "📦 Exporting model..."

# Export the trained model
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$(python3 -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
cargo +nightly run --example export_snake_model --release

echo "✅ Model exported to snake_model.json"
ls -lh snake_model.json
EOF

echo ""
echo "📥 Downloading trained model..."
scp rwalters-sandbox-2:~/thrust/snake_model.json ./web/public/

echo ""
echo "✅ All done! Model ready at web/public/snake_model.json"
