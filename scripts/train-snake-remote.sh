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

# Run training with shared policy mode (faster convergence)
LIBTORCH_USE_PYTORCH=1 \
cargo +nightly run --example train_snake_multi_v2 --release -- \
  --mode shared \
  --epochs 1000 \
  --cuda

echo "✅ Training complete!"
echo "📦 Exporting model..."

# Export the trained model
LIBTORCH_USE_PYTORCH=1 \
cargo +nightly run --example export_snake_model --release

echo "✅ Model exported to snake_model.json"
ls -lh snake_model.json
EOF

echo ""
echo "📥 Downloading trained model..."
scp rwalters-sandbox-2:~/thrust/snake_model.json ./web/public/

echo ""
echo "✅ All done! Model ready at web/public/snake_model.json"
