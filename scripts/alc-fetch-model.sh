#!/usr/bin/env bash
# Fetch trained model from alc-2, export to JSON for WASM, copy to web/public/
#
# Usage:
#   ./scripts/alc-fetch-model.sh [--with-metadata]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

REMOTE="alc-2"
WITH_METADATA="${1:-}"

# --- 1. Export the model to JSON on the remote machine ----------------------
echo "📦 Exporting model to JSON on $REMOTE..."
ssh $REMOTE << ENDSSH
set -euo pipefail
cd ~/thrust

source ~/.cargo/env
export LIBTORCH_USE_PYTORCH=1
PYTORCH_LIB=\$(/usr/bin/python3 -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))')
export LD_LIBRARY_PATH="\$PYTORCH_LIB:\${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="\$PYTORCH_LIB/libtorch_cuda.so:\$PYTORCH_LIB/libtorch.so"
export RUSTFLAGS="-C link-arg=-Wl,--no-as-needed"
export LIBTORCH_BYPASS_VERSION_CHECK=1

# Find the latest checkpoint (multi_v2 shared policy, then single-agent fallback)
MODEL=\$(ls -t models/snake_policy_shared.safetensors models/snake_policy.shared_epoch*.safetensors models/snake_single_update*.safetensors models/snake_single_final.safetensors 2>/dev/null | head -1 || echo "")
if [ -z "\$MODEL" ]; then
    echo "❌ No trained model found in models/"
    ls -lh models/ 2>/dev/null || true
    exit 1
fi
echo "✅ Using model: \$MODEL"

cargo run --example export_snake_model --release --features training -- \\
    "\$MODEL" snake_model.json ${WITH_METADATA:+--with-metadata}

echo "✅ Exported to snake_model.json"
ls -lh snake_model.json
ENDSSH

# --- 2. Download model files to local ---------------------------------------
echo ""
echo "📥 Downloading model files..."
mkdir -p "$PROJECT_ROOT/models"
rsync -az --progress alc-2:~/thrust/snake_model.json "$PROJECT_ROOT/web/public/snake_model.json"
rsync -az --progress alc-2:~/thrust/models/ "$PROJECT_ROOT/models/"

echo ""
echo "✅ Model ready at web/public/snake_model.json"
ls -lh "$PROJECT_ROOT/web/public/snake_model.json"
echo ""
echo "Deploy with: cd web && npm run build && npm run deploy"
