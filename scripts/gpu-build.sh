#!/usr/bin/env bash
# Build on GPU machine (to be executed ON the remote)
#
# Usage (on GPU machine):
#   ./scripts/gpu-build.sh [--release]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Activate Python venv and Rust
source venv/bin/activate
source ~/.cargo/env

# Set up libtorch paths
export LIBTORCH="$(pwd)/libtorch"
export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1

# Parse arguments
RELEASE_FLAG=""
if [[ "${1:-}" == "--release" ]]; then
    RELEASE_FLAG="--release"
    echo "🔨 Building in release mode..."
else
    echo "🔨 Building in debug mode..."
fi

cargo build $RELEASE_FLAG

echo "✅ Build complete!"
