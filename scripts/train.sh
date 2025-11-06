#!/usr/bin/env bash
# Helper script to run training with proper libtorch setup
#
# Usage:
#   ./scripts/train.sh [example_name]
#
# Examples:
#   ./scripts/train.sh train_cartpole
#   ./scripts/train.sh train_cartpole --release

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Setup libtorch paths
export LIBTORCH="$PROJECT_ROOT/libtorch"
export DYLD_LIBRARY_PATH="$LIBTORCH/lib"

# Default example
EXAMPLE="${1:-train_cartpole}"

# Check if release flag is provided
RELEASE_FLAG=""
if [[ "${2:-}" == "--release" ]]; then
    RELEASE_FLAG="--release"
fi

echo "🚀 Running example: $EXAMPLE"
echo "📦 Using libtorch: $LIBTORCH"
echo "🔧 Mode: ${RELEASE_FLAG:-debug}"
echo ""

cd "$PROJECT_ROOT"
cargo +nightly run --example "$EXAMPLE" $RELEASE_FLAG
