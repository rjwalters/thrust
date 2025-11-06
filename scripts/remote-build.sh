#!/usr/bin/env bash
# Build project on remote GPU machine
#
# Usage:
#   ./scripts/remote-build.sh [--release]

set -euo pipefail

REMOTE="rwalters-sandbox-2"
REMOTE_DIR="thrust"

# Parse arguments
RELEASE_FLAG=""
if [[ "${1:-}" == "--release" ]]; then
    RELEASE_FLAG="--release"
    echo "🔨 Building in release mode..."
else
    echo "🔨 Building in debug mode..."
fi

echo "📡 Syncing code to $REMOTE..."
rsync -avz --exclude 'target' --exclude '.git' --exclude 'libtorch' --exclude 'venv' \
    ./ $REMOTE:~/$REMOTE_DIR/

echo "🏗️  Building on remote..."
ssh $REMOTE "cd $REMOTE_DIR && source venv/bin/activate && source ~/.cargo/env && \
    export LIBTORCH=\$(pwd)/libtorch && \
    export LD_LIBRARY_PATH=\$LIBTORCH/lib:\$LD_LIBRARY_PATH && \
    export LIBTORCH_USE_PYTORCH=1 && \
    cargo build $RELEASE_FLAG"

echo "✅ Build complete!"
