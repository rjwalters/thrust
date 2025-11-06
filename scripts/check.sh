#!/usr/bin/env bash
# Helper script to run checks (clippy, fmt) with proper libtorch setup
#
# Usage:
#   ./scripts/check.sh
#
# This runs cargo fmt, cargo clippy, and cargo test

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Setup libtorch paths
export LIBTORCH="$PROJECT_ROOT/libtorch"
export DYLD_LIBRARY_PATH="$LIBTORCH/lib"

echo "🔍 Running full CI checks"
echo "📦 Using libtorch: $LIBTORCH"
echo ""

cd "$PROJECT_ROOT"

echo "📝 Checking formatting..."
cargo fmt -- --check

echo ""
echo "🔧 Running clippy..."
cargo +nightly clippy --all-targets --all-features -- -D warnings

echo ""
echo "🧪 Running tests..."
cargo +nightly test

echo ""
echo "✅ All checks passed!"
