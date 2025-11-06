#!/usr/bin/env bash
# Helper script to run tests with proper libtorch setup
#
# Usage:
#   ./scripts/test.sh [test_args...]
#
# Examples:
#   ./scripts/test.sh
#   ./scripts/test.sh --test policy
#   ./scripts/test.sh -- --nocapture

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Setup libtorch paths
export LIBTORCH="$PROJECT_ROOT/libtorch"
export DYLD_LIBRARY_PATH="$LIBTORCH/lib"

echo "🧪 Running tests"
echo "📦 Using libtorch: $LIBTORCH"
echo ""

cd "$PROJECT_ROOT"
cargo +nightly test "$@"
