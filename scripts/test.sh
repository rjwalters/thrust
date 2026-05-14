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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Platform-specific dynamic linker var
if [ "$(uname -s)" = "Darwin" ]; then
    LIB_PATH_VAR="DYLD_LIBRARY_PATH"
else
    LIB_PATH_VAR="LD_LIBRARY_PATH"
fi

# Prefer the pip venv set up by scripts/setup-libtorch.sh.
if [ -d "$PROJECT_ROOT/venv" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/venv/bin/activate"
    export LIBTORCH_USE_PYTORCH=1
    export LIBTORCH_BYPASS_VERSION_CHECK=1
    TORCH_LIB="$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")"
    export "$LIB_PATH_VAR"="${TORCH_LIB}:${!LIB_PATH_VAR:-}"
    echo "Using libtorch from venv: $TORCH_LIB"
elif [ -d "$PROJECT_ROOT/libtorch" ]; then
    export LIBTORCH="$PROJECT_ROOT/libtorch"
    export "$LIB_PATH_VAR"="$LIBTORCH/lib:${!LIB_PATH_VAR:-}"
    echo "Using vendored libtorch: $LIBTORCH"
else
    echo "ERROR: neither ./venv nor ./libtorch found." >&2
    echo "       Run ./scripts/setup-libtorch.sh first." >&2
    exit 1
fi

echo "Running tests"
echo

cargo +nightly test "$@"
