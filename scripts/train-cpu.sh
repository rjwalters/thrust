#!/bin/bash
# CPU training script
#
# Works on macOS (DYLD_LIBRARY_PATH) and Linux (LD_LIBRARY_PATH). Uses the
# Python venv set up by scripts/setup-libtorch.sh to source libtorch via
# LIBTORCH_USE_PYTORCH=1, which avoids the Homebrew bottle-drift problem on
# macOS (see docs/LIBTORCH_SETUP.md).

set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <example_name> [example_args...]"
    echo "Example: $0 train_cartpole"
    exit 1
fi

EXAMPLE_NAME="$1"
shift

echo "Starting CPU training: $EXAMPLE_NAME"
echo

# Detect platform to pick the right dynamic-linker env var
UNAME_S="$(uname -s)"
if [ "$UNAME_S" = "Darwin" ]; then
    LIB_PATH_VAR="DYLD_LIBRARY_PATH"
else
    LIB_PATH_VAR="LD_LIBRARY_PATH"
fi

# Activate venv if it exists
if [ -d "venv" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
else
    echo "WARNING: ./venv not found. Run ./scripts/setup-libtorch.sh first." >&2
fi

# Source cargo environment if available
if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

# Have tch-rs locate libtorch via the venv's PyTorch install
export LIBTORCH_USE_PYTORCH=1
# tch-rs's version check is overly strict on patch versions; bypass it (the
# Linux GPU script does the same).
export LIBTORCH_BYPASS_VERSION_CHECK=1

# Locate the PyTorch lib directory and put it on the dynamic-linker search path
TORCH_LIB="$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")"
export "$LIB_PATH_VAR"="${TORCH_LIB}:${!LIB_PATH_VAR:-}"

# Run the example, forwarding any extra args
if [ $# -gt 0 ]; then
    cargo run --example "$EXAMPLE_NAME" --release -- "$@"
else
    cargo run --example "$EXAMPLE_NAME" --release
fi
