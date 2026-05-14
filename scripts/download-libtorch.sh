#!/usr/bin/env bash
# Download a self-contained libtorch zip from pytorch.org into ./libtorch/.
#
# This is the "no Python" alternative to scripts/setup-libtorch.sh. The
# downloaded archive bundles its own protobuf / abseil so it is immune to
# Homebrew bottle drift on macOS.
#
# Default: PyTorch 2.9.0 CPU (matches tch-rs 0.22 expectation).
#
# Usage:
#   ./scripts/download-libtorch.sh                # default: cpu, 2.9.0
#   PYTORCH_VERSION=2.9.0 ./scripts/download-libtorch.sh
#   CUDA_TAG=cu121      ./scripts/download-libtorch.sh   # Linux GPU
#
# After running:
#     export LIBTORCH="$(pwd)/libtorch"
#     export DYLD_LIBRARY_PATH="$LIBTORCH/lib:$DYLD_LIBRARY_PATH"   # macOS
#     export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"        # Linux

set -euo pipefail

PYTORCH_VERSION="${PYTORCH_VERSION:-2.9.0}"
CUDA_TAG="${CUDA_TAG:-cpu}"   # cpu, cu121, cu124, etc.

UNAME_S="$(uname -s)"
UNAME_M="$(uname -m)"

case "$UNAME_S/$UNAME_M" in
    Darwin/arm64)
        URL="https://download.pytorch.org/libtorch/${CUDA_TAG}/libtorch-macos-arm64-${PYTORCH_VERSION}.zip"
        ;;
    Darwin/x86_64)
        URL="https://download.pytorch.org/libtorch/${CUDA_TAG}/libtorch-macos-x86_64-${PYTORCH_VERSION}.zip"
        ;;
    Linux/x86_64)
        # The cxx11-abi flavor is what tch-rs expects on Linux.
        URL="https://download.pytorch.org/libtorch/${CUDA_TAG}/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2B${CUDA_TAG}.zip"
        ;;
    *)
        echo "ERROR: Unsupported platform: $UNAME_S/$UNAME_M" >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -d libtorch ]; then
    echo "WARNING: ./libtorch already exists."
    read -r -p "Remove and re-download? [y/N] " ans
    case "$ans" in
        y|Y) rm -rf libtorch ;;
        *)   echo "Aborting."; exit 1 ;;
    esac
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

ARCHIVE="$TMPDIR/libtorch.zip"
echo "Downloading: $URL"
if command -v curl >/dev/null 2>&1; then
    curl -fL --progress-bar "$URL" -o "$ARCHIVE"
elif command -v wget >/dev/null 2>&1; then
    wget --show-progress -O "$ARCHIVE" "$URL"
else
    echo "ERROR: neither curl nor wget is installed." >&2
    exit 1
fi

echo "Extracting..."
unzip -q "$ARCHIVE" -d "$TMPDIR"
mv "$TMPDIR/libtorch" "$PROJECT_ROOT/libtorch"

echo
echo "Installed libtorch $PYTORCH_VERSION ($CUDA_TAG) to: $PROJECT_ROOT/libtorch"
echo
if [ "$UNAME_S" = "Darwin" ]; then
    echo "Add the following to your shell to use it:"
    echo "    export LIBTORCH=\"$PROJECT_ROOT/libtorch\""
    echo "    export DYLD_LIBRARY_PATH=\"\$LIBTORCH/lib:\${DYLD_LIBRARY_PATH:-}\""
else
    echo "Add the following to your shell to use it:"
    echo "    export LIBTORCH=\"$PROJECT_ROOT/libtorch\""
    echo "    export LD_LIBRARY_PATH=\"\$LIBTORCH/lib:\${LD_LIBRARY_PATH:-}\""
fi
echo
echo "Or just use the wrapper scripts (./scripts/test.sh, ./scripts/train.sh)"
echo "which auto-detect ./libtorch and set these for you."
