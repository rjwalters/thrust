#!/usr/bin/env bash
# DEPRECATED: Use ./scripts/remote-optimize.sh or ./scripts/gpu-optimize.sh instead
#
# This script is kept for backwards compatibility but the new scripts
# have proper GPU support with CUDA library preloading.

echo "⚠️  DEPRECATED: This script doesn't properly enable GPU acceleration"
echo ""
echo "Please use one of these instead:"
echo ""
echo "  From local machine (launches on remote):"
echo "    ./scripts/remote-optimize.sh [--sync-first] [--trials N]"
echo ""
echo "  On GPU machine directly:"
echo "    ./scripts/gpu-optimize.sh [--trials N]"
echo ""
echo "See scripts/train-gpu.sh for the proper GPU setup pattern."
echo ""
exit 1
