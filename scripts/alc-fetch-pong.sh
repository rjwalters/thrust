#!/usr/bin/env bash
# Fetch trained Pong model from alc-2 and copy to web/public/
#
# Usage:
#   ./scripts/alc-fetch-pong.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

REMOTE="alc-2"

echo "📥 Downloading pong_model.json from $REMOTE..."
rsync -az --progress "$REMOTE:~/thrust/pong_model.json" "$PROJECT_ROOT/web/public/pong_model.json"

echo ""
echo "✅ Model ready at web/public/pong_model.json"
ls -lh "$PROJECT_ROOT/web/public/pong_model.json"
echo ""
echo "Test locally:  cd web && npm run dev"
echo "Deploy:        git add web/public/pong_model.json && git commit -m 'Deploy trained Pong model' && git push"
