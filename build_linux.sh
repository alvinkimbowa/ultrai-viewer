#!/usr/bin/env bash
set -euo pipefail

VERSION=$(cat VERSION)
export OS="linux"
export ARCH="x86_64"
export APP_NAME="UltAIViewer-v${VERSION}-${OS}-${ARCH}"

python -m PyInstaller --noconfirm --clean MTGNeuriteTracer.spec

chmod +x dist/${APP_NAME}/${APP_NAME}
echo "Build complete: dist/${APP_NAME}"
