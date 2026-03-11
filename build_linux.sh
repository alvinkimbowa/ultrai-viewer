#!/usr/bin/env bash
set -euo pipefail

VERSION=$(cat VERSION)
export OS="linux"
export ARCH="x86_64"
export APP_NAME="UltrAi Viewer-v${VERSION}-${OS}-${ARCH}"

python -m PyInstaller --noconfirm --clean "UltrAi Viewer.spec"

cp "User Guide - UltrAi Viewer.pdf" "dist/${APP_NAME}/User Guide - UltrAi Viewer.pdf"

chmod +x dist/${APP_NAME}/${APP_NAME}
echo "Build complete: dist/${APP_NAME}"
