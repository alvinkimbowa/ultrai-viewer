#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="$(tr -d '[:space:]' < "$ROOT_DIR/VERSION")"
BUNDLE_NAME="ultrai-annotator-web-v$VERSION"
OUTPUT_DIR="$ROOT_DIR/dist"
TAR_PATH="$OUTPUT_DIR/$BUNDLE_NAME.tar.gz"
ZIP_PATH="$OUTPUT_DIR/$BUNDLE_NAME.zip"

for path in README.md VERSION web start_web.sh start_web.command start_web.bat; do
  if [[ ! -e "$ROOT_DIR/$path" ]]; then
    echo "Missing required release file: $path" >&2
    exit 1
  fi
done

for command_name in tar zip; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf -- "$TEMP_DIR"' EXIT
BUNDLE_DIR="$TEMP_DIR/$BUNDLE_NAME"
mkdir -p "$BUNDLE_DIR"

cp -R "$ROOT_DIR/web" "$BUNDLE_DIR/web"
cp "$ROOT_DIR/README.md" "$ROOT_DIR/VERSION" "$BUNDLE_DIR/"
cp "$ROOT_DIR/start_web.sh" "$ROOT_DIR/start_web.command" "$ROOT_DIR/start_web.bat" "$BUNDLE_DIR/"
chmod +x "$BUNDLE_DIR/start_web.sh" "$BUNDLE_DIR/start_web.command"

rm -f "$TAR_PATH" "$ZIP_PATH"
tar -C "$TEMP_DIR" -czf "$TAR_PATH" "$BUNDLE_NAME"
(
  cd "$TEMP_DIR"
  zip -qr "$ZIP_PATH" "$BUNDLE_NAME"
)

echo "Created web-app releases:"
echo "  $TAR_PATH"
echo "  $ZIP_PATH"
