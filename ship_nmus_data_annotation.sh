#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$ROOT_DIR/dist"
ARCHIVE_PATH="$OUTPUT_DIR/nmus_data_annotation.tar.gz"

for path in README.md VERSION web videos start_web.sh start_web.command start_web.bat; do
  if [[ ! -e "$ROOT_DIR/$path" ]]; then
    echo "Missing required release file: $path" >&2
    exit 1
  fi
done

if ! command -v tar >/dev/null 2>&1; then
  echo "Required command not found: tar" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf -- "$TEMP_DIR"' EXIT
APP_DIR="$TEMP_DIR/UltrAiViewer"

mkdir -p "$APP_DIR" "$TEMP_DIR/annotations"
cp -R "$ROOT_DIR/web" "$APP_DIR/web"
cp "$ROOT_DIR/README.md" "$ROOT_DIR/VERSION" "$APP_DIR/"
cp "$ROOT_DIR/start_web.sh" "$ROOT_DIR/start_web.command" "$ROOT_DIR/start_web.bat" "$APP_DIR/"
chmod +x "$APP_DIR/start_web.sh" "$APP_DIR/start_web.command"

rm -f "$ARCHIVE_PATH"
tar -czf "$ARCHIVE_PATH" -C "$ROOT_DIR" videos -C "$TEMP_DIR" annotations UltrAiViewer

echo "Created annotation data release:"
echo "  $ARCHIVE_PATH"
