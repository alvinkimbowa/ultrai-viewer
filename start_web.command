#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
open -a "Google Chrome" --args --app="file://$ROOT_DIR/web/index.html"
