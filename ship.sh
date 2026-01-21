#!/bin/bash

VERSION=$(cat VERSION)
APP_NAME="UltrAiViewer-v${VERSION}-linux-x86_64"
ARCHIVE_SIZE=$(du -sb "dist/${APP_NAME}" | awk '{print $1}')
tar -cf - -C dist "${APP_NAME}" | pv -s "${ARCHIVE_SIZE}" | gzip > "${APP_NAME}.tar.gz"
