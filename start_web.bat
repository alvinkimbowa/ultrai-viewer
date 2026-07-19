@echo off
set "ROOT_DIR=%~dp0"
start "" chrome.exe --app="file:///%ROOT_DIR%web/index.html"
