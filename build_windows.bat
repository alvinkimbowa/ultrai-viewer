@echo off
setlocal

set /p VERSION=<VERSION
set "ULTAI_VERSION=%VERSION%"
set "ULTAI_OS=windows"
set "ULTAI_ARCH=x64"
set "APP_NAME=UltrAiViewer-v%VERSION%-%ULTAI_OS%-%ULTAI_ARCH%"

py -m PyInstaller --noconfirm --clean UltrAiViewer.spec

copy /Y "User Guide - UltrAi Viewer.pdf" "dist\%APP_NAME%\User Guide - UltrAi Viewer.pdf"

if errorlevel 1 exit /b 1

echo Build complete: dist\%APP_NAME%
