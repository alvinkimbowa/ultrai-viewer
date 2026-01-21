@echo off
setlocal

set /p VERSION=<VERSION
set "ULTAI_VERSION=%VERSION%"
set "ULTAI_OS=windows"
set "ULTAI_ARCH=x86"
set "APP_NAME=UltAIViewer-%VERSION%-%ULTAI_OS%-%ULTAI_ARCH%"

py -m PyInstaller --noconfirm --clean UltAiViewer.spec
if errorlevel 1 exit /b 1

echo Build complete: dist\%APP_NAME%
