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

set "ISCC_EXE=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist "%ISCC_EXE%" (
    "%ISCC_EXE%" "UltrAiViewerInstaller.iss"
    if errorlevel 1 exit /b 1
    echo Build complete: dist\%APP_NAME%
    echo Installer complete: dist\%APP_NAME%-installer.exe
) else (
    echo Build complete: dist\%APP_NAME%
    echo Inno Setup not found. To build the installer, install Inno Setup 6 and rerun this script.
)
