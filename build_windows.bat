@echo off
setlocal

set /p VERSION=<VERSION
set "ULTAI_VERSION=%VERSION%"
set "ULTAI_OS=windows"
set "ULTAI_ARCH=x64"
set "APP_NAME=UltrAiViewer-v%VERSION%-%ULTAI_OS%-%ULTAI_ARCH%"

py -m PyInstaller --noconfirm --clean UltrAiViewer.spec

copy /Y "User Guide - UltrAi Viewer.pdf" "dist\User Guide - UltrAi Viewer.pdf"

if errorlevel 1 exit /b 1

set "ISCC_EXE="
set "PF86=%ProgramFiles(x86)%"

where ISCC.exe >nul 2>nul
if not errorlevel 1 set "ISCC_EXE=ISCC.exe"

if not defined ISCC_EXE if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" (
    set "ISCC_EXE=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
)

if not defined ISCC_EXE if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" (
    set "ISCC_EXE=%ProgramFiles%\Inno Setup 6\ISCC.exe"
)

if not defined ISCC_EXE if defined PF86 if exist "%PF86%\Inno Setup 6\ISCC.exe" (
    set "ISCC_EXE=%PF86%\Inno Setup 6\ISCC.exe"
)

if defined ISCC_EXE (
    "%ISCC_EXE%" "UltrAiViewerInstaller.iss"
    if errorlevel 1 exit /b 1
    echo Build complete: dist\%APP_NAME%
    echo Installer complete: dist\%APP_NAME%-installer.exe
) else (
    echo Build complete: dist\%APP_NAME%
    echo Inno Setup not found. To build the installer, install Inno Setup 6 and rerun this script.
)
