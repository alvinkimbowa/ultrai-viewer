@echo off
setlocal

set /p VERSION=<VERSION
set "ULTAI_VERSION=%VERSION%"
set "ULTAI_OS=windows"
set "ULTAI_ARCH=x64"
set "APP_NAME=UltrAiViewer-v%VERSION%-%ULTAI_OS%-%ULTAI_ARCH%"

.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean UltrAiViewer.spec
if errorlevel 1 exit /b 1

copy /Y "User Guide - UltrAi Viewer.pdf" "dist\User Guide - UltrAi Viewer.pdf"

if errorlevel 1 exit /b 1

set "ISCC_EXE="
where ISCC.exe >nul 2>nul
if not errorlevel 1 set "ISCC_EXE=ISCC.exe"

if not defined ISCC_EXE if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC_EXE=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not defined ISCC_EXE if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC_EXE=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not defined ISCC_EXE if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC_EXE=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"

echo Build complete: dist\%APP_NAME%
if not defined ISCC_EXE (
  echo Inno Setup 6 not found; installer was not created.
  exit /b 0
)

"%ISCC_EXE%" UltrAiViewerInstaller.iss
if errorlevel 1 exit /b 1

echo Installer complete: dist\%APP_NAME%-installer.exe
