@echo off
setlocal

set "ROOT_DIR=%~dp0"
pushd "%ROOT_DIR%" >nul || exit /b 1

set /p "VERSION="<VERSION
if not defined VERSION (
  echo Missing or empty release file: VERSION
  popd
  exit /b 1
)

set "BUNDLE_NAME=UltrAiViewer-web-v%VERSION%"
set "OUTPUT_DIR=%ROOT_DIR%dist"
set "TAR_PATH=%OUTPUT_DIR%\%BUNDLE_NAME%.tar.gz"
set "ZIP_PATH=%OUTPUT_DIR%\%BUNDLE_NAME%.zip"
set "TEMP_DIR=%TEMP%\%BUNDLE_NAME%-%RANDOM%-%RANDOM%"
set "BUNDLE_DIR=%TEMP_DIR%\%BUNDLE_NAME%"

for %%F in (README.md VERSION web start_web.sh start_web.command start_web.bat) do (
  if not exist "%%F" (
    echo Missing required release file: %%F
    popd
    exit /b 1
  )
)

where tar >nul 2>&1
if errorlevel 1 (
  echo Required command not found: tar
  popd
  exit /b 1
)

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if errorlevel 1 goto :error

mkdir "%BUNDLE_DIR%"
if errorlevel 1 goto :error

xcopy "web" "%BUNDLE_DIR%\web\" /E /I /Q /Y >nul
if errorlevel 1 goto :error

copy /Y "README.md" "%BUNDLE_DIR%\" >nul
if errorlevel 1 goto :error
copy /Y "VERSION" "%BUNDLE_DIR%\" >nul
if errorlevel 1 goto :error
copy /Y "start_web.sh" "%BUNDLE_DIR%\" >nul
if errorlevel 1 goto :error
copy /Y "start_web.command" "%BUNDLE_DIR%\" >nul
if errorlevel 1 goto :error
copy /Y "start_web.bat" "%BUNDLE_DIR%\" >nul
if errorlevel 1 goto :error

if exist "%TAR_PATH%" del /Q "%TAR_PATH%"
if exist "%ZIP_PATH%" del /Q "%ZIP_PATH%"

tar -C "%TEMP_DIR%" -czf "%TAR_PATH%" "%BUNDLE_NAME%"
if errorlevel 1 goto :error

tar -C "%TEMP_DIR%" -a -cf "%ZIP_PATH%" "%BUNDLE_NAME%"
if errorlevel 1 goto :error

rmdir /S /Q "%TEMP_DIR%"
popd

echo Created web-app releases:
echo   %TAR_PATH%
echo   %ZIP_PATH%
exit /b 0

:error
set "EXIT_CODE=%ERRORLEVEL%"
if exist "%TEMP_DIR%" rmdir /S /Q "%TEMP_DIR%"
popd
echo Failed to create web-app release.
exit /b %EXIT_CODE%
