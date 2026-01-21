@echo off
setlocal

set /p VERSION=<VERSION
set "OS=windows"
set "ARCH=x86"
set "APP_NAME=UltrAiViewer-v%VERSION%-%OS%-%ARCH%"
set "ARCHIVE_NAME=dist\\%APP_NAME%.zip"

if not exist "dist\%APP_NAME%" (
  echo Missing build output: dist\%APP_NAME%
  exit /b 1
)

tar -a -cvf "%ARCHIVE_NAME%" -C dist "%APP_NAME%"
if errorlevel 1 exit /b 1

echo Shipped: %ARCHIVE_NAME%
