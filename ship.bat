@echo off
setlocal

set /p VERSION=<VERSION
set "OS=windows"
set "ARCH=x64"
set "APP_NAME=UltrAiViewer-v%VERSION%-%OS%-%ARCH%"
set "ARCHIVE_NAME=dist\\%APP_NAME%.zip"
set "INSTALLER_NAME=%APP_NAME%-installer.exe"
set "USER_GUIDE=User Guide - UltrAi Viewer.pdf"

if not exist "dist\%APP_NAME%" (
  echo Missing build output: dist\%APP_NAME%
  exit /b 1
)

if not exist "dist\%INSTALLER_NAME%" (
  echo Missing installer: dist\%INSTALLER_NAME%
  echo Run build_windows.bat first.
  exit /b 1
)

if not exist "dist\%USER_GUIDE%" (
  echo Missing user guide: dist\%USER_GUIDE%
  exit /b 1
)

tar -a -cvf "%ARCHIVE_NAME%" -C dist "%APP_NAME%" "%INSTALLER_NAME%" "%USER_GUIDE%"
if errorlevel 1 exit /b 1

echo Shipped: %ARCHIVE_NAME%
