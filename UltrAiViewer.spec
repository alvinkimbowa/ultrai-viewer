#!/usr/bin/env python3
# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

app_name = os.getenv("APP_NAME").strip()

onnxruntime_bins = collect_dynamic_libs("onnxruntime")
if sys.platform.startswith("linux"):
    onnxruntime_bins.append(("vendor/linux/libxcb-cursor.so.0", "vendor/linux"))

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=onnxruntime_bins,
    datas=[
        ("assets", "assets"),
        ("VERSION", "."),
    ],
    hiddenimports=collect_submodules("pydicom.pixels"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["src/runtime_hook_onnxruntime.py"],
    excludes=['PySide6', 'shiboken6'],  # Keep only PyQt6
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=app_name,
)
