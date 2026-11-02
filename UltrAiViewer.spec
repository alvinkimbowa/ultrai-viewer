#!/usr/bin/env python3
# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_dynamic_libs

app_name = os.getenv("APP_NAME").strip()

onnxruntime_bins = collect_dynamic_libs("onnxruntime")

# The models packed into the app. Other models in assets/ are left out.
MODELS = [
    "nnunet205.onnx",
    "monounet205.onnx",
]

missing = [name for name in MODELS if not os.path.isfile(os.path.join("assets", name))]
if missing:
    raise SystemExit(f"Models missing from assets/: {', '.join(missing)}")

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=onnxruntime_bins,
    datas=[
        ("assets/icons", "assets/icons"),
        *((os.path.join("assets", name), "assets") for name in MODELS),
        ("VERSION", "."),
    ],
    hiddenimports=[],
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
