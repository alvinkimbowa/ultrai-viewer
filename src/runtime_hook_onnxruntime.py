"""Load ONNX Runtime before PyInstaller initializes Qt on Windows."""

# PyInstaller's built-in PyQt6 runtime hook runs before app.py. Importing ORT
# here prevents Qt's bundled MSVC DLLs from being selected for ORT's native
# extension.
import onnxruntime  # noqa: F401
