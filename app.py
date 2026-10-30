"""
Minimal app entry point for loading and viewing images.
"""

import ctypes
import sys
from pathlib import Path

# Load ONNX Runtime before Qt creates QApplication. On Windows, Qt loads its
# own copies of the MSVC runtime DLLs; if those are loaded first, ONNX Runtime's
# native extension can fail to initialize with WinError 1114.
import onnxruntime  # noqa: F401

from PyQt6.QtWidgets import QApplication
from src.main_window import MainWindow
from src.theme import apply_theme


def load_xcb_cursor():
    """Make libxcb-cursor available to Qt's xcb plugin on Linux.

    The plugin links against this library, which no pip package provides and
    which many distributions leave out. Once a library is loaded, the dynamic
    linker reuses it by soname, so the plugin finds the copy in vendor/. A
    system install takes precedence.
    """
    if not sys.platform.startswith("linux"):
        return
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    bundled = root / "vendor" / "linux" / "libxcb-cursor.so.0"
    for library in ("libxcb-cursor.so.0", bundled):
        try:
            ctypes.CDLL(str(library))
            return
        except OSError:
            continue


def main():
    load_xcb_cursor()
    app = QApplication(sys.argv)
    apply_theme(app)
    app.setApplicationName("UltrAi Nerve Viewer")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
