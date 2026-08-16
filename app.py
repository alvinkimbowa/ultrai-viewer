"""
Minimal app entry point for loading and viewing images.
"""

import sys

# Load ONNX Runtime before Qt creates QApplication. On Windows, Qt loads its
# own copies of the MSVC runtime DLLs; if those are loaded first, ONNX Runtime's
# native extension can fail to initialize with WinError 1114.
import onnxruntime  # noqa: F401

from PyQt6.QtWidgets import QApplication
from src.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("UltAI Viewer")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
