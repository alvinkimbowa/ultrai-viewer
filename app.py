"""
Minimal app entry point for loading and viewing images.
"""

import sys
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
