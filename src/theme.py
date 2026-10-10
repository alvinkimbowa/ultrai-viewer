"""
Dark blue look of the app, built on UBC blue.
"""

from PyQt6.QtGui import QColor, QPalette

BACKGROUND = "#001329"
SURFACE = "#000A18"
BUTTON = "#002145"
BUTTON_HOVER = "#003A7D"
BUTTON_PRESSED = "#002A5E"
PRIMARY = "#0055B7"
ACCENT = "#2A9FD8"
BORDER = "#12335A"
TEXT = "#E8EEF5"
TEXT_DISABLED = "#5F7693"

STYLE_SHEET = f"""
QPushButton {{
    background-color: {BUTTON};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 3px 5px;
}}
QPushButton:hover {{
    background-color: {BUTTON_HOVER};
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background-color: {BUTTON_PRESSED};
}}
QPushButton:focus {{
    border-color: {ACCENT};
}}
QPushButton:disabled {{
    background-color: {BACKGROUND};
    color: {TEXT_DISABLED};
    border-color: {BORDER};
}}

QSlider::groove:horizontal {{
    height: 6px;
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 4px;
}}
QSlider::handle:horizontal {{
    background: {TEXT};
    border: 2px solid {ACCENT};
    width: 12px;
    margin: -5px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{
    background: {ACCENT};
}}
QSlider::sub-page:horizontal:disabled {{
    background: {BORDER};
}}
QSlider::handle:horizontal:disabled {{
    background: {BORDER};
    border-color: {BORDER};
}}

QCheckBox::indicator {{
    width: 13px;
    height: 13px;
    background: {SURFACE};
    border: 1px solid {TEXT_DISABLED};
    border-radius: 3px;
}}
QCheckBox::indicator:hover {{
    border-color: {ACCENT};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}

QProgressBar {{
    background: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    text-align: center;
}}
QProgressBar::chunk {{
    background: {PRIMARY};
    border-radius: 3px;
}}

QToolTip {{
    background-color: {SURFACE};
    color: {TEXT};
    border: 1px solid {ACCENT};
    padding: 4px;
}}

QMenuBar {{
    background-color: {SURFACE};
    color: {TEXT};
}}
QMenuBar::item:selected {{
    background-color: {BUTTON_HOVER};
}}
QMenu {{
    background-color: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
}}
QMenu::item:selected {{
    background-color: {BUTTON_HOVER};
}}
QMenu::item:disabled {{
    color: {TEXT_DISABLED};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 8px;
}}

QStatusBar {{
    background-color: {SURFACE};
    color: {TEXT};
    border-top: 1px solid {BORDER};
}}

QLabel#sectionHeader {{
    color: {ACCENT};
    font-size: 10pt;
    font-weight: bold;
}}
"""


def _palette():
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: BACKGROUND,
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: SURFACE,
        QPalette.ColorRole.AlternateBase: BACKGROUND,
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.Button: BUTTON,
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.BrightText: "#FFFFFF",
        QPalette.ColorRole.Highlight: PRIMARY,
        QPalette.ColorRole.HighlightedText: "#FFFFFF",
        QPalette.ColorRole.ToolTipBase: SURFACE,
        QPalette.ColorRole.ToolTipText: TEXT,
        QPalette.ColorRole.PlaceholderText: TEXT_DISABLED,
        QPalette.ColorRole.Link: ACCENT,
        QPalette.ColorRole.Light: BORDER,
        QPalette.ColorRole.Midlight: BUTTON,
        QPalette.ColorRole.Mid: BORDER,
        QPalette.ColorRole.Dark: SURFACE,
        QPalette.ColorRole.Shadow: "#000814",
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    disabled = {
        QPalette.ColorRole.WindowText: TEXT_DISABLED,
        QPalette.ColorRole.Text: TEXT_DISABLED,
        QPalette.ColorRole.ButtonText: TEXT_DISABLED,
        QPalette.ColorRole.Button: BACKGROUND,
        QPalette.ColorRole.Base: BACKGROUND,
        QPalette.ColorRole.Highlight: BORDER,
        QPalette.ColorRole.HighlightedText: TEXT_DISABLED,
    }
    for role, color in disabled.items():
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(color))
    return palette


def apply_theme(app):
    # Fusion draws every control from the palette, so the colours below reach
    # the dropdowns, spin boxes, checkboxes, scroll bars and dialogs as well.
    app.setStyle("Fusion")
    app.setPalette(_palette())
    app.setStyleSheet(STYLE_SHEET)
