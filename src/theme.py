"""
Light grey look of the app with near-black text, for reading in bright light.
Blue marks what is selected or active.
"""

from PyQt6.QtGui import QColor, QPalette

BACKGROUND = "#EDEDED"
SURFACE = "#FFFFFF"
BUTTON = "#E4E4E4"
BUTTON_HOVER = "#D4D4D4"
BUTTON_PRESSED = "#C2C2C2"
PRIMARY = "#0B5CAD"
ACCENT = "#0B5CAD"
BORDER = "#9A9A9A"
TEXT = "#141414"
TEXT_DISABLED = "#8A8A8A"
# Behind the image, darker than the panels so the image stands out.
CANVAS = "#6E6E6E"

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
    background-color: {BACKGROUND};
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
    background-color: {BACKGROUND};
    color: {TEXT};
    border-top: 1px solid {BORDER};
}}

QLabel#sectionHeader {{
    color: {TEXT};
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
        QPalette.ColorRole.Light: SURFACE,
        QPalette.ColorRole.Midlight: BUTTON,
        QPalette.ColorRole.Mid: BORDER,
        QPalette.ColorRole.Dark: BORDER,
        QPalette.ColorRole.Shadow: "#5A5A5A",
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
