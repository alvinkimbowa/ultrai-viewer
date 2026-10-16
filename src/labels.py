"""
Label sets (locations, nerves, other anatomy), the per-file location and side,
and the chip buttons they are picked with.

Both are kept in nerve_manifest.json in the selected output folder, in the
format the web annotator reads and writes. Videos are listed under "videos"
by file name, as the web annotator lists them; images go under "images".
"""

import json
import os
from pathlib import Path

from PyQt6.QtCore import QPoint, QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import QButtonGroup, QLayout, QPushButton, QWidget

from .instances import clean_class, display_class
from .theme import BORDER, BUTTON, BUTTON_HOVER, PRIMARY, SURFACE, TEXT

DEFAULT_LOCATIONS = ["wrist", "mid-arm", "elbow"]
DEFAULT_NERVES = [
    "median", "ulnar", "radial", "plex", "lfcn", "peroneal", "fibular",
    "tibial", "sural", "proximal", "accessory", "quad", "sciatic", "unknown",
]
DEFAULT_ANATOMY = [
    "muscle", "artery", "vein", "skin", "subcutaneous tissue", "cartilage",
    "tendon", "bone",
]
MANIFEST_FILE_NAME = "nerve_manifest.json"
SIDES = ("left", "right")


def class_color(name, class_order):
    """Colour of a class: hues a golden angle apart in the order of the label
    sets, the same colours the web annotator uses."""
    try:
        index = class_order.index(clean_class(name))
    except ValueError:
        index = len(class_order)
    hue = (index * 137.507764) % 360
    color = QColor.fromHslF(hue / 360.0, 0.78, 0.55)
    return color.red(), color.green(), color.blue()


def _merged(defaults, extra):
    labels = list(defaults)
    for label in extra if isinstance(extra, list) else []:
        clean = clean_class(label)
        if clean and clean not in labels:
            labels.append(clean)
    return labels


class NerveManifest:
    """nerve_manifest.json of one output folder."""

    def __init__(self, folder=None):
        self.path = None if folder is None else Path(folder) / MANIFEST_FILE_NAME
        self.locations = list(DEFAULT_LOCATIONS)
        self.nerves = list(DEFAULT_NERVES)
        self.anatomy = list(DEFAULT_ANATOMY)
        # Entries of the file as read, by "videos"/"images" and file name, so
        # that fields this app does not know are written back unchanged.
        self._entries = {"videos": {}, "images": {}}
        if self.path is not None and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.locations = _merged(DEFAULT_LOCATIONS, data.get("location_set"))
            self.nerves = _merged(DEFAULT_NERVES, data.get("label_set"))
            self.anatomy = _merged(DEFAULT_ANATOMY, data.get("anatomy_set"))
            for kind, key in (("videos", "video_name"), ("images", "image_name")):
                for item in data.get(kind) or []:
                    if isinstance(item, dict) and item.get(key):
                        self._entries[kind][str(item[key])] = dict(item)

    def class_order(self):
        return self.nerves + self.anatomy

    def _entry(self, file_name, is_video):
        return self._entries["videos" if is_video else "images"].get(file_name, {})

    def location(self, file_name, is_video):
        return clean_class(self._entry(file_name, is_video).get("location_label")) or None

    def side(self, file_name, is_video):
        side = clean_class(self._entry(file_name, is_video).get("side"))
        return side if side in SIDES else None

    def _update(self, file_name, is_video, field, value):
        kind, key = ("videos", "video_name") if is_video else ("images", "image_name")
        entry = self._entries[kind].setdefault(file_name, {key: file_name})
        entry[field] = value or ""

    def set_location(self, file_name, is_video, location):
        self._update(file_name, is_video, "location_label", location)

    def set_side(self, file_name, is_video, side):
        self._update(file_name, is_video, "side", side)

    def add_label(self, kind, label):
        """Add a label to "locations", "nerves" or "anatomy"; False if it exists."""
        label = clean_class(label)
        taken = self.locations if kind == "locations" else self.class_order()
        if not label or label in taken:
            return False
        getattr(self, kind).append(label)
        return True

    def save(self):
        """Write the file; a manifest of no folder keeps its changes in memory."""
        if self.path is None:
            return
        payload = {
            "version": 3,
            "location_set": self.locations,
            "label_set": self.nerves,
            "anatomy_set": self.anatomy,
            "videos": list(self._entries["videos"].values()),
            "images": list(self._entries["images"].values()),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_name(f".{self.path.name}.partial")
        try:
            partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(partial, self.path)
        finally:
            partial.unlink(missing_ok=True)


class FlowLayout(QLayout):
    """Lays its widgets out left to right, wrapping onto new lines."""

    def __init__(self, parent=None, h_spacing=4, v_spacing=4):
        super().__init__(parent)
        self._items = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize(0, 0)
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def _do_layout(self, rect, test_only):
        x, y = rect.x(), rect.y()
        line_height = 0
        for item in self._items:
            hint = item.sizeHint()
            if line_height > 0 and x + hint.width() > rect.right():
                x = rect.x()
                y += line_height + self._v_spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._h_spacing
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y()


CHIP_STYLE = f"""
QPushButton {{
    padding: 3px 9px;
    border: 1px solid {BORDER};
    border-radius: 10px;
    background: {SURFACE};
    color: {TEXT};
}}
QPushButton:hover {{ background: {BUTTON_HOVER}; }}
QPushButton:checked {{
    background: {PRIMARY};
    border-color: {PRIMARY};
    color: white;
}}
"""
ADD_CHIP_STYLE = CHIP_STYLE.replace(f"background: {SURFACE}", f"background: {BUTTON}")


class LabelChips(QWidget):
    """A row of chips of which at most one is checked, ending in a "+ Add" chip.

    picked carries the label of a chip the user checks, or "" when they uncheck
    the checked chip. add_requested asks for a new label to be added.
    """

    picked = pyqtSignal(str)
    add_requested = pyqtSignal()

    def __init__(self, add_text, parent=None):
        super().__init__(parent)
        self._layout = FlowLayout(self)
        self._group = None
        self._buttons = {}
        self._checked = None
        self._add_button = QPushButton(add_text)
        self._add_button.setStyleSheet(ADD_CHIP_STYLE)
        self._add_button.clicked.connect(self.add_requested)

    def set_labels(self, labels, colors=None):
        """Rebuild the chips; colors maps a label to the (r, g, b) of its dot."""
        while self._layout.count():
            widget = self._layout.takeAt(0).widget()
            if widget is not None and widget is not self._add_button:
                widget.deleteLater()
        self._buttons = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(False)
        for label in labels:
            button = QPushButton(display_class(label))
            button.setCheckable(True)
            button.setStyleSheet(CHIP_STYLE)
            if colors and label in colors:
                dot = QPixmap(10, 10)
                dot.fill(QColor(*colors[label]))
                button.setIcon(QIcon(dot))
            button.setChecked(label == self._checked)
            button.clicked.connect(
                lambda checked, label=label: self._on_clicked(label, checked)
            )
            self._group.addButton(button)
            self._buttons[label] = button
            self._layout.addWidget(button)
        self._layout.addWidget(self._add_button)
        if self._checked not in self._buttons:
            self._checked = None
        self.updateGeometry()

    def checked(self):
        return self._checked

    def set_checked(self, label):
        """Check a chip, or none with None, without emitting picked."""
        self._checked = label if label in self._buttons else None
        for name, button in self._buttons.items():
            button.setChecked(name == self._checked)

    def _on_clicked(self, label, checked):
        self.set_checked(label if checked else None)
        self.picked.emit(self._checked or "")

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._layout.heightForWidth(width)

    def sizeHint(self):
        return QSize(self.width() or 200, self.heightForWidth(self.width() or 200))
