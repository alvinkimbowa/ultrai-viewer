"""
The classes objects can be named with, kept per output folder in classes.json,
and the colour each class is drawn in.
"""

import json
import os
from pathlib import Path

from PyQt6.QtGui import QColor

from .instances import UNLABELED, clean_class

CLASSES_FILE_NAME = "classes.json"
UNLABELED_COLOR = (150, 150, 150)


def class_color(name, class_order):
    """Colour of a class: hues a golden angle apart in the order of the class
    list. Unnamed objects are grey."""
    name = clean_class(name)
    if name == UNLABELED:
        return UNLABELED_COLOR
    try:
        index = class_order.index(name)
    except ValueError:
        index = len(class_order)
    hue = (index * 137.507764) % 360
    color = QColor.fromHslF(hue / 360.0, 0.78, 0.55)
    return color.red(), color.green(), color.blue()


class ClassList:
    """classes.json of one output folder: the classes in the order they were
    added, which sets their colours."""

    def __init__(self, folder=None):
        self.path = None if folder is None else Path(folder) / CLASSES_FILE_NAME
        self.classes = []
        if self.path is not None and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for name in data.get("classes") or []:
                self.add(name)

    def class_order(self):
        return list(self.classes)

    def add(self, name):
        """Add a class; False when it is there already or is no name."""
        name = clean_class(name)
        if not name or name == UNLABELED or name in self.classes:
            return False
        self.classes.append(name)
        return True

    def rename(self, old, new):
        """Rename a class in place, keeping its colour; False when new is taken."""
        old, new = clean_class(old), clean_class(new)
        if old not in self.classes or not new or new == UNLABELED or new in self.classes:
            return False
        self.classes[self.classes.index(old)] = new
        return True

    def remove(self, name):
        name = clean_class(name)
        if name in self.classes:
            self.classes.remove(name)

    def save(self):
        """Write the file; a list of no folder keeps its changes in memory."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_name(f".{self.path.name}.partial")
        try:
            partial.write_text(
                json.dumps({"version": 1, "classes": self.classes}, indent=2), encoding="utf-8"
            )
            os.replace(partial, self.path)
        finally:
            partial.unlink(missing_ok=True)
