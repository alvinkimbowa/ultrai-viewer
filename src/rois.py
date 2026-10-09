"""
The region of interest (ROI) drawn on each image or video frame.
"""

import csv
from pathlib import Path

from .measurements import MEASUREMENTS_FILE_NAME

ROIS_FILE_NAME = "rois.csv"
ROI_COLUMNS = ["file", "frame", "left", "right", "top", "bottom"]


def _read_boxes(path, columns):
    """Boxes of a CSV file keyed by (file, frame); rows without a full box are left out."""
    boxes = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                box = tuple(int(float(row[column])) for column in columns)
            except (KeyError, TypeError, ValueError):
                continue
            boxes[(row.get("file", ""), row.get("frame", ""))] = box
    return boxes


class RoiStore:
    """The rois.csv of one output folder, one row per image or video frame with an ROI.

    A box is (left, right, top, bottom) in image pixels. `frame` is the frame number
    for a video and "" for an image.
    """

    def __init__(self, output_dir):
        self.path = Path(output_dir) / ROIS_FILE_NAME
        self._boxes = {}
        self._dirty = False
        if self.path.exists():
            self._boxes = _read_boxes(self.path, ROI_COLUMNS[2:])
            return
        # Folders written before rois.csv existed hold each ROI only in the
        # measurements row of the mask it was used for.
        measurements = self.path.with_name(MEASUREMENTS_FILE_NAME)
        if measurements.exists():
            self._boxes = _read_boxes(
                measurements,
                ("limit_left", "limit_right", "limit_top", "limit_bottom"),
            )
            self._dirty = bool(self._boxes)

    def get(self, file_name, frame=""):
        return self._boxes.get((str(file_name), str(frame)))

    def set(self, file_name, frame, box):
        key = (str(file_name), str(frame))
        box = tuple(int(value) for value in box)
        if self._boxes.get(key) != box:
            self._boxes[key] = box
            self._dirty = True

    def remove(self, file_name, frame=""):
        if self._boxes.pop((str(file_name), str(frame)), None) is not None:
            self._dirty = True

    def remove_files(self, file_names):
        """Drop every ROI of the named files, all frames included; returns how many."""
        keys = [key for key in self._boxes if key[0] in file_names]
        for key in keys:
            del self._boxes[key]
        self._dirty = self._dirty or bool(keys)
        return len(keys)

    def boxes(self):
        return dict(self._boxes)

    def save(self):
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)

        def order(key):
            file_name, frame = key
            return file_name, int(frame) if frame.isdigit() else -1

        with open(self.path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(ROI_COLUMNS)
            for key in sorted(self._boxes, key=order):
                writer.writerow([*key, *self._boxes[key]])
        self._dirty = False
