"""
Area of each segmented instance, and the measurements file that records it.
"""

import csv
import os
from pathlib import Path

import cv2
import numpy as np


def to_grayscale(image):
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image[:, :, 0]


def area_unit(px_per_mm_x, px_per_mm_y):
    """"mm" when both resolutions are known, "px" otherwise."""
    return "mm" if px_per_mm_x and px_per_mm_y else "px"


def measure_instances(instances, px_per_mm_x, px_per_mm_y):
    """Area of each instance: [{"class", "id", "area_px", "area"}].

    "area" is in mm² when both resolutions are known and in pixels otherwise.
    """
    scale = 1.0
    if area_unit(px_per_mm_x, px_per_mm_y) == "mm":
        scale = 1.0 / (px_per_mm_x * px_per_mm_y)
    results = []
    for instance in instances:
        pixels = int(np.count_nonzero(instance["mask"]))
        if pixels == 0:
            continue
        results.append(
            {
                "class": instance["class"],
                "id": instance["id"],
                "area_px": pixels,
                "area": pixels * scale,
            }
        )
    return results


MEASUREMENTS_FILE_NAME = "measurements.csv"
LIMIT_COLUMNS = ["limit_left", "limit_right", "limit_top", "limit_bottom"]
MEASUREMENT_COLUMNS = [
    "file",
    "frame",
    "side",
    "location",
    "class",
    "instance_id",
    "area_px",
    "area",
    "unit",
    "pixels_per_mm_x",
    "pixels_per_mm_y",
    "rotation",
] + LIMIT_COLUMNS


class MeasurementLog:
    """A measurements file of one output folder, one row per instance.

    `frame` is the frame number for a video and "" for an image. The rows of one
    image or frame are kept together and always replaced together.
    """

    def __init__(self, output_dir, file_name=MEASUREMENTS_FILE_NAME):
        self.path = Path(output_dir) / file_name
        self._rows = {}
        self._dirty = False
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    key = (row.get("file", ""), row.get("frame", ""))
                    self._rows.setdefault(key, []).append(
                        {column: row.get(column, "") for column in MEASUREMENT_COLUMNS}
                    )

    def get(self, file_name, frame):
        """The rows of an image or frame, or None when it has none."""
        return self._rows.get((str(file_name), str(frame)))

    def put(self, file_name, frame, rows):
        """Store rows exactly as given, or remove them when given None."""
        key = (str(file_name), str(frame))
        if not rows:
            self.remove(file_name, frame)
        elif self._rows.get(key) != rows:
            self._rows[key] = [dict(row) for row in rows]
            self._dirty = True

    def keys(self):
        """(file, frame) of every image or frame with rows."""
        return list(self._rows)

    def set(
        self,
        file_name,
        frame,
        instances,
        px_per_mm_x,
        px_per_mm_y,
        side=None,
        location=None,
        limits=None,
        rotation=0.0,
    ):
        """Measure the instances of an image or frame and store them as its rows.

        side is "left", "right" or None, location the scanned site or None.
        limits is the ROI box (left, right, top, bottom) the masks were confined
        to, if any. rotation is the angle, in degrees clockwise, the image and
        masks were turned by before being measured.
        """
        unit = area_unit(px_per_mm_x, px_per_mm_y)
        shared = {
            "file": str(file_name),
            "frame": str(frame),
            "side": side or "",
            "location": location or "",
            "unit": unit,
            "pixels_per_mm_x": f"{px_per_mm_x:.3f}" if px_per_mm_x else "",
            "pixels_per_mm_y": f"{px_per_mm_y:.3f}" if px_per_mm_y else "",
            "rotation": f"{rotation:.1f}",
        }
        for column, value in zip(LIMIT_COLUMNS, limits or ("",) * 4):
            shared[column] = str(value)
        rows = []
        for result in measure_instances(instances, px_per_mm_x, px_per_mm_y):
            rows.append(
                {
                    **shared,
                    "class": result["class"],
                    "instance_id": str(result["id"]),
                    "area_px": str(result["area_px"]),
                    "area": f"{result['area']:.4f}",
                }
            )
        self.put(file_name, frame, rows)

    def limits(self, file_name, frame):
        """The ROI box (left, right, top, bottom) recorded for a frame, or None."""
        rows = self.get(file_name, frame)
        if not rows:
            return None
        try:
            return tuple(int(rows[0][column]) for column in LIMIT_COLUMNS)
        except (KeyError, TypeError, ValueError):
            return None

    def rotation(self, file_name, frame):
        """The angle a frame was measured at; 0 for one that records none."""
        rows = self.get(file_name, frame)
        try:
            return float(rows[0]["rotation"])
        except (TypeError, IndexError, KeyError, ValueError):
            return 0.0

    def matches_inputs(
        self, file_name, frame, side, location, limits, rotation, px_per_mm
    ):
        """True when a frame has rows and they record this side, location, ROI
        box, rotation and pixels per mm."""
        rows = self.get(file_name, frame)
        if not rows:
            return False
        first = rows[0]
        if first.get("side") != (side or "") or first.get("location") != (
            location or ""
        ):
            return False
        if self.rotation(file_name, frame) != round(float(rotation), 1):
            return False
        x, y = px_per_mm
        if first.get("pixels_per_mm_x") != (f"{x:.3f}" if x else ""):
            return False
        if first.get("pixels_per_mm_y") != (f"{y:.3f}" if y else ""):
            return False
        recorded = self.limits(file_name, frame)
        return recorded == (None if limits is None else tuple(limits))

    def clear_limits(self, file_names):
        """Blank the recorded ROI box in every row of the named files."""
        for (file_name, _), rows in self._rows.items():
            if file_name not in file_names:
                continue
            for row in rows:
                for column in LIMIT_COLUMNS:
                    if row.get(column):
                        row[column] = ""
                        self._dirty = True

    def remove(self, file_name, frame):
        if self._rows.pop((str(file_name), str(frame)), None) is not None:
            self._dirty = True

    def save(self):
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)

        def order(key):
            file_name, frame = key
            return file_name, int(frame) if frame.isdigit() else -1

        # Written under another name and then moved into place, so that the
        # file is never read half-written.
        partial = self.path.with_name(f".{self.path.name}.partial")
        try:
            with open(partial, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=MEASUREMENT_COLUMNS)
                writer.writeheader()
                for key in sorted(self._rows, key=order):
                    writer.writerows(self._rows[key])
            os.replace(partial, self.path)
        finally:
            partial.unlink(missing_ok=True)
        self._dirty = False
