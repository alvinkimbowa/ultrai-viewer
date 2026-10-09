"""
The rotation a file is viewed, segmented and measured at.
"""

import csv
from pathlib import Path

import cv2
import numpy as np

ROTATIONS_FILE_NAME = "rotations.csv"
ROTATION_COLUMNS = ["file", "angle"]


def _transform(shape, angle):
    """Matrix turning an image of this shape clockwise by angle degrees about its
    centre, and the (width, height) that holds the whole turned image."""
    height, width = shape[:2]
    # OpenCV counts angles counter-clockwise.
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), -angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(np.ceil(width * cos + height * sin - 1e-9))
    new_height = int(np.ceil(width * sin + height * cos - 1e-9))
    matrix[0, 2] += new_width / 2.0 - width / 2.0
    matrix[1, 2] += new_height / 2.0 - height / 2.0
    return matrix, (new_width, new_height)


def rotate_image(image, angle):
    """Turn an image clockwise by angle degrees, enlarged so that nothing is cut
    off; the corners left empty are black."""
    if not angle:
        return image
    matrix, size = _transform(image.shape, angle)
    return cv2.warpAffine(image, matrix, size, flags=cv2.INTER_LINEAR)


def rotate_mask(mask, angle):
    """Turn a mask the way rotate_image turns its image. Returns a bool array."""
    binary = np.asarray(mask) > 0
    if not angle:
        return binary
    matrix, size = _transform(binary.shape, angle)
    turned = cv2.warpAffine(
        binary.astype(np.uint8), matrix, size, flags=cv2.INTER_NEAREST
    )
    return turned > 0


def unrotate_mask(mask, angle, original_shape):
    """Bring a mask drawn on a turned image back onto the image as stored.

    original_shape is the (height, width) of the stored image. Returns a bool array.
    """
    binary = np.asarray(mask) > 0
    if not angle:
        return binary
    matrix, _ = _transform(original_shape, angle)
    height, width = original_shape[:2]
    turned = cv2.warpAffine(
        binary.astype(np.uint8),
        cv2.invertAffineTransform(matrix),
        (width, height),
        flags=cv2.INTER_NEAREST,
    )
    return turned > 0


class RotationStore:
    """The rotations.csv of one output folder, one row per rotated image or video.

    An angle is in degrees, clockwise; a video has one angle for all its frames.
    """

    def __init__(self, output_dir):
        self.path = Path(output_dir) / ROTATIONS_FILE_NAME
        self._angles = {}
        self._dirty = False
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    try:
                        angle = float(row["angle"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if angle:
                        self._angles[row.get("file", "")] = angle

    def get(self, file_name):
        return self._angles.get(str(file_name), 0.0)

    def set(self, file_name, angle):
        key, angle = str(file_name), round(float(angle), 1)
        if self.get(key) == angle:
            return
        if angle:
            self._angles[key] = angle
        else:
            del self._angles[key]
        self._dirty = True

    def angles(self):
        return dict(self._angles)

    def save(self):
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(ROTATION_COLUMNS)
            for file_name in sorted(self._angles):
                writer.writerow([file_name, f"{self._angles[file_name]:.1f}"])
        self._dirty = False
