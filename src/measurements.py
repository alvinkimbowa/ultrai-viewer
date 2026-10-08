"""
Measurements computed from a segmented cartilage mask.
"""

import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from skimage.morphology import skeletonize


def binarize_mask(mask):
    return (mask > 0).astype(np.uint8)


def compute_geodesic_skeleton_length_mm(skeleton, scale_x, scale_y):
    coords = np.column_stack(np.nonzero(skeleton))
    if len(coords) == 0:
        return np.nan

    coord_to_idx = {tuple(coord): i for i, coord in enumerate(coords)}
    rows, cols, weights = [], [], []
    degrees = np.zeros(len(coords), dtype=int)

    for i, (y, x) in enumerate(coords):
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                neighbor = (y + dy, x + dx)
                j = coord_to_idx.get(neighbor)
                if j is None:
                    continue
                degrees[i] += 1
                if j > i:
                    dist = np.sqrt((dy * scale_y) ** 2 + (dx * scale_x) ** 2)
                    rows.extend([i, j])
                    cols.extend([j, i])
                    weights.extend([dist, dist])

    graph = csr_matrix((weights, (rows, cols)), shape=(len(coords), len(coords)))
    endpoints = np.where(degrees == 1)[0]

    if len(endpoints) >= 2:
        dists = shortest_path(graph, directed=False, indices=endpoints)
        endpoint_dists = dists[:, endpoints]
        finite = endpoint_dists[np.isfinite(endpoint_dists)]
    else:
        dists = shortest_path(graph, directed=False)
        finite = dists[np.isfinite(dists)]

    finite = finite[finite > 0]
    if finite.size == 0:
        return np.nan
    return float(finite.max())


def compute_cartilage_thickness(mask, px_per_mm_x, px_per_mm_y):
    """Average thickness in mm: mask area divided by the length of its centre line."""
    binary_mask = binarize_mask(mask)
    if not binary_mask.any() or not px_per_mm_x or not px_per_mm_y:
        return np.nan

    scale_x_mm = 1.0 / px_per_mm_x
    scale_y_mm = 1.0 / px_per_mm_y
    skeleton = skeletonize(binary_mask.astype(bool))
    length_mm = compute_geodesic_skeleton_length_mm(skeleton, scale_x_mm, scale_y_mm)
    if length_mm == 0 or np.isnan(length_mm):
        return np.nan

    area_mm2 = binary_mask.sum() * scale_x_mm * scale_y_mm
    return area_mm2 / length_mm


def to_grayscale(image):
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image[:, :, 0]


def compute_echo_intensity(image, mask):
    """Mean grey level of the image pixels that lie inside the mask."""
    binary_mask = binarize_mask(mask).astype(bool)
    if not binary_mask.any():
        return np.nan
    return float(np.mean(to_grayscale(image)[binary_mask]))


def measure(image, mask, px_per_mm_x, px_per_mm_y):
    """Measure a mask, returning (thickness, thickness unit, echo intensity).

    Thickness is in mm when both resolutions are known and in pixels otherwise.
    Values that cannot be computed are NaN.
    """
    in_mm = bool(px_per_mm_x and px_per_mm_y)
    thickness = np.nan
    echo_intensity = np.nan
    if mask is not None:
        thickness = compute_cartilage_thickness(
            mask,
            px_per_mm_x if in_mm else 1.0,
            px_per_mm_y if in_mm else 1.0,
        )
        if image is not None and image.shape[:2] == mask.shape[:2]:
            echo_intensity = compute_echo_intensity(image, mask)
    return thickness, "mm" if in_mm else "px", echo_intensity


MEASUREMENTS_FILE_NAME = "measurements.csv"
MEASUREMENT_COLUMNS = [
    "file",
    "frame",
    "average_thickness",
    "thickness_unit",
    "average_echo_intensity_au",
    "pixels_per_mm_x",
    "pixels_per_mm_y",
]


class MeasurementLog:
    """The measurements.csv of one output folder, one row per image or video frame.

    `frame` is the frame number for a video and "" for an image.
    """

    def __init__(self, output_dir):
        self.path = Path(output_dir) / MEASUREMENTS_FILE_NAME
        self._rows = {}
        self._dirty = False
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    key = (row.get("file", ""), row.get("frame", ""))
                    self._rows[key] = {
                        column: row.get(column, "") for column in MEASUREMENT_COLUMNS
                    }

    def set(self, file_name, frame, image, mask, px_per_mm_x, px_per_mm_y):
        thickness, unit, echo_intensity = measure(image, mask, px_per_mm_x, px_per_mm_y)
        row = {
            "file": str(file_name),
            "frame": str(frame),
            "average_thickness": "" if np.isnan(thickness) else f"{thickness:.4f}",
            "thickness_unit": unit,
            "average_echo_intensity_au": (
                "" if np.isnan(echo_intensity) else f"{echo_intensity:.2f}"
            ),
            "pixels_per_mm_x": f"{px_per_mm_x:.3f}" if px_per_mm_x else "",
            "pixels_per_mm_y": f"{px_per_mm_y:.3f}" if px_per_mm_y else "",
        }
        key = (row["file"], row["frame"])
        if self._rows.get(key) != row:
            self._rows[key] = row
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

        with open(self.path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MEASUREMENT_COLUMNS)
            writer.writeheader()
            for key in sorted(self._rows, key=order):
                writer.writerow(self._rows[key])
        self._dirty = False
