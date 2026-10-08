"""
Measurements computed from a segmented cartilage mask.
"""

import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


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


def cartilage_edges(mask):
    """Columns the mask covers, with the top and bottom mask row of each.

    The probe is at the top of the image, so the top row is the cartilage surface
    and the bottom row is the cartilage-bone interface.
    """
    binary = np.asarray(mask) > 0
    height = binary.shape[0]
    columns = np.nonzero(binary.any(axis=0))[0]
    top = binary.argmax(axis=0)[columns]
    bottom = (height - 1 - binary[::-1].argmax(axis=0))[columns]
    return columns, top, bottom


def bottom_edge_line(mask):
    """The cartilage-bone interface as a one-pixel-wide line image.

    Holds the lowest mask pixel of every column. Where neighbouring columns differ
    by more than one row, the deeper column is filled upwards so the line stays
    connected. Columns the mask does not cover leave a gap.
    """
    line = np.zeros(np.asarray(mask).shape[:2], dtype=bool)
    columns, _, bottom = cartilage_edges(mask)
    line[bottom, columns] = True
    for index in range(len(columns) - 1):
        if columns[index + 1] != columns[index] + 1:
            continue
        y0, y1 = int(bottom[index]), int(bottom[index + 1])
        if y1 - y0 > 1:
            line[y0 + 1 : y1, columns[index + 1]] = True
        elif y0 - y1 > 1:
            line[y1 + 1 : y0, columns[index]] = True
    return line


def interface_length(line, scale_x, scale_y):
    """Geodesic length of a bottom-edge line, summed over its connected stretches."""
    count, labels = cv2.connectedComponents(line.astype(np.uint8), connectivity=8)
    total = 0.0
    for label in range(1, count):
        length = compute_geodesic_skeleton_length_mm(labels == label, scale_x, scale_y)
        if not np.isnan(length):
            total += length
    return total


def suggest_centre_x(mask):
    """Column where the cartilage's top surface dips deepest into the image."""
    columns, top, _ = cartilage_edges(mask)
    if len(columns) == 0:
        return None
    # The rounded ends of a mask also turn downwards, so the outer tenth of the
    # columns on each side is left out of the search.
    margin = len(columns) // 10
    inner = slice(margin, len(columns) - margin)
    deepest = columns[inner][top[inner] == top[inner].max()]
    return float(deepest[len(deepest) // 2])


def surface_point(mask, centre_x):
    """The point on the top surface at centre_x, or at the nearest covered column."""
    columns, top, _ = cartilage_edges(mask)
    if len(columns) == 0:
        return None
    index = int(np.abs(columns - centre_x).argmin())
    return float(centre_x), float(top[index])


INTERCONDYLAR_WIDTH_FRACTION = 0.25
REGION_NAMES = ("lateral", "intercondylar", "medial")


def region_columns(image_width, centre_x, knee_side):
    """Column range (start, stop) of each region for a knee side of "right" or "left".

    The intercondylar region is a fixed share of the image width centred on
    centre_x. On a right knee the lateral condyle is on the left of the image.
    """
    half_width = INTERCONDYLAR_WIDTH_FRACTION * image_width / 2.0
    start = int(round(min(max(centre_x - half_width, 0), image_width)))
    stop = int(round(min(max(centre_x + half_width, 0), image_width)))
    left_name, right_name = (
        ("lateral", "medial") if knee_side == "right" else ("medial", "lateral")
    )
    return {
        left_name: (0, start),
        "intercondylar": (start, stop),
        right_name: (stop, image_width),
    }


def to_grayscale(image):
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image[:, :, 0]


MEASURES = ("area", "length", "thickness", "echo_mean", "echo_sd")


def _measure_columns(gray, binary, line, start, stop, scale_x, scale_y):
    values = dict.fromkeys(MEASURES, np.nan)
    region = binary[:, start:stop]
    if not region.any():
        return values
    values["area"] = float(region.sum()) * scale_x * scale_y
    values["length"] = interface_length(line[:, start:stop], scale_x, scale_y)
    if values["length"] > 0:
        values["thickness"] = values["area"] / values["length"]
    if gray is not None:
        pixels = gray[:, start:stop][region]
        values["echo_mean"] = float(np.mean(pixels))
        values["echo_sd"] = float(np.std(pixels))
    return values


def measure(image, mask, px_per_mm_x, px_per_mm_y, centre_x=None, knee_side="right"):
    """Measure a mask as a whole and split into lateral, intercondylar and medial.

    Returns {"unit", "centre_x", "regions"}; "regions" maps "whole" and each region
    name to its area, cartilage-bone interface length, thickness (area / length),
    and the mean and standard deviation of the grey levels inside it. Lengths are in
    mm when both resolutions are known and in pixels otherwise. centre_x of None
    uses the suggested centre. Values that cannot be computed are NaN.
    """
    in_mm = bool(px_per_mm_x and px_per_mm_y)
    scale_x = 1.0 / px_per_mm_x if in_mm else 1.0
    scale_y = 1.0 / px_per_mm_y if in_mm else 1.0
    result = {
        "unit": "mm" if in_mm else "px",
        "centre_x": None,
        "regions": {
            name: dict.fromkeys(MEASURES, np.nan) for name in ("whole",) + REGION_NAMES
        },
    }
    if mask is None:
        return result
    binary = np.asarray(mask) > 0
    if not binary.any():
        return result
    gray = None
    if image is not None and image.shape[:2] == binary.shape[:2]:
        gray = to_grayscale(image)
    line = bottom_edge_line(binary)
    width = binary.shape[1]
    if centre_x is None:
        centre_x = suggest_centre_x(binary)
    result["centre_x"] = centre_x
    spans = {"whole": (0, width), **region_columns(width, centre_x, knee_side)}
    for name, (start, stop) in spans.items():
        result["regions"][name] = _measure_columns(
            gray, binary, line, start, stop, scale_x, scale_y
        )
    return result


MEASUREMENTS_FILE_NAME = "measurements.csv"
MEASUREMENT_COLUMNS = [
    "file",
    "frame",
    "knee_side",
    "centre_x",
    "centre_set_by",
    "roi_state",
    "limit_left",
    "limit_right",
    "limit_top",
    "limit_bottom",
    "pixels_per_mm_x",
    "pixels_per_mm_y",
    "unit",
] + [
    f"{region}_{measure_name}"
    for region in ("whole",) + REGION_NAMES
    for measure_name in MEASURES
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

    def get(self, file_name, frame):
        return self._rows.get((str(file_name), str(frame)))

    def saved_roi(self, file_name, frame):
        """The ROI stored for a row.

        A box (left, right, top, bottom) when the row was saved with ROI on, False
        when ROI was switched off for it, None when it has no ROI of its own.
        """
        row = self.get(file_name, frame)
        if row is None:
            return None
        if row.get("roi_state") == "off":
            return False
        try:
            return tuple(
                int(row[column])
                for column in ("limit_left", "limit_right", "limit_top", "limit_bottom")
            )
        except (KeyError, TypeError, ValueError):
            return None

    def set(
        self,
        file_name,
        frame,
        image,
        mask,
        px_per_mm_x,
        px_per_mm_y,
        centre_x=None,
        knee_side="right",
        prefer_saved=False,
        limits=None,
        roi_off=False,
    ):
        """Measure a mask and store it as the row of its image or frame.

        centre_x of None means the suggested centre. With prefer_saved, a knee side
        and a user-placed centre already stored for the row win over the arguments.
        limits is the box (left, right, top, bottom) the mask was confined to, if any;
        roi_off records that ROI was switched off for this row in particular.
        """
        key = (str(file_name), str(frame))
        saved = self._rows.get(key)
        if prefer_saved and saved is not None:
            if saved.get("knee_side") in ("right", "left"):
                knee_side = saved["knee_side"]
            if saved.get("centre_set_by") == "user":
                try:
                    centre_x = float(saved["centre_x"])
                except (TypeError, ValueError):
                    pass
        result = measure(image, mask, px_per_mm_x, px_per_mm_y, centre_x, knee_side)
        row = {
            "file": key[0],
            "frame": key[1],
            "knee_side": knee_side,
            "centre_x": (
                "" if result["centre_x"] is None else f"{result['centre_x']:.1f}"
            ),
            "centre_set_by": "auto" if centre_x is None else "user",
            "roi_state": "on" if limits is not None else ("off" if roi_off else ""),
            "limit_left": "" if limits is None else str(limits[0]),
            "limit_right": "" if limits is None else str(limits[1]),
            "limit_top": "" if limits is None else str(limits[2]),
            "limit_bottom": "" if limits is None else str(limits[3]),
            "pixels_per_mm_x": f"{px_per_mm_x:.3f}" if px_per_mm_x else "",
            "pixels_per_mm_y": f"{px_per_mm_y:.3f}" if px_per_mm_y else "",
            "unit": result["unit"],
        }
        for region, values in result["regions"].items():
            for measure_name, value in values.items():
                row[f"{region}_{measure_name}"] = (
                    "" if np.isnan(value) else f"{value:.4f}"
                )
        if saved != row:
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
