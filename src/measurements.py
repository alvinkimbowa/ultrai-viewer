"""
Measurements computed from a segmented cartilage mask.
"""

import csv
import os
from pathlib import Path

import cv2
import numpy as np


def binarize_mask(mask):
    return (mask > 0).astype(np.uint8)


def largest_piece(mask):
    """The mask with only its largest connected piece kept, as a bool array.

    Pixels that touch at a corner count as connected. Of pieces of equal size,
    the one met first going down the image from the top-left is kept.
    """
    binary = np.asarray(mask) > 0
    count, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    if count <= 2:
        return binary
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def outline_sides(mask):
    """The outline of a single-piece mask, split into its top and bottom sides.

    The probe is at the top of the image, so the top side is the cartilage surface
    and the bottom side is the cartilage-bone interface. Each side is the (x, y)
    pixels of the outline from the leftmost to the rightmost column of the mask,
    pixel by pixel as traced. Where an end of the mask is a vertical edge, the top
    side ends at its highest pixel and the bottom side at its lowest, so the edge
    itself belongs to neither side.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = contours[0][:, 0, :].astype(np.int64)
    xs, ys = points[:, 0], points[:, 1]
    left = np.nonzero(xs == xs.min())[0]
    right = np.nonzero(xs == xs.max())[0]
    count = len(points)

    def stretch(first, last):
        """The outline points met going forward round the loop from first to last."""
        return points[(first + np.arange((last - first) % count + 1)) % count]

    top_left, top_right = int(left[ys[left].argmin()]), int(right[ys[right].argmin()])
    bottom_left, bottom_right = int(left[ys[left].argmax()]), int(right[ys[right].argmax()])
    # The sign of the enclosed area gives the tracing direction: positive runs
    # left end -> top -> right end -> bottom on screen, where y points down.
    area = np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys)
    if area > 0:
        return stretch(top_left, top_right), stretch(bottom_right, bottom_left)[::-1]
    return stretch(top_right, top_left)[::-1], stretch(bottom_left, bottom_right)


def top_surface(top_side):
    """Columns the top side of an outline covers, with its row in each.

    The row is the highest point of the top side in that column.
    """
    first = top_side[:, 0].min()
    columns = np.arange(first, top_side[:, 0].max() + 1)
    rows = np.full(len(columns), np.iinfo(np.int64).max)
    np.minimum.at(rows, top_side[:, 0] - first, top_side[:, 1])
    return columns, rows


def step_lengths(pixels, scale_x=1.0, scale_y=1.0):
    """Length of each step between consecutive (x, y) pixels of a path, in the
    units of the two scales."""
    moves = np.diff(pixels, axis=0)
    return np.hypot(moves[:, 0] * scale_x, moves[:, 1] * scale_y)


def path_length(path, start, stop):
    """Length of the part of a path that lies in columns start to stop.

    A step between two columns counts half towards each.
    """
    pixels, steps = path
    inside = ((pixels[:, 0] >= start) & (pixels[:, 0] < stop)).astype(float)
    return float(np.sum(steps * (inside[:-1] + inside[1:]) / 2.0))


def suggest_centre_x(columns, rows):
    """Column where a top surface, as given by top_surface, dips deepest."""
    deepest = columns[rows == rows.max()]
    return float(deepest[len(deepest) // 2])


def surface_point(columns, rows, centre_x):
    """The point of a top surface at centre_x, or at the nearest covered column."""
    index = int(np.abs(columns - centre_x).argmin())
    return float(centre_x), float(rows[index])


INTERCONDYLAR_WIDTH_FRACTION = 0.25
REGION_NAMES = ("lateral", "intercondylar", "medial")


def region_columns(image_width, centre_x, knee_side, limits=None):
    """Column range (start, stop) of each region for a knee side of "right" or "left".

    The intercondylar region is centred on centre_x and is a fixed share of the
    width of the ROI box `limits` (left, right, top, bottom), or of the image width
    when there is no ROI. On a right knee the lateral condyle is on the left of the
    image.
    """
    base_width = image_width if limits is None else limits[1] - limits[0]
    half_width = INTERCONDYLAR_WIDTH_FRACTION * base_width / 2.0
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


def _measure_columns(gray, binary, path, start, stop, scale_x, scale_y):
    values = dict.fromkeys(MEASURES, np.nan)
    region = binary[:, start:stop]
    if not region.any():
        return values
    values["area"] = float(region.sum()) * scale_x * scale_y
    values["length"] = path_length(path, start, stop)
    if values["length"] > 0:
        values["thickness"] = values["area"] / values["length"]
    if gray is not None:
        pixels = gray[:, start:stop][region]
        values["echo_mean"] = float(np.mean(pixels))
        values["echo_sd"] = float(np.std(pixels))
    return values


def measure(
    image,
    mask,
    px_per_mm_x,
    px_per_mm_y,
    centre_x=None,
    knee_side="right",
    limits=None,
):
    """Measure a mask as a whole and split into lateral, intercondylar and medial.

    Only the largest connected piece of the mask is measured.
    Returns {"unit", "centre_x", "regions"}; "regions" maps "whole" and each region
    name to its area, cartilage-bone interface length, thickness (area / length),
    and the mean and standard deviation of the grey levels inside it. Lengths are in
    mm when both resolutions are known and in pixels otherwise. centre_x of None
    uses the suggested centre. limits is the ROI box (left, right, top, bottom) the
    mask is confined to, if any; the intercondylar region is sized from it. Values
    that cannot be computed are NaN.
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
    binary = largest_piece(mask)
    if not binary.any():
        return result
    gray = None
    if image is not None and image.shape[:2] == binary.shape[:2]:
        gray = to_grayscale(image)
    top_side, bottom_side = outline_sides(binary)
    path = bottom_side, step_lengths(bottom_side, scale_x, scale_y)
    width = binary.shape[1]
    if centre_x is None:
        centre_x = suggest_centre_x(*top_surface(top_side))
    result["centre_x"] = centre_x
    spans = {
        "whole": (0, width),
        **region_columns(width, centre_x, knee_side, limits),
    }
    for name, (start, stop) in spans.items():
        result["regions"][name] = _measure_columns(
            gray, binary, path, start, stop, scale_x, scale_y
        )
    return result


MEASUREMENTS_FILE_NAME = "measurements.csv"
MEASUREMENT_COLUMNS = [
    "file",
    "frame",
    "knee_side",
    "centre_x",
    "centre_set_by",
    "limit_left",
    "limit_right",
    "limit_top",
    "limit_bottom",
    "rotation",
    "pixels_per_mm_x",
    "pixels_per_mm_y",
    "unit",
] + [
    f"{region}_{measure_name}"
    for region in ("whole",) + REGION_NAMES
    for measure_name in MEASURES
]


class MeasurementLog:
    """A measurements file of one output folder, one row per image or video frame.

    `frame` is the frame number for a video and "" for an image.
    """

    def __init__(self, output_dir, file_name=MEASUREMENTS_FILE_NAME):
        self.path = Path(output_dir) / file_name
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

    def put(self, file_name, frame, row):
        """Store a row exactly as given, or remove the row when given None."""
        key = (str(file_name), str(frame))
        if row is None:
            self.remove(file_name, frame)
        elif self._rows.get(key) != row:
            self._rows[key] = dict(row)
            self._dirty = True

    def keys(self):
        """(file, frame) of every row."""
        return list(self._rows)

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
        rotation=0.0,
    ):
        """Measure a mask and store it as the row of its image or frame.

        centre_x of None means the suggested centre. With prefer_saved, a knee side
        and a user-placed centre already stored for the row win over the arguments.
        limits is the ROI box (left, right, top, bottom) the mask was confined to,
        if any. rotation is the angle, in degrees clockwise, the image and mask
        were turned by before being measured.
        """
        key = (str(file_name), str(frame))
        saved = self._rows.get(key)
        if prefer_saved and saved is not None:
            if saved.get("knee_side") in ("right", "left"):
                knee_side = saved["knee_side"]
            # A centre placed on a view at another rotation is not kept.
            same_view = self.rotation(file_name, frame) == round(float(rotation), 1)
            if saved.get("centre_set_by") == "user" and same_view:
                try:
                    centre_x = float(saved["centre_x"])
                except (TypeError, ValueError):
                    pass
        result = measure(
            image, mask, px_per_mm_x, px_per_mm_y, centre_x, knee_side, limits
        )
        row = {
            "file": key[0],
            "frame": key[1],
            "knee_side": knee_side,
            "centre_x": (
                "" if result["centre_x"] is None else f"{result['centre_x']:.1f}"
            ),
            "centre_set_by": "auto" if centre_x is None else "user",
            "limit_left": "" if limits is None else str(limits[0]),
            "limit_right": "" if limits is None else str(limits[1]),
            "limit_top": "" if limits is None else str(limits[2]),
            "limit_bottom": "" if limits is None else str(limits[3]),
            "rotation": f"{rotation:.1f}",
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

    def limits(self, file_name, frame):
        """The ROI box (left, right, top, bottom) recorded in a row, or None."""
        row = self.get(file_name, frame)
        if row is None:
            return None
        try:
            return tuple(
                int(row[column])
                for column in ("limit_left", "limit_right", "limit_top", "limit_bottom")
            )
        except (KeyError, TypeError, ValueError):
            return None

    def rotation(self, file_name, frame):
        """The angle a row was measured at; 0 for a row that records none."""
        row = self.get(file_name, frame)
        try:
            return float(row["rotation"])
        except (KeyError, TypeError, ValueError):
            return 0.0

    def matches_inputs(
        self, file_name, frame, centre_x, knee_side, limits, rotation=0.0
    ):
        """True when a row exists and records this centre, knee side, ROI box and
        rotation.

        centre_x of None means the suggested centre.
        """
        row = self.get(file_name, frame)
        if row is None or row.get("knee_side") != knee_side:
            return False
        if self.rotation(file_name, frame) != round(float(rotation), 1):
            return False
        if centre_x is None:
            if row.get("centre_set_by") != "auto":
                return False
        elif row.get("centre_set_by") != "user" or row.get("centre_x") != (
            f"{centre_x:.1f}"
        ):
            return False
        recorded = self.limits(file_name, frame)
        return recorded == (None if limits is None else tuple(limits))

    def clear_limits(self, file_names):
        """Blank the recorded ROI box in every row of the named files."""
        for (file_name, _), row in self._rows.items():
            if file_name not in file_names:
                continue
            for column in ("limit_left", "limit_right", "limit_top", "limit_bottom"):
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
                    writer.writerow(self._rows[key])
            os.replace(partial, self.path)
        finally:
            partial.unlink(missing_ok=True)
        self._dirty = False
