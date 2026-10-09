"""
Measurements computed from a segmented cartilage mask.
"""

import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def binarize_mask(mask):
    return (mask > 0).astype(np.uint8)


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


def _bottom_side(points):
    """The stretch of a closed outline that runs along the bottom of its shape.

    points are the (x, y) pixels of the outline in tracing order. The stretch goes
    from the left end to the right end of the shape; where an end is a vertical
    edge, the end is its lowest pixel.
    """
    xs, ys = points[:, 0], points[:, 1]
    left = np.nonzero(xs == xs.min())[0]
    right = np.nonzero(xs == xs.max())[0]
    start = int(left[ys[left].argmax()])
    end = int(right[ys[right].argmax()])
    # The sign of the enclosed area gives the tracing direction: positive runs
    # left end -> top -> right end -> bottom on screen, where y points down.
    area = np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys)
    count = len(points)
    if area > 0:
        steps = (start - end) % count
        order = (end + np.arange(steps + 1)) % count
        return points[order][::-1]
    steps = (end - start) % count
    order = (start + np.arange(steps + 1)) % count
    return points[order]


def _shortest_path(points, scale_x, scale_y):
    """Shortest route from the first to the last of a set of pixels.

    The route steps between pixels of the set that touch, corners included, and a
    step is as long as the pixel is wide (scale_x), tall (scale_y) or diagonal.
    Returns the (x, y) pixels of the route in order and the length of each step.
    """
    index = {}
    for x, y in points:
        index.setdefault((int(x), int(y)), len(index))
    start = index[(int(points[0][0]), int(points[0][1]))]
    end = index[(int(points[-1][0]), int(points[-1][1]))]
    pixels = np.array(list(index), dtype=int)
    if start == end:
        return pixels[[start]], np.zeros(0)
    rows, cols, weights = [], [], []
    for (x, y), i in index.items():
        for dx, dy in ((1, -1), (1, 0), (1, 1), (0, 1)):
            j = index.get((x + dx, y + dy))
            if j is not None:
                rows.append(i)
                cols.append(j)
                weights.append(np.hypot(dx * scale_x, dy * scale_y))
    graph = csr_matrix((weights, (rows, cols)), shape=(len(index), len(index)))
    _, before = dijkstra(graph, directed=False, indices=start, return_predecessors=True)
    route = [end]
    while route[-1] != start:
        route.append(int(before[route[-1]]))
    route = pixels[route[::-1]]
    moves = np.diff(route, axis=0)
    return route, np.hypot(moves[:, 0] * scale_x, moves[:, 1] * scale_y)


def bottom_surface_paths(mask, scale_x=1.0, scale_y=1.0):
    """The cartilage-bone interface of a mask: the bottom side of its outline.

    One path per connected piece of the mask, each the shortest route along the
    bottom side of that piece's outline from its left end to its right end.
    Returns a list of (pixels, steps): the (x, y) pixels of a path in order and
    the length of each step between them, in the units of the two scales.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    paths = []
    for contour in contours:
        points = contour[:, 0, :].astype(np.int64)
        paths.append(_shortest_path(_bottom_side(points), scale_x, scale_y))
    return paths


def path_length(paths, start, stop):
    """Length of the parts of paths that lie in columns start to stop.

    A step between two columns counts half towards each.
    """
    total = 0.0
    for pixels, steps in paths:
        inside = ((pixels[:, 0] >= start) & (pixels[:, 0] < stop)).astype(float)
        total += float(np.sum(steps * (inside[:-1] + inside[1:]) / 2.0))
    return total


def suggest_centre_x(mask):
    """Column where the cartilage's top surface dips deepest into the image."""
    columns, top, _ = cartilage_edges(mask)
    if len(columns) == 0:
        return None
    deepest = columns[top == top.max()]
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


def _measure_columns(gray, binary, paths, start, stop, scale_x, scale_y):
    values = dict.fromkeys(MEASURES, np.nan)
    region = binary[:, start:stop]
    if not region.any():
        return values
    values["area"] = float(region.sum()) * scale_x * scale_y
    values["length"] = path_length(paths, start, stop)
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
    binary = np.asarray(mask) > 0
    if not binary.any():
        return result
    gray = None
    if image is not None and image.shape[:2] == binary.shape[:2]:
        gray = to_grayscale(image)
    paths = bottom_surface_paths(binary, scale_x, scale_y)
    width = binary.shape[1]
    if centre_x is None:
        centre_x = suggest_centre_x(binary)
    result["centre_x"] = centre_x
    spans = {
        "whole": (0, width),
        **region_columns(width, centre_x, knee_side, limits),
    }
    for name, (start, stop) in spans.items():
        result["regions"][name] = _measure_columns(
            gray, binary, paths, start, stop, scale_x, scale_y
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

        with open(self.path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MEASUREMENT_COLUMNS)
            writer.writeheader()
            for key in sorted(self._rows, key=order):
                writer.writerow(self._rows[key])
        self._dirty = False
