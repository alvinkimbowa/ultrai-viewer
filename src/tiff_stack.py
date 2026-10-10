"""
TIFF files that hold several images in one, read frame by frame.
"""

from pathlib import Path

import cv2
import numpy as np
import tifffile

TIFF_EXTENSIONS = (".tif", ".tiff")


def is_tiff(path):
    return str(path).lower().endswith(TIFF_EXTENSIONS)


def _frame_layout(series):
    """(number of frames, shape of one frame) of a TIFF series.

    Every axis ahead of the image rows counts as frames, except a leading
    samples axis, which is how a colour image stored plane by plane appears.
    """
    axes, shape = series.axes, series.shape
    if "Y" not in axes:
        return 1, shape
    rows = axes.index("Y")
    if "S" in axes[:rows]:
        return 1, shape
    return int(np.prod(shape[:rows], dtype=np.int64)), shape[rows:]


def tiff_frame_count(path):
    """Number of images a TIFF file holds; 1 for a file that cannot be read."""
    try:
        with tifffile.TiffFile(path) as tif:
            return _frame_layout(tif.series[0])[0]
    except Exception:
        return 1


def _read_with_opencv(path, index):
    """One image of a TIFF file read by OpenCV, colours put in RGB order."""
    success, frames = cv2.imreadmulti(
        str(path), start=index, count=1, flags=cv2.IMREAD_UNCHANGED
    )
    if not success or not frames:
        raise ValueError(f"Unsupported image format: {path}")
    frame = frames[0]
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
    return frame


def read_tiff_frame(path, index):
    """One image of a TIFF file, as stored (colour images are RGB)."""
    try:
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            count, frame_shape = _frame_layout(series)
            if count <= 1:
                return series.asarray()
            if len(series.pages) == count:
                return tif.asarray(key=index, series=0)
            return series.asarray().reshape((count, *frame_shape))[index]
    except Exception:
        # tifffile cannot decode some compressions (LZW among them) without an
        # extra package; OpenCV reads those.
        return _read_with_opencv(path, index)


class TiffStackCapture:
    """A TIFF file holding several images, read the way cv2.VideoCapture reads a video.

    Colour frames come back in BGR order, as cv2.VideoCapture gives them, so the
    same code can handle frames from either.
    """

    def __init__(self, path):
        self._path = str(path)
        self._count = tiff_frame_count(path)
        self._position = 0

    def isOpened(self):
        return self._count > 1

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._count)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._position)
        return 0.0

    def set(self, prop, value):
        if prop != cv2.CAP_PROP_POS_FRAMES:
            return False
        self._position = int(value)
        return True

    def grab(self):
        if not 0 <= self._position < self._count:
            return False
        self._position += 1
        return True

    def read(self):
        if not 0 <= self._position < self._count:
            return False, None
        try:
            frame = read_tiff_frame(self._path, self._position)
        except Exception:
            return False, None
        self._position += 1
        if frame.ndim == 3 and frame.shape[2] >= 3:
            frame = np.ascontiguousarray(frame[:, :, 2::-1])
        return True, frame

    def release(self):
        self._count = 0


def open_capture(path):
    """Frame reader for a video file or for a TIFF file holding several images."""
    if is_tiff(path):
        return TiffStackCapture(path)
    return cv2.VideoCapture(str(path))
