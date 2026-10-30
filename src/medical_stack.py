"""
DICOM and NIfTI files, which hold one image, a video or a 3D volume.

A file of one frame is an image. A file of several frames is gone through frame
by frame as a video is, whether they are moments in time or slices of a volume.
This module is also loaded by file from the SAM 3.1 helper, so it imports nothing
from the viewer.
"""

import cv2
import numpy as np

DICOM_EXTENSIONS = (".dcm", ".dicom")
NIFTI_EXTENSIONS = (".nii", ".nii.gz")
MEDICAL_EXTENSIONS = DICOM_EXTENSIONS + NIFTI_EXTENSIONS


def is_medical(path):
    return str(path).lower().endswith(MEDICAL_EXTENSIONS)


def _is_dicom(path):
    return str(path).lower().endswith(DICOM_EXTENSIONS)


def _dicom_header(path):
    import pydicom

    header = pydicom.dcmread(str(path), stop_before_pixels=True)
    if "Rows" not in header:
        raise ValueError(f"No image in {path}")
    return header


def medical_frame_count(path):
    """Number of frames in a file, read from its header. Raises on a file that
    cannot be read."""
    if _is_dicom(path):
        return int(_dicom_header(path).get("NumberOfFrames") or 1)
    import nibabel

    shape = nibabel.load(str(path)).shape
    if len(shape) < 2:
        raise ValueError(f"No image in {path}")
    return int(np.prod(shape[2:], dtype=np.int64))


def medical_frame_rate(path):
    """Frames per second a DICOM file gives; 0 when it gives none."""
    if not _is_dicom(path):
        return 0.0
    header = _dicom_header(path)
    frame_time = float(header.get("FrameTime") or 0.0)
    if frame_time > 0:
        return 1000.0 / frame_time
    return float(header.get("CineRate") or 0.0)


def _to_uint8(frames):
    """8-bit frames, scaled by the range of the whole file so that brightness
    does not change from one frame to the next."""
    if frames.dtype == np.uint8:
        return frames
    frames = np.nan_to_num(frames.astype(np.float32))
    low, high = float(frames.min()), float(frames.max())
    if high <= low:
        return np.zeros(frames.shape, dtype=np.uint8)
    return ((frames - low) / (high - low) * 255.0).astype(np.uint8)


def _read_dicom(path):
    import pydicom
    from pydicom.pixels import pixel_array

    dataset = pydicom.dcmread(str(path))
    pixels = pixel_array(dataset)
    if int(dataset.get("NumberOfFrames") or 1) == 1:
        pixels = pixels[None]
    frames = _to_uint8(pixels)
    # MONOCHROME1 stores its brightest pixels as the lowest values.
    if dataset.get("PhotometricInterpretation") == "MONOCHROME1":
        frames = 255 - frames
    return frames


def _read_nifti(path):
    import nibabel

    volume = np.asanyarray(nibabel.load(str(path)).dataobj)
    columns, rows = volume.shape[:2]
    # NIfTI's first axis runs across the image, and its last axes vary slowest.
    stacked = volume.reshape((columns, rows, -1), order="F")
    return _to_uint8(stacked.transpose(2, 1, 0))


def read_medical_frames(path):
    """All frames of a file as 8-bit (frames, rows, columns), with a last axis of
    three for RGB."""
    return _read_dicom(path) if _is_dicom(path) else _read_nifti(path)


class MedicalStackCapture:
    """A DICOM or NIfTI file, read the way cv2.VideoCapture reads a video.

    Colour frames come back in BGR order, as cv2.VideoCapture gives them. The
    frames are read from the file on the first read.
    """

    def __init__(self, path):
        self._path = str(path)
        self._frames = None
        self._position = 0
        try:
            self._count = medical_frame_count(path)
            self._rate = medical_frame_rate(path)
        except Exception:
            self._count = 0
            self._rate = 0.0

    def isOpened(self):
        return self._count > 0

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._count)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._position)
        if prop == cv2.CAP_PROP_FPS:
            return self._rate
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
        if self._frames is None:
            try:
                self._frames = read_medical_frames(self._path)
            except Exception:
                return False, None
        frame = self._frames[self._position]
        self._position += 1
        if frame.ndim == 3:
            frame = np.ascontiguousarray(frame[:, :, ::-1])
        return True, frame.copy()

    def release(self):
        self._count = 0
        self._frames = None
