"""
UltraSam for the viewer, run in a helper process (ultrasam_worker.py).

UltraSam needs PyTorch, which the viewer does not ship, so the helper runs with
the Python of a separate environment that has PyTorch and Meta's
segment_anything package: ULTRAI_ULTRASAM_PYTHON, else the .venv of the ultrasam
project. ULTRAI_ULTRASAM_REPO is that project, by default
~/UltrAi/projects/ultrasam, and ULTRAI_ULTRASAM_CHECKPOINT the released UltraSam
checkpoint, by default weights/UltraSam.pth in it.
"""

import os
import sys
from pathlib import Path

import cv2

from .helper_process import HelperProcess
from .sam_integration import prompt_mask_fields, read_mask

ULTRASAM_MODEL_NAME = "ultrasam"
ULTRASAM_MODEL_LABEL = "UltraSam"
WORKER = Path(__file__).with_name("ultrasam_worker.py")
LOG_FILE = Path(__file__).resolve().parent.parent / "ultrasam_worker.log"


def ultrasam_repo():
    return Path(
        os.environ.get("ULTRAI_ULTRASAM_REPO") or Path.home() / "UltrAi" / "projects" / "ultrasam"
    )


def ultrasam_python():
    configured = os.environ.get("ULTRAI_ULTRASAM_PYTHON")
    if configured:
        return Path(configured)
    venv = ultrasam_repo() / ".venv"
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def ultrasam_checkpoint():
    configured = os.environ.get("ULTRAI_ULTRASAM_CHECKPOINT")
    if configured:
        return Path(configured)
    return ultrasam_repo() / "weights" / "UltraSam.pth"


def ultrasam_available():
    if getattr(sys, "frozen", False):
        return False
    return WORKER.is_file() and ultrasam_python().is_file() and ultrasam_checkpoint().is_file()


class UltraSamPrompter(HelperProcess):
    model_name = ULTRASAM_MODEL_NAME

    def __init__(self):
        super().__init__(
            ULTRASAM_MODEL_LABEL,
            [ultrasam_python(), "-u", WORKER, ultrasam_checkpoint()],
            None,
            LOG_FILE,
            "ultrai_ultrasam_",
        )

    def segment_points(self, image, key, points, labels, mask=None):
        """The mask points select on an 8-bit RGB image: (x, y) pixels of it,
        labelled 1 to include, 0 to exclude, and 2 and 3 for the corners of a box.
        A mask of the image's size, when given, is a prompt as well. The helper
        keeps the encoding of the image while key stays the same."""
        folder = self.scratch_dir("points")
        image_path = folder / "image.png"
        out = folder / "mask.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        self.request(
            "points",
            image=str(image_path),
            key=key,
            points=[[float(x), float(y)] for x, y in points],
            labels=[int(label) for label in labels],
            out=str(out),
            **prompt_mask_fields(folder, mask),
        )
        return read_mask(out)
