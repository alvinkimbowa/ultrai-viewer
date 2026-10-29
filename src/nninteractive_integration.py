"""
nnInteractive for the viewer, run in a helper process (nninteractive_worker.py).

nnInteractive needs PyTorch, which the viewer does not ship, so the helper runs
with the Python of a separate environment that has PyTorch and the nnInteractive
package: ULTRAI_NNINTERACTIVE_PYTHON, else the .venv of the nnInteractive
project. ULTRAI_NNINTERACTIVE_REPO is that project, by default
~/UltrAi/projects/nnInteractive, and ULTRAI_NNINTERACTIVE_MODEL the folder of
the trained model, by default weights/models/nnInteractive_v1.0 in it.
"""

import os
import sys
from pathlib import Path

import cv2

from .helper_process import HelperProcess
from .sam_integration import prompt_mask_fields, read_mask

NNINTERACTIVE_MODEL_NAME = "nninteractive"
NNINTERACTIVE_MODEL_LABEL = "nnInteractive"
WORKER = Path(__file__).with_name("nninteractive_worker.py")
LOG_FILE = Path(__file__).resolve().parent.parent / "nninteractive_worker.log"


def nninteractive_repo():
    return Path(
        os.environ.get("ULTRAI_NNINTERACTIVE_REPO")
        or Path.home() / "UltrAi" / "projects" / "nnInteractive"
    )


def nninteractive_python():
    configured = os.environ.get("ULTRAI_NNINTERACTIVE_PYTHON")
    if configured:
        return Path(configured)
    venv = nninteractive_repo() / ".venv"
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def nninteractive_model():
    configured = os.environ.get("ULTRAI_NNINTERACTIVE_MODEL")
    if configured:
        return Path(configured)
    return nninteractive_repo() / "weights" / "models" / "nnInteractive_v1.0"


def nninteractive_available():
    if getattr(sys, "frozen", False):
        return False
    return (
        WORKER.is_file()
        and nninteractive_python().is_file()
        and (nninteractive_model() / "plans.json").is_file()
    )


class NnInteractivePrompter(HelperProcess):
    model_name = NNINTERACTIVE_MODEL_NAME

    def __init__(self):
        super().__init__(
            NNINTERACTIVE_MODEL_LABEL,
            [nninteractive_python(), "-u", WORKER, nninteractive_model()],
            None,
            LOG_FILE,
            "ultrai_nninteractive_",
        )

    def segment_points(self, image, key, points, labels, mask=None):
        """The mask points select on an 8-bit RGB image: (x, y) pixels of it,
        labelled 1 to include, 0 to exclude, and 2 and 3 for the corners of a box.
        A mask of the image's size, when given, is an outline of the structure.
        The helper keeps the image while key stays the same."""
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
