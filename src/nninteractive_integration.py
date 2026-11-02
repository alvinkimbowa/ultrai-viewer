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
import numpy as np

from .helper_process import HelperProcess
from .sam_integration import read_mask

NNINTERACTIVE_MODEL_NAME = "nninteractive"
NNINTERACTIVE_MODEL_LABEL = "nnInteractive"
WORKER = Path(__file__).with_name("nninteractive_worker.py")
LOG_FILE = Path(__file__).resolve().parent.parent / "nninteractive_worker.log"
# Pixels across a scribble, as nnInteractive prefers them (its model folder's
# inference_session_class.json).
SCRIBBLE_THICKNESS = 2
# Frames of a video given to nnInteractive at a time. It looks at a patch of 192
# slices around a prompt and zooms out at most 4 times, so a prompt reaches no
# further than this.
VOLUME_FRAMES = 768


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
    exclude_boxes = True
    exclude_outlines = True
    scribbles = True
    ordered_prompts = True
    volume_frames = True

    def __init__(self):
        super().__init__(
            NNINTERACTIVE_MODEL_LABEL,
            [nninteractive_python(), "-u", WORKER, nninteractive_model()],
            None,
            LOG_FILE,
            "ultrai_nninteractive_",
        )
        self._volume_key = None

    def segment_prompts(self, image, key, prompts):
        """The mask that prompts select on an 8-bit RGB image. prompts are
        (id, prompt) in the order given, each id kept by its prompt for as long as
        it stands; a prompt is ("point", x, y, include), ("box", left, top, right,
        bottom), ("exclude_box", left, top, right, bottom), ("mask", outline),
        ("exclude_mask", outline) or ("scribble", ((x, y), ...), include), in
        pixels of the image. The helper keeps the image while key stays the
        same, and adds only the prompts it has not had yet."""
        folder = self.scratch_dir("prompts")
        image_path = folder / "image.png"
        out = folder / "mask.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        sent = self._encode_prompts(prompts, image.shape[:2], folder)
        self.request("prompts", image=str(image_path), key=key, prompts=sent, out=str(out))
        return read_mask(out)

    def segment_volume(self, frames, key, target_object, prompts, shape, offset):
        """{frame: mask} of the frames whose mask for target_object changed.
        frames() gives the 8-bit (frames, rows, columns) of the video and is
        called only when key changes. offset is the first frame it gives, shape
        the (rows, columns) of a frame. prompts are as for segment_prompts, each
        with the frame it lies on as a third item."""
        folder = self.scratch_dir("prompts")
        volume = self._scratch_volume()
        if key != self._volume_key:
            self._volume_key = None
            np.save(volume, frames())
            self._volume_key = key
        sent = self._encode_prompts([entry[:2] for entry in prompts], shape, folder)
        for encoded, entry in zip(sent, prompts):
            encoded["z"] = entry[2] - offset
        out = folder / "masks.npz"
        self.request(
            "prompts", volume=str(volume), key=key, object=target_object, prompts=sent, out=str(out)
        )
        with np.load(out) as changed:
            return {
                int(frame) + offset: mask
                for frame, mask in zip(changed["frames"], changed["masks"])
            }

    def _scratch_volume(self):
        return Path(self._scratch.name) / "volume.npy"

    def _encode_prompts(self, prompts, shape, folder):
        """The prompts as the helper takes them, with the files they need in folder."""
        sent = []
        for prompt_id, prompt in prompts:
            kind = prompt[0]
            if kind == "point":
                x, y, include = prompt[1:]
                sent.append(
                    {"id": prompt_id, "kind": "point", "include": bool(include),
                     "x": float(x), "y": float(y)}
                )
            elif kind in ("box", "exclude_box"):
                sent.append(
                    {"id": prompt_id, "kind": "box", "include": kind == "box",
                     "box": [float(v) for v in prompt[1:]]}
                )
            elif kind == "scribble":
                path = folder / f"scribble_{prompt_id}.png"
                stroke = np.zeros(shape, dtype=np.uint8)
                corners = np.array(prompt[1], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(stroke, [corners], False, 255, SCRIBBLE_THICKNESS)
                cv2.imwrite(str(path), stroke)
                sent.append(
                    {"id": prompt_id, "kind": "scribble", "include": bool(prompt[2]),
                     "mask": str(path)}
                )
            else:
                path = folder / f"outline_{prompt_id}.png"
                cv2.imwrite(str(path), prompt[1].astype("uint8") * 255)
                sent.append(
                    {"id": prompt_id, "kind": "lasso", "include": kind == "mask",
                     "mask": str(path)}
                )
        return sent
