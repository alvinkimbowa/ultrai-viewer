"""
SAM 3.1 for the viewer, run in a helper process (sam_worker.py).

SAM 3.1 needs PyTorch, which the viewer does not ship, so the helper runs with
the Python of a separate environment that has PyTorch and Meta's sam3 package:
ULTRAI_SAM_PYTHON, else the .venv of the sam3 checkout. ULTRAI_SAM_REPO is that
checkout, by default ~/UltrAi/projects/sam3.
"""

import json
import os
import sys
from pathlib import Path

import cv2

from .helper_process import HelperCanceled, HelperProcess
from .instances import clean_class

SAM_MODEL_NAME = "sam3.1"
SAM_MODEL_LABEL = "SAM 3.1"
WORKER = Path(__file__).with_name("sam_worker.py")
LOG_FILE = Path(__file__).resolve().parent.parent / "sam_worker.log"


def sam_repo():
    return Path(os.environ.get("ULTRAI_SAM_REPO") or Path.home() / "UltrAi" / "projects" / "sam3")


def sam_python():
    configured = os.environ.get("ULTRAI_SAM_PYTHON")
    if configured:
        return Path(configured)
    venv = sam_repo() / ".venv"
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def sam_available():
    if getattr(sys, "frozen", False):
        return False
    return WORKER.is_file() and sam_python().is_file() and (sam_repo() / "sam3").is_dir()


PROMPTS_FILE_NAME = "sam_prompts.json"


def _read_prompts_file(folder):
    try:
        return json.loads((Path(folder) / PROMPTS_FILE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _prompts_from(frames, masks):
    """{frame: None, or the set of (label, number) of the masks that prompt}
    from the lists kept in a prompts file."""
    prompts = {}
    for frame in frames:
        keys = masks.get(str(int(frame)))
        prompts[int(frame)] = (
            None if keys is None else {(clean_class(label), int(number)) for label, number in keys}
        )
    return prompts


def _prompts_lists(prompts):
    """The frames and masks lists of a prompts file, from {frame: keys}."""
    frames = sorted(int(frame) for frame in prompts)
    masks = {
        str(frame): sorted([label, int(number)] for label, number in prompts[frame])
        for frame in frames
        if prompts[frame] is not None
    }
    return frames, masks


def load_prompts(folder):
    """The masks of a video that SAM propagates from, as kept in its results
    folder: {frame: None when every mask on the frame prompts, else the set of
    (label, number) of those that do}. None while none are kept.

    The other masks there are what SAM found, which must not prompt it again.
    """
    data = _read_prompts_file(folder)
    if data is None:
        return None
    return _prompts_from(data.get("frames", []), data.get("masks", {}))


def load_propagated(folder):
    """The prompts, as load_prompts gives them, of the last propagation of a
    video that was saved in full, or None when none was."""
    data = _read_prompts_file(folder)
    if data is None or "propagated" not in data:
        return None
    return _prompts_from(data["propagated"], data.get("propagated_masks", {}))


def save_prompts(folder, prompts, propagated=None):
    """Keep a video's prompts, and, given propagated, the prompts of a
    propagation just saved in full; otherwise those kept stay."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    frames, masks = _prompts_lists(prompts)
    data = {"frames": frames, "masks": masks}
    if propagated is None:
        propagated = load_propagated(folder)
    if propagated is not None:
        data["propagated"], data["propagated_masks"] = _prompts_lists(propagated)
    (folder / PROMPTS_FILE_NAME).write_text(json.dumps(data), encoding="utf-8")


SamCanceled = HelperCanceled


def read_mask(path):
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"SAM returned no readable mask: {path}")
    return raw > 127


def prompt_mask_fields(folder, mask):
    """The fields of a points request that carry a prompt mask, saved as a PNG
    in folder; none without one."""
    if mask is None:
        return {}
    path = Path(folder) / "prompt.png"
    cv2.imwrite(str(path), mask.astype("uint8") * 255)
    return {"mask": str(path)}


class SamTracker(HelperProcess):
    def __init__(self):
        super().__init__(
            SAM_MODEL_LABEL,
            [sam_python(), "-u", WORKER],
            sam_repo(),
            LOG_FILE,
            "ultrai_sam_",
        )

    def open(self, video):
        return self.request("open", video=str(video))

    def open_image(self, image, key):
        """Prompt an 8-bit RGB image as a video of one frame; the helper keeps it
        while key stays the same."""
        path = self.scratch_dir("image") / "image.png"
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return self.request("image", image=str(path), key=key)

    def segment_image_points(self, image, key, points, labels, mask=None):
        """The mask points select on an image, as segment_points gives it for a
        frame; points and mask are in pixels of the image."""
        self.open_image(image, key)
        return self._points(0, points, labels, mask)

    def find_image_text(self, image, key, text):
        """A mask for each match of a phrase on an image."""
        self.open_image(image, key)
        return self._text(0, text)

    def segment_points(self, video, frame, points, labels, mask=None):
        """The mask points select on a frame: (x, y) pixels of the stored frame,
        labelled 1 to include, 0 to exclude, and 2 and 3 for the corners of a box.
        A mask of the frame's size, when given, is prompted first and the points
        refine it."""
        self.open(video)
        return self._points(frame, points, labels, mask)

    def find_text(self, video, frame, text):
        """A mask for each match of a phrase on a frame."""
        self.open(video)
        return self._text(frame, text)

    def _points(self, frame, points, labels, mask):
        folder = self.scratch_dir("points")
        out = folder / "mask.png"
        self.request(
            "points",
            frame=int(frame),
            points=[[float(x), float(y)] for x, y in points],
            labels=[int(label) for label in labels],
            out=str(out),
            **prompt_mask_fields(folder, mask),
        )
        return read_mask(out)

    def _text(self, frame, text):
        reply = self.request(
            "text", frame=int(frame), text=text, out_dir=str(self.scratch_dir("text"))
        )
        return [read_mask(path) for path in reply["masks"]]

    def propagate(self, video, prompts, start, direction, on_progress=None):
        """Track objects through a video.

        prompts maps each object's key to {frame: mask}. on_progress gets
        (done, total, index of the object in prompts, frame). Returns the frames
        tracked and, for each of them, {key: mask} of the objects found there.
        """
        self.open(video)
        prompt_dir = self.scratch_dir("prompts")
        out_dir = self.scratch_dir("propagated")
        keys = list(prompts)
        objects = []
        for index, key in enumerate(keys):
            paths = {}
            for frame, mask in prompts[key].items():
                path = prompt_dir / f"{index}_{frame}.png"
                cv2.imwrite(str(path), mask.astype("uint8") * 255)
                paths[str(frame)] = str(path)
            objects.append({"prompts": paths})
        reply = self.request(
            "propagate",
            on_progress=on_progress,
            objects=objects,
            start=int(start),
            direction=direction,
            out_dir=str(out_dir),
        )
        results = {frame: {} for frame in reply["frames"]}
        for frame, indices in reply["found"].items():
            for index in indices:
                results[int(frame)][keys[index]] = read_mask(out_dir / f"{frame}_{index}.png")
        return results
