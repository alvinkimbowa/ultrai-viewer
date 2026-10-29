"""
nnInteractive helper process for UltrAi Viewer.

nnInteractive is MIC-DKFZ's promptable nnU-Net, trained on 3D volumes. A 2D
image is given to it as a volume of DEPTH copies of the image, every prompt lies
on the middle slice, and the mask is that slice of the result. The viewer has no PyTorch, so it runs this script with the Python of an
environment that has PyTorch and the nnInteractive package, and talks to it
through the pipes: one JSON request per line on stdin, one JSON reply per line
on stdout. Every reply carries the "id" of its request. Anything else the
libraries print goes to stderr.

Requests:
    points  {"image", "key", "points", "labels", "out", "mask"?}  the mask that
            include (1) and exclude (0) points, and a box given as corner points
            labelled 2 and 3, select on an 8-bit RGB PNG; points are its pixels.
            "mask", a PNG of the image's size, is a closed outline of the
            structure (nnInteractive's lasso). The image is kept while requests
            carry the same key.
    quit

Each request answers its prompts from scratch: the session's interactions are
cleared, the outline and box are added, then the points, and nnInteractive
predicts once, after the last of them.
"""

import json
import os
import sys
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The viewer's modules sit next to this script; generic names among them
# (labels, measurements) must not shadow modules PyTorch and its dependencies import.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != SCRIPT_DIR]

PROTOCOL = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
os.dup2(2, 1)
sys.stdout = sys.stderr

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# Copies of the image stacked into a volume. A volume of a single slice segments
# worse: on breast ultrasound (BUSI) the mean Dice of a box prompt is 0.82 with
# one slice and 0.85 with nine, of one point 0.25 and 0.47, of an outline 0.87
# and 0.89.
DEPTH = 9
MIDDLE = DEPTH // 2


def send(message):
    PROTOCOL.write(json.dumps(message) + "\n")
    PROTOCOL.flush()


class NnInteractive:
    def __init__(self, model_folder):
        from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.session = nnInteractiveInferenceSession(
            device=device,
            use_torch_compile=False,
            verbose=False,
            torch_n_threads=os.cpu_count() or 4,
            do_autozoom=True,
            enable_undo=False,
        )
        self.session.initialize_from_trained_model_folder(str(model_folder))
        self._key = None
        self._target = None

    def _set_image(self, image, key):
        if key is not None and key == self._key and self._target is not None:
            return
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        self.session.set_image(np.repeat(grey[None, None], DEPTH, axis=1))
        self._target = torch.zeros((DEPTH, *grey.shape), dtype=torch.uint8)
        self.session.set_target_buffer(self._target)
        self._key = key

    @torch.inference_mode()
    def points(self, image, key, points, labels, prompt_mask=None):
        self._set_image(image, key)
        self.session.reset_interactions()
        height, width = image.shape[:2]
        steps = []
        if prompt_mask is not None and prompt_mask.any():
            lasso = prompt_mask[None].astype(np.uint8)
            middle = [[MIDDLE, MIDDLE + 1], [0, height], [0, width]]
            steps.append(
                lambda run: self.session.add_lasso_interaction(
                    lasso, True, run, interaction_bbox=middle
                )
            )
        corners = [p for p, label in zip(points, labels) if int(label) in (2, 3)]
        if len(corners) == 2:
            (x0, y0), (x1, y1) = corners
            box = [
                [MIDDLE, MIDDLE + 1],
                [int(min(y0, y1)), max(int(min(y0, y1)) + 1, int(round(max(y0, y1))))],
                [int(min(x0, x1)), max(int(min(x0, x1)) + 1, int(round(max(x0, x1))))],
            ]
            steps.append(lambda run: self.session.add_bbox_interaction(box, True, run))
        for (x, y), label in zip(points, labels):
            if int(label) in (0, 1):
                where = (MIDDLE, min(max(int(y), 0), height - 1), min(max(int(x), 0), width - 1))
                include = int(label) == 1
                steps.append(
                    lambda run, where=where, include=include: self.session.add_point_interaction(
                        where, include, run
                    )
                )
        for index, step in enumerate(steps):
            step(index == len(steps) - 1)
        return self._target[MIDDLE].numpy() > 0


def main():
    model_folder = sys.argv[1]
    model = None
    send({"ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        cmd, reply_id = request.get("cmd"), request.get("id")
        if cmd == "quit":
            return
        try:
            if cmd != "points":
                raise RuntimeError(f"Unknown request {cmd!r}")
            if model is None:
                model = NnInteractive(model_folder)
            image = cv2.imread(request["image"], cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read {request['image']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            prompt_mask = None
            if request.get("mask"):
                prompt_mask = cv2.imread(request["mask"], cv2.IMREAD_GRAYSCALE) > 127
            mask = model.points(
                image, request.get("key"), request["points"], request["labels"], prompt_mask
            )
            cv2.imwrite(request["out"], mask.astype(np.uint8) * 255)
            send({"id": reply_id, "ok": True, "area": int(mask.sum())})
        except Exception as exc:
            traceback.print_exc()
            send({"id": reply_id, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
