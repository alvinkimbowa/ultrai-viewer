"""
nnInteractive helper process for UltrAi Viewer.

nnInteractive is MIC-DKFZ's promptable nnU-Net, trained on 3D volumes. A 2D
image is given to it as a volume of DEPTH copies of the image, every prompt lies
on the middle slice, and the mask is that slice of the result. The viewer has
no PyTorch, so it runs this script with the Python of an environment that has
PyTorch and the nnInteractive package, and talks to it
through the pipes: one JSON request per line on stdin, one JSON reply per line
on stdout. Every reply carries the "id" of its request. Anything else the
libraries print goes to stderr.

Requests:
    prompts  {"image", "key", "prompts", "out"}  the mask that the prompts select
             on an 8-bit RGB PNG, in the order they were given. Each prompt is
             {"id", "kind", "include"} and, by kind, "x" and "y" for a "point",
             "box" [left, top, right, bottom] for a "box", or "mask", a PNG of the
             image's size holding a closed outline, for a "lasso"; positions are
             pixels of the image. The image is kept while requests carry the same
             key.
    load     {}  loads the model, so that the first prompt need not wait for it
    quit

nnInteractive is used as it is meant to be: one session per image, to which
each prompt is added in turn and followed by a prediction, so that later prompts
weigh more and each prediction starts from the one before. A request whose
prompts are those already added and more adds only the new ones; one that drops
the last prompt takes it back with nnInteractive's undo; any other change clears
the session and adds the prompts again in their order.
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
            enable_undo=True,
        )
        self.session.initialize_from_trained_model_folder(str(model_folder))
        self._key = None
        self._target = None
        # Ids of the prompts in the session, in the order added; None when the
        # session no longer matches them, after a failed request.
        self._added = []
        self._can_undo = False

    def _set_image(self, image, key):
        if key is not None and key == self._key and self._target is not None:
            return
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        self.session.set_image(np.repeat(grey[None, None], DEPTH, axis=1))
        self._target = torch.zeros((DEPTH, *grey.shape), dtype=torch.uint8)
        self.session.set_target_buffer(self._target)
        self._key = key
        self._added = []
        self._can_undo = False

    def _add(self, prompt, height, width):
        include = bool(prompt["include"])
        if prompt["kind"] == "point":
            x = min(max(int(prompt["x"]), 0), width - 1)
            y = min(max(int(prompt["y"]), 0), height - 1)
            self.session.add_point_interaction((MIDDLE, y, x), include, True)
        elif prompt["kind"] == "box":
            left, top, right, bottom = prompt["box"]
            box = [
                [MIDDLE, MIDDLE + 1],
                [int(top), max(int(top) + 1, int(round(bottom)))],
                [int(left), max(int(left) + 1, int(round(right)))],
            ]
            self.session.add_bbox_interaction(box, include, True)
        else:
            outline = cv2.imread(prompt["mask"], cv2.IMREAD_GRAYSCALE)
            if outline is None:
                raise RuntimeError(f"Could not read {prompt['mask']}")
            middle = [[MIDDLE, MIDDLE + 1], [0, height], [0, width]]
            self.session.add_lasso_interaction(
                (outline > 127)[None].astype(np.uint8), include, True, interaction_bbox=middle
            )
        self._added.append(prompt["id"])
        self._can_undo = True

    @torch.inference_mode()
    def prompts(self, image, key, prompts):
        self._set_image(image, key)
        height, width = image.shape[:2]
        ids = [prompt["id"] for prompt in prompts]
        added = self._added
        try:
            if added is not None and ids[: len(added)] == added:
                new = prompts[len(added):]
            elif (
                added is not None
                and ids == added[:-1]
                and self._can_undo
                and self.session.undo()
            ):
                self._added = ids
                self._can_undo = False
                new = []
            else:
                self.session.reset_interactions()
                self._added = []
                self._can_undo = False
                new = prompts
            for prompt in new:
                self._add(prompt, height, width)
        except Exception:
            self._added = None
            raise
        if not self._added:
            return np.zeros((height, width), dtype=bool)
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
            if cmd not in ("prompts", "load"):
                raise RuntimeError(f"Unknown request {cmd!r}")
            if model is None:
                model = NnInteractive(model_folder)
            if cmd == "load":
                send({"id": reply_id, "ok": True})
                continue
            image = cv2.imread(request["image"], cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read {request['image']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = model.prompts(image, request.get("key"), request["prompts"])
            cv2.imwrite(request["out"], mask.astype(np.uint8) * 255)
            send({"id": reply_id, "ok": True, "area": int(mask.sum())})
        except Exception as exc:
            traceback.print_exc()
            send({"id": reply_id, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
