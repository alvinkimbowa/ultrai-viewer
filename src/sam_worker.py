"""
SAM 3.1 helper process for UltrAi Viewer.

The viewer has no PyTorch, so it runs this script with the Python of an
environment that has PyTorch and the sam3 package, and talks to it through the
pipes: one JSON request per line on stdin, one JSON reply per line on stdout.
Every reply carries the "id" of its request; a propagation also sends progress
lines before its reply. Masks are exchanged as PNG files in folders the viewer
names. Anything else the libraries print goes to stderr.

Requests:
    open       {"video"}  frame count and size of the video, decoded as the viewer
               decodes it
    points     {"frame", "points", "labels", "out"}  the mask that include (1) and
               exclude (0) points, and a box given as corner points labelled 2
               and 3, select on one frame; points are pixels of the stored frame
    text       {"frame", "text", "out_dir"}  a mask for each match of a phrase
    propagate  {"objects", "start", "direction", "out_dir"}  tracks each object
               from its prompt masks ({frame: png path}) through the video, from
               frame start "both", "forward" or "backward", and saves the masks
               found as <out_dir>/<frame>_<object index>.png
    cancel     stops a running propagation
    quit
"""

import glob
import importlib.util
import json
import os
import queue
import sys
import tempfile
import threading
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The viewer's modules sit next to this script; generic names among them
# (labels, measurements) must not shadow modules sam3 and its dependencies import.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != SCRIPT_DIR]

PROTOCOL = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
os.dup2(2, 1)
sys.stdout = sys.stderr

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

CHECKPOINT_REPO = "facebook/sam3.1"
CHECKPOINT_FILE = "sam3.1_multiplex.pt"


def send(message):
    PROTOCOL.write(json.dumps(message) + "\n")
    PROTOCOL.flush()


def open_capture(path):
    spec = importlib.util.spec_from_file_location(
        "ultrai_tiff_stack", os.path.join(SCRIPT_DIR, "tiff_stack.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.open_capture(path)


def checkpoint_path():
    cached = glob.glob(
        os.path.expanduser(
            f"~/.cache/huggingface/hub/models--facebook--sam3.1/snapshots/*/{CHECKPOINT_FILE}"
        )
    )
    if cached:
        return cached[0]
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=CHECKPOINT_REPO, filename=CHECKPOINT_FILE)


def write_mask(path, mask):
    cv2.imwrite(path, mask.astype(np.uint8) * 255)


def read_mask(path):
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"Could not read prompt mask {path}")
    return raw > 127


class Canceled(Exception):
    pass


class Sam:
    def __init__(self, requests):
        self._requests = requests
        self._tracker = None
        self._predictor = None
        self._video = None
        self._frames = None
        self._images = None

    def tracker(self):
        """The SAM 3.1 tracker with its own image encoder.

        The checkpoint holds the detector and the tracker of the full model; the
        tracker's image encoder is the detector's vision backbone.
        """
        if self._tracker is None:
            from sam3.model_builder import build_sam3_multiplex_video_model

            model = build_sam3_multiplex_video_model(
                checkpoint_path=None, load_from_HF=False, device="cpu"
            )
            weights = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)
            weights = weights.get("model", weights)
            state = {}
            for key, value in weights.items():
                if key.startswith("tracker.model."):
                    state[key[len("tracker.model."):]] = value
                elif key.startswith("detector.backbone.vision_backbone."):
                    state["backbone." + key[len("detector.backbone."):]] = value
            missing, _ = model.load_state_dict(state, strict=False)
            if missing:
                raise RuntimeError(f"SAM 3.1 checkpoint lacks {len(missing)} tracker weights")
            del weights, state
            model = model.to(self.device()).eval()
            # Each prompt mask becomes a conditioning frame of its object's track.
            model.add_all_frames_to_correct_as_cond = True
            encode = model.forward_image
            # Without the detector, only the tracker's image features exist.
            model.forward_image = lambda image, **kw: encode(
                image, **{**kw, "need_sam3_out": False}
            )
            self._tracker = model
        return self._tracker

    def predictor(self):
        """The full SAM 3.1 model, loaded only for text prompts."""
        if self._predictor is None:
            from sam3.model_builder import build_sam3_multiplex_video_predictor

            self._predictor = build_sam3_multiplex_video_predictor(
                checkpoint_path=checkpoint_path(),
                use_fa3=False,
                async_loading_frames=False,
            )
        return self._predictor

    @staticmethod
    def device():
        return "cuda" if torch.cuda.is_available() else "cpu"

    def autocast(self):
        if self.device() == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.autocast("cpu", enabled=False)

    def open(self, video):
        if self._video == video:
            return self._size()
        capture = open_capture(video)
        frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            capture.release()
        if not frames:
            raise RuntimeError(f"No readable frames in {os.path.basename(video)}")
        side = self.tracker().image_size
        images = torch.stack(
            [
                torch.from_numpy(cv2.resize(f, (side, side), interpolation=cv2.INTER_LINEAR))
                for f in frames
            ]
        ).permute(0, 3, 1, 2)
        self._images = ((images.float() / 255.0 - 0.5) / 0.5).to(self.device())
        self._frames = frames
        self._video = video
        return self._size()

    def _size(self):
        height, width = self._frames[0].shape[:2]
        return {"frames": len(self._frames), "height": height, "width": width}

    def _new_state(self, first, count):
        height, width = self._frames[0].shape[:2]
        state = self.tracker().init_state(
            video_height=height, video_width=width, num_frames=count
        )
        state["images"] = self._images[first:first + count]
        return state

    def _check_cancel(self):
        try:
            while True:
                if self._requests.get_nowait().get("cmd") == "cancel":
                    raise Canceled()
        except queue.Empty:
            pass

    @torch.inference_mode()
    def points(self, frame, points, labels, out):
        height, width = self._frames[0].shape[:2]
        relative = torch.tensor(points, dtype=torch.float32) / torch.tensor(
            [width, height], dtype=torch.float32
        )
        with self.autocast():
            state = self._new_state(frame, 1)
            _, _, _, masks = self.tracker().add_new_points(
                state,
                0,
                1,
                relative,
                torch.tensor(labels, dtype=torch.int32),
                clear_old_points=True,
            )
        mask = (masks[0, 0] > 0).cpu().numpy()
        write_mask(out, mask)
        return {"area": int(mask.sum())}

    def text(self, frame, text, out_dir):
        predictor = self.predictor()
        folder = tempfile.mkdtemp(prefix="sam_text_")
        cv2.imwrite(
            os.path.join(folder, "0.jpg"),
            cv2.cvtColor(self._frames[frame], cv2.COLOR_RGB2BGR),
        )
        session = predictor.handle_request(dict(type="start_session", resource_path=folder))[
            "session_id"
        ]
        try:
            outputs = predictor.handle_request(
                dict(type="add_prompt", session_id=session, frame_index=0, text=text)
            )["outputs"]
        finally:
            predictor.handle_request(dict(type="close_session", session_id=session))
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for index, mask in enumerate(outputs["out_binary_masks"]):
            path = os.path.join(out_dir, f"{index}.png")
            write_mask(path, np.asarray(mask))
            paths.append(path)
        return {"masks": paths, "scores": [float(s) for s in outputs["out_probs"]]}

    @torch.inference_mode()
    def propagate(self, objects, start, direction, out_dir, reply_id):
        """Track each object on its own, so objects first seen on later frames
        need no slot in tracks begun before them.

        The tracker takes a mask on a frame only once that frame is tracked: the
        track is run forward from one prompt frame to the next, adding each
        frame's mask on arrival, and then from start in the chosen direction.
        """
        model = self.tracker()
        count = len(self._frames)
        reverse_passes = {"both": [False, True], "forward": [False], "backward": [True]}[
            direction
        ]
        prompts = [
            {int(frame): read_mask(path) for frame, path in obj["prompts"].items()}
            for obj in objects
        ]
        output_frames = set()
        for reverse in reverse_passes:
            output_frames.update(range(start, -1, -1) if reverse else range(start, count))
        total = sum(max(masks) - min(masks) for masks in prompts) + len(prompts) * len(
            output_frames
        )
        done = 0
        os.makedirs(out_dir, exist_ok=True)
        found = {}
        with self.autocast():
            for index, masks in enumerate(prompts):
                state = self._new_state(0, count)

                def add(frame):
                    model.add_new_masks(
                        state,
                        frame,
                        [1],
                        torch.from_numpy(masks[frame][None]).float().to(self.device()),
                    )
                    model.propagate_in_video_preflight(state, run_mem_encoder=True)

                frames = sorted(masks)
                add(frames[0])
                for previous, following in zip(frames, frames[1:]):
                    for _ in model.propagate_in_video(
                        state,
                        start_frame_idx=previous,
                        max_frame_num_to_track=following - previous,
                        reverse=False,
                        tqdm_disable=True,
                    ):
                        self._check_cancel()
                        done += 1
                        send({"id": reply_id, "progress": [done, total]})
                    add(following)
                for reverse in reverse_passes:
                    for frame, _, _, video_masks, _ in model.propagate_in_video(
                        state,
                        start_frame_idx=start,
                        max_frame_num_to_track=None,
                        reverse=reverse,
                        tqdm_disable=True,
                    ):
                        self._check_cancel()
                        mask = (video_masks[0, 0] > 0).cpu().numpy()
                        if mask.any():
                            write_mask(os.path.join(out_dir, f"{frame}_{index}.png"), mask)
                            found.setdefault(frame, []).append(index)
                        done += 1
                        send({"id": reply_id, "progress": [done, total]})
                del state
                if self.device() == "cuda":
                    torch.cuda.empty_cache()
        return {
            "frames": sorted(output_frames),
            "found": {str(frame): indices for frame, indices in found.items()},
        }


def read_requests(requests):
    for line in sys.stdin:
        line = line.strip()
        if line:
            requests.put(json.loads(line))
    requests.put({"cmd": "quit"})


def main():
    requests = queue.Queue()
    threading.Thread(target=read_requests, args=(requests,), daemon=True).start()
    sam = Sam(requests)
    send({"ready": True})
    while True:
        request = requests.get()
        cmd, reply_id = request.get("cmd"), request.get("id")
        if cmd == "quit":
            return
        if cmd == "cancel":
            continue
        try:
            if cmd == "open":
                result = sam.open(request["video"])
            elif cmd == "points":
                result = sam.points(
                    request["frame"], request["points"], request["labels"], request["out"]
                )
            elif cmd == "text":
                result = sam.text(request["frame"], request["text"], request["out_dir"])
            elif cmd == "propagate":
                result = sam.propagate(
                    request["objects"],
                    request["start"],
                    request["direction"],
                    request["out_dir"],
                    reply_id,
                )
            else:
                raise RuntimeError(f"Unknown request {cmd!r}")
            send({"id": reply_id, "ok": True, **result})
        except Canceled:
            send({"id": reply_id, "canceled": True})
        except Exception as exc:
            traceback.print_exc()
            send({"id": reply_id, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
