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
    image      {"image", "key"}  an 8-bit RGB PNG as a video of one frame, which the
               requests after it prompt as frame 0; kept while key stays the same
    points     {"frame", "points", "labels", "out", "mask"?}  the mask that include
               (1) and exclude (0) points, and a box given as corner points
               labelled 2 and 3, select on one frame; points are pixels of the
               stored frame. "mask", a PNG of the frame's size, prompts as
               well: as mask logits, with its bounding box when no box is given
    text       {"frame", "text", "out_dir"}  a mask for each match of a phrase
    propagate  {"objects", "start", "direction", "out_dir"}  tracks each object
               from its prompt masks ({frame: png path}) through the video, from
               frame start "both", "forward" or "backward", and saves the masks
               found as <out_dir>/<frame>_<object index>.png; its progress
               lines are [done, total, object index, frame]
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
# The magnitude of the logits a drawn mask is given, inside and outside it.
PROMPT_MASK_LOGIT = 8.0


def send(message):
    PROTOCOL.write(json.dumps(message) + "\n")
    PROTOCOL.flush()


def viewer_module(name):
    """One of the viewer's modules, loaded from its file."""
    spec = importlib.util.spec_from_file_location(
        f"ultrai_{name}", os.path.join(SCRIPT_DIR, f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def open_capture(path):
    return viewer_module("tiff_stack").open_capture(path)


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
            # Lets the interactive head take a drawn mask as mask logits with the
            # points; see _points_with_mask.
            model.iter_use_prev_mask_pred = True
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
        return self._use_frames(frames, video)

    def open_image(self, path, key):
        """An 8-bit RGB PNG as a video of one frame, kept while key stays the same."""
        if self._video == ("image", key):
            return self._size()
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {os.path.basename(path)}")
        return self._use_frames([cv2.cvtColor(image, cv2.COLOR_BGR2RGB)], ("image", key))

    def _use_frames(self, frames, source):
        side = self.tracker().image_size
        images = torch.stack(
            [
                torch.from_numpy(cv2.resize(f, (side, side), interpolation=cv2.INTER_LINEAR))
                for f in frames
            ]
        ).permute(0, 3, 1, 2)
        self._images = ((images.float() / 255.0 - 0.5) / 0.5).to(self.device())
        self._frames = frames
        self._video = source
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
    def points(self, frame, points, labels, out, prompt_mask=None):
        if prompt_mask is not None and prompt_mask.any():
            mask = self._points_with_mask(frame, points, labels, prompt_mask)
            write_mask(out, mask)
            return {"area": int(mask.sum())}
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

    def _points_with_mask(self, frame, points, labels, prompt_mask):
        """The mask that points and a drawn mask select, from the tracker's
        interactive head, as a new object would get it.

        The tracker's own mask prompts skip that head and come back as drawn, so
        the drawn mask is given as the mask logits the head refines a mask from,
        with its bounding box in place of a box not given.
        """
        model = self.tracker()
        height, width = self._frames[0].shape[:2]
        side = model.image_size
        if 2 not in labels:
            ys, xs = np.nonzero(prompt_mask)
            points = list(points) + [
                [float(xs.min()), float(ys.min())],
                [float(xs.max() + 1), float(ys.max() + 1)],
            ]
            labels = list(labels) + [2, 3]
        coords = torch.tensor(points, dtype=torch.float32)[None] * torch.tensor(
            [side / width, side / height]
        )
        resized = cv2.resize(
            prompt_mask.astype(np.float32), (side, side), interpolation=cv2.INTER_LINEAR
        )
        logits = np.where(resized > 0.5, PROMPT_MASK_LOGIT, -PROMPT_MASK_LOGIT)
        with self.autocast():
            state = self._new_state(frame, 1)
            obj_idx = model._obj_id_to_idx(state, 1)
            point_inputs = {
                "point_coords": coords.to(self.device()),
                "point_labels": torch.tensor([labels], dtype=torch.int32, device=self.device()),
            }
            state["point_inputs_per_obj"][obj_idx][0] = point_inputs
            state["multiplex_state"] = model.multiplex_controller.get_state(
                num_valid_entries=1,
                device=state["device"],
                dtype=torch.float32,
                random=False,
                object_ids=[1],
            )
            current, _ = model._run_single_frame_inference(
                inference_state=state,
                output_dict=state["output_dict"],
                frame_idx=0,
                batch_size=1,
                is_init_cond_frame=True,
                point_inputs=point_inputs,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=False,
                prev_sam_mask_logits=torch.from_numpy(logits.astype(np.float32))[None, None].to(
                    self.device()
                ),
                add_to_existing_state=False,
                new_obj_idxs=[obj_idx],
                new_obj_ids=[1],
                allow_new_buckets=True,
                prefer_new_buckets=True,
                objects_to_interact=[0],
            )
            _, masks = model._get_orig_video_res_output(state, current["pred_masks"])
        return (masks[0, 0] > 0).cpu().numpy()

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
        need no slot in tracks begun before them, in the stretches sam_plan
        gives.

        The tracker takes a mask on a frame only once that frame is tracked: a
        stretch is run forward from one prompt frame to the next, adding each
        frame's mask on arrival, and then through the frames it saves.
        """
        plan_module = viewer_module("sam_plan")
        model = self.tracker()
        count = len(self._frames)
        prompts = [
            {int(frame): read_mask(path) for frame, path in obj["prompts"].items()}
            for obj in objects
        ]
        plans = [plan_module.track_plan(masks, start, direction, count) for masks in prompts]
        total = sum(plan_module.stretch_steps(st) for plan in plans for st in plan)
        done = 0
        os.makedirs(out_dir, exist_ok=True)
        found = {}
        saved_frames = set()

        def step(index, frame):
            nonlocal done
            self._check_cancel()
            done += 1
            send({"id": reply_id, "progress": [done, total, index, frame]})

        with self.autocast():
            for index, (masks, plan) in enumerate(zip(prompts, plans)):
                for stretch in plan:
                    state = self._new_state(0, count)

                    def add(frame):
                        model.add_new_masks(
                            state,
                            frame,
                            [1],
                            torch.from_numpy(masks[frame][None]).float().to(self.device()),
                        )
                        model.propagate_in_video_preflight(state, run_mem_encoder=True)

                    frames = stretch["prompts"]
                    add(frames[0])
                    for previous, following in zip(frames, frames[1:]):
                        for tracked in model.propagate_in_video(
                            state,
                            start_frame_idx=previous,
                            max_frame_num_to_track=following - previous,
                            reverse=False,
                            tqdm_disable=True,
                        ):
                            step(index, tracked[0])
                        add(following)
                    saved = set(stretch["saved"])
                    for first, last, reverse in stretch["passes"]:
                        for frame, _, _, video_masks, _ in model.propagate_in_video(
                            state,
                            start_frame_idx=first,
                            max_frame_num_to_track=abs(last - first),
                            reverse=reverse,
                            tqdm_disable=True,
                        ):
                            step(index, frame)
                            if frame not in saved:
                                continue
                            mask = (video_masks[0, 0] > 0).cpu().numpy()
                            if mask.any():
                                write_mask(os.path.join(out_dir, f"{frame}_{index}.png"), mask)
                                found.setdefault(frame, []).append(index)
                    saved_frames |= saved
                    del state
                    if self.device() == "cuda":
                        torch.cuda.empty_cache()
        return {
            "frames": sorted(saved_frames),
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
            elif cmd == "image":
                result = sam.open_image(request["image"], request.get("key"))
            elif cmd == "points":
                prompt_mask = read_mask(request["mask"]) if request.get("mask") else None
                result = sam.points(
                    request["frame"],
                    request["points"],
                    request["labels"],
                    request["out"],
                    prompt_mask,
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
