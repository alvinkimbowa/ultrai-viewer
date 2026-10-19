"""
SAM 3.1 for the viewer, run in a helper process (sam_worker.py).

SAM 3.1 needs PyTorch, which the viewer does not ship, so the helper runs with
the Python of a separate environment that has PyTorch and Meta's sam3 package:
ULTRAI_SAM_PYTHON, else the .venv of the sam3 checkout. ULTRAI_SAM_REPO is that
checkout, by default ~/UltrAi/projects/sam3.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import cv2

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


def load_prompt_frames(folder):
    """The frames of a video whose masks SAM propagates from, as kept in its
    results folder, or None while none are kept.

    The other frames there hold what SAM found, which must not prompt it again.
    """
    data = _read_prompts_file(folder)
    if data is None:
        return None
    return {int(frame) for frame in data.get("frames", [])}


def load_propagated_frames(folder):
    """The prompt frames of the last propagation of a video that was saved in
    full, or None when none was."""
    data = _read_prompts_file(folder)
    if data is None or "propagated" not in data:
        return None
    return {int(frame) for frame in data["propagated"]}


def save_prompt_frames(folder, frames, propagated=None):
    """Keep a video's prompt frames, and, given propagated, the prompt frames of
    a propagation just saved in full; otherwise those kept stay."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    data = {"frames": sorted(int(f) for f in frames)}
    if propagated is None:
        propagated = load_propagated_frames(folder)
    if propagated is not None:
        data["propagated"] = sorted(int(f) for f in propagated)
    (folder / PROMPTS_FILE_NAME).write_text(json.dumps(data), encoding="utf-8")


class SamCanceled(Exception):
    pass


def read_mask(path):
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"SAM returned no readable mask: {path}")
    return raw > 127


class SamTracker:
    """Requests to the helper process, which is started on first use and keeps
    the model loaded until the viewer closes.

    Requests block until answered, so they are made from worker threads; one
    runs at a time. cancel() may be called from any thread.
    """

    def __init__(self):
        self._process = None
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._next_id = 0
        self._scratch = tempfile.TemporaryDirectory(prefix="ultrai_sam_")

    def scratch_dir(self, name):
        """An empty folder for the masks of one request."""
        folder = Path(self._scratch.name) / name
        if folder.exists():
            for path in folder.iterdir():
                path.unlink()
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _start(self):
        if self._process is not None and self._process.poll() is None:
            return
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(sam_repo()), env.get("PYTHONPATH", "")) if p
        )
        log = open(LOG_FILE, "a", encoding="utf-8")
        self._process = subprocess.Popen(
            [str(sam_python()), "-u", str(WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            env=env,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        log.close()
        first = self._process.stdout.readline()
        if not first or not json.loads(first).get("ready"):
            raise RuntimeError(f"SAM 3.1 could not start. See {LOG_FILE}.")

    def _send(self, message):
        with self._write_lock:
            self._process.stdin.write(json.dumps(message) + "\n")
            self._process.stdin.flush()

    def request(self, cmd, on_progress=None, **fields):
        with self._lock:
            self._start()
            self._next_id += 1
            request_id = self._next_id
            self._send({"cmd": cmd, "id": request_id, **fields})
            while True:
                line = self._process.stdout.readline()
                if not line:
                    raise RuntimeError(f"SAM 3.1 stopped unexpectedly. See {LOG_FILE}.")
                reply = json.loads(line)
                if reply.get("id") != request_id:
                    continue
                if "progress" in reply:
                    if on_progress is not None:
                        on_progress(*reply["progress"])
                    continue
                if reply.get("canceled"):
                    raise SamCanceled()
                if "error" in reply:
                    raise RuntimeError(reply["error"])
                return reply

    def cancel(self):
        if self._process is not None and self._process.poll() is None:
            try:
                self._send({"cmd": "cancel"})
            except OSError:
                pass

    def close(self):
        if self._process is not None and self._process.poll() is None:
            try:
                self._send({"cmd": "quit"})
                self._process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self._process.kill()
        self._process = None
        self._scratch.cleanup()

    def open(self, video):
        return self.request("open", video=str(video))

    def segment_points(self, video, frame, points, labels):
        """The mask points select on a frame: (x, y) pixels of the stored frame,
        labelled 1 to include, 0 to exclude, and 2 and 3 for the corners of a box."""
        self.open(video)
        out = self.scratch_dir("points") / "mask.png"
        self.request(
            "points",
            frame=int(frame),
            points=[[float(x), float(y)] for x, y in points],
            labels=[int(label) for label in labels],
            out=str(out),
        )
        return read_mask(out)

    def find_text(self, video, frame, text):
        """A mask for each match of a phrase on a frame."""
        self.open(video)
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
