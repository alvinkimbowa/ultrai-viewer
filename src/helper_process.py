"""
A model run in a helper process with the Python of a separate environment that
has PyTorch, which the viewer does not ship. Requests and replies are JSON lines
on the helper's stdin and stdout; masks are exchanged as PNG files.
"""

import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path


class HelperCanceled(Exception):
    pass


class HelperProcess:
    """Requests to a helper process, which is started on first use and keeps
    its model loaded until the viewer closes.

    A model whose single_prompt is True was trained on one prompt per mask, one
    point or one box, and one with exclude_points False has no trained exclude
    point; one with exclude_boxes True takes boxes that exclude what they hold.
    One with ordered_prompts True takes its prompts in the order they were given,
    through segment_prompts, rather than as points, a box and a mask.

    Requests block until answered, so they are made from worker threads; one
    runs at a time. cancel() may be called from any thread.
    """

    single_prompt = False
    exclude_points = True
    exclude_boxes = False
    ordered_prompts = False

    def __init__(self, label, command, pythonpath, log_file, scratch_prefix):
        self.label = label
        self._command = [str(part) for part in command]
        self._pythonpath = str(pythonpath) if pythonpath else ""
        self._log_file = Path(log_file)
        self._process = None
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._next_id = 0
        self._scratch = tempfile.TemporaryDirectory(prefix=scratch_prefix)

    def scratch_dir(self, name):
        """An empty folder for the files of one request."""
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
            p for p in (self._pythonpath, env.get("PYTHONPATH", "")) if p
        )
        log = open(self._log_file, "a", encoding="utf-8")
        self._process = subprocess.Popen(
            self._command,
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
            raise RuntimeError(f"{self.label} could not start. See {self._log_file}.")

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
                    raise RuntimeError(
                        f"{self.label} stopped unexpectedly. See {self._log_file}."
                    )
                reply = json.loads(line)
                if reply.get("id") != request_id:
                    continue
                if "progress" in reply:
                    if on_progress is not None:
                        on_progress(*reply["progress"])
                    continue
                if reply.get("canceled"):
                    raise HelperCanceled()
                if "error" in reply:
                    raise RuntimeError(reply["error"])
                return reply

    def warm_up(self):
        """Start the helper and load its model on a background thread, so that
        the first request finds it ready. A request made meanwhile waits for it."""

        def load():
            try:
                self.request("load")
            except Exception:
                pass

        threading.Thread(target=load, daemon=True).start()

    def cancel(self):
        if self._process is not None and self._process.poll() is None:
            try:
                self._send({"cmd": "cancel"})
            except OSError:
                pass

    def stop(self):
        """End the helper, freeing the memory its model holds; the next request
        starts it again."""
        if self._process is not None and self._process.poll() is None:
            try:
                self._send({"cmd": "quit"})
                self._process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self._process.kill()
        self._process = None

    def close(self):
        self.stop()
        self._scratch.cleanup()
