"""
Model integration for single-image ONNX inference (CPU/GPU).

A model with a one-channel input is fed a min-max scaled grey image; one with a
three-channel input is taken to be a DINOv3 model and fed an RGB image prepared
as DINOv3 was trained. Either returns the class index of each pixel, covering
the whole image given, for the caller to resize onto it.

A model whose input is (1, T, C, H, W) reads a clip of T video frames, each
prepared as a single image would be, and gives the class scores of every frame
of it; see segment_video.
"""

import os
import sys
import json
import contextlib
import numpy as np
import cv2


def clip_starts(n_frames: int, length: int, stride: int) -> list[int]:
    """Where windows of length frames start so that, stride apart, they cover
    all n_frames, the last one moved back to end on the last frame. A video
    shorter than a window has one window, at its start."""
    if n_frames <= length:
        return [0]
    starts = list(range(0, n_frames - length + 1, stride))
    if starts[-1] != n_frames - length:
        starts.append(n_frames - length)
    return starts


GPU_FALLBACK_WARNING = (
    "GPU selected but could not be used. Falling back to CPU.\n\n"
    "This usually means there is no supported GPU or required drivers "
    "(NVIDIA driver/CUDA/cuDNN) are missing."
)
# Side of the square input of a DINOv3 model whose input size is not fixed.
DINOV3_DEFAULT_SIZE = 896
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ModelIntegration:
    def __init__(self, model_path: str | None = None):
        self._model_path = model_path or self._locate_model()
        self._session = None
        self._device_provider = "CPUExecutionProvider"
        self._device_warning = None

    def has_model(self) -> bool:
        return bool(self._model_path)

    def available_devices(self) -> list[tuple[str, str]]:
        try:
            import onnxruntime as ort
        except Exception:
            return [("CPU", "CPUExecutionProvider")]

        providers = set(ort.get_available_providers())
        devices = []
        if "CUDAExecutionProvider" in providers:
            devices.append(("GPU", "CUDAExecutionProvider"))
        devices.append(("CPU", "CPUExecutionProvider"))
        return devices

    def set_device(self, provider: str) -> None:
        try:
            import onnxruntime as ort
        except Exception:
            provider = "CPUExecutionProvider"
        else:
            available = ort.get_available_providers()
            if provider not in available:
                raise ValueError(f"Device provider not available: {provider}")
        self._device_provider = provider
        self._session = None

    def current_device(self) -> str:
        return self._device_provider

    def consume_device_warning(self) -> str | None:
        warning = self._device_warning
        self._device_warning = None
        return warning

    def device_warning(self) -> str | None:
        return self._device_warning

    def list_models(self) -> list[str]:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        assets_dir = os.path.join(base_dir, "assets")
        if not os.path.isdir(assets_dir):
            return []
        return [
            os.path.splitext(name)[0]
            for name in sorted(os.listdir(assets_dir))
            if name.endswith((".onnx", ".nnx"))
        ]

    def set_model(self, name: str) -> None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        assets_dir = os.path.join(base_dir, "assets")
        if not name:
            self._model_path = None
            self._session = None
            return
        candidate = os.path.join(assets_dir, name)
        if not candidate.endswith((".onnx", ".nnx")):
            candidate = f"{candidate}.onnx"
        if not os.path.exists(candidate):
            alt = f"{os.path.splitext(candidate)[0]}.nnx"
            if os.path.exists(alt):
                candidate = alt
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Model not found: {name}")
        self._model_path = candidate
        self._session = None

    def current_model(self) -> str | None:
        if not self._model_path:
            return None
        return os.path.splitext(os.path.basename(self._model_path))[0]

    def _locate_model(self) -> str | None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        assets_dir = os.path.join(base_dir, "assets")
        if not os.path.isdir(assets_dir):
            return None
        candidates = []
        for name in sorted(os.listdir(assets_dir)):
            if name.endswith((".onnx", ".nnx")):
                candidates.append(os.path.join(assets_dir, name))
        return candidates[0] if candidates else None

    def _ensure_session(self):
        if self._session is not None:
            return self._session
        if not self._model_path:
            return None
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            ort.preload_dlls()

        self._device_warning = None
        providers = ort.get_available_providers()
        selected = self._device_provider
        if selected not in providers:
            self._device_warning = GPU_FALLBACK_WARNING
            selected = "CPUExecutionProvider"
        provider_list = [selected]
        if selected != "CPUExecutionProvider" and "CPUExecutionProvider" in providers:
            provider_list.append("CPUExecutionProvider")
        try:
            with self._redirect_stderr_to_log():
                self._session = ort.InferenceSession(
                    self._model_path,
                    providers=provider_list,
                )
        except Exception as exc:
            if selected != "CPUExecutionProvider":
                self._device_warning = (
                    GPU_FALLBACK_WARNING
                )
                with self._redirect_stderr_to_log():
                    self._session = ort.InferenceSession(
                        self._model_path,
                        providers=["CPUExecutionProvider"],
                    )
            else:
                raise exc
        if selected != "CPUExecutionProvider" and self._session is not None:
            try:
                active_providers = self._session.get_providers()
            except Exception:
                active_providers = []
            if not active_providers or active_providers[0] != selected:
                self._device_warning = GPU_FALLBACK_WARNING
        return self._session

    @contextlib.contextmanager
    def _redirect_stderr_to_log(self):
        log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inference.log")
        log_file = open(log_path, "a", encoding="utf-8")
        if (
            sys.stderr is None
            or sys.stdout is None
            or not hasattr(sys.stderr, "fileno")
            or not hasattr(sys.stdout, "fileno")
        ):
            try:
                with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                    yield
            finally:
                log_file.close()
            return
        stderr_fd = sys.stderr.fileno()
        stdout_fd = sys.stdout.fileno()
        saved_stderr_fd = os.dup(stderr_fd)
        saved_stdout_fd = os.dup(stdout_fd)
        try:
            os.dup2(log_file.fileno(), stderr_fd)
            os.dup2(log_file.fileno(), stdout_fd)
            yield
        finally:
            os.dup2(saved_stderr_fd, stderr_fd)
            os.dup2(saved_stdout_fd, stdout_fd)
            os.close(saved_stderr_fd)
            os.close(saved_stdout_fd)
            log_file.close()

    def preload(self) -> bool:
        return self._ensure_session() is not None

    def _image_input_shape(self, session) -> list:
        """The model's input shape as (batch, channels, height, width), leaving
        out the frames axis of a model that reads clips."""
        shape = list(session.get_inputs()[0].shape)
        return shape[:1] + shape[2:] if len(shape) == 5 else shape

    def clip_length(self) -> int | None:
        """The number of frames a model that reads clips takes at once, or None
        for a model of single images. Loads the model."""
        session = self._ensure_session()
        if session is None:
            return None
        shape = session.get_inputs()[0].shape
        if len(shape) == 5 and isinstance(shape[1], int) and shape[1] > 0:
            return shape[1]
        return None

    def _model_input_hw(self, session) -> tuple[int, int]:
        shape = self._image_input_shape(session)
        height = shape[2] if len(shape) > 2 and isinstance(shape[2], int) else None
        width = shape[3] if len(shape) > 3 and isinstance(shape[3], int) else None
        if not height or not width:
            size = DINOV3_DEFAULT_SIZE if self._model_channels(session) == 3 else 512
            height, width = size, size
        return height, width

    def _model_channels(self, session) -> int:
        shape = self._image_input_shape(session)
        channels = shape[1] if len(shape) > 1 else None
        return channels if isinstance(channels, int) else 1

    def class_names(self) -> dict[int, str]:
        """Labels of the model's output classes by index, from the "classes"
        list of a <model>.json beside the model; empty when there is none.

        Index 0 is the background and has no label.
        """
        if not self._model_path:
            return {}
        sidecar = os.path.splitext(self._model_path)[0] + ".json"
        try:
            with open(sidecar, encoding="utf-8") as handle:
                classes = json.load(handle).get("classes", [])
        except (OSError, ValueError, AttributeError):
            return {}
        return {
            index: str(name).strip().lower()
            for index, name in enumerate(classes)
            if index > 0 and str(name).strip()
        }

    def _prepare_input(self, image, session):
        """The model input, and the part of the model's output that covers the image.

        The part is (top, left, height, width) in the output at the model's input
        size, or None when the image fills it.
        """
        if self._model_channels(session) == 3:
            return self._prepare_rgb_input(image, session)
        data = image
        if data.ndim == 3:
            if data.shape[2] == 4:
                data = data[:, :, :3]
            if data.shape[2] == 3:
                data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
            elif data.shape[2] == 1:
                data = data[:, :, 0]
        if data.ndim != 2:
            raise ValueError(f"Unsupported image layout: {data.shape}")
        data = data.astype(np.float32, copy=False)
        min_val = float(np.min(data))
        max_val = float(np.max(data))
        if max_val > min_val:
            data = (data - min_val) / (max_val - min_val)
        else:
            data = np.zeros_like(data, dtype=np.float32)
        target_h, target_w = self._model_input_hw(session)
        resized = cv2.resize(data, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return resized[None, None, :, :].astype(np.float32), None

    def _prepare_rgb_input(self, image, session):
        """Input of a DINOv3 model, prepared the way it was trained.

        The image is resized with its aspect ratio kept to fit the model's square
        input, centred on a black canvas, and normalised with the ImageNet mean
        and standard deviation.
        """
        data = np.asarray(image)
        if data.ndim == 2:
            data = np.stack([data] * 3, axis=-1)
        elif data.shape[2] == 1:
            data = np.repeat(data, 3, axis=2)
        elif data.shape[2] == 4:
            data = data[:, :, :3]
        if data.dtype != np.uint8:
            peak = float(data.max())
            data = (data.astype(np.float32) * (255.0 / peak) if peak > 0 else data)
            data = np.clip(data, 0, 255).astype(np.uint8)
        target_h, target_w = self._model_input_hw(session)
        height, width = data.shape[:2]
        scale = min(target_h / height, target_w / width)
        resized_h = max(1, round(height * scale))
        resized_w = max(1, round(width * scale))
        resized = cv2.resize(data, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
        top = (target_h - resized_h) // 2
        left = (target_w - resized_w) // 2
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[top : top + resized_h, left : left + resized_w] = resized
        normalized = (canvas.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        model_input = normalized.transpose(2, 0, 1)[None].astype(np.float32)
        return model_input, (top, left, resized_h, resized_w)

    def _postprocess(self, prediction, input_hw, part) -> np.ndarray:
        """Class index of each pixel, with only the largest piece of each class
        kept, cut to the part of the output that covers the image."""
        data = np.asarray(prediction)
        while data.ndim > 3 and data.shape[0] == 1:
            data = data[0]
        if data.ndim == 3:
            if data.shape[0] > 1:
                data = np.argmax(data, axis=0)
            else:
                scores = data[0]
                cut = 0.5 if 0.0 <= float(scores.min()) and float(scores.max()) <= 1.0 else 0.0
                data = scores > cut
        labels = np.squeeze(data).astype(np.uint8)
        if part is not None:
            if labels.shape != tuple(input_hw):
                labels = cv2.resize(
                    labels, (input_hw[1], input_hw[0]), interpolation=cv2.INTER_NEAREST
                )
            top, left, height, width = part
            labels = labels[top : top + height, left : left + width]
        return self._largest_piece_per_class(labels)

    def _largest_piece_per_class(self, labels):
        kept = np.zeros_like(labels, dtype=np.uint8)
        for index in np.unique(labels):
            if index == 0:
                continue
            count, pieces = cv2.connectedComponents(
                (labels == index).astype(np.uint8), connectivity=8
            )
            if count <= 1:
                continue
            sizes = np.bincount(pieces.ravel())
            sizes[0] = 0
            kept[pieces == int(np.argmax(sizes))] = index
        return kept

    def run_inference(self, image, cancel_event=None) -> np.ndarray:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Inference canceled.")
        session = self._ensure_session()
        if session is None:
            raise RuntimeError("Model session is not available.")
        if self.clip_length():
            # A still image is read as a clip of that image alone.
            return next(self.segment_video([image], cancel_event))
        model_input, part = self._prepare_input(image, session)
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Inference canceled.")
        input_name = session.get_inputs()[0].name
        with self._redirect_stderr_to_log():
            outputs = session.run(None, {input_name: model_input})
        if not outputs:
            raise RuntimeError("Model returned no outputs.")
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Inference canceled.")
        return self._postprocess(outputs[0], self._model_input_hw(session), part)

    def segment_video(self, frames, cancel_event=None):
        """Yield the class index of each pixel of every frame of a video, in
        order, from a model that reads clips.

        frames is an iterable of the video's images, each as run_inference
        takes one. The video is read in windows of clip_length() frames, half a
        window apart, the last one ending on the last frame, and a frame takes
        the mean of the class probabilities every window holding it gave it.
        A video shorter than a window is filled out with its last frame.
        """
        session = self._ensure_session()
        if session is None:
            raise RuntimeError("Model session is not available.")
        length = self.clip_length()
        if not length:
            raise RuntimeError("The model does not read video clips.")
        stride = max(1, length // 2)
        input_hw = self._model_input_hw(session)
        inputs, parts, totals, counts = {}, {}, {}, {}
        state = {"next": 0, "last_start": None}

        def run_window(start, stop):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Inference canceled.")
            window = [inputs[index] for index in range(start, stop)]
            window += [window[-1]] * (length - len(window))
            input_name = session.get_inputs()[0].name
            with self._redirect_stderr_to_log():
                outputs = session.run(None, {input_name: np.stack(window)[None]})
            if not outputs:
                raise RuntimeError("Model returned no outputs.")
            scores = np.asarray(outputs[0], dtype=np.float32)[0]
            if scores.shape[1] > 1:
                scores = np.exp(scores - scores.max(axis=1, keepdims=True))
                probabilities = scores / scores.sum(axis=1, keepdims=True)
            else:
                probabilities = 1.0 / (1.0 + np.exp(-scores))
            for index in range(start, stop):
                frame = probabilities[index - start]
                totals[index] = totals[index] + frame if index in totals else frame
                counts[index] = counts.get(index, 0) + 1
            state["last_start"] = start

        def finished(upto):
            while state["next"] < upto:
                index = state["next"]
                mean = totals.pop(index) / counts.pop(index)
                labels = self._postprocess(mean, input_hw, parts.pop(index))
                inputs.pop(index, None)
                state["next"] += 1
                yield labels

        count = 0
        for index, image in enumerate(frames):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Inference canceled.")
            model_input, parts[index] = self._prepare_input(image, session)
            inputs[index] = model_input[0]
            count = index + 1
            start = count - length
            if start >= 0 and start % stride == 0:
                run_window(start, count)
                # Only the window ending on the last frame starts between this
                # one and the next, so frames up to this start are complete.
                yield from finished(start + 1)
        if not count:
            return
        last_start = max(0, count - length)
        if state["last_start"] != last_start:
            run_window(last_start, count)
        yield from finished(count)
