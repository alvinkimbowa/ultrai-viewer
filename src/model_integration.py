"""
Model integration for single-image ONNX inference (CPU/GPU).
"""

import os
import sys
import contextlib
import numpy as np
import cv2


GPU_FALLBACK_WARNING = (
    "GPU selected but could not be used. Falling back to CPU.\n\n"
    "This usually means there is no supported GPU or required drivers "
    "(NVIDIA driver/CUDA/cuDNN) are missing."
)


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

    def _model_input_hw(self, session) -> tuple[int, int]:
        shape = session.get_inputs()[0].shape
        height = shape[2] if len(shape) > 2 and isinstance(shape[2], int) else None
        width = shape[3] if len(shape) > 3 and isinstance(shape[3], int) else None
        if not height or not width:
            height, width = 512, 512
        return height, width

    def _prepare_input(self, image, session) -> np.ndarray:
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
        return resized[None, None, :, :].astype(np.float32)

    def _postprocess(self, prediction) -> np.ndarray:
        data = np.asarray(prediction)
        if data.ndim == 4:
            data = np.argmax(data, axis=1)
        if data.ndim == 3 and data.shape[0] > 1:
            data = np.argmax(data, axis=0)
        if data.ndim >= 2:
            data = np.squeeze(data)
        return self._largest_component(data)

    def _largest_component(self, mask):
        binary = np.asarray(mask) > 0
        if binary.ndim != 2:
            binary = np.squeeze(binary)
        if binary.size == 0 or not np.any(binary):
            return np.zeros_like(binary, dtype=np.uint8)
        count, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
        if count <= 1:
            return binary.astype(np.uint8)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        largest = int(np.argmax(sizes))
        return (labels == largest).astype(np.uint8)

    def run_inference(self, image, cancel_event=None) -> np.ndarray:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Inference canceled.")
        session = self._ensure_session()
        if session is None:
            raise RuntimeError("Model session is not available.")
        model_input = self._prepare_input(image, session)
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Inference canceled.")
        input_name = session.get_inputs()[0].name
        with self._redirect_stderr_to_log():
            outputs = session.run(None, {input_name: model_input})
        if not outputs:
            raise RuntimeError("Model returned no outputs.")
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Inference canceled.")
        return self._postprocess(outputs[0])
