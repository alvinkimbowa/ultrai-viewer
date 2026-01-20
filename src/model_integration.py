"""
Model integration for single-image ONNX inference (CPU).
"""

import os
import numpy as np
import cv2


class ModelIntegration:
    def __init__(self, model_path: str | None = None):
        self._model_path = model_path or self._locate_model()
        self._session = None

    def has_model(self) -> bool:
        return bool(self._model_path)

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

        self._session = ort.InferenceSession(
            self._model_path,
            providers=["CPUExecutionProvider"],
        )
        return self._session

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
        return data

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
        outputs = session.run(None, {input_name: model_input})
        if not outputs:
            raise RuntimeError("Model returned no outputs.")
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Inference canceled.")
        return self._postprocess(outputs[0])
