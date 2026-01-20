"""
Canvas widget for displaying images.
"""

from PyQt6.QtWidgets import QWidget, QFileDialog, QMessageBox
from PyQt6.QtGui import QImage, QPixmap, QPainter
from PyQt6.QtCore import Qt, QRect
import numpy as np
import cv2
import tifffile


class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.pixmap = None

    def load_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if path:
            self.load_image(path)

    def load_image(self, file_path):
        try:
            raw = self._read_image(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        self.image = raw
        self.pixmap = self._to_pixmap(raw)
        self.update()

    def _read_image(self, file_path):
        lower_path = file_path.lower()
        if lower_path.endswith((".tif", ".tiff")):
            image = tifffile.imread(file_path)
        else:
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Unsupported image format: {file_path}")
            if image.ndim == 3:
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        if image is None:
            raise ValueError(f"Unsupported image format: {file_path}")
        return image

    def _to_pixmap(self, image):
        if image.ndim == 2:
            display = self._normalize_gray(image)
            height, width = display.shape
            bytes_per_line = display.strides[0]
            q_image = QImage(
                display.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8
            )
            return QPixmap.fromImage(q_image)
        if image.ndim == 3 and image.shape[2] in (3, 4):
            if image.shape[2] == 4:
                image = image[:, :, :3]
            display = self._as_uint8(image)
            height, width, _ = display.shape
            bytes_per_line = display.strides[0]
            q_image = QImage(
                display.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
            )
            return QPixmap.fromImage(q_image)
        raise ValueError(f"Unsupported image layout: {image.shape}")

    def _as_uint8(self, image):
        if image.dtype == np.uint8:
            return image
        if image.dtype == np.uint16:
            return (image >> 8).astype(np.uint8)
        return np.clip(image, 0, 255).astype(np.uint8)

    def _normalize_gray(self, image):
        image_float = image.astype(np.float32, copy=False)
        min_val = float(np.min(image_float))
        max_val = float(np.max(image_float))
        if max_val <= min_val:
            return np.zeros(image_float.shape, dtype=np.uint8)
        scaled = (image_float - min_val) / (max_val - min_val)
        scaled = np.clip(scaled, 0.0, 1.0)
        return (scaled * 255.0).astype(np.uint8)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.darkGray)
        if self.pixmap is None:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load an image to begin")
            return
        pix_w = self.pixmap.width()
        pix_h = self.pixmap.height()
        if pix_w <= 0 or pix_h <= 0:
            return
        scale = min(self.width() / pix_w, self.height() / pix_h)
        draw_w = int(pix_w * scale)
        draw_h = int(pix_h * scale)
        offset_x = (self.width() - draw_w) // 2
        offset_y = (self.height() - draw_h) // 2
        target = QRect(offset_x, offset_y, draw_w, draw_h)
        painter.drawPixmap(target, self.pixmap)
