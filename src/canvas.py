"""
Canvas widget for displaying images and mask annotation tools.
"""

import os
import numpy as np
import cv2
import tifffile
from PyQt6.QtWidgets import QWidget, QFileDialog, QMessageBox
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QRect, QPoint


class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.pixmap = None
        self.image_path = None
        self.mask = None
        self.mask_pixmap = None
        self.mask_opacity = 0.5

        self.tool = "select"
        self.brush_radius = 4
        self.fill_roi = False
        self._drawing = False
        self._last_point = None
        self._poly_points = []
        self._freehand_points = []
        self._last_outline = []

        self.setMouseTracking(True)

    def load_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if path:
            self.load_image(path)

    def load_mask_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Mask",
            "",
            "Mask Files (*.tif *.tiff *.png *.bmp *.jpg *.jpeg)",
        )
        if path:
            self.load_mask(path)

    def save_mask_dialog(self):
        if self.mask is None:
            QMessageBox.information(self, "No mask", "Run segmentation before saving a mask.")
            return
        default_path = ""
        if self.image_path:
            default_dir = os.path.dirname(self.image_path)
            default_name = os.path.basename(self.image_path)
            default_path = os.path.join(default_dir, default_name)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mask",
            default_path,
            "Mask Files (*.tif *.tiff *.png)",
        )
        if path:
            self.save_mask(path)

    def save_mask(self, file_path):
        if self.mask is None:
            raise ValueError("No mask available to save.")
        mask_uint8 = (self.mask >= 0.5).astype(np.uint8) * 255
        lower_path = file_path.lower()
        if lower_path.endswith((".tif", ".tiff")):
            tifffile.imwrite(file_path, mask_uint8)
        else:
            cv2.imwrite(file_path, mask_uint8)

    def load_image(self, file_path):
        try:
            raw = self._read_image(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        self.image = raw
        self.pixmap = self._to_pixmap(raw)
        self.image_path = file_path
        height, width = self.image.shape[:2]
        self.mask = np.zeros((height, width), dtype=np.float32)
        self._refresh_mask_pixmap()
        self._last_outline = []
        self.update()

    def load_mask(self, file_path):
        try:
            mask = self._read_mask(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        if self.image is None:
            QMessageBox.information(self, "No image", "Load an image before loading a mask.")
            return
        if mask.shape[:2] != self.image.shape[:2]:
            mask = cv2.resize(
                mask, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        self.set_mask(mask)

    def clear_mask(self):
        if self.mask is None:
            return
        self.mask = None
        self.mask_pixmap = None
        self._last_outline = []
        self.update()

    def clear_image(self):
        if self.image is None:
            return
        self.image = None
        self.pixmap = None
        self.image_path = None
        self.clear_mask()
        self._last_outline = []
        self.update()

    def set_mask(self, mask):
        if self.image is None:
            return
        self.mask = self._normalize_mask(mask)
        self._refresh_mask_pixmap()
        self.update()

    def set_mask_opacity(self, opacity):
        self.mask_opacity = max(0.0, min(1.0, float(opacity)))
        if self.mask is not None:
            self._refresh_mask_pixmap()
        self.update()

    def set_tool(self, tool_name):
        self.tool = tool_name
        self._drawing = False
        self._poly_points = []
        self._freehand_points = []
        self.update()

    def set_brush_radius(self, radius):
        self.brush_radius = max(1, int(radius))

    def set_fill_roi(self, enabled):
        self.fill_roi = bool(enabled)
        if self.fill_roi and self._last_outline:
            self._ensure_mask()
            points = np.array(
                [[p.x(), p.y()] for p in self._last_outline],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
            cv2.fillPoly(self.mask, [points], 1.0)
            self._refresh_mask_pixmap()
            self.update()

    def fit_to_window(self):
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

    def _read_mask(self, file_path):
        lower_path = file_path.lower()
        if lower_path.endswith((".tif", ".tiff")):
            mask = tifffile.imread(file_path)
        else:
            mask = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise ValueError(f"Unsupported mask format: {file_path}")
        if mask is None:
            raise ValueError(f"Unsupported mask format: {file_path}")
        if mask.ndim == 3:
            if mask.shape[-1] in (3, 4):
                mask = mask[:, :, 0]
            else:
                mask = mask[0]
        while mask.ndim > 2:
            mask = mask[0]
        return mask

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

    def _normalize_mask(self, mask):
        data = np.asarray(mask)
        if data.ndim == 4:
            data = data[0]
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        if data.ndim == 3 and data.shape[2] == 1:
            data = data[:, :, 0]
        if data.ndim != 2:
            raise ValueError(f"Unsupported mask layout: {data.shape}")
        if self.image is not None and data.shape[:2] != self.image.shape[:2]:
            data = cv2.resize(
                data, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        if data.dtype != np.float32:
            data = data.astype(np.float32, copy=False)
        if data.max() > 1.0:
            data = data / 255.0
        return np.clip(data, 0.0, 1.0)

    def _refresh_mask_pixmap(self):
        if self.mask is None:
            self.mask_pixmap = None
            return
        height, width = self.mask.shape
        alpha = (self.mask * 255.0 * self.mask_opacity).astype(np.uint8)
        rgb = (self.mask * 255.0).astype(np.uint8)
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[:, :, 0] = rgb
        rgba[:, :, 1] = rgb
        rgba[:, :, 2] = rgb
        rgba[:, :, 3] = alpha
        bytes_per_line = rgba.strides[0]
        q_image = QImage(
            rgba.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888
        )
        self.mask_pixmap = QPixmap.fromImage(q_image)

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

    def _ensure_mask(self):
        if self.image is None:
            return
        if self.mask is None:
            height, width = self.image.shape[:2]
            self.mask = np.zeros((height, width), dtype=np.float32)

    def _current_view(self):
        if self.pixmap is None:
            return 1.0, 0, 0
        pix_w = self.pixmap.width()
        pix_h = self.pixmap.height()
        if pix_w <= 0 or pix_h <= 0:
            return 1.0, 0, 0
        scale = min(self.width() / pix_w, self.height() / pix_h)
        draw_w = int(pix_w * scale)
        draw_h = int(pix_h * scale)
        offset_x = (self.width() - draw_w) // 2
        offset_y = (self.height() - draw_h) // 2
        return scale, offset_x, offset_y

    def _screen_to_image(self, pos):
        scale, offset_x, offset_y = self._current_view()
        if scale <= 0:
            return None
        x = int((pos.x() - offset_x) / scale)
        y = int((pos.y() - offset_y) / scale)
        if self.image is None:
            return None
        height, width = self.image.shape[:2]
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        return QPoint(x, y)

    def _image_to_screen(self, point):
        scale, offset_x, offset_y = self._current_view()
        x = int(offset_x + point.x() * scale)
        y = int(offset_y + point.y() * scale)
        return QPoint(x, y)

    def _brush_thickness(self):
        return max(1, int(self.brush_radius))

    def _draw_point(self, point, value):
        if self.mask is None:
            return
        cv2.circle(self.mask, (point.x(), point.y()), self.brush_radius, value, -1)

    def _draw_line(self, start, end, value):
        if self.mask is None:
            return
        cv2.line(
            self.mask,
            (start.x(), start.y()),
            (end.x(), end.y()),
            value,
            self._brush_thickness(),
        )

    def _finish_polyline(self):
        if len(self._poly_points) < 2:
            self._poly_points = []
            self.update()
            return
        if self.fill_roi:
            self._ensure_mask()
            points = np.array(
                [[p.x(), p.y()] for p in self._poly_points],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
            cv2.fillPoly(self.mask, [points], 1.0)
            self._refresh_mask_pixmap()
        else:
            self._last_outline = list(self._poly_points)
        self._poly_points = []
        self.update()

    def mousePressEvent(self, event):
        if self.image is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            point = self._screen_to_image(event.position().toPoint())
            if point is None:
                return
            if self.tool == "freehand":
                self._drawing = True
                self._freehand_points = [point]
                self._last_outline = []
                self.update()
            elif self.tool == "brush":
                self._ensure_mask()
                self._drawing = True
                self._last_point = point
                self._draw_point(point, 1.0)
                self._refresh_mask_pixmap()
                self.update()
            elif self.tool == "eraser":
                self._ensure_mask()
                self._drawing = True
                self._last_point = point
                self._draw_point(point, 0.0)
                self._refresh_mask_pixmap()
                self.update()
            elif self.tool == "polyline":
                self._poly_points.append(point)
                self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            if self.tool == "polyline":
                self._finish_polyline()

    def mouseMoveEvent(self, event):
        if self.image is None:
            return
        if not self._drawing:
            return
        point = self._screen_to_image(event.position().toPoint())
        if point is None:
            return
        if self.tool == "freehand":
            self._freehand_points.append(point)
            self.update()
            return
        if self._last_point is None:
            return
        if self.tool == "brush":
            self._draw_line(self._last_point, point, 1.0)
        elif self.tool == "eraser":
            self._draw_line(self._last_point, point, 0.0)
        self._last_point = point
        self._refresh_mask_pixmap()
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.tool == "freehand":
                if self.fill_roi and len(self._freehand_points) > 2:
                    self._ensure_mask()
                    points = np.array(
                        [[p.x(), p.y()] for p in self._freehand_points],
                        dtype=np.int32,
                    ).reshape((-1, 1, 2))
                    cv2.fillPoly(self.mask, [points], 1.0)
                    self._refresh_mask_pixmap()
                elif len(self._freehand_points) > 1:
                    self._last_outline = list(self._freehand_points)
                self._freehand_points = []
            self._drawing = False
            self._last_point = None
            self.update()

    def mouseDoubleClickEvent(self, event):
        if self.tool == "polyline":
            self._finish_polyline()

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
        scale, offset_x, offset_y = self._current_view()
        draw_w = int(pix_w * scale)
        draw_h = int(pix_h * scale)
        target = QRect(offset_x, offset_y, draw_w, draw_h)
        painter.drawPixmap(target, self.pixmap)
        if self.mask_pixmap is not None:
            painter.drawPixmap(target, self.mask_pixmap)

        if self.tool == "polyline" and self._poly_points:
            painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine))
            for point in self._poly_points:
                screen_point = self._image_to_screen(point)
                painter.drawEllipse(screen_point, 4, 4)
            for idx in range(len(self._poly_points) - 1):
                start = self._image_to_screen(self._poly_points[idx])
                end = self._image_to_screen(self._poly_points[idx + 1])
                painter.drawLine(start, end)
        if self.tool == "freehand" and self._freehand_points:
            painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine))
            for idx in range(len(self._freehand_points) - 1):
                start = self._image_to_screen(self._freehand_points[idx])
                end = self._image_to_screen(self._freehand_points[idx + 1])
                painter.drawLine(start, end)
        if self._last_outline:
            painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine))
            for idx in range(len(self._last_outline) - 1):
                start = self._image_to_screen(self._last_outline[idx])
                end = self._image_to_screen(self._last_outline[idx + 1])
                painter.drawLine(start, end)
            if len(self._last_outline) > 2:
                start = self._image_to_screen(self._last_outline[-1])
                end = self._image_to_screen(self._last_outline[0])
                painter.drawLine(start, end)
