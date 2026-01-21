"""
Canvas widget for displaying images and mask annotation tools.
"""

import os
import numpy as np
import cv2
import tifffile
from PyQt6.QtWidgets import QWidget, QFileDialog, QMessageBox, QScrollBar, QStyle
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
        self.show_contour_only = True
        self._undo_stack = []
        self._redo_stack = []
        self._mask_touched = False

        self.tool = "select"
        self.brush_radius = 4
        self.fill_roi = False
        self._drawing = False
        self._last_point = None
        self._poly_points = []
        self._freehand_points = []
        self._last_outline = []
        self._cursor_pos = None

        self.scale = 1.0
        self.min_scale = 0.1
        self.max_scale = 10.0
        self._scroll_x = 0
        self._scroll_y = 0

        self.h_scrollbar = QScrollBar(Qt.Orientation.Horizontal, self)
        self.v_scrollbar = QScrollBar(Qt.Orientation.Vertical, self)
        self.h_scrollbar.valueChanged.connect(self._on_h_scroll)
        self.v_scrollbar.valueChanged.connect(self._on_v_scroll)
        self.h_scrollbar.hide()
        self.v_scrollbar.hide()
        self.h_scrollbar.raise_()
        self.v_scrollbar.raise_()

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
        if not self.has_mask_data():
            QMessageBox.information(self, "No mask", "Run segmentation or annotate before saving a mask.")
            return
        default_path = ""
        if self.image_path:
            default_dir = os.path.dirname(self.image_path)
            default_name = f"{os.path.splitext(os.path.basename(self.image_path))[0]}.png"
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
        if not self.has_mask_data():
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
        self._mask_touched = False
        self._reset_history()
        self._reset_view()
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
        if self.image is None:
            self.mask = None
            self.mask_pixmap = None
            self._last_outline = []
            self._mask_touched = False
            self._reset_history()
            self.update()
            return
        height, width = self.image.shape[:2]
        self.mask = np.zeros((height, width), dtype=np.float32)
        self._refresh_mask_pixmap()
        self._last_outline = []
        self._push_history()
        self._mask_touched = True
        self.update()

    def clear_image(self):
        if self.image is None:
            return
        self.image = None
        self.pixmap = None
        self.image_path = None
        self.clear_mask()
        self._last_outline = []
        self._reset_view()
        self.update()

    def set_mask(self, mask):
        if self.image is None:
            return
        self.mask = self._normalize_mask(mask)
        self._refresh_mask_pixmap()
        self._mask_touched = True
        self._reset_history()
        self.update()

    def set_mask_opacity(self, opacity):
        self.mask_opacity = max(0.0, min(1.0, float(opacity)))
        if self.mask is not None:
            self._refresh_mask_pixmap()
        self.update()

    def set_contour_only(self, enabled):
        self.show_contour_only = bool(enabled)
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
        self.show_contour_only = not self.fill_roi
        if self.fill_roi and self._last_outline:
            self._ensure_mask()
            points = np.array(
                [[p.x(), p.y()] for p in self._last_outline],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
            cv2.fillPoly(self.mask, [points], 1.0)
            self._push_history()
            self._mask_touched = True
        if self.mask is not None:
            self._refresh_mask_pixmap()
        self.update()

    def fit_to_window(self):
        if self.pixmap is None:
            return
        pix_w = self.pixmap.width()
        pix_h = self.pixmap.height()
        if pix_w <= 0 or pix_h <= 0:
            return
        scale = min(self.width() / pix_w, self.height() / pix_h)
        self.scale = max(self.min_scale, min(self.max_scale, scale))
        self._scroll_x = 0
        self._scroll_y = 0
        self._update_scrollbars()
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
        if self.show_contour_only:
            binary = (self.mask > 0).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            contour_color = int(255 * self.mask_opacity)
            cv2.drawContours(rgba, contours, -1, (0, 255, 0, contour_color), 1)
        else:
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
            self._reset_history()

    def _current_view(self):
        if self.pixmap is None:
            return 1.0, 0, 0
        pix_w = self.pixmap.width()
        pix_h = self.pixmap.height()
        if pix_w <= 0 or pix_h <= 0:
            return 1.0, 0, 0
        scale = self.scale
        draw_w = int(pix_w * scale)
        draw_h = int(pix_h * scale)
        if draw_w <= self.width():
            offset_x = (self.width() - draw_w) // 2
        else:
            offset_x = -self._scroll_x
        if draw_h <= self.height():
            offset_y = (self.height() - draw_h) // 2
        else:
            offset_y = -self._scroll_y
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
            self._push_history()
            self._mask_touched = True
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
        self._cursor_pos = self._screen_to_image(event.position().toPoint())
        if not self._drawing:
            if self.tool in ("brush", "eraser"):
                self.update()
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
                    self._push_history()
                    self._mask_touched = True
                elif len(self._freehand_points) > 1:
                    self._last_outline = list(self._freehand_points)
                self._freehand_points = []
            elif self.tool in ("brush", "eraser"):
                if self._drawing:
                    self._push_history()
                    self._mask_touched = True
            self._drawing = False
            self._last_point = None
            self.update()

    def leaveEvent(self, event):
        self._cursor_pos = None
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

        if self._cursor_pos is not None and self.tool in ("brush", "eraser"):
            scale, _, _ = self._current_view()
            radius = max(1, int(self.brush_radius * scale))
            center = self._image_to_screen(self._cursor_pos)
            painter.setPen(QPen(QColor(0, 255, 0), 1, Qt.PenStyle.SolidLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, radius, radius)

    def undo(self):
        if len(self._undo_stack) <= 1:
            return
        current = self._undo_stack.pop()
        self._redo_stack.append(current)
        self.mask = np.copy(self._undo_stack[-1])
        self._refresh_mask_pixmap()
        self.update()

    def redo(self):
        if not self._redo_stack:
            return
        state = self._redo_stack.pop()
        self._undo_stack.append(np.copy(state))
        self.mask = np.copy(state)
        self._refresh_mask_pixmap()
        self.update()

    def _reset_history(self):
        self._undo_stack = []
        self._redo_stack = []
        if self.mask is not None:
            self._undo_stack.append(np.copy(self.mask))

    def _push_history(self):
        if self.mask is None:
            return
        self._undo_stack.append(np.copy(self.mask))
        self._redo_stack = []

    def has_mask_data(self):
        return self.mask is not None and self._mask_touched

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scrollbars()

    def wheelEvent(self, event):
        if self.pixmap is None:
            return
        delta = event.angleDelta().y()
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            factor = 1.2 if delta > 0 else 1 / 1.2
            self._apply_zoom(factor)
        elif modifiers & Qt.KeyboardModifier.ShiftModifier:
            if self.h_scrollbar.isVisible():
                self._scroll_horizontal(-delta)
        else:
            self._scroll_vertical(-delta)
        event.accept()

    def _reset_view(self):
        self.scale = 1.0
        self._scroll_x = 0
        self._scroll_y = 0
        if self.pixmap is not None:
            pix_w = self.pixmap.width()
            pix_h = self.pixmap.height()
            if pix_w > 0 and pix_h > 0:
                scale = min(self.width() / pix_w, self.height() / pix_h)
                self.scale = max(self.min_scale, min(self.max_scale, scale))
        self._update_scrollbars()

    def _apply_zoom(self, factor):
        if self.pixmap is None:
            return
        self.scale = max(self.min_scale, min(self.max_scale, self.scale * factor))
        self._update_scrollbars()
        self.update()

    def _update_scrollbars(self):
        if self.pixmap is None:
            self.h_scrollbar.hide()
            self.v_scrollbar.hide()
            return
        pix_w = self.pixmap.width()
        pix_h = self.pixmap.height()
        draw_w = int(pix_w * self.scale)
        draw_h = int(pix_h * self.scale)

        bar_size = self.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
        h_visible = draw_w > self.width()
        v_visible = draw_h > self.height()

        h_height = bar_size if h_visible else 0
        v_width = bar_size if v_visible else 0

        if h_visible:
            self.h_scrollbar.setRange(0, max(0, draw_w - self.width()))
            self.h_scrollbar.setPageStep(max(1, self.width()))
            self.h_scrollbar.setValue(min(self.h_scrollbar.value(), self.h_scrollbar.maximum()))
            self.h_scrollbar.setGeometry(0, self.height() - h_height, self.width() - v_width, h_height)
            self.h_scrollbar.show()
        else:
            self.h_scrollbar.setValue(0)
            self._scroll_x = 0
            self.h_scrollbar.hide()

        if v_visible:
            self.v_scrollbar.setRange(0, max(0, draw_h - self.height()))
            self.v_scrollbar.setPageStep(max(1, self.height()))
            self.v_scrollbar.setValue(min(self.v_scrollbar.value(), self.v_scrollbar.maximum()))
            self.v_scrollbar.setGeometry(self.width() - v_width, 0, v_width, self.height() - h_height)
            self.v_scrollbar.show()
        else:
            self.v_scrollbar.setValue(0)
            self._scroll_y = 0
            self.v_scrollbar.hide()

        self._scroll_x = self.h_scrollbar.value()
        self._scroll_y = self.v_scrollbar.value()
        self.update()

    def _scroll_horizontal(self, delta):
        if not self.h_scrollbar.isVisible():
            return
        step = max(1, int(abs(delta) * 0.25))
        if delta > 0:
            self.h_scrollbar.setValue(self.h_scrollbar.value() + step)
        else:
            self.h_scrollbar.setValue(self.h_scrollbar.value() - step)

    def _scroll_vertical(self, delta):
        if not self.v_scrollbar.isVisible():
            return
        step = max(1, int(abs(delta) * 0.25))
        if delta > 0:
            self.v_scrollbar.setValue(self.v_scrollbar.value() + step)
        else:
            self.v_scrollbar.setValue(self.v_scrollbar.value() - step)

    def _on_h_scroll(self, value):
        self._scroll_x = value
        self.update()

    def _on_v_scroll(self, value):
        self._scroll_y = value
        self.update()
