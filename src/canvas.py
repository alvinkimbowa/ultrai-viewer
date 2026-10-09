"""
Canvas widget for displaying images and mask annotation tools.
"""

import os
import numpy as np
import cv2
import tifffile
from PyQt6.QtWidgets import QWidget, QFileDialog, QMessageBox, QScrollBar, QStyle, QMenu
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPolygonF
from PyQt6.QtCore import Qt, QRect, QPoint, QPointF, pyqtSignal

from .measurements import (
    bottom_surface_paths,
    cartilage_edges,
    region_columns,
    suggest_centre_x,
    surface_point,
)


REGION_COLORS = {
    "lateral": (255, 170, 0),
    "intercondylar": (255, 60, 220),
    "medial": (120, 190, 255),
}
WHOLE_MASK_COLOR = (255, 255, 180)
CENTRE_MARKER_COLOR = (255, 235, 0)
REGION_LABELS = {"lateral": "Lateral", "intercondylar": "Notch", "medial": "Medial"}


class Canvas(QWidget):
    image_loaded = pyqtSignal(str)
    navigation_requested = pyqtSignal(int)
    mask_changed = pyqtSignal()
    # Carries the box the image had before the drawn one replaced it, or None.
    roi_drawn = pyqtSignal(object)
    roi_changed = pyqtSignal()
    delete_requested = pyqtSignal(str)

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
        self.fill_mask = False
        self._drawing = False
        self._last_point = None
        self._poly_points = []
        self._freehand_points = []
        self._last_outline = []
        self._cursor_pos = None

        # Calibration lines in image pixels: "h" holds the y positions of the two
        # horizontal lines, "v" the x positions of the two vertical lines. They
        # outlive a calibration so the next one starts from the same places.
        # Per-region view splits the mask into lateral, intercondylar and medial
        # around a centre point on the cartilage's top surface. A centre_x of None
        # means the suggested centre is used; a number is a position the user set.
        self.region_view = False
        self.knee_side = "right"
        self.centre_x = None
        self._centre_drag = False
        self._region_overlay = None

        # The ROI of the image on screen, or None: "v" holds the x positions of its
        # two vertical edges and "h" the y positions of its two horizontal edges, in
        # image pixels. The mask outside it is hidden, not saved and not measured,
        # but stays in self.mask.
        self._roi = None
        self._roi_drag = None
        # While the whole ROI is being dragged: where the press landed, the edge
        # positions at that moment, and whether the pointer has moved far enough
        # for the press to count as a drag instead of a click.
        self._roi_move = None
        # Opposite corners (x, y) of a box being drawn with the ROI tool.
        self._roi_draft = None
        # What a click with the Select tool picked for deletion: "roi", "mask" or None.
        self.selection = None

        self._calibrating = False
        self._calibration_lines = None
        self._calibration_drag = None

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
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def load_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if path:
            self.load_image(path)

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
        mask_uint8 = (self.visible_mask() >= 0.5).astype(np.uint8) * 255
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
        self._load_image_data(raw, file_path)

    def load_image_array(self, image, source_name=""):
        raw = np.asarray(image)
        self._load_image_data(raw, source_name)

    def _load_image_data(self, raw, source_name):
        self.image = raw
        self.pixmap = self._to_pixmap(raw)
        self.image_path = source_name or None
        height, width = self.image.shape[:2]
        self.mask = np.zeros((height, width), dtype=np.float32)
        self.centre_x = None
        self._roi = None
        self._roi_drag = None
        self._roi_move = None
        self._roi_draft = None
        self.selection = None
        self._refresh_mask_pixmap()
        self._last_outline = []
        self._mask_touched = False
        self._reset_history()
        self._reset_view()
        self.update()
        self.image_loaded.emit(source_name)

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
        if self.selection == "mask":
            self.selection = None
        if self.image is None:
            self.mask = None
            self.mask_pixmap = None
            self._last_outline = []
            self._mask_touched = False
            self._reset_history()
            self.update()
            self.mask_changed.emit()
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
        self.image_loaded.emit("")

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
        self._roi_draft = None
        self.selection = None
        if tool_name == "roi":
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()
        self.update()

    def set_brush_radius(self, radius):
        self.brush_radius = max(1, int(radius))

    def set_fill_mask(self, enabled):
        self.fill_mask = bool(enabled)
        self.show_contour_only = not self.fill_mask
        if self.fill_mask and self._last_outline:
            self.commit_pending_outline_to_mask()
        if self.mask is not None:
            self._refresh_mask_pixmap()
        self.update()

    def commit_pending_outline_to_mask(self):
        if len(self._last_outline) < 2:
            return False
        self._ensure_mask()
        points = np.array(
            [[p.x(), p.y()] for p in self._last_outline],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.fillPoly(self.mask, [points], 1.0)
        self._push_history()
        self._mask_touched = True
        self._last_outline = []
        if self.mask is not None:
            self._refresh_mask_pixmap()
        self.update()
        return True

    def set_roi(self, box):
        """Give the image on screen the ROI box (left, right, top, bottom), or None."""
        self._roi_drag = None
        self._roi_move = None
        if box is None or self.image is None:
            self._roi = None
            if self.selection == "roi":
                self.selection = None
        else:
            height, width = self.image.shape[:2]
            left, right, top, bottom = box
            self._roi = {
                "v": [max(0.0, min(float(width), float(x))) for x in (left, right)],
                "h": [max(0.0, min(float(height), float(y))) for y in (top, bottom)],
            }
        self._refresh_mask_pixmap()
        self.update()

    def roi_box(self):
        """Pixel box (left, right, top, bottom) of the ROI, or None when there is none."""
        if self._roi is None or self.image is None:
            return None
        left, right = sorted(int(round(x)) for x in self._roi["v"])
        top, bottom = sorted(int(round(y)) for y in self._roi["h"])
        return left, right, top, bottom

    def visible_mask(self):
        """The mask as shown, saved and measured: empty outside the ROI."""
        if self.mask is None:
            return None
        box = self.roi_box()
        if box is None:
            return self.mask
        left, right, top, bottom = box
        visible = np.zeros_like(self.mask)
        visible[top:bottom, left:right] = self.mask[top:bottom, left:right]
        return visible

    def _roi_screen_rect(self):
        scale, offset_x, offset_y = self._current_view()
        left = int(offset_x + min(self._roi["v"]) * scale)
        right = int(offset_x + max(self._roi["v"]) * scale)
        top = int(offset_y + min(self._roi["h"]) * scale)
        bottom = int(offset_y + max(self._roi["h"]) * scale)
        return left, right, top, bottom

    def _roi_grips_at(self, pos):
        """The ROI edges under a widget position as (axis, index) pairs.

        One edge along a side, two (a vertical and a horizontal one) at a corner,
        none elsewhere.
        """
        if self._roi is None:
            return []
        scale, offset_x, offset_y = self._current_view()
        left, right, top, bottom = self._roi_screen_rect()
        reach = 6
        grips = []
        for axis, across, along, offset, low, high in (
            ("v", pos.x(), pos.y(), offset_x, top, bottom),
            ("h", pos.y(), pos.x(), offset_y, left, right),
        ):
            if not low - reach <= along <= high + reach:
                continue
            distances = [
                abs(across - (offset + value * scale)) for value in self._roi[axis]
            ]
            index = int(np.argmin(distances))
            if distances[index] <= reach:
                grips.append((axis, index))
        return grips

    def _set_grip_cursor(self, grips):
        if len(grips) < 2:
            self._set_line_cursor(grips[0] if grips else None)
            return
        (_, v_index), (_, h_index) = grips
        at_left = self._roi["v"][v_index] == min(self._roi["v"])
        at_top = self._roi["h"][h_index] == min(self._roi["h"])
        if at_left == at_top:
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)

    def _inside_roi(self, pos):
        if self._roi is None:
            return False
        left, right, top, bottom = self._roi_screen_rect()
        return left <= pos.x() <= right and top <= pos.y() <= bottom

    def _move_roi(self, pos):
        """Slide the whole ROI with the pointer, keeping its size and staying in the image."""
        move = self._roi_move
        if not move["moved"]:
            if (pos - move["press"]).manhattanLength() < 4:
                return
            move["moved"] = True
            self.selection = "roi"
        height, width = self.image.shape[:2]
        scale = self._current_view()[0]
        for axis, delta, extent in (
            ("v", (pos.x() - move["press"].x()) / scale, width),
            ("h", (pos.y() - move["press"].y()) / scale, height),
        ):
            low, high = min(move[axis]), max(move[axis])
            delta = max(-low, min(float(extent) - high, delta))
            self._roi[axis] = [value + delta for value in move[axis]]
        self._refresh_mask_pixmap()
        self.update()

    def _image_position(self, pos):
        """A widget position as image (x, y), pulled inside the image."""
        scale, offset_x, offset_y = self._current_view()
        height, width = self.image.shape[:2]
        return (
            max(0.0, min(float(width), (pos.x() - offset_x) / scale)),
            max(0.0, min(float(height), (pos.y() - offset_y) / scale)),
        )

    def _mask_at(self, pos):
        """True when the mask as shown covers the image pixel under a widget position."""
        mask = self.visible_mask()
        point = self._screen_to_image(pos)
        if mask is None or point is None:
            return False
        height, width = mask.shape[:2]
        if not (0 <= point.x() < width and 0 <= point.y() < height):
            return False
        return bool(mask[point.y(), point.x()] > 0)

    def _outline_at(self, pos):
        """True when a widget position is on or inside the outline not yet filled in."""
        if len(self._last_outline) < 2:
            return False
        x, y = self._image_position(pos)
        points = np.array(
            [[p.x(), p.y()] for p in self._last_outline], dtype=np.float32
        ).reshape((-1, 1, 2))
        scale = self._current_view()[0]
        return cv2.pointPolygonTest(points, (x, y), True) >= -6.0 / scale

    def deletable_at(self, pos):
        """What a right-click at a widget position can delete, nearest thing first."""
        kinds = []
        if self.image is None:
            return kinds
        if self._outline_at(pos):
            kinds.append("outline")
        if self._mask_at(pos):
            kinds.append("mask")
        if self._roi is not None:
            if self._inside_roi(pos) or self._roi_grips_at(pos):
                kinds.append("roi")
        return kinds

    def request_delete(self, kind):
        if kind == "outline":
            self._last_outline = []
            self.update()
        else:
            self.delete_requested.emit(kind)

    def set_region_view(self, enabled):
        self.region_view = bool(enabled)
        self._refresh_mask_pixmap()
        self.update()

    def set_knee_side(self, knee_side):
        self.knee_side = knee_side
        self._refresh_mask_pixmap()
        self.update()

    def set_centre_x(self, centre_x):
        self.centre_x = None if centre_x is None else float(centre_x)
        self._refresh_mask_pixmap()
        self.update()

    def _build_region_overlay(self, binary):
        """Work out what is drawn over the mask: the bone line, and the regions."""
        if not binary.any():
            return None
        width = binary.shape[1]
        if self.region_view:
            centre_x = self.centre_x
            if centre_x is None:
                centre_x = suggest_centre_x(binary)
            spans = region_columns(width, centre_x, self.knee_side, self.roi_box())
            centre = surface_point(binary, centre_x)
        else:
            spans = {"whole": (0, width)}
            centre = None
        # The line drawn is the path the length is measured along, cut where it
        # passes from one region into the next.
        bone_lines = []
        for pixels, _ in bottom_surface_paths(binary):
            for name, (start, stop) in spans.items():
                inside = np.nonzero((pixels[:, 0] >= start) & (pixels[:, 0] < stop))[0]
                if len(inside) == 0:
                    continue
                breaks = np.nonzero(np.diff(inside) > 1)[0] + 1
                for run in np.split(inside, breaks):
                    # One pixel further, so the colours of two regions meet.
                    last = min(run[-1] + 1, len(pixels) - 1)
                    run = pixels[run[0] : last + 1]
                    bone_lines.append((name, run[:, 0] + 0.5, run[:, 1] + 0.5))
        return {"spans": spans, "centre": centre, "bone_lines": bone_lines}

    def _centre_marker_hit(self, pos):
        overlay = self._region_overlay
        if not self.region_view or overlay is None or overlay["centre"] is None:
            return False
        scale, offset_x, offset_y = self._current_view()
        centre_x, centre_y = overlay["centre"]
        return (
            abs(pos.x() - (offset_x + centre_x * scale)) <= 10
            and abs(pos.y() - (offset_y + centre_y * scale)) <= 10
        )

    def start_calibration(self):
        if self.image is None:
            return False
        height, width = self.image.shape[:2]
        if self._calibration_lines is None:
            self._calibration_lines = {
                "h": [height / 3.0, 2.0 * height / 3.0],
                "v": [width / 3.0, 2.0 * width / 3.0],
            }
        lines = self._calibration_lines
        lines["h"] = [max(0.0, min(float(height), y)) for y in lines["h"]]
        lines["v"] = [max(0.0, min(float(width), x)) for x in lines["v"]]
        self._calibrating = True
        self._calibration_drag = None
        self.update()
        return True

    def stop_calibration(self):
        self._calibrating = False
        self._calibration_drag = None
        self.unsetCursor()
        self.update()

    def is_calibrating(self):
        return self._calibrating

    def calibration_lines(self):
        if self._calibration_lines is None:
            return None
        return {axis: list(values) for axis, values in self._calibration_lines.items()}

    def set_calibration_lines(self, lines):
        self._calibration_lines = {
            "h": [float(v) for v in lines["h"]],
            "v": [float(v) for v in lines["v"]],
        }
        self.update()

    def calibration_gaps(self):
        """Pixel distance between the horizontal pair and between the vertical pair."""
        lines = self._calibration_lines
        if lines is None:
            return 0.0, 0.0
        return abs(lines["h"][1] - lines["h"][0]), abs(lines["v"][1] - lines["v"][0])

    def _calibration_line_at(self, pos):
        scale, offset_x, offset_y = self._current_view()
        grab_distance = 6
        best = None
        for axis, screen_value, offset in (
            ("h", pos.y(), offset_y),
            ("v", pos.x(), offset_x),
        ):
            for index, value in enumerate(self._calibration_lines[axis]):
                distance = abs(screen_value - (offset + value * scale))
                if distance <= grab_distance and (best is None or distance < best[0]):
                    best = (distance, axis, index)
        return None if best is None else (best[1], best[2])

    def _move_calibration_line(self, axis, index, pos):
        scale, offset_x, offset_y = self._current_view()
        height, width = self.image.shape[:2]
        if axis == "h":
            value = max(0.0, min(float(height), (pos.y() - offset_y) / scale))
        else:
            value = max(0.0, min(float(width), (pos.x() - offset_x) / scale))
        self._calibration_lines[axis][index] = value
        self.update()

    def _set_line_cursor(self, line):
        if line is None:
            self.unsetCursor()
        elif line[0] == "h":
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeHorCursor)

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
            display = np.ascontiguousarray(self._normalize_gray(image))
            height, width = display.shape
            bytes_per_line = display.strides[0]
            q_image = QImage(
                display.tobytes(),
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_Grayscale8,
            ).copy()
            return QPixmap.fromImage(q_image)
        if image.ndim == 3 and image.shape[2] in (3, 4):
            if image.shape[2] == 4:
                image = image[:, :, :3]
            display = np.ascontiguousarray(self._as_uint8(image))
            height, width, _ = display.shape
            bytes_per_line = display.strides[0]
            q_image = QImage(
                display.tobytes(),
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_RGB888,
            ).copy()
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
        self.mask_changed.emit()
        if self.mask is None:
            self.mask_pixmap = None
            self._region_overlay = None
            return
        height, width = self.mask.shape
        shown = self.visible_mask()
        binary = (shown > 0).astype(np.uint8)
        self._region_overlay = self._build_region_overlay(binary)
        if self.region_view and self._region_overlay is not None:
            colored_spans = [
                (REGION_COLORS[name], span)
                for name, span in self._region_overlay["spans"].items()
            ]
        else:
            colored_spans = [(WHOLE_MASK_COLOR, (0, width))]
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        for color, (start, stop) in colored_spans:
            if stop <= start:
                continue
            if self.show_contour_only:
                region = np.zeros_like(binary)
                region[:, start:stop] = binary[:, start:stop]
                contours, _ = cv2.findContours(
                    region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(
                    rgba,
                    contours,
                    -1,
                    (*color, int(255 * self.mask_opacity)),
                    self._roi_outline_thickness_px(),
                )
            else:
                level = shown[:, start:stop]
                for channel in range(3):
                    rgba[:, start:stop, channel] = (level * color[channel]).astype(
                        np.uint8
                    )
                rgba[:, start:stop, 3] = (level * 255.0 * self.mask_opacity).astype(
                    np.uint8
                )
        rgba = np.ascontiguousarray(rgba)
        bytes_per_line = rgba.strides[0]
        q_image = QImage(
            rgba.tobytes(),
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGBA8888,
        ).copy()
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

    def _roi_outline_thickness_px(self):
        if self.image is None:
            return 1
        height, width = self.image.shape[:2]
        base = min(height, width)
        thickness = (base + 511) // 512
        return max(1, min(6, thickness))

    def _roi_outline_pen_width(self):
        thickness_px = self._roi_outline_thickness_px()
        width = int(round(thickness_px * max(0.6, self.scale)))
        return max(1, min(8, width))

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
        if self.fill_mask:
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
        if self._calibrating:
            if event.button() == Qt.MouseButton.LeftButton:
                self._calibration_drag = self._calibration_line_at(
                    event.position().toPoint()
                )
            return
        if event.button() == Qt.MouseButton.LeftButton:
            point = self._screen_to_image(event.position().toPoint())
            if point is None:
                return
            if self.tool == "select":
                pos = event.position().toPoint()
                self._centre_drag = self._centre_marker_hit(pos)
                if not self._centre_drag:
                    self._roi_drag = self._roi_grips_at(pos) or None
                    if self._roi_drag is not None:
                        self.selection = "roi"
                    elif self._inside_roi(pos):
                        self._roi_move = {
                            "press": pos,
                            "v": list(self._roi["v"]),
                            "h": list(self._roi["h"]),
                            "moved": False,
                        }
                    elif self._mask_at(pos):
                        self.selection = "mask"
                    else:
                        self.selection = None
                    self.update()
            elif self.tool == "roi":
                # On the box already there the ROI tool adjusts it; anywhere else
                # it starts a box that replaces it.
                pos = event.position().toPoint()
                self._roi_drag = self._roi_grips_at(pos) or None
                if self._roi_drag is not None:
                    self.selection = "roi"
                elif self._inside_roi(pos):
                    self._roi_move = {
                        "press": pos,
                        "v": list(self._roi["v"]),
                        "h": list(self._roi["h"]),
                        "moved": False,
                    }
                else:
                    corner = self._image_position(pos)
                    self._roi_draft = [corner, corner]
                self.update()
            elif self.tool == "freehand":
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
        if self._calibrating:
            pos = event.position().toPoint()
            if self._calibration_drag is not None:
                self._move_calibration_line(*self._calibration_drag, pos)
            else:
                self._set_line_cursor(self._calibration_line_at(pos))
            return
        self._cursor_pos = self._screen_to_image(event.position().toPoint())
        if self.tool == "select":
            pos = event.position().toPoint()
            if self._centre_drag:
                scale, offset_x, _ = self._current_view()
                width = self.image.shape[1]
                self.set_centre_x(max(0.0, min(float(width), (pos.x() - offset_x) / scale)))
            elif self._roi_drag is not None:
                x, y = self._image_position(pos)
                for axis, index in self._roi_drag:
                    self._roi[axis][index] = x if axis == "v" else y
                self._refresh_mask_pixmap()
                self.update()
            elif self._roi_move is not None:
                self._move_roi(pos)
            elif self._centre_marker_hit(pos):
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif not self._roi_grips_at(pos) and self._inside_roi(pos):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self._set_grip_cursor(self._roi_grips_at(pos))
            return
        if self.tool == "roi":
            pos = event.position().toPoint()
            if self._roi_draft is not None:
                self._roi_draft[1] = self._image_position(pos)
                self.update()
            elif self._roi_drag is not None:
                x, y = self._image_position(pos)
                for axis, index in self._roi_drag:
                    self._roi[axis][index] = x if axis == "v" else y
                self._refresh_mask_pixmap()
                self.update()
            elif self._roi_move is not None:
                self._move_roi(pos)
            elif self._roi_grips_at(pos):
                self._set_grip_cursor(self._roi_grips_at(pos))
            elif self._inside_roi(pos):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
            return
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
        if self._calibrating:
            self._calibration_drag = None
            return
        self._centre_drag = False
        if self._roi_drag is not None:
            self._roi_drag = None
            self.roi_changed.emit()
        if self._roi_move is not None:
            moved = self._roi_move["moved"]
            self._roi_move = None
            if moved:
                self.roi_changed.emit()
            else:
                # A click inside the ROI picks the mask when it lands on it.
                pos = event.position().toPoint()
                on_mask = self.tool == "select" and self._mask_at(pos)
                self.selection = "mask" if on_mask else "roi"
                self.update()
        if self._roi_draft is not None:
            (x0, y0), (x1, y1) = self._roi_draft
            self._roi_draft = None
            self.update()
            if abs(x1 - x0) >= 1 and abs(y1 - y0) >= 1:
                previous = self.roi_box()
                self.set_roi((x0, x1, y0, y1))
                self.roi_drawn.emit(previous)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.tool == "freehand":
                if self.fill_mask and len(self._freehand_points) > 2:
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

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete and self.selection is not None:
            self.request_delete(self.selection)
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        # A right-click with the Segmented Line tool finishes the line instead.
        if self.tool == "polyline" or self._calibrating:
            return
        kinds = self.deletable_at(event.pos())
        if not kinds:
            return
        titles = {"outline": "Delete outline", "mask": "Delete mask", "roi": "Delete ROI"}
        menu = QMenu(self)
        for kind in kinds:
            menu.addAction(titles[kind]).setData(kind)
        if "roi" in kinds:
            menu.addAction("Clear all ROIs").setData("all_rois")
        chosen = menu.exec(event.globalPos())
        if chosen is not None:
            self.request_delete(chosen.data())

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

        self._paint_region_overlay(painter, scale, offset_x, offset_y)
        self._paint_roi(painter, scale, offset_x, offset_y, draw_w, draw_h)
        self._paint_selected_mask(painter, scale, offset_x, offset_y)

        if self.tool == "polyline" and self._poly_points:
            painter.setPen(QPen(QColor(0, 255, 0), self._roi_outline_pen_width(), Qt.PenStyle.SolidLine))
            for point in self._poly_points:
                screen_point = self._image_to_screen(point)
                painter.drawEllipse(screen_point, 4, 4)
            for idx in range(len(self._poly_points) - 1):
                start = self._image_to_screen(self._poly_points[idx])
                end = self._image_to_screen(self._poly_points[idx + 1])
                painter.drawLine(start, end)
        if self.tool == "freehand" and self._freehand_points:
            painter.setPen(QPen(QColor(0, 255, 0), self._roi_outline_pen_width(), Qt.PenStyle.SolidLine))
            for idx in range(len(self._freehand_points) - 1):
                start = self._image_to_screen(self._freehand_points[idx])
                end = self._image_to_screen(self._freehand_points[idx + 1])
                painter.drawLine(start, end)
        if self._last_outline:
            painter.setPen(QPen(QColor(0, 255, 0), self._roi_outline_pen_width(), Qt.PenStyle.SolidLine))
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

        if self._calibrating and self._calibration_lines is not None:
            painter.setPen(QPen(QColor(0, 220, 255), 2, Qt.PenStyle.SolidLine))
            for y in self._calibration_lines["h"]:
                screen_y = int(offset_y + y * scale)
                painter.drawLine(offset_x, screen_y, offset_x + draw_w, screen_y)
            painter.setPen(QPen(QColor(255, 170, 0), 2, Qt.PenStyle.SolidLine))
            for x in self._calibration_lines["v"]:
                screen_x = int(offset_x + x * scale)
                painter.drawLine(screen_x, offset_y, screen_x, offset_y + draw_h)

    def _paint_roi(self, painter, scale, offset_x, offset_y, draw_w, draw_h):
        if self._roi is not None:
            left, right, top, bottom = self._roi_screen_rect()
            shade = QColor(0, 0, 0, 120)
            painter.fillRect(QRect(offset_x, offset_y, left - offset_x, draw_h), shade)
            painter.fillRect(
                QRect(right, offset_y, offset_x + draw_w - right, draw_h), shade
            )
            painter.fillRect(QRect(left, offset_y, right - left, top - offset_y), shade)
            painter.fillRect(
                QRect(left, bottom, right - left, offset_y + draw_h - bottom), shade
            )
            if self.selection == "roi":
                pen = QPen(QColor(255, 255, 255), 3, Qt.PenStyle.SolidLine)
            else:
                pen = QPen(QColor(120, 255, 120), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRect(left, top, right - left, bottom - top))
            for x in (left, right):
                for y in (top, bottom):
                    painter.fillRect(QRect(x - 3, y - 3, 7, 7), pen.color())
        if self._roi_draft is not None:
            (x0, y0), (x1, y1) = self._roi_draft
            painter.setPen(QPen(QColor(120, 255, 120), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(
                QRect(
                    QPoint(int(offset_x + x0 * scale), int(offset_y + y0 * scale)),
                    QPoint(int(offset_x + x1 * scale), int(offset_y + y1 * scale)),
                ).normalized()
            )

    def _paint_selected_mask(self, painter, scale, offset_x, offset_y):
        if self.selection != "mask":
            return
        mask = self.visible_mask()
        if mask is None:
            return
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        painter.setPen(QPen(QColor(255, 255, 255), 2, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for contour in contours:
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(offset_x + (x + 0.5) * scale, offset_y + (y + 0.5) * scale)
                        for x, y in contour[:, 0, :]
                    ]
                )
            )

    def _paint_region_overlay(self, painter, scale, offset_x, offset_y):
        overlay = self._region_overlay
        if overlay is None or self.mask_opacity <= 0:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # The bone line follows the mask outline, so it is drawn as coloured dots
        # on a dark line to stay visible on top of an outline of the same colour.
        for name, xs, ys in overlay["bone_lines"]:
            polygon = QPolygonF(
                [
                    QPointF(offset_x + x * scale, offset_y + y * scale)
                    for x, y in zip(xs, ys)
                ]
            )
            color = REGION_COLORS.get(name, WHOLE_MASK_COLOR)
            painter.setPen(QPen(QColor(0, 0, 0), 4, Qt.PenStyle.SolidLine))
            painter.drawPolyline(polygon)
            dotted = QPen(QColor(*color), 3, Qt.PenStyle.DotLine)
            dotted.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(dotted)
            painter.drawPolyline(polygon)
        if self.region_view and overlay["centre"] is not None:
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            columns, top, _ = cartilage_edges(self.visible_mask() > 0)
            for name, (start, stop) in overlay["spans"].items():
                inside = (columns >= start) & (columns < stop)
                if not inside.any():
                    continue
                label = REGION_LABELS[name]
                middle_x = offset_x + (columns[inside].mean() + 0.5) * scale
                label_y = offset_y + top[inside].min() * scale - 8
                painter.setPen(QColor(*REGION_COLORS[name]))
                painter.drawText(
                    QPointF(middle_x - metrics.horizontalAdvance(label) / 2, label_y),
                    label,
                )
            centre_x, centre_y = overlay["centre"]
            x = offset_x + centre_x * scale
            y = offset_y + centre_y * scale
            size = 7
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.setBrush(QColor(*CENTRE_MARKER_COLOR))
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(x, y - size),
                        QPointF(x + size, y),
                        QPointF(x, y + size),
                        QPointF(x - size, y),
                    ]
                )
            )
        painter.restore()

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
        if delta == 0:
            delta = event.pixelDelta().y()
        if delta == 0:
            event.ignore()
            return
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            factor = 1.2 if delta > 0 else 1 / 1.2
            self._apply_zoom(factor)
        elif modifiers & Qt.KeyboardModifier.ShiftModifier:
            if self.h_scrollbar.isVisible():
                self._scroll_horizontal(-delta)
        else:
            self.navigation_requested.emit(-1 if delta > 0 else 1)
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
