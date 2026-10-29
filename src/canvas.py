"""
Canvas widget for displaying images and annotating them with instance masks.
"""

import numpy as np
import cv2
from .tiff_stack import read_tiff_frame
from PyQt6.QtWidgets import QWidget, QMessageBox, QScrollBar, QStyle, QMenu
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPolygonF, QFont
from PyQt6.QtCore import Qt, QRect, QRectF, QPoint, QPointF, pyqtSignal

from .instances import (
    clean_class,
    clip_instances,
    copy_instances,
    display_class,
    instance_title,
    make_instance,
    next_instance_id,
)
from .labels import class_color
from .theme import CANVAS

# Tools that make or change a mask of a label, and so need a label picked or a
# mask selected before they start.
LABELLED_TOOLS = ("freehand", "polyline", "brush", "sam_point", "sam_box", "sam_mask")


def label_anchor(mask):
    """The mask pixel (x, y) nearest the mask's centroid, so that a label put
    there sits on the mask even when the mask is curved or hollow."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    nearest = int(np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))
    return int(xs[nearest]), int(ys[nearest])


class Canvas(QWidget):
    image_loaded = pyqtSignal(str)
    navigation_requested = pyqtSignal(int)
    mask_changed = pyqtSignal()
    roi_drawn = pyqtSignal()
    roi_changed = pyqtSignal()
    delete_requested = pyqtSignal(str)
    # A message for the status bar, such as why a drawing was not made.
    notice = pyqtSignal(str)
    # A SAM prompt, in image pixels: ("point", x, y, include) or
    # ("box", left, top, right, bottom).
    sam_prompt_added = pyqtSignal(tuple)
    # Why a drawing tool was not started: no label is picked and no mask selected.
    label_needed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.image = None
        self.pixmap = None
        self.image_path = None
        # The masks of the image on screen, each {"class", "id", "mask"} with a
        # uint8 0/1 mask the size of the image. Later ones are drawn on top.
        self.instances = []
        # Index in self.instances of the instance the Select tool picked, or None.
        self.selected_index = None
        # (label, number) of the masks kept out of view. They are still saved and
        # measured, but are not drawn, picked or erased.
        self.hidden_keys = set()
        # The label new masks get, and the order of the labels, which gives each
        # its colour.
        self.active_class = None
        self.class_order = []
        # The labels offered when relabelling a mask: (nerves, other anatomy).
        self.label_choices = ([], [])
        self.mask_pixmap = None
        self._label_anchors = []
        # The screen rectangle of each mask's name tag as last drawn, with the
        # mask's (label, number); later ones are drawn on top.
        self._label_rects = []
        self.mask_opacity = 0.5
        self.show_contour_only = True
        # Each history entry is (instances, ROI box): the state after one edit.
        self._undo_stack = []
        self._redo_stack = []
        # edit_count goes up with every edit recorded, history_resets with every
        # fresh start of the history; both let the window order its own undo
        # steps against the canvas's.
        self.edit_count = 0
        self.history_resets = 0
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
        # The instance a brush stroke paints into.
        self._stroke_target = None

        # The ROI of the image on screen, or None: "v" holds the x positions of its
        # two vertical edges and "h" the y positions of its two horizontal edges, in
        # image pixels. The masks outside it are hidden, not saved and not
        # measured, but stay in self.instances.
        self._roi = None
        self._roi_drag = None
        # While the whole ROI is being dragged: where the press landed, the edge
        # positions at that moment, and whether the pointer has moved far enough
        # for the press to count as a drag instead of a click.
        self._roi_move = None
        # Opposite corners (x, y) of a box being drawn with the ROI tool.
        self._roi_draft = None
        # While a mask is being dragged with the Select tool: where the press
        # landed, the mask at that moment, and whether the pointer has moved far
        # enough for the press to count as a drag instead of a click.
        self._mask_move = None
        # What a click with the Select tool picked for deletion: "roi", "mask" or None.
        self.selection = None
        # Opposite corners (x, y) of a box being drawn with the Box prompt tool.
        self._sam_box_draft = None
        # Where a press of the Mask prompt tool landed, in widget pixels, and
        # whether the pointer has since moved away, while the button is down: a
        # press that moves draws freehand, one that does not adds a vertex of a
        # segmented outline.
        self._sam_mask_press = None
        # The SAM prompts of the object being prompted, drawn over the image:
        # points (x, y, include), a box (left, top, right, bottom) or None, and the
        # outlines of a drawn mask.
        self._sam_points = []
        self._sam_box = None
        self._sam_mask_outline = []

        # Calibration lines in image pixels: "h" holds the y positions of the two
        # horizontal lines, "v" the x positions of the two vertical lines. They
        # outlive a calibration so the next one starts from the same places.
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
        self.instances = []
        self.selected_index = None
        self._roi = None
        self._roi_drag = None
        self._roi_move = None
        self._roi_draft = None
        self._mask_move = None
        self.selection = None
        self._refresh_mask_pixmap()
        self._last_outline = []
        self._mask_touched = False
        self._reset_history()
        self._reset_view()
        self.update()
        self.image_loaded.emit(source_name)

    def clear_mask(self):
        """Remove every mask of the image on screen, as a step Undo takes back."""
        if self.selection == "mask":
            self.selection = None
        self.selected_index = None
        self._last_outline = []
        if self.image is None:
            self.instances = []
            self.mask_pixmap = None
            self._mask_touched = False
            self._reset_history()
            self.update()
            self.mask_changed.emit()
            return
        if not self.instances:
            self.update()
            return
        self.instances = []
        self._refresh_mask_pixmap()
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

    def set_instances(self, instances):
        """Show these masks, as the state Undo starts from."""
        if self.image is None:
            return
        height, width = self.image.shape[:2]
        self.instances = [
            make_instance(
                i["class"],
                i["id"],
                i["mask"]
                if i["mask"].shape[:2] == (height, width)
                else cv2.resize(
                    i["mask"].astype(np.uint8), (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ),
            )
            for i in instances
        ]
        self.selected_index = None
        if self.selection == "mask":
            self.selection = None
        self._refresh_mask_pixmap()
        self._mask_touched = True
        self._reset_history()
        self.update()

    def replace_classes(self, instances, classes):
        """Put new masks in place of those of the given classes, as a step Undo
        takes back. The new masks are numbered on from the masks kept."""
        if self.image is None:
            return
        classes = {clean_class(c) for c in classes}
        kept = [i for i in self.instances if i["class"] not in classes]
        for instance in instances:
            kept.append(
                make_instance(
                    instance["class"],
                    next_instance_id(kept, instance["class"]),
                    instance["mask"],
                )
            )
        self.instances = kept
        self.selected_index = None
        if self.selection == "mask":
            self.selection = None
        self._refresh_mask_pixmap()
        self._push_history()
        self._mask_touched = True
        self.update()

    def put_instance(self, instance):
        """Show a mask in place of the one with its label and number, or add it,
        as a step Undo takes back."""
        if self.image is None:
            return
        key = (clean_class(instance["class"]), int(instance["id"]))
        kept = [i for i in self.instances if (i["class"], i["id"]) != key]
        kept.append(make_instance(key[0], key[1], instance["mask"]))
        self.instances = kept
        self.selected_index = len(kept) - 1
        self._refresh_mask_pixmap()
        self._push_history()
        self._mask_touched = True
        self.update()

    def add_instances(self, instances):
        """Add masks, numbered on from those of their label, as a step Undo takes
        back."""
        if self.image is None or not instances:
            return
        for instance in instances:
            self.instances.append(
                make_instance(
                    instance["class"],
                    next_instance_id(self.instances, instance["class"]),
                    instance["mask"],
                )
            )
        self.selected_index = None
        self._refresh_mask_pixmap()
        self._push_history()
        self._mask_touched = True
        self.update()

    def display_rgb(self):
        """The image as drawn: 8-bit RGB, gray images scaled to their range."""
        if self.image is None:
            return None
        if self.image.ndim == 2:
            gray = self._normalize_gray(self.image)
            return np.ascontiguousarray(np.stack([gray] * 3, axis=-1))
        return np.ascontiguousarray(self._as_uint8(self.image[:, :, :3]))

    def set_sam_prompts(self, points, box, mask=None):
        self._sam_points = list(points)
        self._sam_box = box
        self._sam_mask_outline = []
        if mask is not None:
            self._sam_mask_outline, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
        self.update()

    def set_class_order(self, order):
        self.class_order = [clean_class(c) for c in order]
        self._refresh_mask_pixmap()
        self.update()

    def set_label_choices(self, nerves, anatomy):
        self.label_choices = (list(nerves), list(anatomy))

    def set_active_class(self, name):
        self.active_class = clean_class(name) or None

    def color_of(self, class_name):
        return class_color(class_name, self.class_order)

    def set_mask_opacity(self, opacity):
        self.mask_opacity = max(0.0, min(1.0, float(opacity)))
        self._refresh_mask_pixmap()
        self.update()

    def set_contour_only(self, enabled):
        self.show_contour_only = bool(enabled)
        self._refresh_mask_pixmap()
        self.update()

    def set_tool(self, tool_name):
        self.tool = tool_name
        self._drawing = False
        self._poly_points = []
        self._freehand_points = []
        self._roi_draft = None
        self._sam_box_draft = None
        self._sam_mask_press = None
        # A selected mask stays selected, so that the brush and eraser keep to it.
        if self.selection != "mask":
            self.selection = None
        if tool_name in ("roi", "sam_point", "sam_box", "sam_mask"):
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
        self._refresh_mask_pixmap()
        self.update()

    def commit_pending_outline_to_mask(self):
        if len(self._last_outline) < 2:
            return False
        outline = self._last_outline
        self._last_outline = []
        made = self._add_polygon(outline)
        self.update()
        return made

    def _add_polygon(self, points):
        """Fill a closed outline as a new mask of the active label.

        With no label picked the outline is added to the selected mask instead,
        and with neither nothing is drawn. Returns whether a mask changed.
        """
        if self.image is None:
            return False
        outline = np.array([[p.x(), p.y()] for p in points], dtype=np.int32)
        if self.active_class is not None:
            height, width = self.image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [outline.reshape((-1, 1, 2))], 1)
            self.instances.append(
                make_instance(
                    self.active_class,
                    next_instance_id(self.instances, self.active_class),
                    mask,
                )
            )
            self.selected_index = len(self.instances) - 1
        elif self._selected_instance() is not None:
            cv2.fillPoly(
                self._selected_instance()["mask"], [outline.reshape((-1, 1, 2))], 1
            )
        else:
            self.label_needed.emit("Pick a nerve or anatomy label before drawing a mask.")
            return False
        self._refresh_mask_pixmap()
        self._push_history()
        self._mask_touched = True
        return True

    def _selected_instance(self):
        if self.selected_index is None or self.selected_index >= len(self.instances):
            return None
        return self.instances[self.selected_index]

    def set_roi(self, box, record=False):
        """Give the image on screen the ROI box (left, right, top, bottom), or None.

        record makes the change a step that Undo takes back, for a change the
        user made; without it the box becomes part of the state Undo returns to.
        """
        self._apply_roi(box)
        if record:
            self._push_history()
        elif self._undo_stack:
            self._undo_stack[-1] = (self._undo_stack[-1][0], self.roi_box())

    def history_depth(self):
        return len(self._undo_stack)

    def can_redo(self):
        return bool(self._redo_stack)

    def _apply_roi(self, box):
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

    def visible_instances(self):
        """The masks as shown, saved and measured: emptied outside the ROI, with
        those left empty dropped."""
        return clip_instances(self.instances, self.roi_box())

    def is_shown(self, instance):
        return (instance["class"], instance["id"]) not in self.hidden_keys

    def set_hidden_keys(self, keys):
        """Keep the masks of these (label, number) keys out of view."""
        self.hidden_keys = set(keys)
        selected = self._selected_instance()
        if selected is not None and not self.is_shown(selected):
            self.selected_index = None
            if self.selection == "mask":
                self.selection = None
        self._refresh_mask_pixmap()
        self.update()

    def has_visible_mask(self):
        return bool(self.visible_instances())

    def has_hidden_mask(self):
        """True when the ROI keeps every mask on the image out of view."""
        return any(i["mask"].any() for i in self.instances) and not self.has_visible_mask()

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

    def _move_mask(self, pos):
        """Slide the selected mask with the pointer by whole pixels; what is
        pushed past the edge of the image is cut off."""
        move = self._mask_move
        instance = self._selected_instance()
        if instance is None:
            self._mask_move = None
            return
        if not move["moved"]:
            if (pos - move["press"]).manhattanLength() < 4:
                return
            move["moved"] = True
        scale = self._current_view()[0]
        dx = int(round((pos.x() - move["press"].x()) / scale))
        dy = int(round((pos.y() - move["press"].y()) / scale))
        source = move["mask"]
        height, width = source.shape[:2]
        shifted = np.zeros_like(source)
        if abs(dx) < width and abs(dy) < height:
            shifted[max(0, dy):height + min(0, dy), max(0, dx):width + min(0, dx)] = source[
                max(0, -dy):height + min(0, -dy), max(0, -dx):width + min(0, -dx)
            ]
        instance["mask"] = shifted
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

    def _instance_at(self, pos):
        """Index of the topmost mask shown at a widget position, or None."""
        point = self._screen_to_image(pos)
        if point is None or self.image is None:
            return None
        x, y = point.x(), point.y()
        box = self.roi_box()
        if box is not None:
            left, right, top, bottom = box
            if not (left <= x < right and top <= y < bottom):
                return None
        for index in range(len(self.instances) - 1, -1, -1):
            if self.instances[index]["mask"][y, x] and self.is_shown(self.instances[index]):
                return index
        return None

    def _mask_at(self, pos):
        return self._instance_at(pos) is not None

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
        elif kind == "mask":
            self.delete_selected_instance()
        else:
            self.delete_requested.emit(kind)

    def delete_selected_instance(self):
        instance = self._selected_instance()
        if instance is None:
            return
        del self.instances[self.selected_index]
        self.selected_index = None
        self.selection = None
        self._refresh_mask_pixmap()
        self._push_history()
        self._mask_touched = True
        self.notice.emit(f"Deleted {instance_title(instance)}")
        self.update()

    def relabel_selected_instance(self, class_name):
        """Give the selected mask another label, numbered on from that label's masks."""
        instance = self._selected_instance()
        class_name = clean_class(class_name)
        if instance is None or not class_name or instance["class"] == class_name:
            return
        old_title = instance_title(instance)
        others = [i for i in self.instances if i is not instance]
        instance["id"] = next_instance_id(others, class_name)
        instance["class"] = class_name
        self._refresh_mask_pixmap()
        self._push_history()
        self._mask_touched = True
        self.notice.emit(f"{old_title} is now {instance_title(instance)}")
        self.update()

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
            # Of a TIFF holding several images, the first one is used.
            image = read_tiff_frame(file_path, 0)
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

    def _refresh_mask_pixmap(self):
        self.mask_changed.emit()
        if self.image is None:
            self.mask_pixmap = None
            self._label_anchors = []
            return
        height, width = self.image.shape[:2]
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        alpha = int(255 * self.mask_opacity)
        self._label_anchors = []
        for instance in self.visible_instances():
            if not self.is_shown(instance):
                continue
            color = self.color_of(instance["class"])
            mask = instance["mask"]
            if self.show_contour_only:
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(
                    rgba, contours, -1, (*color, alpha), self._roi_outline_thickness_px()
                )
            else:
                rgba[mask > 0] = (*color, alpha)
            anchor = label_anchor(mask)
            if anchor is not None:
                self._label_anchors.append((instance, anchor))
        rgba = np.ascontiguousarray(rgba)
        q_image = QImage(
            rgba.tobytes(), width, height, rgba.strides[0], QImage.Format.Format_RGBA8888
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

    def _brush_target(self):
        """The mask a brush stroke paints into: the selected mask when it has the
        active label or no label is picked, otherwise a new mask of the active
        label. None, with a notice, when there is neither."""
        selected = self._selected_instance()
        if selected is not None and self.active_class in (None, selected["class"]):
            return selected
        if self.active_class is None:
            self.label_needed.emit("Pick a nerve or anatomy label before painting a mask.")
            return None
        height, width = self.image.shape[:2]
        instance = make_instance(
            self.active_class,
            next_instance_id(self.instances, self.active_class),
            np.zeros((height, width), dtype=np.uint8),
        )
        self.instances.append(instance)
        self.selected_index = len(self.instances) - 1
        return instance

    def _paint(self, start, end):
        """Paint with the brush, or erase with the eraser, along a stroke from
        start to end (a point when both are the same). The eraser erases from the
        selected mask alone when one is selected, else from every mask shown."""
        if self.tool == "eraser":
            selected = self._selected_instance() if self.selection == "mask" else None
            if selected is not None and self.is_shown(selected):
                targets = [selected]
            else:
                targets = [i for i in self.instances if self.is_shown(i)]
            value = 0
        elif self._stroke_target is not None:
            targets, value = [self._stroke_target], 1
        else:
            return
        for instance in targets:
            if start == end:
                cv2.circle(
                    instance["mask"], (start.x(), start.y()), self.brush_radius, value, -1
                )
            else:
                cv2.line(
                    instance["mask"],
                    (start.x(), start.y()),
                    (end.x(), end.y()),
                    value,
                    self._brush_thickness(),
                )

    def _drop_empty_instances(self):
        selected = self._selected_instance()
        self.instances = [i for i in self.instances if i["mask"].any()]
        self.selected_index = next(
            (n for n, i in enumerate(self.instances) if i is selected), None
        )

    def _finish_polyline(self):
        if len(self._poly_points) < 2:
            self._poly_points = []
            self.update()
            return
        if self.fill_mask:
            self._add_polygon(self._poly_points)
        else:
            self._last_outline = list(self._poly_points)
        self._poly_points = []
        self.update()

    def _emit_mask_prompt(self, points):
        """Send the filled outline of points as a mask prompt."""
        if len(points) > 2:
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            outline = np.array([[p.x(), p.y()] for p in points], dtype=np.int32)
            cv2.fillPoly(mask, [outline.reshape((-1, 1, 2))], 1)
            if mask.any():
                self.sam_prompt_added.emit(("mask", mask.astype(bool)))
        self._freehand_points = []
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
            pos = event.position().toPoint()
            index = self._label_at(pos)
            if index is not None:
                if self.tool == "select":
                    # With Select, a tag is a handle: it selects the mask and drags it.
                    self.selected_index = index
                    self.selection = "mask"
                    self.notice.emit(f"Selected {instance_title(self.instances[index])}")
                    self._mask_move = {
                        "press": pos,
                        "mask": self.instances[index]["mask"].copy(),
                        "moved": False,
                    }
                    self.update()
                else:
                    self._select_by_label(index)
                return
        if self.tool in LABELLED_TOOLS and event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        ):
            if self.active_class is None and self._selected_instance() is None:
                if event.button() == Qt.MouseButton.LeftButton or self.tool == "sam_point":
                    self.label_needed.emit(
                        "Pick a nerve or anatomy label under Labels before drawing "
                        "or prompting a mask, or select the mask to change."
                    )
                    return
        if event.button() == Qt.MouseButton.LeftButton:
            point = self._screen_to_image(event.position().toPoint())
            if point is None:
                return
            if self.tool == "select":
                pos = event.position().toPoint()
                self._roi_drag = self._roi_grips_at(pos) or None
                if self._roi_drag is not None:
                    self.selection = "roi"
                elif self._mask_at(pos):
                    self._select_instance_at(pos)
                    self._mask_move = {
                        "press": pos,
                        "mask": self.instances[self.selected_index]["mask"].copy(),
                        "moved": False,
                    }
                elif self._inside_roi(pos):
                    self._roi_move = {
                        "press": pos,
                        "v": list(self._roi["v"]),
                        "h": list(self._roi["h"]),
                        "moved": False,
                    }
                else:
                    self._select_instance_at(pos)
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
            elif self.tool in ("brush", "eraser"):
                self._stroke_target = (
                    self._brush_target() if self.tool == "brush" else None
                )
                if self.tool == "brush" and self._stroke_target is None:
                    return
                self._drawing = True
                self._last_point = point
                self._paint(point, point)
                self._refresh_mask_pixmap()
                self.update()
            elif self.tool == "polyline":
                self._poly_points.append(point)
                self.update()
            elif self.tool == "sam_point":
                self.sam_prompt_added.emit(("point", point.x(), point.y(), True))
            elif self.tool == "sam_box":
                self._sam_box_draft = [point, point]
                self.update()
            elif self.tool == "sam_mask":
                self._sam_mask_press = {"pos": event.position().toPoint(), "moved": False}
                self._drawing = True
                self._freehand_points = [point]
                self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            if self.tool == "polyline":
                self._finish_polyline()
            elif self.tool == "sam_mask":
                self._emit_mask_prompt(self._poly_points)
            elif self.tool == "sam_point":
                point = self._screen_to_image(event.position().toPoint())
                if point is not None:
                    self.sam_prompt_added.emit(("point", point.x(), point.y(), False))

    def _label_at(self, pos):
        """Index in self.instances of the mask whose name tag is at a widget
        position, or None."""
        point = QPointF(pos)
        for rect, key in reversed(self._label_rects):
            if rect.contains(point):
                for index, instance in enumerate(self.instances):
                    if (instance["class"], instance["id"]) == key:
                        return index
        return None

    def _select_by_label(self, index):
        """Select the mask of a name tag clicked, or deselect it when it is
        selected already."""
        if self.selection == "mask" and self.selected_index == index:
            self.selected_index = None
            self.selection = None
            self.notice.emit("No mask selected")
        else:
            self.selected_index = index
            self.selection = "mask"
            self.notice.emit(f"Selected {instance_title(self.instances[index])}")
        self.update()

    def _select_instance_at(self, pos):
        self.selected_index = self._instance_at(pos)
        self.selection = None if self.selected_index is None else "mask"
        if self.selected_index is not None:
            self.notice.emit(f"Selected {instance_title(self.instances[self.selected_index])}")

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
            if self._mask_move is not None:
                self._move_mask(pos)
            elif self._roi_drag is not None:
                x, y = self._image_position(pos)
                for axis, index in self._roi_drag:
                    self._roi[axis][index] = x if axis == "v" else y
                self._refresh_mask_pixmap()
                self.update()
            elif self._roi_move is not None:
                self._move_roi(pos)
            elif not self._roi_grips_at(pos) and (self._mask_at(pos) or self._inside_roi(pos)):
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
        if self._sam_box_draft is not None:
            point = self._screen_to_image(event.position().toPoint())
            if point is not None:
                self._sam_box_draft[1] = point
                self.update()
            return
        if not self._drawing:
            if self.tool in ("brush", "eraser"):
                if self._label_at(event.position().toPoint()) is not None:
                    self.setCursor(Qt.CursorShape.PointingHandCursor)
                else:
                    self.unsetCursor()
                self.update()
            return
        point = self._screen_to_image(event.position().toPoint())
        if point is None:
            return
        if self.tool in ("freehand", "sam_mask"):
            press = self._sam_mask_press
            if press is not None and (
                event.position().toPoint() - press["pos"]
            ).manhattanLength() >= 4:
                press["moved"] = True
            self._freehand_points.append(point)
            self.update()
            return
        if self._last_point is None:
            return
        if self.tool in ("brush", "eraser"):
            self._paint(self._last_point, point)
        self._last_point = point
        self._refresh_mask_pixmap()
        self.update()

    def mouseReleaseEvent(self, event):
        if self._calibrating:
            self._calibration_drag = None
            return
        if self._mask_move is not None:
            moved = self._mask_move["moved"]
            self._mask_move = None
            if moved:
                self._drop_empty_instances()
                self._refresh_mask_pixmap()
                self._push_history()
                self._mask_touched = True
            self.update()
            return
        if self._roi_drag is not None:
            self._roi_drag = None
            self._push_history()
            self.roi_changed.emit()
        if self._roi_move is not None:
            moved = self._roi_move["moved"]
            self._roi_move = None
            if moved:
                self._push_history()
                self.roi_changed.emit()
            else:
                # A click inside the ROI picks the mask when it lands on it.
                pos = event.position().toPoint()
                if self.tool == "select" and self._mask_at(pos):
                    self._select_instance_at(pos)
                else:
                    self.selection = "roi"
                self.update()
        if self._sam_box_draft is not None:
            first, second = self._sam_box_draft
            self._sam_box_draft = None
            self.update()
            left, right = sorted((first.x(), second.x()))
            top, bottom = sorted((first.y(), second.y()))
            if right - left >= 2 and bottom - top >= 2:
                self.sam_prompt_added.emit(("box", left, top, right, bottom))
            return
        if self._roi_draft is not None:
            (x0, y0), (x1, y1) = self._roi_draft
            self._roi_draft = None
            self.update()
            if abs(x1 - x0) >= 1 and abs(y1 - y0) >= 1:
                self.set_roi((x0, x1, y0, y1), record=True)
                self.roi_drawn.emit()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.tool == "sam_mask" and self._sam_mask_press is not None:
                moved = self._sam_mask_press["moved"]
                self._sam_mask_press = None
                if moved and len(self._freehand_points) > 2:
                    self._emit_mask_prompt(self._freehand_points)
                elif self._freehand_points:
                    self._poly_points.append(self._freehand_points[0])
                    self._freehand_points = []
            elif self.tool == "freehand":
                if self.fill_mask and len(self._freehand_points) > 2:
                    self._add_polygon(self._freehand_points)
                elif len(self._freehand_points) > 1:
                    self._last_outline = list(self._freehand_points)
                self._freehand_points = []
            elif self.tool in ("brush", "eraser"):
                if self._drawing:
                    self._drop_empty_instances()
                    self._refresh_mask_pixmap()
                    self._push_history()
                    self._mask_touched = True
                self._stroke_target = None
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
        if event.key() == Qt.Key.Key_Escape and self.selection == "mask":
            self.selected_index = None
            self.selection = None
            self.update()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        # A right-click with the Segmented Line or Mask prompt tool finishes the
        # line instead, and with the Point prompt tool adds an exclude point.
        if self.tool in ("polyline", "sam_point", "sam_mask") or self._calibrating:
            return
        kinds = self.deletable_at(event.pos())
        if not kinds:
            return
        menu = QMenu(self)
        if "outline" in kinds:
            menu.addAction("Delete outline").setData("outline")
        if "mask" in kinds:
            self._select_instance_at(event.pos())
            self.update()
            instance = self._selected_instance()
            label_menu = menu.addMenu(f"Label of {instance_title(instance)}")
            nerves, anatomy = self.label_choices
            for group_index, group in enumerate((nerves, anatomy)):
                if group_index and group:
                    label_menu.addSeparator()
                for name in group:
                    action = label_menu.addAction(display_class(name))
                    action.setCheckable(True)
                    action.setChecked(name == instance["class"])
                    action.setData(("label", name))
            menu.addAction("Delete mask").setData("mask")
        if "roi" in kinds:
            menu.addAction("Delete ROI").setData("roi")
            menu.addAction("Apply ROI to all").setData("apply_roi")
            menu.addAction("Clear all ROIs").setData("all_rois")
        chosen = menu.exec(event.globalPos())
        if chosen is None:
            return
        data = chosen.data()
        if isinstance(data, tuple) and data[0] == "label":
            self.relabel_selected_instance(data[1])
        else:
            self.request_delete(data)

    def mouseDoubleClickEvent(self, event):
        if self.tool == "polyline":
            self._finish_polyline()
        elif self.tool == "sam_mask":
            self._emit_mask_prompt(self._poly_points)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(CANVAS))
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

        self._paint_roi(painter, scale, offset_x, offset_y, draw_w, draw_h)
        self._paint_selected_mask(painter, scale, offset_x, offset_y)
        self._paint_labels(painter, scale, offset_x, offset_y, draw_w, draw_h)

        if self.tool in ("polyline", "sam_mask") and self._poly_points:
            painter.setPen(QPen(QColor(0, 255, 0), self._roi_outline_pen_width(), Qt.PenStyle.SolidLine))
            for point in self._poly_points:
                screen_point = self._image_to_screen(point)
                painter.drawEllipse(screen_point, 4, 4)
            for idx in range(len(self._poly_points) - 1):
                start = self._image_to_screen(self._poly_points[idx])
                end = self._image_to_screen(self._poly_points[idx + 1])
                painter.drawLine(start, end)
        if self.tool in ("freehand", "sam_mask") and self._freehand_points:
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

        self._paint_sam_prompts(painter)

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

    def _paint_sam_prompts(self, painter):
        width = self._roi_outline_pen_width()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        boxes = [self._sam_box]
        if self._sam_box_draft is not None:
            first, second = self._sam_box_draft
            boxes.append((first.x(), first.y(), second.x(), second.y()))
        painter.setPen(QPen(QColor(255, 220, 0), width, Qt.PenStyle.DashLine))
        for box in boxes:
            if box is None:
                continue
            start = self._image_to_screen(QPoint(box[0], box[1]))
            end = self._image_to_screen(QPoint(box[2], box[3]))
            painter.drawRect(QRect(start, end).normalized())
        for contour in self._sam_mask_outline:
            corners = [self._image_to_screen(QPoint(int(x), int(y))) for x, y in contour[:, 0]]
            painter.drawPolygon(QPolygonF([QPointF(c) for c in corners]))
        for x, y, include in self._sam_points:
            center = self._image_to_screen(QPoint(int(x), int(y)))
            painter.setPen(QPen(QColor(0, 0, 0), width + 3))
            self._draw_sam_marker(painter, center, include)
            painter.setPen(QPen(QColor(0, 230, 0) if include else QColor(255, 60, 60), width + 1))
            self._draw_sam_marker(painter, center, include)

    @staticmethod
    def _draw_sam_marker(painter, center, include):
        """A plus for an include point, a minus for an exclude point."""
        arm = 7
        painter.drawLine(center.x() - arm, center.y(), center.x() + arm, center.y())
        if include:
            painter.drawLine(center.x(), center.y() - arm, center.x(), center.y() + arm)

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
        instance = self._selected_instance()
        if self.selection != "mask" or instance is None or not self.is_shown(instance):
            return
        shown = clip_instances([instance], self.roi_box())
        if not shown:
            return
        contours, _ = cv2.findContours(
            shown[0]["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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

    def _paint_labels(self, painter, scale, offset_x, offset_y, draw_w, draw_h):
        """A tag naming each mask, on the mask and kept inside the image."""
        self._label_rects = []
        if not self._label_anchors or self.mask_opacity <= 0:
            return
        selected = self._selected_instance() if self.selection == "mask" else None
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = QFont(painter.font())
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        for instance, (x, y) in self._label_anchors:
            text = instance_title(instance)
            width = metrics.horizontalAdvance(text) + 14
            height = metrics.height() + 6
            centre_x = offset_x + (x + 0.5) * scale
            centre_y = offset_y + (y + 0.5) * scale
            if draw_w >= width:
                centre_x = max(offset_x + width / 2, min(offset_x + draw_w - width / 2, centre_x))
            if draw_h >= height:
                centre_y = max(offset_y + height / 2, min(offset_y + draw_h - height / 2, centre_y))
            rect = QRectF(centre_x - width / 2, centre_y - height / 2, width, height)
            self._label_rects.append((rect, (instance["class"], instance["id"])))
            is_selected = selected is not None and instance["class"] == selected["class"] and instance["id"] == selected["id"]
            painter.setPen(
                QPen(
                    QColor(255, 255, 255) if is_selected else QColor(*self.color_of(instance["class"])),
                    2 if is_selected else 1.5,
                )
            )
            painter.setBrush(QColor(20, 27, 34, 230))
            painter.drawRoundedRect(rect, 5, 5)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

    def undo(self):
        if len(self._undo_stack) <= 1:
            return
        current = self._undo_stack.pop()
        self._redo_stack.append(current)
        self._restore_history_entry(self._undo_stack[-1])

    def redo(self):
        if not self._redo_stack:
            return
        state = self._redo_stack.pop()
        self._undo_stack.append(state)
        self._restore_history_entry(state)

    def _restore_history_entry(self, entry):
        instances, box = entry
        self.instances = copy_instances(instances)
        self.selected_index = None
        if self.selection == "mask":
            self.selection = None
        if box != self.roi_box():
            self._apply_roi(box)
            self.roi_changed.emit()
        else:
            self._refresh_mask_pixmap()
            self.update()

    def _reset_history(self):
        self._undo_stack = []
        self._redo_stack = []
        self.history_resets += 1
        if self.image is not None:
            self._undo_stack.append((copy_instances(self.instances), self.roi_box()))

    def _push_history(self):
        if self.image is None:
            return
        self._undo_stack.append((copy_instances(self.instances), self.roi_box()))
        self._redo_stack = []
        self.edit_count += 1

    def has_mask_data(self):
        return self.image is not None and self._mask_touched

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
