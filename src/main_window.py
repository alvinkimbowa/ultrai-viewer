"""
Main window for the knee ultrasound app (new MVP shell).
"""

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QApplication,
    QMessageBox,
    QProgressDialog,
    QProgressBar,
    QFileDialog,
    QDialog,
    QLineEdit,
    QFormLayout,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
)
from PyQt6.QtCore import Qt, QObject, QThread, QTimer, QSize, QEvent, QSettings, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QShortcut

from .canvas import Canvas, REGION_COLORS
from .model_integration import ModelIntegration, GPU_FALLBACK_WARNING
from .rois import RoiStore
from .measurements import (
    MEASUREMENTS_FILE_NAME,
    MeasurementLog,
    REGION_NAMES,
    measure,
)
from .rotations import RotationStore, rotate_image, rotate_mask, unrotate_mask
from threading import Event
from collections import OrderedDict
import os
from pathlib import Path
import numpy as np
import cv2
import tifffile

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv")
MASK_EXTENSIONS = (".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg")
MIN_LIMITED_SIZE = 8


NO_MODEL_LABEL = "No model"
PREDICTIONS_SUFFIX = "_predictions"


def find_mask_files(folder, image_path, images=()):
    """Mask files in `folder` that belong to an image, the best match first.

    A mask belongs to an image when its file name starts with the image's name
    without its ending: "scan.png", "scan.tif.png" and "scan_mask.png" all belong
    to "scan.tif". The character after that start must not be a letter or a digit,
    which keeps "scan_15.png" from being taken as the mask of "scan_1.tif". The
    exact name "scan.png" comes first, then the shortest name.

    `images` are the loaded image files; they are never taken as masks, since
    the output folder can be the folder the images are in.
    """
    image_path = Path(image_path)
    stem = image_path.stem
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    skip = {os.path.abspath(path) for path in images}
    skip.add(os.path.abspath(image_path))
    found = []
    for name in names:
        if not name.startswith(stem) or not name.lower().endswith(MASK_EXTENSIONS):
            continue
        if name[len(stem) : len(stem) + 1].isalnum():
            continue
        path = Path(folder) / name
        if os.path.abspath(path) in skip or not path.is_file():
            continue
        found.append(path)
    return sorted(found, key=lambda path: (path.stem != stem, len(path.name), path.name))


def write_mask_file(mask_path, mask, angle=0.0, original_shape=None):
    """Save a mask as a 0/255 image that lines up with the image file as stored.

    mask is the mask as viewed. For an image viewed turned by angle, it is turned
    back onto the stored image, whose (height, width) is original_shape.
    """
    mask_path = Path(mask_path)
    stored = np.asarray(mask) > 0
    if angle:
        stored = unrotate_mask(stored, angle, original_shape)
    mask_uint8 = stored.astype(np.uint8) * 255
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if mask_path.suffix.lower() in (".tif", ".tiff"):
        tifffile.imwrite(str(mask_path), mask_uint8)
    elif not cv2.imwrite(str(mask_path), mask_uint8):
        raise OSError(f"Failed to save mask: {mask_path.name}")


def box_to_segment(image, box):
    """Clamp an ROI box (left, right, top, bottom) to an image.

    Returns None when there is no box or it leaves too little to segment.
    """
    if box is None:
        return None
    height, width = image.shape[:2]
    left, right = min(box[0], width), min(box[1], width)
    top, bottom = min(box[2], height), min(box[3], height)
    if right - left < MIN_LIMITED_SIZE or bottom - top < MIN_LIMITED_SIZE:
        return None
    return left, right, top, bottom


def crop_to_box(image, box):
    if box is None:
        return image
    left, right, top, bottom = box
    return np.ascontiguousarray(image[top:bottom, left:right])


def place_prediction(prediction, image, box):
    """Turn a model output into a mask the size of the image.

    The model's output has its own fixed size. With a box (left, right, top,
    bottom) it covers only that part of the image, and the rest of the mask is
    left empty.
    """
    height, width = image.shape[:2]
    left, right, top, bottom = (0, width, 0, height) if box is None else box
    part = np.asarray(prediction).astype(np.float32)
    if part.shape[:2] != (bottom - top, right - left):
        part = cv2.resize(
            part, (right - left, bottom - top), interpolation=cv2.INTER_NEAREST
        )
    mask = np.zeros((height, width), dtype=np.float32)
    mask[top:bottom, left:right] = part
    return mask


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._base_title = "UltAI Viewer"
        self.setWindowTitle(self._base_title)
        self._sidebar_width = 340

        self._create_menu_bar()

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)

        self.canvas = Canvas()
        self.canvas.image_loaded.connect(self._update_title_with_image)
        self.canvas.navigation_requested.connect(self._on_canvas_navigation_requested)

        sidebar = self._build_sidebar()
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setWidget(sidebar)
        sidebar_scroll.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        scroll_extent = sidebar_scroll.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
        sidebar_scroll.setMinimumWidth(self._sidebar_width + scroll_extent + 4)
        sidebar_scroll.setMaximumWidth(self._sidebar_width + scroll_extent + 4)
        root_layout.addWidget(sidebar_scroll, stretch=0)

        canvas_panel = QWidget()
        canvas_layout = QVBoxLayout(canvas_panel)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)
        canvas_layout.addWidget(self.canvas, stretch=1)
        canvas_layout.addSpacing(6)
        frame_nav_row = QHBoxLayout()
        frame_nav_row.setContentsMargins(0, 0, 0, 0)
        frame_nav_row.setSpacing(4)
        self.play_btn = QPushButton(">")
        self.play_btn.setEnabled(False)
        self.play_btn.setFixedWidth(26)
        self.play_btn.setFixedHeight(30)
        self.frame_first_btn = QPushButton("|<")
        self.frame_first_btn.setEnabled(False)
        self.frame_first_btn.setFixedWidth(26)
        self.frame_first_btn.setFixedHeight(30)
        self.frame_prev_btn = QPushButton("<")
        self.frame_prev_btn.setEnabled(False)
        self.frame_prev_btn.setFixedWidth(26)
        self.frame_prev_btn.setFixedHeight(30)
        self.frame_prev_btn.setAutoRepeat(True)
        self.frame_prev_btn.setAutoRepeatDelay(250)
        self.frame_prev_btn.setAutoRepeatInterval(40)
        self.frame_next_btn = QPushButton(">")
        self.frame_next_btn.setEnabled(False)
        self.frame_next_btn.setFixedWidth(26)
        self.frame_next_btn.setFixedHeight(30)
        self.frame_next_btn.setAutoRepeat(True)
        self.frame_next_btn.setAutoRepeatDelay(250)
        self.frame_next_btn.setAutoRepeatInterval(40)
        self.frame_last_btn = QPushButton(">|")
        self.frame_last_btn.setEnabled(False)
        self.frame_last_btn.setFixedWidth(26)
        self.frame_last_btn.setFixedHeight(30)
        frame_nav_row.addWidget(self.frame_first_btn, stretch=0)
        frame_nav_row.addWidget(self.frame_prev_btn, stretch=0)
        frame_nav_row.addWidget(self.play_btn, stretch=0)
        frame_nav_row.addWidget(self.frame_next_btn, stretch=0)
        frame_nav_row.addWidget(self.frame_last_btn, stretch=0)
        self._init_transport_icons()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setFixedHeight(26)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setPageStep(1)
        self.frame_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 12px;
                background: #6b6b6b;
                border-radius: 6px;
            }
            QSlider::sub-page:horizontal {
                background: #9a9a9a;
                border-radius: 6px;
            }
            QSlider::handle:horizontal {
                background: #e6e6e6;
                border: 1px solid #4a4a4a;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            """
        )
        frame_nav_row.addWidget(self.frame_slider, stretch=1)
        canvas_layout.addLayout(frame_nav_row, stretch=0)
        root_layout.addWidget(canvas_panel, stretch=1)

        self.statusBar().showMessage("Ready")
        self._prev_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self._next_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self._prev_shortcut.setEnabled(False)
        self._next_shortcut.setEnabled(False)
        self._arrow_repeat_timer = QTimer(self)
        self._arrow_repeat_timer.timeout.connect(self._on_arrow_repeat_timeout)
        self._arrow_repeat_direction = 0
        self._arrow_repeat_started = False
        QApplication.instance().installEventFilter(self)
        self._wire_actions()
        self._model = ModelIntegration()
        self._last_prediction = None
        self._inference_thread = None
        self._inference_worker = None
        self._inference_box = None
        self._inference_dialog = None
        self._sequence_paths = []
        self._sequence_index = -1
        self._sequence_base_dir = None
        self._mode = "none"
        self._video_path = None
        self._video_paths = []
        self._video_list_index = -1
        self._video_capture = None
        self._video_frame_count = 0
        self._video_fps = 0.0
        self._video_frame_index = -1
        self._video_base_dir = None
        self._video_frame_cache = OrderedDict()
        self._video_decode_pos = -1
        self._video_cache_limit = 9
        self._video_use_random_seek = False
        self._last_image_input_dir = ""
        self._last_image_output_dir = ""
        self._last_video_input_dir = ""
        self._last_video_output_dir = ""
        self._last_load_kind = "image"
        self._load_persisted_paths()
        self._update_output_dir_tooltip()
        # Recomputing on every mask change would stall drawing, so changes are
        # collected and the measurements refresh once the mask has been still
        # for a moment.
        self._measurement_logs = {}
        self._roi_stores = {}
        self._rotation_stores = {}
        # (height, width) of the image on screen as stored, before any rotation,
        # and the angle it is shown at.
        self._original_shape = None
        self._shown_angle = 0.0
        self._rotation_preview = None
        # Steps that changed several files at once (Apply ROI to all, Clear all
        # ROIs), kept apart from the canvas's own history of the image on screen.
        self._bulk_undo = []
        self._bulk_redo = []
        self._calibration_changed = False
        self._measurement_timer = QTimer(self)
        self._measurement_timer.setSingleShot(True)
        self._measurement_timer.setInterval(250)
        self._measurement_timer.timeout.connect(self._update_measurements)
        self.canvas.mask_changed.connect(self._measurement_timer.start)
        settings = self._settings()
        for spin, key in (
            (self.px_per_mm_x_spin, "measurements/px_per_mm_x"),
            (self.px_per_mm_y_spin, "measurements/px_per_mm_y"),
        ):
            spin.setValue(float(settings.value(key, 0.0, float) or 0.0))
            spin.valueChanged.connect(self._on_px_per_mm_changed)
        self._build_calibration_dialog()
        self.h_lines_mm_spin.setValue(
            float(settings.value("calibration/h_lines_mm", 0.0, float) or 0.0)
        )
        self.v_lines_mm_spin.setValue(
            float(settings.value("calibration/v_lines_mm", 0.0, float) or 0.0)
        )
        saved_lines = str(settings.value("calibration/lines", "", str) or "").split(",")
        if len(saved_lines) == 4:
            try:
                y1, y2, x1, x2 = (float(v) for v in saved_lines)
                self.canvas.set_calibration_lines({"h": [y1, y2], "v": [x1, x2]})
            except ValueError:
                pass
        self.calibrate_btn.clicked.connect(self._start_calibration)
        self.canvas.roi_drawn.connect(self._on_roi_drawn)
        self.canvas.roi_changed.connect(self._save_current_roi)
        self.canvas.delete_requested.connect(self._on_delete_requested)
        self.view_picker.setCurrentIndex(
            0 if settings.value("measurements/view", "region", str) == "full" else 1
        )
        self.knee_picker.setCurrentIndex(
            max(0, self.knee_picker.findData(
                str(settings.value("measurements/knee_side", "right", str))
            ))
        )
        self.view_picker.currentIndexChanged.connect(self._on_measurement_view_changed)
        self.knee_picker.currentIndexChanged.connect(self._on_knee_side_changed)
        self.reset_centre_btn.clicked.connect(lambda: self.canvas.set_centre_x(None))
        self.canvas.set_knee_side(self.knee_picker.currentData())
        self._on_measurement_view_changed()
        self.canvas.image_loaded.connect(self._on_image_loaded_during_calibration)
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_playback)
        self._playback_interval_ms = 33
        self._batch_thread = None
        self._batch_worker = None
        self._batch_dialog = None
        self._gpu_warning = None
        self._init_device_picker()
        self._preload_model()
        self._init_model_picker()

        self._start_size = self._initial_window_size()
        self.setGeometry(10, 10, *self._start_size)
        self._center_window()

    def _create_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        self.load_files_action = QAction("Load files...", self)
        file_menu.addAction(self.load_files_action)
        self.segment_action = QAction("Segment image", self)
        file_menu.addAction(self.segment_action)
        self.segment_video_action = QAction("Segment video", self)
        self.segment_video_action.setEnabled(False)
        file_menu.addAction(self.segment_video_action)
        self.segment_all_action = QAction("Segment all files", self)
        file_menu.addAction(self.segment_all_action)
        file_menu.addSeparator()
        self.save_mask_action = QAction("Save masks", self)
        self.save_mask_action.setShortcut(QKeySequence.StandardKey.Save)
        file_menu.addAction(self.save_mask_action)
        self.delete_mask_action = QAction("Delete mask", self)
        file_menu.addAction(self.delete_mask_action)
        self.close_file_action = QAction("Close file", self)
        file_menu.addAction(self.close_file_action)
        file_menu.addSeparator()
        self.exit_action = QAction("Exit", self)
        file_menu.addAction(self.exit_action)

        edit_menu = menubar.addMenu("Edit")
        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        edit_menu.addAction(self.undo_action)
        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut(QKeySequence("Ctrl+Y"))
        edit_menu.addAction(self.redo_action)
        edit_menu.addSeparator()
        self.clear_rois_action = QAction("Clear all ROIs", self)
        edit_menu.addAction(self.clear_rois_action)
        self.rotate_action = QAction("Rotate image", self)
        edit_menu.addAction(self.rotate_action)

        tools_menu = menubar.addMenu("Tools")
        self.select_pan_action = QAction("Select", self)
        tools_menu.addAction(self.select_pan_action)
        self.freehand_action = QAction("Freehand Line", self)
        tools_menu.addAction(self.freehand_action)
        self.polyline_action = QAction("Segmented Line", self)
        tools_menu.addAction(self.polyline_action)
        self.paint_action = QAction("Paint Brush", self)
        tools_menu.addAction(self.paint_action)
        self.eraser_action = QAction("Eraser", self)
        tools_menu.addAction(self.eraser_action)

    def _results_dir(self, base_dir):
        """Folder the masks of the selected model are kept in.

        The subfolder "<model>_predictions" of the selected output folder, or the
        output folder itself when no model is selected.
        """
        if not base_dir:
            return None
        model = self._model.current_model()
        if not model:
            return str(base_dir)
        return str(Path(base_dir) / f"{model}{PREDICTIONS_SUFFIX}")

    def _measurement_file(self, results_dir):
        """(folder, file name) of the measurements file that goes with a results
        folder.

        The masks of a model sit in "<model>_predictions"; its measurements sit
        beside that folder as "measurements_<model>.csv". Masks kept directly in
        the selected output folder go with its "measurements.csv".
        """
        folder = Path(results_dir)
        selected = (self._sequence_base_dir, self._video_base_dir)
        if str(results_dir) in selected or not folder.name.endswith(PREDICTIONS_SUFFIX):
            return folder, MEASUREMENTS_FILE_NAME
        model = folder.name[: -len(PREDICTIONS_SUFFIX)]
        return folder.parent, f"measurements_{model}.csv"

    @property
    def _sequence_output_dir(self):
        return self._results_dir(self._sequence_base_dir)

    @_sequence_output_dir.setter
    def _sequence_output_dir(self, selected_dir):
        self._sequence_base_dir = selected_dir

    @property
    def _video_output_dir(self):
        return self._results_dir(self._video_base_dir)

    @_video_output_dir.setter
    def _video_output_dir(self, selected_dir):
        self._video_base_dir = selected_dir

    def _base_dir_of(self, results_dir):
        """The selected output folder a results folder belongs to."""
        for base_dir in (self._sequence_base_dir, self._video_base_dir):
            if base_dir and str(results_dir) == self._results_dir(base_dir):
                return str(base_dir)
        return str(results_dir)

    def _settings(self):
        return QSettings("UltAI", "UltAI Viewer")

    def _load_persisted_paths(self):
        settings = self._settings()
        self._last_image_input_dir = str(settings.value("paths/image_input_dir", "", str) or "")
        self._last_image_output_dir = str(settings.value("paths/image_output_dir", "", str) or "")
        self._last_video_input_dir = str(settings.value("paths/video_input_dir", "", str) or "")
        self._last_video_output_dir = str(settings.value("paths/video_output_dir", "", str) or "")
        self._last_load_kind = str(settings.value("paths/last_load_kind", "image", str) or "image")

    def _save_persisted_paths(self):
        settings = self._settings()
        settings.setValue("paths/image_input_dir", self._last_image_input_dir)
        settings.setValue("paths/image_output_dir", self._last_image_output_dir)
        settings.setValue("paths/video_input_dir", self._last_video_input_dir)
        settings.setValue("paths/video_output_dir", self._last_video_output_dir)
        settings.setValue("paths/last_load_kind", self._last_load_kind)

    def _on_px_per_mm_changed(self):
        settings = self._settings()
        settings.setValue("measurements/px_per_mm_x", self.px_per_mm_x_spin.value())
        settings.setValue("measurements/px_per_mm_y", self.px_per_mm_y_spin.value())
        if self._current_frame_key() is not None:
            self._calibration_changed = True
        self._update_measurements()

    def _build_calibration_dialog(self):
        # Shown without blocking the main window, so the lines on the image can
        # be dragged while the dialog is open.
        self.calibration_dialog = QDialog(self)
        self.calibration_dialog.setWindowTitle("Calibrate")
        dialog_layout = QVBoxLayout(self.calibration_dialog)
        hint = QLabel(
            "Drag the lines on the image onto a known distance, then enter the "
            "physical distance between each pair of lines."
        )
        hint.setWordWrap(True)
        dialog_layout.addWidget(hint)
        form = QFormLayout()
        self.h_lines_mm_spin = QDoubleSpinBox()
        self.v_lines_mm_spin = QDoubleSpinBox()
        for spin in (self.h_lines_mm_spin, self.v_lines_mm_spin):
            spin.setRange(0.0, 10000.0)
            spin.setDecimals(2)
            spin.setSuffix(" mm")
            spin.setSpecialValueText("Not set")
        form.addRow("Distance between horizontal lines:", self.h_lines_mm_spin)
        form.addRow("Distance between vertical lines:", self.v_lines_mm_spin)
        dialog_layout.addLayout(form)
        self.calibration_buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.calibration_buttons.accepted.connect(self._apply_calibration)
        self.calibration_buttons.rejected.connect(self.calibration_dialog.reject)
        self.calibration_dialog.rejected.connect(self._end_calibration)
        dialog_layout.addWidget(self.calibration_buttons)

    def _start_calibration(self):
        if not self.canvas.start_calibration():
            QMessageBox.information(
                self, "Calibrate", "Load an image or video before calibrating."
            )
            return
        self.calibrate_btn.setEnabled(False)
        self.calibration_dialog.show()
        self.statusBar().showMessage("Calibrating: drag the lines on the image")

    def _end_calibration(self):
        self.canvas.stop_calibration()
        self.calibration_dialog.hide()
        self.calibrate_btn.setEnabled(True)
        lines = self.canvas.calibration_lines()
        if lines is not None:
            self._settings().setValue(
                "calibration/lines",
                ",".join(f"{v:.3f}" for v in lines["h"] + lines["v"]),
            )

    def _on_image_loaded_during_calibration(self, source_name):
        # An empty name means the image was closed, which leaves nothing to
        # place the lines on.
        if not source_name and self.canvas.is_calibrating():
            self._end_calibration()

    def _apply_calibration(self):
        h_mm = self.h_lines_mm_spin.value()
        v_mm = self.v_lines_mm_spin.value()
        h_gap_px, v_gap_px = self.canvas.calibration_gaps()
        if not h_mm and not v_mm:
            QMessageBox.information(
                self,
                "Calibrate",
                "Enter the real distance for at least one pair of lines.",
            )
            return
        if (h_mm and h_gap_px == 0) or (v_mm and v_gap_px == 0):
            QMessageBox.information(
                self,
                "Calibrate",
                "The two lines of a pair are on top of each other. "
                "Move them apart first.",
            )
            return
        # The horizontal lines are spaced down the image, so they give the y
        # resolution; the vertical lines are spaced across it and give x.
        if h_mm:
            self.px_per_mm_y_spin.setValue(h_gap_px / h_mm)
        if v_mm:
            self.px_per_mm_x_spin.setValue(v_gap_px / v_mm)
        settings = self._settings()
        settings.setValue("calibration/h_lines_mm", h_mm)
        settings.setValue("calibration/v_lines_mm", v_mm)
        self._end_calibration()
        self.statusBar().showMessage("Calibration applied")

    def _on_measurement_view_changed(self):
        region_view = bool(self.view_picker.currentData())
        self._settings().setValue(
            "measurements/view", "region" if region_view else "full"
        )
        self.canvas.set_region_view(region_view)
        self.knee_picker.setEnabled(region_view)
        self.reset_centre_btn.setEnabled(region_view)
        for region in self._result_columns:
            visible = (region != "whole") if region_view else (region == "whole")
            self._result_headers[region].setVisible(visible and region_view)
            for (value_region, _), label in self._result_values.items():
                if value_region == region:
                    label.setVisible(visible)
        self._update_measurements()

    def _on_knee_side_changed(self):
        knee_side = self.knee_picker.currentData()
        self._settings().setValue("measurements/knee_side", knee_side)
        self.canvas.set_knee_side(knee_side)

    def _current_frame_key(self):
        """(output folder, file name, frame) of the image or frame on screen, or None."""
        if self._mode == "video" and self._video_path:
            output_dir = self._video_output_dir
            file_name, frame = Path(self._video_path).name, self._video_frame_index
        elif self._mode == "sequence" and 0 <= self._sequence_index < len(
            self._sequence_paths
        ):
            output_dir = self._sequence_output_dir
            file_name, frame = Path(self._sequence_paths[self._sequence_index]).name, ""
        else:
            return None
        if not output_dir:
            return None
        return str(output_dir), file_name, str(frame)

    def _restore_frame_inputs(self):
        """Bring back the ROI, knee side and centre point of the frame on screen.

        A frame with no saved knee side keeps the one already chosen, and one with
        no centre point of its own uses the suggested centre.
        """
        key = self._current_frame_key()
        if key is None:
            return
        output_dir, file_name, frame = key
        try:
            log = self._measurement_log(self._shown_result_dir())
            row = log.get(file_name, frame)
            # A saved result keeps the ROI it was measured with; rois.csv only
            # supplies the ROI of a frame that has no saved result yet.
            # A row measured at another rotation describes a view that is no
            # longer the one on screen, so its ROI and centre point do not apply.
            if row is not None and log.rotation(file_name, frame) != self._shown_angle:
                row = None
            if row is not None:
                self.canvas.set_roi(log.limits(file_name, frame))
            else:
                self.canvas.set_roi(self._roi_store(output_dir).get(file_name, frame))
        except OSError:
            return
        if row is None:
            return
        knee_index = self.knee_picker.findData(row.get("knee_side"))
        if knee_index >= 0:
            self.knee_picker.setCurrentIndex(knee_index)
        if row.get("centre_set_by") == "user":
            try:
                self.canvas.set_centre_x(float(row["centre_x"]))
            except (TypeError, ValueError):
                pass

    def _shown_result_dir(self):
        """Folder holding the saved result of the image or frame on screen.

        The selected model's results folder, or the selected output folder itself
        when only that holds a mask for it (results saved before models had
        folders of their own).
        """
        if self._mode == "video":
            results_dir, base_dir = self._video_output_dir, self._video_base_dir
        else:
            results_dir, base_dir = self._sequence_output_dir, self._sequence_base_dir
        shown = self._shown_mask_path()
        if shown is not None and base_dir and results_dir != str(base_dir):
            if Path(results_dir) not in Path(shown).parents:
                return str(base_dir)
        return results_dir

    def _shown_mask_path(self):
        """Saved mask file of the image or frame on screen, or None."""
        if self._mode == "video" and self._video_path and self._video_frame_index >= 0:
            for base_dir in (self._video_output_dir, self._video_base_dir):
                if base_dir:
                    path = (
                        Path(base_dir)
                        / Path(self._video_path).stem
                        / f"frame_{int(self._video_frame_index):06d}.png"
                    )
                    if path.is_file():
                        return path
            return None
        if self._mode == "sequence" and 0 <= self._sequence_index < len(
            self._sequence_paths
        ):
            image_path = Path(self._sequence_paths[self._sequence_index])
            for base_dir in (self._sequence_output_dir, self._sequence_base_dir):
                if not base_dir:
                    continue
                found = find_mask_files(base_dir, image_path, self._sequence_paths)
                if found:
                    return found[0]
        return None

    def _restore_folder_inputs(self):
        """Show the ROI, knee side and centre point the output folder holds for the
        frame on screen, in place of those of the folder it was switched from."""
        self.canvas.set_centre_x(None)
        self._restore_frame_inputs()

    def _roi_store(self, output_dir):
        # rois.csv is shared by all models, so it sits in the selected output
        # folder rather than in the results folder of one model.
        key = self._base_dir_of(output_dir)
        if key not in self._roi_stores:
            self._roi_stores[key] = RoiStore(key)
        return self._roi_stores[key]

    def _save_current_roi(self):
        """Store the ROI on screen, or its absence, for this image or frame."""
        key = self._current_frame_key()
        if key is None:
            return False
        output_dir, file_name, frame = key
        box = self.canvas.roi_box()
        try:
            store = self._roi_store(output_dir)
            if box is None:
                store.remove(file_name, frame)
            else:
                store.set(file_name, frame, box)
            store.save()
        except OSError as exc:
            self.statusBar().showMessage(f"Could not save the ROI: {exc}")
            return False
        return True

    def _rotation_store(self, output_dir):
        # Like rois.csv, rotations.csv is shared by all models.
        key = self._base_dir_of(output_dir)
        if key not in self._rotation_stores:
            self._rotation_stores[key] = RotationStore(key)
        return self._rotation_stores[key]

    def _angle_of(self, output_dir, file_name):
        """Angle, in degrees clockwise, a file is viewed and measured at."""
        if not output_dir:
            return 0.0
        try:
            return self._rotation_store(output_dir).get(file_name)
        except OSError:
            return 0.0

    def _current_angle(self):
        key = self._current_frame_key()
        return 0.0 if key is None else self._angle_of(key[0], key[1])

    def _read_saved_mask(self, mask_path, angle=0.0):
        """A saved mask file as a bool array, turned the way its image is viewed."""
        raw = self.canvas._read_mask(str(mask_path))
        return rotate_mask(raw >= (128 if raw.max() > 1 else 0.5), angle)

    def _show_image(self, image, source_name):
        """Put an image as stored on the canvas, turned by its file's angle."""
        self._original_shape = image.shape[:2]
        self._shown_angle = self._current_angle()
        self.canvas.load_image_array(rotate_image(image, self._shown_angle), source_name)

    def _show_saved_mask(self):
        """Put the saved mask of the image or frame on screen on the canvas."""
        mask_path = self._shown_mask_path()
        if mask_path is None:
            self.canvas.clear_mask()
        elif not self._shown_angle:
            self.canvas.load_mask(str(mask_path))
        else:
            try:
                mask = self._read_saved_mask(mask_path, self._shown_angle)
            except Exception as exc:
                QMessageBox.critical(self, "Load failed", str(exc))
                return
            self.canvas.set_mask(mask.astype(np.float32))

    def _rotate_dialog(self):
        """Let the user turn the image on screen, previewing while a slider moves."""
        if not self._begin_rotation():
            return
        start = self._current_angle()
        dialog = QDialog(self)
        dialog.setWindowTitle("Rotate image")
        dialog_layout = QVBoxLayout(dialog)
        hint = QLabel(
            "Move the slider to turn the image. Positive angles turn it clockwise. "
            "The image file itself is not changed."
        )
        hint.setWordWrap(True)
        dialog_layout.addWidget(hint)
        row = QHBoxLayout()
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(-180, 180)
        slider.setMinimumWidth(320)
        angle_spin = QDoubleSpinBox()
        angle_spin.setRange(-180.0, 180.0)
        angle_spin.setDecimals(1)
        angle_spin.setSingleStep(0.1)
        angle_spin.setSuffix("°")
        row.addWidget(slider, stretch=1)
        row.addWidget(angle_spin)
        dialog_layout.addLayout(row)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Reset
        )
        dialog_layout.addWidget(buttons)

        def on_slider(value):
            # The slider moves in whole degrees; the box keeps a finer angle
            # until the slider is moved to another degree.
            if round(angle_spin.value()) != value:
                angle_spin.setValue(float(value))

        def on_spin(value):
            slider.blockSignals(True)
            slider.setValue(round(value))
            slider.blockSignals(False)
            self._preview_rotation(value)

        slider.setValue(round(start))
        angle_spin.setValue(start)
        slider.valueChanged.connect(on_slider)
        angle_spin.valueChanged.connect(on_spin)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        buttons.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(
            lambda: angle_spin.setValue(0.0)
        )
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        self._finish_rotation(angle_spin.value() if accepted else None)

    def _begin_rotation(self):
        """Save what is on screen and keep the stored image and mask for previews."""
        if self._batch_thread and self._batch_thread.isRunning():
            return False
        if self.canvas.image is None or self._current_frame_key() is None:
            QMessageBox.information(self, "No files", "Load files first.")
            return False
        self._stop_playback()
        try:
            if self._mode == "video":
                self._stash_video_mask_for_current_frame()
                image = self._decode_video_frame(self._video_frame_index)
            else:
                self._save_sequence_mask_if_needed()
                image = self.canvas._read_image(self._sequence_paths[self._sequence_index])
            mask_path = self._shown_mask_path()
            mask = None if mask_path is None else self._read_saved_mask(mask_path)
        except Exception as exc:
            QMessageBox.warning(self, "Rotate image", str(exc))
            return False
        if image is None:
            return False
        self._rotation_preview = {
            "image": image,
            "mask": mask,
            "source": self.canvas.image_path or "",
        }
        return True

    def _preview_rotation(self, angle):
        """Show the image and its saved mask turned by angle, without storing it."""
        preview = self._rotation_preview
        if preview is None:
            return
        self.canvas.load_image_array(
            rotate_image(preview["image"], angle), preview["source"]
        )
        if preview["mask"] is not None:
            self.canvas.set_mask(rotate_mask(preview["mask"], angle).astype(np.float32))

    def _finish_rotation(self, angle):
        """Keep an angle chosen in a preview, or with None go back to the stored one."""
        self._rotation_preview = None
        key = self._current_frame_key()
        if key is None:
            return
        output_dir, file_name, _ = key
        changed = angle is not None and round(float(angle), 1) != self._current_angle()
        if changed:
            try:
                store = self._rotation_store(output_dir)
                store.set(file_name, angle)
                store.save()
                # An ROI is a box on the view, and the view changes size as the
                # image turns, so the ROIs of the file are removed.
                rois = self._roi_store(output_dir)
                rois.remove_files({file_name})
                rois.save()
                log = self._measurement_log(output_dir)
                log.clear_limits({file_name})
                log.save()
            except OSError as exc:
                QMessageBox.warning(self, "Rotate image", str(exc))
        self._show_stored_image()
        self._show_saved_mask()
        self._restore_frame_inputs()
        if not changed:
            return
        if self._mode == "video":
            self._remeasure_video_frames()
        elif self._canvas_has_mask() and self._shown_result_dir() == output_dir:
            self._record_measurement(output_dir, file_name)
        self.statusBar().showMessage(f"Rotation of {file_name}: {self._shown_angle:g}°")

    def _rois_for_segmenting(self, output_dir):
        """ROI of each image and frame for a background run, keyed (file, frame).

        The boxes of rois.csv, except that a frame with a saved result in the
        results folder keeps the ROI recorded with that result.
        """
        boxes = self._roi_store(output_dir).boxes()
        log = MeasurementLog(*self._measurement_file(output_dir))
        for file_name, frame in log.keys():
            if log.rotation(file_name, frame) != self._angle_of(output_dir, file_name):
                continue
            recorded = log.limits(file_name, frame)
            if recorded is None:
                boxes.pop((file_name, frame), None)
            else:
                boxes[(file_name, frame)] = recorded
        return boxes

    def _other_frames(self):
        """Every loaded image and video frame other than the one on screen.

        Returns (source path, file name, frame) tuples per output folder.
        """
        current = self._current_frame_key()
        found = {}
        if self._sequence_output_dir and self._sequence_paths:
            folder = str(self._sequence_output_dir)
            for image_path in self._sequence_paths:
                name = Path(image_path).name
                if (folder, name, "") != current:
                    found.setdefault(folder, []).append((image_path, name, ""))
        if self._video_output_dir and self._video_paths:
            folder = str(self._video_output_dir)
            for video_path in self._video_paths:
                capture = cv2.VideoCapture(video_path)
                frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                capture.release()
                name = Path(video_path).name
                for frame in range(frame_count):
                    if (folder, name, str(frame)) != current:
                        found.setdefault(folder, []).append((video_path, name, frame))
        return found

    def _on_roi_drawn(self, previous):
        if self._offer_roi_to_others(can_discard=True) == QMessageBox.StandardButton.Cancel:
            self.canvas.drop_last_history()
            self.canvas.set_roi(previous)
            self._save_current_roi()

    def _undo(self):
        """Take back the latest step: an edit on the canvas or a step across files."""
        step = self._bulk_undo[-1] if self._bulk_undo else None
        canvas_is_later = (
            step is not None
            and step["resets"] == self.canvas.history_resets
            and self.canvas.history_depth() > step["depth"]
        )
        if step is None or canvas_is_later:
            self.canvas.undo()
            return
        self._bulk_undo.pop()
        self._swap_bulk(step, "prev")
        step["edit_count"] = self.canvas.edit_count
        self._bulk_redo.append(step)

    def _redo(self):
        if self.canvas.can_redo():
            self.canvas.redo()
            return
        if not self._bulk_redo:
            return
        step = self._bulk_redo.pop()
        if step["edit_count"] != self.canvas.edit_count:
            # An edit made since the step was undone starts a different history.
            self._bulk_redo.clear()
            return
        self._swap_bulk(step, "after")
        self._record_bulk(step)

    def _record_bulk(self, step):
        step["resets"] = self.canvas.history_resets
        step["depth"] = self.canvas.history_depth()
        self._bulk_undo.append(step)

    def _bulk_entry(self, folder, name, frame, mask_path=None):
        """What one image or frame holds before a step across files changes it."""
        entry = {
            "folder": str(folder),
            "base": self._base_dir_of(folder),
            "name": name,
            "frame": str(frame),
            "mask_path": None if mask_path is None else Path(mask_path),
        }
        self._fill_bulk_entry(entry, "prev")
        return entry

    def _fill_bulk_entry(self, entry, side):
        name, frame = entry["name"], entry["frame"]
        row = self._measurement_log(entry["folder"]).get(name, frame)
        entry[f"{side}_roi"] = self._roi_store(entry["base"]).get(name, frame)
        entry[f"{side}_row"] = None if row is None else dict(row)
        path = entry["mask_path"]
        entry[f"{side}_bytes"] = (
            path.read_bytes() if path is not None and path.is_file() else None
        )

    def _swap_bulk(self, step, side):
        """Put the files of a step across files into their "prev" or "after" state.

        An image or frame whose ROI or mask file is no longer what the step left
        it as was edited since, and is left alone.
        """
        if self._mode == "video":
            self._stash_video_mask_for_current_frame()
        elif self._mode == "sequence":
            self._save_sequence_mask_if_needed()
        other = "after" if side == "prev" else "prev"
        current = self._current_frame_key()
        on_screen = False
        skipped = 0
        touched = {}
        try:
            for entry in step["entries"]:
                name, frame = entry["name"], entry["frame"]
                store = self._roi_store(entry["base"])
                log = self._measurement_log(entry["folder"])
                path = entry["mask_path"]
                held = path.read_bytes() if path is not None and path.is_file() else None
                if store.get(name, frame) != entry[f"{other}_roi"] or (
                    path is not None and held != entry[f"{other}_bytes"]
                ):
                    skipped += 1
                    continue
                if entry[f"{side}_roi"] is None:
                    store.remove(name, frame)
                else:
                    store.set(name, frame, entry[f"{side}_roi"])
                if path is not None and held != entry[f"{side}_bytes"]:
                    if entry[f"{side}_bytes"] is None:
                        path.unlink()
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(entry[f"{side}_bytes"])
                log.put(name, frame, entry[f"{side}_row"])
                touched[id(store)] = store
                touched[id(log)] = log
                if current == (entry["folder"], name, frame):
                    on_screen = True
            for item in touched.values():
                item.save()
        except OSError as exc:
            QMessageBox.warning(self, step["title"], str(exc))
        if on_screen:
            self._show_saved_result()
        verb = "Undid" if side == "prev" else "Redid"
        message = f"{verb}: {step['title']}"
        if skipped:
            message += f"; {skipped} edited since were left as they are"
        self.statusBar().showMessage(message)

    def _offer_roi_to_others(self, can_discard=False):
        """Ask whether the ROI on screen should go to every other image and frame.

        Returns the button chosen, or None when nothing was asked. can_discard adds
        a Cancel button for throwing away a box that was just drawn.
        """
        if self.canvas.roi_box() is None or not self._save_current_roi():
            return None
        targets = self._other_frames()
        count = sum(len(frames) for frames in targets.values())
        if count == 0:
            return None
        others = (
            "the 1 other image or frame"
            if count == 1
            else f"all {count} other images and frames"
        )
        text = (
            f"Apply this ROI to {others}?\n\n"
            "It replaces any ROI they have, and where one of them already has a "
            "saved mask, the mask is trimmed to the ROI."
        )
        buttons = QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        if can_discard:
            text += "\n\nCancel discards the box you just drew."
            buttons |= QMessageBox.StandardButton.Cancel
        choice = QMessageBox.question(
            self, "Apply ROI", text, buttons, QMessageBox.StandardButton.No
        )
        if choice == QMessageBox.StandardButton.Yes:
            self._apply_roi_to(targets, self.canvas.roi_box(), count)
        return choice

    def _apply_roi_to(self, targets, box, count):
        """Give a box to the frames listed by _other_frames, trimming saved masks."""
        progress = QProgressDialog("Applying ROI...", None, 0, count, self)
        progress.setWindowTitle("Apply ROI")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(400)
        done = 0
        applied = 0
        entries = []
        try:
            for folder, frames in targets.items():
                store = self._roi_store(folder)
                log = self._measurement_log(folder)
                capture = None
                capture_path = None
                position = 0
                try:
                    for source_path, name, frame in frames:
                        done += 1
                        if done % 25 == 0:
                            progress.setValue(done)
                        if frame == "":
                            mask_path = self._find_sequence_mask_path(source_path)
                            if mask_path and Path(mask_path).resolve() == Path(
                                source_path
                            ).resolve():
                                mask_path = None
                        else:
                            mask_path = self._video_mask_path(source_path, frame)
                        if mask_path is None or not Path(mask_path).is_file():
                            entries.append(self._bulk_entry(folder, name, frame))
                            store.set(name, frame, box)
                            applied += 1
                            continue
                        entries.append(self._bulk_entry(folder, name, frame, mask_path))
                        if frame == "":
                            image = self.canvas._read_image(source_path)
                        else:
                            if capture_path != source_path:
                                if capture is not None:
                                    capture.release()
                                capture = cv2.VideoCapture(source_path)
                                capture_path, position = source_path, 0
                            # Frames are read in order because seeking to a frame
                            # number is not exact in every video format.
                            while position < frame:
                                capture.grab()
                                position += 1
                            success, image = capture.read()
                            position += 1
                            if not success:
                                image = None
                            elif image.ndim == 3 and image.shape[2] == 3:
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        if self._trim_saved_mask(
                            Path(mask_path),
                            box,
                            image,
                            log,
                            name,
                            frame,
                            self._angle_of(folder, name),
                        ):
                            store.set(name, frame, box)
                            applied += 1
                finally:
                    if capture is not None:
                        capture.release()
                    store.save()
                    log.save()
        except Exception as exc:
            progress.close()
            QMessageBox.warning(self, "Apply ROI", str(exc))
            return
        for entry in entries:
            self._fill_bulk_entry(entry, "after")
        self._bulk_redo.clear()
        self._record_bulk({"title": "Apply ROI to all", "entries": entries})
        progress.setValue(count)
        message = f"ROI applied to {applied} images and frames"
        if applied < count:
            message += f"; {count - applied} left out because the ROI does not fit them"
        self.statusBar().showMessage(message)

    def _trim_saved_mask(self, mask_path, box, image, log, file_name, frame, angle=0.0):
        """Blank a saved mask outside a box and bring its measurements up to date.

        The box is in the view of the file, which is turned by angle. Returns
        False, changing nothing, when the box does not fit the mask.
        """
        stored_shape = self.canvas._read_mask(str(mask_path)).shape[:2]
        mask = self._read_saved_mask(mask_path, angle)
        if image is not None:
            image = rotate_image(image, angle)
        fitted = box_to_segment(mask, box)
        if fitted is None:
            return False
        left, right, top, bottom = fitted
        trimmed = np.zeros_like(mask)
        trimmed[top:bottom, left:right] = mask[top:bottom, left:right]
        if not trimmed.any():
            mask_path.unlink()
            log.remove(file_name, frame)
            return True
        write_mask_file(mask_path, trimmed, angle, stored_shape)
        log.set(
            file_name,
            frame,
            image,
            trimmed,
            self.px_per_mm_x_spin.value(),
            self.px_per_mm_y_spin.value(),
            knee_side=self.knee_picker.currentData(),
            prefer_saved=True,
            limits=fitted,
            rotation=angle,
        )
        return True

    def _on_delete_requested(self, kind):
        if kind == "mask":
            self._delete_current_mask()
        elif kind == "roi":
            self.canvas.set_roi(None, record=True)
            if self._save_current_roi():
                self.statusBar().showMessage("ROI deleted")
        elif kind == "apply_roi":
            if self._offer_roi_to_others() is None:
                self.statusBar().showMessage("No other images or frames are loaded")
        elif kind == "all_rois":
            self._clear_all_rois()

    def _clear_all_rois(self):
        """Remove the ROI of every loaded image and video, after asking."""
        if self._batch_thread and self._batch_thread.isRunning():
            return
        loaded = {}
        for folder, paths in (
            (self._sequence_output_dir, self._sequence_paths),
            (self._video_output_dir, self._video_paths),
        ):
            if folder and paths:
                loaded.setdefault(str(folder), set()).update(
                    Path(path).name for path in paths
                )
        if not loaded and self.canvas.roi_box() is None:
            return
        choice = QMessageBox.question(
            self,
            "Clear all ROIs",
            "Remove the ROI from every loaded image and frame?\n\n"
            "Saved masks stay as they are.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return
        removed = 0
        entries = []
        try:
            for folder, names in loaded.items():
                store = self._roi_store(folder)
                log = self._measurement_log(folder)
                keys = {key for key in store.boxes() if key[0] in names}
                keys.update(
                    key
                    for key in log.keys()
                    if key[0] in names and log.limits(*key) is not None
                )
                entries.extend(self._bulk_entry(folder, *key) for key in sorted(keys))
                removed += store.remove_files(names)
                store.save()
                log.clear_limits(names)
                log.save()
        except OSError as exc:
            QMessageBox.warning(self, "Clear all ROIs", str(exc))
            return
        for entry in entries:
            self._fill_bulk_entry(entry, "after")
        self._bulk_redo.clear()
        self._record_bulk({"title": "Clear all ROIs", "entries": entries})
        self.canvas.set_roi(None)
        self.statusBar().showMessage(f"Removed {removed} ROIs")

    def _update_measurements(self):
        mask = self.canvas.visible_mask()
        result = measure(
            self.canvas.image,
            None if mask is None else mask >= 0.5,
            self.px_per_mm_x_spin.value(),
            self.px_per_mm_y_spin.value(),
            self.canvas.centre_x,
            self.canvas.knee_side,
            self.canvas.roi_box(),
        )
        unit = result["unit"]
        row_titles = {
            "area": f"Area ({unit}²)",
            "length": f"Length ({unit})",
            "thickness": f"Thickness ({unit})",
            "echo_mean": "Echo intensity (AU)",
            "echo_sd": "Variation (AU)",
        }
        for measure_name, title in row_titles.items():
            self._result_row_labels[measure_name].setText(title)
        for (region, measure_name), label in self._result_values.items():
            value = result["regions"][region][measure_name]
            if np.isnan(value):
                label.setText("–")
            else:
                decimals = 1 if measure_name in ("echo_mean", "echo_sd") else 2
                label.setText(f"{value:.{decimals}f}")

    def _measurement_log(self, output_dir):
        # Logs are kept in memory so that stepping through video frames does not
        # re-read the file each time; they are dropped whenever a background
        # segmentation run has written to the same files.
        key = str(output_dir)
        if key not in self._measurement_logs:
            self._measurement_logs[key] = MeasurementLog(
                *self._measurement_file(output_dir)
            )
        return self._measurement_logs[key]

    def _record_measurement(self, output_dir, file_name, frame=""):
        """Store the measurements of the mask on screen next to its saved mask."""
        try:
            log = self._measurement_log(output_dir)
            log.set(
                file_name,
                frame,
                self.canvas.image,
                np.asarray(self.canvas.visible_mask()) >= 0.5,
                self.px_per_mm_x_spin.value(),
                self.px_per_mm_y_spin.value(),
                self.canvas.centre_x,
                self.canvas.knee_side,
                limits=self.canvas.roi_box(),
                rotation=self._shown_angle,
            )
            log.save()
        except OSError as exc:
            self.statusBar().showMessage(f"Could not save measurements: {exc}")

    def _saved_mask_matches(self, mask_path):
        """True when a saved mask file holds the same mask as the one on screen."""
        if mask_path is None or not Path(mask_path).is_file():
            return False
        try:
            stored = self._read_saved_mask(mask_path)
        except Exception:
            return False
        shown = np.asarray(self.canvas.visible_mask()) >= 0.5
        if self._shown_angle:
            # Turning a mask is not exactly reversible, so the file counts as
            # the same mask when it gives the mask on screen once turned, or
            # when the mask on screen gives the file once turned back.
            if stored.ndim != 2:
                return False
            if np.array_equal(rotate_mask(stored, self._shown_angle), shown):
                return True
            return bool(
                np.array_equal(
                    unrotate_mask(shown, self._shown_angle, stored.shape), stored
                )
            )
        saved = stored
        if saved.shape != shown.shape:
            if saved.ndim != 2:
                return False
            saved = cv2.resize(
                saved.astype(np.uint8),
                (shown.shape[1], shown.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return bool(np.array_equal(saved, shown))

    def _write_shown_mask(self, mask_path):
        """Save the mask on screen so that it lines up with the stored image."""
        write_mask_file(
            mask_path,
            np.asarray(self.canvas.visible_mask()) >= 0.5,
            self._shown_angle,
            self._original_shape,
        )

    def _measurement_is_current(self, output_dir, file_name, frame=""):
        """True when the saved row was measured with the inputs on screen."""
        try:
            return self._measurement_log(output_dir).matches_inputs(
                file_name,
                frame,
                self.canvas.centre_x,
                self.canvas.knee_side,
                self.canvas.roi_box(),
                self._shown_angle,
            )
        except OSError:
            return False

    def _remeasure_video_frames(self):
        """Measure every saved frame mask of the video on screen again.

        Each frame keeps the knee side, centre point and ROI its row records.
        """
        root = self._video_output_root_for_path(self._video_path)
        if root is None or not root.is_dir():
            return
        frames = sorted(
            int(path.stem.split("_")[-1])
            for path in root.glob("frame_*.png")
            if path.stem.split("_")[-1].isdigit()
        )
        if not frames:
            return
        name = Path(self._video_path).name
        progress = QProgressDialog("Updating measurements...", None, 0, len(frames), self)
        progress.setWindowTitle("Measurements")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(400)
        capture = cv2.VideoCapture(self._video_path)
        position = 0
        try:
            log = self._measurement_log(self._video_output_dir)
            for done, frame in enumerate(frames):
                if done % 25 == 0:
                    progress.setValue(done)
                # Frames are read in order because seeking to a frame number is
                # not exact in every video format.
                while position < frame:
                    capture.grab()
                    position += 1
                success, image = capture.read()
                position += 1
                if not success:
                    image = None
                elif image.ndim == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = self._load_saved_video_mask_for_frame(self._video_path, frame)
                if mask is None or not mask.any():
                    continue
                angle = self._angle_of(self._video_output_dir, name)
                mask = rotate_mask(mask, angle)
                if image is not None:
                    image = rotate_image(image, angle)
                log.set(
                    name,
                    frame,
                    image,
                    mask,
                    self.px_per_mm_x_spin.value(),
                    self.px_per_mm_y_spin.value(),
                    knee_side=self.knee_picker.currentData(),
                    prefer_saved=True,
                    limits=(
                        log.limits(name, frame)
                        if log.rotation(name, frame) == angle
                        else None
                    ),
                    rotation=angle,
                )
            log.save()
        except OSError as exc:
            self.statusBar().showMessage(f"Could not save measurements: {exc}")
        finally:
            capture.release()
            progress.setValue(len(frames))

    def _forget_measurement(self, output_dir, file_name, frame=""):
        try:
            log = self._measurement_log(output_dir)
            log.remove(file_name, frame)
            log.save()
        except OSError as exc:
            self.statusBar().showMessage(f"Could not save measurements: {exc}")

    def _transport_icon(self, name):
        icon_path = Path(__file__).resolve().parent.parent / "assets" / "icons" / f"{name}.svg"
        if not icon_path.exists():
            return None
        return QIcon(str(icon_path))

    def _init_transport_icons(self):
        icon_size = QSize(14, 14)
        self.play_btn.setText("")
        self.play_btn.setIconSize(icon_size)
        self.play_btn.setToolTip("Play/Pause")
        self.frame_first_btn.setText("")
        self.frame_first_btn.setIconSize(icon_size)
        self.frame_first_btn.setToolTip("First frame")
        self.frame_prev_btn.setText("")
        self.frame_prev_btn.setIconSize(icon_size)
        self.frame_prev_btn.setToolTip("Previous frame")
        self.frame_next_btn.setText("")
        self.frame_next_btn.setIconSize(icon_size)
        self.frame_next_btn.setToolTip("Next frame")
        self.frame_last_btn.setText("")
        self.frame_last_btn.setIconSize(icon_size)
        self.frame_last_btn.setToolTip("Last frame")

        first_icon = self._transport_icon("first")
        if first_icon is not None:
            self.frame_first_btn.setIcon(first_icon)
        else:
            self.frame_first_btn.setText("|<")

        prev_icon = self._transport_icon("prev")
        if prev_icon is not None:
            self.frame_prev_btn.setIcon(prev_icon)
        else:
            self.frame_prev_btn.setText("<")

        next_icon = self._transport_icon("next")
        if next_icon is not None:
            self.frame_next_btn.setIcon(next_icon)
        else:
            self.frame_next_btn.setText(">")

        last_icon = self._transport_icon("last")
        if last_icon is not None:
            self.frame_last_btn.setIcon(last_icon)
        else:
            self.frame_last_btn.setText(">|")

        self._play_btn_set_play()

    def _build_sidebar(self):
        panel = QWidget()
        panel.setMaximumWidth(self._sidebar_width)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        def configure_button(button):
            min_height = max(26, int(button.fontMetrics().height() * 1.6))
            button.setMinimumHeight(min_height)
            button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        layout.addWidget(QLabel("Files"))
        files_row = QHBoxLayout()
        self.load_btn = QPushButton("Load files")
        configure_button(self.load_btn)
        files_row.addWidget(self.load_btn)
        self.output_btn = QPushButton("Select output folder")
        configure_button(self.output_btn)
        files_row.addWidget(self.output_btn)
        layout.addLayout(files_row)

        self.file_combo = QComboBox()
        self.file_combo.setEnabled(False)
        layout.addWidget(self.file_combo)

        nav_row = QHBoxLayout()
        self.prev_btn = QPushButton("Prev")
        self.prev_btn.setEnabled(False)
        configure_button(self.prev_btn)
        nav_row.addWidget(self.prev_btn)
        self.next_btn = QPushButton("Next")
        self.next_btn.setEnabled(False)
        configure_button(self.next_btn)
        nav_row.addWidget(self.next_btn)
        layout.addLayout(nav_row)

        close_row = QHBoxLayout()
        self.close_btn = QPushButton("Close file")
        configure_button(self.close_btn)
        close_row.addWidget(self.close_btn)
        self.clear_files_btn = QPushButton("Clear files")
        configure_button(self.clear_files_btn)
        close_row.addWidget(self.clear_files_btn)
        layout.addLayout(close_row)

        layout.addSpacing(16)
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep3)

        layout.addWidget(QLabel("Image Analysis"))
        model_row = QHBoxLayout()
        self.model_picker = QComboBox()
        self.model_picker.addItem("No models loaded")
        self.model_picker.setEnabled(False)
        model_row.addWidget(QLabel("Model:"))
        model_row.addWidget(self.model_picker, stretch=1)
        model_row.addSpacing(8)
        self.device_picker = QComboBox()
        self.device_picker.setEnabled(False)
        model_row.addWidget(QLabel("Device:"))
        model_row.addWidget(self.device_picker)
        layout.addLayout(model_row)

        segment_row = QHBoxLayout()
        self.segment_btn = QPushButton("Segment image")
        self.segment_btn.setToolTip("Segment the image or video frame on screen")
        configure_button(self.segment_btn)
        segment_row.addWidget(self.segment_btn)
        self.segment_video_btn = QPushButton("Segment video")
        self.segment_video_btn.setEnabled(False)
        configure_button(self.segment_video_btn)
        segment_row.addWidget(self.segment_video_btn)
        layout.addLayout(segment_row)

        self.segment_all_btn = QPushButton("Segment all files")
        configure_button(self.segment_all_btn)
        layout.addWidget(self.segment_all_btn)

        mask_row = QHBoxLayout()
        self.delete_mask_btn = QPushButton("Delete mask")
        configure_button(self.delete_mask_btn)
        mask_row.addWidget(self.delete_mask_btn)
        self.save_btn = QPushButton("Save masks")
        configure_button(self.save_btn)
        mask_row.addWidget(self.save_btn)
        layout.addLayout(mask_row)

        layout.addSpacing(16)
        sep4 = QFrame()
        sep4.setFrameShape(QFrame.Shape.HLine)
        sep4.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep4)

        layout.addWidget(QLabel("Edit"))
        tools_row = QHBoxLayout()
        tools_row.addWidget(QLabel("Tools:"))
        self.tool_picker = QComboBox()
        self.tool_picker.addItems(
            [
                "Select",
                "Freehand Line",
                "Segmented Line",
                "Paint Brush",
                "Eraser",
                "ROI Box",
            ]
        )
        self.tool_picker.setToolTip(
            "ROI Box: drag a box on the image to segment only inside it. Drag its "
            "edges or corners to resize it, or inside it to move it. Right-click, "
            "or click and press Delete, to remove the ROI or the mask."
        )
        self.tool_picker.setCurrentIndex(0)
        tools_row.addWidget(self.tool_picker)
        self.fill_mask_checkbox = QCheckBox("Fill mask")
        tools_row.addWidget(self.fill_mask_checkbox)
        self.fill_mask_checkbox.setChecked(False)
        layout.addLayout(tools_row)

        layout.addSpacing(5)

        opacity_row = QHBoxLayout()
        opacity_label = QLabel("Mask Opacity:")
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(50)
        opacity_row.addWidget(opacity_label)
        opacity_row.addWidget(self.opacity_slider)
        layout.addLayout(opacity_row)

        brush_row = QHBoxLayout()
        brush_label = QLabel("Tool Radius:")
        self.brush_radius = QSlider(Qt.Orientation.Horizontal)
        self.brush_radius.setMinimum(1)
        self.brush_radius.setMaximum(50)
        self.brush_radius.setValue(4)
        self.brush_radius_label = QLabel("4 px")
        brush_row.addWidget(brush_label)
        brush_row.addWidget(self.brush_radius)
        brush_row.addWidget(self.brush_radius_label)
        layout.addLayout(brush_row)

        layout.addSpacing(10)

        view_buttons_row = QHBoxLayout()
        self.fit_btn = QPushButton("Fit to Window")
        configure_button(self.fit_btn)
        view_buttons_row.addWidget(self.fit_btn)
        self.rotate_btn = QPushButton("Rotate image")
        self.rotate_btn.setToolTip(
            "Turn the image on screen. The image file is not changed, and the "
            "rotation is remembered for this file."
        )
        configure_button(self.rotate_btn)
        view_buttons_row.addWidget(self.rotate_btn)
        layout.addLayout(view_buttons_row)

        layout.addSpacing(5)

        undo_redo_row = QHBoxLayout()
        self.undo_btn = QPushButton("Undo")
        configure_button(self.undo_btn)
        undo_redo_row.addWidget(self.undo_btn)
        self.redo_btn = QPushButton("Redo")
        configure_button(self.redo_btn)
        undo_redo_row.addWidget(self.redo_btn)
        layout.addLayout(undo_redo_row)

        layout.addSpacing(6)
        sep_end = QFrame()
        sep_end.setFrameShape(QFrame.Shape.HLine)
        sep_end.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep_end)

        layout.addWidget(QLabel("Measurements"))
        self.px_per_mm_x_spin = QDoubleSpinBox()
        self.px_per_mm_y_spin = QDoubleSpinBox()
        for spin, direction in (
            (self.px_per_mm_x_spin, "across"),
            (self.px_per_mm_y_spin, "down"),
        ):
            spin.setRange(0.0, 999.99)
            spin.setDecimals(2)
            spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            spin.setSpecialValueText("Not set")
            spin.setToolTip(
                f"How many image pixels make up one millimetre going {direction} "
                "the image. While either value is not set, thickness is shown "
                "in pixels."
            )
        px_per_mm_row = QHBoxLayout()
        px_per_mm_row.addWidget(QLabel("Pixels per mm:"))
        px_per_mm_row.addWidget(QLabel("x"))
        px_per_mm_row.addWidget(self.px_per_mm_x_spin)
        px_per_mm_row.addSpacing(8)
        px_per_mm_row.addWidget(QLabel("y"))
        px_per_mm_row.addWidget(self.px_per_mm_y_spin)
        px_per_mm_row.addStretch(1)
        layout.addLayout(px_per_mm_row)

        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.setToolTip(
            "Work out the pixels per mm by placing lines a known distance apart"
        )
        configure_button(self.calibrate_btn)
        layout.addWidget(self.calibrate_btn)

        view_row = QHBoxLayout()
        self.view_picker = QComboBox()
        self.view_picker.addItem("Full cartilage", False)
        self.view_picker.addItem("Per region", True)
        view_row.addWidget(QLabel("View:"))
        view_row.addWidget(self.view_picker, stretch=1)
        view_row.addSpacing(8)
        self.knee_picker = QComboBox()
        self.knee_picker.addItem("Right", "right")
        self.knee_picker.addItem("Left", "left")
        self.knee_picker.setToolTip(
            "Which knee the image shows. It decides which side is lateral and "
            "which is medial."
        )
        view_row.addWidget(QLabel("Knee:"))
        view_row.addWidget(self.knee_picker)
        layout.addLayout(view_row)
        self.reset_centre_btn = QPushButton("Reset centre point")
        self.reset_centre_btn.setToolTip(
            "Move the diamond back to the position the app suggests"
        )
        configure_button(self.reset_centre_btn)
        layout.addWidget(self.reset_centre_btn)

        # One column of values per region; the view decides which columns show.
        results_grid = QGridLayout()
        results_grid.setHorizontalSpacing(8)
        self._result_columns = ("whole",) + REGION_NAMES
        column_titles = {
            "whole": "Whole",
            "lateral": "Lateral",
            "intercondylar": "Notch",
            "medial": "Medial",
        }
        self._result_headers = {}
        self._result_row_labels = {}
        self._result_values = {}
        for column, region in enumerate(self._result_columns, start=1):
            header = QLabel(column_titles[region])
            header.setAlignment(Qt.AlignmentFlag.AlignRight)
            if region in REGION_COLORS:
                red, green, blue = REGION_COLORS[region]
                header.setStyleSheet(
                    f"color: rgb({red}, {green}, {blue}); font-weight: bold;"
                )
            results_grid.addWidget(header, 0, column)
            self._result_headers[region] = header
        for row, measure_name in enumerate(
            ("area", "length", "thickness", "echo_mean", "echo_sd"), start=1
        ):
            row_label = QLabel()
            results_grid.addWidget(row_label, row, 0)
            self._result_row_labels[measure_name] = row_label
            for column, region in enumerate(self._result_columns, start=1):
                value = QLabel("–")
                value.setAlignment(Qt.AlignmentFlag.AlignRight)
                results_grid.addWidget(value, row, column)
                self._result_values[(region, measure_name)] = value
        results_grid.setColumnStretch(0, 1)
        layout.addLayout(results_grid)

        layout.addSpacing(6)
        sep_measurements = QFrame()
        sep_measurements.setFrameShape(QFrame.Shape.HLine)
        sep_measurements.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep_measurements)

        layout.addStretch()

        return panel

    def _wire_actions(self):
        self.load_btn.clicked.connect(self._load_files)
        self.load_files_action.triggered.connect(self._load_files)
        self.output_btn.clicked.connect(self._change_output_dir)
        self.exit_action.triggered.connect(self.close)
        self.segment_btn.clicked.connect(self._segment_current)
        self.segment_action.triggered.connect(self._segment_current)
        self.segment_video_btn.clicked.connect(self._run_current_video_inference)
        self.segment_video_action.triggered.connect(self._run_current_video_inference)
        self.segment_all_btn.clicked.connect(self._segment_all)
        self.segment_all_action.triggered.connect(self._segment_all)
        self.save_btn.clicked.connect(self._save_current_mask)
        self.save_mask_action.triggered.connect(self._save_current_mask)
        self.delete_mask_btn.clicked.connect(self._delete_current_mask)
        self.delete_mask_action.triggered.connect(self._delete_current_mask)
        self.clear_rois_action.triggered.connect(self._clear_all_rois)
        self.close_btn.clicked.connect(self._close_current_file)
        self.close_file_action.triggered.connect(self._close_current_file)
        self.tool_picker.currentIndexChanged.connect(self._on_tool_changed)
        self.brush_radius.valueChanged.connect(self._on_brush_radius_changed)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self.fit_btn.clicked.connect(self.canvas.fit_to_window)
        self.rotate_btn.clicked.connect(self._rotate_dialog)
        self.rotate_action.triggered.connect(self._rotate_dialog)
        self.fill_mask_checkbox.toggled.connect(self.canvas.set_fill_mask)
        self.undo_btn.clicked.connect(self._undo)
        self.redo_btn.clicked.connect(self._redo)
        self.undo_action.triggered.connect(self._undo)
        self.redo_action.triggered.connect(self._redo)
        self.model_picker.currentIndexChanged.connect(self._on_model_changed)
        self.device_picker.currentIndexChanged.connect(self._on_device_changed)
        self.clear_files_btn.clicked.connect(self._clear_sequence)
        self.file_combo.currentIndexChanged.connect(self._on_file_selected)
        self.prev_btn.clicked.connect(self._show_previous_sequence)
        self.next_btn.clicked.connect(self._show_next_sequence)
        self.play_btn.clicked.connect(self._toggle_playback)
        self.frame_first_btn.clicked.connect(self._show_first_frame)
        self.frame_prev_btn.clicked.connect(self._show_previous_frame)
        self.frame_next_btn.clicked.connect(self._show_next_frame)
        self.frame_last_btn.clicked.connect(self._show_last_frame)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
                if event.modifiers() != Qt.KeyboardModifier.NoModifier:
                    return False
                if not self.isActiveWindow():
                    return False
                if self._mode == "sequence":
                    if event.key() == Qt.Key.Key_Left:
                        self._show_previous_sequence()
                    else:
                        self._show_next_sequence()
                    return True
                if self._mode != "video":
                    return False
                if event.isAutoRepeat():
                    return True
                direction = -1 if event.key() == Qt.Key.Key_Left else 1
                self._start_arrow_repeat(direction)
                return True
        if event.type() == QEvent.Type.KeyRelease:
            if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
                if event.isAutoRepeat():
                    return True
                direction = -1 if event.key() == Qt.Key.Key_Left else 1
                if direction == self._arrow_repeat_direction:
                    self._stop_arrow_repeat()
                    return True
        return super().eventFilter(obj, event)

    def _start_arrow_repeat(self, direction):
        self._stop_arrow_repeat()
        self._arrow_repeat_direction = int(direction)
        self._arrow_repeat_started = False
        if self._arrow_repeat_direction < 0:
            self._show_previous_frame()
        else:
            self._show_next_frame()
        self._arrow_repeat_timer.start(250)

    def _stop_arrow_repeat(self):
        if self._arrow_repeat_timer.isActive():
            self._arrow_repeat_timer.stop()
        self._arrow_repeat_timer.setInterval(250)
        self._arrow_repeat_direction = 0
        self._arrow_repeat_started = False

    def _on_arrow_repeat_timeout(self):
        if self._arrow_repeat_direction == 0:
            self._stop_arrow_repeat()
            return
        if not self._arrow_repeat_started:
            self._arrow_repeat_started = True
            self._arrow_repeat_timer.setInterval(40)
        if self._arrow_repeat_direction < 0:
            self._show_previous_frame()
        else:
            self._show_next_frame()

    def _center_window(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        frame = self.frameGeometry()
        frame.moveCenter(screen.availableGeometry().center())
        self.move(frame.topLeft())

    def _update_title_with_image(self, image_path):
        if self._mode == "video" and self._video_path and self._video_frame_count > 0:
            base = Path(self._video_path).name
            index = self._video_frame_index + 1
            if self._video_paths and self._video_list_index >= 0:
                v_num = self._video_list_index + 1
                v_total = len(self._video_paths)
                self.setWindowTitle(
                    f"{self._base_title} - {base} ({v_num}/{v_total}) "
                    f"[frame {index}/{self._video_frame_count}]"
                )
            else:
                self.setWindowTitle(f"{self._base_title} - {base} [frame {index}/{self._video_frame_count}]")
            return
        name = Path(image_path).name if image_path else ""
        if name:
            self.setWindowTitle(f"{self._base_title} - {name}")
        else:
            self.setWindowTitle(self._base_title)

    def _saved_mask_paths_for_current(self):
        if self._mode == "sequence" and 0 <= self._sequence_index < len(self._sequence_paths):
            if not self._sequence_output_dir:
                return []
            image_path = Path(self._sequence_paths[self._sequence_index])
            return find_mask_files(
                self._shown_result_dir(), image_path, self._sequence_paths
            )
        if self._mode == "video" and self._video_path and self._video_frame_index >= 0:
            mask_path = self._shown_mask_path()
            if mask_path is not None:
                return [mask_path]
        return []

    def _delete_current_mask(self):
        self._commit_pending_outline()
        mask_paths = self._saved_mask_paths_for_current()
        result_dir = self._shown_result_dir()
        if not mask_paths and not self._canvas_has_mask():
            return
        if self._mode == "video":
            target = f"frame {self._video_frame_index + 1} of {Path(self._video_path).name}"
        elif self._mode == "sequence":
            target = Path(self._sequence_paths[self._sequence_index]).name
        else:
            target = "this image"
        choice = QMessageBox.question(
            self,
            "Delete mask",
            f"Delete the mask for {target}?\n\nThe mask will be permanently deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return
        self.canvas.clear_mask()
        try:
            for mask_path in mask_paths:
                mask_path.unlink()
        except OSError as exc:
            QMessageBox.warning(self, "Delete error", str(exc))
            return
        if self._mode == "video":
            self._forget_measurement(
                result_dir,
                Path(self._video_path).name,
                self._video_frame_index,
            )
        elif self._mode == "sequence" and self._sequence_output_dir:
            self._forget_measurement(
                result_dir,
                Path(self._sequence_paths[self._sequence_index]).name,
            )
        if mask_paths:
            self.statusBar().showMessage("Mask deleted from the output folder")

    def _close_current_file(self):
        if self._mode == "video":
            paths, index = self._video_paths, self._video_list_index
        elif self._mode == "sequence":
            paths, index = self._sequence_paths, self._sequence_index
        else:
            return
        if index < 0 or index >= len(paths):
            return
        closed_name = Path(paths[index]).name
        choice = QMessageBox.question(
            self,
            "Close file",
            f"Close {closed_name}?\n\n"
            "The file will be closed but not deleted, and the masks will be saved.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return
        self._stop_playback()
        self._stash_video_mask_for_current_frame()
        self._save_sequence_mask_if_needed()
        if len(paths) == 1:
            self._clear_sequence()
            self.statusBar().showMessage(f"Closed {closed_name}")
            return
        del paths[index]
        self.file_combo.blockSignals(True)
        self.file_combo.removeItem(index)
        self.file_combo.blockSignals(False)
        next_index = min(index, len(paths) - 1)
        if self._mode == "video":
            self._open_video_at_index(next_index, start_frame=0)
        else:
            self._set_slider_state(0, len(paths) - 1, enabled=True)
            self._set_sequence_index(next_index)
        self.statusBar().showMessage(f"Closed {closed_name}")

    def _screen_bounds(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            return None
        return screen.availableGeometry()

    def _initial_window_size(self):
        screen_rect = self._screen_bounds()
        if screen_rect is None:
            self._sidebar_width = 200
            return (600, 900)
        margin = 40
        max_w = max(700, screen_rect.width() - margin)
        max_h = max(300, screen_rect.height() - margin)
        target_w = int(screen_rect.width() * 0.35)
        target_h = int(screen_rect.height() * 0.45)
        width = min(target_w, max_w)
        height = min(target_h, max_h)
        min_w = min(1000, max_w)
        min_h = min(650, max_h)
        width = max(min_w, width)
        height = max(min_h, height)
        self._sidebar_width = min(340, max(180, int(screen_rect.width() * 0.22)))
        return (width, height)

    def _on_tool_changed(self, index):
        tool_map = {
            0: "select",
            1: "freehand",
            2: "polyline",
            3: "brush",
            4: "eraser",
            5: "roi",
        }
        tool = tool_map.get(index, "select")
        self.canvas.set_tool(tool)
        if tool in ("brush", "eraser"):
            self.fill_mask_checkbox.setChecked(True)

    def _on_opacity_changed(self, value):
        self.canvas.set_mask_opacity(value / 100.0)

    def _on_brush_radius_changed(self, value):
        self.canvas.set_brush_radius(value)
        if hasattr(self, "brush_radius_label"):
            self.brush_radius_label.setText(f"{int(value)} px")

    def _run_current_image_inference(self):
        if self._mode != "sequence":
            QMessageBox.information(self, "No files", "Load files first.")
            return
        if 0 <= self._sequence_index < len(self._sequence_paths):
            existing_path = self._find_sequence_mask_path(
                self._sequence_paths[self._sequence_index]
            )
            if existing_path and not self._confirm_overwrite_existing(1, "image segmentation"):
                return
        self._run_inference()

    def _run_current_video_frame_inference(self):
        if self._mode != "video" or self._video_frame_index < 0:
            QMessageBox.information(self, "No video", "Load a video first.")
            return
        selected_index = self.file_combo.currentIndex()
        if selected_index < 0 or selected_index >= len(self._video_paths):
            QMessageBox.information(self, "No video", "Select a video first.")
            return
        selected_video_path = self._video_paths[selected_index]
        existing_path = self._video_mask_path(
            selected_video_path,
            self._video_frame_index,
        )
        has_existing_segmentation = bool(
            (existing_path and existing_path.exists()) or self._canvas_has_mask()
        )
        if has_existing_segmentation:
            scope = (
                f"{Path(selected_video_path).name}, "
                f"frame {self._video_frame_index + 1}"
            )
            if not self._confirm_overwrite_existing(
                1,
                "frame segmentation",
                scope,
            ):
                return
        self._run_inference()

    def _confirm_overwrite_existing(self, count, description, scope=""):
        scope_text = f" for {scope}" if scope else ""
        choice = QMessageBox.question(
            self,
            "Existing segmentations",
            f"Found {count} existing {description}(s){scope_text}.\n\n"
            "Overwrite them?\n\n"
            "Yes: overwrite existing masks.\n"
            "No: keep existing masks and segment only missing items.",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.No,
        )
        if choice == QMessageBox.StandardButton.Cancel:
            return None
        return choice == QMessageBox.StandardButton.Yes

    def _run_batch_inference(self):
        if not self._sequence_paths:
            QMessageBox.information(self, "No sequence", "Load an image sequence first.")
            return
        if not self._sequence_output_dir:
            QMessageBox.information(self, "No output folder", "Select an output folder first.")
            return
        if not self._model.has_model():
            QMessageBox.warning(
                self,
                "No model",
                "Select a model in the Model list first.",
            )
            return
        if self._inference_thread and self._inference_thread.isRunning():
            self.statusBar().showMessage("Segmentation already running...")
            return
        if self._batch_thread and self._batch_thread.isRunning():
            self.statusBar().showMessage("Batch segmentation already running...")
            return

        overwrite_existing = True
        existing_count = sum(
            1 for image_path in self._sequence_paths if self._find_sequence_mask_path(image_path)
        )
        if existing_count:
            overwrite_existing = self._confirm_overwrite_existing(
                existing_count,
                "image segmentation",
            )
            if overwrite_existing is None:
                return

        total = len(self._sequence_paths)
        self.statusBar().showMessage("Running batch segmentation...")
        self._show_batch_dialog(total)

        self._batch_thread = QThread(self)
        self._measurement_logs.clear()
        self._batch_worker = BatchInferenceWorker(
            self._model,
            list(self._sequence_paths),
            self._sequence_output_dir,
            overwrite_existing=overwrite_existing,
            px_per_mm=(
                self.px_per_mm_x_spin.value(),
                self.px_per_mm_y_spin.value(),
            ),
            knee_side=self.knee_picker.currentData(),
            rois=self._rois_for_segmenting(self._sequence_output_dir),
            rotations=self._rotation_store(self._sequence_output_dir).angles(),
            measurements=self._measurement_file(self._sequence_output_dir),
        )
        self._batch_worker.moveToThread(self._batch_thread)
        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.image_started.connect(self._on_batch_image_started)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.canceled.connect(self._on_batch_canceled)
        self._batch_worker.error.connect(self._on_batch_error)
        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_worker.canceled.connect(self._batch_thread.quit)
        self._batch_worker.error.connect(self._batch_thread.quit)
        self._batch_worker.finished.connect(self._batch_worker.deleteLater)
        self._batch_worker.canceled.connect(self._batch_worker.deleteLater)
        self._batch_worker.error.connect(self._batch_worker.deleteLater)
        self._batch_thread.finished.connect(self._on_batch_thread_done)
        self._batch_thread.finished.connect(self._batch_thread.deleteLater)
        self._batch_thread.start()

    def _run_current_video_inference(self):
        if self._mode != "video":
            QMessageBox.information(self, "No video", "Load a video first.")
            return
        selected_index = self.file_combo.currentIndex()
        if selected_index < 0 or selected_index >= len(self._video_paths):
            QMessageBox.information(self, "No video", "Select a video first.")
            return
        selected_video_path = self._video_paths[selected_index]
        self._start_video_batch_inference(
            [selected_video_path],
            "Segment video",
            overwrite_scope=Path(selected_video_path).name,
        )

    def _run_all_videos_inference(self):
        if self._mode != "video" or not self._video_paths:
            QMessageBox.information(self, "No videos", "Load one or more videos first.")
            return
        self._start_video_batch_inference(list(self._video_paths), "Segment all videos")

    def _start_video_batch_inference(self, video_paths, title, overwrite_scope=""):
        if not self._video_output_dir:
            QMessageBox.information(self, "No output folder", "Select an output folder first.")
            return
        if not self._model.has_model():
            QMessageBox.warning(self, "No model", "Select a model in the Model list first.")
            return
        if self._inference_thread and self._inference_thread.isRunning():
            self.statusBar().showMessage("Segmentation already running...")
            return
        if self._batch_thread and self._batch_thread.isRunning():
            self.statusBar().showMessage("Batch segmentation already running...")
            return

        overwrite_existing = True
        existing_count = sum(
            1
            for video_path in video_paths
            for _ in (Path(self._video_output_dir) / Path(video_path).stem).glob("frame_*.png")
        )
        if existing_count:
            overwrite_existing = self._confirm_overwrite_existing(
                existing_count,
                "frame segmentation",
                overwrite_scope,
            )
            if overwrite_existing is None:
                return

        total_frames = 0
        for video_path in video_paths:
            capture = cv2.VideoCapture(video_path)
            if capture.isOpened():
                total_frames += max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            capture.release()
        if total_frames <= 0:
            QMessageBox.warning(self, "Video error", "The selected videos contain no readable frames.")
            return

        self._stash_video_mask_for_current_frame()
        self.statusBar().showMessage(f"{title} running...")
        self._show_video_batch_dialog(len(video_paths), title)

        self._batch_thread = QThread(self)
        self._measurement_logs.clear()
        self._batch_worker = VideoBatchInferenceWorker(
            self._model,
            video_paths,
            self._video_output_dir,
            overwrite_existing=overwrite_existing,
            px_per_mm=(
                self.px_per_mm_x_spin.value(),
                self.px_per_mm_y_spin.value(),
            ),
            knee_side=self.knee_picker.currentData(),
            rois=self._rois_for_segmenting(self._video_output_dir),
            rotations=self._rotation_store(self._video_output_dir).angles(),
            measurements=self._measurement_file(self._video_output_dir),
        )
        self._batch_worker.moveToThread(self._batch_thread)
        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.video_started.connect(self._on_video_started)
        self._batch_worker.frame_started.connect(self._on_video_frame_started)
        self._batch_worker.frame_progress.connect(self._on_video_frame_progress)
        self._batch_worker.video_finished.connect(self._on_video_finished)
        self._batch_worker.finished.connect(self._on_video_batch_finished)
        self._batch_worker.canceled.connect(self._on_batch_canceled)
        self._batch_worker.error.connect(self._on_batch_error)
        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_worker.canceled.connect(self._batch_thread.quit)
        self._batch_worker.error.connect(self._batch_thread.quit)
        self._batch_worker.finished.connect(self._batch_worker.deleteLater)
        self._batch_worker.canceled.connect(self._batch_worker.deleteLater)
        self._batch_worker.error.connect(self._batch_worker.deleteLater)
        self._batch_thread.finished.connect(self._on_batch_thread_done)
        self._batch_thread.finished.connect(self._batch_thread.deleteLater)
        self._batch_thread.start()

    def _show_video_batch_dialog(self, video_count, title):
        if self._batch_dialog is not None:
            self._batch_dialog.close()
        self._batch_dialog = VideoBatchProgressDialog(video_count, title, self)
        self._batch_dialog.canceled.connect(self._request_batch_cancel)
        self._batch_dialog.show()

    def _show_batch_dialog(self, total, title="Batch segment"):
        if self._batch_dialog is not None:
            self._batch_dialog.close()
        self._batch_dialog = QProgressDialog(
            "Running batch segmentation...",
            "Cancel",
            0,
            total,
            self,
        )
        self._batch_dialog.setWindowTitle(title)
        self._batch_dialog.setMinimumDuration(0)
        self._batch_dialog.setAutoClose(False)
        self._batch_dialog.setAutoReset(False)
        self._batch_dialog.canceled.connect(self._request_batch_cancel)
        self._batch_dialog.show()

    def _close_batch_dialog(self):
        if self._batch_dialog is None:
            return
        self._batch_dialog.close()
        self._batch_dialog = None

    def _request_batch_cancel(self):
        if self._batch_worker:
            self._batch_worker.cancel()
        self.statusBar().showMessage("Canceling batch segmentation...")
        if self._batch_dialog:
            if isinstance(self._batch_dialog, VideoBatchProgressDialog):
                self._batch_dialog.mark_canceling()
            else:
                self._batch_dialog.setLabelText("Canceling batch segmentation...")

    def _on_batch_progress(self, current, total):
        if self._batch_dialog:
            self._batch_dialog.setValue(current)
            self._batch_dialog.setLabelText(f"Segmenting image {current}/{total}")

    def _on_batch_image_started(self, image_path, index, total):
        if self._batch_dialog:
            name = Path(image_path).name
            self._batch_dialog.setLabelText(f"Segmenting {name} ({index}/{total})")

    def _on_batch_finished(self, processed, skipped):
        message = f"Batch segmentation complete: {processed} image(s) segmented"
        if skipped:
            message += f", {skipped} existing image(s) kept"
        self.statusBar().showMessage(f"{message}.")
        self._close_batch_dialog()
        if self._sequence_paths and self._sequence_index >= 0:
            self._load_sequence_image()

    def _on_video_started(self, video_name, video_number, video_count, frame_count):
        if isinstance(self._batch_dialog, VideoBatchProgressDialog):
            self._batch_dialog.start_video(
                video_name,
                video_number,
                video_count,
                frame_count,
            )

    def _on_video_frame_started(self, video_name, frame_number, frame_count, current, total):
        if isinstance(self._batch_dialog, VideoBatchProgressDialog):
            self._batch_dialog.start_frame(frame_number, frame_count)

    def _on_video_frame_progress(self, frame_number, frame_count):
        if isinstance(self._batch_dialog, VideoBatchProgressDialog):
            self._batch_dialog.set_frame_progress(frame_number, frame_count)

    def _on_video_finished(self, video_number, video_count):
        if isinstance(self._batch_dialog, VideoBatchProgressDialog):
            self._batch_dialog.set_video_progress(video_number, video_count)

    def _on_video_batch_finished(self, processed, skipped):
        message = f"Video segmentation complete: {processed} frame(s) segmented"
        if skipped:
            message += f", {skipped} existing frame(s) kept"
        self.statusBar().showMessage(f"{message}.")
        if isinstance(self._batch_dialog, VideoBatchProgressDialog):
            self._batch_dialog.mark_complete()
        self._close_batch_dialog()
        if self._mode == "video" and self._video_path and self._video_frame_index >= 0:
            if self._shown_mask_path() is not None:
                self._show_saved_mask()

    def _on_batch_canceled(self):
        self.statusBar().showMessage("Batch segmentation canceled")
        if isinstance(self._batch_dialog, VideoBatchProgressDialog):
            self._batch_dialog.mark_canceling()
        self._close_batch_dialog()

    def _on_batch_error(self, message):
        QMessageBox.warning(self, "Batch error", message)
        self.statusBar().showMessage("Batch segmentation failed")
        if isinstance(self._batch_dialog, VideoBatchProgressDialog):
            self._batch_dialog.mark_stopped()
        self._close_batch_dialog()

    def _on_batch_thread_done(self):
        self._batch_thread = None
        self._batch_worker = None
        self._measurement_logs.clear()
        self.statusBar().showMessage("Ready")

    def _load_files(self):
        dialog_result = self._show_load_dialog()
        if dialog_result is None:
            return
        kind, paths, output_dir = dialog_result
        if kind == "video":
            self._load_video(paths, output_dir)
        else:
            self._load_sequence(paths, output_dir)

    def _load_sequence(self, paths, output_dir):
        self._stop_playback()
        if self._mode == "video":
            self._stash_video_mask_for_current_frame()
            self._clear_video_state()
        self._mode = "sequence"
        self._sequence_paths = list(paths)
        self._sequence_index = 0
        self._sequence_output_dir = output_dir
        self._update_output_dir_tooltip()
        self.file_combo.setEnabled(True)
        self.file_combo.clear()
        self.file_combo.addItems([Path(p).name for p in self._sequence_paths])
        self._set_slider_state(0, len(self._sequence_paths) - 1, enabled=bool(self._sequence_paths))
        self._set_sequence_index(0)
        self.statusBar().showMessage(f"Sequence loaded: {len(self._sequence_paths)} images")

    def _load_video(self, video_paths, output_dir):
        self._stop_playback()
        if self._mode == "sequence":
            self._save_sequence_mask_if_needed()
            self._clear_sequence_state(clear_canvas=False)
        self._stash_video_mask_for_current_frame()
        self._clear_video_state()
        self._video_paths = sorted(str(Path(p)) for p in video_paths)
        self._video_list_index = 0
        self._video_output_dir = output_dir
        self._last_video_output_dir = output_dir
        self._save_persisted_paths()
        self._update_output_dir_tooltip()
        self._mode = "video"
        self.file_combo.setEnabled(True)
        self.file_combo.clear()
        self.file_combo.addItems([Path(p).name for p in self._video_paths])
        self._open_video_at_index(0, start_frame=0)

    def _change_output_dir(self):
        if self._mode == "video":
            self._change_video_output_dir()
        elif self._mode == "sequence":
            self._change_image_output_dir()
        else:
            QMessageBox.information(self, "No files", "Load files first.")

    def _segment_current(self):
        if self._mode == "video":
            self._run_current_video_frame_inference()
        else:
            self._run_current_image_inference()

    def _segment_all(self):
        if self._mode == "video":
            self._run_all_videos_inference()
        elif self._mode == "sequence":
            self._run_batch_inference()
        else:
            QMessageBox.information(self, "No files", "Load files first.")

    def _on_file_selected(self, index):
        if self._mode == "video":
            self._on_video_selected(index)
        else:
            self._on_sequence_selected(index)

    def _change_image_output_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            self._sequence_base_dir or self._last_image_output_dir or self._last_image_input_dir,
        )
        if not directory or directory == self._sequence_base_dir:
            return
        # The mask on screen belongs to the previous folder, so it is written there
        # and the canvas then shows whatever the chosen folder holds for this image.
        self._save_sequence_mask_if_needed()
        self._sequence_output_dir = directory
        self._last_image_output_dir = directory
        self._save_persisted_paths()
        self._update_output_dir_tooltip()
        self._show_saved_result()
        self.statusBar().showMessage(f"Output folder: {directory}")

    def _change_video_output_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            self._video_base_dir or self._last_video_output_dir or self._last_video_input_dir,
        )
        if not directory or directory == self._video_base_dir:
            return
        self._stop_playback()
        # Stashing writes the on-screen mask to the folder in effect, so it has to
        # happen before the switch and the canvas reloaded from the chosen folder after.
        self._stash_video_mask_for_current_frame()
        self._video_output_dir = directory
        self._last_video_output_dir = directory
        self._save_persisted_paths()
        self._update_output_dir_tooltip()
        self._show_saved_result()
        self.statusBar().showMessage(f"Output folder: {directory}")

    def _show_saved_result(self):
        """Put the saved mask, ROI, knee side and centre point of the image or
        frame on screen back on the canvas, as held by the results folder in use."""
        if self.canvas.image is None or self._current_frame_key() is None:
            return
        if self._current_angle() != self._shown_angle:
            # The folder switched to keeps this file at another rotation.
            self._show_stored_image()
        self._show_saved_mask()
        self._restore_folder_inputs()

    def _show_stored_image(self):
        """Put the image or frame on screen back on the canvas from its file."""
        if self._mode == "video":
            frame = self._decode_video_frame(self._video_frame_index)
            if frame is not None:
                self._show_image(frame, self._video_path or "")
        elif self._mode == "sequence":
            self._show_image_file(self._sequence_paths[self._sequence_index])

    def _update_output_dir_tooltip(self):
        self.output_btn.setToolTip(
            self._sequence_base_dir or self._video_base_dir or "No output folder selected"
        )

    def _clear_sequence_state(self, clear_canvas):
        self._sequence_paths = []
        self._sequence_index = -1
        self._sequence_output_dir = None
        self._update_output_dir_tooltip()
        self.file_combo.clear()
        self.file_combo.setEnabled(False)
        if clear_canvas:
            self.canvas.clear_image()

    def _clear_video_state(self):
        self._stop_playback()
        if self._video_capture is not None:
            self._video_capture.release()
        self.file_combo.clear()
        self.file_combo.setEnabled(False)
        self._video_paths = []
        self._video_list_index = -1
        self._video_path = None
        self._video_capture = None
        self._video_frame_count = 0
        self._video_fps = 0.0
        self._video_frame_index = -1
        self._video_output_dir = None
        self._update_output_dir_tooltip()
        self._video_frame_cache.clear()
        self._video_decode_pos = -1
        self._video_use_random_seek = False

    def _video_output_root_for_path(self, video_path):
        output_dir = (self._video_output_dir or "").strip()
        if not output_dir:
            return None
        return Path(output_dir) / Path(video_path).stem

    def _video_mask_path(self, video_path, frame_index):
        root = self._video_output_root_for_path(video_path)
        if root is None:
            return None
        return root / f"frame_{int(frame_index):06d}.png"

    def _load_saved_video_mask_for_frame(self, video_path, frame_index):
        mask_path = self._video_mask_path(video_path, frame_index)
        if mask_path is None or not mask_path.exists():
            return None
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            return None
        return (mask_gray >= 128).astype(np.float32)

    def _open_video_at_index(self, video_index, start_frame=0):
        if video_index < 0 or video_index >= len(self._video_paths):
            return
        self._stop_playback()
        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None
        self._video_list_index = video_index
        self._video_path = self._video_paths[video_index]
        capture = cv2.VideoCapture(self._video_path)
        if not capture.isOpened():
            QMessageBox.warning(self, "Load failed", f"Could not open video: {Path(self._video_path).name}")
            capture.release()
            return
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            QMessageBox.warning(self, "Load failed", f"Video has no readable frames: {Path(self._video_path).name}")
            capture.release()
            return
        self._video_capture = capture
        self._video_frame_count = frame_count
        self._video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._video_use_random_seek = self._supports_random_seek(self._video_path)
        if self._video_fps > 0:
            self._playback_interval_ms = max(15, int(round(1000.0 / self._video_fps)))
        else:
            self._playback_interval_ms = 33
        self._play_timer.setInterval(self._playback_interval_ms)
        self._video_frame_index = -1
        self._video_decode_pos = -1
        self._video_frame_cache.clear()
        self.file_combo.blockSignals(True)
        self.file_combo.setCurrentIndex(video_index)
        self.file_combo.blockSignals(False)
        self._set_slider_state(0, frame_count - 1, enabled=frame_count > 0)
        target_frame = int(start_frame)
        if target_frame < 0:
            target_frame = frame_count - 1
        target_frame = min(frame_count - 1, max(0, target_frame))
        self._set_video_frame_index(target_frame, force=True)
        self.statusBar().showMessage(
            f"Video loaded: {Path(self._video_path).name} "
            f"({video_index + 1}/{len(self._video_paths)})"
        )

    def _clear_sequence(self):
        self._stop_playback()
        self._stash_video_mask_for_current_frame()
        self._clear_video_state()
        self._clear_sequence_state(clear_canvas=False)
        self._mode = "none"
        self._set_slider_state(0, 0, enabled=False)
        self._update_navigation_buttons()
        self.canvas.clear_image()
        self.statusBar().showMessage("Files cleared")

    def _on_sequence_selected(self, index):
        if self._mode != "sequence":
            return
        if index < 0 or index >= len(self._sequence_paths):
            return
        if index == self._sequence_index:
            return
        self._save_sequence_mask_if_needed()
        self._set_sequence_index(index)

    def _on_video_selected(self, index):
        if self._mode != "video":
            return
        if index < 0 or index >= len(self._video_paths):
            return
        if index == self._video_list_index:
            return
        self._stash_video_mask_for_current_frame()
        self._open_video_at_index(index, start_frame=0)

    def _show_previous_sequence(self):
        if self._mode == "video":
            if self._video_list_index <= 0:
                return
            self._stash_video_mask_for_current_frame()
            self._open_video_at_index(self._video_list_index - 1, start_frame=0)
            return
        if self._mode != "sequence":
            return
        if self._sequence_index <= 0:
            return
        self._save_sequence_mask_if_needed()
        self._set_sequence_index(self._sequence_index - 1)

    def _show_next_sequence(self):
        if self._mode == "video":
            if self._video_list_index >= len(self._video_paths) - 1:
                return
            self._stash_video_mask_for_current_frame()
            self._open_video_at_index(self._video_list_index + 1, start_frame=0)
            return
        if self._mode != "sequence":
            return
        if self._sequence_index < 0:
            return
        if self._sequence_index >= len(self._sequence_paths) - 1:
            return
        self._save_sequence_mask_if_needed()
        self._set_sequence_index(self._sequence_index + 1)

    def _show_previous_frame(self):
        if self._mode == "sequence":
            self._show_previous_sequence()
            return
        if self._mode != "video":
            return
        if self._video_frame_index <= 0:
            return
        self._set_video_frame_index(self._video_frame_index - 1)

    def _show_next_frame(self):
        if self._mode == "sequence":
            self._show_next_sequence()
            return
        if self._mode != "video":
            return
        if self._video_frame_index < 0:
            return
        if self._video_frame_index >= self._video_frame_count - 1:
            return
        self._set_video_frame_index(self._video_frame_index + 1)

    def _on_canvas_navigation_requested(self, direction):
        if self._mode == "sequence":
            if direction < 0:
                self._show_previous_sequence()
            else:
                self._show_next_sequence()
            return
        if self._mode == "video":
            if direction < 0:
                self._show_previous_frame()
            else:
                self._show_next_frame()

    def _show_first_frame(self):
        if self._mode == "sequence":
            if self._sequence_index <= 0:
                return
            self._save_sequence_mask_if_needed()
            self._set_sequence_index(0)
            return
        if self._mode != "video":
            return
        if self._video_frame_index <= 0:
            return
        self._set_video_frame_index(0)

    def _show_last_frame(self):
        if self._mode == "sequence":
            if not self._sequence_paths:
                return
            last_index = len(self._sequence_paths) - 1
            if self._sequence_index >= last_index:
                return
            self._save_sequence_mask_if_needed()
            self._set_sequence_index(last_index)
            return
        if self._mode != "video":
            return
        if self._video_frame_count <= 0:
            return
        last_index = self._video_frame_count - 1
        if self._video_frame_index >= last_index:
            return
        self._set_video_frame_index(last_index)

    def _show_image_file(self, path):
        """Show an image file; returns False, with an empty canvas, if it cannot be shown."""
        try:
            self._show_image(self.canvas._read_image(path), path)
        except Exception as exc:
            self.canvas.clear_image()
            QMessageBox.critical(self, "Load failed", f"{Path(path).name}: {exc}")
            return False
        return True

    def _load_sequence_image(self):
        if self._mode != "sequence":
            return
        if self._sequence_index < 0 or self._sequence_index >= len(self._sequence_paths):
            return
        path = self._sequence_paths[self._sequence_index]
        if self._show_image_file(path):
            self._show_saved_mask()
            self._restore_frame_inputs()
        self._update_navigation_buttons()
        self._set_slider_value(self._sequence_index)

    def _set_sequence_index(self, index):
        if index < 0 or index >= len(self._sequence_paths):
            return
        self._sequence_index = index
        self.file_combo.blockSignals(True)
        self.file_combo.setCurrentIndex(index)
        self.file_combo.blockSignals(False)
        self._load_sequence_image()

    def _reopen_video_capture(self):
        if not self._video_path:
            return False
        if self._video_capture is not None:
            self._video_capture.release()
        capture = cv2.VideoCapture(self._video_path)
        if not capture.isOpened():
            self._video_capture = None
            return False
        self._video_capture = capture
        self._video_decode_pos = -1
        return True

    def _supports_random_seek(self, video_path):
        suffix = Path(video_path).suffix.lower()
        return suffix in {".mp4", ".m4v", ".mov"}

    def _decode_video_frame(self, frame_index):
        if frame_index in self._video_frame_cache:
            frame = self._video_frame_cache.pop(frame_index)
            self._video_frame_cache[frame_index] = frame
            return frame
        if self._video_capture is None:
            return None

        frame = None
        if self._video_use_random_seek:
            if self._video_decode_pos + 1 == frame_index:
                success, frame = self._video_capture.read()
                if success and frame is not None:
                    self._video_decode_pos = frame_index
                else:
                    frame = None
            if frame is None:
                self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = self._video_capture.read()
                if success and frame is not None:
                    self._video_decode_pos = frame_index
                else:
                    if not self._reopen_video_capture():
                        return None
                    while self._video_decode_pos < frame_index:
                        success, frame = self._video_capture.read()
                        if not success or frame is None:
                            return None
                        self._video_decode_pos += 1
        else:
            # Sequential decode only. Reopen to restart when requesting older frames.
            if frame_index <= self._video_decode_pos:
                if not self._reopen_video_capture():
                    return None
            while self._video_decode_pos < frame_index:
                success, frame = self._video_capture.read()
                if not success or frame is None:
                    return None
                self._video_decode_pos += 1

        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._video_frame_cache[frame_index] = frame
        while len(self._video_frame_cache) > self._video_cache_limit:
            self._video_frame_cache.popitem(last=False)
        return frame

    def _set_video_frame_index(self, frame_index, force=False):
        if self._mode != "video":
            return
        if frame_index < 0 or frame_index >= self._video_frame_count:
            return
        if not force and frame_index == self._video_frame_index:
            return
        self._stash_video_mask_for_current_frame()
        frame = self._decode_video_frame(frame_index)
        if frame is None:
            self._stop_playback()
            QMessageBox.warning(self, "Frame error", f"Could not decode frame {frame_index}.")
            return
        self._video_frame_index = frame_index
        self._show_image(frame, self._video_path or "")
        self._show_saved_mask()
        self._restore_frame_inputs()
        self._set_slider_value(frame_index)
        self._update_navigation_buttons()
        self._update_title_with_image(self._video_path or "")

    def _canvas_has_mask(self):
        mask = self.canvas.visible_mask()
        if mask is None:
            return False
        return bool(np.any(np.asarray(mask) >= 0.5))

    def _mask_hidden_by_roi(self):
        """True when the canvas holds mask that the ROI keeps out of view."""
        mask = self.canvas.mask
        if mask is None or self.canvas.roi_box() is None:
            return False
        return bool(np.any(np.asarray(mask) >= 0.5)) and not self._canvas_has_mask()

    def _commit_pending_outline(self):
        if self.canvas is None:
            return
        self.canvas.commit_pending_outline_to_mask()

    def _stash_video_mask_for_current_frame(self, force=False):
        """Save the mask and measurements of the frame on screen if they changed.

        A mask equal to its saved file is not written again, and its measurements
        row is left alone while it records the knee side, centre point and ROI on
        screen. force writes both regardless. After a change of pixels per mm,
        every saved frame of the video is measured again.
        """
        if self._mode != "video":
            return
        if self._video_frame_index < 0 or not self._video_path:
            return
        self._commit_pending_outline()
        mask_path = self._video_mask_path(self._video_path, self._video_frame_index)
        if mask_path is None:
            return
        name = Path(self._video_path).name
        calibration_changed = self._calibration_changed
        self._calibration_changed = False
        self._stash_video_frame(mask_path, name, force)
        if calibration_changed:
            self._remeasure_video_frames()

    def _stash_video_frame(self, mask_path, name, force):
        if self._canvas_has_mask():
            shown_dir = self._shown_result_dir()
            if not force and shown_dir != self._video_output_dir:
                # The mask shown comes from the selected output folder itself. It
                # is copied into the model's results folder only once it differs
                # from what that folder holds.
                if self._saved_mask_matches(self._shown_mask_path()) and (
                    self._measurement_log(shown_dir).get(name, self._video_frame_index)
                    is None
                    or self._measurement_is_current(
                        shown_dir, name, self._video_frame_index
                    )
                ):
                    return
            mask_same = not force and self._saved_mask_matches(mask_path)
            if not mask_same:
                try:
                    self._write_shown_mask(mask_path)
                except OSError as exc:
                    QMessageBox.warning(self, "Save error", str(exc))
                    return
            if not mask_same or not self._measurement_is_current(
                self._video_output_dir, name, self._video_frame_index
            ):
                self._record_measurement(
                    self._video_output_dir, name, self._video_frame_index
                )
            return
        if self._mask_hidden_by_roi():
            return
        if mask_path.exists():
            try:
                mask_path.unlink()
            except OSError as exc:
                QMessageBox.warning(self, "Save error", str(exc))
                return
            self._forget_measurement(
                self._video_output_dir,
                Path(self._video_path).name,
                self._video_frame_index,
            )

    def _update_navigation_buttons(self):
        is_video = self._mode == "video"
        self.segment_video_btn.setEnabled(is_video)
        self.segment_video_action.setEnabled(is_video)
        if is_video:
            can_prev = self._video_list_index > 0
            can_next = 0 <= self._video_list_index < len(self._video_paths) - 1
            can_prev_frame = self._video_frame_index > 0
            can_next_frame = (
                self._video_frame_index >= 0 and self._video_frame_index < self._video_frame_count - 1
            )
            self.prev_btn.setEnabled(can_prev)
            self.next_btn.setEnabled(can_next)
            self.play_btn.setEnabled(self._video_frame_count > 1)
            self.frame_first_btn.setEnabled(can_prev_frame)
            self.frame_prev_btn.setEnabled(can_prev_frame)
            self.frame_next_btn.setEnabled(can_next_frame)
            self.frame_last_btn.setEnabled(can_next_frame)
            return
        if self._mode == "sequence":
            can_prev = self._sequence_index > 0
            can_next = self._sequence_index < len(self._sequence_paths) - 1
            self.prev_btn.setEnabled(can_prev)
            self.next_btn.setEnabled(can_next)
            self.play_btn.setEnabled(False)
            self.frame_first_btn.setEnabled(can_prev)
            self.frame_prev_btn.setEnabled(can_prev)
            self.frame_next_btn.setEnabled(can_next)
            self.frame_last_btn.setEnabled(can_next)
            return
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.frame_first_btn.setEnabled(False)
        self.frame_prev_btn.setEnabled(False)
        self.frame_next_btn.setEnabled(False)
        self.frame_last_btn.setEnabled(False)

    def _set_slider_state(self, minimum, maximum, enabled):
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(int(minimum), int(maximum))
        self.frame_slider.setEnabled(bool(enabled))
        self.frame_slider.blockSignals(False)
        if not enabled:
            self.play_btn.setEnabled(False)
            self.frame_first_btn.setEnabled(False)
            self.frame_prev_btn.setEnabled(False)
            self.frame_next_btn.setEnabled(False)
            self.frame_last_btn.setEnabled(False)

    def _set_slider_value(self, value):
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(int(value))
        self.frame_slider.blockSignals(False)

    def _on_frame_slider_changed(self, value):
        if self._mode == "video":
            self._set_video_frame_index(int(value))
            return
        if self._mode != "sequence":
            return
        index = int(value)
        if index == self._sequence_index:
            return
        self._save_sequence_mask_if_needed()
        self._set_sequence_index(index)

    def _toggle_playback(self):
        if self._play_timer.isActive():
            self._stop_playback()
            return
        if self._mode != "video":
            return
        if self._video_frame_count <= 1:
            return
        self._start_playback()

    def _start_playback(self):
        if self._mode != "video":
            return
        if self._video_frame_index >= self._video_frame_count - 1:
            if self._video_list_index >= 0 and self._video_list_index < len(self._video_paths) - 1:
                self._stash_video_mask_for_current_frame()
                self._open_video_at_index(self._video_list_index + 1, start_frame=0)
            else:
                self._set_video_frame_index(0)
        self._play_btn_set_pause()
        self._play_timer.start(max(1, int(self._playback_interval_ms)))

    def _stop_playback(self):
        if self._play_timer.isActive():
            self._play_timer.stop()
        self._play_btn_set_play()

    def _advance_playback(self):
        if self._mode != "video":
            self._stop_playback()
            return
        if self._video_frame_index < self._video_frame_count - 1:
            self._set_video_frame_index(self._video_frame_index + 1)
            return
        if self._video_list_index >= 0 and self._video_list_index < len(self._video_paths) - 1:
            self._stash_video_mask_for_current_frame()
            self._open_video_at_index(self._video_list_index + 1, start_frame=0)
            return
        if self._video_frame_index >= self._video_frame_count - 1:
            self._stop_playback()

    def _play_btn_set_play(self):
        icon = self._transport_icon("play")
        if icon is not None:
            self.play_btn.setText("")
            self.play_btn.setIcon(icon)
        else:
            self.play_btn.setIcon(QIcon())
            self.play_btn.setText(">")

    def _play_btn_set_pause(self):
        icon = self._transport_icon("pause")
        if icon is not None:
            self.play_btn.setText("")
            self.play_btn.setIcon(icon)
        else:
            self.play_btn.setIcon(QIcon())
            self.play_btn.setText("||")

    def _save_sequence_mask_if_needed(self):
        """Save the mask and measurements of the image on screen if they changed.

        A mask equal to its saved file is not written again, and its measurements
        row is left alone while it records the knee side, centre point and ROI on
        screen and pixels per mm was not changed with this image on screen.
        """
        if self._mode != "sequence":
            return
        if not self._sequence_output_dir:
            return
        if self._sequence_index < 0 or self._sequence_index >= len(self._sequence_paths):
            return
        self._commit_pending_outline()
        image_path = Path(self._sequence_paths[self._sequence_index])
        calibration_changed = self._calibration_changed
        self._calibration_changed = False
        if not self._canvas_has_mask():
            self._remove_emptied_sequence_mask(image_path)
            return
        shown_dir = self._shown_result_dir()
        if shown_dir != self._sequence_output_dir and not calibration_changed:
            # The mask shown comes from the selected output folder itself. It is
            # copied into the model's results folder only once it differs from
            # what that folder holds.
            if self._saved_mask_matches(self._shown_mask_path()) and (
                self._measurement_log(shown_dir).get(image_path.name, "") is None
                or self._measurement_is_current(shown_dir, image_path.name)
            ):
                return
        saved_path = self._find_sequence_mask_path(image_path)
        # With the images' own folder as output folder, a PNG image is found as
        # its own mask.
        if saved_path and Path(saved_path).resolve() == image_path.resolve():
            saved_path = None
        mask_same = self._saved_mask_matches(saved_path)
        output_path = Path(self._sequence_output_dir) / f"{image_path.stem}.png"
        try:
            if not mask_same:
                self._write_shown_mask(output_path)
                self.statusBar().showMessage(f"Saved mask: {output_path.name}")
            if (
                not mask_same
                or calibration_changed
                or not self._measurement_is_current(
                    self._sequence_output_dir, image_path.name
                )
            ):
                self._record_measurement(self._sequence_output_dir, image_path.name)
        except Exception as exc:
            QMessageBox.warning(self, "Save error", str(exc))

    def _remove_emptied_sequence_mask(self, image_path):
        """Delete the saved mask of an image whose mask was erased down to nothing."""
        # An empty canvas only means "erased" when a mask was loaded or drawn for
        # the image on screen; a saved mask that was never shown must be kept.
        if not self.canvas.has_mask_data() or self._mask_hidden_by_roi():
            return
        if not self.canvas.image_path or Path(self.canvas.image_path) != image_path:
            return
        if self._shown_result_dir() != self._sequence_output_dir:
            return
        try:
            for mask_path in self._saved_mask_paths_for_current():
                mask_path.unlink()
        except OSError as exc:
            QMessageBox.warning(self, "Save error", str(exc))
            return
        self._forget_measurement(self._sequence_output_dir, image_path.name)

    def _save_current_mask(self):
        if self._mode == "video":
            self._save_video_masks()
            return
        self._commit_pending_outline()
        if not self._canvas_has_mask():
            QMessageBox.information(self, "No mask", "Run segmentation or annotate before saving a mask.")
            return
        if self._sequence_output_dir and self._sequence_index >= 0:
            image_path = Path(self._sequence_paths[self._sequence_index])
            output_path = Path(self._sequence_output_dir) / f"{image_path.stem}.png"
            try:
                self._write_shown_mask(output_path)
                self.statusBar().showMessage(f"Saved mask: {output_path.name}")
                self._record_measurement(self._sequence_output_dir, image_path.name)
            except Exception as exc:
                QMessageBox.warning(self, "Save error", str(exc))
            return
        self.canvas.save_mask_dialog()

    def _save_video_frame_mask_dialog(self):
        if self._mode != "video":
            QMessageBox.information(self, "No video", "Load a video first.")
            return
        self._commit_pending_outline()
        if not self._canvas_has_mask():
            QMessageBox.information(self, "No mask", "Run segmentation or annotate before saving a mask.")
            return
        default_dir = str(Path(self._video_path).parent) if self._video_path else ""
        video_stem = Path(self._video_path).stem if self._video_path else "video"
        default_name = f"{video_stem}_frame_{self._video_frame_index:06d}.png"
        default_path = str(Path(default_dir) / default_name) if default_dir else default_name
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save frame mask",
            default_path,
            "Mask Files (*.tif *.tiff *.png)",
        )
        if not path:
            return
        try:
            self.canvas.save_mask(path)
            self._stash_video_mask_for_current_frame()
            self.statusBar().showMessage(f"Saved mask: {Path(path).name}")
        except Exception as exc:
            QMessageBox.warning(self, "Save error", str(exc))

    def _save_video_masks(self):
        if self._mode != "video":
            QMessageBox.information(self, "No video", "Load a video first.")
            return
        if not self._video_path:
            QMessageBox.information(self, "No video", "Load a video first.")
            return
        self._commit_pending_outline()
        self._stash_video_mask_for_current_frame(force=True)
        output_dir = (self._video_output_dir or "").strip()
        if not output_dir:
            output_dir = QFileDialog.getExistingDirectory(
                self,
                "Select output folder",
                self._last_video_output_dir or self._last_video_input_dir,
            )
            if not output_dir:
                return
            self._video_output_dir = output_dir
            self._last_video_output_dir = output_dir
            self._save_persisted_paths()
            self._update_output_dir_tooltip()
        video_output_root = self._video_output_root_for_path(self._video_path)
        annotated_count = 0
        if video_output_root.exists() and video_output_root.is_dir():
            for mask_path in sorted(video_output_root.glob("frame_*.png")):
                suffix = mask_path.stem.split("_")[-1]
                if suffix.isdigit():
                    annotated_count += 1
        if annotated_count <= 0:
            QMessageBox.information(self, "No masks", "No annotated frames found.")
            return
        self.statusBar().showMessage(
            f"Saved {annotated_count}/{self._video_frame_count} annotated frames "
            f"for {Path(self._video_path).name}"
        )

    def _find_sequence_mask_path(self, image_path):
        if not self._sequence_output_dir:
            return None
        found = find_mask_files(self._sequence_output_dir, image_path, self._sequence_paths)
        return str(found[0]) if found else None

    def _show_load_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Load files")
        layout = QFormLayout(dialog)

        input_line = QLineEdit(dialog)
        output_line = QLineEdit(dialog)
        input_line.setReadOnly(True)

        selected_paths = []
        selection = {"kind": self._last_load_kind}

        def input_dir():
            if selection["kind"] == "video":
                return self._last_video_input_dir or self._last_image_input_dir
            return self._last_image_input_dir or self._last_video_input_dir

        def default_output_dir():
            if selection["kind"] == "video":
                remembered = self._last_video_output_dir
            else:
                remembered = self._last_image_output_dir
            directory = self._sequence_base_dir or self._video_base_dir or remembered
            return directory if directory and Path(directory).is_dir() else ""

        # The suggested folder follows the kind of file selected until the user
        # picks or types a folder of their own.
        suggested = {"dir": default_output_dir()}
        output_line.setText(suggested["dir"])

        def with_suffix(paths, extensions):
            return [str(path) for path in paths if Path(path).suffix.lower() in extensions]

        def choose(kind, paths, description, directory):
            selected_paths[:] = paths
            selection["kind"] = kind
            noun = "videos" if kind == "video" else "images"
            input_line.setText(f"{description}{len(paths)} {noun}")
            if kind == "video":
                self._last_video_input_dir = directory
            else:
                self._last_image_input_dir = directory
            self._last_load_kind = kind
            self._save_persisted_paths()
            if output_line.text().strip() == suggested["dir"]:
                suggested["dir"] = default_output_dir()
                output_line.setText(suggested["dir"])

        def ask_kind():
            box = QMessageBox(dialog)
            box.setWindowTitle("Images or videos")
            box.setText("This folder contains both images and videos. Which do you want to load?")
            images_btn = box.addButton("Images", QMessageBox.ButtonRole.AcceptRole)
            videos_btn = box.addButton("Videos", QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() is images_btn:
                return "image"
            if box.clickedButton() is videos_btn:
                return "video"
            return None

        def select_folder():
            directory = QFileDialog.getExistingDirectory(dialog, "Select input folder", input_dir())
            if not directory:
                return
            entries = sorted(Path(directory).iterdir())
            images = with_suffix(entries, IMAGE_EXTENSIONS)
            videos = with_suffix(entries, VIDEO_EXTENSIONS)
            if images and videos:
                kind = ask_kind()
                if kind is None:
                    return
            elif images or videos:
                kind = "image" if images else "video"
            else:
                QMessageBox.information(
                    dialog, "No files", "No images or videos found in the selected folder."
                )
                return
            choose(kind, videos if kind == "video" else images, f"{directory}: ", directory)

        def select_files():
            image_filter = " ".join(f"*{ext}" for ext in IMAGE_EXTENSIONS)
            video_filter = " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS)
            files, _ = QFileDialog.getOpenFileNames(
                dialog,
                "Select files",
                input_dir(),
                f"Images and videos ({image_filter} {video_filter});;"
                f"Images ({image_filter});;Videos ({video_filter})",
            )
            if not files:
                return
            images = with_suffix(files, IMAGE_EXTENSIONS)
            videos = with_suffix(files, VIDEO_EXTENSIONS)
            if images and videos:
                QMessageBox.information(
                    dialog,
                    "Mixed selection",
                    "Select either images or videos, not both.",
                )
                return
            if not images and not videos:
                QMessageBox.information(dialog, "No files", "Select image or video files.")
                return
            kind = "image" if images else "video"
            choose(kind, images or videos, "", str(Path(files[0]).parent))

        def browse_output():
            if selection["kind"] == "video":
                start_dir = self._last_video_output_dir or self._last_video_input_dir
            else:
                start_dir = self._last_image_output_dir or self._last_image_input_dir
            directory = QFileDialog.getExistingDirectory(
                dialog, "Select output folder", output_line.text().strip() or start_dir
            )
            if directory:
                output_line.setText(directory)

        input_row = QHBoxLayout()
        folder_btn = QPushButton("Select folder")
        files_btn = QPushButton("Select files")
        folder_btn.clicked.connect(select_folder)
        files_btn.clicked.connect(select_files)
        input_row.addWidget(folder_btn)
        input_row.addWidget(files_btn)
        layout.addRow("Input source:", input_row)
        layout.addRow("Input selection:", input_line)

        output_row = QHBoxLayout()
        output_btn = QPushButton("Browse")
        output_btn.clicked.connect(browse_output)
        output_row.addWidget(output_line)
        output_row.addWidget(output_btn)
        layout.addRow("Output folder:", output_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addRow(buttons)

        def validate_and_accept():
            if not selected_paths:
                QMessageBox.information(dialog, "Missing fields", "Select input files or a folder.")
                return
            if not output_line.text().strip():
                QMessageBox.information(dialog, "Missing fields", "Select an output folder.")
                return
            dialog.accept()

        buttons.accepted.connect(validate_and_accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        output_dir = output_line.text().strip()
        if selection["kind"] == "video":
            self._last_video_output_dir = output_dir
        else:
            self._last_image_output_dir = output_dir
        self._save_persisted_paths()
        return selection["kind"], list(selected_paths), output_dir

    def _preload_model(self):
        if not self._model.has_model():
            return
        self.statusBar().showMessage("Loading model...")
        try:
            gpu_provider = None
            for _label, provider in self._model.available_devices():
                if provider == "CUDAExecutionProvider":
                    gpu_provider = provider
                    break
            if gpu_provider:
                self._model.set_device(gpu_provider)
                self._model.preload()
                warning = self._model.device_warning()
                if warning:
                    self._gpu_warning = warning
                    self._select_cpu_device(show_warning=False)
                else:
                    self._set_device_picker(gpu_provider)
            else:
                self._model.preload()
            self.statusBar().showMessage("Ready")
        except Exception as exc:
            self.statusBar().showMessage("Model load failed")
            QMessageBox.warning(self, "Model error", str(exc))


    def _init_model_picker(self):
        models = self._model.list_models()
        if not models:
            self.model_picker.clear()
            self.model_picker.addItem("No models loaded")
            self.model_picker.setEnabled(False)
            return
        self.model_picker.clear()
        self.model_picker.addItems(models)
        self.model_picker.addItem(NO_MODEL_LABEL)
        current = self._model.current_model()
        if current and current in models:
            self.model_picker.setCurrentText(current)
        self.model_picker.setEnabled(True)

    def _init_device_picker(self):
        devices = self._model.available_devices()
        self.device_picker.clear()
        for label, provider in devices:
            self.device_picker.addItem(label, provider)
        if not devices:
            self.device_picker.addItem("CPU", "CPUExecutionProvider")
        current = self._model.current_device()
        index = self.device_picker.findData(current)
        if index >= 0:
            self.device_picker.setCurrentIndex(index)
        self.device_picker.setEnabled(True)

    def _set_device_picker(self, provider):
        index = self.device_picker.findData(provider)
        if index >= 0:
            self.device_picker.blockSignals(True)
            self.device_picker.setCurrentIndex(index)
            self.device_picker.blockSignals(False)

    def _select_cpu_device(self, show_warning):
        self._model.set_device("CPUExecutionProvider")
        self._model.preload()
        self._set_device_picker("CPUExecutionProvider")
        if show_warning and self._gpu_warning:
            QMessageBox.warning(self, "GPU unavailable", self._gpu_warning)

    def _on_model_changed(self, index):
        if not self.model_picker.isEnabled():
            return
        name = self.model_picker.currentText().strip()
        if not name or name == "No models loaded":
            return
        # Results are kept per model: what is on screen is saved under the model
        # being left before the canvas shows what the chosen model has saved.
        self._stop_playback()
        if self._mode == "video":
            self._stash_video_mask_for_current_frame()
        elif self._mode == "sequence":
            self._save_sequence_mask_if_needed()
        try:
            self._model.set_model("" if name == NO_MODEL_LABEL else name)
            self.statusBar().showMessage(f"Model selected: {name}")
        except Exception as exc:
            QMessageBox.warning(self, "Model error", str(exc))
        self._show_saved_result()

    def _on_device_changed(self, index):
        if not self.device_picker.isEnabled():
            return
        provider = self.device_picker.currentData()
        if not provider:
            return
        if provider != "CUDAExecutionProvider":
            self._model.set_device(provider)
            if self._model.preload():
                self.statusBar().showMessage(f"Device selected: {self.device_picker.currentText()}")
            return

        if self._gpu_warning:
            self._select_cpu_device(show_warning=True)
            return
        try:
            self._model.set_device(provider)
            if not self._model.preload():
                raise RuntimeError("Model session could not be created.")
            warning = self._model.device_warning()
            if warning:
                self._gpu_warning = warning
                self._select_cpu_device(show_warning=True)
                return
            self._gpu_warning = None
            self.statusBar().showMessage(f"Device selected: {self.device_picker.currentText()}")
        except Exception as exc:
            self._gpu_warning = f"{GPU_FALLBACK_WARNING}\n\nDetails: {exc}"
            self._select_cpu_device(show_warning=True)

    def _run_inference(self):
        if self.canvas.image is None:
            QMessageBox.warning(self, "No image", "Load an image before running segmentation.")
            return
        if not self._model.has_model():
            QMessageBox.warning(
                self,
                "No model",
                "Select a model in the Model list first.",
            )
            return
        if self._inference_thread and self._inference_thread.isRunning():
            self.statusBar().showMessage("Segmentation already running...")
            return
        if self._batch_thread and self._batch_thread.isRunning():
            self.statusBar().showMessage("Batch segmentation already running...")
            return

        self.statusBar().showMessage("Running segmentation...")
        self._show_inference_dialog()

        self._inference_thread = QThread(self)
        self._inference_box = box_to_segment(self.canvas.image, self.canvas.roi_box())
        self._inference_worker = InferenceWorker(
            self._model, crop_to_box(self.canvas.image, self._inference_box)
        )
        self._inference_worker.moveToThread(self._inference_thread)
        self._inference_thread.started.connect(self._inference_worker.run)
        self._inference_worker.finished.connect(self._on_inference_finished)
        self._inference_worker.canceled.connect(self._on_inference_canceled)
        self._inference_worker.error.connect(self._on_inference_error)
        self._inference_worker.finished.connect(self._inference_thread.quit)
        self._inference_worker.canceled.connect(self._inference_thread.quit)
        self._inference_worker.error.connect(self._inference_thread.quit)
        self._inference_worker.finished.connect(self._inference_worker.deleteLater)
        self._inference_worker.canceled.connect(self._inference_worker.deleteLater)
        self._inference_worker.error.connect(self._inference_worker.deleteLater)
        self._inference_thread.finished.connect(self._on_inference_thread_done)
        self._inference_thread.finished.connect(self._inference_thread.deleteLater)
        self._inference_thread.start()

    def _show_inference_dialog(self):
        if self._inference_dialog is not None:
            self._inference_dialog.close()
        self._inference_dialog = QProgressDialog(
            "Running segmentation...",
            "Cancel",
            0,
            0,
            self,
        )
        self._inference_dialog.setWindowTitle("Segment")
        self._inference_dialog.setMinimumDuration(0)
        self._inference_dialog.setAutoClose(False)
        self._inference_dialog.setAutoReset(False)
        self._inference_dialog.canceled.connect(self._request_inference_cancel)
        self._inference_dialog.show()

    def _close_inference_dialog(self):
        if self._inference_dialog is None:
            return
        self._inference_dialog.close()
        self._inference_dialog = None

    def _request_inference_cancel(self):
        if self._inference_worker:
            self._inference_worker.cancel()
        self.statusBar().showMessage("Canceling segmentation...")
        if self._inference_dialog:
            self._inference_dialog.setLabelText("Canceling segmentation...")

    def _on_inference_finished(self, prediction):
        self._last_prediction = prediction
        if self._last_prediction is not None and self.canvas.image is not None:
            self.canvas.set_mask(
                place_prediction(
                    self._last_prediction, self.canvas.image, self._inference_box
                )
            )
        self.statusBar().showMessage("Segmentation complete")
        self._close_inference_dialog()

    def _on_inference_canceled(self):
        self.statusBar().showMessage("Segmentation canceled")
        self._close_inference_dialog()

    def _on_inference_error(self, message):
        QMessageBox.warning(self, "Model error", message)
        self.statusBar().showMessage("Segmentation failed")
        self._close_inference_dialog()

    def _on_inference_thread_done(self):
        self._inference_thread = None
        self._inference_worker = None
        self.statusBar().showMessage("Ready")

    def closeEvent(self, event):
        self._stop_playback()
        if self.canvas.is_calibrating():
            self._end_calibration()
        self._save_sequence_mask_if_needed()
        self._stash_video_mask_for_current_frame()
        # A segmentation still running in the background is told to stop and
        # given time to finish the file it is writing.
        for worker, thread in (
            (self._inference_worker, self._inference_thread),
            (self._batch_worker, self._batch_thread),
        ):
            if worker is not None and thread is not None and thread.isRunning():
                worker.cancel()
                thread.quit()
                thread.wait(5000)
        self._clear_video_state()
        super().closeEvent(event)


class VideoBatchProgressDialog(QDialog):
    canceled = pyqtSignal()

    def __init__(self, video_count, title, parent=None):
        super().__init__(parent)
        self._video_count = max(1, int(video_count))
        self._cancel_requested = False
        self._finished = False
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        self.status_label = QLabel("Preparing videos...")
        layout.addWidget(self.status_label)

        self.video_label = QLabel(f"Videos: 0/{self._video_count}")
        layout.addWidget(self.video_label)
        self.video_progress = QProgressBar(self)
        self.video_progress.setRange(0, self._video_count)
        self.video_progress.setValue(0)
        layout.addWidget(self.video_progress)

        self.frame_label = QLabel("Frames: 0/0")
        layout.addWidget(self.frame_label)
        self.frame_progress = QProgressBar(self)
        self.frame_progress.setRange(0, 1)
        self.frame_progress.setValue(0)
        layout.addWidget(self.frame_progress)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._request_cancel)
        layout.addWidget(self.cancel_btn)

    def start_video(self, video_name, video_number, video_count, frame_count):
        self._video_count = max(1, int(video_count))
        self.status_label.setText(f"Segmenting {video_name}")
        self.video_label.setText(f"Videos: {video_number}/{video_count}")
        self.video_progress.setRange(0, self._video_count)
        self.video_progress.setValue(max(0, int(video_number) - 1))
        self.frame_label.setText(f"Frames: 0/{frame_count}")
        self.frame_progress.setRange(0, max(1, int(frame_count)))
        self.frame_progress.setValue(0)

    def start_frame(self, frame_number, frame_count):
        self.frame_label.setText(f"Frames: {frame_number}/{frame_count}")

    def set_frame_progress(self, frame_number, frame_count):
        self.frame_progress.setRange(0, max(1, int(frame_count)))
        self.frame_progress.setValue(int(frame_number))

    def set_video_progress(self, video_number, video_count):
        self.video_progress.setRange(0, max(1, int(video_count)))
        self.video_progress.setValue(int(video_number))

    def mark_canceling(self):
        self._cancel_requested = True
        self.status_label.setText("Canceling video segmentation...")
        self.cancel_btn.setEnabled(False)

    def mark_complete(self):
        self._finished = True
        self.video_progress.setValue(self.video_progress.maximum())
        self.frame_progress.setValue(self.frame_progress.maximum())

    def mark_stopped(self):
        self._finished = True

    def _request_cancel(self):
        if self._cancel_requested:
            return
        self.mark_canceling()
        self.canceled.emit()

    def closeEvent(self, event):
        if not self._finished and not self._cancel_requested:
            self._request_cancel()
        super().closeEvent(event)


class InferenceWorker(QObject):
    finished = pyqtSignal(object)
    canceled = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model, image):
        super().__init__()
        self._model = model
        self._image = image
        self._cancel_event = Event()

    def cancel(self):
        self._cancel_event.set()

    @pyqtSlot()
    def run(self):
        if self._cancel_event.is_set():
            self.canceled.emit()
            return
        try:
            prediction = self._model.run_inference(
                self._image,
                cancel_event=self._cancel_event,
            )
        except Exception as exc:
            if str(exc).lower().startswith("inference canceled"):
                self.canceled.emit()
                return
            self.error.emit(str(exc))
            return
        if self._cancel_event.is_set():
            self.canceled.emit()
            return
        self.finished.emit(prediction)


class BatchInferenceWorker(QObject):
    finished = pyqtSignal(int, int)
    canceled = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    image_started = pyqtSignal(str, int, int)

    def __init__(
        self,
        model,
        image_paths,
        output_dir,
        overwrite_existing=True,
        px_per_mm=(0.0, 0.0),
        knee_side="right",
        rois=None,
        rotations=None,
        measurements=None,
    ):
        super().__init__()
        self._rotations = rotations or {}
        # (folder, file name) of the measurements file the results go into.
        self._measurements = measurements or (output_dir, MEASUREMENTS_FILE_NAME)
        self._model = model
        self._image_paths = list(image_paths)
        self._output_dir = output_dir
        self._overwrite_existing = bool(overwrite_existing)
        self._px_per_mm = px_per_mm
        self._knee_side = knee_side
        self._rois = rois or {}
        self._measurement_log = None
        self._cancel_event = Event()

    def cancel(self):
        self._cancel_event.set()

    @pyqtSlot()
    def run(self):
        try:
            self._measurement_log = MeasurementLog(*self._measurements)
            self._segment_images()
            self._measurement_log.save()
        except OSError as exc:
            self.error.emit(f"Could not save measurements: {exc}")

    def _segment_images(self):
        total = len(self._image_paths)
        if total == 0:
            self.error.emit("No images found for batch segmentation.")
            return
        processed = 0
        skipped = 0
        for idx, image_path in enumerate(self._image_paths, start=1):
            if self._cancel_event.is_set():
                self.canceled.emit()
                return
            self.image_started.emit(image_path, idx, total)
            if not self._overwrite_existing and self._existing_mask_path(image_path):
                skipped += 1
                self.progress.emit(idx, total)
                continue
            try:
                angle = self._rotations.get(Path(image_path).name, 0.0)
                image = self._load_image(image_path)
                stored_shape = image.shape[:2]
                image = rotate_image(image, angle)
                box = box_to_segment(
                    image, self._rois.get((Path(image_path).name, ""))
                )
                prediction = self._model.run_inference(
                    crop_to_box(image, box),
                    cancel_event=self._cancel_event,
                )
                if prediction is not None:
                    mask = place_prediction(prediction, image, box)
                    write_mask_file(
                        Path(self._output_dir) / f"{Path(image_path).stem}.png",
                        mask >= 0.5,
                        angle,
                        stored_shape,
                    )
                    self._measurement_log.set(
                        Path(image_path).name,
                        "",
                        image,
                        mask >= 0.5,
                        *self._px_per_mm,
                        knee_side=self._knee_side,
                        prefer_saved=True,
                        limits=box,
                        rotation=angle,
                    )
            except Exception as exc:
                if str(exc).lower().startswith("inference canceled"):
                    self.canceled.emit()
                    return
                self.error.emit(str(exc))
                return
            processed += 1
            self.progress.emit(idx, total)
        self.finished.emit(processed, skipped)

    def _existing_mask_path(self, image_path):
        found = find_mask_files(self._output_dir, image_path, self._image_paths)
        return found[0] if found else None

    def _load_image(self, file_path):
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


class VideoBatchInferenceWorker(QObject):
    finished = pyqtSignal(int, int)
    canceled = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    video_started = pyqtSignal(str, int, int, int)
    video_finished = pyqtSignal(int, int)
    frame_started = pyqtSignal(str, int, int, int, int)
    frame_progress = pyqtSignal(int, int)

    def __init__(
        self,
        model,
        video_paths,
        output_dir,
        overwrite_existing=True,
        px_per_mm=(0.0, 0.0),
        knee_side="right",
        rois=None,
        rotations=None,
        measurements=None,
    ):
        super().__init__()
        self._rotations = rotations or {}
        # (folder, file name) of the measurements file the results go into.
        self._measurements = measurements or (output_dir, MEASUREMENTS_FILE_NAME)
        self._model = model
        self._video_paths = list(video_paths)
        self._output_dir = Path(output_dir)
        self._overwrite_existing = bool(overwrite_existing)
        self._px_per_mm = px_per_mm
        self._knee_side = knee_side
        self._rois = rois or {}
        self._cancel_event = Event()

    def cancel(self):
        self._cancel_event.set()

    @pyqtSlot()
    def run(self):
        try:
            total_frames = 0
            for video_path in self._video_paths:
                capture = cv2.VideoCapture(video_path)
                if not capture.isOpened():
                    capture.release()
                    raise RuntimeError(f"Could not open video: {Path(video_path).name}")
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count <= 0:
                    capture.release()
                    raise RuntimeError(f"Video has no readable frames: {Path(video_path).name}")
                total_frames += frame_count
                capture.release()

            processed = 0
            skipped = 0
            video_count = len(self._video_paths)
            measurement_log = MeasurementLog(*self._measurements)
            for video_number, video_path in enumerate(self._video_paths, start=1):
                video_name = Path(video_path).name
                capture = cv2.VideoCapture(video_path)
                if not capture.isOpened():
                    capture.release()
                    raise RuntimeError(f"Could not open video: {video_name}")
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_started.emit(video_name, video_number, video_count, frame_count)
                video_output_dir = self._output_dir / Path(video_path).stem
                video_output_dir.mkdir(parents=True, exist_ok=True)
                try:
                    for frame_index in range(frame_count):
                        if self._cancel_event.is_set():
                            self.canceled.emit()
                            return
                        self.frame_started.emit(
                            video_name,
                            frame_index + 1,
                            frame_count,
                            processed + 1,
                            total_frames,
                        )
                        output_path = video_output_dir / f"frame_{frame_index:06d}.png"
                        success, frame = capture.read()
                        if not success or frame is None:
                            raise RuntimeError(
                                f"Could not decode {video_name} frame {frame_index}."
                            )
                        if not self._overwrite_existing and output_path.exists():
                            skipped += 1
                            self.progress.emit(processed + skipped, total_frames)
                            self.frame_progress.emit(frame_index + 1, frame_count)
                            continue
                        if frame.ndim == 3 and frame.shape[2] == 3:
                            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        else:
                            image = frame
                        angle = self._rotations.get(video_name, 0.0)
                        stored_shape = image.shape[:2]
                        image = rotate_image(image, angle)
                        box = box_to_segment(
                            image, self._rois.get((video_name, str(frame_index)))
                        )
                        prediction = self._model.run_inference(
                            crop_to_box(image, box),
                            cancel_event=self._cancel_event,
                        )
                        if self._cancel_event.is_set():
                            self.canceled.emit()
                            return
                        mask = place_prediction(prediction, image, box)
                        write_mask_file(output_path, mask >= 0.5, angle, stored_shape)
                        measurement_log.set(
                            video_name,
                            frame_index,
                            image,
                            mask >= 0.5,
                            *self._px_per_mm,
                            knee_side=self._knee_side,
                            prefer_saved=True,
                            limits=box,
                            rotation=angle,
                        )
                        processed += 1
                        self.progress.emit(processed, total_frames)
                        self.frame_progress.emit(frame_index + 1, frame_count)
                finally:
                    capture.release()
                    measurement_log.save()
                self.video_finished.emit(video_number, video_count)
            self.finished.emit(processed, skipped)
        except Exception as exc:
            if self._cancel_event.is_set() or str(exc).lower().startswith("inference canceled"):
                self.canceled.emit()
            else:
                self.error.emit(str(exc))
