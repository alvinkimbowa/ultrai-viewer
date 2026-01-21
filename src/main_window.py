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
    QComboBox,
    QCheckBox,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QApplication,
    QMessageBox,
    QProgressDialog,
    QFileDialog,
    QDialog,
    QLineEdit,
    QFormLayout,
    QDialogButtonBox,
)
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QKeySequence, QShortcut

from .canvas import Canvas
from .model_integration import ModelIntegration
from threading import Event
from pathlib import Path
import numpy as np
import cv2
import tifffile


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Knee Ultrasound Viewer")
        self._sidebar_width = 220

        self._create_menu_bar()

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)

        self.canvas = Canvas()

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

        root_layout.addWidget(self.canvas, stretch=1)

        self.statusBar().showMessage("Ready")
        self._prev_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self._next_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self._wire_actions()
        self._model = ModelIntegration()
        self._last_prediction = None
        self._inference_thread = None
        self._inference_worker = None
        self._inference_dialog = None
        self._sequence_paths = []
        self._sequence_index = -1
        self._sequence_output_dir = None
        self._batch_thread = None
        self._batch_worker = None
        self._batch_dialog = None
        self._preload_model()
        self._init_model_picker()

        self._start_size = self._initial_window_size()
        self.setGeometry(10, 10, *self._start_size)
        self._center_window()

    def _create_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        self.load_image_action = QAction("Load Image...", self)
        file_menu.addAction(self.load_image_action)
        file_menu.addAction(QAction("Load Mask...", self))
        file_menu.addAction(QAction("Load image sequence...", self))
        self.segment_action = QAction("Segment", self)
        file_menu.addAction(self.segment_action)
        self.segment_batch_action = QAction("Batch segment", self)
        file_menu.addAction(self.segment_batch_action)
        file_menu.addSeparator()
        self.save_mask_action = QAction("Save mask...", self)
        self.save_mask_action.setShortcut(QKeySequence.StandardKey.Save)
        file_menu.addAction(self.save_mask_action)
        self.clear_mask_action = QAction("Clear Mask", self)
        file_menu.addAction(self.clear_mask_action)
        self.close_image_action = QAction("Close Image", self)
        file_menu.addAction(self.close_image_action)
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

        layout.addWidget(QLabel("Single Image:"))
        self.open_image_btn = QPushButton("Load Image")
        configure_button(self.open_image_btn)
        layout.addWidget(self.open_image_btn)

        self.open_mask_btn = QPushButton("Load Mask")
        configure_button(self.open_mask_btn)
        layout.addWidget(self.open_mask_btn)

        self.fit_btn = QPushButton("Fit to Window")
        configure_button(self.fit_btn)
        layout.addWidget(self.fit_btn)

        layout.addSpacing(10)

        layout.addWidget(QLabel("Image Sequence:"))
        self.sequence_btn = QPushButton("Load image sequence")
        configure_button(self.sequence_btn)
        layout.addWidget(self.sequence_btn)
        self.clear_sequence_btn = QPushButton("Clear sequence")
        configure_button(self.clear_sequence_btn)
        layout.addWidget(self.clear_sequence_btn)
        self.sequence_combo = QComboBox()
        self.sequence_combo.setEnabled(False)
        layout.addWidget(self.sequence_combo)

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

        layout.addSpacing(10)

        layout.addWidget(QLabel("Model:"))
        self.model_picker = QComboBox()
        self.model_picker.addItem("No models loaded")
        self.model_picker.setEnabled(False)
        layout.addWidget(self.model_picker)

        self.run_btn = QPushButton("Segment")
        configure_button(self.run_btn)
        layout.addWidget(self.run_btn)

        self.run_batch_btn = QPushButton("Batch segment")
        configure_button(self.run_batch_btn)
        layout.addWidget(self.run_batch_btn)

        layout.addSpacing(20)

        layout.addWidget(QLabel("Tools:"))
        self.tool_picker = QComboBox()
        self.tool_picker.addItems(
            ["Select", "Freehand Line", "Segmented Line", "Paint Brush", "Eraser"]
        )
        self.tool_picker.setCurrentIndex(0)
        layout.addWidget(self.tool_picker)

        self.fill_roi_checkbox = QCheckBox("Fill ROI")
        self.fill_roi_checkbox.setChecked(False)
        layout.addWidget(self.fill_roi_checkbox)

        layout.addSpacing(10)

        layout.addWidget(QLabel("Brush Radius:"))
        self.brush_radius = QSlider(Qt.Orientation.Horizontal)
        self.brush_radius.setMinimum(1)
        self.brush_radius.setMaximum(50)
        self.brush_radius.setValue(4)
        layout.addWidget(self.brush_radius)
        self.brush_radius_label = QLabel("4 px")
        layout.addWidget(self.brush_radius_label)

        layout.addSpacing(10)

        self.undo_btn = QPushButton("Undo")
        configure_button(self.undo_btn)
        layout.addWidget(self.undo_btn)

        self.redo_btn = QPushButton("Redo")
        configure_button(self.redo_btn)
        layout.addWidget(self.redo_btn)

        layout.addSpacing(20)

        layout.addWidget(QLabel("Mask Opacity:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(50)
        layout.addWidget(self.opacity_slider)

        layout.addSpacing(10)

        self.save_btn = QPushButton("Save mask")
        configure_button(self.save_btn)
        layout.addWidget(self.save_btn)

        self.clear_btn = QPushButton("Clear Mask")
        configure_button(self.clear_btn)
        layout.addWidget(self.clear_btn)

        self.close_btn = QPushButton("Close Image")
        configure_button(self.close_btn)
        layout.addWidget(self.close_btn)

        layout.addStretch()

        return panel

    def _wire_actions(self):
        self.open_image_btn.clicked.connect(self.canvas.load_image_dialog)
        self.load_image_action.triggered.connect(self.canvas.load_image_dialog)
        self.open_mask_btn.clicked.connect(self.canvas.load_mask_dialog)
        self.exit_action.triggered.connect(self.close)
        self.run_btn.clicked.connect(self._run_inference)
        self.segment_action.triggered.connect(self._run_inference)
        self.run_batch_btn.clicked.connect(self._run_batch_inference)
        self.segment_batch_action.triggered.connect(self._run_batch_inference)
        self.save_btn.clicked.connect(self.canvas.save_mask_dialog)
        self.save_mask_action.triggered.connect(self.canvas.save_mask_dialog)
        self.clear_btn.clicked.connect(self.canvas.clear_mask)
        self.clear_mask_action.triggered.connect(self.canvas.clear_mask)
        self.close_btn.clicked.connect(self.canvas.clear_image)
        self.close_image_action.triggered.connect(self.canvas.clear_image)
        self.tool_picker.currentIndexChanged.connect(self._on_tool_changed)
        self.brush_radius.valueChanged.connect(self._on_brush_radius_changed)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self.fit_btn.clicked.connect(self.canvas.fit_to_window)
        self.fill_roi_checkbox.toggled.connect(self.canvas.set_fill_roi)
        self.undo_btn.clicked.connect(self.canvas.undo)
        self.redo_btn.clicked.connect(self.canvas.redo)
        self.undo_action.triggered.connect(self.canvas.undo)
        self.redo_action.triggered.connect(self.canvas.redo)
        self.model_picker.currentIndexChanged.connect(self._on_model_changed)
        self.sequence_btn.clicked.connect(self._load_sequence)
        self.clear_sequence_btn.clicked.connect(self._clear_sequence)
        self.sequence_combo.currentIndexChanged.connect(self._on_sequence_selected)
        self.prev_btn.clicked.connect(self._show_previous_sequence)
        self.next_btn.clicked.connect(self._show_next_sequence)
        self._prev_shortcut.activated.connect(self._show_previous_sequence)
        self._next_shortcut.activated.connect(self._show_next_sequence)

    def _center_window(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        frame = self.frameGeometry()
        frame.moveCenter(screen.availableGeometry().center())
        self.move(frame.topLeft())

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
        max_w = max(300, screen_rect.width() - margin)
        max_h = max(300, screen_rect.height() - margin)
        target_w = int(screen_rect.width() * 0.75)
        target_h = int(screen_rect.height() * 0.75)
        width = min(target_w, max_w)
        height = min(target_h, max_h)
        min_w = min(250, max_w)
        min_h = min(250, max_h)
        width = max(min_w, width)
        height = max(min_h, height)
        self._sidebar_width = min(240, max(160, int(screen_rect.width() * 0.2)))
        return (width, height)

    def _on_tool_changed(self, index):
        tool_map = {
            0: "select",
            1: "freehand",
            2: "polyline",
            3: "brush",
            4: "eraser",
        }
        tool = tool_map.get(index, "select")
        self.canvas.set_tool(tool)
        if tool in ("brush", "eraser"):
            self.fill_roi_checkbox.setChecked(True)

    def _on_opacity_changed(self, value):
        self.canvas.set_mask_opacity(value / 100.0)

    def _on_brush_radius_changed(self, value):
        self.canvas.set_brush_radius(value)
        if hasattr(self, "brush_radius_label"):
            self.brush_radius_label.setText(f"{int(value)} px")

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
                "No ONNX model found in assets/.",
            )
            return
        if self._batch_thread and self._batch_thread.isRunning():
            self.statusBar().showMessage("Batch segmentation already running...")
            return

        total = len(self._sequence_paths)
        self.statusBar().showMessage("Running batch segmentation...")
        self._show_batch_dialog(total)

        self._batch_thread = QThread(self)
        self._batch_worker = BatchInferenceWorker(
            self._model,
            list(self._sequence_paths),
            self._sequence_output_dir,
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

    def _show_batch_dialog(self, total):
        if self._batch_dialog is not None:
            self._batch_dialog.close()
        self._batch_dialog = QProgressDialog(
            "Running batch segmentation...",
            "Cancel",
            0,
            total,
            self,
        )
        self._batch_dialog.setWindowTitle("Batch segment")
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
            self._batch_dialog.setLabelText("Canceling batch segmentation...")

    def _on_batch_progress(self, current, total):
        if self._batch_dialog:
            self._batch_dialog.setValue(current)
            self._batch_dialog.setLabelText(f"Segmenting image {current}/{total}")

    def _on_batch_image_started(self, image_path, index, total):
        if self._batch_dialog:
            name = Path(image_path).name
            self._batch_dialog.setLabelText(f"Segmenting {name} ({index}/{total})")

    def _on_batch_finished(self, processed):
        self.statusBar().showMessage(f"Batch segmentation complete: {processed} images.")
        self._close_batch_dialog()

    def _on_batch_canceled(self):
        self.statusBar().showMessage("Batch segmentation canceled")
        self._close_batch_dialog()

    def _on_batch_error(self, message):
        QMessageBox.warning(self, "Batch error", message)
        self.statusBar().showMessage("Batch segmentation failed")
        self._close_batch_dialog()

    def _on_batch_thread_done(self):
        self._batch_thread = None
        self._batch_worker = None

    def _load_sequence(self):
        dialog_result = self._show_sequence_dialog()
        if dialog_result is None:
            return
        directory, output_dir = dialog_result
        extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        paths = [
            str(path)
            for path in sorted(Path(directory).iterdir())
            if path.suffix.lower() in extensions
        ]
        if not paths:
            QMessageBox.information(self, "No images", "No images found in the selected folder.")
            return
        self._sequence_paths = paths
        self._sequence_index = 0
        self._sequence_output_dir = output_dir
        self.sequence_combo.setEnabled(True)
        self.sequence_combo.clear()
        self.sequence_combo.addItems([Path(p).name for p in self._sequence_paths])
        self.sequence_combo.setCurrentIndex(0)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(len(self._sequence_paths) > 1)
        self._load_sequence_image()

    def _clear_sequence(self):
        self._sequence_paths = []
        self._sequence_index = -1
        self._sequence_output_dir = None
        self.sequence_combo.clear()
        self.sequence_combo.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.canvas.clear_image()
        self.statusBar().showMessage("Sequence cleared")

    def _on_sequence_selected(self, index):
        if index < 0 or index >= len(self._sequence_paths):
            return
        self._sequence_index = index
        self._load_sequence_image()

    def _show_previous_sequence(self):
        if self._sequence_index <= 0:
            return
        self._save_sequence_mask_if_needed()
        self._sequence_index -= 1
        self.sequence_combo.setCurrentIndex(self._sequence_index)

    def _show_next_sequence(self):
        if self._sequence_index < 0:
            return
        if self._sequence_index >= len(self._sequence_paths) - 1:
            return
        self._save_sequence_mask_if_needed()
        self._sequence_index += 1
        self.sequence_combo.setCurrentIndex(self._sequence_index)

    def _load_sequence_image(self):
        if self._sequence_index < 0 or self._sequence_index >= len(self._sequence_paths):
            return
        path = self._sequence_paths[self._sequence_index]
        self.canvas.load_image(path)
        mask_path = self._find_sequence_mask_path(path)
        if mask_path:
            self.canvas.load_mask(mask_path)
        else:
            self.canvas.clear_mask()
        self.prev_btn.setEnabled(self._sequence_index > 0)
        self.next_btn.setEnabled(self._sequence_index < len(self._sequence_paths) - 1)

    def _save_sequence_mask_if_needed(self):
        if not self._sequence_output_dir:
            return
        if self._sequence_index < 0 or self._sequence_index >= len(self._sequence_paths):
            return
        if not self.canvas.has_mask_data():
            return
        image_path = Path(self._sequence_paths[self._sequence_index])
        output_path = Path(self._sequence_output_dir) / image_path.name
        try:
            self.canvas.save_mask(str(output_path))
            self.statusBar().showMessage(f"Saved mask: {output_path.name}")
        except Exception as exc:
            QMessageBox.warning(self, "Save error", str(exc))

    def _find_sequence_mask_path(self, image_path):
        if not self._sequence_output_dir:
            return None
        stem = Path(image_path).stem
        output_dir = Path(self._sequence_output_dir)
        for ext in (".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"):
            candidate = output_dir / f"{stem}{ext}"
            if candidate.exists():
                return str(candidate)
        return None

    def _show_sequence_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Load image sequence")
        layout = QFormLayout(dialog)

        input_line = QLineEdit(dialog)
        output_line = QLineEdit(dialog)

        def browse_input():
            directory = QFileDialog.getExistingDirectory(dialog, "Select input folder")
            if directory:
                input_line.setText(directory)

        def browse_output():
            directory = QFileDialog.getExistingDirectory(dialog, "Select output folder")
            if directory:
                output_line.setText(directory)

        input_row = QHBoxLayout()
        input_btn = QPushButton("Browse")
        input_btn.clicked.connect(browse_input)
        input_row.addWidget(input_line)
        input_row.addWidget(input_btn)
        layout.addRow("Input folder:", input_row)

        output_row = QHBoxLayout()
        output_btn = QPushButton("Browse")
        output_btn.clicked.connect(browse_output)
        output_row.addWidget(output_line)
        output_row.addWidget(output_btn)
        layout.addRow("Output folder:", output_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addRow(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        input_dir = input_line.text().strip()
        output_dir = output_line.text().strip()
        if not input_dir or not output_dir:
            QMessageBox.information(self, "Missing fields", "Select both input and output folders.")
            return None
        return input_dir, output_dir

    def _preload_model(self):
        if not self._model.has_model():
            return
        self.statusBar().showMessage("Loading model...")
        try:
            if self._model.preload():
                self.statusBar().showMessage("Model loaded")
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
        current = self._model.current_model()
        if current and current in models:
            self.model_picker.setCurrentText(current)
        self.model_picker.setEnabled(True)

    def _on_model_changed(self, index):
        if not self.model_picker.isEnabled():
            return
        name = self.model_picker.currentText().strip()
        if not name or name == "No models loaded":
            return
        try:
            self._model.set_model(name)
            self.statusBar().showMessage(f"Model selected: {name}")
        except Exception as exc:
            QMessageBox.warning(self, "Model error", str(exc))

    def _run_inference(self):
        if self.canvas.image is None:
            QMessageBox.warning(self, "No image", "Load an image before running segmentation.")
            return
        if not self._model.has_model():
            QMessageBox.warning(
                self,
                "No model",
                "No ONNX model found in assets/.",
            )
            return
        if self._inference_thread and self._inference_thread.isRunning():
            self.statusBar().showMessage("Segmentation already running...")
            return

        self.statusBar().showMessage("Running segmentation...")
        self._show_inference_dialog()

        self._inference_thread = QThread(self)
        self._inference_worker = InferenceWorker(self._model, self.canvas.image)
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
        if self._last_prediction is not None:
            self.canvas.set_mask(self._last_prediction)
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
    finished = pyqtSignal(int)
    canceled = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    image_started = pyqtSignal(str, int, int)

    def __init__(self, model, image_paths, output_dir):
        super().__init__()
        self._model = model
        self._image_paths = list(image_paths)
        self._output_dir = output_dir
        self._cancel_event = Event()

    def cancel(self):
        self._cancel_event.set()

    @pyqtSlot()
    def run(self):
        total = len(self._image_paths)
        if total == 0:
            self.error.emit("No images found for batch segmentation.")
            return
        processed = 0
        for idx, image_path in enumerate(self._image_paths, start=1):
            if self._cancel_event.is_set():
                self.canceled.emit()
                return
            self.image_started.emit(image_path, idx, total)
            try:
                image = self._load_image(image_path)
                prediction = self._model.run_inference(
                    image,
                    cancel_event=self._cancel_event,
                )
                self._save_mask(prediction, image_path)
            except Exception as exc:
                if str(exc).lower().startswith("inference canceled"):
                    self.canceled.emit()
                    return
                self.error.emit(str(exc))
                return
            processed += 1
            self.progress.emit(processed, total)
        self.finished.emit(processed)

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

    def _save_mask(self, prediction, image_path):
        if prediction is None:
            return
        mask_uint8 = (prediction >= 0.5).astype(np.uint8) * 255
        output_path = Path(self._output_dir) / f"{Path(image_path).stem}.png"
        cv2.imwrite(str(output_path), mask_uint8)
