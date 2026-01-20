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
    QScrollArea,
    QSizePolicy,
    QStyle,
    QApplication,
    QMessageBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

from .canvas import Canvas
from .model_integration import ModelIntegration


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
        self._wire_actions()
        self._model = ModelIntegration()
        self._last_prediction = None

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
        edit_menu.addAction(self.undo_action)
        self.redo_action = QAction("Redo", self)
        edit_menu.addAction(self.redo_action)

        tools_menu = menubar.addMenu("Tools")
        self.select_pan_action = QAction("Select/Pan", self)
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

        layout.addSpacing(10)

        layout.addWidget(QLabel("Image Sequence:"))
        self.sequence_btn = QPushButton("Load image sequence")
        configure_button(self.sequence_btn)
        layout.addWidget(self.sequence_btn)
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
            ["Select/Pan", "Freehand Line", "Segmented Line", "Paint Brush", "Eraser"]
        )
        self.tool_picker.setCurrentIndex(0)
        layout.addWidget(self.tool_picker)

        layout.addSpacing(10)

        layout.addWidget(QLabel("Brush Radius:"))
        self.brush_radius = QSpinBox()
        self.brush_radius.setMinimum(1)
        self.brush_radius.setMaximum(50)
        self.brush_radius.setValue(4)
        layout.addWidget(self.brush_radius)

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

        self.fit_btn = QPushButton("Fit to Window")
        configure_button(self.fit_btn)
        layout.addWidget(self.fit_btn)

        layout.addSpacing(20)

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
        self.exit_action.triggered.connect(self.close)
        self.run_btn.clicked.connect(self._run_inference)
        self.segment_action.triggered.connect(self._run_inference)

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
        target_w = int(screen_rect.width() * 0.35)
        target_h = int(screen_rect.height() * 0.45)
        width = min(target_w, max_w)
        height = min(target_h, max_h)
        min_w = min(250, max_w)
        min_h = min(250, max_h)
        width = max(min_w, width)
        height = max(min_h, height)
        self._sidebar_width = min(240, max(160, int(screen_rect.width() * 0.2)))
        return (width, height)

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
        self.statusBar().showMessage("Running segmentation...")
        try:
            self._last_prediction = self._model.run_inference(self.canvas.image)
        except Exception as exc:
            QMessageBox.warning(self, "Model error", str(exc))
            self.statusBar().showMessage("Segmentation failed")
            return
        if self._last_prediction is not None:
            self.canvas.set_mask(self._last_prediction)
        self.statusBar().showMessage("Segmentation complete")
