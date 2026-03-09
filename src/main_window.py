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
    QFrame,
    QButtonGroup,
    QRadioButton,
    QInputDialog,
    QGridLayout,
)
from PyQt6.QtCore import Qt, QObject, QThread, QTimer, QSize, QEvent, QSettings, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QShortcut

from .canvas import Canvas
from .model_integration import ModelIntegration, GPU_FALLBACK_WARNING
from threading import Event
from collections import OrderedDict
from pathlib import Path
from datetime import datetime, timezone
import json
import re
import sys
import numpy as np
import cv2
import tifffile


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._base_title = "UltAI Viewer"
        self.setWindowTitle(self._base_title)
        self._sidebar_width = 260

        self._create_menu_bar()

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)

        self.canvas = Canvas()
        self.canvas.image_loaded.connect(self._update_title_with_image)

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
        canvas_layout.setSpacing(6)
        nerve_panel = QWidget()
        nerve_panel_layout = QVBoxLayout(nerve_panel)
        nerve_panel_layout.setContentsMargins(0, 0, 0, 0)
        nerve_panel_layout.setSpacing(4)
        nerve_panel_layout.addWidget(QLabel("Nerve:"))
        self.nerve_labels_container = QWidget()
        self.nerve_labels_layout = QGridLayout(self.nerve_labels_container)
        self.nerve_labels_layout.setContentsMargins(0, 0, 0, 0)
        self.nerve_labels_layout.setSpacing(3)
        nerve_panel_layout.addWidget(self.nerve_labels_container)
        nerve_actions_row = QHBoxLayout()
        nerve_actions_row.setContentsMargins(0, 0, 0, 0)
        nerve_actions_row.setSpacing(6)
        self.add_nerve_btn = QPushButton("+ Add nerve")
        self.nerve_summary_label = QLabel("Label: - (0/0 labeled)")
        nerve_actions_row.addWidget(self.add_nerve_btn, stretch=0)
        nerve_actions_row.addWidget(self.nerve_summary_label, stretch=1)
        nerve_panel_layout.addLayout(nerve_actions_row)
        nerve_sep = QFrame()
        nerve_sep.setFrameShape(QFrame.Shape.HLine)
        nerve_sep.setFrameShadow(QFrame.Shadow.Sunken)
        nerve_panel_layout.addWidget(nerve_sep)
        canvas_layout.addWidget(nerve_panel, stretch=0)
        canvas_layout.addWidget(self.canvas, stretch=1)
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
        self._undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self._redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        self._redo_alt_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        self._clear_mask_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self)
        self._prev_video_shortcut = QShortcut(QKeySequence(Qt.Key.Key_P), self)
        self._next_video_shortcut = QShortcut(QKeySequence(Qt.Key.Key_N), self)
        self._play_pause_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self._prev_shortcut.setEnabled(False)
        self._next_shortcut.setEnabled(False)
        self._undo_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._redo_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._redo_alt_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._clear_mask_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._prev_video_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._next_video_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._play_pause_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
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
        self._inference_dialog = None
        self._sequence_paths = []
        self._sequence_index = -1
        self._sequence_output_dir = None
        self._mode = "none"
        self._video_path = None
        self._video_paths = []
        self._video_list_index = -1
        self._video_capture = None
        self._video_frame_count = 0
        self._video_fps = 0.0
        self._video_frame_index = -1
        self._video_output_dir = None
        self._video_frame_cache = OrderedDict()
        self._video_decode_pos = -1
        self._video_cache_limit = 9
        self._video_use_random_seek = False
        self._default_nerves = [
            "ulnar",
            "median",
            "radial",
            "plex",
            "lfcn",
            "peroneal",
            "fibular",
            "tibial",
            "sural",
            "proximal",
            "accessory",
            "quad",
            "sciatic",
            "unknown",
        ]
        self._nerve_labels = list(self._default_nerves)
        self._video_nerve_map = {}
        self._video_nerve_updated_at = {}
        self._video_manifest_name = "nerve_manifest.json"
        self._classification_ui_updating = False
        self._nerve_button_group = None
        self._nerve_buttons = {}
        self._nerve_manifest_warning_shown = False
        self._last_image_input_dir = ""
        self._last_image_output_dir = ""
        self._last_video_input_dir = ""
        self._last_video_output_dir = ""
        self._load_persisted_paths()
        self._build_nerve_label_controls()
        self._update_nerve_summary_label()
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
        self.load_image_action = QAction("Load Image...", self)
        file_menu.addAction(self.load_image_action)
        self.load_video_action = QAction("Load Video...", self)
        file_menu.addAction(self.load_video_action)
        self.load_mask_action = QAction("Load Mask...", self)
        file_menu.addAction(self.load_mask_action)
        self.segment_action = QAction("Segment", self)
        file_menu.addAction(self.segment_action)
        self.segment_batch_action = QAction("Batch segment", self)
        file_menu.addAction(self.segment_batch_action)
        file_menu.addSeparator()
        self.save_mask_action = QAction("Save mask...", self)
        self.save_mask_action.setShortcut(QKeySequence.StandardKey.Save)
        file_menu.addAction(self.save_mask_action)
        self.save_video_masks_action = QAction("Save video masks...", self)
        file_menu.addAction(self.save_video_masks_action)
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

    def _settings(self):
        return QSettings("UltAI", "UltAI Viewer")

    def _runtime_base_dir(self):
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent
        return Path(__file__).resolve().parent.parent

    def _manifest_path(self):
        output_dir = (self._video_output_dir or "").strip()
        if not output_dir:
            return None
        return Path(output_dir) / self._video_manifest_name

    def _build_nerve_label_controls(self):
        if not hasattr(self, "nerve_labels_layout"):
            return
        while self.nerve_labels_layout.count():
            item = self.nerve_labels_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._nerve_buttons = {}
        self._nerve_button_group = QButtonGroup(self)
        self._nerve_button_group.setExclusive(True)
        for index, label in enumerate(self._nerve_labels):
            button = QRadioButton(label)
            button.toggled.connect(lambda checked, lbl=label: self._on_nerve_label_selected(lbl, checked))
            self._nerve_button_group.addButton(button)
            self._nerve_buttons[label] = button
            self.nerve_labels_layout.addWidget(button, index % 2, index // 2)
        has_video = self._mode == "video" and bool(self._video_path)
        self.nerve_labels_container.setEnabled(has_video)
        self.add_nerve_btn.setEnabled(self._mode == "video")

    def _labeled_video_count(self):
        if not self._video_paths:
            return 0
        return sum(1 for path in self._video_paths if self._video_nerve_map.get(path))

    def _update_nerve_summary_label(self):
        if not hasattr(self, "nerve_summary_label"):
            return
        total = len(self._video_paths)
        labeled = self._labeled_video_count()
        label_text = "-"
        if self._mode == "video" and self._video_path:
            label_text = self._video_nerve_map.get(self._video_path, "-")
        self.nerve_summary_label.setText(f"Label: {label_text} ({labeled}/{total} labeled)")

    def _set_current_video_label_ui(self, video_path):
        if not hasattr(self, "nerve_labels_container"):
            return
        self._classification_ui_updating = True
        try:
            selected = self._video_nerve_map.get(video_path)
            if self._nerve_button_group is not None:
                self._nerve_button_group.setExclusive(False)
            for label, button in self._nerve_buttons.items():
                button.setChecked(bool(selected and label == selected))
            if self._nerve_button_group is not None:
                self._nerve_button_group.setExclusive(True)
        finally:
            self._classification_ui_updating = False
        self._update_nerve_summary_label()

    def _infer_nerve_label_from_video_path(self, video_path):
        folder_name = Path(video_path).parent.name.strip().lower()
        if not folder_name:
            return None
        normalized = re.sub(r"[^a-z0-9]+", " ", folder_name)
        for label in sorted(self._nerve_labels, key=len, reverse=True):
            if re.search(rf"\b{re.escape(label.lower())}\b", normalized):
                return label
        return None

    def _ensure_current_video_nerve_label(self):
        if self._mode != "video" or not self._video_path:
            return None
        current = self._video_nerve_map.get(self._video_path)
        if current:
            return current
        inferred_label = self._infer_nerve_label_from_video_path(self._video_path)
        if not inferred_label:
            return None
        self._video_nerve_map[self._video_path] = inferred_label
        self._video_nerve_updated_at[self._video_path] = datetime.now(timezone.utc).isoformat()
        return inferred_label

    def _persist_current_video_nerve_label(self):
        if self._mode != "video" or not self._video_path:
            return
        label = self._video_nerve_map.get(self._video_path)
        if not label:
            return
        if self._video_path not in self._video_nerve_updated_at:
            self._video_nerve_updated_at[self._video_path] = datetime.now(timezone.utc).isoformat()
        self._save_nerve_manifest(show_errors=False)

    def _load_nerve_manifest(self):
        self._nerve_labels = list(self._default_nerves)
        self._video_nerve_map = {}
        self._video_nerve_updated_at = {}
        self._nerve_manifest_warning_shown = False
        manifest_path = self._manifest_path()
        if manifest_path is None or not manifest_path.exists():
            return
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            labels = data.get("label_set", [])
            if isinstance(labels, list):
                merged = list(self._default_nerves)
                seen = set(merged)
                for label in labels:
                    if not isinstance(label, str):
                        continue
                    clean = label.strip().lower()
                    if not clean or clean in seen:
                        continue
                    seen.add(clean)
                    merged.append(clean)
                self._nerve_labels = merged
            videos = data.get("videos", [])
            if isinstance(videos, list):
                for item in videos:
                    if not isinstance(item, dict):
                        continue
                    video_path = str(item.get("video_path", "")).strip()
                    label = str(item.get("nerve_label", "")).strip().lower()
                    if not video_path or not label:
                        continue
                    self._video_nerve_map[video_path] = label
                    updated = str(item.get("updated_at", "")).strip()
                    if updated:
                        self._video_nerve_updated_at[video_path] = updated
        except Exception as exc:
            self._video_nerve_map = {}
            self._video_nerve_updated_at = {}
            self._nerve_labels = list(self._default_nerves)
            if not self._nerve_manifest_warning_shown:
                QMessageBox.warning(self, "Manifest warning", f"Could not read nerve manifest: {exc}")
                self._nerve_manifest_warning_shown = True

    def _save_nerve_manifest(self, show_errors=True):
        manifest_path = self._manifest_path()
        if manifest_path is None:
            if show_errors:
                QMessageBox.information(self, "Missing output folder", "Select video output folder first.")
            return False
        existing_videos = {}
        existing_labels = []
        if manifest_path.exists():
            try:
                existing_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                raw_labels = existing_data.get("label_set", [])
                if isinstance(raw_labels, list):
                    existing_labels = [str(label).strip().lower() for label in raw_labels if str(label).strip()]
                raw_videos = existing_data.get("videos", [])
                if isinstance(raw_videos, list):
                    for item in raw_videos:
                        if not isinstance(item, dict):
                            continue
                        video_path = str(item.get("video_path", "")).strip()
                        if not video_path:
                            continue
                        existing_videos[video_path] = {
                            "video_path": video_path,
                            "video_name": str(item.get("video_name", Path(video_path).name)),
                            "nerve_label": str(item.get("nerve_label", "")).strip().lower(),
                            "updated_at": str(item.get("updated_at", "")).strip(),
                        }
            except Exception:
                # Fall back to rewriting a clean manifest from current in-memory state.
                existing_videos = {}
                existing_labels = []
        for video_path in self._video_paths:
            label = self._video_nerve_map.get(video_path)
            if not label:
                continue
            existing_videos[str(video_path)] = {
                "video_path": str(video_path),
                "video_name": Path(video_path).name,
                "nerve_label": str(label),
                "updated_at": self._video_nerve_updated_at.get(video_path, ""),
            }
        merged_labels = list(self._default_nerves)
        seen_labels = set(merged_labels)
        for label in list(existing_labels) + list(self._nerve_labels):
            clean = str(label).strip().lower()
            if not clean or clean in seen_labels:
                continue
            seen_labels.add(clean)
            merged_labels.append(clean)
        videos = sorted(
            (
                item
                for item in existing_videos.values()
                if item.get("video_path") and item.get("nerve_label")
            ),
            key=lambda item: str(item["video_path"]).lower(),
        )
        payload = {
            "version": 1,
            "label_set": merged_labels,
            "video_count": len(videos),
            "videos": videos,
        }
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return True
        except Exception as exc:
            if show_errors:
                QMessageBox.warning(self, "Manifest save error", str(exc))
            return False

    def _on_nerve_label_selected(self, label, checked):
        if not checked or self._classification_ui_updating:
            return
        if self._mode != "video" or not self._video_path:
            return
        clean = str(label).strip().lower()
        if not clean:
            return
        self._video_nerve_map[self._video_path] = clean
        self._video_nerve_updated_at[self._video_path] = datetime.now(timezone.utc).isoformat()
        self._update_nerve_summary_label()
        if self._save_nerve_manifest(show_errors=True):
            self.statusBar().showMessage(f"Saved label '{clean}' for {Path(self._video_path).name}")

    def _add_custom_nerve_label(self):
        text, ok = QInputDialog.getText(self, "Add nerve", "Nerve name:")
        if not ok:
            return
        label = str(text).strip().lower()
        if not label:
            QMessageBox.information(self, "Invalid label", "Nerve name cannot be empty.")
            return
        existing = {lbl.lower() for lbl in self._nerve_labels}
        if label in existing:
            QMessageBox.information(self, "Duplicate label", "That nerve label already exists.")
            return
        self._nerve_labels.append(label)
        self._build_nerve_label_controls()
        if self._mode == "video" and self._video_path:
            self._set_current_video_label_ui(self._video_path)
        self._save_nerve_manifest(show_errors=True)

    def _load_persisted_paths(self):
        settings = self._settings()
        self._last_image_input_dir = str(settings.value("paths/image_input_dir", "", str) or "")
        self._last_image_output_dir = str(settings.value("paths/image_output_dir", "", str) or "")
        self._last_video_input_dir = str(settings.value("paths/video_input_dir", "", str) or "")
        self._last_video_output_dir = str(settings.value("paths/video_output_dir", "", str) or "")

    def _save_persisted_paths(self):
        settings = self._settings()
        settings.setValue("paths/image_input_dir", self._last_image_input_dir)
        settings.setValue("paths/image_output_dir", self._last_image_output_dir)
        settings.setValue("paths/video_input_dir", self._last_video_input_dir)
        settings.setValue("paths/video_output_dir", self._last_video_output_dir)

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

        layout.addWidget(QLabel("Images:"))
        load_row = QHBoxLayout()
        self.open_image_btn = QPushButton("Load image(s)")
        configure_button(self.open_image_btn)
        load_row.addWidget(self.open_image_btn)
        layout.addLayout(load_row)

        mask_row = QHBoxLayout()
        self.open_mask_btn = QPushButton("Load mask(s)")
        configure_button(self.open_mask_btn)
        mask_row.addWidget(self.open_mask_btn)
        self.clear_sequence_btn = QPushButton("Clear image(s)")
        configure_button(self.clear_sequence_btn)
        mask_row.addWidget(self.clear_sequence_btn)
        layout.addLayout(mask_row)

        self.sequence_combo = QComboBox()
        self.sequence_combo.setEnabled(False)
        layout.addWidget(self.sequence_combo)
        image_nav_row = QHBoxLayout()
        self.image_prev_btn = QPushButton("Prev")
        self.image_prev_btn.setEnabled(False)
        configure_button(self.image_prev_btn)
        image_nav_row.addWidget(self.image_prev_btn)
        self.image_next_btn = QPushButton("Next")
        self.image_next_btn.setEnabled(False)
        configure_button(self.image_next_btn)
        image_nav_row.addWidget(self.image_next_btn)
        layout.addLayout(image_nav_row)
        self.save_btn = QPushButton("Save masks")
        configure_button(self.save_btn)
        layout.addWidget(self.save_btn)

        layout.addSpacing(16)
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep2)

        layout.addWidget(QLabel("Videos:"))
        self.video_btn = QPushButton("Load video(s)")
        configure_button(self.video_btn)
        layout.addWidget(self.video_btn)
        self.clear_video_btn = QPushButton("Clear video(s)")
        configure_button(self.clear_video_btn)
        layout.addWidget(self.clear_video_btn)
        self.video_combo = QComboBox()
        self.video_combo.setEnabled(False)
        layout.addWidget(self.video_combo)

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
        self.save_video_btn = QPushButton("Save masks")
        configure_button(self.save_video_btn)
        layout.addWidget(self.save_video_btn)

        layout.addSpacing(16)
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep3)

        layout.addWidget(QLabel("Models:"))
        model_device_row = QHBoxLayout()
        model_device_row.addWidget(QLabel("Model:"))
        self.model_picker = QComboBox()
        self.model_picker.addItem("No models loaded")
        self.model_picker.setEnabled(False)
        model_device_row.addWidget(self.model_picker)
        model_device_row.addSpacing(6)
        model_device_row.addWidget(QLabel("Device:"))
        self.device_picker = QComboBox()
        self.device_picker.setEnabled(False)
        model_device_row.addWidget(self.device_picker)
        layout.addLayout(model_device_row)

        layout.addSpacing(3)

        segment_row = QHBoxLayout()
        self.run_btn = QPushButton("Segment")
        configure_button(self.run_btn)
        segment_row.addWidget(self.run_btn)
        self.run_batch_btn = QPushButton("Batch segment")
        configure_button(self.run_batch_btn)
        segment_row.addWidget(self.run_batch_btn)
        layout.addLayout(segment_row)

        layout.addSpacing(16)
        sep4 = QFrame()
        sep4.setFrameShape(QFrame.Shape.HLine)
        sep4.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep4)

        layout.addWidget(QLabel("Edit:"))
        tools_row = QHBoxLayout()
        tools_row.addWidget(QLabel("Tools:"))
        self.tool_picker = QComboBox()
        self.tool_picker.addItems(
            ["Select", "Freehand Line", "Segmented Line", "Paint Brush", "Eraser"]
        )
        self.tool_picker.setCurrentIndex(1)
        tools_row.addWidget(self.tool_picker)
        self.fill_roi_checkbox = QCheckBox("Fill ROI")
        tools_row.addWidget(self.fill_roi_checkbox)
        self.fill_roi_checkbox.setChecked(False)
        layout.addLayout(tools_row)

        self.show_mask_checkbox = QCheckBox("Toggle mask")
        self.show_mask_checkbox.setChecked(True)
        layout.addWidget(self.show_mask_checkbox)

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

        self.fit_btn = QPushButton("Fit to Window")
        configure_button(self.fit_btn)
        layout.addWidget(self.fit_btn)

        layout.addSpacing(5)

        undo_redo_row = QHBoxLayout()
        self.undo_btn = QPushButton("Undo")
        configure_button(self.undo_btn)
        undo_redo_row.addWidget(self.undo_btn)
        self.redo_btn = QPushButton("Redo")
        configure_button(self.redo_btn)
        undo_redo_row.addWidget(self.redo_btn)
        layout.addLayout(undo_redo_row)

        clear_row = QHBoxLayout()
        self.clear_btn = QPushButton("Clear Mask")
        configure_button(self.clear_btn)
        clear_row.addWidget(self.clear_btn)
        self.close_btn = QPushButton("Close Image")
        configure_button(self.close_btn)
        clear_row.addWidget(self.close_btn)
        layout.addLayout(clear_row)

        layout.addSpacing(6)
        sep_end = QFrame()
        sep_end.setFrameShape(QFrame.Shape.HLine)
        sep_end.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep_end)

        layout.addStretch()

        return panel

    def _wire_actions(self):
        self.open_image_btn.clicked.connect(self._load_single_image)
        self.load_image_action.triggered.connect(self._load_single_image)
        self.video_btn.clicked.connect(self._load_video)
        self.load_video_action.triggered.connect(self._load_video)
        self.open_mask_btn.clicked.connect(self.canvas.load_mask_dialog)
        self.load_mask_action.triggered.connect(self.canvas.load_mask_dialog)
        self.exit_action.triggered.connect(self.close)
        self.run_btn.clicked.connect(self._run_inference)
        self.segment_action.triggered.connect(self._run_inference)
        self.run_batch_btn.clicked.connect(self._run_batch_inference)
        self.segment_batch_action.triggered.connect(self._run_batch_inference)
        self.save_btn.clicked.connect(self._save_current_mask)
        self.save_mask_action.triggered.connect(self._save_current_mask)
        self.save_video_btn.clicked.connect(self._save_video_masks)
        self.save_video_masks_action.triggered.connect(self._save_video_masks)
        self.clear_btn.clicked.connect(self.canvas.clear_mask)
        self.clear_mask_action.triggered.connect(self.canvas.clear_mask)
        self._clear_mask_shortcut.activated.connect(self.canvas.clear_mask)
        self.close_btn.clicked.connect(self._close_current_image)
        self.close_image_action.triggered.connect(self._close_current_image)
        self.tool_picker.currentIndexChanged.connect(self._on_tool_changed)
        self.brush_radius.valueChanged.connect(self._on_brush_radius_changed)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self.fit_btn.clicked.connect(self.canvas.fit_to_window)
        self.fill_roi_checkbox.toggled.connect(self.canvas.set_fill_roi)
        self.show_mask_checkbox.toggled.connect(self.canvas.set_mask_visible)
        self.undo_btn.clicked.connect(self.canvas.undo)
        self.redo_btn.clicked.connect(self.canvas.redo)
        self.undo_action.triggered.connect(self.canvas.undo)
        self.redo_action.triggered.connect(self.canvas.redo)
        self._undo_shortcut.activated.connect(self.canvas.undo)
        self._redo_shortcut.activated.connect(self.canvas.redo)
        self._redo_alt_shortcut.activated.connect(self.canvas.redo)
        self._prev_video_shortcut.activated.connect(self._show_previous_sequence)
        self._next_video_shortcut.activated.connect(self._show_next_sequence)
        self._play_pause_shortcut.activated.connect(self._toggle_playback)
        self.add_nerve_btn.clicked.connect(self._add_custom_nerve_label)
        self.model_picker.currentIndexChanged.connect(self._on_model_changed)
        self.device_picker.currentIndexChanged.connect(self._on_device_changed)
        self.clear_sequence_btn.clicked.connect(self._clear_sequence)
        self.sequence_combo.currentIndexChanged.connect(self._on_sequence_selected)
        self.image_prev_btn.clicked.connect(self._show_previous_sequence)
        self.image_next_btn.clicked.connect(self._show_next_sequence)
        self.clear_video_btn.clicked.connect(self._clear_video_sequence)
        self.video_combo.currentIndexChanged.connect(self._on_video_selected)
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
                if event.modifiers() != Qt.KeyboardModifier.NoModifier or self._mode != "video":
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

    def _load_single_image(self):
        image_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load image(s)",
            self._last_image_input_dir,
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not image_paths:
            return
        self._last_image_input_dir = str(Path(image_paths[0]).parent)
        self._save_persisted_paths()
        self._stop_playback()
        if self._mode == "video":
            self._stash_video_mask_for_current_frame()
            self._clear_video_state()
        self._mode = "sequence"
        self._sequence_paths = list(image_paths)
        self._sequence_index = 0
        self._sequence_output_dir = None
        self.sequence_combo.setEnabled(True)
        self.sequence_combo.clear()
        self.sequence_combo.addItems([Path(p).name for p in self._sequence_paths])
        self._set_slider_state(0, len(self._sequence_paths) - 1, enabled=bool(self._sequence_paths))
        self._set_sequence_index(0)
        self.statusBar().showMessage(f"Loaded {len(self._sequence_paths)} image(s)")

    def _close_current_image(self):
        self._stop_playback()
        self._stash_video_mask_for_current_frame()
        self._clear_video_state()
        self._clear_sequence_state(clear_canvas=False)
        self._mode = "none"
        self._set_slider_state(0, 0, enabled=False)
        self.canvas.clear_image()

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
        self._sidebar_width = min(260, max(180, int(screen_rect.width() * 0.22)))
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
        if self._mode == "video":
            QMessageBox.information(
                self,
                "Video mode",
                "Batch segmentation is currently available for image sequences only.",
            )
            return
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
        if self._sequence_paths and self._sequence_index >= 0:
            self._load_sequence_image()

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
        self.statusBar().showMessage("Ready")

    def _load_sequence(self):
        dialog_result = self._show_sequence_dialog()
        if dialog_result is None:
            return
        self._stop_playback()
        if self._mode == "video":
            self._stash_video_mask_for_current_frame()
            self._clear_video_state()
        paths, output_dir = dialog_result
        self._mode = "sequence"
        self._sequence_paths = list(paths)
        self._sequence_index = 0
        self._sequence_output_dir = output_dir
        self.sequence_combo.setEnabled(True)
        self.sequence_combo.clear()
        self.sequence_combo.addItems([Path(p).name for p in self._sequence_paths])
        self._set_slider_state(0, len(self._sequence_paths) - 1, enabled=bool(self._sequence_paths))
        self._set_sequence_index(0)
        self.statusBar().showMessage(f"Sequence loaded: {len(self._sequence_paths)} images")

    def _load_video(self):
        dialog_result = self._show_video_dialog()
        if dialog_result is None:
            return
        video_paths, output_dir = dialog_result
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
        self._video_manifest_name = "nerve_manifest.json"
        self._save_persisted_paths()
        self._mode = "video"
        self._load_nerve_manifest()
        allowed_paths = set(self._video_paths)
        self._video_nerve_map = {
            path: label for path, label in self._video_nerve_map.items() if path in allowed_paths
        }
        self._video_nerve_updated_at = {
            path: ts for path, ts in self._video_nerve_updated_at.items() if path in allowed_paths
        }
        self.video_combo.setEnabled(True)
        self.video_combo.clear()
        self.video_combo.addItems([Path(p).name for p in self._video_paths])
        self._build_nerve_label_controls()
        self._update_nerve_summary_label()
        resume_video_index, resume_frame_index = self._find_resume_video_position()
        self._open_video_at_index(resume_video_index, start_frame=resume_frame_index)

    def _clear_sequence_state(self, clear_canvas):
        self._sequence_paths = []
        self._sequence_index = -1
        self._sequence_output_dir = None
        self.sequence_combo.clear()
        self.sequence_combo.setEnabled(False)
        if clear_canvas:
            self.canvas.clear_image()

    def _clear_video_state(self):
        self._stop_playback()
        if self._video_capture is not None:
            self._video_capture.release()
        self.video_combo.clear()
        self.video_combo.setEnabled(False)
        self._video_paths = []
        self._video_list_index = -1
        self._video_path = None
        self._video_capture = None
        self._video_frame_count = 0
        self._video_fps = 0.0
        self._video_frame_index = -1
        self._video_output_dir = None
        self._video_frame_cache.clear()
        self._video_decode_pos = -1
        self._video_use_random_seek = False
        self._video_nerve_map = {}
        self._video_nerve_updated_at = {}
        self._nerve_labels = list(self._default_nerves)
        self._build_nerve_label_controls()
        self._update_nerve_summary_label()

    def _clear_video_sequence(self):
        if self._mode != "video":
            return
        self._stop_playback()
        self._stash_video_mask_for_current_frame()
        self._clear_video_state()
        self._mode = "none"
        self._set_slider_state(0, 0, enabled=False)
        self.image_prev_btn.setEnabled(False)
        self.image_next_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.canvas.clear_image()
        self.statusBar().showMessage("Video sequence cleared")

    def _video_output_root_for_path(self, video_path):
        output_dir = (self._video_output_dir or "").strip()
        if not output_dir:
            return None
        video_path = Path(video_path)
        return Path(output_dir) / video_path.stem

    def _video_mask_path(self, video_path, frame_index):
        root = self._video_output_root_for_path(video_path)
        if root is None:
            return None
        return root / f"frame_{int(frame_index):06d}.png"

    def _last_annotated_frame_for_video(self, video_path):
        video_output_root = self._video_output_root_for_path(video_path)
        if video_output_root is None or not video_output_root.exists():
            return -1
        last_frame = -1
        for mask_path in sorted(video_output_root.glob("frame_*.png")):
            stem = mask_path.stem
            if not stem.startswith("frame_"):
                continue
            try:
                frame_index = int(stem.split("_")[-1])
            except ValueError:
                continue
            if frame_index > last_frame:
                last_frame = frame_index
        return last_frame

    def _find_resume_video_position(self):
        if not self._video_paths:
            return 0, 0
        last_video_index = -1
        last_frame_index = -1
        for video_index, video_path in enumerate(self._video_paths):
            frame_index = self._last_annotated_frame_for_video(video_path)
            if frame_index >= 0:
                last_video_index = video_index
                last_frame_index = frame_index
        if last_video_index < 0:
            return 0, 0
        return last_video_index, last_frame_index

    def _load_saved_video_mask_for_frame(self, video_path, frame_index):
        mask_path = self._video_mask_path(video_path, frame_index)
        if mask_path is None or not mask_path.exists():
            return None
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            return None
        return (mask_gray >= 128).astype(np.float32)

    def _propagate_mask_with_optical_flow(self, src_frame, dst_frame, src_mask):
        if src_frame is None or dst_frame is None or src_mask is None:
            return None
        src_gray = cv2.cvtColor(np.asarray(src_frame), cv2.COLOR_RGB2GRAY)
        dst_gray = cv2.cvtColor(np.asarray(dst_frame), cv2.COLOR_RGB2GRAY)
        mask = (np.asarray(src_mask, dtype=np.float32) >= 0.5).astype(np.uint8)
        if mask.ndim != 2 or not np.any(mask):
            return None

        flow = cv2.calcOpticalFlowFarneback(
            src_gray,
            dst_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        height, width = src_gray.shape
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        map_x = grid_x - flow[:, :, 0]
        map_y = grid_y - flow[:, :, 1]
        warped = cv2.remap(
            mask.astype(np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        binary = (warped >= 0.5).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        if not np.any(binary):
            return None
        return binary.astype(np.float32)

    def _maybe_propagate_mask_to_next_frame(self, src_index, dst_index, src_frame, dst_frame):
        if self._mode != "video":
            return None
        if src_index < 0 or dst_index < 0:
            return None
        if dst_index != src_index + 1:
            return None
        if self._load_saved_video_mask_for_frame(self._video_path, dst_index) is not None:
            return None
        self._commit_pending_outline()
        if not self._canvas_has_roi():
            return None
        try:
            propagated = self._propagate_mask_with_optical_flow(src_frame, dst_frame, self.canvas.mask)
        except Exception:
            return None
        return propagated

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
        self.video_combo.blockSignals(True)
        self.video_combo.setCurrentIndex(video_index)
        self.video_combo.blockSignals(False)
        self._ensure_current_video_nerve_label()
        self._set_slider_state(0, frame_count - 1, enabled=frame_count > 0)
        target_frame = int(start_frame)
        if target_frame < 0:
            target_frame = frame_count - 1
        target_frame = min(frame_count - 1, max(0, target_frame))
        self._set_video_frame_index(target_frame, force=True)
        self._set_current_video_label_ui(self._video_path)
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
        self.image_prev_btn.setEnabled(False)
        self.image_next_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.canvas.clear_image()
        self.statusBar().showMessage("Sequence/video cleared")

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
        self._persist_current_video_nerve_label()
        self._stash_video_mask_for_current_frame()
        self._open_video_at_index(index, start_frame=0)

    def _show_previous_sequence(self):
        if self._mode == "video":
            if self._video_list_index <= 0:
                return
            self._persist_current_video_nerve_label()
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
            self._persist_current_video_nerve_label()
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
        if self._mode != "video":
            return
        if self._video_frame_index <= 0:
            return
        self._set_video_frame_index(self._video_frame_index - 1)

    def _show_next_frame(self):
        if self._mode != "video":
            return
        if self._video_frame_index < 0:
            return
        if self._video_frame_index >= self._video_frame_count - 1:
            return
        self._set_video_frame_index(self._video_frame_index + 1)

    def _show_first_frame(self):
        if self._mode != "video":
            return
        if self._video_frame_index <= 0:
            return
        self._set_video_frame_index(0)

    def _show_last_frame(self):
        if self._mode != "video":
            return
        if self._video_frame_count <= 0:
            return
        last_index = self._video_frame_count - 1
        if self._video_frame_index >= last_index:
            return
        self._set_video_frame_index(last_index)

    def _load_sequence_image(self):
        if self._mode != "sequence":
            return
        if self._sequence_index < 0 or self._sequence_index >= len(self._sequence_paths):
            return
        path = self._sequence_paths[self._sequence_index]
        self.canvas.load_image(path)
        mask_path = self._find_sequence_mask_path(path)
        if mask_path:
            self.canvas.load_mask(mask_path)
        else:
            self.canvas.clear_mask()
        self._update_navigation_buttons()
        self._set_slider_value(self._sequence_index)

    def _set_sequence_index(self, index):
        if index < 0 or index >= len(self._sequence_paths):
            return
        self._sequence_index = index
        self.sequence_combo.blockSignals(True)
        self.sequence_combo.setCurrentIndex(index)
        self.sequence_combo.blockSignals(False)
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
        previous_index = self._video_frame_index
        source_frame = None if self.canvas.image is None else np.copy(self.canvas.image)
        self._stash_video_mask_for_current_frame()
        frame = self._decode_video_frame(frame_index)
        if frame is None:
            self._stop_playback()
            QMessageBox.warning(self, "Frame error", f"Could not decode frame {frame_index}.")
            return
        propagated_mask = None
        if source_frame is not None:
            propagated_mask = self._maybe_propagate_mask_to_next_frame(
                previous_index,
                frame_index,
                source_frame,
                frame,
            )
        self._video_frame_index = frame_index
        self.canvas.load_image_array(frame, self._video_path or "")
        cached_mask = self._load_saved_video_mask_for_frame(self._video_path, frame_index)
        if cached_mask is not None:
            self.canvas.set_mask(np.copy(cached_mask))
        elif propagated_mask is not None:
            self.canvas.set_mask(propagated_mask)
            self.statusBar().showMessage(f"Propagated mask to frame {frame_index + 1}")
        else:
            self.canvas.clear_mask()
        self._set_slider_value(frame_index)
        self._update_navigation_buttons()
        self._update_title_with_image(self._video_path or "")

    def _canvas_has_roi(self):
        if self.canvas.mask is None:
            return False
        return bool(np.any(np.asarray(self.canvas.mask) >= 0.5))

    def _commit_pending_outline(self):
        if self.canvas is None:
            return
        self.canvas.commit_pending_outline_to_mask()

    def _stash_video_mask_for_current_frame(self):
        if self._mode != "video":
            return
        if self._video_frame_index < 0 or not self._video_path:
            return
        self._commit_pending_outline()
        mask_path = self._video_mask_path(self._video_path, self._video_frame_index)
        if mask_path is None:
            return
        if self._canvas_has_roi():
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask = np.asarray(self.canvas.mask, dtype=np.float32)
            mask_uint8 = (mask >= 0.5).astype(np.uint8) * 255
            if not cv2.imwrite(str(mask_path), mask_uint8):
                QMessageBox.warning(self, "Save error", f"Failed to save mask: {mask_path.name}")
            return
        if mask_path.exists():
            try:
                mask_path.unlink()
            except OSError as exc:
                QMessageBox.warning(self, "Save error", str(exc))

    def _update_navigation_buttons(self):
        if self._mode == "video":
            can_prev = self._video_list_index > 0
            can_next = 0 <= self._video_list_index < len(self._video_paths) - 1
            can_prev_frame = self._video_frame_index > 0
            can_next_frame = (
                self._video_frame_index >= 0 and self._video_frame_index < self._video_frame_count - 1
            )
            self.image_prev_btn.setEnabled(False)
            self.image_next_btn.setEnabled(False)
            self.prev_btn.setEnabled(can_prev)
            self.next_btn.setEnabled(can_next)
            self.play_btn.setEnabled(self._video_frame_count > 1)
            self.frame_first_btn.setEnabled(can_prev_frame)
            self.frame_prev_btn.setEnabled(can_prev_frame)
            self.frame_next_btn.setEnabled(can_next_frame)
            self.frame_last_btn.setEnabled(can_next_frame)
            self.nerve_labels_container.setEnabled(True)
            self.add_nerve_btn.setEnabled(True)
            self._update_nerve_summary_label()
            return
        if self._mode == "sequence":
            self.image_prev_btn.setEnabled(self._sequence_index > 0)
            self.image_next_btn.setEnabled(self._sequence_index < len(self._sequence_paths) - 1)
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.play_btn.setEnabled(False)
            self.frame_first_btn.setEnabled(False)
            self.frame_prev_btn.setEnabled(False)
            self.frame_next_btn.setEnabled(False)
            self.frame_last_btn.setEnabled(False)
            self.nerve_labels_container.setEnabled(False)
            self.add_nerve_btn.setEnabled(False)
            self._update_nerve_summary_label()
            return
        self.image_prev_btn.setEnabled(False)
        self.image_next_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.frame_first_btn.setEnabled(False)
        self.frame_prev_btn.setEnabled(False)
        self.frame_next_btn.setEnabled(False)
        self.frame_last_btn.setEnabled(False)
        self.nerve_labels_container.setEnabled(False)
        self.add_nerve_btn.setEnabled(False)
        self._update_nerve_summary_label()

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
            self._stop_playback()
            return
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
        if self._mode != "sequence":
            return
        if not self._sequence_output_dir:
            return
        if self._sequence_index < 0 or self._sequence_index >= len(self._sequence_paths):
            return
        self._commit_pending_outline()
        if not self._canvas_has_roi():
            return
        image_path = Path(self._sequence_paths[self._sequence_index])
        output_path = Path(self._sequence_output_dir) / f"{image_path.stem}.png"
        try:
            self.canvas.save_mask(str(output_path))
            self.statusBar().showMessage(f"Saved mask: {output_path.name}")
        except Exception as exc:
            QMessageBox.warning(self, "Save error", str(exc))

    def _save_current_mask(self):
        if self._mode == "video":
            self._save_video_masks()
            return
        self._commit_pending_outline()
        if not self._canvas_has_roi():
            QMessageBox.information(self, "No mask", "Run segmentation or annotate before saving a mask.")
            return
        if self._sequence_output_dir and self._sequence_index >= 0:
            image_path = Path(self._sequence_paths[self._sequence_index])
            output_path = Path(self._sequence_output_dir) / f"{image_path.stem}.png"
            try:
                self.canvas.save_mask(str(output_path))
                self.statusBar().showMessage(f"Saved mask: {output_path.name}")
            except Exception as exc:
                QMessageBox.warning(self, "Save error", str(exc))
            return
        self.canvas.save_mask_dialog()

    def _save_video_frame_mask_dialog(self):
        if self._mode != "video":
            QMessageBox.information(self, "No video", "Load a video first.")
            return
        self._commit_pending_outline()
        if not self._canvas_has_roi():
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
        self._stash_video_mask_for_current_frame()
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
        video_output_root = Path(output_dir) / Path(self._video_path).stem
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
        input_line.setReadOnly(True)

        selected_paths = []

        def select_folder():
            directory = QFileDialog.getExistingDirectory(
                dialog, "Select input folder", self._last_image_input_dir
            )
            if directory:
                paths = sorted(
                    str(path)
                    for path in Path(directory).iterdir()
                    if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
                )
                if paths:
                    selected_paths[:] = paths
                    input_line.setText(f"{directory} ({len(paths)} images)")
                    self._last_image_input_dir = directory
                    self._save_persisted_paths()
                else:
                    QMessageBox.information(
                        dialog, "No images", "No images found in the selected folder."
                    )

        def select_files():
            files, _ = QFileDialog.getOpenFileNames(
                dialog,
                "Select images",
                self._last_image_input_dir,
                "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
            )
            if files:
                selected_paths[:] = list(files)
                input_line.setText(f"{len(files)} images selected")
                self._last_image_input_dir = str(Path(files[0]).parent)
                self._save_persisted_paths()

        def browse_output():
            directory = QFileDialog.getExistingDirectory(
                dialog,
                "Select output folder",
                self._last_image_output_dir or self._last_image_input_dir,
            )
            if directory:
                output_line.setText(directory)
                self._last_image_output_dir = directory
                self._save_persisted_paths()

        input_row = QHBoxLayout()
        folder_btn = QPushButton("Select folder")
        files_btn = QPushButton("Select images")
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
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        if not selected_paths:
            QMessageBox.information(self, "Missing fields", "Select input images or a folder.")
            return None
        output_dir = output_line.text().strip()
        if not output_dir:
            QMessageBox.information(self, "Missing fields", "Select an output folder.")
            return None
        self._last_image_output_dir = output_dir
        if selected_paths:
            self._last_image_input_dir = str(Path(selected_paths[0]).parent)
        self._save_persisted_paths()
        return selected_paths, output_dir

    def _show_video_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Load video sequence")
        layout = QFormLayout(dialog)

        input_line = QLineEdit(dialog)
        output_line = QLineEdit(dialog)
        base_dir = self._runtime_base_dir()
        output_line.setText(str(base_dir / "annotations"))
        input_line.setReadOnly(True)

        selected_paths = []

        def select_folder():
            directory = QFileDialog.getExistingDirectory(
                dialog, "Select input folder", self._last_video_input_dir
            )
            if directory:
                paths = sorted(
                    str(path)
                    for path in Path(directory).iterdir()
                    if path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv")
                )
                if paths:
                    selected_paths[:] = paths
                    input_line.setText(f"{directory} ({len(paths)} videos)")
                    self._last_video_input_dir = directory
                    self._save_persisted_paths()
                else:
                    QMessageBox.information(
                        dialog, "No videos", "No videos found in the selected folder."
                    )

        def select_files():
            files, _ = QFileDialog.getOpenFileNames(
                dialog,
                "Select video(s)",
                self._last_video_input_dir,
                "Video Files (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv)",
            )
            if files:
                selected_paths[:] = list(files)
                input_line.setText(f"{len(files)} videos selected")
                self._last_video_input_dir = str(Path(files[0]).parent)
                self._save_persisted_paths()

        def browse_output():
            directory = QFileDialog.getExistingDirectory(
                dialog,
                "Select output folder",
                self._video_output_dir or self._last_video_output_dir or self._last_video_input_dir,
            )
            if directory:
                output_line.setText(directory)
                self._last_video_output_dir = directory
                self._save_persisted_paths()

        input_row = QHBoxLayout()
        folder_btn = QPushButton("Select folder")
        files_btn = QPushButton("Select videos")
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
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        if not selected_paths:
            QMessageBox.information(self, "Missing fields", "Select input videos or a folder.")
            return None
        output_dir = output_line.text().strip()
        if not output_dir:
            QMessageBox.information(self, "Missing fields", "Select an output folder.")
            return None
        self._last_video_output_dir = output_dir
        if selected_paths:
            self._last_video_input_dir = str(Path(selected_paths[0]).parent)
        self._save_persisted_paths()
        return selected_paths, output_dir

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
        try:
            self._model.set_model(name)
            self.statusBar().showMessage(f"Model selected: {name}")
        except Exception as exc:
            QMessageBox.warning(self, "Model error", str(exc))

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
        self.statusBar().showMessage("Ready")

    def closeEvent(self, event):
        self._stash_video_mask_for_current_frame()
        self._clear_video_state()
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
