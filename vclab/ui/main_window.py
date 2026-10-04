"""PySide6 desktop application for Visual Continuity Lab.

The widgets in this module are intentionally self-contained.  They consume the
small adapter contract from :mod:`vclab.ui.adapter` and do not reach into core
engine implementation details.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageChops, ImageEnhance, ImageOps

from .adapter import AnalysisAdapter, FrameRecord, result_to_json

try:  # Importing this module should fail cleanly when Qt is not installed.
    from PySide6.QtCore import QThread, QTimer, Qt, Signal, QRectF, QPointF
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSlider,
        QSplitter,
        QStackedWidget,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError:  # pragma: no cover - exercised only on minimal systems
    QApplication = None  # type: ignore[assignment]


if QApplication is not None:

    def _qimage_from_pil(image: Image.Image) -> QImage:
        image = image.convert("RGBA")
        data = image.tobytes("raw", "RGBA")
        qimage = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
        return qimage.copy()


    def _pixmap_for_path(path: str, max_w: int = 900, max_h: int = 560) -> QPixmap:
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                return QPixmap.fromImage(_qimage_from_pil(image))
        except Exception:
            return QPixmap()


    class ImageCanvas(QWidget):
        """Image preview with normalized rectangle ROI drawing."""

        roi_changed = Signal(object)

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setMinimumSize(320, 220)
            self.setMouseTracking(True)
            self._pixmap = QPixmap()
            self._roi: tuple[float, float, float, float] | None = None
            self.editable = False
            self.locked = False
            self._drag_start: QPointF | None = None
            self._drag_current: QPointF | None = None
            self.setStyleSheet("background:#111820;border:1px solid #304050;border-radius:5px;")

        def set_image(self, path: str | None) -> None:
            self._pixmap = _pixmap_for_path(path) if path else QPixmap()
            self.update()

        def set_roi(self, roi: tuple[float, float, float, float] | None) -> None:
            self._roi = roi
            self.update()

        def roi(self) -> tuple[float, float, float, float] | None:
            return self._roi

        def image_rect(self) -> QRectF:
            if self._pixmap.isNull():
                return QRectF()
            scaled = self._pixmap.size()
            scaled.scale(self.size(), Qt.KeepAspectRatio)
            x = (self.width() - scaled.width()) / 2
            y = (self.height() - scaled.height()) / 2
            return QRectF(x, y, scaled.width(), scaled.height())

        def _to_normalized(self, point: QPointF) -> tuple[float, float]:
            rect = self.image_rect()
            if rect.width() <= 0 or rect.height() <= 0:
                return (0.0, 0.0)
            x = max(0.0, min(1.0, (point.x() - rect.x()) / rect.width()))
            y = max(0.0, min(1.0, (point.y() - rect.y()) / rect.height()))
            return (x, y)

        def _roi_rect(self, roi: tuple[float, float, float, float]) -> QRectF:
            rect = self.image_rect()
            return QRectF(rect.x() + roi[0] * rect.width(), rect.y() + roi[1] * rect.height(), roi[2] * rect.width(), roi[3] * rect.height())

        def paintEvent(self, _event: Any) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("#111820"))
            if not self._pixmap.isNull():
                painter.drawPixmap(self.image_rect().toRect(), self._pixmap)
            roi = self._roi
            if self._drag_start is not None and self._drag_current is not None:
                x0, y0 = self._to_normalized(self._drag_start)
                x1, y1 = self._to_normalized(self._drag_current)
                roi = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            if roi and roi[2] > 0 and roi[3] > 0:
                painter.setPen(QPen(QColor("#55d6be"), 2, Qt.SolidLine))
                painter.fillRect(self._roi_rect(roi), QColor(80, 214, 190, 35))
                painter.drawRect(self._roi_rect(roi))
            painter.end()

        def mousePressEvent(self, event: Any) -> None:  # noqa: N802
            if self.editable and not self.locked and event.button() == Qt.LeftButton:
                self._drag_start = event.position()
                self._drag_current = event.position()
                self.update()
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
            if self._drag_start is not None:
                self._drag_current = event.position()
                self.update()
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
            if self._drag_start is not None and event.button() == Qt.LeftButton:
                self._drag_current = event.position()
                x0, y0 = self._to_normalized(self._drag_start)
                x1, y1 = self._to_normalized(self._drag_current)
                roi = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
                if roi[2] >= 0.01 and roi[3] >= 0.01:
                    self._roi = roi
                    self.roi_changed.emit(roi)
                self._drag_start = None
                self._drag_current = None
                self.update()
            super().mouseReleaseEvent(event)


    class CompareViewer(QWidget):
        """Reference/current viewer with six comparison modes."""

        MODES = ("Side by Side", "Overlay", "Difference", "Slider", "Blink", "Onion Skin")

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.records: list[FrameRecord] = []
            self.master_index = 0
            self.current_index = 0
            self.mode = "Side by Side"
            self._blink_on = False
            self._timer = QTimer(self)
            self._timer.setInterval(450)
            self._timer.timeout.connect(self._toggle_blink)

            self.left = QLabel("No frame selected")
            self.right = QLabel("No master reference")
            for label in (self.left, self.right):
                label.setAlignment(Qt.AlignCenter)
                label.setMinimumSize(220, 150)
                label.setStyleSheet("background:#111820;border:1px solid #304050;border-radius:4px;")
                label.setScaledContents(False)
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(0, 100)
            self.slider.setValue(50)
            self.slider.valueChanged.connect(self._render)
            self.slider.setVisible(False)
            layout = QVBoxLayout(self)
            panes = QHBoxLayout()
            panes.setSpacing(8)
            panes.addWidget(self.left, 1)
            panes.addWidget(self.right, 1)
            layout.addLayout(panes, 1)
            layout.addWidget(self.slider)

        def set_records(self, records: Sequence[FrameRecord]) -> None:
            self.records = list(records)
            self._render()

        def set_indices(self, current: int, master: int) -> None:
            self.current_index = max(0, min(len(self.records) - 1, current)) if self.records else 0
            self.master_index = max(0, min(len(self.records) - 1, master)) if self.records else 0
            self._render()

        def set_mode(self, mode: str) -> None:
            self.mode = mode
            self._timer.stop()
            self.slider.setVisible(mode == "Slider")
            if mode == "Blink":
                self._timer.start()
            self._render()

        def _toggle_blink(self) -> None:
            self._blink_on = not self._blink_on
            self._render()

        def _fit_pair(self) -> tuple[Image.Image | None, Image.Image | None]:
            if not self.records:
                return None, None
            try:
                with Image.open(self.records[self.current_index].path) as current_source:
                    current = ImageOps.exif_transpose(current_source).convert("RGB")
                with Image.open(self.records[self.master_index].path) as master_source:
                    master = ImageOps.exif_transpose(master_source).convert("RGB")
            except Exception:
                return None, None
            width = min(900, max(current.width, master.width))
            height = min(560, max(current.height, master.height))
            current.thumbnail((width, height), Image.Resampling.LANCZOS)
            master.thumbnail((width, height), Image.Resampling.LANCZOS)
            # Put both on a common canvas so blending is stable for different sizes.
            size = (max(current.width, master.width), max(current.height, master.height))
            def canvas(image: Image.Image) -> Image.Image:
                out = Image.new("RGB", size, (17, 24, 32))
                out.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
                return out
            return canvas(current), canvas(master)

        def _set_image(self, label: QLabel, image: Image.Image | None) -> None:
            if image is None:
                label.setText("No image")
                label.setPixmap(QPixmap())
                return
            pixmap = QPixmap.fromImage(_qimage_from_pil(image))
            label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            label.setText("")

        def resizeEvent(self, event: Any) -> None:  # noqa: N802
            super().resizeEvent(event)
            self._render()

        def _render(self) -> None:
            current, master = self._fit_pair()
            if current is None or master is None:
                self._set_image(self.left, None)
                self._set_image(self.right, None)
                return
            mode = self.mode
            if mode == "Side by Side":
                self._set_image(self.left, current)
                self._set_image(self.right, master)
                self.right.setVisible(True)
            else:
                self.right.setVisible(False)
                if mode == "Overlay":
                    composed = Image.blend(master, current, 0.5)
                elif mode == "Difference":
                    composed = ImageOps.autocontrast(ImageChops.difference(master, current))
                elif mode == "Slider":
                    alpha = self.slider.value() / 100.0
                    composed = master.copy()
                    split = int(composed.width * alpha)
                    if split > 0:
                        composed.paste(current.crop((0, 0, split, current.height)), (0, 0))
                elif mode == "Blink":
                    composed = current if self._blink_on else master
                else:  # Onion Skin
                    composed = Image.blend(master, current, 0.32)
                self._set_image(self.left, composed)


    class TimelineWidget(QWidget):
        frame_clicked = Signal(int)

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setMinimumHeight(155)
            self.rows: list[dict[str, Any]] = []
            self.selected = 0
            self.setStyleSheet("background:#111820;border:1px solid #304050;border-radius:5px;")

        def set_data(self, rows: Sequence[dict[str, Any]]) -> None:
            self.rows = list(rows)
            self.update()

        def set_selected(self, frame: int) -> None:
            self.selected = frame
            self.update()

        def mousePressEvent(self, event: Any) -> None:  # noqa: N802
            if not self.rows:
                return
            margin = 38
            index = round((event.position().x() - margin) / max(1, self.width() - margin - 12) * (len(self.rows) - 1))
            index = max(0, min(len(self.rows) - 1, index))
            self.frame_clicked.emit(index + 1)

        def paintEvent(self, _event: Any) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("#111820"))
            if not self.rows:
                painter.setPen(QColor("#93a6b7"))
                painter.drawText(self.rect(), Qt.AlignCenter, "Run analysis to populate drift timeline")
                painter.end()
                return
            left, right, top, bottom = 38, 12, 15, 25
            w = self.width() - left - right
            h = self.height() - top - bottom
            painter.setPen(QPen(QColor("#486070"), 1))
            painter.drawLine(left, top + h, left + w, top + h)
            painter.drawLine(left, top, left, top + h)
            colors = [QColor("#55d6be"), QColor("#f7c873"), QColor("#ec6b73"), QColor("#8da8ff")]
            names = ["overall", "color", "brightness", "identity"]
            for series, name in enumerate(names):
                points: list[QPointF] = []
                for i, row in enumerate(self.rows):
                    dimensions = row.get("dimensions", {})
                    if name == "overall":
                        value = float(row.get("score", 0.0))
                    elif name == "color":
                        value = float(dimensions.get("Color Drift", {}).get("score", 0.0))
                    elif name == "brightness":
                        value = float(dimensions.get("Brightness Flicker", {}).get("score", 0.0))
                    else:
                        value = float(dimensions.get("Identity Drift", {}).get("score", 0.0))
                    x = left + (w * i / max(1, len(self.rows) - 1))
                    y = top + h * (1.0 - max(0.0, min(1.0, value)))
                    points.append(QPointF(x, y))
                painter.setPen(QPen(colors[series], 2))
                for a, b in zip(points, points[1:]):
                    painter.drawLine(a, b)
            if self.selected:
                x = left + w * max(0, min(len(self.rows) - 1, self.selected - 1)) / max(1, len(self.rows) - 1)
                painter.setPen(QPen(QColor("#ffffff"), 1, Qt.DashLine))
                painter.drawLine(x, top, x, top + h)
            painter.setPen(QColor("#93a6b7"))
            painter.drawText(6, 14, "drift")
            painter.drawText(left - 2, self.height() - 7, "F1")
            painter.drawText(left + w - 28, self.height() - 7, f"F{len(self.rows) + 1}")
            painter.end()


    class AnalyzeWorker(QThread):
        completed = Signal(object)
        failed = Signal(str)

        def __init__(self, adapter: AnalysisAdapter, records: Sequence[FrameRecord], roi: Any, master: int, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.adapter = adapter
            self.records = list(records)
            self.roi = roi
            self.master = master

        def run(self) -> None:  # noqa: D401
            try:
                self.completed.emit(self.adapter.analyze(self.records, roi=self.roi, master_index=self.master))
            except Exception as exc:  # pragma: no cover - defensive UI boundary
                self.failed.emit(str(exc))


    class HealthCard(QFrame):
        def __init__(self, name: str, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setFrameShape(QFrame.StyledPanel)
            self.setStyleSheet("QFrame{background:#182631;border:1px solid #304050;border-radius:6px;} QLabel{border:0px;}")
            layout = QVBoxLayout(self)
            layout.setContentsMargins(10, 7, 10, 7)
            self.name = QLabel(name)
            self.name.setStyleSheet("color:#9fb2c0;font-size:11px;")
            self.value = QLabel("—")
            self.value.setStyleSheet("color:#55d6be;font-size:20px;font-weight:600;")
            layout.addWidget(self.name)
            layout.addWidget(self.value)

        def set_value(self, value: Any) -> None:
            if value is None:
                self.value.setText("—")
            else:
                self.value.setText(f"{float(value):.1f}")


    class MainWindow(QMainWindow):
        """Main Visual Continuity Lab window."""

        def __init__(self, adapter: AnalysisAdapter | None = None) -> None:
            super().__init__()
            self.adapter = adapter or AnalysisAdapter()
            self.records: list[FrameRecord] = []
            self.result: dict[str, Any] = {}
            self.roi: tuple[float, float, float, float] | None = None
            self._worker: AnalyzeWorker | None = None
            self.current_index = 0
            self.master_index = 0
            self.setWindowTitle("Visual Continuity Lab — AI 图像角色一致性与连续性实验室")
            self.resize(1440, 930)
            self._build_ui()
            self._apply_theme()

        def _apply_theme(self) -> None:
            self.setStyleSheet(
                """
                QMainWindow, QWidget { background:#0d151d; color:#d8e4ec; }
                QGroupBox { border:1px solid #304050; border-radius:6px; margin-top:8px; padding-top:12px; font-weight:600; }
                QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; color:#9fb2c0; }
                QPushButton, QToolButton { background:#203342; border:1px solid #3c5768; border-radius:4px; padding:6px 10px; color:#e5f1f6; }
                QPushButton:hover, QToolButton:hover { background:#2c4759; }
                QComboBox, QLineEdit { background:#182631; border:1px solid #3c5768; border-radius:4px; padding:5px; }
                QListWidget, QTableWidget { background:#111c25; border:1px solid #304050; alternate-background-color:#15232d; }
                QHeaderView::section { background:#203342; color:#cddbe3; padding:5px; border:0px; }
                QSlider::groove:horizontal { height:4px; background:#304050; }
                QSlider::handle:horizontal { width:12px; background:#55d6be; margin:-5px 0; border-radius:6px; }
                """
            )

        def _build_ui(self) -> None:
            self.status = QStatusBar()
            self.setStatusBar(self.status)
            self.status.showMessage("Ready — add a sequence to begin")

            root = QWidget()
            self.setCentralWidget(root)
            root_layout = QHBoxLayout(root)
            root_layout.setContentsMargins(10, 10, 10, 10)
            root_layout.setSpacing(10)

            left_panel = QWidget()
            left_panel.setMinimumWidth(300)
            left_panel.setMaximumWidth(370)
            left_layout = QVBoxLayout(left_panel)
            left_layout.setContentsMargins(0, 0, 0, 0)

            sequence_box = QGroupBox("Sequence")
            seq_layout = QVBoxLayout(sequence_box)
            buttons = QGridLayout()
            self.add_images_btn = QPushButton("Add Images")
            self.add_folder_btn = QPushButton("Add Folder")
            self.add_video_btn = QPushButton("Add Video")
            self.import_json_btn = QPushButton("Import JSON")
            self.clear_btn = QPushButton("Clear")
            buttons.addWidget(self.add_images_btn, 0, 0)
            buttons.addWidget(self.add_folder_btn, 0, 1)
            buttons.addWidget(self.add_video_btn, 1, 0)
            buttons.addWidget(self.import_json_btn, 1, 1)
            buttons.addWidget(self.clear_btn, 2, 0, 1, 2)
            seq_layout.addLayout(buttons)
            self.sequence_list = QListWidget()
            self.sequence_list.setAlternatingRowColors(True)
            self.sequence_list.currentRowChanged.connect(self._select_frame)
            seq_layout.addWidget(self.sequence_list, 1)
            self.sequence_summary = QLabel("0 frames")
            self.sequence_summary.setStyleSheet("color:#8da1af;")
            seq_layout.addWidget(self.sequence_summary)
            left_layout.addWidget(sequence_box, 1)

            roi_box = QGroupBox("Character Region / ROI")
            roi_layout = QVBoxLayout(roi_box)
            self.roi_state = QLabel("No ROI — draw a rectangle on the preview")
            self.roi_state.setWordWrap(True)
            self.roi_state.setStyleSheet("color:#9fb2c0;font-size:11px;")
            roi_layout.addWidget(self.roi_state)
            roi_buttons = QHBoxLayout()
            self.roi_btn = QPushButton("Draw ROI")
            self.roi_reset_btn = QPushButton("Reset")
            self.roi_lock = QCheckBox("Lock region")
            roi_buttons.addWidget(self.roi_btn)
            roi_buttons.addWidget(self.roi_reset_btn)
            roi_layout.addLayout(roi_buttons)
            roi_layout.addWidget(self.roi_lock)
            left_layout.addWidget(roi_box)

            labels_box = QGroupBox("Manual QC Label")
            labels_layout = QHBoxLayout(labels_box)
            self.label_combo = QComboBox()
            self.label_combo.addItems(["Good", "Bad Face", "Bad Hair", "Bad Costume", "Bad Pose", "Bad Background"])
            self.label_btn = QPushButton("Apply")
            labels_layout.addWidget(self.label_combo, 1)
            labels_layout.addWidget(self.label_btn)
            left_layout.addWidget(labels_box)

            ref_box = QGroupBox("Reference Lock")
            ref_layout = QVBoxLayout(ref_box)
            ref_layout.addWidget(QLabel("Master Reference"))
            self.master_combo = QComboBox()
            self.master_combo.currentIndexChanged.connect(self._master_changed)
            ref_layout.addWidget(self.master_combo)
            left_layout.addWidget(ref_box)

            actions = QHBoxLayout()
            self.analyze_btn = QPushButton("Analyze Sequence")
            self.export_btn = QPushButton("Export JSON")
            self.report_btn = QPushButton("Report")
            self.analyze_btn.setMinimumHeight(36)
            actions.addWidget(self.analyze_btn, 2)
            actions.addWidget(self.export_btn, 1)
            actions.addWidget(self.report_btn, 1)
            left_layout.addLayout(actions)

            root_layout.addWidget(left_panel)

            center = QWidget()
            center_layout = QVBoxLayout(center)
            center_layout.setContentsMargins(0, 0, 0, 0)

            preview_box = QGroupBox("Character Region Preview")
            preview_layout = QVBoxLayout(preview_box)
            self.preview = ImageCanvas()
            preview_layout.addWidget(self.preview, 1)
            self.frame_label = QLabel("Frame —")
            self.frame_label.setStyleSheet("color:#9fb2c0;")
            preview_layout.addWidget(self.frame_label)
            center_layout.addWidget(preview_box, 4)

            compare_box = QGroupBox("Compare Viewer")
            compare_layout = QVBoxLayout(compare_box)
            compare_tools = QHBoxLayout()
            compare_tools.addWidget(QLabel("Mode"))
            self.compare_mode = QComboBox()
            self.compare_mode.addItems(CompareViewer.MODES)
            self.compare_mode.currentTextChanged.connect(self._compare_mode_changed)
            compare_tools.addWidget(self.compare_mode)
            compare_tools.addStretch(1)
            self.compare_hint = QLabel("Current frame vs master")
            self.compare_hint.setStyleSheet("color:#8da1af;")
            compare_tools.addWidget(self.compare_hint)
            compare_layout.addLayout(compare_tools)
            self.compare = CompareViewer()
            self.compare.setMinimumHeight(220)
            compare_layout.addWidget(self.compare, 1)
            center_layout.addWidget(compare_box, 3)
            root_layout.addWidget(center, 5)

            right_panel = QWidget()
            right_panel.setMinimumWidth(430)
            right_layout = QVBoxLayout(right_panel)
            right_layout.setContentsMargins(0, 0, 0, 0)
            health_box = QGroupBox("Sequence Health (stability %) ")
            health_layout = QGridLayout(health_box)
            self.health_cards: dict[str, HealthCard] = {}
            names = ["Identity Stability", "Motion Stability", "Color Stability", "Lighting Stability", "Background Stability", "Overall"]
            for i, name in enumerate(names):
                card = HealthCard(name)
                self.health_cards[name] = card
                health_layout.addWidget(card, i // 2, i % 2)
            right_layout.addWidget(health_box)

            timeline_box = QGroupBox("Drift Timeline")
            timeline_layout = QVBoxLayout(timeline_box)
            timeline_tools = QHBoxLayout()
            timeline_tools.addWidget(QLabel("Comparison"))
            self.consistency_mode = QComboBox()
            self.consistency_mode.addItems(["Adjacent Consistency", "Master Consistency"])
            self.consistency_mode.currentIndexChanged.connect(self._consistency_mode_changed)
            timeline_tools.addWidget(self.consistency_mode)
            timeline_tools.addStretch(1)
            timeline_layout.addLayout(timeline_tools)
            self.timeline = TimelineWidget()
            self.timeline.frame_clicked.connect(self._timeline_select)
            timeline_layout.addWidget(self.timeline)
            legend = QLabel("overall · color · brightness · identity   (click to inspect a frame)")
            legend.setStyleSheet("color:#8da1af;font-size:11px;")
            timeline_layout.addWidget(legend)
            right_layout.addWidget(timeline_box)

            issues_box = QGroupBox("Detected Issues")
            issues_layout = QVBoxLayout(issues_box)
            self.issue_table = QTableWidget(0, 6)
            self.issue_table.setHorizontalHeaderLabels(["Type", "Start", "End", "Severity", "Evidence", "Notes"])
            self.issue_table.setWordWrap(True)
            self.issue_table.setAlternatingRowColors(True)
            self.issue_table.setSelectionBehavior(QTableWidget.SelectRows)
            self.issue_table.cellDoubleClicked.connect(self._issue_selected)
            header = self.issue_table.horizontalHeader()
            header.setStretchLastSection(True)
            for col in (0, 1, 2, 3):
                header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.Stretch)
            issues_layout.addWidget(self.issue_table)
            right_layout.addWidget(issues_box, 2)
            root_layout.addWidget(right_panel, 4)

            self.add_images_btn.clicked.connect(self._add_images)
            self.add_folder_btn.clicked.connect(self._add_folder)
            self.add_video_btn.clicked.connect(self._add_video)
            self.import_json_btn.clicked.connect(self._import_json)
            self.clear_btn.clicked.connect(self._clear)
            self.roi_btn.clicked.connect(self._toggle_roi)
            self.roi_reset_btn.clicked.connect(self._reset_roi)
            self.roi_lock.toggled.connect(self._roi_lock_changed)
            self.analyze_btn.clicked.connect(self._analyze)
            self.export_btn.clicked.connect(self._export)
            self.report_btn.clicked.connect(self._export_report)
            self.label_btn.clicked.connect(self._apply_label)
            self.preview.roi_changed.connect(self._roi_changed)

        # ---- sequence and import -------------------------------------------------
        def _add_images(self) -> None:
            files, _ = QFileDialog.getOpenFileNames(self, "Add images", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)")
            if files:
                self._add_paths(files)

        def _add_folder(self) -> None:
            folder = QFileDialog.getExistingDirectory(self, "Add image folder")
            if folder:
                self._add_paths([folder])

        def _add_video(self) -> None:
            file, _ = QFileDialog.getOpenFileName(self, "Add video", "", "Video (*.mp4 *.mov *.mkv *.avi *.webm *.m4v)")
            if file:
                self._add_paths([file])

        def _import_json(self) -> None:
            file, _ = QFileDialog.getOpenFileName(self, "Import sequence / continuity JSON", "", "JSON (*.json)")
            if not file:
                return
            try:
                data = json.loads(Path(file).read_text(encoding="utf-8"))
                frame_data = data.get("frames", data.get("sequence", []))
                if not frame_data and data.get("shots"):
                    first_shot = (data.get("shots") or [{}])[0]
                    frame_data = first_shot.get("frames", []) if isinstance(first_shot, dict) else []
                if isinstance(frame_data, dict):
                    frame_data = list(frame_data.values())
                imported: list[FrameRecord] = []
                base = Path(file).parent
                for index, item in enumerate(frame_data, 1):
                    if not isinstance(item, dict):
                        continue
                    # VCL manifests use ``file`` while FrameForge exports use
                    # ``path``.  Accept both so a generated Sequence JSON can
                    # be opened directly in the desktop UI.
                    raw_value = item.get("path") or item.get("file")
                    if not raw_value:
                        continue
                    raw_path = Path(str(raw_value))
                    path = raw_path if raw_path.is_absolute() else base / raw_path
                    if not path.exists() and (base / "frames" / raw_path.name).exists():
                        path = base / "frames" / raw_path.name
                    if not path.exists():
                        continue
                    frame_id = item.get("frame_id", item.get("frameId", item.get("id", f"F{index:04d}")))
                    frame_number = item.get("frame_number", item.get("frameNumber", item.get("frame", index)))
                    imported.append(
                        FrameRecord(
                            frame_id=str(frame_id),
                            frame_number=int(frame_number or index),
                            path=str(path),
                            timestamp=float(item.get("timestamp", 0.0) or 0.0),
                            character=str(item.get("character", "Character 1")),
                            shot=str(item.get("shot", "Shot 1")),
                            status=str(item.get("status", "Pending")),
                            metadata={"manifest": item, "markers": item.get("markers", {})},
                        )
                    )
                if not imported:
                    raise ValueError("JSON did not contain readable frame paths")
                self.records.extend(imported)
                for i, record in enumerate(self.records, 1):
                    record.frame_number, record.frame_id = i, f"F{i:04d}"
                self._refresh_sequence()
                self.status.showMessage(f"Imported {len(imported)} frame(s) from JSON")
            except Exception as exc:
                QMessageBox.warning(self, "Import failed", str(exc))

        def _add_paths(self, paths: Sequence[str]) -> None:
            self.status.showMessage("Loading sequence…")
            records = self.adapter.load_paths(paths)
            if not records:
                QMessageBox.warning(self, "No images", "No supported image/video frames were found.")
                return
            self.records.extend(records)
            # Re-number globally after multiple imports.
            for i, record in enumerate(self.records, 1):
                record.frame_number = i
                record.frame_id = f"F{i:04d}"
            self._refresh_sequence()
            self.status.showMessage(f"Loaded {len(records)} frame(s)")

        def _refresh_sequence(self) -> None:
            self.sequence_list.blockSignals(True)
            self.sequence_list.clear()
            self.master_combo.blockSignals(True)
            self.master_combo.clear()
            for i, record in enumerate(self.records):
                item = QListWidgetItem(f"{record.frame_id}   {Path(record.path).name}")
                item.setData(Qt.UserRole, i)
                self.sequence_list.addItem(item)
                self.master_combo.addItem(f"{record.frame_id} — {Path(record.path).name}")
            self.sequence_list.blockSignals(False)
            self.master_combo.blockSignals(False)
            self.sequence_summary.setText(f"{len(self.records)} frame(s) · {'core' if self.adapter.core_available else 'baseline'} analyser")
            if self.records:
                self.sequence_list.setCurrentRow(0)
                self.master_combo.setCurrentIndex(min(self.master_index, len(self.records) - 1))
                self.compare.set_records(self.records)
                self._select_frame(0)

        def _select_frame(self, row: int) -> None:
            if row < 0 or row >= len(self.records):
                return
            self.current_index = row
            record = self.records[row]
            self.preview.set_image(record.path)
            self.preview.set_roi(self.roi)
            self.frame_label.setText(f"{record.frame_id}  ·  frame {record.frame_number}  ·  {record.timestamp:.3f}s  ·  {record.status}")
            self.compare.set_indices(row, self.master_index)
            self.compare_hint.setText(f"{record.frame_id} vs {self.records[self.master_index].frame_id if self.records else '—'}")
            self.timeline.set_selected(row)

        def _master_changed(self, index: int) -> None:
            if 0 <= index < len(self.records):
                self.master_index = index
                self.compare.set_indices(self.current_index, self.master_index)
                self.compare_hint.setText(f"{self.records[self.current_index].frame_id} vs {self.records[index].frame_id}")

        def _apply_label(self) -> None:
            if not self.records or not (0 <= self.current_index < len(self.records)):
                return
            label = self.label_combo.currentText()
            record = self.records[self.current_index]
            record.status = label
            # Keep labels in a UI-owned interchange field.  The core adapter can
            # consume this list later without making the UI depend on a model.
            record.metadata = getattr(record, "metadata", {})
            record.metadata["manual_label"] = label
            item = self.sequence_list.item(self.current_index)
            if item is not None:
                item.setText(f"{record.frame_id}   {Path(record.path).name}   [{label}]")
            self.frame_label.setText(f"{record.frame_id}  ·  {label}")
            self.status.showMessage(f"Labeled {record.frame_id}: {label}")

        # ---- ROI -----------------------------------------------------------------
        def _toggle_roi(self) -> None:
            active = not self.preview.editable
            self.preview.editable = active
            self.roi_btn.setText("Finish ROI" if active else "Draw ROI")
            self.status.showMessage("Drag a rectangle around the character" if active else "ROI selection finished")

        def _roi_lock_changed(self, checked: bool) -> None:
            self.preview.locked = checked
            self.roi_state.setText("ROI locked" if checked else "ROI unlocked — draw a rectangle on the preview")

        def _roi_changed(self, roi: tuple[float, float, float, float]) -> None:
            self.roi = roi
            self.roi_state.setText(f"Locked region: x={roi[0]:.2f}, y={roi[1]:.2f}, w={roi[2]:.2f}, h={roi[3]:.2f}" if self.roi_lock.isChecked() else f"Region: x={roi[0]:.2f}, y={roi[1]:.2f}, w={roi[2]:.2f}, h={roi[3]:.2f}")

        def _reset_roi(self) -> None:
            self.roi = None
            self.preview.set_roi(None)
            self.roi_state.setText("No ROI — draw a rectangle on the preview")

        # ---- analysis and output --------------------------------------------------
        def _analyze(self) -> None:
            if not self.records:
                QMessageBox.information(self, "No sequence", "Add images, a folder, or a video first.")
                return
            if self._worker and self._worker.isRunning():
                return
            self.analyze_btn.setEnabled(False)
            self.status.showMessage("Analyzing sequence…")
            self._worker = AnalyzeWorker(self.adapter, self.records, self.roi, self.master_index, self)
            self._worker.completed.connect(self._analysis_done)
            self._worker.failed.connect(self._analysis_failed)
            self._worker.start()

        def _analysis_done(self, result: dict[str, Any]) -> None:
            self.result = result or {}
            health = self.result.get("health", {})
            for name, card in self.health_cards.items():
                card.set_value(health.get(name))
            self.timeline.set_data(self.result.get("drift", []))
            self._populate_issues(self.result.get("issues", []))
            for record in self.records:
                record.status = "Analyzed"
            self._select_frame(self.current_index)
            self.analyze_btn.setEnabled(True)
            self.status.showMessage(f"Analysis complete — {len(self.result.get('issues', []))} issue(s) detected")

        def _analysis_failed(self, message: str) -> None:
            self.analyze_btn.setEnabled(True)
            self.status.showMessage("Analysis failed")
            QMessageBox.critical(self, "Analysis failed", message)

        def _populate_issues(self, issues: Sequence[dict[str, Any]]) -> None:
            self.issue_table.setRowCount(0)
            for issue in issues:
                row = self.issue_table.rowCount()
                self.issue_table.insertRow(row)
                values = [
                    issue.get("type", "Unknown"),
                    issue.get("start_frame", "—"),
                    issue.get("end_frame", "—"),
                    issue.get("severity", "—"),
                    issue.get("evidence", "—"),
                    issue.get("notes", "—"),
                ]
                for col, value in enumerate(values):
                    item = QTableWidgetItem(str(value))
                    item.setToolTip(str(value))
                    self.issue_table.setItem(row, col, item)

        def _timeline_select(self, frame: int) -> None:
            self.sequence_list.setCurrentRow(max(0, min(len(self.records) - 1, frame - 1)))

        def _issue_selected(self, row: int, _column: int) -> None:
            item = self.issue_table.item(row, 1)
            if item is not None:
                try:
                    self._timeline_select(int(item.text()))
                except ValueError:
                    pass

        def _compare_mode_changed(self, mode: str) -> None:
            self.compare.set_mode(mode)

        def _consistency_mode_changed(self, index: int) -> None:
            if not self.result:
                return
            rows = self.result.get("master_drift", []) if index == 1 else self.result.get("drift", [])
            self.timeline.set_data(rows)
            self.status.showMessage("Showing master-reference consistency" if index == 1 else "Showing adjacent-frame consistency")

        def _export(self) -> None:
            if not self.result:
                QMessageBox.information(self, "No result", "Run analysis before exporting.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Export continuity result", "continuity_result.json", "JSON (*.json)")
            if not path:
                return
            payload = dict(self.result)
            payload["sequence"] = [record.to_dict() for record in self.records]
            payload["roi"] = self.roi
            payload["master_reference"] = self.records[self.master_index].frame_id if self.records else None
            payload["manual_labels"] = [
                {"frame_id": record.frame_id, "label": record.metadata.get("manual_label")}
                for record in self.records
                if getattr(record, "metadata", {}).get("manual_label")
            ]
            try:
                Path(path).write_text(result_to_json(payload), encoding="utf-8")
                self.status.showMessage(f"Exported {path}")
            except OSError as exc:
                QMessageBox.critical(self, "Export failed", str(exc))

        def _export_report(self) -> None:
            if not self.result:
                QMessageBox.information(self, "No result", "Run analysis before exporting a report.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Export Shot Continuity Report", "Shot_Continuity_Report.md", "Markdown (*.md)")
            if not path:
                return
            health = self.result.get("health", {})
            lines = ["# Shot Continuity Report", "", f"Frames: {len(self.records)}", "", "## Sequence Health", "", "| Dimension | Stability |", "|---|---:|"]
            for key, value in health.items():
                if key == "sample_count":
                    continue
                lines.append(f"| {key} | {value} |")
            lines += ["", "## Issues", ""]
            issues = self.result.get("issues", [])
            if not issues:
                lines.append("No issues detected.")
            for issue in issues:
                lines.append(f"- **{issue.get('severity', 'Unknown')}** `{issue.get('type', 'Unknown')}` — F{issue.get('start_frame', '?')} → F{issue.get('end_frame', '?')}: {issue.get('evidence', '')}")
            recommended = self.result.get("recommended_frames", []) or []
            lines += ["", "## Recommended Frames to Regenerate", "", ", ".join(str(x) for x in recommended) if recommended else "See high-severity issue ranges above.", ""]
            try:
                Path(path).write_text("\n".join(lines), encoding="utf-8")
                self.status.showMessage(f"Exported report {path}")
            except OSError as exc:
                QMessageBox.critical(self, "Report export failed", str(exc))

        def _clear(self) -> None:
            self.records.clear()
            self.result = {}
            self.roi = None
            self.sequence_list.clear()
            self.master_combo.clear()
            self.preview.set_image(None)
            self.preview.set_roi(None)
            self.compare.set_records([])
            self.timeline.set_data([])
            self.issue_table.setRowCount(0)
            self.sequence_summary.setText("0 frames")
            for card in self.health_cards.values():
                card.set_value(None)
            self.status.showMessage("Ready — add a sequence to begin")


else:  # pragma: no cover - makes import diagnostics friendly without PySide6

    class MainWindow:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("PySide6 is not installed")
