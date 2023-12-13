import numpy as np
from pathlib import Path
from typing import Callable
from PyQt5.QtCore import Qt, QLineF, QPointF
from PyQt5.QtGui import QPixmap, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsLineItem,
    QGraphicsEllipseItem,
    QFileDialog,
    QAction,
    QStatusBar,
    QLabel,
)


class Pose2DLabeler(QMainWindow):
    def __init__(
        self,
        viewer_scale=1.0,
        point_radius=10,
        line_width=5,
        center_color=Qt.red,
        direction_color=Qt.blue,
        center_key=Qt.RightButton,
        direction_key=Qt.LeftButton,
        next_key=Qt.Key_D,
        prev_key=Qt.Key_A,
        save_key=Qt.Key_S,
        save_key_modifier=Qt.ControlModifier,
        name_pose2d_file: Callable[[str], str] = None,
        default_img_pattern: str = "*.bmp",
    ):
        super().__init__()

        self.setWindowTitle("2DPose Labeler")
        self.resize(800, 600)

        self.graphic_view = QGraphicsView(self)
        self.annotator = Pose2DAnnotator(
            circle_radius=point_radius,
            line_width=line_width,
            center_color=center_color,
            direction_color=direction_color,
            center_key=center_key,
            direction_key=direction_key,
        )
        self.graphic_view.setScene(self.annotator)
        self.viewer_scale = viewer_scale
        self.graphic_view.scale(viewer_scale, viewer_scale)
        self.setCentralWidget(self.graphic_view)

        self.status_bar = CustomStatusBar(self)
        self.setStatusBar(self.status_bar)

        open_img_action = QAction("Open Image", self)
        open_img_action.triggered.connect(self.open_image)
        self.menuBar().addAction(open_img_action)

        open_folder_action = QAction("Open Folder", self)
        open_folder_action.triggered.connect(self.open_folder)
        self.menuBar().addAction(open_folder_action)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_pose2d)
        self.menuBar().addAction(save_action)

        save_as_action = QAction("Save As", self)
        save_as_action.triggered.connect(self.save_as_pose2d)
        self.menuBar().addAction(save_as_action)

        self.next_key = next_key
        self.prev_key = prev_key
        self.save_key = save_key
        self.save_key_modifier = save_key_modifier

        if name_pose2d_file is not None:
            self._get_pose2d_filepath = name_pose2d_file
        self.default_img_pattern = default_img_pattern

    def open_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *bmp)"
        )
        if filename:
            self.img_list = [filename]
            self.cur_idx = 0
            self._load_image()

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Folder", "")
        if folder:
            self.img_list = [
                str(p) for p in Path(folder).glob(self.default_img_pattern)
            ]
            if len(self.img_list) == 0:
                self.status_bar.set_message("No image found.")
                return
            self.cur_idx = 0
            self._load_image()

    def _load_image(self):
        self.cur_img_path = self.img_list[self.cur_idx]
        self.cur_pose2d_path = self._get_pose2d_filepath(self.cur_img_path)

        pixmap = QPixmap(self.cur_img_path)
        pose2d = self._load_pose2d()
        self.annotator.set_pixmap_pose2d(pixmap, pose2d)

        self._resize_window()

        self.status_bar.set_message(f"Image: {self.cur_img_path}")
        self.status_bar.set_idx_indicator(self.cur_idx, len(self.img_list))

    def _load_pose2d(self):
        if Path(self.cur_pose2d_path).exists():
            with open(self.cur_pose2d_path, "r") as f:
                return [float(x) for x in f.readline().split()]
        return None

    def _next_image(self):
        if hasattr(self, "cur_idx") and hasattr(self, "img_list"):
            if self.cur_idx == len(self.img_list) - 1:
                self.cur_idx = 0
            else:
                self.cur_idx += 1
            self._load_image()

    def _prev_image(self):
        if hasattr(self, "cur_idx") and hasattr(self, "img_list"):
            if self.cur_idx == 0:
                self.cur_idx = len(self.img_list) - 1
            else:
                self.cur_idx -= 1
            self._load_image()

    def _get_pose2d_filepath(self, img_path):
        return str(Path(img_path).with_suffix(".pose2d"))

    def save_pose2d(self):
        if self.annotator.is_annotated():
            pose2d = self.annotator.get_pose2d()
            self._write_pose2d(pose2d, self.cur_pose2d_path)
            self.status_bar.set_message(f"Saved to: {self.cur_pose2d_path}")

    def save_as_pose2d(self):
        if self.annotator.is_annotated():
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Pose2D", "", "Pose2D (*.pose2d)"
            )
            if filename:
                pose2d = self.annotator.get_pose2d()
                self._write_pose2d(pose2d, filename)
                self.status_bar.set_message(f"Saved to: {filename}")

    def _write_pose2d(self, pose2d, filepath):
        with open(filepath, "w") as f:
            f.write(f"{pose2d[0]} {pose2d[1]} {pose2d[2]}")

    def keyPressEvent(self, event):
        if event.key() == self.next_key:
            self._next_image()
        elif event.key() == self.prev_key:
            self._prev_image()
        elif (
            event.key() == self.save_key and event.modifiers() == self.save_key_modifier
        ):
            self.save_pose2d()
        else:
            super().keyPressEvent(event)

    def _resize_window(self):
        width, height = self.annotator.width(), self.annotator.height()
        width, height = width * self.viewer_scale, height * self.viewer_scale
        self.resize(int(width) + 20, int(height) + 80)


class Pose2DAnnotator(QGraphicsScene):
    def __init__(
        self,
        circle_radius=10,
        line_width=5,
        center_color=Qt.red,
        direction_color=Qt.blue,
        center_key=Qt.RightButton,
        direction_key=Qt.LeftButton,
    ):
        super().__init__()

        self.circle_radius = circle_radius
        self.line_width = line_width
        self.center_color = center_color
        self.direction_color = direction_color
        self.center_key = center_key
        self.direction_key = direction_key

        self.cx, self.cy, self.angle = None, None, None

    def set_pixmap_pose2d(self, pixmap, pose2d=None):
        self.clear()

        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.addItem(self.pixmap_item)

        if pose2d is None:
            self.cx, self.cy, self.angle = None, None, None
        else:
            self.cx, self.cy, shifted_angle = pose2d
            self.angle = (shifted_angle + 90 + 360) % 360
            self._draw_center()
            self._draw_direction_from_center()

    def _draw_center(self):
        self._safe_remove_item("center_item")
        self.center_item = QGraphicsEllipseItem(
            self.cx - self.circle_radius,
            self.cy - self.circle_radius,
            self.circle_radius * 2,
            self.circle_radius * 2,
        )
        self.center_item.setPen(QPen(self.center_color, self.line_width))
        self.addItem(self.center_item)

    def _start_drawing_direction(self):
        self._safe_remove_item("line_item")
        self._safe_remove_item("line_start_item")

        # Draw line.
        self.line_item = QGraphicsLineItem(QLineF(self.start_point, self.start_point))
        self.line_item.setPen(QPen(self.direction_color, self.line_width))
        self.addItem(self.line_item)

        # Draw line start point marker.
        self.line_start_item = QGraphicsEllipseItem(
            self.start_point.x() - self.circle_radius,
            self.start_point.y() - self.circle_radius,
            self.circle_radius * 2,
            self.circle_radius * 2,
        )
        self.line_start_item.setPen(QPen(self.direction_color, self.line_width))
        self.addItem(self.line_start_item)

    def mouseMoveEvent(self, event):
        if not self.is_pixmap_loaded():
            return

        # Update line.
        if event.buttons() == self.direction_key and self._has_item("line_item"):
            line = QLineF(self.start_point, event.scenePos())
            self.line_item.setLine(line)
            self.angle = line.angle()
        else:
            super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if not self.is_pixmap_loaded():
            return

        if event.button() == self.center_key:
            self.cx, self.cy = event.scenePos().x(), event.scenePos().y()
            self._draw_center()
        elif event.button() == self.direction_key:
            self.start_point = event.scenePos()
            self._start_drawing_direction()
        else:
            super().mousePressEvent(event)

    def _draw_direction_from_center(self):
        pixmap_width = self.pixmap_item.pixmap().width()
        pixmap_height = self.pixmap_item.pixmap().height()
        length = min(pixmap_width, pixmap_height) / 4
        start = QPointF(self.cx, self.cy)
        end = QPointF(
            self.cx + length * np.cos(np.deg2rad(-self.angle)),
            self.cy + length * np.sin(np.deg2rad(-self.angle)),
        )
        self.line_item = QGraphicsLineItem(QLineF(start, end))
        self.line_item.setPen(QPen(self.direction_color, self.line_width))
        self.addItem(self.line_item)

    def _has_item(self, name):
        return hasattr(self, name) and getattr(self, name) in self.items()

    def _safe_remove_item(self, name):
        if self._has_item(name):
            self.removeItem(getattr(self, name))

    def get_pose2d(self):
        cx, cy = np.round(self.cx), np.round(self.cy)
        shifted_angle = (self.angle - 90 + 180) % 360 - 180
        return cx, cy, shifted_angle

    def is_annotated(self):
        return self.cx is not None and self.cy is not None and self.angle is not None

    def is_pixmap_loaded(self):
        return self._has_item("pixmap_item")


class CustomStatusBar(QStatusBar):
    def __init__(self, parent):
        super().__init__(parent)
        self.message = QLabel()
        self.idx_indicator = QLabel()
        self.addPermanentWidget(self.message, stretch=1)
        self.addPermanentWidget(self.idx_indicator)

    def set_message(self, msg):
        self.message.setText(msg)

    def set_idx_indicator(self, idx, list_len):
        self.idx_indicator.setText(f"[{idx+1}/{list_len}]")


if __name__ == "__main__":
    from argparse import ArgumentParser
    import sys

    parser = ArgumentParser()
    parser.add_argument("-i", "--img-pattern", type=str, default="*.bmp")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = Pose2DLabeler(default_img_pattern=args.img_pattern)
    window.show()
    sys.exit(app.exec_())
