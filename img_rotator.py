import cv2
from pathlib import Path
from typing import Callable
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QFileDialog,
    QAction,
    QStatusBar,
    QLabel,
)


class ImageRotator(QMainWindow):
    def __init__(
        self,
        rot_pose2d=True,
        rot_counterclockwise_key=Qt.Key_Q,
        rot_clockwise_key=Qt.Key_E,
        next_key=Qt.Key_D,
        prev_key=Qt.Key_A,
        save_key=Qt.Key_S,
        save_key_modifier=Qt.ControlModifier,
        name_pose2d_file: Callable[[str], str] = None,
        default_img_pattern: str = "*.bmp",
    ):
        super().__init__()

        self.setWindowTitle("Image Rotator")
        self.resize(800, 600)

        self.graphic_view = QGraphicsView(self)
        self.scene = ImageRotatorScene(
            rot_counterclockwise_key=rot_counterclockwise_key,
            rot_clockwise_key=rot_clockwise_key,
        )
        self.graphic_view.setScene(self.scene)
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
        save_action.triggered.connect(self.save)
        self.menuBar().addAction(save_action)

        save_as_action = QAction("Save As", self)
        save_as_action.triggered.connect(self.save_as)
        self.menuBar().addAction(save_as_action)

        self.next_key = next_key
        self.prev_key = prev_key
        self.save_key = save_key
        self.save_key_modifier = save_key_modifier

        if name_pose2d_file is not None:
            self._get_pose2d_filepath = name_pose2d_file
        self.default_img_pattern = default_img_pattern
        self.rot_pose2d = rot_pose2d

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
        pixmap = QPixmap(self.cur_img_path)
        self.scene.set_pixmap(pixmap)
        self.cur_img = cv2.imread(self.cur_img_path, cv2.IMREAD_GRAYSCALE)

        if self.rot_pose2d:
            self.cur_pose2d_path = self._get_pose2d_filepath(self.cur_img_path)
            self.cur_pose2d = self._load_pose2d()

        width, height = self.scene.width(), self.scene.height()
        self.resize(int(width) + 20, int(height) + 80)

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

    def save(self):
        if self.scene.is_pixmap_loaded():
            rot_angle = self.scene.get_rotation()
            rotated_img = self._rotate_image(self.cur_img, rot_angle)
            cv2.imwrite(self.cur_img_path, rotated_img)
            if self.rot_pose2d and self.cur_pose2d is not None:
                rotated_pose2d = self._rotate_pose2d(
                    self.cur_pose2d, rot_angle, *self.cur_img.shape
                )
                self._write_pose2d(rotated_pose2d, self.cur_pose2d_path)
            self.status_bar.set_message(f"Saved to: {self.cur_img_path}")

    def save_as(self):
        if self.scene.is_pixmap_loaded():
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Image", "", "Images (*.png *.jpg *bmp)"
            )
            if filename:
                rot_angle = self.scene.get_rotation()
                rotated_img = self._rotate_image(self.cur_img, rot_angle)
                cv2.imwrite(filename, rotated_img)
                self.status_bar.set_message(f"Saved to: {filename}")

    def _write_pose2d(self, pose2d, filepath):
        with open(filepath, "w") as f:
            f.write(f"{pose2d[0]} {pose2d[1]} {pose2d[2]}")

    def _rotate_image(self, img, rot_angle):
        angle = int(rot_angle % 360)
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def _rotate_pose2d(self, pose2d, rot_angle, img_height, img_width):
        cx, cy, angle = pose2d
        rotated_angle = (angle - rot_angle + 180) % 360 - 180
        rot_angle = int(rot_angle % 360)
        if rot_angle == 90:
            cx, cy = img_height - 1 - cy, cx
        elif rot_angle == 180:
            cx, cy = img_width - 1 - cx, img_height - 1 - cy
        elif rot_angle == 270:
            cx, cy = cy, img_width - 1 - cx
        return cx, cy, rotated_angle

    def keyPressEvent(self, event):
        if event.key() == self.next_key:
            self._next_image()
        elif event.key() == self.prev_key:
            self._prev_image()
        elif (
            event.key() == self.save_key and event.modifiers() == self.save_key_modifier
        ):
            self.save()
        else:
            super().keyPressEvent(event)


class ImageRotatorScene(QGraphicsScene):
    def __init__(
        self,
        rot_counterclockwise_key=Qt.Key_Q,
        rot_clockwise_key=Qt.Key_E,
    ):
        super().__init__()

        self.cur_rot_angle = 0
        self.rot_counterclockwise_key = rot_counterclockwise_key
        self.rot_clockwise_key = rot_clockwise_key

    def set_pixmap(self, pixmap):
        self.clear()
        self.cur_rot_angle = 0

        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.addItem(self.pixmap_item)

        rot_origin = pixmap.rect().center()
        self.pixmap_item.setTransformOriginPoint(rot_origin)

    def rotate_pixmap(self, angle):
        self.cur_rot_angle = (self.cur_rot_angle + angle) % 360
        self.pixmap_item.setRotation(self.cur_rot_angle)

    def keyPressEvent(self, event):
        if event.key() == self.rot_counterclockwise_key:
            self.rotate_pixmap(-90)
        elif event.key() == self.rot_clockwise_key:
            self.rotate_pixmap(90)
        else:
            super().keyPressEvent(event)

    def get_rotation(self):
        return self.cur_rot_angle

    def _has_item(self, name):
        return hasattr(self, name) and getattr(self, name) in self.items()

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
    import sys

    app = QApplication(sys.argv)
    window = ImageRotator()
    window.show()
    sys.exit(app.exec_())
