import cv2
import numpy as np
from pathlib import Path
from typing import Callable
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
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


class Pose2DAlignmentLabeler(QMainWindow):
    def __init__(
        self,
        query_color=(255, 0, 0, 150),
        template_color=(0, 0, 255, 255),
        rotation_key=Qt.RightButton,
        rotation_speed=0.8,
        move_up_key=Qt.Key_Up,
        move_down_key=Qt.Key_Down,
        move_left_key=Qt.Key_Left,
        move_right_key=Qt.Key_Right,
        key_move_step=1,
        next_key=Qt.Key_D,
        prev_key=Qt.Key_A,
        save_key=Qt.Key_S,
        name_query_pose2d_file: Callable[[str], str] = None,
        name_template_pose2d_file: Callable[[str], str] = None,
    ):
        super().__init__()

        self.setWindowTitle("2DPose Alignment Labeler")
        self.resize(1200, 800)

        self.graphic_view = QGraphicsView(self)
        self.fp_aligner = FingerprintManualAligner(
            query_color=query_color,
            template_color=template_color,
            rotation_key=rotation_key,
            rotation_speed=rotation_speed,
            move_up_key=move_up_key,
            move_down_key=move_down_key,
            move_left_key=move_left_key,
            move_right_key=move_right_key,
            key_move_step=key_move_step,
        )
        self.graphic_view.setScene(self.fp_aligner)
        self.setCentralWidget(self.graphic_view)

        self.status_bar = CustomStatusBar(self)
        self.setStatusBar(self.status_bar)

        open_query_img_action = QAction("Open Query Image", self)
        open_query_img_action.triggered.connect(self.open_query_image)
        self.menuBar().addAction(open_query_img_action)

        open_query_folder_action = QAction("Open Query Folder", self)
        open_query_folder_action.triggered.connect(self.open_query_folder)
        self.menuBar().addAction(open_query_folder_action)

        open_template_img_action = QAction("Open Template Image", self)
        open_template_img_action.triggered.connect(self.open_template_image)
        self.menuBar().addAction(open_template_img_action)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_query_pose2d)
        self.menuBar().addAction(save_action)

        save_as_action = QAction("Save As", self)
        save_as_action.triggered.connect(self.save_as_query_pose2d)
        self.menuBar().addAction(save_as_action)

        self.next_key = next_key
        self.prev_key = prev_key
        self.save_key = save_key

        if name_query_pose2d_file is not None:
            self._get_query_pose2d_filepath = name_query_pose2d_file
        else:
            self._get_query_pose2d_filepath = self._get_pose2d_filepath

        if name_template_pose2d_file is not None:
            self._get_template_pose2d_filepath = name_template_pose2d_file
        else:
            self._get_template_pose2d_filepath = self._get_pose2d_filepath

    def open_query_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Query Image", "", "Images (*.png *.jpg)"
        )
        if filename:
            self.query_list = [filename]
            self.cur_idx = 0
            self._load_query()

    def open_query_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Query Folder", "")
        if folder:
            self.query_list = [str(p) for p in Path(folder).glob("*.png")]
            self.cur_idx = 0
            self._load_query()

    def open_template_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Template Image", "", "Images (*.png *.jpg)"
        )
        if filename:
            self.template_path = filename
            self.template_pose2d_path = self._get_template_pose2d_filepath(filename)
            self._load_template()
            self._load_template_pose2d()
            self.status_bar.set_addition_message(f"Template: {self.template_path}")

    def _load_query(self):
        self.cur_query_path = self.query_list[self.cur_idx]

        query_pixmap = QPixmap(self.cur_query_path)
        self.fp_aligner.set_query_pixmap(query_pixmap)

        width, height = self.fp_aligner.width(), self.fp_aligner.height()
        self.resize(int(width) + 20, int(height) + 100)

        self.status_bar.set_message(f"Image: {self.cur_query_path}")
        self.status_bar.set_idx_indicator(self.cur_idx, len(self.query_list))

    def _load_template(self):
        template_pixmap = QPixmap(self.template_path)
        self.fp_aligner.set_template_pixmap(template_pixmap)

        width, height = self.fp_aligner.width(), self.fp_aligner.height()
        self.resize(int(width) + 20, int(height) + 100)

    def _load_template_pose2d(self):
        if Path(self.template_pose2d_path).exists():
            with open(self.template_pose2d_path, "r") as f:
                self.template_pose2d = [float(x) for x in f.readline().split()]
        else:
            raise FileNotFoundError(f"{self.template_pose2d_path} not found.")

    def _next_query(self):
        if hasattr(self, "cur_idx") and hasattr(self, "query_list"):
            if self.cur_idx == len(self.query_list) - 1:
                self.cur_idx = 0
            else:
                self.cur_idx += 1
            self._load_query()

    def _prev_query(self):
        if hasattr(self, "cur_idx") and hasattr(self, "query_list"):
            if self.cur_idx == 0:
                self.cur_idx = len(self.query_list) - 1
            else:
                self.cur_idx -= 1
            self._load_query()

    def _get_pose2d_filepath(self, img_path):
        return str(Path(img_path).with_suffix(".pose2d"))

    def save_query_pose2d(self):
        if self.fp_aligner.is_aligned():
            transfrom_params = self.fp_aligner.get_query_transform()
            pose2d = self._compute_query_pose2d(*transfrom_params)
            save_path = self._get_query_pose2d_filepath(self.cur_query_path)
            self._write_pose2d(pose2d, save_path)
            self.status_bar.set_message(f"Saved to: {save_path}")

    def save_as_query_pose2d(self):
        if self.fp_aligner.is_aligned():
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Pose2D", "", "Pose2D (*.pose2d)"
            )
            if filename:
                tx, ty, rot = self.fp_aligner.get_query_transform()
                pose2d = self._compute_query_pose2d(tx, ty, rot)
                self._write_pose2d(pose2d, filename)
                self.status_bar.set_message(f"Saved to: {filename}")

    def _write_pose2d(self, pose2d, filepath):
        with open(filepath, "w") as f:
            f.write(f"{pose2d[0]} {pose2d[1]} {pose2d[2]}")

    def _compute_query_pose2d(self, tx, ty, rot, rot_dx, rot_dy):
        rd = np.deg2rad(rot)
        R = np.array([[np.cos(rd), -np.sin(rd)], [np.sin(rd), np.cos(rd)]])
        t1 = np.array([rot_dx, rot_dy])
        t2 = np.array([tx, ty])
        t = -R @ t1 + t1 + t2
        affine_mat = np.hstack([R, t.reshape(2, 1)])
        affine_mat = np.vstack([affine_mat, [0, 0, 1]])
        affine_mat = np.linalg.inv(affine_mat)
        point = np.array([*self.template_pose2d[:2], 1])
        cx, cy = np.matmul(affine_mat, point)[:2]
        angle = rot + self.template_pose2d[2]
        return cx, cy, angle

    def keyPressEvent(self, event):
        if event.key() == self.next_key:
            self._next_query()
        elif event.key() == self.prev_key:
            self._prev_query()
        elif event.key() == self.save_key:
            self.save_query_pose2d()
        else:
            super().keyPressEvent(event)


class FingerprintManualAligner(QGraphicsScene):
    def __init__(
        self,
        query_color=(255, 0, 0, 255),
        template_color=(0, 0, 255, 255),
        grayscale_th=250,
        rotation_key=Qt.RightButton,
        rotation_speed=1.0,
        move_up_key=Qt.Key_Up,
        move_down_key=Qt.Key_Down,
        move_left_key=Qt.Key_Left,
        move_right_key=Qt.Key_Right,
        key_move_step=1,
    ):
        super().__init__()

        self.query_color = query_color
        self.template_color = template_color
        self.grayscale_th = grayscale_th
        self.rotation_key = rotation_key
        self.rotation_speed = rotation_speed
        self.move_up_key = move_up_key
        self.move_down_key = move_down_key
        self.move_left_key = move_left_key
        self.move_right_key = move_right_key
        self.key_move_step = key_move_step

    def set_query_pixmap(self, query_pixmap):
        self._safe_remove_item("query_item")
        color_query_pixmap, center = self._color_ridge(query_pixmap, self.query_color)
        self.query_item = DraggablePixmapItem(
            color_query_pixmap,
            transform_origin=center,
            rotation_key=self.rotation_key,
            rotation_speed=self.rotation_speed,
        )
        self.addItem(self.query_item)
        self.query_item.setZValue(1)
        self.query_item.setFocus()

    def set_template_pixmap(self, template_pixmap):
        self._safe_remove_item("template_item")
        color_template_pixmap, _ = self._color_ridge(
            template_pixmap, self.template_color
        )
        self.template_item = QGraphicsPixmapItem(color_template_pixmap)
        self.addItem(self.template_item)
        self.template_item.setZValue(0)

    def _has_item(self, name):
        return hasattr(self, name) and getattr(self, name) in self.items()

    def _safe_remove_item(self, name):
        if self._has_item(name):
            self.removeItem(getattr(self, name))

    def _color_ridge(self, pixmap, color):
        img = self._pixmap_to_array(pixmap)
        ridge_mask = np.where(img < self.grayscale_th, 255, 0).astype(np.uint8)
        center = self._compute_mask_center(ridge_mask)
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
        color_img[ridge_mask == 0] = (255, 255, 255, 1)
        color_img[ridge_mask > 0] = color
        color_pixmap = self._array_to_pixmap(color_img)
        return color_pixmap, center

    def _compute_mask_center(self, mask):
        M = cv2.moments(mask)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        return center_x, center_y

    def _pixmap_to_array(self, pixmap):
        qimage = pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
        width, height = qimage.width(), qimage.height()
        bytes_per_line = qimage.bytesPerLine()
        ptr = qimage.bits()
        ptr.setsize(height * bytes_per_line)
        array = np.frombuffer(ptr, np.uint8).reshape((height, bytes_per_line))
        array = array[:, :width]
        return array

    def _array_to_pixmap(self, array):
        height, width = array.shape[:2]
        image = QImage(array.data, width, height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(image)
        return pixmap

    def get_query_transform(self):
        tx, ty = self.query_item.pos().x(), self.query_item.pos().y()
        rot = self.query_item.rotation()
        rot_dx, rot_dy = self.query_item.origin_x, self.query_item.origin_y
        return tx, ty, rot, rot_dx, rot_dy

    def is_aligned(self):
        return self._has_item("query_item") and self._has_item("template_item")

    def keyPressEvent(self, event):
        if event.key() == self.move_up_key:
            if self._has_item("query_item"):
                self.query_item.moveBy(0, -self.key_move_step)
        elif event.key() == self.move_down_key:
            if self._has_item("query_item"):
                self.query_item.moveBy(0, self.key_move_step)
        elif event.key() == self.move_left_key:
            if self._has_item("query_item"):
                self.query_item.moveBy(-self.key_move_step, 0)
        elif event.key() == self.move_right_key:
            if self._has_item("query_item"):
                self.query_item.moveBy(self.key_move_step, 0)
        else:
            super().keyPressEvent(event)


class DraggablePixmapItem(QGraphicsPixmapItem):
    def __init__(
        self,
        pixmap,
        transform_origin=(0, 0),
        rotation_key=Qt.RightButton,
        rotation_speed=1.0,
    ):
        super().__init__(pixmap)
        self.setFlag(self.ItemIsMovable)
        self.setFlag(self.ItemIsSelectable)

        self.origin_x, self.origin_y = transform_origin
        self.setTransformOriginPoint(*transform_origin)

        self.cur_rot_angle = 0
        self.rotation_key = rotation_key
        self.rot_speed = rotation_speed

    def mousePressEvent(self, event):
        if event.button() == self.rotation_key:
            x, y = event.pos().x(), event.pos().y()
            x, y = x - self.origin_x, y - self.origin_y
            self.ref_angle = np.rad2deg(np.arctan2(y, x))
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == self.rotation_key:
            x, y = event.pos().x(), event.pos().y()
            x, y = x - self.origin_x, y - self.origin_y
            cur_angle = np.rad2deg(np.arctan2(y, x))
            angle_diff = self._diff_angle(self.ref_angle, cur_angle)
            self.cur_rot_angle += angle_diff * self.rot_speed
            self.cur_rot_angle %= 360
            self.setRotation(self.cur_rot_angle)
            self.ref_angle = cur_angle
        else:
            super().mouseMoveEvent(event)

    def _diff_angle(self, angle_from, angle_to):
        return (angle_to - angle_from + 180) % 360 - 180


class CustomStatusBar(QStatusBar):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_message = ""
        self.addition_message = ""
        self.message = QLabel()
        self.idx_indicator = QLabel()
        self.addPermanentWidget(self.message, stretch=1)
        self.addPermanentWidget(self.idx_indicator)

    def set_message(self, msg):
        self.main_message = msg
        self._set_all_message()

    def set_addition_message(self, msg):
        self.addition_message = msg
        self._set_all_message()

    def _set_all_message(self):
        msg = f"{self.main_message}\n{self.addition_message}"
        self.message.setText(msg)

    def set_idx_indicator(self, idx, list_len):
        self.idx_indicator.setText(f"[{idx+1}/{list_len}]")


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = Pose2DAlignmentLabeler()
    window.show()
    sys.exit(app.exec_())