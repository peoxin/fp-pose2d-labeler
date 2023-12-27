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
        viewer_scale=1.0,
        query_color=(255, 0, 0, 150),
        template_color=(0, 0, 255, 255),
        grayscale_th=250,
        rotation_key=Qt.RightButton,
        rotation_speed=0.8,
        rot_counterclockwise_key=Qt.Key_Q,
        rot_clockwise_key=Qt.Key_E,
        key_rotate_step=1,
        move_up_key=Qt.Key_Up,
        move_down_key=Qt.Key_Down,
        move_left_key=Qt.Key_Left,
        move_right_key=Qt.Key_Right,
        key_move_step=1,
        next_key=Qt.Key_D,
        prev_key=Qt.Key_A,
        save_key=Qt.Key_S,
        save_key_modifier=Qt.ControlModifier,
        template_calibrated=False,
        name_query_pose2d_file: Callable[[str], str] = None,
        name_template_pose2d_file: Callable[[str], str] = None,
        enable_auto_find_template=False,
        template_auto_finder: Callable[[str], str] = None,
        default_query_pattern: str = "*.png",
    ):
        super().__init__()

        self.setWindowTitle("2DPose Alignment Labeler")
        self.resize(1200, 800)

        self.graphic_view = QGraphicsView(self)
        self.fp_aligner = FingerprintManualAligner(
            query_color=query_color,
            template_color=template_color,
            grayscale_th=grayscale_th,
            rotation_key=rotation_key,
            rotation_speed=rotation_speed,
            rot_counterclockwise_key=rot_counterclockwise_key,
            rot_clockwise_key=rot_clockwise_key,
            key_rotate_step=key_rotate_step,
            move_up_key=move_up_key,
            move_down_key=move_down_key,
            move_left_key=move_left_key,
            move_right_key=move_right_key,
            key_move_step=key_move_step,
        )
        self.graphic_view.setScene(self.fp_aligner)
        self.viewer_scale = viewer_scale
        self.graphic_view.scale(viewer_scale, viewer_scale)
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
        self.save_key_modifier = save_key_modifier

        if name_query_pose2d_file is not None:
            self._get_query_pose2d_file = name_query_pose2d_file
        else:
            self._get_query_pose2d_file = self._get_pose2d_filepath

        if name_template_pose2d_file is not None:
            self._get_template_pose2d_file = name_template_pose2d_file
        else:
            self._get_template_pose2d_file = self._get_pose2d_filepath

        if not enable_auto_find_template:
            self._get_template_from_query = None
        elif template_auto_finder is not None:
            self._get_template_from_query = template_auto_finder

        self.template_calibrated = template_calibrated
        self.default_query_pattern = default_query_pattern

    def open_query_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Query Image", "", "Images (*.png *.jpg *bmp)"
        )
        if filename:
            self.query_list = [filename]
            self.cur_idx = 0
            self._load_query()

    def open_query_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Query Folder", "")
        if folder:
            self.query_list = [
                str(p) for p in Path(folder).glob(self.default_query_pattern)
            ]
            if len(self.query_list) == 0:
                self.status_bar.set_message("No image found.")
                return
            self.cur_idx = 0
            self._load_query()

    def open_template_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Template Image", "", "Images (*.png *.jpg *bmp)"
        )
        if filename:
            self._open_template(filename)

    def _auto_open_template(self):
        if self._get_template_from_query is not None:
            try:
                template_path = self._get_template_from_query(self.cur_query_path)
                if template_path is not None and Path(template_path).exists():
                    self._open_template(template_path)
            except Exception as e:
                print(e)

    def _open_template(self, filepath):
        self.template_path = filepath
        self._load_template()
        if not self.template_calibrated:
            self.template_pose2d_path = self._get_template_pose2d_file(filepath)
            self._load_template_pose2d()
        else:
            self._set_calibrated_template_pose2d()

        self._transform_query_by_pose2d()
        self.status_bar.set_addition_message(f"Template: {self.template_path}")

    def _load_query(self):
        self.cur_query_path = self.query_list[self.cur_idx]
        self.cur_query_pose2d_path = self._get_query_pose2d_file(self.cur_query_path)

        query_img = cv2.imread(self.cur_query_path, cv2.IMREAD_GRAYSCALE)
        self.fp_aligner.set_query_image(query_img)
        self._load_query_pose2d()
        self._transform_query_by_pose2d()

        if self._get_template_from_query is not None:
            self._auto_open_template()

        self._resize_window()

        self.status_bar.set_message(f"Image: {self.cur_query_path}")
        self.status_bar.set_idx_indicator(self.cur_idx, len(self.query_list))

    def _load_template(self):
        self.template_img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
        self.fp_aligner.set_template_image(self.template_img)

        self._resize_window()

    def _load_query_pose2d(self):
        if Path(self.cur_query_pose2d_path).exists():
            with open(self.cur_query_pose2d_path, "r") as f:
                self.query_pose2d = [float(x) for x in f.readline().split()]
                self.status_bar.set_tool_message("pose2d exists")
        else:
            self.query_pose2d = None
            self.status_bar.clear_tool_message()

    def _load_template_pose2d(self):
        if Path(self.template_pose2d_path).exists():
            with open(self.template_pose2d_path, "r") as f:
                self.template_pose2d = [float(x) for x in f.readline().split()]
        else:
            raise FileNotFoundError(f"{self.template_pose2d_path} not found.")

    def _set_calibrated_template_pose2d(self):
        cx = float(self.template_img.shape[1] // 2)
        cy = float(self.template_img.shape[0] // 2)
        angle = 0.0
        self.template_pose2d = [cx, cy, angle]

    def _transform_query_by_pose2d(self):
        if self._has_var("query_pose2d") and self._has_var("template_pose2d"):
            rot_center = self.fp_aligner.get_query_rot_center()
            transfrom_params = self._get_transform_from_pose2d(rot_center)
            self.fp_aligner.transform_query_item(*transfrom_params)

    def _next_query(self):
        if self._has_var("cur_idx") and self._has_var("query_list"):
            if self.cur_idx == len(self.query_list) - 1:
                self.cur_idx = 0
            else:
                self.cur_idx += 1
            self._load_query()

    def _prev_query(self):
        if self._has_var("cur_idx") and self._has_var("query_list"):
            if self.cur_idx == 0:
                self.cur_idx = len(self.query_list) - 1
            else:
                self.cur_idx -= 1
            self._load_query()

    def _get_pose2d_filepath(self, img_path):
        return str(Path(img_path).with_suffix(".pose2d"))

    def _get_template_from_query(self, query_path):
        filepath = Path(query_path).with_suffix(".template")
        if filepath.exists():
            with open(filepath, "r") as f:
                return f.readline().strip()
        return None

    def save_query_pose2d(self):
        if self.fp_aligner.is_aligned():
            transfrom_params = self.fp_aligner.get_query_transform()
            pose2d = self._compute_query_pose2d(*transfrom_params)
            save_path = self._get_query_pose2d_file(self.cur_query_path)
            self._write_pose2d(pose2d, save_path)
            self.status_bar.set_message(f"Saved to: {save_path}")
            self.status_bar.set_tool_message("pose2d exists")

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
        rr = np.deg2rad(rot)
        R = np.array([[np.cos(rr), -np.sin(rr)], [np.sin(rr), np.cos(rr)]])
        t1 = np.array([rot_dx, rot_dy])
        t2 = np.array([tx, ty])
        t = -R @ t1 + t1 + t2
        affine_mat = np.hstack([R, t.reshape(2, 1)])
        affine_mat = np.vstack([affine_mat, [0, 0, 1]])
        affine_mat = np.linalg.inv(affine_mat)
        point = np.array([*self.template_pose2d[:2], 1])
        cx, cy = np.matmul(affine_mat, point)[:2]
        cx, cy = np.round(cx), np.round(cy)
        angle = (rot + self.template_pose2d[2] + 180) % 360 - 180
        return cx, cy, angle

    def _get_transform_from_pose2d(self, query_rot_center):
        query_cx, query_cy, query_angle = self.query_pose2d
        query_c = np.array([query_cx, query_cy])
        template_cx, template_cy, template_angle = self.template_pose2d
        template_c = np.array([template_cx, template_cy])
        rot_c = np.array(query_rot_center)

        rd = (query_angle - template_angle + 180) % 360 - 180
        rr = np.deg2rad(rd)
        R = np.array([[np.cos(rr), -np.sin(rr)], [np.sin(rr), np.cos(rr)]])
        tx, ty = template_c - R @ (query_c - rot_c) - rot_c
        return tx, ty, rd

    def keyPressEvent(self, event):
        if event.key() == self.next_key:
            self._next_query()
        elif event.key() == self.prev_key:
            self._prev_query()
        elif (
            event.key() == self.save_key and event.modifiers() == self.save_key_modifier
        ):
            self.save_query_pose2d()
        else:
            super().keyPressEvent(event)

    def _resize_window(self):
        bound_rect = self.fp_aligner.itemsBoundingRect()
        width, height = bound_rect.width(), bound_rect.height()
        width, height = width * self.viewer_scale, height * self.viewer_scale
        self.resize(int(width) + 20, int(height) + 100)

    def _has_var(self, name):
        return hasattr(self, name) and getattr(self, name) is not None


class FingerprintManualAligner(QGraphicsScene):
    def __init__(
        self,
        query_color=(255, 0, 0, 255),
        template_color=(0, 0, 255, 255),
        grayscale_th=250,
        rotation_key=Qt.RightButton,
        rotation_speed=1.0,
        rot_counterclockwise_key=Qt.Key_Q,
        rot_clockwise_key=Qt.Key_E,
        key_rotate_step=1,
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
        self.rot_counterclockwise_key = rot_counterclockwise_key
        self.rot_clockwise_key = rot_clockwise_key
        self.key_rotate_step = key_rotate_step
        self.move_up_key = move_up_key
        self.move_down_key = move_down_key
        self.move_left_key = move_left_key
        self.move_right_key = move_right_key
        self.key_move_step = key_move_step

    def set_query_image(self, query_img):
        self._safe_remove_item("query_item")
        color_query_img, center = self._color_ridge(query_img, self.query_color)
        color_query_pixmap = self._array_to_pixmap(color_query_img)
        self.query_item = DraggablePixmapItem(
            color_query_pixmap,
            transform_origin=center,
            rotation_key=self.rotation_key,
            rotation_speed=self.rotation_speed,
        )
        self.addItem(self.query_item)
        self.query_item.setZValue(1)
        self.query_item.setFocus()

    def set_template_image(self, template_img):
        self._safe_remove_item("template_item")
        color_template_img, _ = self._color_ridge(template_img, self.template_color)
        color_template_pixmap = self._array_to_pixmap(color_template_img)
        self.template_item = QGraphicsPixmapItem(color_template_pixmap)
        self.addItem(self.template_item)
        self.template_item.setZValue(0)

    def get_query_rot_center(self):
        if self._has_item("query_item"):
            return self.query_item.origin_x, self.query_item.origin_y
        return None

    def transform_query_item(self, tx, ty, rot):
        self.query_item.setPos(tx, ty)
        self.query_item.set_rotation(rot)

    def _has_item(self, name):
        return hasattr(self, name) and getattr(self, name) in self.items()

    def _safe_remove_item(self, name):
        if self._has_item(name):
            self.removeItem(getattr(self, name))

    def _color_ridge(self, img, color):
        ridge_mask = np.where(img < self.grayscale_th, 255, 0).astype(np.uint8)
        center = self._compute_mask_center(ridge_mask)
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
        color_img[ridge_mask == 0] = (255, 255, 255, 1)
        color_img[ridge_mask > 0] = color
        return color_img, center

    def _compute_mask_center(self, mask):
        M = cv2.moments(mask)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        return center_x, center_y

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
        elif event.key() == self.rot_counterclockwise_key:
            if self._has_item("query_item") and not self.query_item.is_mouse_rotating:
                cur_rotation = self.query_item.rotation()
                self.query_item.setRotation(cur_rotation - self.key_rotate_step)
        elif event.key() == self.rot_clockwise_key:
            if self._has_item("query_item") and not self.query_item.is_mouse_rotating:
                cur_rotation = self.query_item.rotation()
                self.query_item.setRotation(cur_rotation + self.key_rotate_step)
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
        self.is_mouse_rotating = False
        self.rotation_key = rotation_key
        self.rot_speed = rotation_speed

    def set_rotation(self, angle):
        self.cur_rot_angle = angle
        self.setRotation(angle)

    def mousePressEvent(self, event):
        if event.button() == self.rotation_key:
            x, y = event.pos().x(), event.pos().y()
            x, y = x - self.origin_x, y - self.origin_y
            self.ref_angle = np.rad2deg(np.arctan2(y, x))
            self.is_mouse_rotating = True
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

    def mouseReleaseEvent(self, event):
        if event.button() == self.rotation_key:
            self.is_mouse_rotating = False
        else:
            super().mouseReleaseEvent(event)

    def _diff_angle(self, angle_from, angle_to):
        return (angle_to - angle_from + 180) % 360 - 180


class CustomStatusBar(QStatusBar):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_message = ""
        self.addition_message = ""
        self.tool_message = ""
        self.message = QLabel()
        self.tool = QLabel()
        self.idx_indicator = QLabel()
        self.addPermanentWidget(self.message, stretch=1)
        self.addPermanentWidget(self.tool)
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

    def set_tool_message(self, msg):
        self.tool_message = msg
        self.tool.setText(msg)

    def clear_tool_message(self):
        self.tool_message = ""
        self.tool.setText("")

    def set_idx_indicator(self, idx, list_len):
        self.idx_indicator.setText(f"[{idx+1}/{list_len}]")


if __name__ == "__main__":
    from argparse import ArgumentParser
    import sys

    parser = ArgumentParser()
    parser.add_argument("-q", "--query-pattern", type=str, default="*.png")
    parser.add_argument("-c", "--template-calibrated", action="store_true")
    parser.add_argument("-a", "--auto-find-template", action="store_true")
    args = parser.parse_args()

    def find_template_path(query_path):
        src_root_dir = r"D:\peoxin\Projects\oxi_dataset\oxi"
        log_root_dir = r"D:\peoxin\Projects\oxi_dataset\oxi_pose2d"
        relative_path = Path(query_path).relative_to(src_root_dir)
        log_path = Path(log_root_dir) / relative_path
        log_path = log_path.with_suffix(".txt")
        with open(log_path, "r") as f:
            line = f.readline().strip()
            name, score, _, _, _ = line.split()
        print(name, score)

        template_root_dir = r"D:\peoxin\Projects\oxi_dataset\roll"
        person = relative_path.parents[1].name
        template_path = Path(template_root_dir) / person / f"{name}.bmp"
        return str(template_path)

    def find_calibrated_template_path(query_path):
        src_root_dir = r"D:\peoxin\Projects\oxi_dataset\oxi"
        log_root_dir = r"D:\peoxin\Projects\oxi_dataset\new_oxi_pose2d"
        # src_root_dir = r"D:\peoxin\Projects\oxi_dataset\oxi-copy-selected"
        # log_root_dir = r"D:\peoxin\Projects\oxi_dataset\oxi_pose2d"

        relative_path = Path(query_path).relative_to(src_root_dir)
        log_path = Path(log_root_dir) / relative_path
        log_path = log_path.with_suffix(".txt")
        with open(log_path, "r") as f:
            line = f.readline().strip()
            name, score, _, _, _ = line.split()
        print(name, score)

        template_root_dir = r"D:\peoxin\Projects\oxi_dataset\roll_calibrated"
        # template_root_dir = r"D:\peoxin\Projects\oxi_dataset\roll_rot1024"
        person = relative_path.parents[1].name
        template_path = Path(template_root_dir) / person / f"{name}.bmp"
        return str(template_path)

    app = QApplication(sys.argv)
    window = Pose2DAlignmentLabeler(
        viewer_scale=1.0,
        grayscale_th=240,
        template_calibrated=args.template_calibrated,
        enable_auto_find_template=args.auto_find_template,
        template_auto_finder=find_calibrated_template_path,
        default_query_pattern=args.query_pattern,
    )
    window.show()
    sys.exit(app.exec_())
