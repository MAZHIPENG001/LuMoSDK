#!/usr/bin/env python3
"""检测 ChArUco 板并发布板坐标系到彩色相机坐标系的位姿。

发布的 PoseStamped 表示 ^camera T_target（target -> camera），这正是
cv2.calibrateHandEye 的 R_target2cam / t_target2cam 输入。
"""

import os
import sys

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 部署时请将修正版相机文件放在 device/realsense_camera.py。
from device.realsense_camera import RealSenseCamera


class CharucoDetectorNode(Node):
    def __init__(self):
        super().__init__('charuco_detector_node')

        # 这些参数必须与实际打印板完全一致；长度单位为米。
        self.declare_parameter('squares_x', 7)
        self.declare_parameter('squares_y', 5)
        self.declare_parameter('square_length_m', 0.01812)
        self.declare_parameter('marker_length_m', 0.01446)
        self.declare_parameter('legacy_pattern', False)
        self.declare_parameter('min_charuco_corners', 6)
        self.declare_parameter('max_reprojection_error_px', 1.5)
        self.declare_parameter('camera_serial', '')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 60)

        self.squares_x = int(self.get_parameter('squares_x').value)
        self.squares_y = int(self.get_parameter('squares_y').value)
        self.square_length = float(
            self.get_parameter('square_length_m').value
        )
        self.marker_length = float(
            self.get_parameter('marker_length_m').value
        )
        self.min_charuco_corners = int(
            self.get_parameter('min_charuco_corners').value
        )
        self.max_reprojection_error_px = float(
            self.get_parameter('max_reprojection_error_px').value
        )
        self.fps = int(self.get_parameter('fps').value)

        if self.squares_x < 2 or self.squares_y < 2:
            raise ValueError('squares_x 和 squares_y 必须至少为 2')
        if not 0.0 < self.marker_length < self.square_length:
            raise ValueError('必须满足 0 < marker_length < square_length')
        if self.min_charuco_corners < 4:
            raise ValueError('min_charuco_corners 不能小于 4')

        serial = str(self.get_parameter('camera_serial').value).strip()
        self.camera = RealSenseCamera(
            width=int(self.get_parameter('width').value),
            height=int(self.get_parameter('height').value),
            fps=self.fps,
            serial_number=serial or None,
        )
        if not self.camera.start():
            self.get_logger().error('相机启动失败，请检查连接和流配置。')
            raise RuntimeError('RealSense camera start failed')

        try:
            self.initialize_detector()
        except Exception:
            # __init__ 失败时 main() 得不到 node 对象，必须在这里主动关闭
            # RealSense；否则 SDK 后台线程会在解释器退出时触发 abort。
            self.camera.stop()
            raise

    def initialize_detector(self):
        # K、D 必须来自当前正在使用的彩色流分辨率。
        self.camera_matrix = self.camera.get_color_intrinsic_matrix().astype(
            np.float64
        )
        distortion_model = self.camera.get_color_distortion_model()
        self.use_undistorted_pnp_points = (
            not self.camera.color_distortion_is_opencv_compatible()
        )
        if self.use_undistorted_pnp_points:
            if 'inverse_brown_conrady' not in str(distortion_model):
                raise RuntimeError(
                    f'暂不支持彩色流畸变模型: {distortion_model}'
                )
            # 角点将由 RealSense 转换到使用同一 K 的无畸变虚拟图像，
            # 因此 solvePnP 使用全零 D。
            self.dist_coeffs = np.zeros((1, 5), dtype=np.float64)
            self.get_logger().warn(
                '彩色流为 inverse_brown_conrady：将先用 RealSense 反投影'
                '矫正 ChArUco 角点，再使用 K + 零畸变进行 PnP。'
            )
        else:
            self.dist_coeffs = (
                self.camera.get_color_distortion_coeffs().astype(np.float64)
            )

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_6X6_250
        )
        self.board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length,
            self.marker_length,
            self.aruco_dict,
        )
        if bool(self.get_parameter('legacy_pattern').value):
            self.board.setLegacyPattern(True)

        self.detector_params = cv2.aruco.DetectorParameters()
        # ChArUco 内角点会在 detectBoard 内部做亚像素细化；不强制对 ArUco
        # 角点做 sub-pixel，避免黑白棋盘边缘把 marker 角点拉偏。
        self.detector_params.cornerRefinementMethod = (
            cv2.aruco.CORNER_REFINE_NONE
        )

        self.charuco_params = cv2.aruco.CharucoParameters()
        if not self.use_undistorted_pnp_points:
            self.charuco_params.cameraMatrix = self.camera_matrix
            self.charuco_params.distCoeffs = self.dist_coeffs
        self.charuco_params.tryRefineMarkers = True
        self.charuco_detector = cv2.aruco.CharucoDetector(
            self.board,
            self.charuco_params,
            self.detector_params,
        )

        self.pose_pub = self.create_publisher(PoseStamped, '/aruco_pose', 10)
        self.timer = self.create_timer(1.0 / self.fps, self.process_frame)
        self.get_logger().info(
            'ChArUco 检测已启动；/aruco_pose 表示 target -> camera。'
        )

    def process_frame(self):
        color_image, _, frame_metadata = self.camera.get_images(
            return_metadata=True
        )
        if color_image is None or frame_metadata is None:
            return

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        (
            charuco_corners,
            charuco_ids,
            marker_corners,
            marker_ids,
        ) = self.charuco_detector.detectBoard(gray)

        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(
                color_image, marker_corners, marker_ids
            )

        pose = self.estimate_board_pose(charuco_corners, charuco_ids)
        if charuco_ids is not None and len(charuco_ids) > 0:
            cv2.aruco.drawDetectedCornersCharuco(
                color_image,
                charuco_corners,
                charuco_ids,
                (0, 255, 0),
            )

        if pose is not None:
            rvec, tvec, reprojection_rmse = pose
            self.draw_pose_axes(color_image, rvec, tvec)
            cv2.putText(
                color_image,
                f'PnP RMSE: {reprojection_rmse:.2f} px',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            self.publish_pose(
                rvec,
                tvec,
                frame_metadata.get('capture_time_ns'),
            )

        cv2.imshow('ChArUco Detection', color_image)
        cv2.waitKey(1)

    def estimate_board_pose(self, charuco_corners, charuco_ids):
        """返回 (rvec, tvec, reprojection_rmse)，不合格帧返回 None。"""
        if charuco_ids is None or charuco_corners is None:
            return None
        if len(charuco_ids) < self.min_charuco_corners:
            return None
        if self.board.checkCharucoCornersCollinear(charuco_ids):
            return None

        obj_points, img_points = self.board.matchImagePoints(
            charuco_corners, charuco_ids
        )
        obj_points = np.asarray(obj_points, dtype=np.float64).reshape(-1, 3)
        img_points = np.asarray(img_points, dtype=np.float64).reshape(-1, 2)
        if self.use_undistorted_pnp_points:
            pnp_img_points = self.camera.color_points_to_undistorted_pixels(
                img_points
            )
        else:
            pnp_img_points = img_points

        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            pnp_img_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not success:
            return None

        # 用 LM 对 IPPE 初值做最终非线性优化。
        if hasattr(cv2, 'solvePnPRefineLM'):
            rvec, tvec = cv2.solvePnPRefineLM(
                obj_points,
                pnp_img_points,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
            )

        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        if not np.all(np.isfinite(tvec)) or tvec[2, 0] <= 0.0:
            return None

        projected, _ = cv2.projectPoints(
            obj_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        projected = projected.reshape(-1, 2)
        reprojection_rmse = float(
            np.sqrt(
                np.mean(np.sum((projected - pnp_img_points) ** 2, axis=1))
            )
        )
        if (
            not np.isfinite(reprojection_rmse)
            or reprojection_rmse > self.max_reprojection_error_px
        ):
            return None

        return rvec, tvec, reprojection_rmse

    def draw_pose_axes(self, color_image, rvec, tvec):
        """在原始彩色图上绘制坐标轴，并正确处理 inverse 畸变。"""
        axis_length = self.square_length * 2.0
        if not self.use_undistorted_pnp_points:
            cv2.drawFrameAxes(
                color_image,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
                axis_length,
            )
            return

        axis_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [axis_length, 0.0, 0.0],
                [0.0, axis_length, 0.0],
                [0.0, 0.0, axis_length],
            ],
            dtype=np.float64,
        )
        undistorted_pixels, _ = cv2.projectPoints(
            axis_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        raw_pixels = self.camera.undistorted_color_points_to_raw_pixels(
            undistorted_pixels.reshape(-1, 2)
        )
        if not np.all(np.isfinite(raw_pixels)):
            return

        raw_pixels = np.rint(raw_pixels).astype(np.int32)
        origin = tuple(raw_pixels[0])
        # 与 cv2.drawFrameAxes 一致：X 红、Y 绿、Z 蓝。
        cv2.line(color_image, origin, tuple(raw_pixels[1]), (0, 0, 255), 2)
        cv2.line(color_image, origin, tuple(raw_pixels[2]), (0, 255, 0), 2)
        cv2.line(color_image, origin, tuple(raw_pixels[3]), (255, 0, 0), 2)

    def publish_pose(self, rvec, tvec, capture_time_ns):
        """发布 ^camera T_target。"""
        msg = PoseStamped()
        if capture_time_ns is None:
            msg.header.stamp = self.get_clock().now().to_msg()
        else:
            msg.header.stamp = TimeMsg(
                sec=int(capture_time_ns // 1_000_000_000),
                nanosec=int(capture_time_ns % 1_000_000_000),
            )
        msg.header.frame_id = 'camera_color_optical_frame'

        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        msg.pose.position.x = float(tvec[0])
        msg.pose.position.y = float(tvec[1])
        msg.pose.position.z = float(tvec[2])

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        quat = Rotation.from_matrix(rotation_matrix).as_quat()
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        self.pose_pub.publish(msg)

    def destroy_node(self):
        self.camera.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CharucoDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
