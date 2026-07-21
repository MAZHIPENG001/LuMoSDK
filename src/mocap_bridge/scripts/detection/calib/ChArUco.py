#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time as TimeMsg
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from device.realsense_camera import RealSenseCamera


class CharucoDetectorNode(Node):
    def __init__(self):
        super().__init__('charuco_detector_node')

        self.squares_x = 7  # 标定板 X 方向（列）的棋盘格方块数量
        self.squares_y = 5  # 标定板 Y 方向（行）的棋盘格方块数量

        # 单位必须是米 (m)！例如 40mm 就是 0.040
        self.square_length = 0.01812  # 棋盘格方块的物理边长
        self.marker_length = 0.01446  # 内部 ArUco 码的物理边长

        # ==========================================

        # 实例化并启动 RealSense 相机
        self.camera = RealSenseCamera(width=640, height=480, fps=60)
        if not self.camera.start():
            self.get_logger().error("相机启动失败，请检查连接！")
            raise SystemExit

        # 获取相机内参矩阵
        self.camera_matrix = self.camera.get_color_intrinsic_matrix()
        self.dist_coeffs = self.camera.get_color_distortion_coeffs()

        # 1. 初始化 ArUco 字典
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

        # 2. 创建 ChArUco 标定板对象
        self.board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length,
            self.marker_length,
            self.aruco_dict
        )

        # 3. 初始化检测器
        self.detector_params = cv2.aruco.DetectorParameters()
        self.charuco_params = cv2.aruco.CharucoParameters()

        # 使用 OpenCV 4.7+ 的 CharucoDetector (集成了 Aruco 基础检测)
        self.charuco_detector = cv2.aruco.CharucoDetector(
            self.board, self.charuco_params, self.detector_params
        )

        # 创建发布者 (话题名保持不变，兼容手眼标定节点)
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco_pose', 10)
        self.timer = self.create_timer(1.0 / 60.0, self.process_frame)

        self.get_logger().info("ChArUco 检测节点已启动，正在发布 /aruco_pose")

    def process_frame(self):
        color_image, _, frame_metadata = self.camera.get_images(
            return_metadata=True
        )
        if color_image is None:
            return

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # 1. 检测 ChArUco 标定板的角点和 ID
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)

        # 2. 如果检测到了基础的 ArUco 码，可以先画出来 (用于调试)
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(color_image, marker_corners, marker_ids)

        # 3. 如果成功插值出足够的 ChArUco 内部棋盘格角点 (至少需要 4 个角点来解算 3D 姿态)
        if charuco_ids is not None and len(charuco_ids) >= 4:
            # 绘制亚像素级别的 ChArUco 角点
            cv2.aruco.drawDetectedCornersCharuco(color_image, charuco_corners, charuco_ids, (0, 255, 0))

            # 获取对应的 3D 物理坐标和 2D 图像坐标
            obj_points, img_points = self.board.matchImagePoints(charuco_corners, charuco_ids)

            # 4. 解算 3D 位姿
            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                img_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE  # 多点解算使用迭代法最稳定
            )

            if success:
                # 绘制 3D 坐标轴 (原点在标定板左下角)
                cv2.drawFrameAxes(color_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.square_length * 2)

                # 发布位姿
                self.publish_pose(
                    rvec, tvec, frame_metadata['capture_time_ns']
                )
        else:
            pass
        cv2.imshow('ChArUco Detection', color_image)
        cv2.waitKey(1)

    def publish_pose(self, rvec, tvec, capture_time_ns):
        # print(f"\33[92mtvec:{tvec}\33[0m")
        msg = PoseStamped()
        if capture_time_ns is None:
            msg.header.stamp = self.get_clock().now().to_msg()
        else:
            msg.header.stamp = TimeMsg(
                sec=int(capture_time_ns // 1_000_000_000),
                nanosec=int(capture_time_ns % 1_000_000_000),
            )
        msg.header.frame_id = 'camera_color_optical_frame'

        msg.pose.position.x = float(tvec[0][0])
        msg.pose.position.y = float(tvec[1][0])
        msg.pose.position.z = float(tvec[2][0])

        r_matrix, _ = cv2.Rodrigues(rvec)
        quat = R.from_matrix(r_matrix).as_quat()

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
    node = CharucoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
