#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from mocap_bridge.msg import MocapData
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading
import sys
import select
import termios
import tty
import os


class HandEyeCalibrator(Node):
    def __init__(self):
        super().__init__('hand_eye_calibrator')

        # 数据存储
        self.R_gripper2base = []
        self.t_gripper2base = []
        self.R_target2cam = []
        self.t_target2cam = []

        # 缓存最新一帧的数据，用于手动触发保存
        self.latest_mocap_pose = None
        self.latest_aruco_pose = None

        # 订阅动捕话题 (对应相机刚体 rigid_id == 4)[cite: 1]
        self.mocap_sub = self.create_subscription(
            MocapData,
            '/mocap_data',
            self.mocap_callback,
            10
        )

        # 订阅 ArUco 姿态话题[cite: 1]
        self.aruco_sub = self.create_subscription(
            PoseStamped,
            '/aruco_pose',
            self.aruco_callback,
            10
        )

        self.get_logger().info("手眼标定节点已启动。[cite: 1]")
        self.get_logger().info("请在终端中按 's' 保存当前对齐的位姿，按 'c' 进行计算并退出。[cite: 1]")

        # 启动键盘监听线程[cite: 1]
        self.key_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.key_thread.start()

    def mocap_callback(self, msg):
        # 提取 rigid_id == 4 的相机刚体位姿[cite: 1]
        for rb in msg.rigid_bodies:
            if rb.rigid_id == 4 and rb.is_track:
                self.latest_mocap_pose = rb
                break

    def aruco_callback(self, msg):
        # 更新最新的 ArUco 姿态[cite: 1]
        self.latest_aruco_pose = msg.pose

    def keyboard_loop(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key.lower() == 's':
                        self.save_current_pose_pair()
                    elif key.lower() == 'c':
                        self.calculate_calibration()
                        os._exit(0)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def save_current_pose_pair(self):
        if self.latest_mocap_pose is None or self.latest_aruco_pose is None:
            self.get_logger().warn("数据未对齐或缺失，无法保存！请确保相机在动捕视野内且拍到了 ArUco。[cite: 1]")
            return

        m_pose = self.latest_mocap_pose
        a_pose = self.latest_aruco_pose

        # 1. 处理动捕数据 (Gripper to Base)[cite: 1]
        r_mocap = R.from_quat([m_pose.qx, m_pose.qy, m_pose.qz, m_pose.qw]).as_matrix()
        # 动捕数据除以 1000 转换为米[cite: 1]
        t_mocap = np.array([[m_pose.x / 1000.0],
                            [m_pose.y / 1000.0],
                            [m_pose.z / 1000.0]])

        # 2. 处理视觉数据 (Target to Camera)[cite: 1]
        r_cam = R.from_quat([
            a_pose.orientation.x, a_pose.orientation.y,
            a_pose.orientation.z, a_pose.orientation.w
        ]).as_matrix()
        # ArUco 视觉节点输出已经是米[cite: 1]
        t_cam = np.array([
            [a_pose.position.x], [a_pose.position.y], [a_pose.position.z]
        ])

        self.R_gripper2base.append(r_mocap)
        self.t_gripper2base.append(t_mocap)
        self.R_target2cam.append(r_cam)
        self.t_target2cam.append(t_cam)

        self.get_logger().info(
            f"✅ 成功保存第 {len(self.R_gripper2base)} 组数据！")

        # 清空缓存，防止在同一位置误触多次按键
        self.latest_mocap_pose = None
        self.latest_aruco_pose = None

    def calculate_calibration(self):
        if len(self.R_gripper2base) < 5:
            self.get_logger().error(f"数据量不足，仅有 {len(self.R_gripper2base)} 组。标定失败。")
            return

        self.get_logger().info("正在计算相机到动捕刚体的变换矩阵 (Hand-Eye Calibration)...")

        # 定义 OpenCV 支持的所有标定算法
        methods = {
            'Tsai-Lenz': cv2.CALIB_HAND_EYE_TSAI,
            'Park': cv2.CALIB_HAND_EYE_PARK,
            'Horaud': cv2.CALIB_HAND_EYE_HORAUD,
            'Andreff': cv2.CALIB_HAND_EYE_ANDREFF,
            'Daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS
        }

        print("\n" + "=" * 55)
        print("🎉 标定完成！(Camera -> Mocap Rigid Body) 结果对比")
        print("=" * 55)

        # 遍历每一种算法进行求解
        for name, method_flag in methods.items():
            try:
                R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                    R_gripper2base=self.R_gripper2base,
                    t_gripper2base=self.t_gripper2base,
                    R_target2cam=self.R_target2cam,
                    t_target2cam=self.t_target2cam,
                    method=method_flag
                )

                quat_cam2gripper = R.from_matrix(R_cam2gripper).as_quat()

                print(f"# 🟢 【算法: {name}】")
                print(f"# 平移向量 (X, Y, Z) 单位:米[cite: 1]")
                print(f"calib_t_x_m = {t_cam2gripper[0][0]:.4f}")
                print(f"calib_t_y_m = {t_cam2gripper[1][0]:.4f}")
                print(f"calib_t_z_m = {t_cam2gripper[2][0]:.4f}")
                print(f"# 旋转四元数 (x, y, z, w):[cite: 1]")
                print(
                    f"calib_qx, calib_qy, calib_qz, calib_qw = {quat_cam2gripper[0]:.4f}, {quat_cam2gripper[1]:.4f}, {quat_cam2gripper[2]:.4f}, {quat_cam2gripper[3]:.4f}")
                print("-" * 55)

            except cv2.error as e:
                # 某些算法在特定奇异数据下可能无解，增加异常捕获
                print(f"🔴 【算法: {name}】 计算失败！")
                print(f"错误信息: {e}")
                print("-" * 55)
        print("=" * 55)


def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()