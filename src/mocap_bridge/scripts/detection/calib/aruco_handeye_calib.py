#!/usr/bin/env python3
"""RealSense 彩色相机与动捕刚体的 eye-in-hand 标定。

坐标约定：
  - 动捕输入为 ^world T_rigid（rigid -> mocap world）。
  - /aruco_pose 为 ^camera T_target（target -> camera）。
  - OpenCV 输出 ^rigid T_camera（camera -> mocap rigid body）。

前提：相机与 rigid_id 对应的刚体刚性连接，ChArUco 板在动捕世界中固定。
"""

from collections import deque
import json
from pathlib import Path
import select
import sys
import termios
import threading
import tty

import cv2
from geometry_msgs.msg import PoseStamped
from mocap_bridge.msg import MocapData
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation


def _stamp_to_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _make_transform(rotation_matrix, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def _invert_transform(rotation_matrix, translation):
    rotation_inv = np.asarray(rotation_matrix, dtype=np.float64).T
    translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)
    translation_inv = -rotation_inv @ translation
    return rotation_inv, translation_inv


class HandEyeCalibrator(Node):
    def __init__(self):
        super().__init__('hand_eye_calibrator')

        self.declare_parameter('rigid_id', 4)
        self.declare_parameter('mocap_position_scale', 0.001)
        self.declare_parameter('mocap_pose_is_rigid_to_world', True)
        self.declare_parameter('use_mocap_header_stamp', True)
        self.declare_parameter('max_pair_delta_sec', 0.03)
        self.declare_parameter('min_samples', 10)
        self.declare_parameter('duplicate_translation_m', 0.005)
        self.declare_parameter('duplicate_rotation_deg', 3.0)
        self.declare_parameter('output_file', 'handeye_calibration.json')

        self.rigid_id = int(self.get_parameter('rigid_id').value)
        self.mocap_position_scale = float(
            self.get_parameter('mocap_position_scale').value
        )
        self.mocap_pose_is_rigid_to_world = bool(
            self.get_parameter('mocap_pose_is_rigid_to_world').value
        )
        self.use_mocap_header_stamp = bool(
            self.get_parameter('use_mocap_header_stamp').value
        )
        self.max_pair_delta_sec = float(
            self.get_parameter('max_pair_delta_sec').value
        )
        self.min_samples = max(
            3, int(self.get_parameter('min_samples').value)
        )
        self.duplicate_translation_m = float(
            self.get_parameter('duplicate_translation_m').value
        )
        self.duplicate_rotation_deg = float(
            self.get_parameter('duplicate_rotation_deg').value
        )
        self.output_file = str(self.get_parameter('output_file').value)

        # 每个元素为 (timestamp_sec, R, t)。队列用于从同一时间轴上找最近帧，
        # 而不是错误地把两个话题各自的“最新帧”直接拼成一对。
        self.mocap_buffer = deque(maxlen=600)
        self.aruco_buffer = deque(maxlen=300)
        self.data_lock = threading.Lock()

        self.R_gripper2base = []
        self.t_gripper2base = []
        self.R_target2cam = []
        self.t_target2cam = []
        self.sample_timestamps = []

        self.mocap_sub = self.create_subscription(
            MocapData,
            '/mocap_data',
            self.mocap_callback,
            qos_profile_sensor_data,
        )
        self.aruco_sub = self.create_subscription(
            PoseStamped,
            '/aruco_pose',
            self.aruco_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f'手眼标定已启动：rigid_id={self.rigid_id}。'
        )
        self.get_logger().info(
            "每次将相机静止后按 's' 保存；采集 15–25 个方向充分不同的姿态，"
            "再按 'c' 计算。"
        )
        if not self.use_mocap_header_stamp:
            self.get_logger().warn(
                '动捕将使用 ROS 回调接收时间；适用于消息头不与系统时钟同步的情况。'
            )

        self.key_thread = threading.Thread(
            target=self.keyboard_loop, daemon=True
        )
        self.key_thread.start()

    def mocap_callback(self, msg):
        receive_time = self.get_clock().now().nanoseconds * 1e-9
        header_time = _stamp_to_seconds(msg.header.stamp)
        stamp_sec = (
            header_time
            if self.use_mocap_header_stamp and header_time > 0.0
            else receive_time
        )

        for rigid_body in msg.rigid_bodies:
            if rigid_body.rigid_id != self.rigid_id or not rigid_body.is_track:
                continue

            quaternion = np.array(
                [
                    rigid_body.qx,
                    rigid_body.qy,
                    rigid_body.qz,
                    rigid_body.qw,
                ],
                dtype=np.float64,
            )
            quaternion_norm = np.linalg.norm(quaternion)
            if quaternion_norm < 1e-12 or not np.isfinite(quaternion_norm):
                return
            quaternion /= quaternion_norm

            rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
            translation = self.mocap_position_scale * np.array(
                [[rigid_body.x], [rigid_body.y], [rigid_body.z]],
                dtype=np.float64,
            )

            # calibrateHandEye 需要 ^world T_rigid。如果动捕驱动给的是反方向，
            # 在此统一反转，避免在标定调用处混淆坐标定义。
            if not self.mocap_pose_is_rigid_to_world:
                rotation_matrix, translation = _invert_transform(
                    rotation_matrix, translation
                )

            with self.data_lock:
                self.mocap_buffer.append(
                    (stamp_sec, rotation_matrix, translation)
                )
            return

    def aruco_callback(self, msg):
        stamp_sec = _stamp_to_seconds(msg.header.stamp)
        if stamp_sec <= 0.0:
            stamp_sec = self.get_clock().now().nanoseconds * 1e-9

        quaternion = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        quaternion_norm = np.linalg.norm(quaternion)
        if quaternion_norm < 1e-12 or not np.isfinite(quaternion_norm):
            return
        quaternion /= quaternion_norm

        rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
        translation = np.array(
            [
                [msg.pose.position.x],
                [msg.pose.position.y],
                [msg.pose.position.z],
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(translation)) or translation[2, 0] <= 0.0:
            return

        with self.data_lock:
            self.aruco_buffer.append(
                (stamp_sec, rotation_matrix, translation)
            )

    def keyboard_loop(self):
        if not sys.stdin.isatty():
            self.get_logger().error(
                '当前标准输入不是终端，无法监听 s/c；请在交互式终端运行此节点。'
            )
            return

        try:
            old_settings = termios.tcgetattr(sys.stdin)
        except termios.error as error:
            self.get_logger().error(f'无法读取终端设置: {error}')
            return

        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                if not select.select([sys.stdin], [], [], 0.1)[0]:
                    continue
                key = sys.stdin.read(1).lower()
                if key == 's':
                    self.save_current_pose_pair()
                elif key == 'c':
                    if self.calculate_calibration():
                        rclpy.shutdown()
                        return
        finally:
            termios.tcsetattr(
                sys.stdin, termios.TCSADRAIN, old_settings
            )

    def save_current_pose_pair(self):
        with self.data_lock:
            if not self.mocap_buffer or not self.aruco_buffer:
                self.get_logger().warn(
                    '缺少动捕或 ChArUco 数据，当前姿态未保存。'
                )
                return False

            aruco_time, r_target2cam, t_target2cam = self.aruco_buffer[-1]
            mocap_sample = min(
                self.mocap_buffer,
                key=lambda sample: abs(sample[0] - aruco_time),
            )
            mocap_time, r_gripper2base, t_gripper2base = mocap_sample
            pair_delta = abs(mocap_time - aruco_time)

            if pair_delta > self.max_pair_delta_sec:
                self.get_logger().warn(
                    f'最近位姿对相差 {pair_delta * 1000.0:.1f} ms，超过 '
                    f'{self.max_pair_delta_sec * 1000.0:.0f} ms。若两个消息头并非'
                    '同一时钟，请设置 use_mocap_header_stamp:=false。'
                )
                return False

            if self.R_gripper2base:
                translation_delta = float(
                    np.linalg.norm(
                        t_gripper2base - self.t_gripper2base[-1]
                    )
                )
                relative_rotation = (
                    self.R_gripper2base[-1].T @ r_gripper2base
                )
                rotation_delta_deg = float(
                    np.degrees(
                        Rotation.from_matrix(relative_rotation).magnitude()
                    )
                )
                if (
                    translation_delta < self.duplicate_translation_m
                    and rotation_delta_deg < self.duplicate_rotation_deg
                ):
                    self.get_logger().warn(
                        '与上一姿态过于接近，未保存；请改变位置或绕不同轴旋转。'
                    )
                    return False

            self.R_gripper2base.append(r_gripper2base.copy())
            self.t_gripper2base.append(t_gripper2base.copy())
            self.R_target2cam.append(r_target2cam.copy())
            self.t_target2cam.append(t_target2cam.copy())
            self.sample_timestamps.append(
                {
                    'mocap_sec': mocap_time,
                    'camera_sec': aruco_time,
                    'delta_ms': pair_delta * 1000.0,
                }
            )
            sample_count = len(self.R_gripper2base)

        self.get_logger().info(
            f'已保存第 {sample_count} 组，时间差 '
            f'{pair_delta * 1000.0:.1f} ms。'
        )
        return True

    @staticmethod
    def _has_sufficient_rotation_excitation(rotation_matrices):
        """检查是否至少存在两条不近似平行的有效旋转轴。"""
        axes = []
        min_motion_rad = np.radians(5.0)
        for first in range(len(rotation_matrices)):
            for second in range(first + 1, len(rotation_matrices)):
                relative = (
                    rotation_matrices[first].T
                    @ rotation_matrices[second]
                )
                rotation_vector = Rotation.from_matrix(relative).as_rotvec()
                angle = float(np.linalg.norm(rotation_vector))
                if angle >= min_motion_rad:
                    axes.append(rotation_vector / angle)

        for first in range(len(axes)):
            for second in range(first + 1, len(axes)):
                # 轴正负等价；叉乘范数直接衡量是否平行。
                if np.linalg.norm(np.cross(axes[first], axes[second])) > np.sin(
                    np.radians(15.0)
                ):
                    return True
        return False

    def calculate_calibration(self):
        with self.data_lock:
            r_gripper2base = [value.copy() for value in self.R_gripper2base]
            t_gripper2base = [value.copy() for value in self.t_gripper2base]
            r_target2cam = [value.copy() for value in self.R_target2cam]
            t_target2cam = [value.copy() for value in self.t_target2cam]
            timestamps = list(self.sample_timestamps)

        sample_count = len(r_gripper2base)
        if sample_count < self.min_samples:
            self.get_logger().error(
                f'仅有 {sample_count} 组；至少需要 {self.min_samples} 组。'
            )
            return False
        if not self._has_sufficient_rotation_excitation(r_gripper2base):
            self.get_logger().error(
                '姿态旋转轴不充分：请补采绕至少两个不平行轴旋转的姿态。'
            )
            return False

        methods = {
            'Tsai-Lenz': cv2.CALIB_HAND_EYE_TSAI,
            'Park': cv2.CALIB_HAND_EYE_PARK,
            'Horaud': cv2.CALIB_HAND_EYE_HORAUD,
            'Andreff': cv2.CALIB_HAND_EYE_ANDREFF,
            'Daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
        }

        results = []
        for method_name, method_flag in methods.items():
            try:
                r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                    R_gripper2base=r_gripper2base,
                    t_gripper2base=t_gripper2base,
                    R_target2cam=r_target2cam,
                    t_target2cam=t_target2cam,
                    method=method_flag,
                )
                r_cam2gripper = np.asarray(
                    r_cam2gripper, dtype=np.float64
                ).reshape(3, 3)
                t_cam2gripper = np.asarray(
                    t_cam2gripper, dtype=np.float64
                ).reshape(3, 1)
                if not (
                    np.all(np.isfinite(r_cam2gripper))
                    and np.all(np.isfinite(t_cam2gripper))
                ):
                    raise ValueError('结果包含 NaN 或 Inf')

                residual = self.calculate_static_target_residual(
                    r_cam2gripper,
                    t_cam2gripper,
                    r_gripper2base,
                    t_gripper2base,
                    r_target2cam,
                    t_target2cam,
                )
                results.append(
                    {
                        'method': method_name,
                        'rotation': r_cam2gripper,
                        'translation': t_cam2gripper,
                        'quaternion_xyzw': Rotation.from_matrix(
                            r_cam2gripper
                        ).as_quat(),
                        **residual,
                    }
                )
            except (cv2.error, ValueError) as error:
                self.get_logger().warn(
                    f'{method_name} 求解失败: {error}'
                )

        if not results:
            self.get_logger().error('所有手眼标定算法均求解失败。')
            return False

        # 训练样本上固定标定板的位置离散越小越好；若接近，再比较旋转离散。
        results.sort(
            key=lambda result: (
                result['translation_rmse_mm'],
                result['rotation_rmse_deg'],
            )
        )
        best = results[0]
        output_path = self.save_results(
            best, results, timestamps, sample_count
        )
        self.print_results(results, best, output_path)
        return True

    @staticmethod
    def calculate_static_target_residual(
        r_cam2gripper,
        t_cam2gripper,
        r_gripper2base,
        t_gripper2base,
        r_target2cam,
        t_target2cam,
    ):
        """计算固定标定板在动捕世界中的位姿离散程度。"""
        target_positions = []
        target_rotations = []

        for r_b_g, t_b_g, r_c_t, t_c_t in zip(
            r_gripper2base,
            t_gripper2base,
            r_target2cam,
            t_target2cam,
        ):
            # ^base T_target = ^base T_gripper * ^gripper T_camera
            #                  * ^camera T_target
            r_b_t = r_b_g @ r_cam2gripper @ r_c_t
            t_b_t = r_b_g @ (
                r_cam2gripper @ t_c_t + t_cam2gripper
            ) + t_b_g
            target_positions.append(t_b_t.reshape(3))
            target_rotations.append(r_b_t)

        target_positions = np.asarray(target_positions)
        mean_position = target_positions.mean(axis=0)
        translation_errors_m = np.linalg.norm(
            target_positions - mean_position, axis=1
        )

        rotations = Rotation.from_matrix(np.asarray(target_rotations))
        mean_rotation = rotations.mean()
        rotation_errors_deg = np.degrees(
            (mean_rotation.inv() * rotations).magnitude()
        )

        return {
            'translation_rmse_mm': float(
                1000.0 * np.sqrt(np.mean(translation_errors_m ** 2))
            ),
            'rotation_rmse_deg': float(
                np.sqrt(np.mean(rotation_errors_deg ** 2))
            ),
            'translation_max_mm': float(
                1000.0 * np.max(translation_errors_m)
            ),
            'rotation_max_deg': float(np.max(rotation_errors_deg)),
        }

    def save_results(self, best, results, timestamps, sample_count):
        output_path = Path(self.output_file).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        r_rigid_camera = best['rotation']
        t_rigid_camera = best['translation']
        r_camera_rigid, t_camera_rigid = _invert_transform(
            r_rigid_camera, t_rigid_camera
        )

        serializable_results = []
        for result in results:
            serializable_results.append(
                {
                    'method': result['method'],
                    'T_rigid_camera': _make_transform(
                        result['rotation'], result['translation']
                    ).tolist(),
                    'quaternion_xyzw': result[
                        'quaternion_xyzw'
                    ].tolist(),
                    'translation_rmse_mm': result[
                        'translation_rmse_mm'
                    ],
                    'rotation_rmse_deg': result['rotation_rmse_deg'],
                    'translation_max_mm': result['translation_max_mm'],
                    'rotation_max_deg': result['rotation_max_deg'],
                }
            )

        payload = {
            'transform_convention': {
                'selected': 'T_rigid_camera maps camera points into the mocap rigid frame',
                'inverse': 'T_camera_rigid maps mocap rigid-frame points into the camera frame',
            },
            'selected_method': best['method'],
            'selection_rule': 'minimum static-target translation RMSE, then rotation RMSE',
            'sample_count': sample_count,
            'rigid_id': self.rigid_id,
            'mocap_position_scale': self.mocap_position_scale,
            'T_rigid_camera': _make_transform(
                r_rigid_camera, t_rigid_camera
            ).tolist(),
            'T_camera_rigid': _make_transform(
                r_camera_rigid, t_camera_rigid
            ).tolist(),
            'selected_quaternion_xyzw': best[
                'quaternion_xyzw'
            ].tolist(),
            'selected_translation_m': t_rigid_camera.reshape(3).tolist(),
            'selected_residual': {
                'translation_rmse_mm': best['translation_rmse_mm'],
                'rotation_rmse_deg': best['rotation_rmse_deg'],
                'translation_max_mm': best['translation_max_mm'],
                'rotation_max_deg': best['rotation_max_deg'],
            },
            'all_methods': serializable_results,
            'sample_timestamps': timestamps,
        }
        with output_path.open('w', encoding='utf-8') as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.write('\n')
        return output_path

    @staticmethod
    def print_results(results, best, output_path):
        print('\n' + '=' * 76)
        print('Hand-eye result: T_rigid_camera (camera -> mocap rigid body)')
        print('=' * 76)
        for result in results:
            mark = '  <-- selected' if result is best else ''
            print(
                f"{result['method']:<12}  "
                f"translation RMSE={result['translation_rmse_mm']:.3f} mm, "
                f"rotation RMSE={result['rotation_rmse_deg']:.3f} deg"
                f'{mark}'
            )

        print('-' * 76)
        print(_make_transform(best['rotation'], best['translation']))
        print('quaternion xyzw:', best['quaternion_xyzw'])
        print('saved to:', output_path)
        print('=' * 76)


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