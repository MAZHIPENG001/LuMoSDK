#!/usr/bin/env python3
"""Calibrate a RealSense color camera against mocap rigid body 4.

This is an eye-in-hand calibration:

    mocap provides        T_world_rigid
    ChArUco PnP provides  T_camera_target
    result is             T_rigid_camera

The camera and rigid body must be rigidly connected.  The ChArUco board must
remain fixed in the mocap world while the camera/rigid assembly is moved to
different poses.

The board defaults below intentionally match detection/calib/ChArUco.py:
7 x 5 squares, 18.12 mm square length, 14.46 mm marker length, DICT_6X6_250.
"""

from collections import deque
import json
import os
from pathlib import Path
import select
import sys
import termios
import threading
import tty

import cv2
from mocap_bridge.msg import MocapData
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_DIR = os.path.dirname(SCRIPT_DIR)
if DETECTION_DIR not in sys.path:
    sys.path.append(DETECTION_DIR)

from device.realsense_camera import RealSenseCamera  # noqa: E402


def stamp_to_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def make_transform(rotation_matrix, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(rotation_matrix, translation):
    rotation_inv = np.asarray(rotation_matrix, dtype=np.float64).T
    translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)
    return rotation_inv, -rotation_inv @ translation


def mean_pose(rotation_matrices, translations):
    rotations = Rotation.from_matrix(np.asarray(rotation_matrices))
    mean_rotation = rotations.mean()
    translation_array = np.asarray(translations, dtype=np.float64).reshape(-1, 3)
    mean_translation = translation_array.mean(axis=0).reshape(3, 1)

    translation_errors = np.linalg.norm(
        translation_array - mean_translation.reshape(1, 3), axis=1
    )
    rotation_errors_deg = np.degrees(
        (mean_rotation.inv() * rotations).magnitude()
    )
    return (
        mean_rotation.as_matrix(),
        mean_translation,
        translation_errors,
        rotation_errors_deg,
    )


class CharucoMocapHandEye(Node):
    def __init__(self):
        super().__init__('charuco_mocap_handeye_calibrator')

        # Board parameters: identical to ChArUco.py.
        self.declare_parameter('squares_x', 7)
        self.declare_parameter('squares_y', 5)
        self.declare_parameter('square_length_m', 0.01812)
        self.declare_parameter('marker_length_m', 0.01446)
        self.declare_parameter('legacy_pattern', False)
        self.declare_parameter('min_charuco_corners', 8)
        self.declare_parameter('max_reprojection_error_px', 1.0)

        # Camera parameters.
        self.declare_parameter('camera_serial', '')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 60)
        self.declare_parameter('show_image', True)

        # Mocap and time-pairing parameters.
        self.declare_parameter('rigid_id', 4)
        self.declare_parameter('mocap_position_scale', 0.001)
        self.declare_parameter('mocap_pose_direction', 'auto')
        self.declare_parameter('use_mocap_header_stamp', True)
        self.declare_parameter('max_pair_delta_sec', 0.03)

        # One saved pose is an average of a short stationary window.
        self.declare_parameter('averaging_window_sec', 0.40)
        self.declare_parameter('min_window_pairs', 8)
        self.declare_parameter('stationary_mocap_translation_m', 0.002)
        self.declare_parameter('stationary_mocap_rotation_deg', 0.5)
        self.declare_parameter('stationary_pnp_translation_m', 0.003)
        self.declare_parameter('stationary_pnp_rotation_deg', 0.8)
        self.declare_parameter('duplicate_translation_m', 0.010)
        self.declare_parameter('duplicate_rotation_deg', 5.0)

        self.declare_parameter('min_samples', 12)
        self.declare_parameter('max_outlier_fraction', 0.25)
        self.declare_parameter('output_file', 'handeye_calibration.json')

        self.squares_x = int(self.get_parameter('squares_x').value)
        self.squares_y = int(self.get_parameter('squares_y').value)
        self.square_length = float(
            self.get_parameter('square_length_m').value
        )
        self.marker_length = float(
            self.get_parameter('marker_length_m').value
        )
        self.min_charuco_corners = max(
            4, int(self.get_parameter('min_charuco_corners').value)
        )
        self.max_reprojection_error_px = float(
            self.get_parameter('max_reprojection_error_px').value
        )
        self.show_image = bool(self.get_parameter('show_image').value)

        self.rigid_id = int(self.get_parameter('rigid_id').value)
        self.mocap_position_scale = float(
            self.get_parameter('mocap_position_scale').value
        )
        self.mocap_pose_direction = str(
            self.get_parameter('mocap_pose_direction').value
        ).strip().lower()
        if self.mocap_pose_direction not in {
            'auto', 'rigid_to_world', 'world_to_rigid'
        }:
            raise ValueError(
                'mocap_pose_direction must be auto, rigid_to_world, or '
                'world_to_rigid'
            )
        self.use_mocap_header_stamp = bool(
            self.get_parameter('use_mocap_header_stamp').value
        )
        self.max_pair_delta_sec = float(
            self.get_parameter('max_pair_delta_sec').value
        )
        self.averaging_window_sec = float(
            self.get_parameter('averaging_window_sec').value
        )
        self.min_window_pairs = max(
            3, int(self.get_parameter('min_window_pairs').value)
        )
        self.stationary_mocap_translation_m = float(
            self.get_parameter('stationary_mocap_translation_m').value
        )
        self.stationary_mocap_rotation_deg = float(
            self.get_parameter('stationary_mocap_rotation_deg').value
        )
        self.stationary_pnp_translation_m = float(
            self.get_parameter('stationary_pnp_translation_m').value
        )
        self.stationary_pnp_rotation_deg = float(
            self.get_parameter('stationary_pnp_rotation_deg').value
        )
        self.duplicate_translation_m = float(
            self.get_parameter('duplicate_translation_m').value
        )
        self.duplicate_rotation_deg = float(
            self.get_parameter('duplicate_rotation_deg').value
        )
        self.min_samples = max(
            3, int(self.get_parameter('min_samples').value)
        )
        self.max_outlier_fraction = float(
            self.get_parameter('max_outlier_fraction').value
        )
        self.output_file = str(self.get_parameter('output_file').value)

        if self.squares_x < 2 or self.squares_y < 2:
            raise ValueError('squares_x and squares_y must be at least 2')
        if not 0.0 < self.marker_length < self.square_length:
            raise ValueError('require 0 < marker_length < square_length')

        # Buffers contain poses expressed exactly as received from mocap and
        # target->camera poses from PnP.  A saved sample is an average, not a
        # single noisy frame.
        self.data_lock = threading.Lock()
        self.mocap_buffer = deque(maxlen=1200)
        self.paired_buffer = deque(maxlen=600)
        self.samples = []
        self.latest_pair_delta_ms = None
        self.latest_rmse_px = None

        serial = str(self.get_parameter('camera_serial').value).strip()
        self.camera = RealSenseCamera(
            width=int(self.get_parameter('width').value),
            height=int(self.get_parameter('height').value),
            fps=int(self.get_parameter('fps').value),
            serial_number=serial or None,
        )
        if not self.camera.start():
            raise RuntimeError('RealSense camera start failed')

        try:
            self.initialize_detector()
        except Exception:
            self.camera.stop()
            raise

        self.mocap_sub = self.create_subscription(
            MocapData,
            '/mocap_data',
            self.mocap_callback,
            qos_profile_sensor_data,
        )
        fps = int(self.get_parameter('fps').value)
        self.timer = self.create_timer(1.0 / max(1, fps), self.process_frame)

        self.get_logger().info(
            f'subscribed to /mocap_data; rigid_id={self.rigid_id}; '
            f'translation scale={self.mocap_position_scale:g}'
        )
        self.get_logger().info(
            "Keep the board fixed. Stop at each pose, then press 's'. "
            "Use 15-25 poses with rotations about at least two axes. "
            "Keys: s=save, u=undo, c=calculate, q=quit."
        )

        self.key_thread = threading.Thread(
            target=self.keyboard_loop, daemon=True
        )
        self.key_thread.start()

    def initialize_detector(self):
        self.camera_matrix = (
            self.camera.get_color_intrinsic_matrix().astype(np.float64)
        )
        self.color_intrinsics = self.camera.color_intrinsics
        self.distortion_model = str(self.color_intrinsics.model).lower()
        self.raw_dist_coeffs = np.asarray(
            self.color_intrinsics.coeffs, dtype=np.float64
        ).reshape(1, 5)

        # OpenCV coefficients describe ideal->distorted Brown-Conrady.  A
        # RealSense inverse_brown_conrady stream stores the opposite mapping;
        # passing those coefficients directly to solvePnP is incorrect.
        self.inverse_brown = 'inverse_brown_conrady' in self.distortion_model
        if self.inverse_brown:
            self.pnp_dist_coeffs = np.zeros((1, 5), dtype=np.float64)
            self.get_logger().warn(
                'inverse_brown_conrady color stream detected: ChArUco points '
                'will be converted to ideal pixels before solvePnP.'
            )
        elif (
            'brown_conrady' in self.distortion_model
            or 'none' in self.distortion_model
        ):
            self.pnp_dist_coeffs = self.raw_dist_coeffs
        else:
            raise RuntimeError(
                f'unsupported RealSense color distortion model: '
                f'{self.distortion_model}'
            )

        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_6X6_250
        )
        self.board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length,
            self.marker_length,
            dictionary,
        )
        if bool(self.get_parameter('legacy_pattern').value):
            self.board.setLegacyPattern(True)

        detector_params = cv2.aruco.DetectorParameters()
        charuco_params = cv2.aruco.CharucoParameters()
        if not self.inverse_brown:
            charuco_params.cameraMatrix = self.camera_matrix
            charuco_params.distCoeffs = self.pnp_dist_coeffs
        charuco_params.tryRefineMarkers = True
        self.detector = cv2.aruco.CharucoDetector(
            self.board, charuco_params, detector_params
        )

    def mocap_callback(self, msg):
        receive_time = self.get_clock().now().nanoseconds * 1e-9
        header_time = stamp_to_seconds(msg.header.stamp)
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
            norm = np.linalg.norm(quaternion)
            if not np.isfinite(norm) or norm < 1e-12:
                return
            quaternion /= norm
            rotation = Rotation.from_quat(quaternion).as_matrix()
            translation = self.mocap_position_scale * np.array(
                [[rigid_body.x], [rigid_body.y], [rigid_body.z]],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(translation)):
                return

            with self.data_lock:
                self.mocap_buffer.append((stamp_sec, rotation, translation))
            return

    def raw_to_ideal_pixels(self, raw_pixels):
        ideal_pixels = []
        for pixel in np.asarray(raw_pixels, dtype=np.float64).reshape(-1, 2):
            point = rs.rs2_deproject_pixel_to_point(
                self.color_intrinsics,
                [float(pixel[0]), float(pixel[1])],
                1.0,
            )
            if abs(point[2]) < 1e-12:
                raise ValueError('invalid undistorted ChArUco point')
            ideal_pixels.append(
                [
                    self.color_intrinsics.fx * point[0] / point[2]
                    + self.color_intrinsics.ppx,
                    self.color_intrinsics.fy * point[1] / point[2]
                    + self.color_intrinsics.ppy,
                ]
            )
        return np.asarray(ideal_pixels, dtype=np.float64)

    def estimate_board_pose(self, gray):
        corners, ids, marker_corners, marker_ids = self.detector.detectBoard(
            gray
        )
        if ids is None or corners is None:
            return None, marker_corners, marker_ids
        if len(ids) < self.min_charuco_corners:
            return None, marker_corners, marker_ids
        if self.board.checkCharucoCornersCollinear(ids):
            return None, marker_corners, marker_ids

        object_points, image_points = self.board.matchImagePoints(corners, ids)
        object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
        image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        pnp_points = (
            self.raw_to_ideal_pixels(image_points)
            if self.inverse_brown
            else image_points
        )

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            pnp_points,
            self.camera_matrix,
            self.pnp_dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not ok:
            return None, marker_corners, marker_ids
        if hasattr(cv2, 'solvePnPRefineLM'):
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points,
                pnp_points,
                self.camera_matrix,
                self.pnp_dist_coeffs,
                rvec,
                tvec,
            )

        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        if not np.all(np.isfinite(tvec)) or tvec[2, 0] <= 0.0:
            return None, marker_corners, marker_ids
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.pnp_dist_coeffs,
        )
        projected = projected.reshape(-1, 2)
        rmse = float(
            np.sqrt(np.mean(np.sum((projected - pnp_points) ** 2, axis=1)))
        )
        if not np.isfinite(rmse) or rmse > self.max_reprojection_error_px:
            return None, marker_corners, marker_ids

        rotation, _ = cv2.Rodrigues(rvec)
        result = {
            'rotation': rotation,
            'translation': tvec,
            'rvec': rvec,
            'rmse_px': rmse,
            'corner_count': int(len(ids)),
            'corners': corners,
            'ids': ids,
        }
        return result, marker_corners, marker_ids

    def process_frame(self):
        color_image, _, metadata = self.camera.get_images(return_metadata=True)
        if color_image is None or metadata is None:
            return

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        pose, marker_corners, marker_ids = self.estimate_board_pose(gray)
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(
                color_image, marker_corners, marker_ids
            )

        if pose is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                color_image, pose['corners'], pose['ids'], (0, 255, 0)
            )
            camera_time = metadata.get('capture_time_ns')
            camera_time = (
                camera_time * 1e-9
                if camera_time is not None
                else self.get_clock().now().nanoseconds * 1e-9
            )
            self.add_synchronized_pair(camera_time, pose)
            self.latest_rmse_px = pose['rmse_px']

            if not self.inverse_brown:
                cv2.drawFrameAxes(
                    color_image,
                    self.camera_matrix,
                    self.pnp_dist_coeffs,
                    pose['rvec'],
                    pose['translation'],
                    2.0 * self.square_length,
                )

        if self.show_image:
            count = len(self.samples)
            status = f'saved={count}'
            if self.latest_rmse_px is not None:
                status += f'  PnP={self.latest_rmse_px:.2f}px'
            if self.latest_pair_delta_ms is not None:
                status += f'  dt={self.latest_pair_delta_ms:.1f}ms'
            cv2.putText(
                color_image,
                status,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
            cv2.imshow('ChArUco + mocap hand-eye calibration', color_image)
            cv2.waitKey(1)

    def add_synchronized_pair(self, camera_time, pose):
        with self.data_lock:
            if not self.mocap_buffer:
                return
            mocap = min(
                self.mocap_buffer,
                key=lambda item: abs(item[0] - camera_time),
            )
            delta = abs(mocap[0] - camera_time)
            if delta > self.max_pair_delta_sec:
                self.latest_pair_delta_ms = delta * 1000.0
                return
            self.paired_buffer.append(
                {
                    'camera_time': camera_time,
                    'mocap_time': mocap[0],
                    'pair_delta_sec': delta,
                    'R_mocap_raw': mocap[1].copy(),
                    't_mocap_raw': mocap[2].copy(),
                    'R_target2camera': pose['rotation'].copy(),
                    't_target2camera': pose['translation'].copy(),
                    'pnp_rmse_px': pose['rmse_px'],
                    'corner_count': pose['corner_count'],
                }
            )
            self.latest_pair_delta_ms = delta * 1000.0

    def keyboard_loop(self):
        if not sys.stdin.isatty():
            self.get_logger().error(
                'stdin is not a terminal; run this script in an interactive '
                'terminal to use s/u/c/q.'
            )
            return
        try:
            old_settings = termios.tcgetattr(sys.stdin)
        except termios.error as error:
            self.get_logger().error(f'cannot read terminal settings: {error}')
            return

        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                if not select.select([sys.stdin], [], [], 0.1)[0]:
                    continue
                key = sys.stdin.read(1).lower()
                if key == 's':
                    self.save_stationary_pose()
                elif key == 'u':
                    self.undo_last_pose()
                elif key == 'c':
                    if self.calculate_calibration():
                        rclpy.shutdown()
                        return
                elif key == 'q':
                    rclpy.shutdown()
                    return
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def save_stationary_pose(self):
        with self.data_lock:
            if not self.paired_buffer:
                self.get_logger().warn('no synchronized ChArUco/mocap data')
                return False
            newest_time = self.paired_buffer[-1]['camera_time']
            window = [
                item.copy()
                for item in self.paired_buffer
                if newest_time - item['camera_time'] <= self.averaging_window_sec
            ]

        if len(window) < self.min_window_pairs:
            self.get_logger().warn(
                f'only {len(window)} synchronized frames in the averaging '
                f'window; need {self.min_window_pairs}'
            )
            return False

        r_mocap, t_mocap, mocap_t_error, mocap_r_error = mean_pose(
            [item['R_mocap_raw'] for item in window],
            [item['t_mocap_raw'] for item in window],
        )
        r_target, t_target, pnp_t_error, pnp_r_error = mean_pose(
            [item['R_target2camera'] for item in window],
            [item['t_target2camera'] for item in window],
        )

        checks = [
            (
                float(np.max(mocap_t_error)),
                self.stationary_mocap_translation_m,
                'mocap translation',
                'm',
            ),
            (
                float(np.max(mocap_r_error)),
                self.stationary_mocap_rotation_deg,
                'mocap rotation',
                'deg',
            ),
            (
                float(np.max(pnp_t_error)),
                self.stationary_pnp_translation_m,
                'PnP translation',
                'm',
            ),
            (
                float(np.max(pnp_r_error)),
                self.stationary_pnp_rotation_deg,
                'PnP rotation',
                'deg',
            ),
        ]
        failed = [
            f'{name} spread={value:.4g}{unit} > {limit:.4g}{unit}'
            for value, limit, name, unit in checks
            if value > limit
        ]
        if failed:
            self.get_logger().warn(
                'pose is not stationary; not saved: ' + '; '.join(failed)
            )
            return False

        with self.data_lock:
            if self.samples:
                last = self.samples[-1]
                translation_delta = float(
                    np.linalg.norm(t_mocap - last['t_mocap_raw'])
                )
                relative = last['R_mocap_raw'].T @ r_mocap
                rotation_delta = float(
                    np.degrees(Rotation.from_matrix(relative).magnitude())
                )
                if (
                    translation_delta < self.duplicate_translation_m
                    and rotation_delta < self.duplicate_rotation_deg
                ):
                    self.get_logger().warn(
                        'pose is too similar to the previous sample; move and '
                        'rotate the camera before saving again'
                    )
                    return False

            sample = {
                'R_mocap_raw': r_mocap,
                't_mocap_raw': t_mocap,
                'R_target2camera': r_target,
                't_target2camera': t_target,
                'frame_count': len(window),
                'mean_pair_delta_ms': float(
                    1000.0 * np.mean(
                        [item['pair_delta_sec'] for item in window]
                    )
                ),
                'max_pair_delta_ms': float(
                    1000.0 * np.max(
                        [item['pair_delta_sec'] for item in window]
                    )
                ),
                'mean_pnp_rmse_px': float(
                    np.mean([item['pnp_rmse_px'] for item in window])
                ),
                'mean_corner_count': float(
                    np.mean([item['corner_count'] for item in window])
                ),
            }
            self.samples.append(sample)
            count = len(self.samples)

        self.get_logger().info(
            f'saved pose {count}: averaged {len(window)} frames, '
            f'PnP={sample["mean_pnp_rmse_px"]:.3f}px, '
            f'dt={sample["mean_pair_delta_ms"]:.1f}ms'
        )
        return True

    def undo_last_pose(self):
        with self.data_lock:
            if not self.samples:
                self.get_logger().warn('no saved pose to remove')
                return
            self.samples.pop()
            count = len(self.samples)
        self.get_logger().info(f'removed last pose; {count} remain')

    @staticmethod
    def has_rotation_excitation(rotation_matrices):
        axes = []
        for first in range(len(rotation_matrices)):
            for second in range(first + 1, len(rotation_matrices)):
                relative = (
                    rotation_matrices[first].T @ rotation_matrices[second]
                )
                rotvec = Rotation.from_matrix(relative).as_rotvec()
                angle = float(np.linalg.norm(rotvec))
                if angle >= np.radians(8.0):
                    axes.append(rotvec / angle)
        for first in range(len(axes)):
            for second in range(first + 1, len(axes)):
                if np.linalg.norm(np.cross(axes[first], axes[second])) > np.sin(
                    np.radians(20.0)
                ):
                    return True
        return False

    @staticmethod
    def target_residual(
        r_cam2rigid,
        t_cam2rigid,
        r_rigid2world,
        t_rigid2world,
        r_target2camera,
        t_target2camera,
    ):
        target_positions = []
        target_rotations = []
        for r_w_r, t_w_r, r_c_t, t_c_t in zip(
            r_rigid2world,
            t_rigid2world,
            r_target2camera,
            t_target2camera,
        ):
            r_w_t = r_w_r @ r_cam2rigid @ r_c_t
            t_w_t = r_w_r @ (
                r_cam2rigid @ t_c_t + t_cam2rigid
            ) + t_w_r
            target_positions.append(t_w_t.reshape(3))
            target_rotations.append(r_w_t)

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
            'translation_errors_m': translation_errors_m,
            'rotation_errors_deg': rotation_errors_deg,
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

    def solve_candidates(self, samples, pose_direction):
        r_raw = [sample['R_mocap_raw'] for sample in samples]
        t_raw = [sample['t_mocap_raw'] for sample in samples]
        if pose_direction == 'rigid_to_world':
            r_rigid2world = [value.copy() for value in r_raw]
            t_rigid2world = [value.copy() for value in t_raw]
        else:
            inverted = [
                invert_transform(rotation, translation)
                for rotation, translation in zip(r_raw, t_raw)
            ]
            r_rigid2world = [value[0] for value in inverted]
            t_rigid2world = [value[1] for value in inverted]

        r_target = [sample['R_target2camera'] for sample in samples]
        t_target = [sample['t_target2camera'] for sample in samples]
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
                rotation, translation = cv2.calibrateHandEye(
                    R_gripper2base=r_rigid2world,
                    t_gripper2base=t_rigid2world,
                    R_target2cam=r_target,
                    t_target2cam=t_target,
                    method=method_flag,
                )
                rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
                translation = np.asarray(
                    translation, dtype=np.float64
                ).reshape(3, 1)
                if not (
                    np.all(np.isfinite(rotation))
                    and np.all(np.isfinite(translation))
                    and abs(np.linalg.det(rotation) - 1.0) < 1e-3
                ):
                    raise ValueError('non-finite or invalid rigid transform')
                residual = self.target_residual(
                    rotation,
                    translation,
                    r_rigid2world,
                    t_rigid2world,
                    r_target,
                    t_target,
                )
                results.append(
                    {
                        'method': method_name,
                        'mocap_pose_direction': pose_direction,
                        'rotation': rotation,
                        'translation': translation,
                        **residual,
                    }
                )
            except (cv2.error, ValueError) as error:
                self.get_logger().warn(
                    f'{pose_direction}/{method_name} failed: {error}'
                )
        return results

    def fit(self, samples):
        directions = (
            ['rigid_to_world', 'world_to_rigid']
            if self.mocap_pose_direction == 'auto'
            else [self.mocap_pose_direction]
        )
        results = []
        for direction in directions:
            results.extend(self.solve_candidates(samples, direction))
        results.sort(
            key=lambda item: (
                item['translation_rmse_mm'],
                item['rotation_rmse_deg'],
            )
        )
        return results

    @staticmethod
    def robust_inlier_mask(best):
        translation = np.asarray(best['translation_errors_m'])
        rotation = np.asarray(best['rotation_errors_deg'])

        t_median = np.median(translation)
        t_mad = np.median(np.abs(translation - t_median))
        r_median = np.median(rotation)
        r_mad = np.median(np.abs(rotation - r_median))
        t_limit = t_median + max(0.002, 3.5 * 1.4826 * t_mad)
        r_limit = r_median + max(0.3, 3.5 * 1.4826 * r_mad)
        return (translation <= t_limit) & (rotation <= r_limit)

    def calculate_calibration(self):
        with self.data_lock:
            samples = list(self.samples)
        if len(samples) < self.min_samples:
            self.get_logger().error(
                f'only {len(samples)} saved poses; need at least '
                f'{self.min_samples}'
            )
            return False
        if not self.has_rotation_excitation(
            [sample['R_mocap_raw'] for sample in samples]
        ):
            self.get_logger().error(
                'insufficient rotation excitation; collect poses rotated '
                'about at least two non-parallel axes'
            )
            return False

        initial_results = self.fit(samples)
        if not initial_results:
            self.get_logger().error('all hand-eye solvers failed')
            return False

        mask = self.robust_inlier_mask(initial_results[0])
        rejected = int(len(mask) - np.count_nonzero(mask))
        max_rejected = int(np.floor(len(mask) * self.max_outlier_fraction))
        if 0 < rejected <= max_rejected and np.count_nonzero(mask) >= self.min_samples:
            inlier_samples = [
                sample for sample, keep in zip(samples, mask) if keep
            ]
            results = self.fit(inlier_samples)
            if results:
                samples_used = inlier_samples
                rejected_indices = np.flatnonzero(~mask).astype(int).tolist()
            else:
                results = initial_results
                samples_used = samples
                rejected_indices = []
        else:
            results = initial_results
            samples_used = samples
            rejected_indices = []

        best = results[0]
        output_path = self.save_result(
            best, results, samples, samples_used, rejected_indices
        )
        self.print_result(best, results, output_path, rejected_indices)
        return True

    @staticmethod
    def result_summary(result):
        return {
            'method': result['method'],
            'mocap_pose_direction': result['mocap_pose_direction'],
            'T_rigid_camera': make_transform(
                result['rotation'], result['translation']
            ).tolist(),
            'quaternion_xyzw': Rotation.from_matrix(
                result['rotation']
            ).as_quat().tolist(),
            'translation_m': result['translation'].reshape(3).tolist(),
            'translation_rmse_mm': result['translation_rmse_mm'],
            'rotation_rmse_deg': result['rotation_rmse_deg'],
            'translation_max_mm': result['translation_max_mm'],
            'rotation_max_deg': result['rotation_max_deg'],
        }

    def save_result(
        self, best, results, all_samples, samples_used, rejected_indices
    ):
        output_path = Path(self.output_file).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        r_camera_rigid, t_camera_rigid = invert_transform(
            best['rotation'], best['translation']
        )
        payload = {
            'transform_convention': {
                'T_rigid_camera': (
                    'maps camera_color_optical_frame points into mocap rigid '
                    'body 4 coordinates'
                ),
                'T_camera_rigid': (
                    'inverse; maps rigid body 4 points into camera coordinates'
                ),
            },
            'selected': self.result_summary(best),
            'T_camera_rigid': make_transform(
                r_camera_rigid, t_camera_rigid
            ).tolist(),
            'sample_count_collected': len(all_samples),
            'sample_count_used': len(samples_used),
            'rejected_sample_indices_zero_based': rejected_indices,
            'rigid_id': self.rigid_id,
            'mocap_position_scale': self.mocap_position_scale,
            'board': {
                'dictionary': 'DICT_6X6_250',
                'squares_x': self.squares_x,
                'squares_y': self.squares_y,
                'square_length_m': self.square_length,
                'marker_length_m': self.marker_length,
            },
            'camera': {
                'matrix': self.camera_matrix.tolist(),
                'distortion_model': self.distortion_model,
                'raw_distortion_coefficients': self.raw_dist_coeffs.tolist(),
            },
            'all_solver_results': [
                self.result_summary(result) for result in results
            ],
            'sample_quality': [
                {
                    'frame_count': sample['frame_count'],
                    'mean_pair_delta_ms': sample['mean_pair_delta_ms'],
                    'max_pair_delta_ms': sample['max_pair_delta_ms'],
                    'mean_pnp_rmse_px': sample['mean_pnp_rmse_px'],
                    'mean_corner_count': sample['mean_corner_count'],
                }
                for sample in all_samples
            ],
        }
        with output_path.open('w', encoding='utf-8') as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.write('\n')
        return output_path

    @staticmethod
    def print_result(best, results, output_path, rejected_indices):
        print('\n' + '=' * 80)
        print('T_rigid_camera: camera_color_optical_frame -> mocap rigid body')
        print('=' * 80)
        for result in results:
            marker = '  <-- selected' if result is best else ''
            print(
                f"{result['mocap_pose_direction']:<15} "
                f"{result['method']:<12} "
                f"translation RMSE={result['translation_rmse_mm']:.3f} mm, "
                f"rotation RMSE={result['rotation_rmse_deg']:.3f} deg"
                f'{marker}'
            )
        print('-' * 80)
        print(make_transform(best['rotation'], best['translation']))
        print(
            'quaternion xyzw:',
            Rotation.from_matrix(best['rotation']).as_quat(),
        )
        if rejected_indices:
            print('rejected sample indices (zero-based):', rejected_indices)
        print('saved to:', output_path)
        print('=' * 80)

    def destroy_node(self):
        self.camera.stop()
        if self.show_image:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CharucoMocapHandEye()
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
