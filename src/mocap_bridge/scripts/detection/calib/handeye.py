#!/usr/bin/env python3
"""D435/ZED + ChArUco eye-in-hand calibration.

The camera and the tracked rigid body (the robot gripper) must be rigidly
connected, while the ChArUco board remains fixed in the mocap world::

    T_world_gripper @ T_gripper_camera @ T_camera_board = T_world_board

``/mocap_data`` supplies ``T_world_gripper`` and ChArUco PnP supplies
``T_camera_board``.  OpenCV ``calibrateHandEye`` then estimates
``T_gripper_camera`` (camera optical frame -> gripper/rigid-body frame).

Examples:

    python3 handeye.py --camera d435 --rigid-id 5
    python3 handeye.py --camera zed --rigid-id 5 --auto-capture

The equivalent ROS 2 parameter form is also supported::

    python3 handeye.py --ros-args -p camera_type:=zed -p rigid_id:=5 \
        -p auto_capture:=true

Keys: ``s`` save a stationary pose, ``u`` undo, ``c`` calculate, ``q`` quit.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import select
import sys
import termios
import threading
import time
import tty
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
DETECTION_DIR = SCRIPT_DIR.parent
if str(DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTION_DIR))

DICTIONARY_NAME = "DICT_6X6_250"
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH_M = 0.01812
MARKER_LENGTH_M = 0.01446


@dataclass
class BoardPose:
    """A board-to-camera pose returned by ChArUco PnP."""

    rotation: np.ndarray
    translation: np.ndarray
    rvec: np.ndarray
    corners: np.ndarray
    ids: np.ndarray
    corner_count: int
    reprojection_rmse_px: float


@dataclass
class PoseSample:
    """One averaged, synchronized gripper/board observation."""

    rotation_gripper_raw: np.ndarray
    translation_gripper_raw: np.ndarray
    rotation_board_to_camera: np.ndarray
    translation_board_to_camera: np.ndarray
    frame_count: int
    mean_pair_delta_ms: float
    max_pair_delta_ms: float
    mean_reprojection_rmse_px: float
    mean_corner_count: float


@dataclass
class RuntimeConfig:
    camera_type: str
    camera_serial: Optional[str]
    rigid_id: int
    mocap_topic: str
    mocap_position_scale: float
    mocap_pose_direction: str
    use_mocap_header_stamp: bool
    max_pair_delta_sec: float
    max_observation_age_sec: float
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    legacy_pattern: bool
    min_charuco_corners: int
    max_reprojection_error_px: float
    averaging_window_sec: float
    min_window_pairs: int
    stationary_gripper_translation_m: float
    stationary_gripper_rotation_deg: float
    stationary_pnp_translation_m: float
    stationary_pnp_rotation_deg: float
    stationary_wait_timeout_sec: float
    duplicate_translation_m: float
    duplicate_rotation_deg: float
    min_samples: int
    max_outlier_fraction: float
    auto_capture: bool
    auto_target_samples: int
    auto_check_interval_sec: float
    show_image: bool
    output_path: Optional[Path]


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a homogeneous transform from a 3x3 rotation and 3-vector."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_pose(
    rotation: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Invert a rigid pose represented by rotation and translation."""
    rotation_inverse = np.asarray(rotation, dtype=np.float64).reshape(3, 3).T
    translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)
    return rotation_inverse, -rotation_inverse @ translation


def mean_pose(
    rotations: Sequence[np.ndarray], translations: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average SE(3) samples and return per-sample translation/angle errors."""
    rotation_group = Rotation.from_matrix(np.asarray(rotations, dtype=np.float64))
    average_rotation = rotation_group.mean()
    translation_array = np.asarray(translations, dtype=np.float64).reshape(-1, 3)
    average_translation = translation_array.mean(axis=0)
    translation_errors = np.linalg.norm(
        translation_array - average_translation, axis=1
    )
    rotation_errors_deg = np.degrees(
        (average_rotation.inv() * rotation_group).magnitude()
    )
    return (
        average_rotation.as_matrix(),
        average_translation.reshape(3, 1),
        translation_errors,
        rotation_errors_deg,
    )


def stamp_to_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def depth_preview(depth_image: np.ndarray, depth_scale: float) -> np.ndarray:
    """Create a common 0-5 m color preview for D435 and ZED depth."""
    depth_m = np.asarray(depth_image, dtype=np.float32) * float(depth_scale)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    depth_u8 = np.zeros(depth_m.shape, dtype=np.uint8)
    depth_u8[valid] = np.clip(depth_m[valid] * 51.0, 0.0, 255.0).astype(
        np.uint8
    )
    display = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    display[~valid] = 0
    return display


class CharucoPoseEstimator:
    """Detect the configured ChArUco board and estimate ``T_camera_board``."""

    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion_coefficients: np.ndarray,
        distortion_model: str,
        squares_x: int = SQUARES_X,
        squares_y: int = SQUARES_Y,
        square_length_m: float = SQUARE_LENGTH_M,
        marker_length_m: float = MARKER_LENGTH_M,
        min_corners: int = 8,
        max_reprojection_error_px: float = 1.0,
        legacy_pattern: bool = False,
        realsense_sdk: Any = None,
        realsense_intrinsics: Any = None,
    ) -> None:
        if squares_x < 2 or squares_y < 2:
            raise ValueError("squares_x and squares_y must both be >= 2")
        if not 0.0 < marker_length_m < square_length_m:
            raise ValueError("require 0 < marker_length_m < square_length_m")
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV was built without the aruco module")
        if not hasattr(cv2.aruco, "CharucoDetector"):
            raise RuntimeError(
                "OpenCV CharucoDetector is required; install a current "
                "opencv-contrib-python build"
            )

        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(
            3, 3
        )
        raw_distortion = np.asarray(
            distortion_coefficients, dtype=np.float64
        ).reshape(1, -1)
        if raw_distortion.size == 0:
            raw_distortion = np.zeros((1, 5), dtype=np.float64)
        self.raw_distortion = raw_distortion
        self.distortion_model = str(distortion_model).strip().lower()
        self.inverse_brown = "inverse_brown_conrady" in self.distortion_model
        self.realsense_sdk = realsense_sdk
        self.realsense_intrinsics = realsense_intrinsics
        self.pnp_distortion = (
            np.zeros((1, 5), dtype=np.float64)
            if self.inverse_brown
            else self.raw_distortion
        )
        self.min_corners = max(4, int(min_corners))
        self.max_reprojection_error_px = float(max_reprojection_error_px)
        self.axis_length_m = 2.0 * float(square_length_m)

        supported_models = ("none", "brown_conrady", "plumb_bob")
        if not self.inverse_brown and not any(
            name in self.distortion_model for name in supported_models
        ):
            raise RuntimeError(
                f"unsupported camera distortion model: {self.distortion_model}"
            )
        if self.inverse_brown and (
            self.realsense_sdk is None or self.realsense_intrinsics is None
        ):
            raise RuntimeError(
                "inverse Brown-Conrady pixels require RealSense SDK intrinsics"
            )

        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_6X6_250
        )
        self.board = cv2.aruco.CharucoBoard(
            (int(squares_x), int(squares_y)),
            float(square_length_m),
            float(marker_length_m),
            dictionary,
        )
        if legacy_pattern:
            self.board.setLegacyPattern(True)

        charuco_parameters = cv2.aruco.CharucoParameters()
        if not self.inverse_brown:
            charuco_parameters.cameraMatrix = self.camera_matrix
            charuco_parameters.distCoeffs = self.pnp_distortion
        charuco_parameters.tryRefineMarkers = True
        self.detector = cv2.aruco.CharucoDetector(
            self.board,
            charuco_parameters,
            cv2.aruco.DetectorParameters(),
        )

    def _to_ideal_pixels(self, pixels: np.ndarray) -> np.ndarray:
        if not self.inverse_brown:
            return np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        ideal_pixels = []
        for u, v in np.asarray(pixels, dtype=np.float64).reshape(-1, 2):
            point = self.realsense_sdk.rs2_deproject_pixel_to_point(
                self.realsense_intrinsics, [float(u), float(v)], 1.0
            )
            if not np.all(np.isfinite(point)) or abs(point[2]) < 1e-12:
                raise ValueError("invalid inverse Brown-Conrady pixel")
            ideal_pixels.append(
                [
                    self.realsense_intrinsics.fx * point[0] / point[2]
                    + self.realsense_intrinsics.ppx,
                    self.realsense_intrinsics.fy * point[1] / point[2]
                    + self.realsense_intrinsics.ppy,
                ]
            )
        return np.asarray(ideal_pixels, dtype=np.float64)

    def estimate(
        self, gray_image: np.ndarray
    ) -> tuple[Optional[BoardPose], Any, Any]:
        """Return pose, detected marker corners, and marker IDs."""
        corners, ids, marker_corners, marker_ids = self.detector.detectBoard(
            gray_image
        )
        if ids is None or corners is None or len(ids) < self.min_corners:
            return None, marker_corners, marker_ids
        if self.board.checkCharucoCornersCollinear(ids):
            return None, marker_corners, marker_ids

        object_points, image_points = self.board.matchImagePoints(corners, ids)
        object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
        raw_image_points = np.asarray(image_points, dtype=np.float64).reshape(
            -1, 2
        )
        pnp_image_points = self._to_ideal_pixels(raw_image_points)

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            pnp_image_points,
            self.camera_matrix,
            self.pnp_distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not ok:
            return None, marker_corners, marker_ids
        if hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points,
                pnp_image_points,
                self.camera_matrix,
                self.pnp_distortion,
                rvec,
                tvec,
            )

        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        if not np.all(np.isfinite(tvec)) or tvec[2, 0] <= 0.0:
            return None, marker_corners, marker_ids
        projected_points, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.pnp_distortion,
        )
        error = projected_points.reshape(-1, 2) - pnp_image_points
        reprojection_rmse_px = float(
            np.sqrt(np.mean(np.sum(error * error, axis=1)))
        )
        if (
            not np.isfinite(reprojection_rmse_px)
            or reprojection_rmse_px > self.max_reprojection_error_px
        ):
            return None, marker_corners, marker_ids

        rotation, _ = cv2.Rodrigues(rvec)
        return (
            BoardPose(
                rotation=rotation,
                translation=tvec,
                rvec=rvec,
                corners=corners,
                ids=ids,
                corner_count=int(len(ids)),
                reprojection_rmse_px=reprojection_rmse_px,
            ),
            marker_corners,
            marker_ids,
        )


class HandEyeSolver:
    """Solve and score eye-in-hand candidates using a fixed-board residual."""

    METHODS = {
        "Tsai-Lenz": cv2.CALIB_HAND_EYE_TSAI,
        "Park": cv2.CALIB_HAND_EYE_PARK,
        "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    @staticmethod
    def has_rotation_excitation(rotations: Sequence[np.ndarray]) -> bool:
        """Require useful relative rotations about at least two axes."""
        axes = []
        for first in range(len(rotations)):
            for second in range(first + 1, len(rotations)):
                relative = rotations[first].T @ rotations[second]
                rotation_vector = Rotation.from_matrix(relative).as_rotvec()
                angle = float(np.linalg.norm(rotation_vector))
                if angle >= np.radians(8.0):
                    axes.append(rotation_vector / angle)
        minimum_cross_product = np.sin(np.radians(20.0))
        for first in range(len(axes)):
            for second in range(first + 1, len(axes)):
                if (
                    np.linalg.norm(np.cross(axes[first], axes[second]))
                    > minimum_cross_product
                ):
                    return True
        return False

    @staticmethod
    def _fixed_board_residual(
        rotation_camera_to_gripper: np.ndarray,
        translation_camera_to_gripper: np.ndarray,
        rotations_gripper_to_world: Sequence[np.ndarray],
        translations_gripper_to_world: Sequence[np.ndarray],
        rotations_board_to_camera: Sequence[np.ndarray],
        translations_board_to_camera: Sequence[np.ndarray],
    ) -> dict[str, Any]:
        positions = []
        rotations = []
        for r_w_g, t_w_g, r_c_b, t_c_b in zip(
            rotations_gripper_to_world,
            translations_gripper_to_world,
            rotations_board_to_camera,
            translations_board_to_camera,
        ):
            rotation_world_to_board = (
                r_w_g @ rotation_camera_to_gripper @ r_c_b
            )
            translation_world_to_board = r_w_g @ (
                rotation_camera_to_gripper @ t_c_b
                + translation_camera_to_gripper
            ) + t_w_g
            positions.append(translation_world_to_board.reshape(3))
            rotations.append(rotation_world_to_board)

        position_array = np.asarray(positions, dtype=np.float64)
        position_errors_m = np.linalg.norm(
            position_array - position_array.mean(axis=0), axis=1
        )
        rotation_group = Rotation.from_matrix(np.asarray(rotations))
        average_rotation = rotation_group.mean()
        rotation_errors_deg = np.degrees(
            (average_rotation.inv() * rotation_group).magnitude()
        )
        return {
            "translation_errors_m": position_errors_m,
            "rotation_errors_deg": rotation_errors_deg,
            "translation_rmse_mm": float(
                1000.0 * np.sqrt(np.mean(position_errors_m**2))
            ),
            "rotation_rmse_deg": float(
                np.sqrt(np.mean(rotation_errors_deg**2))
            ),
            "translation_max_mm": float(1000.0 * np.max(position_errors_m)),
            "rotation_max_deg": float(np.max(rotation_errors_deg)),
        }

    @classmethod
    def fit_direction(
        cls, samples: Sequence[PoseSample], pose_direction: str
    ) -> tuple[list[dict[str, Any]], list[str]]:
        raw_rotations = [sample.rotation_gripper_raw for sample in samples]
        raw_translations = [sample.translation_gripper_raw for sample in samples]
        if pose_direction == "rigid_to_world":
            rotations_gripper_to_world = [value.copy() for value in raw_rotations]
            translations_gripper_to_world = [
                value.copy() for value in raw_translations
            ]
        elif pose_direction == "world_to_rigid":
            inverted = [
                invert_pose(rotation, translation)
                for rotation, translation in zip(
                    raw_rotations, raw_translations
                )
            ]
            rotations_gripper_to_world = [item[0] for item in inverted]
            translations_gripper_to_world = [item[1] for item in inverted]
        else:
            raise ValueError(f"invalid pose direction: {pose_direction}")

        rotations_board_to_camera = [
            sample.rotation_board_to_camera for sample in samples
        ]
        translations_board_to_camera = [
            sample.translation_board_to_camera for sample in samples
        ]
        candidates = []
        warnings = []
        for method_name, method_flag in cls.METHODS.items():
            try:
                rotation, translation = cv2.calibrateHandEye(
                    R_gripper2base=rotations_gripper_to_world,
                    t_gripper2base=translations_gripper_to_world,
                    R_target2cam=rotations_board_to_camera,
                    t_target2cam=translations_board_to_camera,
                    method=method_flag,
                )
                rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
                translation = np.asarray(translation, dtype=np.float64).reshape(
                    3, 1
                )
                valid = (
                    np.all(np.isfinite(rotation))
                    and np.all(np.isfinite(translation))
                    and np.linalg.norm(rotation.T @ rotation - np.eye(3)) < 1e-3
                    and abs(np.linalg.det(rotation) - 1.0) < 1e-3
                )
                if not valid:
                    raise ValueError("solver returned a non-rigid transform")
                residual = cls._fixed_board_residual(
                    rotation,
                    translation,
                    rotations_gripper_to_world,
                    translations_gripper_to_world,
                    rotations_board_to_camera,
                    translations_board_to_camera,
                )
                candidates.append(
                    {
                        "method": method_name,
                        "mocap_pose_direction": pose_direction,
                        "rotation": rotation,
                        "translation": translation,
                        **residual,
                    }
                )
            except (cv2.error, ValueError) as error:
                warnings.append(f"{pose_direction}/{method_name}: {error}")
        return candidates, warnings

    @classmethod
    def fit(
        cls, samples: Sequence[PoseSample], pose_direction: str
    ) -> tuple[list[dict[str, Any]], list[str]]:
        directions = (
            ("rigid_to_world", "world_to_rigid")
            if pose_direction == "auto"
            else (pose_direction,)
        )
        candidates = []
        warnings = []
        for direction in directions:
            direction_candidates, direction_warnings = cls.fit_direction(
                samples, direction
            )
            candidates.extend(direction_candidates)
            warnings.extend(direction_warnings)
        candidates.sort(
            key=lambda item: (
                item["translation_rmse_mm"],
                item["rotation_rmse_deg"],
            )
        )
        return candidates, warnings

    @staticmethod
    def robust_inlier_mask(candidate: dict[str, Any]) -> np.ndarray:
        translation = np.asarray(candidate["translation_errors_m"])
        rotation = np.asarray(candidate["rotation_errors_deg"])
        translation_median = np.median(translation)
        translation_mad = np.median(np.abs(translation - translation_median))
        rotation_median = np.median(rotation)
        rotation_mad = np.median(np.abs(rotation - rotation_median))
        translation_limit = translation_median + max(
            0.002, 3.5 * 1.4826 * translation_mad
        )
        rotation_limit = rotation_median + max(
            0.3, 3.5 * 1.4826 * rotation_mad
        )
        return (translation <= translation_limit) & (rotation <= rotation_limit)


class HandEyeCalibrationNode(Node):
    """ROS 2 acquisition node for synchronized mocap and ChArUco poses."""

    def __init__(self, cli: argparse.Namespace) -> None:
        super().__init__("charuco_eye_in_hand_calibrator")
        self._declare_parameters(cli)
        self.config = self._read_config()
        self._validate_config()

        self.data_lock = threading.Lock()
        self.capture_lock = threading.Lock()
        self.mocap_buffer: deque[tuple[float, np.ndarray, np.ndarray]] = deque(
            maxlen=1200
        )
        self.paired_buffer: deque[dict[str, Any]] = deque(maxlen=600)
        self.samples: list[PoseSample] = []
        self.latest_pair_delta_ms: Optional[float] = None
        self.latest_rmse_px: Optional[float] = None
        self.latest_corner_count: Optional[int] = None
        self.capture_pending = False
        self.capture_deadline: Optional[float] = None
        self.next_auto_check = time.monotonic()
        self.calculation_started = False
        self.shutdown_requested = False
        self.realsense_sdk = None

        self.camera, self.camera_description, self.camera_fps = self._open_camera()
        if not self.camera.start():
            self.camera.stop()
            raise RuntimeError(f"{self.config.camera_type} camera start failed")
        try:
            self.estimator = self._create_pose_estimator()
        except Exception:
            self.camera.stop()
            raise

        try:
            from mocap_bridge.msg import MocapData
        except ImportError as error:
            self.camera.stop()
            raise RuntimeError(
                "cannot import mocap_bridge.msg; source the built ROS 2 "
                "workspace before running handeye.py"
            ) from error

        self.subscription = self.create_subscription(
            MocapData,
            self.config.mocap_topic,
            self._mocap_callback,
            qos_profile_sensor_data,
        )
        self.timer = self.create_timer(1.0 / self.camera_fps, self._process_frame)

        self.get_logger().info(
            f"camera={self.camera_description}; topic={self.config.mocap_topic}; "
            f"rigid_id={self.config.rigid_id}; board={self.config.squares_x}x"
            f"{self.config.squares_y}, {self.config.square_length_m * 1000:.2f} "
            f"mm/{self.config.marker_length_m * 1000:.2f} mm"
        )
        if self.config.auto_capture:
            self.get_logger().info(
                "自动采样已开启：标定板保持不动，每次移动相机后短暂停稳；"
                f"收集 {self.config.auto_target_samples} 个有效姿态后自动计算。"
            )
        else:
            self.get_logger().info(
                "标定板保持不动；移动相机/刚体并停稳。按键："
                "s=采样，u=撤销，c=计算，q=退出。"
            )
        self.keyboard_thread = threading.Thread(
            target=self._keyboard_loop, daemon=True
        )
        self.keyboard_thread.start()

    def _declare_parameters(self, cli: argparse.Namespace) -> None:
        values = {
            "camera_type": cli.camera,
            "camera_serial": cli.camera_serial or "",
            "rigid_id": cli.rigid_id,
            "mocap_topic": cli.mocap_topic,
            "mocap_position_scale": cli.mocap_position_scale,
            "mocap_pose_direction": cli.mocap_pose_direction,
            "use_mocap_header_stamp": True,
            "max_pair_delta_sec": 0.03,
            # Never save a window after camera/board detection has gone stale.
            "max_observation_age_sec": 0.25,
            "squares_x": SQUARES_X,
            "squares_y": SQUARES_Y,
            "square_length_m": SQUARE_LENGTH_M,
            "marker_length_m": MARKER_LENGTH_M,
            "legacy_pattern": False,
            "min_charuco_corners": 8,
            "max_reprojection_error_px": 1.0,
            "averaging_window_sec": 0.40,
            "min_window_pairs": 8,
            "stationary_gripper_translation_m": 0.002,
            "stationary_gripper_rotation_deg": 0.5,
            "stationary_pnp_translation_m": 0.003,
            "stationary_pnp_rotation_deg": 0.8,
            "stationary_wait_timeout_sec": 3.0,
            "duplicate_translation_m": 0.010,
            "duplicate_rotation_deg": 5.0,
            "min_samples": cli.min_samples,
            "max_outlier_fraction": 0.25,
            "auto_capture": cli.auto_capture,
            "auto_target_samples": cli.auto_target_samples,
            "auto_check_interval_sec": 0.10,
            "show_image": not cli.no_gui,
            "output_path": cli.output or "",
        }
        for name, default in values.items():
            self.declare_parameter(name, default)

    def _read_config(self) -> RuntimeConfig:
        parameter = lambda name: self.get_parameter(name).value
        camera_name = str(parameter("camera_type")).strip().lower()
        camera_aliases = {
            "d435": "d435",
            "d435i": "d435",
            "realsense": "d435",
            "rs": "d435",
            "zed": "zed",
            "zedx": "zed",
            "zed_x": "zed",
        }
        if camera_name not in camera_aliases:
            raise ValueError(
                "camera_type must be d435/realsense/rs or zed/zedx"
            )
        output_value = str(parameter("output_path")).strip()
        camera_serial = str(parameter("camera_serial")).strip()
        return RuntimeConfig(
            camera_type=camera_aliases[camera_name],
            camera_serial=camera_serial or None,
            rigid_id=int(parameter("rigid_id")),
            mocap_topic=str(parameter("mocap_topic")),
            mocap_position_scale=float(parameter("mocap_position_scale")),
            mocap_pose_direction=str(parameter("mocap_pose_direction"))
            .strip()
            .lower(),
            use_mocap_header_stamp=bool(parameter("use_mocap_header_stamp")),
            max_pair_delta_sec=float(parameter("max_pair_delta_sec")),
            max_observation_age_sec=float(
                parameter("max_observation_age_sec")
            ),
            squares_x=int(parameter("squares_x")),
            squares_y=int(parameter("squares_y")),
            square_length_m=float(parameter("square_length_m")),
            marker_length_m=float(parameter("marker_length_m")),
            legacy_pattern=bool(parameter("legacy_pattern")),
            min_charuco_corners=max(4, int(parameter("min_charuco_corners"))),
            max_reprojection_error_px=float(
                parameter("max_reprojection_error_px")
            ),
            averaging_window_sec=float(parameter("averaging_window_sec")),
            min_window_pairs=max(3, int(parameter("min_window_pairs"))),
            stationary_gripper_translation_m=float(
                parameter("stationary_gripper_translation_m")
            ),
            stationary_gripper_rotation_deg=float(
                parameter("stationary_gripper_rotation_deg")
            ),
            stationary_pnp_translation_m=float(
                parameter("stationary_pnp_translation_m")
            ),
            stationary_pnp_rotation_deg=float(
                parameter("stationary_pnp_rotation_deg")
            ),
            stationary_wait_timeout_sec=max(
                0.0, float(parameter("stationary_wait_timeout_sec"))
            ),
            duplicate_translation_m=float(parameter("duplicate_translation_m")),
            duplicate_rotation_deg=float(parameter("duplicate_rotation_deg")),
            min_samples=max(3, int(parameter("min_samples"))),
            max_outlier_fraction=float(parameter("max_outlier_fraction")),
            auto_capture=bool(parameter("auto_capture")),
            auto_target_samples=max(
                3, int(parameter("auto_target_samples"))
            ),
            auto_check_interval_sec=max(
                0.02, float(parameter("auto_check_interval_sec"))
            ),
            show_image=bool(parameter("show_image")),
            output_path=Path(output_value).expanduser().resolve()
            if output_value
            else None,
        )

    def _validate_config(self) -> None:
        config = self.config
        if config.squares_x < 2 or config.squares_y < 2:
            raise ValueError("squares_x and squares_y must both be >= 2")
        if not 0.0 < config.marker_length_m < config.square_length_m:
            raise ValueError("require 0 < marker_length_m < square_length_m")
        if config.mocap_position_scale <= 0.0:
            raise ValueError("mocap_position_scale must be positive")
        if config.mocap_pose_direction not in {
            "auto",
            "rigid_to_world",
            "world_to_rigid",
        }:
            raise ValueError(
                "mocap_pose_direction must be auto, rigid_to_world, or "
                "world_to_rigid"
            )
        if config.max_pair_delta_sec <= 0.0:
            raise ValueError("max_pair_delta_sec must be positive")
        if config.max_observation_age_sec <= 0.0:
            raise ValueError("max_observation_age_sec must be positive")
        if config.averaging_window_sec <= 0.0:
            raise ValueError("averaging_window_sec must be positive")
        nonnegative_thresholds = {
            "stationary_gripper_translation_m": (
                config.stationary_gripper_translation_m
            ),
            "stationary_gripper_rotation_deg": (
                config.stationary_gripper_rotation_deg
            ),
            "stationary_pnp_translation_m": (
                config.stationary_pnp_translation_m
            ),
            "stationary_pnp_rotation_deg": config.stationary_pnp_rotation_deg,
            "duplicate_translation_m": config.duplicate_translation_m,
            "duplicate_rotation_deg": config.duplicate_rotation_deg,
        }
        invalid_thresholds = [
            name for name, value in nonnegative_thresholds.items() if value < 0.0
        ]
        if invalid_thresholds:
            raise ValueError(
                "thresholds must be nonnegative: " + ", ".join(invalid_thresholds)
            )
        if not 0.0 <= config.max_outlier_fraction < 1.0:
            raise ValueError("max_outlier_fraction must be in [0, 1)")
        config.auto_target_samples = max(
            config.min_samples, config.auto_target_samples
        )

    def _open_camera(self) -> tuple[Any, str, float]:
        serial = self.config.camera_serial
        if self.config.camera_type == "d435":
            try:
                import pyrealsense2 as rs
                from device.realsense_camera import RealSenseCamera
            except ImportError as error:
                raise RuntimeError(
                    "D435 requires pyrealsense2 and device/realsense_camera.py"
                ) from error
            self.realsense_sdk = rs
            return (
                RealSenseCamera(
                    width=640, height=480, fps=60, serial_number=serial
                ),
                "Intel RealSense D435 640x480@60",
                60.0,
            )

        try:
            import pyzed.sl as sl
            from device.zed_camera import ZEDCamera
        except ImportError as error:
            raise RuntimeError(
                "ZED requires the ZED SDK Python API and device/zed_camera.py"
            ) from error
        zed_serial = int(serial) if serial is not None else None
        return (
            ZEDCamera(
                resolution=sl.RESOLUTION.SVGA,
                fps=120,
                serial_number=zed_serial,
            ),
            "ZED rectified left SVGA@120",
            120.0,
        )

    def _create_pose_estimator(self) -> CharucoPoseEstimator:
        camera_matrix = self.camera.get_color_intrinsic_matrix().astype(
            np.float64
        )
        intrinsics = self.camera.color_intrinsics
        distortion_model = str(getattr(intrinsics, "model", "none"))
        distortion = np.asarray(
            getattr(intrinsics, "coeffs", np.zeros(5)), dtype=np.float64
        )
        self.camera_matrix = camera_matrix
        self.raw_distortion = distortion.reshape(1, -1)
        self.distortion_model = distortion_model
        estimator = CharucoPoseEstimator(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion,
            distortion_model=distortion_model,
            squares_x=self.config.squares_x,
            squares_y=self.config.squares_y,
            square_length_m=self.config.square_length_m,
            marker_length_m=self.config.marker_length_m,
            min_corners=self.config.min_charuco_corners,
            max_reprojection_error_px=self.config.max_reprojection_error_px,
            legacy_pattern=self.config.legacy_pattern,
            realsense_sdk=self.realsense_sdk,
            realsense_intrinsics=intrinsics,
        )
        if estimator.inverse_brown:
            self.get_logger().warn(
                "D435 color stream is inverse Brown-Conrady; ChArUco points "
                "will be converted to ideal pixels before PnP."
            )
        return estimator

    def _mocap_callback(self, message: Any) -> None:
        receive_time = time.time()
        header_time = stamp_to_seconds(message.header.stamp)
        sample_time = (
            header_time
            if self.config.use_mocap_header_stamp and header_time > 0.0
            else receive_time
        )
        for rigid_body in message.rigid_bodies:
            if (
                int(rigid_body.rigid_id) != self.config.rigid_id
                or not rigid_body.is_track
            ):
                continue
            quaternion = np.array(
                [rigid_body.qx, rigid_body.qy, rigid_body.qz, rigid_body.qw],
                dtype=np.float64,
            )
            quaternion_norm = float(np.linalg.norm(quaternion))
            if not np.isfinite(quaternion_norm) or quaternion_norm < 1e-12:
                return
            rotation = Rotation.from_quat(
                quaternion / quaternion_norm
            ).as_matrix()
            translation = self.config.mocap_position_scale * np.array(
                [[rigid_body.x], [rigid_body.y], [rigid_body.z]],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(translation)):
                return
            with self.data_lock:
                self.mocap_buffer.append((sample_time, rotation, translation))
            return

    def _process_frame(self) -> None:
        color, depth, metadata = self.camera.get_images(return_metadata=True)
        if color is None or depth is None or metadata is None:
            self._process_pending_capture()
            return

        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        pose, marker_corners, marker_ids = self.estimator.estimate(gray)
        if marker_ids is not None and len(marker_ids):
            cv2.aruco.drawDetectedMarkers(color, marker_corners, marker_ids)
        if pose is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                color, pose.corners, pose.ids, (0, 255, 0)
            )
            capture_time_ns = metadata.get("capture_time_ns")
            camera_time = (
                float(capture_time_ns) * 1e-9
                if capture_time_ns is not None
                else time.time()
            )
            self._add_synchronized_pair(camera_time, pose)
            self.latest_rmse_px = pose.reprojection_rmse_px
            self.latest_corner_count = pose.corner_count
            if not self.estimator.inverse_brown:
                cv2.drawFrameAxes(
                    color,
                    self.camera_matrix,
                    self.estimator.pnp_distortion,
                    pose.rvec,
                    pose.translation,
                    self.estimator.axis_length_m,
                )

        self._process_pending_capture()
        self._process_auto_capture()
        if self.config.show_image:
            self._show_preview(color, depth)

    def _show_preview(self, color: np.ndarray, depth: np.ndarray) -> None:
        with self.data_lock:
            sample_count = len(self.samples)
        status = (
            f"AUTO {sample_count}/{self.config.auto_target_samples}"
            if self.config.auto_capture
            else f"saved={sample_count}"
        )
        if self.latest_rmse_px is not None:
            status += f"  PnP={self.latest_rmse_px:.2f}px"
        if self.latest_corner_count is not None:
            status += f"  corners={self.latest_corner_count}"
        if self.latest_pair_delta_ms is not None:
            status += f"  dt={self.latest_pair_delta_ms:.1f}ms"
        if self.capture_pending:
            status += "  HOLD STILL"
        cv2.putText(
            color,
            status,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )
        depth_image = depth_preview(depth, self.camera.depth_scale)
        if depth_image.shape[:2] != color.shape[:2]:
            depth_image = cv2.resize(
                depth_image,
                (color.shape[1], color.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        cv2.imshow("D435/ZED ChArUco eye-in-hand", cv2.hconcat([color, depth_image]))
        key = cv2.waitKey(1) & 0xFF
        if key != 0xFF:
            self._handle_key(chr(key).lower())

    def _add_synchronized_pair(
        self, camera_time: float, board_pose: BoardPose
    ) -> None:
        with self.data_lock:
            if not self.mocap_buffer:
                return
            if (
                self.paired_buffer
                and camera_time == self.paired_buffer[-1]["camera_time"]
            ):
                return
            mocap_time, rotation, translation = min(
                self.mocap_buffer,
                key=lambda item: abs(item[0] - camera_time),
            )
            delta = abs(mocap_time - camera_time)
            self.latest_pair_delta_ms = 1000.0 * delta
            if delta > self.config.max_pair_delta_sec:
                return
            self.paired_buffer.append(
                {
                    "camera_time": camera_time,
                    "paired_monotonic": time.monotonic(),
                    "mocap_time": mocap_time,
                    "pair_delta_sec": delta,
                    "rotation_gripper_raw": rotation.copy(),
                    "translation_gripper_raw": translation.copy(),
                    "rotation_board_to_camera": board_pose.rotation.copy(),
                    "translation_board_to_camera": board_pose.translation.copy(),
                    "reprojection_rmse_px": board_pose.reprojection_rmse_px,
                    "corner_count": board_pose.corner_count,
                }
            )

    def _keyboard_loop(self) -> None:
        if not sys.stdin.isatty():
            if not self.config.auto_capture:
                self.get_logger().warn(
                    "stdin is not a terminal; use the preview window keys, "
                    "or enable auto_capture/no GUI."
                )
            return
        try:
            old_settings = termios.tcgetattr(sys.stdin)
        except termios.error as error:
            self.get_logger().warn(f"cannot configure terminal keys: {error}")
            return
        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok() and not self.shutdown_requested:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    self._handle_key(sys.stdin.read(1).lower())
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def _handle_key(self, key: str) -> None:
        if key == "s" and not self.config.auto_capture:
            self._request_stationary_pose()
        elif key == "u":
            self._undo_last_pose()
        elif key == "c":
            if self._calculate_calibration():
                self._request_shutdown()
        elif key == "q":
            self._request_shutdown()

    def _request_shutdown(self) -> None:
        self.shutdown_requested = True
        if rclpy.ok():
            rclpy.shutdown()

    def _request_stationary_pose(self) -> bool:
        with self.capture_lock:
            if self.capture_pending:
                return False
            saved, retryable, reason = self._try_save_stationary_pose()
            if saved:
                return True
            if not retryable or self.config.stationary_wait_timeout_sec <= 0.0:
                self.get_logger().warn(reason)
                return False
            self.capture_pending = True
            self.capture_deadline = (
                time.monotonic() + self.config.stationary_wait_timeout_sec
            )
        self.get_logger().info(
            "采样请求已收到，请保持相机静止，正在等待有效平均窗口。"
        )
        return False

    def _process_pending_capture(self) -> None:
        if not self.capture_pending or not self.capture_lock.acquire(False):
            return
        try:
            saved, retryable, reason = self._try_save_stationary_pose()
            if saved:
                self.capture_pending = False
                self.capture_deadline = None
                return
            timed_out = (
                self.capture_deadline is not None
                and time.monotonic() >= self.capture_deadline
            )
            if not retryable or timed_out:
                self.capture_pending = False
                self.capture_deadline = None
                self.get_logger().warn(
                    reason if not timed_out else f"等待静止超时，未保存：{reason}"
                )
        finally:
            self.capture_lock.release()

    def _process_auto_capture(self) -> None:
        if (
            not self.config.auto_capture
            or self.capture_pending
            or self.calculation_started
            or time.monotonic() < self.next_auto_check
        ):
            return
        self.next_auto_check = (
            time.monotonic() + self.config.auto_check_interval_sec
        )
        if not self.capture_lock.acquire(False):
            return
        try:
            saved, _, _ = self._try_save_stationary_pose()
        finally:
            self.capture_lock.release()
        if not saved:
            return
        with self.data_lock:
            count = len(self.samples)
            rotations = [item.rotation_gripper_raw.copy() for item in self.samples]
        if count < self.config.auto_target_samples:
            return
        if not HandEyeSolver.has_rotation_excitation(rotations):
            self.get_logger().info(
                "姿态数量已达到目标，但旋转方向不够丰富；请继续增加俯仰、"
                "偏航和滚转姿态。"
            )
            return
        self.calculation_started = True
        if self._calculate_calibration():
            self._request_shutdown()
        else:
            self.calculation_started = False

    def _try_save_stationary_pose(self) -> tuple[bool, bool, str]:
        with self.data_lock:
            if not self.paired_buffer:
                return False, True, "没有同步的 ChArUco/动捕数据"
            observation_age = (
                time.monotonic()
                - self.paired_buffer[-1]["paired_monotonic"]
            )
            if observation_age > self.config.max_observation_age_sec:
                return (
                    False,
                    True,
                    "最新同步观测已过期 "
                    f"({observation_age * 1000.0:.0f} ms)，"
                    "请确认标定板仍被清晰检测且动捕正常",
                )
            newest_time = self.paired_buffer[-1]["camera_time"]
            window = [
                dict(item)
                for item in self.paired_buffer
                if newest_time - item["camera_time"]
                <= self.config.averaging_window_sec
            ]
        if len(window) < self.config.min_window_pairs:
            return (
                False,
                True,
                f"平均窗口只有 {len(window)} 帧，需要 "
                f"{self.config.min_window_pairs} 帧",
            )

        r_gripper, t_gripper, gripper_t_error, gripper_r_error = mean_pose(
            [item["rotation_gripper_raw"] for item in window],
            [item["translation_gripper_raw"] for item in window],
        )
        r_board, t_board, board_t_error, board_r_error = mean_pose(
            [item["rotation_board_to_camera"] for item in window],
            [item["translation_board_to_camera"] for item in window],
        )
        checks = (
            (
                float(np.max(gripper_t_error)),
                self.config.stationary_gripper_translation_m,
                "末端平移",
                "m",
            ),
            (
                float(np.max(gripper_r_error)),
                self.config.stationary_gripper_rotation_deg,
                "末端旋转",
                "deg",
            ),
            (
                float(np.max(board_t_error)),
                self.config.stationary_pnp_translation_m,
                "PnP 平移",
                "m",
            ),
            (
                float(np.max(board_r_error)),
                self.config.stationary_pnp_rotation_deg,
                "PnP 旋转",
                "deg",
            ),
        )
        failures = [
            f"{name}离散={value:.4g}{unit}>{limit:.4g}{unit}"
            for value, limit, name, unit in checks
            if value > limit
        ]
        if failures:
            return False, True, "当前姿态不静止：" + "; ".join(failures)

        with self.data_lock:
            for index, sample in enumerate(self.samples, start=1):
                translation_delta = float(
                    np.linalg.norm(t_gripper - sample.translation_gripper_raw)
                )
                relative_rotation = sample.rotation_gripper_raw.T @ r_gripper
                rotation_delta = float(
                    np.degrees(
                        Rotation.from_matrix(relative_rotation).magnitude()
                    )
                )
                if (
                    translation_delta < self.config.duplicate_translation_m
                    and rotation_delta < self.config.duplicate_rotation_deg
                ):
                    return (
                        False,
                        False,
                        f"与第 {index} 个姿态过于相似，请继续移动或旋转",
                    )
            sample = PoseSample(
                rotation_gripper_raw=r_gripper,
                translation_gripper_raw=t_gripper,
                rotation_board_to_camera=r_board,
                translation_board_to_camera=t_board,
                frame_count=len(window),
                mean_pair_delta_ms=float(
                    1000.0
                    * np.mean([item["pair_delta_sec"] for item in window])
                ),
                max_pair_delta_ms=float(
                    1000.0
                    * np.max([item["pair_delta_sec"] for item in window])
                ),
                mean_reprojection_rmse_px=float(
                    np.mean(
                        [item["reprojection_rmse_px"] for item in window]
                    )
                ),
                mean_corner_count=float(
                    np.mean([item["corner_count"] for item in window])
                ),
            )
            self.samples.append(sample)
            count = len(self.samples)
        self.get_logger().info(
            f"已保存姿态 {count}：平均 {len(window)} 帧，PnP="
            f"{sample.mean_reprojection_rmse_px:.3f}px，"
            f"dt={sample.mean_pair_delta_ms:.1f}ms"
        )
        return True, False, ""

    def _undo_last_pose(self) -> None:
        with self.data_lock:
            if not self.samples:
                self.get_logger().warn("没有可撤销的姿态")
                return
            self.samples.pop()
            count = len(self.samples)
        self.get_logger().info(f"已撤销最后一个姿态，剩余 {count} 个")

    def _calculate_calibration(self) -> bool:
        if not self.capture_lock.acquire(False):
            self.get_logger().warn("正在采样，请稍后再计算")
            return False
        try:
            with self.data_lock:
                samples = list(self.samples)
            if len(samples) < self.config.min_samples:
                self.get_logger().error(
                    f"只有 {len(samples)} 个姿态，至少需要 "
                    f"{self.config.min_samples} 个"
                )
                return False
            if not HandEyeSolver.has_rotation_excitation(
                [sample.rotation_gripper_raw for sample in samples]
            ):
                self.get_logger().error(
                    "旋转激励不足：需要绕至少两个不平行轴采集姿态"
                )
                return False

            candidates, warnings = HandEyeSolver.fit(
                samples, self.config.mocap_pose_direction
            )
            for warning in warnings:
                self.get_logger().warn(f"手眼求解器失败：{warning}")
            if not candidates:
                self.get_logger().error("全部手眼求解器均失败")
                return False

            inlier_mask = HandEyeSolver.robust_inlier_mask(candidates[0])
            rejected_count = int(
                len(inlier_mask) - np.count_nonzero(inlier_mask)
            )
            maximum_rejected = int(
                np.floor(len(samples) * self.config.max_outlier_fraction)
            )
            used_samples = samples
            rejected_indices: list[int] = []
            if (
                0 < rejected_count <= maximum_rejected
                and np.count_nonzero(inlier_mask) >= self.config.min_samples
            ):
                proposed_samples = [
                    sample
                    for sample, keep in zip(samples, inlier_mask)
                    if keep
                ]
                refitted, refit_warnings = HandEyeSolver.fit(
                    proposed_samples, self.config.mocap_pose_direction
                )
                for warning in refit_warnings:
                    self.get_logger().warn(f"鲁棒重算求解器失败：{warning}")
                if refitted:
                    candidates = refitted
                    used_samples = proposed_samples
                    rejected_indices = (
                        np.flatnonzero(~inlier_mask).astype(int).tolist()
                    )

            best = candidates[0]
            output_path = self._save_result(
                best,
                candidates,
                samples,
                used_samples,
                rejected_indices,
            )
            self._print_result(best, candidates, output_path, rejected_indices)
            return True
        finally:
            self.capture_lock.release()

    @staticmethod
    def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
        transform = make_transform(
            candidate["rotation"], candidate["translation"]
        )
        return {
            "method": candidate["method"],
            "mocap_pose_direction": candidate["mocap_pose_direction"],
            "T_gripper_camera": transform.tolist(),
            # Project compatibility: the mocap rigid body is the gripper.
            "T_rigid_camera": transform.tolist(),
            "quaternion_xyzw": Rotation.from_matrix(
                candidate["rotation"]
            ).as_quat().tolist(),
            "translation_m": candidate["translation"].reshape(3).tolist(),
            "translation_rmse_mm": candidate["translation_rmse_mm"],
            "rotation_rmse_deg": candidate["rotation_rmse_deg"],
            "translation_max_mm": candidate["translation_max_mm"],
            "rotation_max_deg": candidate["rotation_max_deg"],
        }

    def _result_path(self) -> Path:
        if self.config.output_path is not None:
            path = self.config.output_path
            if path.suffix.lower() != ".json":
                path = path / "handeye_calibration.json"
            return path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return (
            SCRIPT_DIR
            / f"{self.config.camera_type}_{timestamp}"
            / "handeye_calibration.json"
        )

    def _save_result(
        self,
        best: dict[str, Any],
        candidates: Sequence[dict[str, Any]],
        all_samples: Sequence[PoseSample],
        used_samples: Sequence[PoseSample],
        rejected_indices: Sequence[int],
    ) -> Path:
        output_path = self._result_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rotation_gripper_to_camera, translation_gripper_to_camera = invert_pose(
            best["rotation"], best["translation"]
        )
        payload = {
            "calibration_type": "eye_in_hand",
            "transform_convention": {
                "equation": (
                    "T_world_gripper * T_gripper_camera * "
                    "T_camera_board = T_world_board"
                ),
                "T_gripper_camera": (
                    "maps camera optical-frame points into gripper/rigid-body "
                    "coordinates"
                ),
                "T_camera_gripper": "inverse of T_gripper_camera",
            },
            "selected": self._candidate_summary(best),
            "T_camera_gripper": make_transform(
                rotation_gripper_to_camera, translation_gripper_to_camera
            ).tolist(),
            # Backward-compatible naming used by plot_auto_calib.py.
            "T_camera_rigid": make_transform(
                rotation_gripper_to_camera, translation_gripper_to_camera
            ).tolist(),
            "sample_count_collected": len(all_samples),
            "sample_count_used": len(used_samples),
            "rejected_sample_indices_zero_based": list(rejected_indices),
            "rigid_id": self.config.rigid_id,
            "mocap_topic": self.config.mocap_topic,
            "mocap_position_scale": self.config.mocap_position_scale,
            "board": {
                "dictionary": DICTIONARY_NAME,
                "squares_x": self.config.squares_x,
                "squares_y": self.config.squares_y,
                "square_length_m": self.config.square_length_m,
                "marker_length_m": self.config.marker_length_m,
                "board_square_area_width_m": (
                    self.config.squares_x * self.config.square_length_m
                ),
                "board_square_area_height_m": (
                    self.config.squares_y * self.config.square_length_m
                ),
            },
            "camera": {
                "type": self.config.camera_type,
                "description": self.camera_description,
                "requested_serial": self.config.camera_serial,
                "serial": self._actual_camera_serial(),
                "image_width": int(getattr(self.camera, "width", 0)),
                "image_height": int(getattr(self.camera, "height", 0)),
                "fps": float(self.camera_fps),
                "optical_frame_convention": "x right, y down, z forward",
                "matrix": self.camera_matrix.tolist(),
                "distortion_model": self.distortion_model,
                "distortion_coefficients": self.raw_distortion.tolist(),
            },
            "all_solver_results": [
                self._candidate_summary(candidate) for candidate in candidates
            ],
            "sample_quality": [
                {
                    "frame_count": sample.frame_count,
                    "mean_pair_delta_ms": sample.mean_pair_delta_ms,
                    "max_pair_delta_ms": sample.max_pair_delta_ms,
                    "mean_pnp_rmse_px": (
                        sample.mean_reprojection_rmse_px
                    ),
                    "mean_corner_count": sample.mean_corner_count,
                }
                for sample in all_samples
            ],
        }
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        return output_path

    def _actual_camera_serial(self) -> Optional[str]:
        """Return the connected device serial without depending on one SDK."""
        serial = getattr(self.camera, "serial_number", None)
        if self.config.camera_type == "d435" and self.realsense_sdk is not None:
            try:
                serial = self.camera.profile.get_device().get_info(
                    self.realsense_sdk.camera_info.serial_number
                )
            except Exception:
                pass
        return None if serial is None else str(serial)

    @staticmethod
    def _print_result(
        best: dict[str, Any],
        candidates: Iterable[dict[str, Any]],
        output_path: Path,
        rejected_indices: Sequence[int],
    ) -> None:
        print("\n" + "=" * 78)
        print("眼在手上标定结果：T_gripper_camera（camera -> gripper/rigid）")
        print("=" * 78)
        for candidate in candidates:
            selected = "  <-- selected" if candidate is best else ""
            print(
                f"{candidate['mocap_pose_direction']:<15} "
                f"{candidate['method']:<12} "
                f"平移RMSE={candidate['translation_rmse_mm']:.3f} mm, "
                f"旋转RMSE={candidate['rotation_rmse_deg']:.3f} deg"
                f"{selected}"
            )
        print("-" * 78)
        print(make_transform(best["rotation"], best["translation"]))
        print(
            "quaternion xyzw:",
            Rotation.from_matrix(best["rotation"]).as_quat(),
        )
        if rejected_indices:
            print("剔除的样本索引（从 0 开始）:", list(rejected_indices))
        print("结果文件:", output_path)
        print("=" * 78)

    def destroy_node(self) -> None:
        self.shutdown_requested = True
        self.camera.stop()
        if self.config.show_image:
            cv2.destroyAllWindows()
        super().destroy_node()


def parse_arguments(args: Optional[Sequence[str]] = None) -> tuple[
    argparse.Namespace, list[str]
]:
    parser = argparse.ArgumentParser(
        description="D435/ZED + ChArUco 眼在手上手眼标定"
    )
    parser.add_argument(
        "--camera",
        default="d435",
        help="相机类型：d435/realsense 或 zed（默认 d435）",
    )
    parser.add_argument("--camera-serial", default=None, help="可选相机序列号")
    parser.add_argument(
        "--rigid-id",
        type=int,
        default=5,
        help="与相机刚性连接的动捕刚体 ID（默认 5）",
    )
    parser.add_argument(
        "--mocap-topic", default="/mocap_data", help="动捕消息话题"
    )
    parser.add_argument(
        "--mocap-position-scale",
        type=float,
        default=0.001,
        help="动捕位置换算到米的比例（毫米输入为 0.001）",
    )
    parser.add_argument(
        "--mocap-pose-direction",
        choices=("auto", "rigid_to_world", "world_to_rigid"),
        default="rigid_to_world",
        help="动捕位姿方向（本项目默认 rigid_to_world）",
    )
    parser.add_argument(
        "--min-samples", type=int, default=12, help="求解所需最少姿态数"
    )
    parser.add_argument(
        "--auto-capture", action="store_true", help="自动保存静止且不重复的姿态"
    )
    parser.add_argument(
        "--auto-target-samples",
        type=int,
        default=20,
        help="自动计算的目标姿态数",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 JSON 路径；也可指定一个输出目录",
    )
    parser.add_argument("--no-gui", action="store_true", help="不显示预览窗口")
    parsed, ros_args = parser.parse_known_args(args)
    return parsed, ros_args


def main(args: Optional[Sequence[str]] = None) -> None:
    cli, ros_args = parse_arguments(args)
    rclpy.init(args=ros_args)
    node: Optional[HandEyeCalibrationNode] = None
    try:
        node = HandEyeCalibrationNode(cli)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"handeye.py 启动失败: {error}", file=sys.stderr)
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
