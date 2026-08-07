#!/usr/bin/env python3
import argparse
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from builtin_interfaces.msg import Time as TimeMsg
import cv2
from ultralytics import YOLO
import time
from ball_geometry import (
    estimate_fixed_radius_sphere_from_mask,
    largest_component_mask,
    make_inner_mask,
)
from sphere_fit import fit_fixed_radius_sphere
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_WEIGHTS_DIR = (
    SCRIPT_DIR
    / "model"
    / "pic_zed_ball_seg"
    / "yolo26n_seg_768_b16"
    / "weights"
)
DEFAULT_ENGINE_PATH = MODEL_WEIGHTS_DIR / "best.engine"
DEFAULT_PT_PATH = MODEL_WEIGHTS_DIR / "best.pt"
DEFAULT_MODEL_PATH = (
    DEFAULT_ENGINE_PATH if DEFAULT_ENGINE_PATH.is_file() else DEFAULT_PT_PATH
)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="使用 RealSense 或 ZED 相机发布球位置。"
    )
    parser.add_argument(
        "--camera",
        choices=("realsense", "zed"),
        default="realsense",
        help="相机类型（默认：realsense）",
    )
    parser.add_argument(
        "--position-method",
        choices=("silhouette", "depth"),
        default="silhouette",
        help=(
            "球心恢复方法：silhouette 使用分割轮廓和已知半径，"
            "depth 使用深度球面拟合（默认：silhouette）"
        ),
    )
    parser.add_argument(
        "--depth-fallback",
        choices=("auto", "enabled", "disabled"),
        default="auto",
        help=(
            "silhouette 失败后的深度回退：auto 仅 RealSense 启用，"
            "enabled 强制启用，disabled 禁用（默认：auto）"
        ),
    )
    parser.add_argument(
        "--ball-radius-m",
        type=float,
        default=0.110,
        help="球的实测物理半径，单位米（默认：0.110）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="YOLO 分割模型路径（优先使用同目录 best.engine）",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(480, 768),
        help="YOLO 推理尺寸（默认：480 768）",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="YOLO 置信度阈值（默认：0.5）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Ultralytics 推理设备，例如 0、cuda:0 或 cpu",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="对支持的 PyTorch/CUDA 模型启用 FP16；TensorRT 已固定精度",
    )
    parser.add_argument(
        "--processing-hz",
        type=float,
        default=0.0,
        help="处理定时器频率；0 表示跟随相机采集帧率（默认：0）",
    )
    parser.add_argument(
        "--silhouette-boundary-model",
        choices=("ellipse", "raw"),
        default="ellipse",
        help="轮廓模型：ellipse 使用亚像素椭圆，raw 使用原始掩码边界",
    )
    parser.add_argument(
        "--min-ball-radius-px",
        type=float,
        default=10.0,
        help="接受轮廓球心的最小等效像素半径（默认：10 px）",
    )
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--one-euro-filter",
        dest="use_one_euro_filter",
        action="store_true",
        help="对发布到 /ball_center 的球心启用 One Euro 滤波（默认开启）",
    )
    filter_group.add_argument(
        "--disable-one-euro-filter",
        dest="use_one_euro_filter",
        action="store_false",
        help="关闭 /ball_center 的 One Euro 滤波",
    )
    parser.set_defaults(use_one_euro_filter=True)
    parser.add_argument(
        "--one-euro-min-cutoff",
        type=float,
        default=8.0,
        help="One Euro 最小截止频率（默认：8 Hz，兼顾平滑和低时延）",
    )
    parser.add_argument(
        "--one-euro-beta",
        type=float,
        default=1.0,
        help="One Euro 速度自适应系数（默认：1.0）",
    )
    parser.add_argument(
        "--one-euro-derivative-cutoff",
        type=float,
        default=1.0,
        help="One Euro 导数截止频率（默认：1 Hz）",
    )
    preview_group = parser.add_mutually_exclusive_group()
    preview_group.add_argument(
        "--preview",
        dest="show_preview",
        action="store_true",
        help="显示检测预览（默认开启）",
    )
    preview_group.add_argument(
        "--disable-preview",
        dest="show_preview",
        action="store_false",
        help="关闭全部 OpenCV 绘图，降低端到端时延",
    )
    parser.set_defaults(show_preview=True)
    parser.add_argument(
        "--show-depth-preview",
        action="store_true",
        help="在检测预览右侧显示深度伪彩色图（默认关闭以降低开销）",
    )
    parser.add_argument(
        "--disable-surface-publish",
        action="store_true",
        help="silhouette 成功时跳过深度表面点和 /ball_surface，降低开销",
    )

    parsed, ros_args = parser.parse_known_args(args)
    if parsed.ball_radius_m <= 0.0:
        parser.error("--ball-radius-m 必须大于 0")
    if any(value <= 0 for value in parsed.imgsz):
        parser.error("--imgsz 的 HEIGHT 和 WIDTH 必须大于 0")
    if not 0.0 < parsed.confidence <= 1.0:
        parser.error("--confidence 必须在 (0, 1] 范围内")
    if parsed.processing_hz < 0.0:
        parser.error("--processing-hz 不能小于 0")
    if parsed.min_ball_radius_px <= 0.0:
        parser.error("--min-ball-radius-px 必须大于 0")
    if parsed.one_euro_min_cutoff <= 0.0:
        parser.error("--one-euro-min-cutoff 必须大于 0")
    if parsed.one_euro_beta < 0.0:
        parser.error("--one-euro-beta 不能小于 0")
    if parsed.one_euro_derivative_cutoff <= 0.0:
        parser.error("--one-euro-derivative-cutoff 必须大于 0")
    return parsed, ros_args


def create_camera(camera_type):
    if camera_type == "realsense":
        from device.realsense_camera import RealSenseCamera

        return (
            RealSenseCamera(width=640, height=480, fps=60),
            "RealSense",
        )

    from device.zed_camera import ZEDCamera
    import pyzed.sl as sl

    return (
        ZEDCamera(resolution=sl.RESOLUTION.SVGA, fps=120),
        "ZED",
    )


class OneEuroFilter1D:
    """静止时抑制深度噪声、运动时自动提高响应速度的一维滤波器。"""

    def __init__(self, min_cutoff=8.0, beta=1.0, derivative_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.reset()

    def reset(self):
        self.last_time = None
        self.last_raw = None
        self.filtered = None
        self.filtered_derivative = 0.0

    @staticmethod
    def smoothing_factor(dt, cutoff):
        value = 2.0 * np.pi * cutoff * dt
        return value / (value + 1.0)

    @staticmethod
    def smooth(alpha, value, previous):
        return alpha * value + (1.0 - alpha) * previous

    def filter(self, value, timestamp_sec):
        if self.last_time is None:
            self.last_time = timestamp_sec
            self.last_raw = value
            self.filtered = value
            return value

        dt = timestamp_sec - self.last_time
        # 检测中断后不要用陈旧状态拉回新测量。
        if dt <= 0.0 or dt > 0.25:
            self.reset()
            return self.filter(value, timestamp_sec)
        dt = float(np.clip(dt, 1.0 / 240.0, 0.1))

        derivative = (value - self.last_raw) / dt
        derivative_alpha = self.smoothing_factor(
            dt, self.derivative_cutoff
        )
        self.filtered_derivative = self.smooth(
            derivative_alpha, derivative, self.filtered_derivative
        )

        cutoff = self.min_cutoff + self.beta * abs(self.filtered_derivative)
        value_alpha = self.smoothing_factor(dt, cutoff)
        self.filtered = self.smooth(value_alpha, value, self.filtered)
        self.last_time = timestamp_sec
        self.last_raw = value
        return self.filtered


def ball_center(mask_data):
    M = cv2.moments(mask_data.astype(np.uint8))
    if M["m00"] > 0:
        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])
    else:
        u, v = None, None
    return u, v


def compensate_ball_radius(surf_x, surf_y, surf_z, ball_radius=0.115):
    p_surf = np.array([surf_x, surf_y, surf_z])
    dist_to_surf = np.linalg.norm(p_surf)
    if dist_to_surf <= 0.01:
        return surf_x, surf_y, surf_z
    scale_factor = (dist_to_surf + ball_radius) / dist_to_surf
    p_center = p_surf * scale_factor
    return float(p_center[0]), float(p_center[1]), float(p_center[2])


def make_depth_display(depth_image, depth_scale):
    """将深度图转换为 0～5 米的伪彩色预览图。"""
    max_preview_depth_m = 5.0
    depth_m = np.asarray(depth_image, dtype=np.float32) * float(depth_scale)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)

    depth_8bit = np.zeros(depth_m.shape, dtype=np.uint8)
    depth_8bit[valid] = np.clip(
        depth_m[valid] * (255.0 / max_preview_depth_m),
        0.0,
        255.0,
    ).astype(np.uint8)

    depth_display = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_TURBO)
    depth_display[~valid] = 0
    return depth_display


class BallPublisher(Node):
    def __init__(
        self,
        camera_type="realsense",
        position_method="silhouette",
        ball_radius_m=0.110,
        use_one_euro_filter=True,
        model_path=DEFAULT_MODEL_PATH,
        inference_size=(480, 768),
        confidence=0.5,
        device=None,
        use_half=False,
        processing_hz=0.0,
        silhouette_boundary_model="ellipse",
        min_ball_radius_px=10.0,
        one_euro_min_cutoff=8.0,
        one_euro_beta=1.0,
        one_euro_derivative_cutoff=1.0,
        show_preview=True,
        show_depth_preview=False,
        publish_surface=True,
        use_depth_fallback=True,
    ):
        super().__init__('ball_publisher')
        # Keep-last 1 防止慢订阅端让旧坐标在 DDS 队列中积压。
        self.surf_pub = self.create_publisher(PointStamped, '/ball_surface', 1)
        self.raw_center_pub = self.create_publisher(
            PointStamped, '/ball_center_raw', 1
        )
        self.center_pub = self.create_publisher(PointStamped, '/ball_center', 1)

        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"未找到 YOLO 模型: {model_path}")
        self.get_logger().info(f"加载模型: {model_path}")
        self.model = YOLO(str(model_path), task='segment')
        self.inference_size = tuple(map(int, inference_size))
        self.predict_options = {
            "imgsz": self.inference_size,
            "conf": float(confidence),
            "classes": [0],
            "max_det": 1,
            "verbose": False,
            "retina_masks": True,
        }
        if device is not None:
            self.predict_options["device"] = device
        if use_half and model_path.suffix.lower() != ".engine":
            self.predict_options["half"] = True

        # 启动相机
        self.camera, self.camera_name = create_camera(camera_type)
        self.get_logger().info(f"启动 {self.camera_name} 相机...")
        if not self.camera.start():
            raise RuntimeError(f"{self.camera_name} 相机启动失败")

        self.ball_radius = float(ball_radius_m)
        if self.ball_radius <= 0.0:
            raise ValueError("ball_radius_m must be positive")
        self.position_method = position_method
        self.use_one_euro_filter = bool(use_one_euro_filter)
        self.silhouette_boundary_model = silhouette_boundary_model
        self.min_ball_radius_px = float(min_ball_radius_px)
        self.show_preview = bool(show_preview)
        self.show_depth_preview = bool(show_depth_preview)
        self.publish_surface = bool(publish_surface)
        self.use_depth_fallback = bool(use_depth_fallback)
        self.sphere_fit_failure_count = 0
        self.silhouette_fit_failure_count = 0
        self.center_failure_count = 0
        self.missing_mask_count = 0
        self.latest_sphere_fit_rmse_mm = None
        self.latest_silhouette_fit_rmse_mm = None
        self.latest_silhouette_radius_px = None
        self.latest_center_method = None

        self.position_filters = [
            OneEuroFilter1D(
                min_cutoff=one_euro_min_cutoff,
                beta=one_euro_beta,
                derivative_cutoff=one_euro_derivative_cutoff,
            )
            for _ in range(3)
        ]

        self.get_logger().info(
            f"球心方法: {self.position_method}, "
            f"球半径: {self.ball_radius * 1000.0:.1f} mm, "
            f"轮廓模型: {self.silhouette_boundary_model}, "
            f"推理尺寸: {self.inference_size}, "
            f"One Euro: {'开启' if self.use_one_euro_filter else '关闭'} "
            f"(min_cutoff={one_euro_min_cutoff:.2f}, "
            f"beta={one_euro_beta:.2f})"
        )
        self.get_logger().info(
            f"预览: {'开启' if self.show_preview else '关闭'}, "
            f"深度预览: {'开启' if self.show_depth_preview else '关闭'}, "
            f"表面点发布: {'开启' if self.publish_surface else '关闭'}, "
            f"轮廓失败深度回退: "
            f"{'开启' if self.use_depth_fallback else '关闭'}"
        )

        # 每秒统计图像、推理、处理和采集到发布的端到端时延。
        self.fps_window_start = time.perf_counter()
        self.capture_frame_count = 0
        self.inference_frame_count = 0
        self.inference_elapsed = 0.0
        self.processing_elapsed = 0.0
        self.capture_age_sum_ms = 0.0
        self.capture_age_max_ms = 0.0
        self.capture_age_count = 0

        camera_fps = float(
            getattr(
                self.camera,
                "capture_fps",
                getattr(self.camera, "fps", 60.0),
            )
        )
        if camera_fps <= 0.0:
            camera_fps = 60.0
        self.processing_hz = (
            camera_fps if float(processing_hz) <= 0.0 else float(processing_hz)
        )
        self.get_logger().info(f"处理定时器: {self.processing_hz:.1f} Hz")
        self.timer = self.create_timer(
            1.0 / self.processing_hz,
            self.process_frame,
        )

    @staticmethod
    def _depth_fit_is_acceptable(sphere_fit):
        return (
            sphere_fit.rmse_m <= 0.015
            and sphere_fit.median_abs_residual_m <= 0.010
            and sphere_fit.inlier_fraction >= 0.35
        )

    def _estimate_depth_center(
        self,
        depth_image,
        depth_mask,
        intrinsics,
        fallback_center,
    ):
        if fallback_center is None:
            raise ValueError("掩码中心附近没有可用深度")
        try:
            masked_points = self.camera.get_masked_point_cloud(
                depth_image=depth_image,
                mask=depth_mask,
                intrinsics=intrinsics,
                max_points=2500,
            )
            sphere_fit = fit_fixed_radius_sphere(
                masked_points,
                radius=self.ball_radius,
                initial_center=fallback_center,
                min_points=80,
                robust_scale_m=0.005,
            )
            if not self._depth_fit_is_acceptable(sphere_fit):
                raise ValueError(
                    "球拟合质量不足: "
                    f"RMSE={sphere_fit.rmse_m * 1000.0:.1f}mm, "
                    "median="
                    f"{sphere_fit.median_abs_residual_m * 1000.0:.1f}mm, "
                    f"inliers={sphere_fit.inlier_fraction:.1%}"
                )
            self.latest_sphere_fit_rmse_mm = sphere_fit.rmse_m * 1000.0
            return sphere_fit.center.copy(), "depth-sphere"
        except ValueError as error:
            self.latest_sphere_fit_rmse_mm = None
            self.sphere_fit_failure_count += 1
            if (
                self.sphere_fit_failure_count <= 3
                or self.sphere_fit_failure_count % 60 == 0
            ):
                self.get_logger().warn(
                    "深度球拟合失败，使用视线半径补偿回退值: "
                    f"{error}"
                )
            return np.asarray(fallback_center, dtype=np.float64), "depth-ray"

    def _estimate_center(
        self,
        mask,
        depth_image,
        depth_mask,
        intrinsics,
        fallback_center,
    ):
        self.latest_silhouette_fit_rmse_mm = None
        self.latest_sphere_fit_rmse_mm = None
        self.latest_silhouette_radius_px = None

        if self.position_method == "silhouette":
            try:
                silhouette_fit = estimate_fixed_radius_sphere_from_mask(
                    mask,
                    intrinsics,
                    self.ball_radius,
                    boundary_model=self.silhouette_boundary_model,
                )
                if (
                    silhouette_fit.equivalent_radius_px
                    < self.min_ball_radius_px
                ):
                    raise ValueError(
                        "球轮廓过小: "
                        f"radius={silhouette_fit.equivalent_radius_px:.1f}px "
                        f"< {self.min_ball_radius_px:.1f}px"
                    )
                self.latest_silhouette_fit_rmse_mm = (
                    silhouette_fit.contour_rmse_m * 1000.0
                )
                self.latest_silhouette_radius_px = (
                    silhouette_fit.equivalent_radius_px
                )
                return silhouette_fit.center.copy(), "silhouette"
            except ValueError as error:
                self.silhouette_fit_failure_count += 1
                if (
                    self.silhouette_fit_failure_count <= 3
                    or self.silhouette_fit_failure_count % 60 == 0
                ):
                    fallback_action = (
                        "尝试深度回退"
                        if self.use_depth_fallback
                        else "本帧不发布"
                    )
                    self.get_logger().warn(
                        f"轮廓球心恢复失败，{fallback_action}: {error}"
                    )
                if not self.use_depth_fallback:
                    raise ValueError(
                        f"轮廓球心恢复失败且深度回退已关闭: {error}"
                    ) from error

        return self._estimate_depth_center(
            depth_image,
            depth_mask,
            intrinsics,
            fallback_center,
        )

    def _get_surface_point(
        self,
        depth_image,
        depth_mask,
        u,
        v,
        frame_metadata,
    ):
        """按需获取掩码内表面点，避免轮廓法成功时无谓处理深度。"""
        masked_depths_m = depth_image[depth_mask] * self.camera.depth_scale
        valid_depths = masked_depths_m[
            np.isfinite(masked_depths_m)
            & (masked_depths_m > 0.05)
            & (masked_depths_m < 10.0)
        ]
        if len(valid_depths) <= 10:
            return None
        real_x, real_y, real_z = self.camera.get_real_position(
            u,
            v,
            window_size=9,
            depth_frame=frame_metadata['depth_frame'],
            intrinsics=frame_metadata['depth_intrinsics'],
            mask=depth_mask,
        )
        if real_z is None:
            return None
        return np.array([real_x, real_y, real_z], dtype=np.float64)

    def _record_performance(
        self,
        frame_start,
        inference_elapsed,
        capture_time_ns,
    ):
        self.inference_elapsed += inference_elapsed
        self.inference_frame_count += 1
        self.processing_elapsed += time.perf_counter() - frame_start

        if capture_time_ns is not None:
            capture_age_ms = (time.time_ns() - capture_time_ns) * 1e-6
            # 防止来自非 Epoch 时钟域的时间戳污染统计。
            if 0.0 <= capture_age_ms <= 10_000.0:
                self.capture_age_sum_ms += capture_age_ms
                self.capture_age_max_ms = max(
                    self.capture_age_max_ms,
                    capture_age_ms,
                )
                self.capture_age_count += 1

        now = time.perf_counter()
        window_elapsed = now - self.fps_window_start
        if window_elapsed < 1.0:
            return

        processed_fps = self.capture_frame_count / window_elapsed
        inference_fps = (
            self.inference_frame_count / self.inference_elapsed
            if self.inference_elapsed > 0.0
            else 0.0
        )
        average_inference_ms = (
            self.inference_elapsed * 1000.0 / self.inference_frame_count
            if self.inference_frame_count > 0
            else 0.0
        )
        average_processing_ms = (
            self.processing_elapsed * 1000.0 / self.inference_frame_count
            if self.inference_frame_count > 0
            else 0.0
        )
        latency_text = ""
        if self.capture_age_count > 0:
            latency_text = (
                ", 采集到发布 avg/max="
                f"{self.capture_age_sum_ms / self.capture_age_count:.1f}/"
                f"{self.capture_age_max_ms:.1f} ms"
            )
        self.get_logger().info(
            f"处理 FPS: {processed_fps:.2f}, "
            f"模型 FPS: {inference_fps:.2f} "
            f"({average_inference_ms:.2f} ms), "
            f"整帧处理: {average_processing_ms:.2f} ms"
            f"{latency_text}"
        )
        self.fps_window_start = now
        self.capture_frame_count = 0
        self.inference_frame_count = 0
        self.inference_elapsed = 0.0
        self.processing_elapsed = 0.0
        self.capture_age_sum_ms = 0.0
        self.capture_age_max_ms = 0.0
        self.capture_age_count = 0

    def process_frame(self):
        frame_start = time.perf_counter()
        color_image, depth_image, frame_metadata = self.camera.get_images(
            return_metadata=True
        )
        if color_image is None or depth_image is None:
            return
        self.capture_frame_count += 1

        capture_time_ns = frame_metadata['capture_time_ns']
        if capture_time_ns is None:
            capture_time = self.get_clock().now().to_msg()
        else:
            capture_time = TimeMsg(
                sec=int(capture_time_ns // 1_000_000_000),
                nanosec=int(capture_time_ns % 1_000_000_000),
            )

        inference_start = time.perf_counter()
        results = self.model.predict(
            source=color_image,
            **self.predict_options,
        )
        inference_elapsed = time.perf_counter() - inference_start
        display_image = color_image.copy() if self.show_preview else None

        result = results[0]
        if len(result.boxes) > 0:
            detection_index = int(result.boxes.conf.argmax().item())
            box = result.boxes.xyxy[detection_index].cpu().numpy()
            confidence = float(result.boxes.conf[detection_index].item())

            if result.masks is None or detection_index >= len(result.masks.data):
                self.missing_mask_count += 1
                if (
                    self.missing_mask_count <= 3
                    or self.missing_mask_count % 60 == 0
                ):
                    self.get_logger().warn("YOLO 检测到球，但没有返回分割掩码")
            else:
                mask_data = result.masks.data[detection_index].cpu().numpy()
                if mask_data.shape != color_image.shape[:2]:
                    mask_data = cv2.resize(
                        mask_data.astype(np.uint8),
                        (color_image.shape[1], color_image.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask_data = largest_component_mask(mask_data.astype(bool))

                u, v = ball_center(mask_data)
                if u is None or v is None:
                    u = int((box[0] + box[2]) / 2)
                    v = int((box[1] + box[3]) / 2)

                depth_mask = None
                surface_point = None
                if self.publish_surface or self.position_method == "depth":
                    depth_mask = make_inner_mask(mask_data)
                    surface_point = self._get_surface_point(
                        depth_image,
                        depth_mask,
                        u,
                        v,
                        frame_metadata,
                    )

                fallback_center = (
                    None
                    if surface_point is None
                    else compensate_ball_radius(
                        *surface_point.tolist(),
                        self.ball_radius,
                    )
                )
                try:
                    raw_center, center_method = self._estimate_center(
                        mask_data,
                        depth_image,
                        depth_mask,
                        frame_metadata['depth_intrinsics'],
                        fallback_center,
                    )
                except ValueError as error:
                    # 在低时延模式下，轮廓成功时完全不处理深度；仅当轮廓
                    # 失败时才按需计算一次表面点并尝试深度回退。
                    raw_center = None
                    center_method = None
                    if (
                        surface_point is None
                        and self.position_method == "silhouette"
                        and self.use_depth_fallback
                    ):
                        depth_mask = make_inner_mask(mask_data)
                        surface_point = self._get_surface_point(
                            depth_image,
                            depth_mask,
                            u,
                            v,
                            frame_metadata,
                        )
                        if surface_point is not None:
                            fallback_center = compensate_ball_radius(
                                *surface_point.tolist(),
                                self.ball_radius,
                            )
                            raw_center, center_method = (
                                self._estimate_depth_center(
                                    depth_image,
                                    depth_mask,
                                    frame_metadata['depth_intrinsics'],
                                    fallback_center,
                                )
                            )
                    if raw_center is None:
                        self.center_failure_count += 1
                        if (
                            self.center_failure_count <= 3
                            or self.center_failure_count % 60 == 0
                        ):
                            self.get_logger().warn(
                                f"本帧无法恢复球心: {error}"
                            )

                if raw_center is not None:
                    capture_time_sec = (
                        capture_time.sec + capture_time.nanosec * 1e-9
                    )
                    center = raw_center.copy()
                    self.latest_center_method = center_method

                    raw_center_msg = PointStamped()
                    raw_center_msg.header.stamp = capture_time
                    raw_center_msg.header.frame_id = (
                        "camera_color_optical_frame"
                    )
                    (
                        raw_center_msg.point.x,
                        raw_center_msg.point.y,
                        raw_center_msg.point.z,
                    ) = raw_center.tolist()
                    self.raw_center_pub.publish(raw_center_msg)

                    if self.publish_surface and surface_point is not None:
                        surf_msg = PointStamped()
                        surf_msg.header.stamp = capture_time
                        surf_msg.header.frame_id = (
                            "camera_color_optical_frame"
                        )
                        (
                            surf_msg.point.x,
                            surf_msg.point.y,
                            surf_msg.point.z,
                        ) = surface_point.tolist()
                        self.surf_pub.publish(surf_msg)

                    if self.use_one_euro_filter:
                        center = np.array(
                            [
                                position_filter.filter(
                                    value,
                                    capture_time_sec,
                                )
                                for position_filter, value in zip(
                                    self.position_filters,
                                    center,
                                )
                            ],
                            dtype=np.float64,
                        )

                    center_msg = PointStamped()
                    center_msg.header.stamp = capture_time
                    center_msg.header.frame_id = "camera_color_optical_frame"
                    (
                        center_msg.point.x,
                        center_msg.point.y,
                        center_msg.point.z,
                    ) = center.tolist()
                    self.center_pub.publish(center_msg)

                    if display_image is not None:
                        contours, _ = cv2.findContours(
                            mask_data.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        cv2.drawContours(
                            display_image,
                            contours,
                            -1,
                            (0, 255, 255),
                            2,
                        )
                        cv2.rectangle(
                            display_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (255, 0, 0),
                            1,
                        )
                        cv2.circle(display_image, (u, v), 5, (0, 0, 255), -1)
                        text_y = max(25, v - 15)
                        cv2.putText(
                            display_image,
                            f"Z={center[2] * 1000.0:.1f}mm "
                            f"{center_method} conf={confidence:.2f}",
                            (max(5, u - 120), text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )
                        diagnostic_parts = []
                        if self.latest_silhouette_radius_px is not None:
                            diagnostic_parts.append(
                                f"r={self.latest_silhouette_radius_px:.1f}px"
                            )
                        fit_rmse_mm = self.latest_silhouette_fit_rmse_mm
                        if fit_rmse_mm is None:
                            fit_rmse_mm = self.latest_sphere_fit_rmse_mm
                        if fit_rmse_mm is not None:
                            diagnostic_parts.append(
                                f"fit={fit_rmse_mm:.2f}mm"
                            )
                        if diagnostic_parts:
                            cv2.putText(
                                display_image,
                                " ".join(diagnostic_parts),
                                (
                                    max(5, u - 120),
                                    min(display_image.shape[0] - 10, v + 15),
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 255),
                                2,
                            )

        # 坐标发布已经完成；在 GUI 绘制前记录端到端时延，避免窗口刷新
        # 时间被误算为传感器到 ROS 坐标的发布时延。
        self._record_performance(
            frame_start,
            inference_elapsed,
            capture_time_ns,
        )

        if display_image is not None:
            preview_image = display_image
            window_name = f"{self.camera_name} - Detection"
            if self.show_depth_preview:
                depth_display = make_depth_display(
                    depth_image,
                    self.camera.depth_scale,
                )
                if depth_display.shape[:2] != display_image.shape[:2]:
                    depth_display = cv2.resize(
                        depth_display,
                        (display_image.shape[1], display_image.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                preview_image = cv2.hconcat(
                    [display_image, depth_display]
                )
                window_name += " + Depth"
            cv2.imshow(window_name, preview_image)
            if cv2.waitKey(1) & 0xFF == ord('q') and rclpy.ok():
                exit()
                rclpy.shutdown()

    def destroy_node(self):
        self.camera.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    cli_args, ros_args = parse_args(args)
    rclpy.init(args=ros_args)
    use_depth_fallback = (
        cli_args.depth_fallback == "enabled"
        or (
            cli_args.depth_fallback == "auto"
            and cli_args.camera == "realsense"
        )
    )
    node = BallPublisher(
        camera_type=cli_args.camera,
        position_method=cli_args.position_method,
        ball_radius_m=cli_args.ball_radius_m,
        use_one_euro_filter=cli_args.use_one_euro_filter,
        model_path=cli_args.model,
        inference_size=cli_args.imgsz,
        confidence=cli_args.confidence,
        device=cli_args.device,
        use_half=cli_args.half,
        processing_hz=cli_args.processing_hz,
        silhouette_boundary_model=cli_args.silhouette_boundary_model,
        min_ball_radius_px=cli_args.min_ball_radius_px,
        one_euro_min_cutoff=cli_args.one_euro_min_cutoff,
        one_euro_beta=cli_args.one_euro_beta,
        one_euro_derivative_cutoff=(
            cli_args.one_euro_derivative_cutoff
        ),
        show_preview=cli_args.show_preview,
        show_depth_preview=cli_args.show_depth_preview,
        publish_surface=not cli_args.disable_surface_publish,
        use_depth_fallback=use_depth_fallback,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
