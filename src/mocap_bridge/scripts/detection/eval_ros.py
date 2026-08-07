#!/usr/bin/env python3
import argparse

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from builtin_interfaces.msg import Time as TimeMsg
import cv2
from ultralytics import YOLO
import time
from record.target_tracker import TargetTracker
from ball_geometry import (
    estimate_fixed_radius_sphere_from_mask,
    largest_component_mask,
    make_inner_mask,
)
from motion_gate import BallMotionGate
from sphere_fit import fit_fixed_radius_sphere
import numpy as np

# 建议加上 half=True 开启半精度(FP16)推理，速度更快
# yolo export model=~/GithubDoc/ultralytics/my_model/model/red_ball/yolo26l-seg/best.pt format=engine task=segment half=True


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
        "--ball-radius-m",
        type=float,
        default=0.110,
        help="球的实测物理半径，单位米（默认：0.110）",
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
        "--disable-motion-gate",
        action="store_true",
        help="关闭球心运动一致性门控（默认开启）",
    )
    parser.add_argument(
        "--max-ball-speed-mps",
        type=float,
        default=8.0,
        help="允许运动球的最大表观速度，单位 m/s（默认：8.0）",
    )
    parser.add_argument(
        "--max-motion-innovation-m",
        type=float,
        default=0.25,
        help="测量与匀速预测的最大偏差，单位米（默认：0.25）",
    )
    parser.add_argument(
        "--max-prediction-sec",
        type=float,
        default=0.25,
        help="极值发生后允许发布预测球心的最长时间（默认：0.25 秒）",
    )
    return parser.parse_known_args(args)


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


class KalmanFilter3D:
    def __init__(self, dt=1.0 / 60.0):
        # 状态向量: [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)

        # 状态转移矩阵 F (基于匀速运动模型)
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # 测量矩阵 H (我们只能观测到位置 x, y, z)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # 状态协方差矩阵 P (初始不确定度)
        self.P = np.eye(6) * 1.0

        # 测量噪声协方差矩阵 R (信任传感器程度)
        # 如果 RealSense 深度跳动大，调大这些值；跳动小，调小这些值。单位是米的平方。
        self.R = np.eye(3) * 0.05

        # 过程噪声协方差矩阵 Q (信任预测模型程度)
        # 如果球变速很快（比如突然被踢飞），调大这些值；如果是平稳滚动，调小。
        self.Q = np.eye(6) * 0.001

        self.is_initialized = False

    def predict(self):
        if not self.is_initialized:
            return self.x[:3]
        # X = F * X
        self.x = np.dot(self.F, self.x)
        # P = F * P * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:3]

    def update(self, z):
        if not self.is_initialized:
            # 第一帧检测到数据时，直接初始化状态位置，速度为0
            self.x[:3] = z
            self.is_initialized = True
            return self.x[:3]

        # 计算卡尔曼增益 K
        # S = H * P * H^T + R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # K = P * H^T * S^-1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # 更新状态 X
        # y = z - H * X (测量残差)
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)

        # 更新协方差 P
        # P = (I - K * H) * P
        I = np.eye(6)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

        return self.x[:3]


class OneEuroFilter1D:
    """静止时抑制深度噪声、运动时自动提高响应速度的一维滤波器。"""

    def __init__(self, min_cutoff=2.0, beta=5.0, derivative_cutoff=1.0):
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
        use_motion_gate=True,
        max_ball_speed_mps=8.0,
        max_motion_innovation_m=0.25,
        max_prediction_sec=0.25,
    ):
        super().__init__('ball_publisher')
        # 创建两个发布者：表面点与球心
        self.surf_pub = self.create_publisher(PointStamped, '/ball_surface', 10)
        self.raw_center_pub = self.create_publisher(
            PointStamped, '/ball_center_raw', 10
        )
        self.center_pub = self.create_publisher(PointStamped, '/ball_center', 10)
        # 加载模型
        model_path = "./model/pic_zed_ball_seg/yolo26n_seg_768_b16/weights/best.engine"
        self.get_logger().info(f"加载模型: {model_path}")
        self.model = YOLO(model_path,task='segment')

        # 启动相机
        self.camera, self.camera_name = create_camera(camera_type)
        self.get_logger().info(f"启动 {self.camera_name} 相机...")
        if not self.camera.start():
            raise RuntimeError(f"{self.camera_name} 相机启动失败")

        # 轨迹记录（可选，保留）
        self.tracker=None
        self.tracker = TargetTracker()

        self.ball_radius = float(ball_radius_m)
        if self.ball_radius <= 0.0:
            raise ValueError("ball_radius_m must be positive")
        self.position_method = position_method
        self.use_one_euro_filter = bool(use_one_euro_filter)
        self.use_motion_gate = bool(use_motion_gate)
        self.sphere_fit_failure_count = 0
        self.silhouette_fit_failure_count = 0
        self.latest_sphere_fit_rmse_mm = None
        self.latest_silhouette_fit_rmse_mm = None
        self.latest_center_method = None
        self.motion_rejection_count = 0

        self.motion_gate = BallMotionGate(
            max_speed_mps=max_ball_speed_mps,
            max_innovation_m=max_motion_innovation_m,
            max_prediction_sec=max_prediction_sec,
        )

        self.kf = KalmanFilter3D(dt=1.0 / 60.0)
        self.position_filters = [
            OneEuroFilter1D(
                min_cutoff=2.0, beta=5.0, derivative_cutoff=1.0
            )
            for _ in range(3)
        ]

        self.get_logger().info(
            f"球心方法: {self.position_method}, "
            f"球半径: {self.ball_radius * 1000.0:.1f} mm, "
            f"One Euro: {'开启' if self.use_one_euro_filter else '关闭'}, "
            f"运动门控: {'开启' if self.use_motion_gate else '关闭'}"
        )
        if self.use_motion_gate:
            self.get_logger().info(
                "运动门控阈值: "
                f"speed<={max_ball_speed_mps:.2f}m/s, "
                f"innovation<={max_motion_innovation_m:.3f}m, "
                f"prediction<={max_prediction_sec:.3f}s"
            )

        # 每秒统计一次成功获取图像的帧率和模型纯推理帧率。
        self.fps_window_start = time.perf_counter()
        self.capture_frame_count = 0
        self.inference_frame_count = 0
        self.inference_elapsed = 0.0

        # 创建定时器（ * Hz 处理）
        self.timer = self.create_timer(1/60, self.process_frame)

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

        if self.position_method == "silhouette":
            try:
                silhouette_fit = estimate_fixed_radius_sphere_from_mask(
                    mask,
                    intrinsics,
                    self.ball_radius,
                )
                self.latest_silhouette_fit_rmse_mm = (
                    silhouette_fit.contour_rmse_m * 1000.0
                )
                return silhouette_fit.center.copy(), "silhouette"
            except ValueError as error:
                self.silhouette_fit_failure_count += 1
                if (
                    self.silhouette_fit_failure_count <= 3
                    or self.silhouette_fit_failure_count % 60 == 0
                ):
                    self.get_logger().warn(
                        f"轮廓球心恢复失败，尝试深度回退: {error}"
                    )

        return self._estimate_depth_center(
            depth_image,
            depth_mask,
            intrinsics,
            fallback_center,
        )

    def process_frame(self):
        color_image, depth_image, frame_metadata = self.camera.get_images(
            return_metadata=True
        )
        if color_image is None or depth_image is None:
            return
        self.capture_frame_count += 1

        display_image = color_image.copy()

        capture_time_ns = frame_metadata['capture_time_ns']
        if capture_time_ns is None:
            capture_time = self.get_clock().now().to_msg()
        else:
            capture_time = TimeMsg(
                sec=int(capture_time_ns // 1_000_000_000),
                nanosec=int(capture_time_ns % 1_000_000_000),
            )
        # YOLO 推理
        inference_start = time.perf_counter()
        results = self.model.predict(source=color_image, 
                                    #  imgsz=640, 
                                     imgsz=(480, 768),
                                     conf=0.5, 
                                     classes=[0], 
                                     max_det=1, 
                                     verbose=False, 
                                     retina_masks=True)
        self.inference_elapsed += time.perf_counter() - inference_start
        self.inference_frame_count += 1

        now = time.perf_counter()
        fps_window_elapsed = now - self.fps_window_start
        if fps_window_elapsed >= 1.0:
            capture_fps = self.capture_frame_count / fps_window_elapsed
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
            self.get_logger().info(
                f"图像获取 FPS: {capture_fps:.2f}, "
                f"模型推理 FPS: {inference_fps:.2f} "
                f"({average_inference_ms:.2f} ms/帧)"
            )
            self.fps_window_start = now
            self.capture_frame_count = 0
            self.inference_frame_count = 0
            self.inference_elapsed = 0.0

        if len(results[0].boxes) > 0:
            max_conf_idx = results[0].boxes.conf.argmax().item()
            best_result = results[0][max_conf_idx]
            display_image = best_result.plot()
            box = best_result.boxes.xyxy[0].cpu().numpy()
            mask_data = largest_component_mask(
                best_result.masks.data[0].cpu().numpy().astype(bool)
            )
            depth_mask = make_inner_mask(mask_data)

            u, v = ball_center(mask_data)
            if u is None or v is None:
                u, v = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)

            # 深度表面点仅用于诊断和轮廓法失败时的回退。球心主估计
            # 不再因深度空洞而中断，也不依赖有系统偏差的 ZED 球面深度。
            surface_point = None
            masked_depths_m = (
                depth_image[depth_mask] * self.camera.depth_scale
            )
            valid_depths = masked_depths_m[
                np.isfinite(masked_depths_m)
                & (masked_depths_m > 0.05)
                & (masked_depths_m < 10.0)
            ]
            if len(valid_depths) > 10:
                real_x, real_y, real_z = self.camera.get_real_position(
                    u,
                    v,
                    window_size=9,
                    depth_frame=frame_metadata['depth_frame'],
                    intrinsics=frame_metadata['depth_intrinsics'],
                    mask=depth_mask,
                )
                if real_z is not None:
                    surface_point = np.array(
                        [real_x, real_y, real_z], dtype=np.float64
                    )

            fallback_center = (
                None
                if surface_point is None
                else compensate_ball_radius(
                    *surface_point.tolist(), self.ball_radius
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
                self.get_logger().warn(f"本帧无法恢复球心: {error}")
                raw_center = None
                center_method = None

            if raw_center is not None:
                capture_time_sec = (
                    capture_time.sec + capture_time.nanosec * 1e-9
                )
                measurement_accepted = True
                center = raw_center.copy()
                published_method = center_method
                if self.use_motion_gate:
                    motion_decision = self.motion_gate.update(
                        raw_center, capture_time_sec
                    )
                    measurement_accepted = motion_decision.accepted
                    center = motion_decision.output_position
                    if not motion_decision.accepted:
                        self.motion_rejection_count += 1
                        published_method = (
                            "motion-prediction"
                            if motion_decision.predicted
                            else "motion-rejected"
                        )
                        if (
                            self.motion_rejection_count <= 3
                            or self.motion_rejection_count % 20 == 0
                        ):
                            action = (
                                "发布短时预测"
                                if motion_decision.predicted
                                else "停止发布"
                            )
                            self.get_logger().warn(
                                "拒绝球心极值: "
                                f"reason={motion_decision.reason}, "
                                "speed="
                                f"{motion_decision.apparent_speed_mps:.2f}m/s, "
                                "innovation="
                                f"{motion_decision.innovation_m:.3f}m; "
                                f"{action}"
                            )

                if measurement_accepted:
                    self.latest_center_method = center_method
                    raw_center_x, raw_center_y, raw_center_z = (
                        raw_center.tolist()
                    )
                    raw_center_msg = PointStamped()
                    raw_center_msg.header.stamp = capture_time
                    raw_center_msg.header.frame_id = (
                        "camera_color_optical_frame"
                    )
                    raw_center_msg.point.x = raw_center_x
                    raw_center_msg.point.y = raw_center_y
                    raw_center_msg.point.z = raw_center_z
                    self.raw_center_pub.publish(raw_center_msg)

                    # 表面点来自相同掩码，只有球心通过门控时才发布，避免
                    # /ball_surface 保留同一错误检测产生的极值。
                    if surface_point is not None:
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

                if center is None:
                    cv2.putText(
                        display_image,
                        "Motion outlier - publish stopped",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                    )
                    center_method = published_method
                    continue_publishing = False
                else:
                    continue_publishing = True

                if continue_publishing and self.use_one_euro_filter:
                    center = np.array(
                        [
                            position_filter.filter(value, capture_time_sec)
                            for position_filter, value in zip(
                                self.position_filters, center
                            )
                        ],
                        dtype=np.float64,
                    )
                if continue_publishing:
                    center_x, center_y, center_z = center.tolist()

                    center_msg = PointStamped()
                    center_msg.header.stamp = capture_time
                    center_msg.header.frame_id = "camera_color_optical_frame"
                    center_msg.point.x = center_x
                    center_msg.point.y = center_y
                    center_msg.point.z = center_z
                    self.center_pub.publish(center_msg)

                    if (
                        measurement_accepted
                        and self.tracker is not None
                        and surface_point is not None
                    ):
                        self.tracker.update(
                            *surface_point.tolist(),
                            center_x,
                            center_y,
                            center_z,
                        )

                    annotated = display_image
                    cv2.circle(annotated, (u, v), 5, (0, 0, 255), -1)
                    cv2.putText(
                        annotated,
                        f"Center Z: {center_z * 1000.0:.1f}mm "
                        f"({published_method})",
                        (max(5, u - 80), max(25, v - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (
                            (0, 255, 0)
                            if measurement_accepted
                            else (0, 165, 255)
                        ),
                        2,
                    )
                    fit_rmse_mm = self.latest_silhouette_fit_rmse_mm
                    if fit_rmse_mm is None:
                        fit_rmse_mm = self.latest_sphere_fit_rmse_mm
                    if fit_rmse_mm is not None:
                        cv2.putText(
                            annotated,
                            f"Fit RMSE: {fit_rmse_mm:.2f}mm",
                            (
                                max(5, u - 80),
                                min(annotated.shape[0] - 10, v + 15),
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 255),
                            2,
                        )
                    display_image = annotated

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
        combined_display = cv2.hconcat([display_image, depth_display])
        cv2.imshow(
            f"{self.camera_name} - Detection + Depth",
            combined_display,
        )
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)

    def destroy_node(self):
        self.camera.stop()
        cv2.destroyAllWindows()
        if self.tracker is not None:
            self.get_logger().info("生成轨迹图...")
            # self.tracker.save_and_plot()
        super().destroy_node()


def main(args=None):
    cli_args, ros_args = parse_args(args)
    rclpy.init(args=ros_args)
    node = BallPublisher(
        camera_type=cli_args.camera,
        position_method=cli_args.position_method,
        ball_radius_m=cli_args.ball_radius_m,
        use_one_euro_filter=cli_args.use_one_euro_filter,
        use_motion_gate=not cli_args.disable_motion_gate,
        max_ball_speed_mps=cli_args.max_ball_speed_mps,
        max_motion_innovation_m=cli_args.max_motion_innovation_m,
        max_prediction_sec=cli_args.max_prediction_sec,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
