import threading
import time

import cv2
import numpy as np
import pyzed.sl as sl


class _CameraIntrinsics:
    """RealSense-like intrinsics object used by the shared detection code."""

    def __init__(
        self,
        fx,
        fy,
        ppx,
        ppy,
        width,
        height,
        coeffs=None,
        model="none",
    ):
        self.fx = float(fx)
        self.fy = float(fy)
        self.ppx = float(ppx)
        self.ppy = float(ppy)
        self.width = int(width)
        self.height = int(height)
        self.coeffs = (
            [0.0] * 5
            if coeffs is None
            else np.asarray(coeffs, dtype=np.float64).reshape(-1)[:5].tolist()
        )
        if len(self.coeffs) < 5:
            self.coeffs.extend([0.0] * (5 - len(self.coeffs)))
        self.model = model


class ZEDCamera:
    """ZED camera wrapper compatible with :class:`RealSenseCamera`.

    ZED depth is registered to the rectified left image and is returned in
    metres, so ``depth_scale`` is 1.0.  ``width`` and ``height`` are accepted
    for drop-in compatibility.  If that size is not a native ZED mode, the
    closest native mode is captured and both color and depth are resized
    together; the exported intrinsics are scaled to the returned image size.
    """

    _RESOLUTION_SIZES = (
        ("HD2K", 2208, 1242),
        ("HD1200", 1920, 1200),
        ("HD1080", 1920, 1080),
        ("HD720", 1280, 720),
        ("SVGA", 960, 600),
        ("VGA", 672, 376),
    )

    def __init__(
        self,
        width=None,
        height=None,
        fps=60,
        serial_number=None,
        resolution=None,
    ):
        """
        初始化 ZED 相机。

        参数:
            width: 返回图像宽度；与 RealSenseCamera 参数兼容
            height: 返回图像高度；与 RealSenseCamera 参数兼容
            fps: 相机帧率
            serial_number: 相机序列号
            resolution: 可选的 sl.RESOLUTION；指定后默认返回原生分辨率
        """
        if resolution is None:
            output_width = 640 if width is None else int(width)
            output_height = 480 if height is None else int(height)
            if output_width <= 0 or output_height <= 0:
                raise ValueError("width and height must be positive")
            resolution = self._select_resolution(output_width, output_height)
        else:
            if (width is None) != (height is None):
                raise ValueError(
                    "width and height must be provided together when "
                    "resolution is specified"
                )
            output_width = None if width is None else int(width)
            output_height = None if height is None else int(height)
            if (
                output_width is not None
                and (output_width <= 0 or output_height <= 0)
            ):
                raise ValueError("width and height must be positive")

        self.requested_width = output_width
        self.requested_height = output_height
        self.width = output_width or 0
        self.height = output_height or 0
        self.capture_width = 0
        self.capture_height = 0
        self.fps = int(fps)
        self.serial_number = serial_number
        self.resolution = resolution

        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = resolution
        self.init_params.camera_fps = self.fps
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        self.init_params.coordinate_units = sl.UNIT.METER
        # IMAGE uses the optical convention shared by RealSense:
        # +X right, +Y down, +Z forward.
        if hasattr(sl, "COORDINATE_SYSTEM") and hasattr(
            sl.COORDINATE_SYSTEM, "IMAGE"
        ):
            self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

        if serial_number:
            self.init_params.set_from_serial_number(int(serial_number))

        self.depth_scale = 1.0
        self.color_intrinsics = None
        self.depth_intrinsics = None
        self.camera_config = None

        self.lock = threading.Lock()
        self.thread = None
        self.stopped = False
        self._is_open = False

        self.latest_color_image = None
        self.latest_depth_image = None
        self.latest_capture_time_ns = None
        self.latest_frame_number = -1
        self.last_processed_frame_num = -1
        # Kept for compatibility with code that reads the old ZED attribute.
        self.last_timestamp = 0

    @classmethod
    def _select_resolution(cls, width, height):
        """Select the closest SDK resolution, prioritizing aspect ratio."""
        candidates = []
        target_aspect = float(width) / float(height)
        for name, candidate_width, candidate_height in cls._RESOLUTION_SIZES:
            value = getattr(sl.RESOLUTION, name, None)
            if value is None:
                continue
            candidate_aspect = candidate_width / candidate_height
            aspect_error = abs(np.log(candidate_aspect / target_aspect))
            size_error = abs(np.log(candidate_width / width)) + abs(
                np.log(candidate_height / height)
            )
            candidates.append((4.0 * aspect_error + size_error, value))

        if not candidates:
            auto_resolution = getattr(sl.RESOLUTION, "AUTO", None)
            if auto_resolution is None:
                raise RuntimeError("ZED SDK does not expose a usable resolution")
            return auto_resolution
        return min(candidates, key=lambda item: item[0])[1]

    def start(self):
        """启动相机及后台采集线程。"""
        if self._is_open:
            return True

        try:
            err = self.zed.open(self.init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"\033[91m相机启动失败: {repr(err)}\033[0m")
                return False
            self._is_open = True

            cam_info = self.zed.get_camera_information()
            actual_serial = getattr(cam_info, "serial_number", self.serial_number)
            self.serial_number = actual_serial
            self._initialize_intrinsics(cam_info)

            print(
                f"\033[92m相机serial_number=={actual_serial}   启动成功\033[0m"
            )
            print(f"\033[96m深度标尺: {self.depth_scale}\033[0m")
            self._configure_camera_settings()

            c_fx, c_fy, c_cx, c_cy = self.get_color_intrinsics()
            d_fx, d_fy, d_cx, d_cy = self.get_depth_intrinsics()
            self.camera_config = {
                "intrinsics": {
                    "color": {
                        "fx": c_fx,
                        "fy": c_fy,
                        "ppx": c_cx,
                        "ppy": c_cy,
                    },
                    "depth": {
                        "fx": d_fx,
                        "fy": d_fy,
                        "ppx": d_cx,
                        "ppy": d_cy,
                    },
                    "depth_scale": self.depth_scale,
                }
            }

            self.stopped = False
            self.thread = threading.Thread(
                target=self._update_frames, daemon=True
            )
            self.thread.start()
            time.sleep(0.5)
        except Exception as error:
            self.stopped = True
            if self._is_open:
                try:
                    self.zed.close()
                except Exception:
                    pass
                self._is_open = False
            print(f"\033[91m相机启动异常: {error}\033[0m")
            return False
        return True

    def _initialize_intrinsics(self, cam_info):
        camera_configuration = getattr(
            cam_info, "camera_configuration", None
        )
        if camera_configuration is None:
            raise RuntimeError("ZED SDK did not return camera configuration")

        resolution = getattr(camera_configuration, "resolution", None)
        if resolution is None:
            resolution = getattr(cam_info, "camera_resolution", None)
        if resolution is not None:
            self.capture_width = int(getattr(resolution, "width", 0))
            self.capture_height = int(getattr(resolution, "height", 0))

        calibration = camera_configuration.calibration_parameters
        left_camera = calibration.left_cam
        if self.capture_width <= 0:
            self.capture_width = int(getattr(left_camera, "image_size").width)
        if self.capture_height <= 0:
            self.capture_height = int(getattr(left_camera, "image_size").height)
        if self.capture_width <= 0 or self.capture_height <= 0:
            raise RuntimeError("ZED SDK returned an invalid capture resolution")

        if self.requested_width is None:
            self.width = self.capture_width
            self.height = self.capture_height
        else:
            self.width = self.requested_width
            self.height = self.requested_height

        scale_x = self.width / self.capture_width
        scale_y = self.height / self.capture_height
        fx = float(left_camera.fx) * scale_x
        fy = float(left_camera.fy) * scale_y
        if fx <= 0.0 or fy <= 0.0:
            raise RuntimeError("ZED SDK returned invalid focal lengths")
        # cv2.resize uses half-pixel centers.
        cx = (float(left_camera.cx) + 0.5) * scale_x - 0.5
        cy = (float(left_camera.cy) + 0.5) * scale_y - 0.5

        # VIEW.LEFT and MEASURE.DEPTH are rectified and share the same pinhole
        # projection, so their OpenCV distortion coefficients are zero.
        self.color_intrinsics = _CameraIntrinsics(
            fx, fy, cx, cy, self.width, self.height, model="none"
        )
        self.depth_intrinsics = _CameraIntrinsics(
            fx, fy, cx, cy, self.width, self.height, model="none"
        )
        self.c_fx, self.c_fy, self.c_cx, self.c_cy = fx, fy, cx, cy

    def _configure_camera_settings(self):
        """Apply the manual settings used by the original ZED wrapper."""
        settings = (
            (sl.VIDEO_SETTINGS.AEC_AGC, 0),
            (sl.VIDEO_SETTINGS.EXPOSURE, 80),
            (sl.VIDEO_SETTINGS.GAIN, 60),
        )
        for setting, value in settings:
            try:
                self.zed.set_camera_settings(setting, value)
            except Exception as error:
                # Settings vary by ZED model; image/depth capture remains usable
                # when an optional camera control is unavailable.
                print(
                    f"\033[93m无法设置 ZED 相机参数 {setting}: {error}\033[0m"
                )

    def stop(self):
        """停止后台线程并关闭相机。"""
        self.stopped = True
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        if self._is_open:
            try:
                self.zed.close()
            finally:
                self._is_open = False
        print("相机已停止")

    def __enter__(self):
        if not self.start():
            raise RuntimeError("ZED camera start failed")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _update_frames(self):
        """后台独立线程：原子地保存同一时刻的彩色图和对齐深度图。"""
        runtime_parameters = sl.RuntimeParameters()
        image = sl.Mat()
        depth = sl.Mat()

        while not self.stopped:
            try:
                if (
                    self.zed.grab(runtime_parameters)
                    != sl.ERROR_CODE.SUCCESS
                ):
                    time.sleep(0.001)
                    continue

                self.zed.retrieve_image(image, sl.VIEW.LEFT)
                self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                capture_time_ns = time.time_ns()

                color_image = np.asarray(image.get_data())
                depth_image = np.asarray(depth.get_data())
                if (
                    color_image.ndim != 3
                    or color_image.shape[2] < 3
                    or depth_image.ndim != 2
                ):
                    continue

                color_image = color_image[:, :, :3]
                if (
                    color_image.shape[1] != self.width
                    or color_image.shape[0] != self.height
                ):
                    color_image = cv2.resize(
                        color_image,
                        (self.width, self.height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    depth_image = cv2.resize(
                        depth_image,
                        (self.width, self.height),
                        interpolation=cv2.INTER_NEAREST,
                    )

                with self.lock:
                    self.latest_color_image = np.array(
                        color_image, copy=True, order="C"
                    )
                    self.latest_depth_image = np.array(
                        depth_image, dtype=np.float32, copy=True, order="C"
                    )
                    self.latest_capture_time_ns = capture_time_ns
                    self.latest_frame_number += 1
                    self.last_timestamp = capture_time_ns // 1_000_000
            except Exception as error:
                if not self.stopped:
                    print(f"后台读取帧异常: {error}")
                    time.sleep(0.01)

    def get_frames(self):
        """仅从内存中返回最新彩色图和深度图，不阻塞。"""
        with self.lock:
            color_image = (
                None
                if self.latest_color_image is None
                else self.latest_color_image.copy()
            )
            depth_image = (
                None
                if self.latest_depth_image is None
                else self.latest_depth_image.copy()
            )
        return color_image, depth_image

    def get_frame_bundle(self):
        """原子地返回同一组彩色图、深度图和主机采集时间。"""
        with self.lock:
            color_image = (
                None
                if self.latest_color_image is None
                else self.latest_color_image.copy()
            )
            depth_image = (
                None
                if self.latest_depth_image is None
                else self.latest_depth_image.copy()
            )
            return (
                color_image,
                depth_image,
                self.latest_capture_time_ns,
            )

    def get_images(self, return_metadata=False):
        """
        获取与 RealSenseCamera 相同形式的图像和逐帧元数据。

        ``depth_frame`` 是与返回彩色图严格同步的深度数组快照，单位为米。
        """
        with self.lock:
            if (
                self.latest_color_image is None
                or self.latest_depth_image is None
            ):
                if return_metadata:
                    return None, None, None
                return None, None

            frame_number = self.latest_frame_number
            if frame_number == self.last_processed_frame_num:
                if return_metadata:
                    return None, None, None
                return None, None
            self.last_processed_frame_num = frame_number

            color_image = self.latest_color_image.copy()
            depth_image = self.latest_depth_image.copy()
            capture_time_ns = self.latest_capture_time_ns

        if return_metadata:
            metadata = {
                "depth_frame": depth_image,
                "depth_intrinsics": self.depth_intrinsics,
                "capture_time_ns": capture_time_ns,
                "frame_number": frame_number,
            }
            return color_image, depth_image, metadata
        return color_image, depth_image

    @staticmethod
    def _depth_array(depth_frame):
        if depth_frame is None:
            return None
        if isinstance(depth_frame, np.ndarray):
            return depth_frame
        get_data = getattr(depth_frame, "get_data", None)
        if get_data is None:
            raise TypeError("depth_frame must be a numpy array or sl.Mat")
        return np.asarray(get_data())

    @staticmethod
    def _intrinsic_values(intrinsics):
        if intrinsics is None:
            raise ValueError("intrinsics are not initialized")
        ppx = getattr(intrinsics, "ppx", getattr(intrinsics, "cx", None))
        ppy = getattr(intrinsics, "ppy", getattr(intrinsics, "cy", None))
        if ppx is None or ppy is None:
            raise ValueError("intrinsics must provide ppx/ppy or cx/cy")
        return (
            float(intrinsics.fx),
            float(intrinsics.fy),
            float(ppx),
            float(ppy),
        )

    def get_real_position(
        self,
        u,
        v,
        window_size=9,
        depth_frame=None,
        intrinsics=None,
        mask=None,
    ):
        """用同帧深度在目标掩码内估计像素对应的相机坐标。"""
        if depth_frame is None:
            _, depth_frame = self.get_frames()
        depth_image = self._depth_array(depth_frame)
        if depth_image is None:
            return None, None, None

        depth_image = np.asarray(depth_image)
        if depth_image.ndim != 2:
            raise ValueError("depth_frame must be a two-dimensional depth map")
        mask_data = None if mask is None else np.asarray(mask, dtype=bool)
        if mask_data is not None and mask_data.shape != depth_image.shape:
            raise ValueError(
                f"depth/mask shape mismatch: {depth_image.shape} vs "
                f"{mask_data.shape}"
            )

        u = int(round(float(u)))
        v = int(round(float(v)))
        half_window = max(0, int(window_size) // 2)
        x_min = max(0, u - half_window)
        x_max = min(depth_image.shape[1], u + half_window + 1)
        y_min = max(0, v - half_window)
        y_max = min(depth_image.shape[0], v + half_window + 1)
        if x_min >= x_max or y_min >= y_max:
            return None, None, None

        window = (
            depth_image[y_min:y_max, x_min:x_max].astype(
                np.float64, copy=False
            )
            * self.depth_scale
        )
        valid = np.isfinite(window) & (window > 0.05) & (window < 10.0)
        if mask_data is not None:
            valid &= mask_data[y_min:y_max, x_min:x_max]
        depths = window[valid]
        if len(depths) == 0:
            print(
                f"\033[91m深度无效，像素区域 ({u}, {v}) 无可用深度\033[0m"
            )
            return None, None, None

        depth_median = np.median(depths)
        mad = np.median(np.abs(depths - depth_median))
        if mad > 1e-6:
            robust_sigma = 1.4826 * mad
            inliers = depths[
                np.abs(depths - depth_median) <= 2.5 * robust_sigma
            ]
            if len(inliers) >= max(5, len(depths) // 3):
                depths = inliers
        median_depth = float(np.median(depths))

        if intrinsics is None:
            intrinsics = self.depth_intrinsics
        fx, fy, cx, cy = self._intrinsic_values(intrinsics)
        return (
            float((u - cx) * median_depth / fx),
            float((v - cy) * median_depth / fy),
            median_depth,
        )

    def get_masked_point_cloud(
        self,
        depth_image,
        mask,
        intrinsics,
        max_points=2500,
        min_depth_m=0.05,
        max_depth_m=10.0,
    ):
        """将掩码内的有效深度像素反投影为相机光学坐标系点云。"""
        if depth_image is None or mask is None or intrinsics is None:
            return np.empty((0, 3), dtype=np.float64)

        depth_image = np.asarray(depth_image)
        mask = np.asarray(mask, dtype=bool)
        if depth_image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"depth/mask shape mismatch: {depth_image.shape} vs "
                f"{mask.shape}"
            )

        depth_m = depth_image.astype(np.float64, copy=False) * self.depth_scale
        valid = (
            mask
            & np.isfinite(depth_m)
            & (depth_m > float(min_depth_m))
            & (depth_m < float(max_depth_m))
        )
        rows, cols = np.nonzero(valid)
        if len(rows) == 0:
            return np.empty((0, 3), dtype=np.float64)

        max_points = max(1, int(max_points))
        if len(rows) > max_points:
            selected = np.linspace(
                0, len(rows) - 1, max_points, dtype=np.int64
            )
            rows = rows[selected]
            cols = cols[selected]

        fx, fy, cx, cy = self._intrinsic_values(intrinsics)
        z = depth_m[rows, cols]
        x = (cols.astype(np.float64) - cx) * z / fx
        y = (rows.astype(np.float64) - cy) * z / fy
        return np.column_stack((x, y, z))

    def get_point_cloud(self, depth_frame=None):
        """返回完整的 HxWx3 XYZ 点云矩阵（单位为米）。"""
        if depth_frame is None:
            _, depth_frame = self.get_frames()
        depth_image = self._depth_array(depth_frame)
        if depth_image is None:
            return None

        depth_m = (
            np.asarray(depth_image, dtype=np.float64) * self.depth_scale
        )
        rows, cols = np.indices(depth_m.shape)
        fx, fy, cx, cy = self._intrinsic_values(self.depth_intrinsics)
        x = (cols - cx) * depth_m / fx
        y = (rows - cy) * depth_m / fy
        points = np.stack((x, y, depth_m), axis=-1)
        invalid = (
            ~np.isfinite(depth_m)
            | (depth_m <= 0.05)
            | (depth_m >= 10.0)
        )
        points[invalid] = np.nan
        return points

    def deproject_to_3d(self, u, v, depth_m):
        """已知像素和米制深度，使用返回图像的内参反投影。"""
        if (
            depth_m is None
            or not np.isfinite(depth_m)
            or depth_m <= 0
            or self.color_intrinsics is None
        ):
            return None, None, None
        fx, fy, cx, cy = self._intrinsic_values(self.color_intrinsics)
        return (
            float((float(u) - cx) * depth_m / fx),
            float((float(v) - cy) * depth_m / fy),
            float(depth_m),
        )

    def get_color_intrinsics(self):
        """获取返回彩色图对应的内参。"""
        if self.color_intrinsics is None:
            raise RuntimeError("camera must be started before reading intrinsics")
        intrinsics = self.color_intrinsics
        print(
            "\033[96m彩图内参:"
            f"{intrinsics.fx}, {intrinsics.fy}, "
            f"{intrinsics.ppx}, {intrinsics.ppy}\033[0m"
        )
        return (
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.ppx,
            intrinsics.ppy,
        )

    def get_color_intrinsic_matrix(self):
        c_fx, c_fy, c_cx, c_cy = self.get_color_intrinsics()
        return np.array(
            [[c_fx, 0.0, c_cx], [0.0, c_fy, c_cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def get_color_distortion_coeffs(self):
        """返回 OpenCV 顺序的左目整流图畸变系数。"""
        if self.color_intrinsics is None:
            raise RuntimeError("camera must be started before reading distortion")
        return np.asarray(
            self.color_intrinsics.coeffs, dtype=np.float64
        ).reshape(1, 5)

    def get_depth_intrinsics(self):
        """获取与左目彩色图对齐的深度内参。"""
        if self.depth_intrinsics is None:
            raise RuntimeError("camera must be started before reading intrinsics")
        intrinsics = self.depth_intrinsics
        print(
            "\033[96m深度图内参:"
            f"{intrinsics.fx}, {intrinsics.fy}, "
            f"{intrinsics.ppx}, {intrinsics.ppy}\033[0m"
        )
        return (
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.ppx,
            intrinsics.ppy,
        )

    def display_images(self):
        """实时显示彩色图和深度图。"""
        try:
            while True:
                color_image, depth_image = self.get_images()
                if color_image is None or depth_image is None:
                    continue

                valid_depth = np.nan_to_num(
                    depth_image, nan=0.0, posinf=0.0, neginf=0.0
                )
                depth_display = np.clip(
                    valid_depth / 3.0 * 255.0, 0, 255
                ).astype(np.uint8)
                cv2.imshow("ZED - Color", color_image)
                cv2.imshow("ZED - Depth", depth_display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()

    def save_images(self, save_path):
        color_image, _ = self.get_images()
        if color_image is not None:
            cv2.imwrite(save_path, color_image)


def list_devices():
    """列出当前连接的 ZED 设备。"""
    devices = sl.Camera.get_device_list()
    print("检测到的ZED设备:")
    if not devices:
        print("\033[91m未检测到任何设备\033[0m")
    for index, device in enumerate(devices):
        print(
            f"\033[92m设备 {index}: {device.camera_model}, "
            f"序列号: {device.serial_number}, "
            f"状态: {device.camera_state}\033[0m"
        )


if __name__ == "__main__":
    list_devices()
    camera = ZEDCamera(resolution=sl.RESOLUTION.SVGA, fps=120)
    if not camera.start():
        print("等待相机启动或启动失败退出程序。")
        raise SystemExit(1)

    try:
        camera.display_images()
    finally:
        camera.stop()
