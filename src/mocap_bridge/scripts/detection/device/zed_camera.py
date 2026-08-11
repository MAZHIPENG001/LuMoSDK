import threading
import time

import cv2
import numpy as np
import pyzed.sl as sl


_EXPECTED_SDK_VERSION = "5.0.7"

# Native per-eye ZED X modes.  On SDK 5.0.7 these values are filtered through
# is_resolution_available()/is_FPS_available() before being reported.
_ZED_X_STEREO_PROFILES = (
    ("HD1200", 1920, 1200, (15, 30, 60)),
    ("HD1080", 1920, 1080, (15, 30, 60)),
    ("SVGA", 960, 600, (15, 30, 60, 120)),
)


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
    """Jetson-local ZED X wrapper compatible with :class:`RealSenseCamera`.

    ZED depth is registered to the rectified left image and is returned in
    metres, so ``depth_scale`` is 1.0.  ``width`` and ``height`` are accepted
    for drop-in compatibility.  If that size is not a native ZED X mode, the
    closest native mode is captured and both color and depth are resized
    together; the exported intrinsics are scaled to the returned image size.
    """

    _RESOLUTION_SIZES = (
        ("HD1200", 1920, 1200),
        ("HD1080", 1920, 1080),
        ("SVGA", 960, 600),
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
            serial_number: Jetson 本机 GMSL2 相机序列号
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
        self.capture_fps = 0
        self.camera_model = None
        self.fps = int(fps)
        self.serial_number = serial_number
        self.resolution = resolution

        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = resolution
        self.init_params.camera_fps = self.fps
        # self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        # self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT
        self.init_params.coordinate_units = sl.UNIT.METER
        # IMAGE uses the optical convention shared by RealSense:
        # +X right, +Y down, +Z forward.
        if hasattr(sl, "COORDINATE_SYSTEM") and hasattr(
            sl.COORDINATE_SYSTEM, "IMAGE"
        ):
            self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

        if serial_number is not None:
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
            if err > sl.ERROR_CODE.SUCCESS:
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
            print("\033[96m输入源: Jetson 本机 ZED Link/GMSL2\033[0m")
            print(
                "\033[96m实际相机配置: "
                f"{_model_name(self.camera_model)}, "
                f"{self.capture_width}x{self.capture_height} "
                f"@ {self.capture_fps} FPS\033[0m"
            )
            print(f"\033[96m深度标尺: {self.depth_scale}\033[0m")
            self._configure_camera_settings()

            c_fx, c_fy, c_cx, c_cy = self.get_color_intrinsics()
            d_fx, d_fy, d_cx, d_cy = self.get_depth_intrinsics()
            self.camera_config = {
                "camera": {
                    "model": _model_name(self.camera_model),
                    "serial_number": self.serial_number,
                    "input_mode": "gmsl2",
                    "capture_width": self.capture_width,
                    "capture_height": self.capture_height,
                    "capture_fps": self.capture_fps,
                },
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
        self.camera_model = getattr(cam_info, "camera_model", "未知")
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
        self.capture_fps = int(getattr(camera_configuration, "fps", 0))

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
                if self.zed.grab(runtime_parameters) > sl.ERROR_CODE.SUCCESS:
                    time.sleep(0.001)
                    continue

                self.zed.retrieve_image(image, sl.VIEW.LEFT)
                self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                # The SDK IMAGE timestamp is the sensor frame timestamp in
                # host-clock Epoch nanoseconds.  time.time_ns() here would
                # instead mark the end of grab/retrieve and adds a variable
                # depth-processing delay to moving-ball comparisons.
                image_timestamp = self.zed.get_timestamp(
                    sl.TIME_REFERENCE.IMAGE
                )
                get_nanoseconds = getattr(
                    image_timestamp, "get_nanoseconds", None
                )
                if get_nanoseconds is not None:
                    capture_time_ns = int(get_nanoseconds())
                else:
                    capture_time_ns = int(
                        image_timestamp.get_milliseconds() * 1_000_000
                    )
                if capture_time_ns <= 0:
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


def _model_name(camera_model):
    """在不同 PyZED 枚举实现之间稳定返回型号名称。"""
    enum_name = getattr(camera_model, "name", None)
    if enum_name:
        return str(enum_name)
    model_enum = getattr(sl, "MODEL", None)
    if model_enum is not None:
        for known_name in ("ZED_X", "ZED_XM"):
            known_value = getattr(model_enum, known_name, None)
            if known_value is not None and camera_model == known_value:
                return known_name
    return str(camera_model).rsplit(".", 1)[-1]


def _is_zed_x(camera_model):
    model_enum = getattr(sl, "MODEL", None)
    if model_enum is not None:
        for name in ("ZED_X", "ZED_XM"):
            value = getattr(model_enum, name, None)
            if value is not None and camera_model == value:
                return True
    return _model_name(camera_model).upper() in {"ZED_X", "ZED_XM"}


def _zed_x_profiles(camera_model):
    """使用 SDK 5.0.7 能力接口过滤 ZED X 官方候选模式。"""
    resolution_checker = getattr(sl, "is_resolution_available", None)
    fps_checker = getattr(sl, "is_fps_available", None)
    if fps_checker is None:
        # SDK 5.0.x 的部分 Python wheel 使用大写 FPS 的函数名。
        fps_checker = getattr(sl, "is_FPS_available", None)

    profiles = []
    for name, width, height, fallback_fps in _ZED_X_STEREO_PROFILES:
        resolution = getattr(sl.RESOLUTION, name, None)
        if resolution is None:
            continue

        resolution_verified = False
        if callable(resolution_checker):
            try:
                if not resolution_checker(resolution, camera_model):
                    continue
                resolution_verified = True
            except (AttributeError, TypeError, ValueError):
                pass

        fps_values = []
        fps_verified = False
        if callable(fps_checker):
            try:
                fps_values = [
                    fps
                    for fps in fallback_fps
                    if fps_checker(fps, resolution, camera_model)
                ]
                fps_verified = True
            except (AttributeError, TypeError, ValueError):
                fps_values = []
        if not fps_values and not fps_verified:
            fps_values = list(fallback_fps)
        elif fps_verified and not fps_values:
            continue

        profiles.append(
            {
                "resolution": name,
                "width": width,
                "height": height,
                "fps": list(fps_values),
                "verified_by_sdk": resolution_verified and fps_verified,
            }
        )
    return profiles


def list_devices():
    """列出 Jetson 本机 ZED Link/GMSL2 设备。"""
    devices = sl.Camera.get_device_list()
    print("Jetson 检测到的 ZED Link/GMSL2 设备:")
    if not devices:
        print("\033[91m未检测到 ZED X\033[0m")
    for index, device in enumerate(devices):
        print(
            f"\033[92m设备 {index}: {_model_name(device.camera_model)}, "
            f"序列号: {device.serial_number}, "
            f"状态: {device.camera_state}\033[0m"
        )


def list_camera_capabilities(serial_number=None):
    """列举 Jetson 本机 ZED X，并用 SDK 5.0.7 验证可用视频模式。"""
    sdk_version = str(sl.Camera.get_sdk_version())
    requested_serial = (
        None if serial_number is None else str(serial_number)
    )
    result = {"sdk_version": sdk_version, "devices": []}

    print(f"ZED SDK版本: {sdk_version}")
    if not sdk_version.startswith(_EXPECTED_SDK_VERSION):
        print(
            "\033[93m警告: 此程序按 ZED SDK 5.0.7 编写，当前版本为 "
            f"{sdk_version}\033[0m"
        )
    print("Jetson 本机 ZED Link/GMSL2 设备:")

    for index, device in enumerate(sl.Camera.get_device_list()):
        device_serial = str(getattr(device, "serial_number", "未知"))
        if requested_serial is not None and device_serial != requested_serial:
            continue

        camera_model = getattr(device, "camera_model", "未知")
        model_name = _model_name(camera_model)
        supported = _is_zed_x(camera_model)
        profiles = _zed_x_profiles(camera_model) if supported else []
        device_id = getattr(device, "id", None)
        device_info = {
            "index": index,
            "camera_model": model_name,
            "serial_number": device_serial,
            "camera_state": str(
                getattr(device, "camera_state", "未知")
            ),
            "connection": "Jetson 本机 ZED Link/GMSL2",
            "gmsl_port": device_id,
            "supported": supported,
            "profiles": profiles,
        }
        result["devices"].append(device_info)

        print(f"\n\033[92m设备 {index}: {model_name}\033[0m")
        print(f"  序列号: {device_serial}")
        print(f"  状态: {device_info['camera_state']}")
        print(f"  连接方式: {device_info['connection']}")
        if device_id is not None:
            print(f"  GMSL端口/设备ID: {device_id}")

        if not supported:
            print("  \033[91m此封装只支持双目 ZED X/ZED X Mini\033[0m")
        elif profiles:
            verified = all(
                profile["verified_by_sdk"] for profile in profiles
            )
            source = "SDK 5.0.7 验证" if verified else "官方规格回退"
            print(f"  支持的视频模式（{source}，单目输出尺寸）:")
            for profile in profiles:
                size = f"{profile['width']}x{profile['height']}"
                print(
                    f"    {profile['resolution']:<8} "
                    f"{size:<10} 支持帧率: {profile['fps']}"
                )
        else:
            print("  \033[91mSDK 未返回可用的 ZED X 视频模式\033[0m")

    if not result["devices"]:
        if requested_serial is None:
            print("\033[91m未检测到 Jetson 本机 ZED X\033[0m")
        else:
            print(f"\033[91m未找到序列号 {requested_serial}\033[0m")

    return result


if __name__ == "__main__":
    inventory = list_camera_capabilities()
    supported_devices = [
        device for device in inventory["devices"] if device["supported"]
    ]
    if not supported_devices:
        raise SystemExit(1)

    selected_device = supported_devices[0]
    camera = ZEDCamera(
        resolution=sl.RESOLUTION.SVGA,
        fps=120,
        serial_number=selected_device["serial_number"],
    )
    if not camera.start():
        print("ZED X 启动失败，程序退出。")
        raise SystemExit(1)

    fps_sample_frame = None
    fps_sample_time_ns = None
    actual_fps = 0.0
    fps_update_interval_ns = 500_000_000

    try:
        while True:
            color_image, depth_image, metadata = camera.get_images(
                return_metadata=True
            )
            if color_image is None or depth_image is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            frame_number = metadata["frame_number"]
            capture_time_ns = metadata["capture_time_ns"]
            if fps_sample_frame is None:
                fps_sample_frame = frame_number
                fps_sample_time_ns = capture_time_ns
            else:
                elapsed_ns = capture_time_ns - fps_sample_time_ns
                if elapsed_ns >= fps_update_interval_ns:
                    actual_fps = (
                        (frame_number - fps_sample_frame)
                        * 1_000_000_000.0
                        / elapsed_ns
                    )
                    fps_sample_frame = frame_number
                    fps_sample_time_ns = capture_time_ns

            cv2.putText(
                color_image,
                f"Actual FPS: {actual_fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            valid_mask = np.isfinite(depth_image) & (depth_image > 0.0)
            valid_depth = np.where(valid_mask, depth_image, 0.0)
            depth_display = np.clip(
                valid_depth / 3.0 * 255.0, 0, 255
            ).astype(np.uint8)
            depth_display = cv2.applyColorMap(
                depth_display, cv2.COLORMAP_TURBO
            )
            depth_display[~valid_mask] = 0
            cv2.imshow("ZED - Color", color_image)
            cv2.imshow("ZED - Colorized Depth", depth_display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        camera.stop()
