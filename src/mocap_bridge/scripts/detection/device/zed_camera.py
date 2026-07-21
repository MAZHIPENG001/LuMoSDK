import pyzed.sl as sl
import numpy as np
import cv2
import threading
import time


class ZEDCamera:
    def __init__(self, resolution=sl.RESOLUTION.SVGA, fps=120, serial_number=None):
        """
        初始化 ZED 相机
        参数:
            resolution: 图像分辨率 (sl.RESOLUTION 枚举, 对应ZED X推荐HD1200/HD1080/SVGA)
            fps: 帧率 (15, 30, 60 等)
            serial_number: 相机序列号
        """
        self.fps = fps
        self.serial_number = serial_number

        # 创建 ZED 相机对象和初始化配置
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = resolution
        self.init_params.camera_fps = fps

        # 深度配置 (ULTRA模式提供最高质量的深度图)
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        # 单位设为米，与原代码中提取的3D坐标单位保持一致
        self.init_params.coordinate_units = sl.UNIT.METER

        if serial_number:
            self.init_params.set_from_serial_number(int(serial_number))

        # 多线程同步控制
        self.lock = threading.Lock()
        self.thread = None
        self.stopped = False

        # 用于存储 numpy 数据
        self.latest_color_image = None
        self.latest_depth_image = None
        self.latest_point_cloud = None
        self.last_timestamp = 0

    def start(self):
        """启动相机"""
        try:
            err = self.zed.open(self.init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"\033[91m相机启动失败: {repr(err)}\033[0m")
                return False

            cam_info = self.zed.get_camera_information()
            actual_serial = cam_info.serial_number
            print(f"\033[92m相机serial_number=={actual_serial}   启动成功\033[0m")

            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 80)
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 60)

            # --- 修改部分开始 ---
            # 相机内参提取并保存为类属性
            self.c_fx, self.c_fy, self.c_cx, self.c_cy = self.get_color_intrinsics()

            # 添加 depth_scale，为了与 eval_ros.py 兼容 (ZED 深度已设置为米，因此 scale 为 1.0)
            self.depth_scale = 1.0
            # --- 修改部分结束 ---

            # 启动后台线程
            self.stopped = False
            self.thread = threading.Thread(target=self._update_frames, daemon=True)
            self.thread.start()

            # 短暂等待，确保第一帧数据加载完成
            time.sleep(0.5)

        except Exception as e:
            print(f"\033[91m相机启动异常: {e}\033[0m")
            return False
        return True

    def stop(self):
        """停止相机"""
        self.stopped = True

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        self.zed.close()
        print("相机已停止")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _update_frames(self):
        """后台独立线程：不断获取最新帧"""
        runtime_parameters = sl.RuntimeParameters()

        # 创建 ZED 矩阵用于接收数据
        image = sl.Mat()
        depth = sl.Mat()
        point_cloud = sl.Mat()

        while not self.stopped:
            # 抓取最新帧
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # 检索左视图彩色图像、深度图和 3D 点云
                self.zed.retrieve_image(image, sl.VIEW.LEFT)
                self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)

                with self.lock:
                    # 获取 numpy 副本 (ZED 默认彩色图是 BGRA 格式)
                    # 直接去掉 Alpha 通道以便和原版 cv2.imshow(BGR) 完全兼容
                    self.latest_color_image = image.get_data()[:, :, :3].copy()
                    self.latest_depth_image = depth.get_data().copy()
                    self.latest_point_cloud = point_cloud.get_data().copy()
                    self.last_timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()

    def get_images(self):
        """
        获取彩色图像和深度图像数组
        返回:
            color_image: 彩色图像 (BGR格式)
            depth_image: 深度图像 (浮点数, 单位米)
        """
        with self.lock:
            if self.latest_color_image is None or self.latest_depth_image is None:
                return None, None
            return self.latest_color_image.copy(), self.latest_depth_image.copy()

    def get_real_position(self, u, v, window_size=5):
        """
        获取指定像素点附近的有效深度值，并转换为真实坐标 (使用 ZED 的 XYZ 矩阵直接获取)
        """
        with self.lock:
            if self.latest_point_cloud is None:
                return None, None, None
            pc_data = self.latest_point_cloud.copy()

        height, width = pc_data.shape[:2]
        half_w = window_size // 2
        xs, ys, zs = [], [], []

        for i in range(-half_w, half_w + 1):
            for j in range(-half_w, half_w + 1):
                if 0 <= u + i < width and 0 <= v + j < height:
                    # pc_data 在每个像素包含 [X, Y, Z, 占位符]
                    point = pc_data[v + j, u + i]
                    x, y, z = point[0], point[1], point[2]

                    # ZED 对于无效深度 (例如太近或太远) 会返回 np.nan 或者极值
                    if not np.isnan(z) and not np.isinf(z) and 0.05 < z < 10.0:
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)

        if not zs:
            print(f"\33[91m深度无效，像素区域 ({u}, {v}) 无可用深度\33[0m")
            return None, None, None

        # 取中位数来减小误差
        return np.median(xs), np.median(ys), np.median(zs)

    def get_point_cloud(self):
        """
        返回完整的 XYZ 点云矩阵
        """
        with self.lock:
            if self.latest_point_cloud is None:
                return None
            return self.latest_point_cloud.copy()

    def deproject_to_3d(self, u, v, depth_m):
        """
        已知像素坐标 (u, v) 和真实深度（米），利用针孔模型内参反算 3D 坐标。
        以此兼容 RealSense 的 API。
        """
        if depth_m <= 0 or not hasattr(self, 'c_fx'):
            return None, None, None

        # 根据针孔相机模型反算 3D 坐标
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        x = (u - self.c_cx) * depth_m / self.c_fx
        y = (v - self.c_cy) * depth_m / self.c_fy
        z = float(depth_m)

        return float(x), float(y), float(z)

    def get_color_intrinsics(self):
        """获取左眼相机内参"""
        cam_info = self.zed.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters.left_cam

        c_fx = calib.fx
        c_fy = calib.fy
        c_cx = calib.cx
        c_cy = calib.cy
        print(f'\033[96m左镜头内参:{c_fx}, {c_fy}, {c_cx}, {c_cy}\033[0m')

        return c_fx, c_fy, c_cx, c_cy

    def display_images(self):
        """实时显示彩色和深度图像"""
        try:
            while True:
                color_image, depth_image = self.get_images()
                if color_image is None or depth_image is None:
                    continue

                # 将浮点深度图(米)映射到8位以供显示
                # ZED 深度通常以浮点数返回(例如 0.5代表0.5米)，所以要稍微调整归一化逻辑
                # 假设可视化最大距离为 3.0 米
                max_dist = 3.0
                depth_display = np.clip(depth_image / max_dist * 255, 0, 255).astype(np.uint8)

                cv2.imshow('ZED - Color', color_image)
                cv2.imshow('ZED - Depth', depth_display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
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
    """列出当前连接的ZED设备"""
    devices = sl.Camera.get_device_list()
    print("检测到的ZED设备:")
    if not devices:
        print("\33[91m未检测到任何设备\33[0m")
    for i, dev in enumerate(devices):
        print(f"\33[92m设备 {i}: {dev.camera_model}, 序列号: {dev.serial_number}, 状态: {dev.camera_state}\33[0m")


if __name__ == "__main__":
    list_devices()

    # 使用 ZED X 支持的 HD720 分辨率 (1280x720) HD1200 HD1080 SVGA
    camera = ZEDCamera(resolution=sl.RESOLUTION.SVGA, fps=120)

    if not camera.start():
        print("等待相机启动或启动失败退出程序。")
        exit(1)

    try:
        while True:
            color_img, depth_img = camera.get_images()
            if color_img is None or depth_img is None:
                continue

            # 将ZED以米为单位的深度图转化为0-255用于显示
            # 这里将0-3米内的物体映射到较好对比度
            max_visual_dist = 3.0
            depth_display = np.clip((depth_img / max_visual_dist) * 255, 0, 255).astype(np.uint8)

            # 使用伪彩色
            depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            cv2.imshow('ZED - Color1', color_img)
            cv2.imshow('ZED - Depth1', depth_colormap)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()