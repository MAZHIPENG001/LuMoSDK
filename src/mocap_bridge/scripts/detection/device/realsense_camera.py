import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import time

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=60, serial_number=None):
        """
        初始化RealSense相机
        参数:
            width: 图像宽度
            height: 图像高度
            fps: 帧率
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.serial_number = serial_number
        # 创建管道和配置
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # 如果有指定序列号，则只连接该设备
        if serial_number:
            self.config.enable_device(serial_number)
        else:
            self.serial_number = rs.camera_info.serial_number
        # 配置流
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # 对齐工具（将深度图对齐到彩色图）
        self.align = rs.align(rs.stream.color)

        # 初始化后处理滤波器
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()

        # 深度颜色化工具
        self.colorizer = rs.colorizer()

        self.profile = self.pipeline.start(self.config)
        self.depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
        self.color_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        self.depth_intrinsics = self.depth_profile.get_intrinsics()
        self.color_intrinsics = self.color_profile.get_intrinsics()

        # 多线程
        self.lock = threading.Lock()
        self.thread = None
        self.stopped = False
        self.latest_color_frame = None
        self.latest_depth_frame = None
        self.latest_capture_time_ns = None
        self.last_processed_frame_num = -1

    def start(self):
        """启动相机"""
        try:
            # self.profile = self.pipeline.start(self.config)
            print(f"\033[92m相机serial_number=={self.serial_number}   启动成功\033[0m")

            # 获取深度传感器和深度标尺
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"\033[96m深度标尺: {self.depth_scale}\033[0m")
            sensors = self.profile.get_device().query_sensors()
            for sensor in sensors:
                sensor_name = sensor.get_info(rs.camera_info.name)
                # 1. 设置深度传感器 (Stereo Module)
                if 'Stereo Module' in sensor_name:
                    # 关闭深度图自动曝光
                    sensor.set_option(rs.option.enable_auto_exposure, 0)
                    # 设置深度曝光时间 (单位通常为微秒，例如 5000 = 5毫秒)
                    # 运动较快时建议设置在 3000 - 6000 之间
                    sensor.set_option(rs.option.exposure, 5000)
                    # 还可以稍微提高深度激光发射器(增益)功率来弥补曝光缩短带来的深度缺失
                    if sensor.supports(rs.option.laser_power):
                        sensor.set_option(rs.option.laser_power, 200)  # 默认通常是150，可调高到200-300
                # 2. 设置彩色传感器 (RGB Camera)
                elif 'RGB' in sensor_name:
                    # 关闭彩色图自动曝光
                    sensor.set_option(rs.option.enable_auto_exposure, 0)
                    # 设置彩色曝光时间
                    # 注意：部分 RealSense 型号 RGB 曝光单位是 1/10000 秒 (即 100 = 10毫秒)
                    # 如果你使用的是 60fps[cite: 2]，每帧最大时间只有 16.6ms。建议设置在 80 左右 (8ms)
                    sensor.set_option(rs.option.exposure, 80)

                    # 曝光降低会导致图像变暗，需要提高增益 (Gain) 来补偿
                    if sensor.supports(rs.option.gain):
                        sensor.set_option(rs.option.gain, 64)
            # 相机内参
            c_fx, c_fy, c_cx, c_cy = self.get_color_intrinsics()
            d_fx, d_fy, d_cx, d_cy = self.get_depth_intrinsics()
            self.camera_config = {
                'intrinsics': {
                    'color': {
                        'fx': c_fx,
                        'fy': c_fy,
                        'ppx': c_cx,
                        'ppy': c_cy,
                    },
                    'depth': {
                        'fx': d_fx,
                        'fy': d_fy,
                        'ppx': d_cx,
                        'ppy': d_cy,
                    },
                    'depth_scale': self.depth_scale
                }
            }
            # 启动后台线程
            self.stopped = False
            self.thread = threading.Thread(target=self._update_frames, daemon=True)
            self.thread.start()
            # 短暂等待，确保第一帧数据加载完成
            time.sleep(0.5)
        except Exception as e:
            print(f"\033[91m相机启动失败: {e}\033[0m")
            return False
        return True
    def stop(self):
        """停止相机"""
        # 改变标志位，后台线程跳出 while 循环
        self.stopped = True

        # 阻塞主线程，等待后台线程安全结束 (设置 1 秒超时防止死锁)
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        # 后台线程已退出,关闭相机管道
        try:
            self.pipeline.stop()
        except Exception as e:
            pass  # 忽略重复关闭可能引发的错误

        print("相机已停止")
    def __enter__(self):
        self.start()
        return self

    def _update_frames(self):
        """后台独立线程：以相机原生帧率不断获取最新帧"""
        while not self.stopped:
            try:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)

                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not depth_frame or not color_frame:
                    continue

                # 应用滤波器
                filtered_frame = self.spatial_filter.process(depth_frame)
                # filtered_frame = self.temporal_filter.process(filtered_frame)
                # 将基础 frame 转换回 depth_frame
                depth_frame = filtered_frame.as_depth_frame()

                # 重要：调用 keep() 防止被 SDK 自动释放
                color_frame.keep()
                depth_frame.keep()

                # 加锁更新最新帧
                # 使用系统时钟记录这一对对齐帧到达主机的时刻。该时钟与默认
                # ROS system time 同源，便于和动捕发布端的接收时间进行对齐。
                capture_time_ns = time.time_ns()
                with self.lock:
                    self.latest_color_frame = color_frame
                    self.latest_depth_frame = depth_frame
                    self.latest_capture_time_ns = capture_time_ns

            except Exception as e:
                if not self.stopped:
                    print(f"后台读取帧异常: {e}")

    def get_frames(self):
        """仅从内存中返回最新帧，不阻塞"""
        with self.lock:
            return self.latest_color_frame, self.latest_depth_frame

    def get_frame_bundle(self):
        """原子地返回同一组彩色帧、对齐深度帧和主机采集时间。"""
        with self.lock:
            return (
                self.latest_color_frame,
                self.latest_depth_frame,
                self.latest_capture_time_ns,
            )

    def get_images(self, return_metadata=False):
        """
        获取对齐后的图像数组
        参数:
            return_metadata: 为 True 时额外返回与图像严格对应的深度帧、
                深度图内参和主机采集时间。默认 False 以兼容现有调用。
        返回:
            color_image: 彩色图像 (BGR格式)
            depth_image: 深度图像 (16位)
            metadata（可选）: depth_frame、depth_intrinsics、capture_time_ns
        """
        color_frame, depth_frame, capture_time_ns = self.get_frame_bundle()

        if color_frame is None or depth_frame is None:
            if return_metadata:
                return None, None, None
            return None, None

        current_frame_num = color_frame.get_frame_number()
        if self.last_processed_frame_num == current_frame_num:
            if return_metadata:
                return None, None, None
            return None, None
        self.last_processed_frame_num = current_frame_num

        # 转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        if return_metadata:
            # 深度已通过 rs.align 对齐到彩色流，必须使用当前“对齐后深度帧”
            # 自身携带的内参，不能再使用原始 depth stream 的内参。
            depth_intrinsics = (
                depth_frame.profile.as_video_stream_profile().get_intrinsics()
            )
            metadata = {
                'depth_frame': depth_frame,
                'depth_intrinsics': depth_intrinsics,
                'capture_time_ns': capture_time_ns,
            }
            return color_image, depth_image, metadata

        return color_image, depth_image

    def get_real_position(
        self,
        u,
        v,
        window_size=9,
        depth_frame=None,
        intrinsics=None,
        mask=None,
    ):
        """
        获取指定像素点附近的有效深度值，并转换为真实坐标。

        depth_frame 应传入产生该像素坐标的彩色图对应的对齐深度帧，避免
        YOLO 推理期间后台线程更新深度后发生跨帧取值。
        """
        if depth_frame is None:
            _, depth_frame = self.get_frames()
        if depth_frame is None:
            return None, None, None

        # 在目标掩码内截取中心点附近的局部窗口。球面中心附近的深度最适合
        # 后续沿视线方向进行半径补偿，同时可以避免窗口混入背景/地面。
        half_w = window_size // 2
        depths = []

        for dy in range(-half_w, half_w + 1):
            for dx in range(-half_w, half_w + 1):
                px = u + dx
                py = v + dy
                if not (0 <= px < self.width and 0 <= py < self.height):
                    continue
                if mask is not None and not mask[py, px]:
                    continue
                d = depth_frame.get_distance(px, py)
                if 0.05 < d < 10.0:
                    depths.append(d)

        # 如果这个区域内找不到任何有效深度
        if not depths:
            print(f"\33[91m深度无效，像素区域 ({u}, {v}) 无可用深度\33[0m")
            return None, None, None

        # 先用 MAD 剔除飞点，再取中位数。与均值相比，中位数对 RealSense
        # 在球面边缘产生的空洞、背景穿透和离群深度更稳定。
        depths = np.asarray(depths, dtype=np.float64)
        depth_median = np.median(depths)
        mad = np.median(np.abs(depths - depth_median))
        if mad > 1e-6:
            robust_sigma = 1.4826 * mad
            inliers = depths[np.abs(depths - depth_median) <= 2.5 * robust_sigma]
            if len(inliers) >= max(5, len(depths) // 3):
                depths = inliers
        median_depth = float(np.median(depths))

        # 用中位数深度反算真实 3D 坐标
        if intrinsics is None:
            intrinsics = (
                depth_frame.profile.as_video_stream_profile().get_intrinsics()
            )
        camera_coordinate = rs.rs2_deproject_pixel_to_point(
            intrinsics, [u, v], median_depth
        )
        # print(f"\033[1;93m像素: ({u}, {v}) -> 真实坐标 (米): X={camera_coordinate[0]:.3f}, Y={camera_coordinate[1]:.3f}, Z={camera_coordinate[2]:.3f}\033[0m")
        return camera_coordinate[0], camera_coordinate[1], camera_coordinate[2]
    def get_point_cloud(self, depth_frame=None):
        """
        生成点云数据
        参数:
            depth_frame: 深度帧，如果为None则获取新帧
        返回:
            vertices: 点云顶点数组
        """
        if depth_frame is None:
            _, depth_frame= self.get_frames()
            if depth_frame is None:
                return None

        # 创建点云对象
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)

        return points
    def deproject_to_3d(self, u, v, depth_m):
        """
        已知像素坐标 (u, v) 和真实深度（米），利用内参反算 3D 坐标
        """
        if depth_m <= 0 or not hasattr(self, 'color_intrinsics'):
            return None, None, None

        # 调用 realsense SDK 的反投影函数
        point_3d = rs.rs2_deproject_pixel_to_point(
            self.color_intrinsics,
            [float(u), float(v)],
            float(depth_m)
        )
        return point_3d[0], point_3d[1], point_3d[2]

    def get_color_intrinsics(self):
        """获取相机内参"""
        c_fx = self.color_intrinsics.fx
        c_fy = self.color_intrinsics.fy
        c_cx = self.color_intrinsics.ppx
        c_cy = self.color_intrinsics.ppy
        print(f'\033[96m彩图内参:{c_fx}, {c_fy}, {c_cx}, {c_cy}\033[0m')
        return c_fx, c_fy, c_cx, c_cy
    def get_color_intrinsic_matrix(self):
        c_fx, c_fy, c_cx, c_cy = self.get_color_intrinsics()
        intrinsic_matrix = np.array([[c_fx, 0, c_cx],
                                     [0, c_fy, c_cy],
                                     [0, 0, 1]])
        return intrinsic_matrix
    def get_color_distortion_coeffs(self):
        """返回 OpenCV 顺序的彩色相机畸变系数 [k1, k2, p1, p2, k3]。"""
        return np.asarray(self.color_intrinsics.coeffs, dtype=np.float64).reshape(1, 5)
    def get_depth_intrinsics(self):
        d_fx = self.depth_intrinsics.fx
        d_fy = self.depth_intrinsics.fy
        d_cx = self.depth_intrinsics.ppx
        d_cy = self.depth_intrinsics.ppy
        print(f'\033[96m深度图内参:{d_fx}, {d_fy}, {d_cx}, {d_cy}\033[0m')
        return d_fx, d_fy, d_cx, d_cy

    def display_images(self):
        """实时显示彩色和深度图像"""
        try:
            while True:
                color_image, depth_image = self.get_images()
                if color_image is None or depth_image is None:
                    continue

                # 显示图像
                cv2.imshow('RealSense - Color', color_image)
                cv2.imshow('RealSense - Depth', depth_image)

                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()

    def save_images(self, save_path):
        color_image, _= self.get_images()
        cv2.imwrite(save_path, color_image)

def serial_number():
    ctx = rs.context()
    devices = ctx.query_devices()

    print("检测到的设备:")
    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        print(f"\33[92m设备 {i}: {name}, 序列号: {serial}\33[0m")

def list_camera_framerates(serial_number=None):
    """
    列出相机支持的分辨率和帧率组合
    """
    def list_stream_profiles(device):
        """
        列出设备的流配置
        """
        # 获取所有传感器
        sensors = device.query_sensors()

        for sensor in sensors:
            print(f"  \n传感器类型: {sensor.get_info(rs.camera_info.name)}")

            # 获取传感器支持的所有流配置
            stream_profiles = sensor.get_stream_profiles()

            # 按流类型和分辨率分组
            profiles_by_res = {}

            for profile in stream_profiles:
                # 将profile转换为视频流profile
                if profile.is_video_stream_profile():
                    vprofile = profile.as_video_stream_profile()

                    # 获取流类型
                    stream_type = str(vprofile.stream_type())

                    # 获取分辨率
                    width = vprofile.width()
                    height = vprofile.height()
                    res_key = f"{stream_type}_{width}x{height}"

                    if res_key not in profiles_by_res:
                        profiles_by_res[res_key] = []

                    # 获取帧率
                    fps = vprofile.fps()
                    if fps not in profiles_by_res[res_key]:
                        profiles_by_res[res_key].append(fps)

            # 打印结果
            for res_key, fps_list in profiles_by_res.items():
                stream_type, resolution = res_key.split('_')
                fps_list.sort()
                print(f"    流类型: {stream_type:<15} 分辨率: {resolution:<10} 支持帧率: {fps_list}")

    ctx = rs.context()

    devices = ctx.query_devices()
    for i, dev in enumerate(devices):
        print(f"\n\33[92m设备 {i + 1}:\33[0m")
        print(f"  名称: {dev.get_info(rs.camera_info.name)}")
        print(f"  序列号: {dev.get_info(rs.camera_info.serial_number)}")
        list_stream_profiles(dev)


if __name__ == "__main__":
    import time
    serial_number()
    list_camera_framerates()
    camera1 = RealSenseCamera(width=640, height=480)
    # 加载相机
    # camera1 = RealSenseCamera(width=1280, height=720, serial_number="233622070932")
    # camera2 = RealSen6seCamera(width=640, height=480, serial_number="938422074612")
    # while not camera2.start() and not camera1.start():
    #     print(f"\33[93m等待相机启动...\33[0m")
    #     time.sleep(0.2)  # 避免过度占用 CPU
    while True:
        color_image1, depth_image1 = camera1.get_images()
        # color_image2, depth_image2 = camera2.get_images()

        if color_image1 is None or depth_image1 is None:
            continue

        # 1. 将16位深度图映射到8位 (0-255)
        # alpha 缩放因子：0.03 左右通常能让 0-3米 范围内的物体有较好的对比度
        depth_display = cv2.convertScaleAbs(depth_image1, alpha=0.03)

        # 2. 应用伪彩色（COLORMAP_JET 效果类似于常用的红色表示近，蓝色表示远）
        # depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
        # 显示图像
        cv2.imshow('RealSense - Color1', color_image1)
        cv2.imshow('RealSense - Depth1', depth_display)
        # cv2.imshow('RealSense - Color2', color_image2)

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
