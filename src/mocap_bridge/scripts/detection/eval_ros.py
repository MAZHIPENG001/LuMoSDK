#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import cv2
from ultralytics import YOLO
from device.realsense_camera import RealSenseCamera
import time
from record.target_tracker import TargetTracker
import numpy as np
# from device.zed_camera import ZEDCamera
# import pyzed.sl as sl

# 建议加上 half=True 开启半精度(FP16)推理，速度更快
# yolo export model=/home/ma/GithubDoc/ultralytics/my_model/model/red_ball/yolo26l-seg/best.pt format=engine task=segment half=True

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


class BallPublisher(Node):
    def __init__(self):
        super().__init__('ball_publisher')
        # 创建两个发布者：表面点与球心
        self.surf_pub = self.create_publisher(PointStamped, '/ball_surface', 10)
        self.center_pub = self.create_publisher(PointStamped, '/ball_center', 10)

        # 加载模型
        model_path = "./model/red_ball/yolo26l-seg/best.engine"
        self.get_logger().info(f"加载模型: {model_path}")
        self.model = YOLO(model_path,task='segment')

        # 启动相机
        self.get_logger().info("启动 RealSense 相机...")
        self.camera = RealSenseCamera(width=640, height=480)
        # self.camera = RealSenseCamera(width=640, height=480,fps=60)
        # self.camera = ZEDCamera(resolution=sl.RESOLUTION.HD720, fps=60)
        self.camera.start()

        # 轨迹记录（可选，保留）
        self.tracker=None
        self.tracker = TargetTracker()

        self.ball_radius = 0.115

        self.kf = KalmanFilter3D(dt=1.0 / 60.0)

        # 创建定时器（ * Hz 处理）
        self.timer = self.create_timer(1/60, self.process_frame)

    def process_frame(self):
        capture_time = self.get_clock().now().to_msg()
        color_image, depth_image = self.camera.get_images()
        if color_image is None:
            return
        # YOLO 推理
        # t0=time.time()
        results = self.model.predict(source=color_image, conf=0.5, verbose=False, retina_masks=True)
        # t1=time.time()
        # print(t1-t0)
        if len(results[0].boxes) > 0:
            max_conf_idx = results[0].boxes.conf.argmax().item()
            best_result = results[0][max_conf_idx]
            box = best_result.boxes.xyxy[0].cpu().numpy()
            mask_data = best_result.masks.data[0].cpu().numpy().astype(bool)

            mask_uint8 = mask_data.astype(np.uint8)
            kernel = np.ones((5, 5), np.uint8)  # 可根据实际模糊程度调整核大小
            eroded_mask = cv2.erode(mask_uint8, kernel, iterations=1)
            mask_data = eroded_mask.astype(bool)

            mask_display = (mask_data * 255).astype(np.uint8)
            cv2.imshow("YOLO Seg Mask", mask_display)
            u, v = ball_center(mask_data)
            if u is None and v is None:
                u, v = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            # --- 基于全局掩码的鲁棒深度采样 ---
            # 提取掩码覆盖范围内原始深度值(单位:m)
            masked_depths_m = depth_image[mask_data] * self.camera.depth_scale
            # 过滤无效深度（0 值）和极端飞点（> 10米）
            valid_depths = masked_depths_m[(masked_depths_m > 0.05) & (masked_depths_m < 10.0)]
            if len(valid_depths) > 10:  # 确保有足够的有效像素点
                # # 取 10% 分位数，既能过滤噪点，又能锁定球面最前端的深度
                # surface_z = np.percentile(valid_depths, 10)
                # # 获取 3D 坐标
                # real_x, real_y, real_z = self.camera.deproject_to_3d(u, v, surface_z)
                real_x, real_y, real_z = self.camera.get_real_position(u, v, window_size=5)
                # 获取 3D 坐标
                # real_x, real_y, real_z = self.camera.get_real_position(u,v)
                if real_z is not None:
                    # 表面点
                    surf_msg = PointStamped()
                    surf_msg.header.stamp = capture_time
                    surf_msg.header.frame_id = "camera_link"   # 坐标系名称
                    surf_msg.point.x, surf_msg.point.y, surf_msg.point.z = real_x, real_y, real_z
                    self.surf_pub.publish(surf_msg)


                    # 球心
                    center_x, center_y, center_z = compensate_ball_radius(real_x, real_y, real_z, self.ball_radius)
                    # raw_center_x, raw_center_y, raw_center_z = compensate_ball_radius(real_x, real_y, real_z, self.ball_radius)
                    # self.kf.predict()
                    # z_measurement = np.array([raw_center_x, raw_center_y, raw_center_z])
                    # center_x, center_y, center_z = self.kf.update(z_measurement)


                    center_msg = PointStamped()
                    center_msg.header.stamp = capture_time
                    center_msg.header.frame_id = "camera_link"
                    center_msg.point.x, center_msg.point.y, center_msg.point.z = center_x, center_y, center_z
                    self.center_pub.publish(center_msg)

                    # 记录数据
                    if self.tracker is not None:
                        self.tracker.update(real_x, real_y, real_z, center_x, center_y, center_z)

                    # 可视化
                    annotated = best_result.plot()
                    cv2.circle(annotated, (u, v), 5, (0, 0, 255), -1)
                    cv2.putText(annotated, f"z: {real_z*1000:.6f}mm", (u-20, v-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.putText(annotated, f"Center Z: {center_z * 1000:.6f}mm", (u-20, v+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow("Detection", annotated)
            else:
                self.get_logger().warn("掩码内无有效深度点！")
                cv2.imshow("Detection", best_result.plot())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(0)
        else:
            cv2.imshow("Detection", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(0)

    def destroy_node(self):
        self.camera.stop()
        cv2.destroyAllWindows()
        if self.tracker is not None:
            self.get_logger().info("生成轨迹图...")
            self.tracker.save_and_plot()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BallPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()