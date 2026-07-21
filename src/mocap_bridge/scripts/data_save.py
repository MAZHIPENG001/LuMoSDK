#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from mocap_bridge.msg import MocapData
from geometry_msgs.msg import PointStamped
import os
import csv
from datetime import datetime

class MultiSubscriber(Node):
    def __init__(self):
        super().__init__('multi_subscriber')

        # 创建数据目录（上级目录）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_base_dir = os.path.join(script_dir, 'data')
        os.makedirs(data_base_dir, exist_ok=True)

        # 生成时间戳（启动时刻），用于文件夹命名
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 创建本次运行对应的子文件夹
        self.run_dir = os.path.join(data_base_dir, self.timestamp)
        os.makedirs(self.run_dir, exist_ok=True)

        # 为三个话题分别建立 CSV 文件（文件名不再带时间戳）
        self.mocap_file = open(os.path.join(self.run_dir, 'mocap.csv'), 'w', newline='')
        self.surf_file  = open(os.path.join(self.run_dir, 'surface.csv'), 'w', newline='')
        self.raw_center_file = open(os.path.join(self.run_dir, 'center_raw.csv'), 'w', newline='')
        self.center_file = open(os.path.join(self.run_dir, 'center.csv'), 'w', newline='')

        self.mocap_writer = csv.writer(self.mocap_file)
        self.surf_writer  = csv.writer(self.surf_file)
        self.raw_center_writer = csv.writer(self.raw_center_file)
        self.center_writer = csv.writer(self.center_file)
        self.warned_legacy_mocap_timestamp = False

        # 写入表头
        self.mocap_writer.writerow(['timestamp_sec', 'timestamp_nanosec', 'marker_id', 'x', 'y', 'z',
                                    'rigid_id', 'rx', 'ry', 'rz', 'qx', 'qy', 'qz', 'qw', 'is_track',
                                    'source_frame_id', 'source_timestamp'])
        self.surf_writer.writerow(['timestamp_sec', 'timestamp_nanosec', 'x', 'y', 'z'])
        self.raw_center_writer.writerow(['timestamp_sec', 'timestamp_nanosec', 'x', 'y', 'z'])
        self.center_writer.writerow(['timestamp_sec', 'timestamp_nanosec', 'x', 'y', 'z'])

        # 订阅话题（保持不变）
        self.mocap_sub = self.create_subscription(
            MocapData,
            'mocap_data',
            self.mocap_callback,
            10
        )
        self.surf_sub = self.create_subscription(
            PointStamped,
            '/ball_surface',
            self.surface_callback,
            10
        )
        self.raw_center_sub = self.create_subscription(
            PointStamped,
            '/ball_center_raw',
            self.raw_center_callback,
            10
        )
        self.center_sub = self.create_subscription(
            PointStamped,
            '/ball_center',
            self.center_callback,
            10
        )

        self.get_logger().info(f"已启动多源订阅节点，数据保存至 {self.run_dir}")

    def mocap_callback(self, msg):
        # 使用发布节点在 ReceiveData 返回后立即记录的 ROS 时间，避免把 DDS
        # 传输和本订阅回调排队延迟混入动捕时间戳。兼容尚未重新编译的旧消息。
        if (
            hasattr(msg, 'header')
            and (msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0)
        ):
            sec = msg.header.stamp.sec
            nsec = msg.header.stamp.nanosec
        else:
            now = self.get_clock().now()
            sec, nsec = now.seconds_nanoseconds()
            if not self.warned_legacy_mocap_timestamp:
                self.get_logger().warn(
                    '收到的 MocapData 没有发布端时间戳，暂时使用订阅回调时间。'
                    '请重新 colcon build 并 source install/setup.bash。'
                )
                self.warned_legacy_mocap_timestamp = True

        for marker in msg.markers:
            if marker.marker_id == 1:
                self.mocap_writer.writerow([
                    sec, nsec,
                    marker.marker_id,
                    marker.x, marker.y, marker.z,
                    '', '', '', '', '', '', '', '', '',
                    msg.frame_id, msg.timestamp
                ])
                self.get_logger().info(
                    f"[动捕] 反光球位置: X={marker.x:.3f}, Y={marker.y:.3f}, Z={marker.z:.3f}"
                )

        for rb in msg.rigid_bodies:
            if rb.rigid_id == 4:
                self.mocap_writer.writerow([
                    sec, nsec,
                    '', '', '', '',
                    rb.rigid_id,
                    rb.x, rb.y, rb.z,
                    rb.qx, rb.qy, rb.qz, rb.qw,
                    1 if rb.is_track else 0,
                    msg.frame_id, msg.timestamp
                ])
                if rb.is_track:
                    self.get_logger().info(
                        f"[动捕] 相机位姿: X={rb.x:.3f}, Y={rb.y:.3f}, Z={rb.z:.3f}  "
                        f"四元数: QX={rb.qx:.3f}, QY={rb.qy:.3f}, QZ={rb.qz:.3f}, QW={rb.qw:.3f}"
                    )
                else:
                    self.get_logger().warn("[动捕] 相机追踪丢失")
            if rb.rigid_id == 5:
                self.mocap_writer.writerow([
                    sec, nsec,
                    '', '', '', '',
                    rb.rigid_id,
                    rb.x, rb.y, rb.z,
                    rb.qx, rb.qy, rb.qz, rb.qw,
                    1 if rb.is_track else 0,
                    msg.frame_id, msg.timestamp
                ])
                if rb.is_track:
                    self.get_logger().info(
                        f"[动捕] 红球球心: X={rb.x:.3f}, Y={rb.y:.3f}, Z={rb.z:.3f}  "
                        f"四元数: QX={rb.qx:.3f}, QY={rb.qy:.3f}, QZ={rb.qz:.3f}, QW={rb.qw:.3f}"
                    )
                else:
                    self.get_logger().warn("[动捕] 红球刚体追踪丢失")
        self.mocap_file.flush()

    def surface_callback(self, msg):
        sec = msg.header.stamp.sec
        nsec = msg.header.stamp.nanosec
        self.surf_writer.writerow([sec, nsec, msg.point.x, msg.point.y, msg.point.z])
        self.get_logger().info(
            f"[视觉-表面] 足球表面点 (相机坐标系): "
            f"X={msg.point.x:.3f}, Y={msg.point.y:.3f}, Z={msg.point.z:.3f}"
        )
        self.surf_file.flush()

    def center_callback(self, msg):
        sec = msg.header.stamp.sec
        nsec = msg.header.stamp.nanosec
        self.center_writer.writerow([sec, nsec, msg.point.x, msg.point.y, msg.point.z])
        self.get_logger().info(
            f"[视觉-球心] 足球球心 (补偿后): "
            f"X={msg.point.x:.3f}, Y={msg.point.y:.3f}, Z={msg.point.z:.3f}"
        )
        self.center_file.flush()

    def raw_center_callback(self, msg):
        sec = msg.header.stamp.sec
        nsec = msg.header.stamp.nanosec
        self.raw_center_writer.writerow([
            sec, nsec, msg.point.x, msg.point.y, msg.point.z
        ])
        self.raw_center_file.flush()

    def destroy_node(self):
        for f in [
            self.mocap_file,
            self.surf_file,
            self.raw_center_file,
            self.center_file,
        ]:
            if f and not f.closed:
                f.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MultiSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
