#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from mocap_bridge.msg import MocapData
from geometry_msgs.msg import PointStamped

class MultiSubscriber(Node):
    def __init__(self):
        super().__init__('multi_subscriber')
        # 1. 订阅动捕数据
        self.mocap_sub = self.create_subscription(
            MocapData,
            'mocap_data',
            self.mocap_callback,
            10
        )
        # 2. 订阅视觉表面点
        self.surf_sub = self.create_subscription(
            PointStamped,
            '/ball_surface',
            self.surface_callback,
            10
        )
        # 3. 订阅视觉球心
        self.center_sub = self.create_subscription(
            PointStamped,
            '/ball_center',
            self.center_callback,
            10
        )

        # 打印提示
        self.get_logger().info("已启动多源订阅节点：动捕 + 视觉")

    def mocap_callback(self, msg):
        """处理动捕数据"""
        # 足球位置（Marker ID=1）
        for marker in msg.markers:
            if marker.marker_id == 1:
                self.get_logger().info(
                    f"[动捕] 足球位置: X={marker.x:.3f}, Y={marker.y:.3f}, Z={marker.z:.3f}"
                )
        for rb in msg.rigid_bodies:
            # 相机刚体（Rigid ID=4）
            if rb.rigid_id == 4:
                if rb.is_track:
                    self.get_logger().info(
                        f"[动捕] 相机位姿: X={rb.x:.3f}, Y={rb.y:.3f}, Z={rb.z:.3f}  "
                        f"四元数: QX={rb.qx:.3f}, QY={rb.qy:.3f}, QZ={rb.qz:.3f}, QW={rb.qw:.3f}"
                    )
                else:
                    self.get_logger().warn("[动捕] 相机追踪丢失")
            # box刚体（Rigid ID=5）
            if rb.rigid_id == 5:
                if rb.is_track:
                    self.get_logger().info(
                        f"[动捕] box位姿: X={rb.x:.3f}, Y={rb.y:.3f}, Z={rb.z:.3f}  "
                        f"四元数: QX={rb.qx:.3f}, QY={rb.qy:.3f}, QZ={rb.qz:.3f}, QW={rb.qw:.3f}"
                    )
                else:
                    self.get_logger().warn("[动捕] 相机追踪丢失")

    def surface_callback(self, msg):
        """处理视觉表面点"""
        self.get_logger().info(
            f"[视觉-表面] 足球表面点 (相机坐标系): "
            f"X={msg.point.x:.3f}, Y={msg.point.y:.3f}, Z={msg.point.z:.3f}"
        )

    def center_callback(self, msg):
        """处理视觉球心"""
        self.get_logger().info(
            f"[视觉-球心] 足球球心 (补偿后): "
            f"X={msg.point.x:.3f}, Y={msg.point.y:.3f}, Z={msg.point.z:.3f}"
        )

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