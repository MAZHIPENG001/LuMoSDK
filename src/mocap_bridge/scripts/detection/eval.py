import cv2
from ultralytics import YOLO
from device.realsense_camera import RealSenseCamera
import time
from record.target_tracker import TargetTracker
import numpy as np


def ball_center_hsv(image, box, lower_color, upper_color):
    """
    阶段 2：使用 HSV 颜色分割对 YOLO 的检测框进行微调，提取精确的球心像素坐标。

    参数:
        image: 原始彩色图像 (BGR)
        box: YOLO 预测的边界框 [x1, y1, x2, y2]
        lower_color: HSV 颜色范围下限 (numpy array)
        upper_color: HSV 颜色范围上限 (numpy array)

    返回:
        u, v: 精确的像素坐标
    """
    x1, y1, x2, y2 = map(int, box[:4])
    h_img, w_img = image.shape[:2]

    # 确保坐标在图像边界内，防止数组越界
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    # 提取感兴趣区域 (ROI)
    roi = image[y1:y2, x1:x2]
    return int((x1 + x2) / 2), int((y1 + y2) / 2)
    # 防御性检查：如果 ROI 无效，直接返回 YOLO 框中心
    if roi.size == 0:
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    # 转换到 HSV 空间并生成二值掩码
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, lower_color, upper_color)

    # 计算图像矩 (Moments) 来寻找质心
    M = cv2.moments(mask)
    if M["m00"] > 0:
        # 局部质心坐标
        local_u = int(M["m10"] / M["m00"])
        local_v = int(M["m01"] / M["m00"])
        # 映射回全局像素坐标并返回
        return x1 + local_u, y1 + local_v
    else:
        # 如果掩码未匹配到颜色（例如受强光干扰或遮挡），回退到 YOLO 框几何中心
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

def compensate_ball_radius(surf_x, surf_y, surf_z, ball_radius=0.115):
    """
    足球半径补偿模块：将相机获取的表面 3D 坐标推算为真实的球心坐标。

    注意：此函数假设输入的 (surf_x, surf_y, surf_z) 是在【相机坐标系】下的坐标
    （即相机光心为原点 [0,0,0]）。

    参数:
        surf_x, surf_y, surf_z: 相机测得的表面点坐标
        ball_radius: 足球的物理半径 (默认 0.115m)

    返回:
        center_x, center_y, center_z: 补偿后的球心坐标
    """
    # 构建相机到表面的向量
    P_surf = np.array([surf_x, surf_y, surf_z])

    # 计算距离 (模长)
    dist_to_surf = np.linalg.norm(P_surf)

    # 防御性检查，避免除以 0
    if dist_to_surf <= 0.01:
        return surf_x, surf_y, surf_z

    # 根据相似三角形/向量比例关系，将坐标延伸至球心
    scale_factor = (dist_to_surf + ball_radius) / dist_to_surf
    P_center = P_surf * scale_factor

    return float(P_center[0]), float(P_center[1]), float(P_center[2])

def main():
    # model_path = "/home/ma/GithubDoc/ultralytics/runs/segment/ball_seg/seg_model_v1/weights/best.pt"
    # model_path = "/home/ma/GithubDoc/ultralytics/my_model/model/yolo26n-seg/best.pt"
    model_path = "./model/box/yolo26l-seg/best.engine"
    print(f"正在加载模型: {model_path}")
    model = YOLO(model_path)

    print("正在启动 RealSense 相机...")
    camera = RealSenseCamera(width=640, height=480)
    camera.start()

    # print("正在初始化轨迹记录器...")
    # tracker = TargetTracker()

    lower_ball_color = np.array([10, 100, 100])
    upper_ball_color = np.array([40, 255, 255])

    # SOCCER_BALL_RADIUS = 0.11
    SOCCER_BALL_RADIUS = 0.115


    try:
        while True:
            t0 = time.time()
            color_image, depth_image = camera.get_images()
            if color_image is None:
                time.sleep(0.005)
                continue

            t1 = time.time()
            conf=0.5 # 置信度大于 50%
            results = model.predict(source=color_image, conf=conf, verbose=False)
            t2 = time.time()
            if len(results[0].boxes) > 0:
                max_conf_idx = results[0].boxes.conf.argmax().item()
                best_result = results[0][max_conf_idx]
                annotated_image = best_result.plot()

                # box = best_result.boxes.xyxy[0]
                box = best_result.boxes.xyxy[0].cpu().numpy()
                u, v = ball_center_hsv(color_image, box, lower_ball_color, upper_ball_color)
                # u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
                real_x, real_y, real_z = camera.get_real_position(u, v)
                if real_z is not None:
                    center_x, center_y, center_z = compensate_ball_radius(
                        real_x, real_y, real_z, ball_radius=SOCCER_BALL_RADIUS
                    )
                    # tracker.update(real_x, real_y, real_z, center_x, center_y, center_z)
                    cv2.circle(annotated_image, (u, v), 5, (0, 0, 255), -1)
                    text = f"real_z: {real_z:.4f}m"
                    cv2.putText(annotated_image, text, (u - 20, v - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                annotated_image = color_image.copy()
            cv2.imshow("Detection", annotated_image)
            t3 = time.time()
            # 按 'q' 键退出循环

            # print(f"获取图像时间:{t1 - t0}")
            # print(f"模型推理时间:{t2 - t1}")
            # print(f"结果标记时间:{t3 - t2}\n")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("exit...")
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("\n正在生成轨迹图和数据文件...")
        # tracker.save_and_plot()


if __name__ == "__main__":
    main()