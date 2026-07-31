import sys
from pathlib import Path

DETECTION_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "mocap_bridge"
    / "scripts"
    / "detection"
)
sys.path.insert(0, str(DETECTION_DIR))

from device.realsense_camera import RealSenseCamera
import numpy as np
import cv2
import os



if __name__ == "__main__":
    import time
    import glob

    # serial_number()
    # list_camera_framerates()
    camera = RealSenseCamera(width=640, height=480)
    camera.start()
    # 固定保存到当前脚本所在目录，避免保存位置随启动目录变化。
    save_dir = Path(__file__).resolve().parent / "pic_red_ball"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建文件夹: {save_dir}")
        pic_counter = 1
    else:
        existing_files = glob.glob(os.path.join(save_dir, "pic*.jpg"))
        if existing_files:
            # 从文件名中提取数字部分，例如从 "pic012.jpg" 中提取 12
            # int(x.split('.')[0][-3:]) 的逻辑是：分割文件名，取最后一部分（数字），转为整数
            existing_numbers = [int(os.path.basename(x).split('.')[0][-3:]) for x in existing_files]
            # 计数器从最大编号的下一个开始
            pic_counter = max(existing_numbers) + 1
            print(f"检测到现有图片，将从 pic{pic_counter:03d}.jpg 开始保存")
        else:
            # 文件夹存在但为空，从1开始
            pic_counter = 1
    is_recording = False
    last_save_time = 0.0

    while True:
        color_image1, depth_image1 = camera.get_images()
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

        if is_recording:
            current_time = time.time()
            if current_time - last_save_time >= 0.1:  # 0.1s 间隔
                # 格式化文件名，如 pic001.jpg
                filename = os.path.join(save_dir, f"pic{pic_counter:03d}.jpg")
                # 保存彩色图片 (如果需要存深度图也可在此处增加 cv2.imwrite)
                cv2.imwrite(filename, color_image1)
                print(f"已保存: {filename}")
                pic_counter += 1
                last_save_time = current_time
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if not is_recording:
                print("\033[92m=== 开始记录图片 ===\033[0m")
                is_recording = True
                last_save_time = time.time() - 0.1  # 确保按下立刻保存第一帧
        elif key == ord('s'):
            if is_recording:
                print("\033[93m=== 停止记录图片 ===\033[0m")
                is_recording = False
        elif key == ord('q'):
            print("退出程序...")
            break

    camera.stop()
    cv2.destroyAllWindows()
