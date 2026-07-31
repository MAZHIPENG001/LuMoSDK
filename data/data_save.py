import argparse
import re
import sys
import time
from pathlib import Path


DETECTION_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "mocap_bridge"
    / "scripts"
    / "detection"
)
DEFAULT_SAVE_DIR = Path(__file__).resolve().parent / "pic_red_ball"
IMAGE_PATTERN = re.compile(r"^pic(\d+)\.jpg$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集 RealSense 彩色图像。")
    parser.add_argument(
        "save_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help=f"图片保存目录（默认：{DEFAULT_SAVE_DIR}）",
    )
    return parser.parse_args()


def next_image_number(save_dir: Path) -> int:
    existing_numbers = []
    for image_path in save_dir.iterdir():
        match = IMAGE_PATTERN.fullmatch(image_path.name)
        if match:
            existing_numbers.append(int(match.group(1)))

    return max(existing_numbers, default=0) + 1


def main() -> None:
    args = parse_args()
    save_dir = args.save_dir.expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(DETECTION_DIR))
    import cv2
    from device.realsense_camera import RealSenseCamera

    pic_counter = next_image_number(save_dir)
    if pic_counter == 1:
        print(f"图片将保存到: {save_dir}")
    else:
        print(f"检测到现有图片，将从 pic{pic_counter:03d}.jpg 开始保存")

    camera = RealSenseCamera(width=640, height=480)
    camera.start()
    is_recording = False
    last_save_time = 0.0

    try:
        while True:
            color_image, depth_image = camera.get_images()

            if color_image is None or depth_image is None:
                continue

            # 将 16 位深度图映射到 8 位，以便显示。
            depth_display = cv2.convertScaleAbs(depth_image, alpha=0.03)

            cv2.imshow("RealSense - Color", color_image)
            cv2.imshow("RealSense - Depth", depth_display)

            if is_recording:
                current_time = time.time()
                if current_time - last_save_time >= 0.1:
                    image_path = save_dir / f"pic{pic_counter:03d}.jpg"
                    cv2.imwrite(str(image_path), color_image)
                    print(f"已保存: {image_path}")
                    pic_counter += 1
                    last_save_time = current_time

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c") and not is_recording:
                print("\033[92m=== 开始记录图片 ===\033[0m")
                is_recording = True
                last_save_time = time.time() - 0.1
            elif key == ord("s") and is_recording:
                print("\033[93m=== 停止记录图片 ===\033[0m")
                is_recording = False
            elif key == ord("q"):
                print("退出程序...")
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
