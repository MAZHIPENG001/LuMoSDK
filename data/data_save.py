import argparse
import math
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


def positive_frequency(value: str) -> float:
    try:
        frequency = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("频率必须是数字。") from error

    if not math.isfinite(frequency) or frequency <= 0:
        raise argparse.ArgumentTypeError("频率必须是大于 0 的有限数字。")
    return frequency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集 RealSense 彩色图像。")
    parser.add_argument(
        "save_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help=f"图片保存目录（默认：{DEFAULT_SAVE_DIR}）",
    )
    parser.add_argument(
        "-f",
        "--frequency",
        type=positive_frequency,
        default=10.0,
        metavar="HZ",
        help="图片保存频率，单位为 Hz（默认：10）",
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
    save_interval = 1.0 / args.frequency

    sys.path.insert(0, str(DETECTION_DIR))
    import cv2
    from device.realsense_camera import RealSenseCamera

    pic_counter = next_image_number(save_dir)
    if pic_counter == 1:
        print(f"图片将保存到: {save_dir}")
    else:
        print(f"检测到现有图片，将从 pic{pic_counter:03d}.jpg 开始保存")
    print(
        f"保存频率: {args.frequency:g} Hz"
        f"（最小间隔 {save_interval:.3f} 秒）"
    )

    camera = RealSenseCamera(width=640, height=480)
    camera.start()
    is_recording = False
    next_save_time = 0.0

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
                current_time = time.monotonic()
                if current_time >= next_save_time:
                    image_path = save_dir / f"pic{pic_counter:03d}.jpg"
                    cv2.imwrite(str(image_path), color_image)
                    print(f"已保存: {image_path}")
                    pic_counter += 1

                    # 按固定时间线推进；处理较慢时跳过错过的时刻，不重复保存同一帧。
                    next_save_time += save_interval
                    if next_save_time <= current_time:
                        missed_intervals = (
                            int((current_time - next_save_time) / save_interval) + 1
                        )
                        next_save_time += missed_intervals * save_interval

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c") and not is_recording:
                print("\033[92m=== 开始记录图片 ===\033[0m")
                is_recording = True
                next_save_time = time.monotonic()
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
