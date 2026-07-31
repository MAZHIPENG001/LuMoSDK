import argparse
import re
from pathlib import Path


DEFAULT_IMAGE_DIR = Path("pic")
IMAGE_PATTERN = re.compile(r"^pic(\d+)\.jpg$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按原编号顺序将 pic数字.jpg 重命名为连续的四位编号。"
    )
    parser.add_argument(
        "image_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="待重命名的图片目录（默认：当前工作目录下的 pic）",
    )
    return parser.parse_args()


def rename_images_sequentially(directory: Path) -> None:
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        print(f"错误：找不到目录 '{directory}'，请检查路径是否正确。")
        return

    image_files = []
    for image_path in directory.iterdir():
        match = IMAGE_PATTERN.fullmatch(image_path.name)
        if match:
            image_files.append((int(match.group(1)), image_path))

    if not image_files:
        print("在指定目录下未找到符合 'pic数字.jpg' 格式的文件。")
        return

    image_files.sort(key=lambda item: item[0])
    print(f"共找到 {len(image_files)} 个图片文件，开始重命名...")

    for new_index, (_, old_path) in enumerate(image_files, start=1):
        new_path = directory / f"pic{new_index:04d}.jpg"
        if old_path == new_path:
            continue
        if new_path.exists():
            print(f"跳过: {new_path.name} 已存在。")
            continue

        try:
            old_path.rename(new_path)
            print(f"重命名: {old_path.name} -> {new_path.name}")
        except OSError as error:
            print(f"错误: 无法重命名 {old_path.name}。原因: {error}")

    print("重命名完成！")


def main() -> None:
    args = parse_args()
    rename_images_sequentially(args.image_dir)


if __name__ == "__main__":
    main()
