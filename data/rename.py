import os
import re


def rename_images_sequentially(directory):
    # 1. 获取目录下所有文件
    try:
        files = os.listdir(directory)
    except FileNotFoundError:
        print(f"错误：找不到目录 '{directory}'，请检查路径是否正确。")
        return

    # 2. 筛选出符合 pic+数字.jpg 格式的文件，并提取编号
    # 正则解释：pic(\d+)\.jpg 匹配 pic 开头，中间是数字，结尾是 .jpg 的文件
    image_files = []
    pattern = re.compile(r'^pic(\d+)\.jpg$')

    for filename in files:
        match = pattern.match(filename)
        if match:
            # 提取数字部分并转换为整数，用于排序
            file_number = int(match.group(1))
            image_files.append((file_number, filename))

    if not image_files:
        print("在指定目录下未找到符合 'pic数字.jpg' 格式的文件。")
        return

    # 3. 按照原始编号进行排序
    image_files.sort(key=lambda x: x[0])

    print(f"共找到 {len(image_files)} 个图片文件，开始重命名...")

    # 4. 依次重命名
    # 使用 enumerate 生成新的连续编号，start=1 表示从 1 开始
    for new_index, (old_num, old_name) in enumerate(image_files, start=1):
        # 格式化为 4 位数字，例如 1 -> 0001, 105 -> 0105
        new_name = f"pic{new_index:04d}.jpg"

        # 构建完整路径
        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)

        # 执行重命名
        try:
            os.rename(old_path, new_path)
            print(f"重命名: {old_name} -> {new_name}")
        except FileExistsError:
            print(f"跳过: {new_name} 已存在 (可能已有同名文件)。")
        except Exception as e:
            print(f"错误: 无法重命名 {old_name}。原因: {e}")

    print("重命名完成！")


if __name__ == "__main__":
    # 请根据实际情况修改这里的路径
    # 如果是相对路径，确保脚本在该目录下运行
    target_directory = r'pic'

    rename_images_sequentially(target_directory)