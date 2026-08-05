# 图像数据工具

本目录提供 RealSense/ZED 图像采集和图片批量重命名工具。两个脚本都支持在命令后直接传入目标目录；目录既可以是绝对路径，也可以是相对路径。

## 1. 采集图像

### 1.1 环境要求

两类相机都需要安装 NumPy 和 OpenCV。此外，请根据使用的设备安装对应的相机 SDK：

- RealSense：安装 Intel RealSense SDK 及 Python 模块 `pyrealsense2`。
- ZED：安装与相机及系统匹配的 Stereolabs ZED SDK，并确认 Python 可以导入 `pyzed.sl`。

脚本只会导入 `--camera` 指定的相机模块，因此使用 RealSense 时不要求安装 ZED SDK，使用 ZED 时也不要求安装 `pyrealsense2`。

### 1.2 运行

基本用法：

```bash
python3 ~/GithubDoc/LuMoSDK/data/data_save.py [保存目录] \
  [--camera realsense|zed] [--frequency 保存频率]
```

例如，使用默认的 RealSense 相机，以 5 Hz 保存图片：

```bash
python3 ~/GithubDoc/LuMoSDK/data/data_save.py ./data/pic_ball --frequency 5
```

使用 ZED 相机：

```bash
python3 ~/GithubDoc/LuMoSDK/data/data_save.py ./data/pic_ball --camera zed
```

默认保存到 `data/pic_ball`，使用 RealSense，保存频率为 10 Hz。`--frequency`（或 `-f`）必须大于 0，实际频率受相机帧率、图像处理和磁盘速度限制。采集规格为 RealSense `640×480 @ 120 FPS`，ZED `HD1200 @ 120 FPS`。

运行后可使用以下按键：

- `c`：按设置的频率开始保存彩色图片。
- `s`：停止采集。
- `q`：退出程序。

彩色图片按 `pic001.jpg`、`pic002.jpg` 等格式保存；已有图片时会从最大编号继续。深度图仅用于预览，以 0～5 米范围显示为伪彩色图，无效深度为黑色。

## 2. 批量重命名

运行 `rename.py`，并将待处理的图片目录作为第一个参数传入：

```bash
python3 ~/GithubDoc/LuMoSDK/data/rename.py /path/to/image_dir
```

例如：

```bash
python3 ~/GithubDoc/LuMoSDK/data/rename.py ./pic_ball
```

如果不传入目录，默认处理当前工作目录下的 `pic`：

```bash
python3 ~/GithubDoc/LuMoSDK/data/rename.py
```

脚本只处理名称符合 `pic数字.jpg` 格式的文件，并按原编号顺序重命名为 `pic0001.jpg`、`pic0002.jpg` 等连续编号。

## 3. 查看帮助

```bash
python3 ~/GithubDoc/LuMoSDK/data/data_save.py --help
python3 ~/GithubDoc/LuMoSDK/data/rename.py --help
```

## 4. ModelScope 数据集上传与下载

数据集仓库为 [`MaZp001/ball_detection`](https://modelscope.cn/datasets/MaZp001/ball_detection)。

### 4.1 安装并登录

```bash
python3 -m pip install --upgrade modelscope
modelscope login --token ms-********************************
```

执行 `modelscope login` 后，根据提示输入 ModelScope Access Token。请勿将 Token 直接写入 README 或提交到 Git 仓库。

### 4.2 上传全部数据

以下命令会递归上传 `~/GithubDoc/LuMoSDK/data` 下的全部文件和子目录，并保存到数据集仓库根目录：

```bash
modelscope upload MaZp001/ball_detection \
  ~/GithubDoc/LuMoSDK/data \
  --repo-type dataset \
  --commit-message "Update ball detection dataset"
```

再次执行该命令可以上传新增或修改后的文件。

### 4.3 下载全部数据

为避免覆盖当前 `data` 目录中的文件，以下命令默认下载到同级的 `data_modelscope` 目录：

```bash
modelscope download MaZp001/ball_detection \
  --repo-type dataset \
  --local-dir ~/GithubDoc/LuMoSDK/data_modelscope
```

如果需要直接同步到当前 `data` 目录，可以将 `--local-dir` 后的路径改为 `~/GithubDoc/LuMoSDK/data`。执行前请确认本地同名文件可以被更新。
