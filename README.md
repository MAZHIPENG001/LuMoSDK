# LuMoSDK

LuMoSDK 的 ROS 2 桥接与数据处理工具，用于发布动捕和相机数据，以及订阅、保存和可视化相关数据。

## 环境要求

- ROS 2 Humble
- Python 3
- `colcon`

## 获取源码

```bash
git clone git@github.com:MAZHIPENG001/LuMoSDK.git
cd LuMoSDK
```

下文中的命令均默认在仓库根目录执行。

## ARM64 与 x86_64 架构切换

仓库同时提供 ARM64 和 x86_64 两种架构的 LuMoSDK 动态库：

| 平台架构 | 动态库目录 |
| --- | --- |
| ARM64（`aarch64`、`arm64`） | `lib_arm` |
| x86_64（`x86_64`、`amd64`） | `lib_x86` |

ROS 2 包固定从 `src/mocap_bridge/sdk/lib` 加载 SDK，仓库根目录的独立示例固定从 `lib` 加载 SDK。因此，构建前必须将当前平台对应的架构库复制到以下两个 `lib` 目录：

- `LuMoSDK/lib`
- `LuMoSDK/src/mocap_bridge/sdk/lib`

仓库根目录提供了 [`switch_sdk_arch.sh`](switch_sdk_arch.sh) 切换脚本。自动识别当前平台并切换：

脚本会先确认两处对应架构的源库目录都存在；如果目标 `lib` 已存在，则先将其删除，再复制对应的 `lib_arm` 或 `lib_x86` 并命名为 `lib`。

```bash
chmod +x switch_sdk_arch.sh
./switch_sdk_arch.sh
```

切换后可通过以下命令确认两个 `lib` 均为实际目录，并检查动态库架构：

```bash
test -d lib && test ! -L lib
test -d src/mocap_bridge/sdk/lib && test ! -L src/mocap_bridge/sdk/lib
file lib/libLuMoSDK.so
file src/mocap_bridge/sdk/lib/libLuMoSDK.so
```

脚本仅支持仓库中已有动态库的 ARM64 和 x86_64 平台，不支持 32 位 ARM。切换架构后应清理 CMake 缓存并重新构建，不能复用其他架构的已有构建产物。

## 构建

```bash
source /opt/ros/humble/setup.bash
./switch_sdk_arch.sh
colcon build --packages-select mocap_bridge --cmake-clean-cache
source install/setup.bash
```

如本地环境需要固定 `setuptools` 版本，可执行：

```bash
python3 -m pip install setuptools==59.6.0
```

## 使用方法

### 发布动捕数据

在新终端中加载环境并设置 SDK 动态库路径：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export LD_LIBRARY_PATH="$PWD/src/mocap_bridge/sdk/lib:${LD_LIBRARY_PATH:-}"
```

通过 ROS 2 启动动捕发布节点：

```bash
ros2 run mocap_bridge mocap_publisher
```

也可以直接运行构建后的可执行文件：

```bash
./install/mocap_bridge/lib/mocap_bridge/mocap_publisher
```

### 发布相机数据

进入检测脚本目录：

```bash
source /opt/ros/humble/setup.bash
cd src/mocap_bridge/scripts/detection
```

使用 RealSense（默认），采集配置为 `640×480 @ 120 FPS`：

```bash
python3 eval_ros.py --camera realsense
```

使用 ZED，采集配置为 `HD1200 @ 120 FPS`：

```bash
python3 eval_ros.py --camera zed
```

`--camera` 可选值为 `realsense` 和 `zed`。程序只导入当前选择的相机驱动，
因此选择 ZED 时不会导入 `pyrealsense2`，选择 RealSense 时不会导入
`pyzed.sl`。

### 订阅数据

查看动捕与视觉数据：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export LD_LIBRARY_PATH="$PWD/src/mocap_bridge/sdk/lib:${LD_LIBRARY_PATH:-}"
python3 src/mocap_bridge/scripts/mocap_subscriber.py
```

订阅并保存数据：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export LD_LIBRARY_PATH="$PWD/src/mocap_bridge/sdk/lib:${LD_LIBRARY_PATH:-}"
python3 src/mocap_bridge/scripts/data_save.py
```

数据默认保存到 `src/mocap_bridge/scripts/data/<时间戳>/`。

### 绘制数据

绘制最新一次保存的数据：

```bash
python3 src/mocap_bridge/scripts/plot_auto_calib.py
```

绘制指定目录中的数据：

```bash
python3 src/mocap_bridge/scripts/plot_auto_calib.py --dir /path/to/data
```

使用指定的手眼标定文件：

```bash
python3 src/mocap_bridge/scripts/plot_auto_calib.py \
  --handeye-calib \
  src/mocap_bridge/scripts/detection/calib/calib_realsense/handeye_ball_refined_ralsensed435.json
```
```bash
python3 src/mocap_bridge/scripts/plot_auto_calib.py \
  --handeye-calib \
  src/mocap_bridge/scripts/detection/calib/zed_20260806_182206/handeye_calibration.json
```
## 其他工具

项目内其他工具的详细使用说明如下：

- [图像数据工具](data/README.md)：使用 RealSense 或 ZED 采集图像、批量重命名图片，以及上传和下载 ModelScope 数据集。
- [ChArUco 手眼标定工具](src/mocap_bridge/scripts/detection/calib/README.md)：标定 RealSense 或 ZED 相机光学坐标系与动捕刚体坐标系之间的固定外参。
- [检测模型与数据集工具](src/mocap_bridge/scripts/detection/model/README.md)：通过 ModelScope 上传、下载 YOLO 检测模型及数据集。
