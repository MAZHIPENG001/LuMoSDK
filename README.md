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

使用 RealSense（默认），采集配置为 `640×480 @ 60 FPS`：

```bash
python3 eval_ros.py --camera realsense
```

使用 ZED，采集配置为原生 `SVGA（960×600）@ 120 FPS`：

```bash
python3 eval_ros.py --camera zed
```

`--camera` 可选值为 `realsense` 和 `zed`。程序只导入当前选择的相机驱动，
因此选择 ZED 时不会导入 `pyrealsense2`，选择 RealSense 时不会导入
`pyzed.sl`。

球心默认使用 `silhouette` 方法：根据 YOLO 分割轮廓、相机内参和球的已知物理半径，直接恢复三维球心，
不使用光滑球面上可能存在系统偏差的双目深度。默认半径为 `0.110 m`，使用前必须实测目标球直径并设置正确
半径；半径误差会近似按相同比例变成距离误差：

```bash
python3 eval_ros.py \
  --camera zed \
  --position-method silhouette \
  --ball-radius-m 0.110
```

如需对照原来的深度球面拟合，可使用 `--position-method depth`。轮廓法失败时也会自动回退到深度法，预览中
会显示本帧实际使用的 `silhouette`、`depth-sphere` 或 `depth-ray`。默认发布未滤波球心，便于评估运动目标的
真实误差；确实需要平滑输出时可加 `--one-euro-filter`，此时会对 XYZ 三轴使用相同滤波策略。

检测端默认开启运动一致性门控，阈值为最大表观速度 `8 m/s`、相对匀速预测最大偏差 `1.0 m`。正常测量不会
被平滑或修改；孤立极值不会发布到 `/ball_center_raw` 和 `/ball_surface`，`/ball_center` 最多使用 `0.25 s`
匀速预测维持连续，之后停止发布而不会输出错误位置。预览和终端会分别显示 `motion-prediction` 和拒绝原因。
如果目标实际速度超过默认值，应按真实速度调大阈值，例如：

```bash
python3 eval_ros.py \
  --camera zed \
  --max-ball-speed-mps 12 \
  --max-motion-innovation-m 1.2 \
  --max-prediction-sec 0.25
```

调试时可用 `--disable-motion-gate` 恢复不做在线极值门控的发布行为。

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

绘图默认读取 `center_raw.csv`，避免滤波时延混入球心检测误差。需要检查启用滤波后的发布结果时，添加
`--center-source filtered`。终端统计还会把误差拆成相机径向误差和方向误差：固定球上方向误差小而径向误差
大，通常说明问题在深度/球心距离恢复；二者都随运动明显增大时还应检查时间同步。

## 其他工具

项目内其他工具的详细使用说明如下：

- [图像数据工具](data/README.md)：使用 RealSense 或 ZED 采集图像、批量重命名图片，以及上传和下载 ModelScope 数据集。
- [ChArUco 手眼标定工具](src/mocap_bridge/scripts/detection/calib/README.md)：标定 RealSense 或 ZED 相机光学坐标系与动捕刚体坐标系之间的固定外参。
- [检测模型与数据集工具](src/mocap_bridge/scripts/detection/model/README.md)：通过 ModelScope 上传、下载 YOLO 检测模型及数据集。
