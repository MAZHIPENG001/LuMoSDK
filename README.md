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
不使用光滑球面上可能存在系统偏差的双目深度。轮廓默认先经过鲁棒亚像素椭圆拟合，以抑制小目标二值掩码
边缘的锯齿和局部毛刺；该处理只使用当前帧，不会增加时间滤波延迟。可使用
`--silhouette-boundary-model raw` 恢复直接使用原始边界的行为。

默认半径为 `0.110 m`，使用前必须实测目标球直径并设置正确半径；半径误差会近似按相同比例变成距离误差：

```bash
python3 eval_ros.py \
  --camera zed \
  --position-method silhouette \
  --ball-radius-m 0.110
```

如需对照深度球面拟合，可使用 `--position-method depth`。在 `silhouette` 模式下，深度回退默认采用
`auto`：RealSense 启用，当前球面深度误差较大的 ZED 禁用，避免偶发轮廓失败产生较大的深度跳点。可通过
`--depth-fallback enabled` 或 `--depth-fallback disabled` 强制修改。

`/ball_center` 默认启用 One Euro 自适应滤波，新默认参数为 `min_cutoff=8 Hz`、`beta=1`。与原来的
`2 Hz/5` 相比，它降低静止或低速阶段的滤波滞后，同时避免速度噪声使滤波器过早接近直通。参数可通过
`--one-euro-min-cutoff`、`--one-euro-beta` 和 `--one-euro-derivative-cutoff` 调整；
`/ball_center_raw` 始终保留未经时间滤波的球心。需要完全关闭滤波时使用 `--disable-one-euro-filter`。

#### 检测低时延运行

发布队列只保留最新一个坐标，处理定时器默认跟随相机帧率。检测预览默认只显示彩色图；如需深度伪彩色图，
显式添加 `--show-depth-preview`。生产运行时可关闭全部 GUI，并在不需要 `/ball_surface` 诊断数据时跳过表面
深度处理：

```bash
python3 eval_ros.py \
  --camera zed \
  --position-method silhouette \
  --disable-preview \
  --disable-surface-publish
```

`--model` 可以指定 `.engine` 或 `.pt` 模型；脚本优先使用模型目录下的 `best.engine`，不存在时回退到
`best.pt`。`--imgsz HEIGHT WIDTH` 可调整推理尺寸，减小尺寸通常降低推理时间，但也会降低远距离小球的轮廓
稳定性；TensorRT engine 的输入尺寸在导出时已经固定，修改 `--imgsz` 前应重新导出对应尺寸的 engine。
程序每秒输出处理 FPS、纯模型耗时、整帧处理耗时，以及相机采集到发布的平均/最大时延。

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
# d435
python3 src/mocap_bridge/scripts/plot_auto_calib.py \
  --handeye-calib \
  src/mocap_bridge/scripts/detection/calib/calib_realsense/handeye_ball_refined_ralsensed435.json
```
```bash
# zed
python3 src/mocap_bridge/scripts/plot_auto_calib.py \
  --handeye-calib \
  src/mocap_bridge/scripts/detection/calib/zed_20260806_182206/handeye_calibration.json \
  --center-source filtered \
  --camera-pose-mode auto \
  --dir ~/GithubDoc/LuMoSDK/src/mocap_bridge/scripts/data/20260807_141345
```

绘图默认读取 `center_raw.csv`，避免滤波时延混入球心检测误差。需要检查启用滤波后的发布结果时，添加
`--center-source filtered`。终端统计还会把误差拆成相机径向误差和方向误差：固定球上方向误差小而径向误差大，通常说明问题在深度/球心距离恢复；二者都随运动明显增大时还应检查时间同步。

#### Rigid 5 相机位姿模式

当相机局部坐标中的检测曲线较平滑，但图 2 转换到动捕世界坐标后出现尖峰时，通常是静止相机对应的 Rigid 5 四元数存在少量跳点。`plot_auto_calib.py` 通过 `--camera-pose-mode` 控制相机位姿的处理方式：

- `auto`：默认模式。根据视觉数据覆盖时段内 Rigid 5 的平移和旋转变化自动判断相机是否静止；静止时使用剔除离群值后的鲁棒平均位姿，运动时使用逐帧插值。
- `fixed`：确认采集期间相机固定时，强制使用鲁棒平均位姿，避免四元数跳点被目标距离放大成世界坐标尖峰。
- `interpolated`：相机在采集期间发生移动时，对每个视觉时间戳逐帧执行平移插值和四元数 Slerp。

固定相机可以显式执行：

```bash
python3 src/mocap_bridge/scripts/plot_auto_calib.py \
  --handeye-calib \
  src/mocap_bridge/scripts/detection/calib/zed_20260806_182206/handeye_calibration.json \
  --center-source filtered \
  --camera-pose-mode fixed
```

移动相机应使用：

```bash
python3 src/mocap_bridge/scripts/plot_auto_calib.py \
  --handeye-calib \
  src/mocap_bridge/scripts/detection/calib/zed_20260806_182206/handeye_calibration.json \
  --center-source filtered \
  --camera-pose-mode interpolated
```

`auto` 默认使用平移 P95 不超过 `3 mm`、旋转 P95 不超过 `1 deg` 作为静止判据，可分别通过 `--static-camera-translation-mm` 和 `--static-camera-rotation-deg` 调整。程序会输出最终选择的模式、保留的 Rigid 5 样本数，以及平移和旋转的 P95/最大偏差，便于确认图 2 是否受动捕位姿抖动影响。

## 其他工具

项目内其他工具的详细使用说明如下：

- [图像数据工具](data/README.md)：使用 RealSense 或 ZED 采集图像、批量重命名图片，以及上传和下载 ModelScope 数据集。
- [ChArUco 手眼标定工具](src/mocap_bridge/scripts/detection/calib/README.md)：标定 RealSense 或 ZED 相机光学坐标系与动捕刚体坐标系之间的固定外参。
- [检测模型与数据集工具](src/mocap_bridge/scripts/detection/model/README.md)：通过 ModelScope 上传、下载 YOLO 检测模型及数据集。
