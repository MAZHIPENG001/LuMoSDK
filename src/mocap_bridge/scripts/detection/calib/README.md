# D435 / ZED 与动捕刚体 ChArUco 眼在手上标定

本目录用于标定 Intel RealSense D435 彩色相机或 ZED 整流左目相机光学坐标系与末端动捕刚体坐标系之间的
固定外参。主标定脚本为 [`handeye.py`](handeye.py)。

这是一个 eye-in-hand（眼在手上）标定问题：

- 相机与动捕刚体必须刚性连接，标定期间作为一个整体移动。
- ChArUco 标定板必须固定在动捕世界坐标系中，全程不能移动。
- 动捕提供末端/刚体位姿 `T_world_gripper`，在本项目中也记作 `T_world_rigid`。
- ChArUco PnP 提供标定板到相机的位姿 `T_camera_board`。
- 脚本求解相机到末端的固定变换 `T_gripper_camera`，它等价于项目原有命名中的
  `T_rigid_camera`。

最终的坐标变换关系为：

```text
p_gripper = T_gripper_camera * p_camera
p_world = T_world_gripper * T_gripper_camera * p_camera

T_world_gripper * T_gripper_camera * T_camera_board = T_world_board
```

其中 `p_camera` 位于所选相机的光学坐标系（D435 彩色光学坐标系或 ZED 整流左目光学坐标系），
单位为米。相机光学坐标系为 X 向右、Y 向下、Z 向前。

## 标定板规格

脚本默认使用以下 ChArUco 标定板，实物尺寸必须与参数一致：

| 参数 | 默认值 |
| --- | --- |
| ArUco 字典 | `DICT_6X6_250` |
| 横向方格数 | `7` |
| 纵向方格数 | `5` |
| 单个方格边长 | `18.12 mm` |
| ArUco Marker 边长 | `14.46 mm` |
| 整板方格区域尺寸 | `126.84 mm × 90.60 mm` |

打印标定板时不要使用“适合页面”或其他自动缩放选项。打印后应使用卡尺复核方格边长，并将标定板平整、
牢固地固定在不会随相机运动的位置。

## 环境与设备要求

- ROS 2 Humble，且已构建 `mocap_bridge`，能够导入 `mocap_bridge.msg.MocapData`。
- Intel RealSense D435 及 `pyrealsense2`，或 ZED 相机及 ZED SDK Python API `pyzed.sl`。
- Python 依赖：`numpy`、`scipy` 和带 `aruco` 模块的 OpenCV。
- OpenCV 需要提供 `cv2.aruco.CharucoDetector`；通常应安装与当前 Python 环境兼容的
  `opencv-contrib-python`。
- 动捕系统能够在 `/mocap_data` 发布待标定刚体，默认使用刚体 `5`。
- 手动无界面模式必须在交互式终端中运行；显示预览窗口时也可以直接在窗口中使用 `s/u/c/q` 快捷键。

可以先检查公共 Python 模块及当前使用的相机 SDK（以下以 ZED 为例）：

```bash
python3 -c "import cv2, numpy, scipy, pyzed.sl; print(cv2.__version__); print(hasattr(cv2.aruco, 'CharucoDetector'))"
```

使用 D435 时将 `pyzed.sl` 替换为 `pyrealsense2`。输出的最后一项应为 `True`。

首次使用时，在仓库根目录构建并加载 ROS 2 工作空间：

```bash
source /opt/ros/humble/setup.bash
./switch_sdk_arch.sh
colcon build --packages-select mocap_bridge --cmake-clean-cache
source install/setup.bash
```

## 标定前准备

1. 将相机和动捕刚体牢固安装在一起。标定完成后不能再改变二者的相对位置。
2. 将 ChArUco 板固定在动捕空间中，保证相机移动过程中标定板本身不动。
3. 确认动捕刚体 ID、位姿方向和位置单位。本项目默认刚体 ID 为 `5`，位置单位为毫米，SDK 输出的
   位姿按 `rigid_to_world` 使用。
4. 确认相机能按脚本的固定配置同时开启彩色流与深度流：D435 使用
   `640×480@60 Hz`，ZED 使用原生 `SVGA@120 Hz`。
5. 保证曝光充足、图像清晰，尽量避免反光、运动模糊和标定板占画面过小。

标定时应保持以下装配关系：

```text
固定不动：ChArUco 标定板

移动整体：[D435 或 ZED 相机] ——刚性连接—— [末端动捕刚体]
```

## 运行方法

### 1. 启动动捕发布节点

在仓库根目录打开第一个终端：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export LD_LIBRARY_PATH="$PWD/src/mocap_bridge/sdk/lib:${LD_LIBRARY_PATH:-}"
ros2 run mocap_bridge mocap_publisher
```

另开终端确认话题和刚体数据正常：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic hz /mocap_data
ros2 topic echo /mocap_data --once
```

需要确认目标刚体的 `rigid_id` 正确，并且 `is_track: true`。

### 2. 启动标定脚本

在第二个交互式终端中运行脚本。推荐直接使用命令行参数；脚本同时兼容 ROS 2 参数。输出路径根据
`handeye.py` 自身的位置生成，与启动命令的当前工作目录无关。例如使用 D435 在
`2026-08-05 14:30:20` 完成计算时，默认结果保存为：

```text
<脚本所在目录>/d435_20260805_143020/handeye_calibration.json
```

使用 D435：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

python3 src/mocap_bridge/scripts/detection/calib/handeye.py \
  --camera d435 \
  --rigid-id 5 \
  --mocap-position-scale 0.001 \
  --mocap-pose-direction rigid_to_world
```

使用 ZED 时选择 `zed`。脚本使用与深度对齐的整流左目图像和对应零畸变内参：

```bash
python3 src/mocap_bridge/scripts/detection/calib/handeye.py \
  --camera zed \
  --rigid-id 5 \
  --mocap-position-scale 0.001 \
  --mocap-pose-direction rigid_to_world
```

无需按键的自动采样模式（以下以 ZED 为例）：

```bash
python3 src/mocap_bridge/scripts/detection/calib/handeye.py \
  --camera zed \
  --rigid-id 5 \
  --mocap-position-scale 0.001 \
  --mocap-pose-direction rigid_to_world \
  --auto-capture \
  --auto-target-samples 20
```

需要指定相机序列号、关闭界面或指定结果文件时，可分别使用 `--camera-serial`、`--no-gui` 和
`--output`：

```bash
python3 src/mocap_bridge/scripts/detection/calib/handeye.py \
  --camera d435 \
  --camera-serial 123456789 \
  --rigid-id 5 \
  --auto-capture \
  --no-gui \
  --output /tmp/handeye_calibration.json
```

自动模式会持续检查最近的同步窗口。每次将相机/刚体移动到一个新方向并短暂停稳后，只要该姿态满足静止
阈值、且与所有已保存姿态有足够差异，就会自动保存。达到目标数量且具备至少两个不平行旋转轴后，程序会
自动执行标定计算、写入 `handeye_calibration.json` 并退出；不需要按 `s` 或 `c`。`--no-gui` 应与
`--auto-capture` 一起使用，否则手动模式仍需要终端按键。

需要调整高级参数或沿用 ROS 2 启动方式时，使用等价的 ROS 参数形式：

```bash
python3 src/mocap_bridge/scripts/detection/calib/handeye.py --ros-args \
  -p camera_type:=zed \
  -p rigid_id:=5 \
  -p mocap_position_scale:=0.001 \
  -p mocap_pose_direction:=rigid_to_world \
  -p auto_capture:=true \
  -p auto_target_samples:=20 \
  -p averaging_window_sec:=0.8 \
  -p min_window_pairs:=30 \
  -p min_charuco_corners:=18 \
  -p max_reprojection_error_px:=0.5 \
  -p max_pair_delta_sec:=0.015
```

查看所有直接命令行选项：

```bash
python3 src/mocap_bridge/scripts/detection/calib/handeye.py --help
```

## 采样流程

预览窗口会像采集和检测脚本一样，将 ChArUco 彩色检测画面与 0～5 米伪彩色深度图横向放在同一窗口。
彩色画面左上角会显示：

- `saved`：已经保存的标定位姿数量。
- `AUTO N/M`：自动模式已保存数量和自动计算的目标数量。
- `PnP`：当前 ChArUco PnP 重投影 RMSE，单位为像素，越小越好。
- `dt`：当前相机帧与最近动捕帧的时间差，单位为毫秒，越小越好。

终端快捷键如下：

| 按键 | 功能 |
| --- | --- |
| `s` | 手动模式请求保存当前位姿；自动模式忽略此按键 |
| `u` | 删除最后一个已保存位姿 |
| `c` | 计算标定结果并写入 JSON；成功后程序自动退出 |
| `q` | 不计算，直接退出 |

自动模式推荐按以下方式采样：

1. 启动后保持第一个姿态不动，看到 `已保存姿态 1` 后再移动。
2. 面向标定板改变相机的位置和方向；每到一个新姿态短暂停稳，看到计数增加后再继续。
3. 姿态应覆盖俯仰、偏航和适量滚转，并改变观察距离及标定板在画面中的位置。
4. 达到 `auto_target_samples` 后，若旋转覆盖充分，程序会自动计算、保存并退出；否则按终端提示继续补充
   不同旋转轴的姿态。

持续快速晃动不会被保存，因为这会降低相机/动捕时间配对和手眼标定精度。自动模式省去的是按键操作，
每个采样姿态仍应短暂停稳。

手动模式推荐按以下方式采样：

1. 将相机移动到一个能清晰看到标定板的位置。
2. 停稳相机和刚体并保持不动。可以停稳后立即按 `s`；程序默认会等待最多 3 秒，以取得一个完整的静止窗口。
3. 确认 ChArUco 角点正常显示，`PnP` 和 `dt` 没有超过设定阈值。
4. 按一次 `s`，终端出现 `已保存姿态 N` 后再移动到下一个位姿。
5. 采集 15～25 个差异明显的位姿。程序默认至少需要 12 个。
6. 完成采样后按 `c` 计算并保存结果。

采样不能只做平移。相机应绕至少两个不平行的轴产生充分旋转，同时改变观察距离和画面位置。建议覆盖
俯仰、偏航和适量滚转，但始终保证标定板清晰可见。相邻样本若平移小于 `10 mm` 且旋转小于 `5°`，
默认会被视为重复位姿而拒绝保存。自动模式会与所有历史样本比较，避免绕回旧姿态时重复采样。

## ROS 2 参数

### 标定板与角点检测

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `squares_x` | `7` | 横向方格数 |
| `squares_y` | `5` | 纵向方格数 |
| `square_length_m` | `0.01812` | 方格边长，单位为米 |
| `marker_length_m` | `0.01446` | Marker 边长，单位为米 |
| `legacy_pattern` | `false` | 是否使用 OpenCV ChArUco 旧版图案布局 |
| `min_charuco_corners` | `8` | 单帧 PnP 所需的最少 ChArUco 角点数，程序内部下限为 4 |
| `max_reprojection_error_px` | `1.0` | 接受单帧 PnP 的最大重投影 RMSE，单位为像素 |

### 相机

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `camera_type` | `d435` | 相机后端：`d435`（也接受 `d435i`、`realsense`、`rs`）或 `zed`（也接受 `zedx`、`zed_x`） |
| `camera_serial` | 空 | 可选相机序列号；ZED 序列号必须能够转换为整数 |
| `show_image` | `true` | 是否显示检测预览窗口 |

相机图像规格不通过 ROS 参数配置：RealSense 固定构造为
`RealSenseCamera(width=640, height=480, fps=60)`；ZED 固定构造为
`ZEDCamera(resolution=sl.RESOLUTION.SVGA, fps=120)`。处理定时器分别使用 60 Hz 和 120 Hz。

### 动捕与时间配对

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `rigid_id` | `5` | 与相机刚性连接的动捕刚体 ID |
| `mocap_topic` | `/mocap_data` | `mocap_bridge.msg.MocapData` 输入话题 |
| `mocap_position_scale` | `0.001` | 动捕平移转换到米的比例；输入为毫米时使用 `0.001` |
| `mocap_pose_direction` | `rigid_to_world` | `rigid_to_world`、`world_to_rigid` 或 `auto`；本项目发布端默认使用前者 |
| `use_mocap_header_stamp` | `true` | 优先使用 `/mocap_data` 的 ROS Header 时间戳 |
| `max_pair_delta_sec` | `0.03` | 相机帧与动捕帧允许的最大时间差，单位为秒 |
| `max_observation_age_sec` | `0.25` | 保存采样时允许最新同步观测的最大年龄，避免误存陈旧位姿 |

本项目的动捕发布节点在本机收到 SDK 数据时写入 ROS Header，相机采集时间也使用本机系统时钟，因此默认
可以直接配对。如果接入其他动捕发布节点，应确认两个时间戳来自同一时钟域。

### 静止窗口、样本筛选与输出

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `averaging_window_sec` | `0.40` | 自动检测或按 `s` 时参与平均的最近时间窗口，单位为秒 |
| `min_window_pairs` | `8` | 一个静止位姿至少需要的有效相机/动捕配对帧数 |
| `stationary_gripper_translation_m` | `0.002` | 窗口内末端动捕平移最大离散阈值 |
| `stationary_gripper_rotation_deg` | `0.5` | 窗口内末端动捕旋转最大离散阈值 |
| `stationary_pnp_translation_m` | `0.003` | 窗口内 PnP 平移最大离散阈值 |
| `stationary_pnp_rotation_deg` | `0.8` | 窗口内 PnP 旋转最大离散阈值 |
| `stationary_wait_timeout_sec` | `3.0` | 按 `s` 后等待连续静止窗口的最长时间；设为 `0` 恢复立即判定 |
| `duplicate_translation_m` | `0.010` | 与历史样本比较时的重复平移阈值 |
| `duplicate_rotation_deg` | `5.0` | 与历史样本比较时的重复旋转阈值 |
| `min_samples` | `12` | 计算标定所需的最少已保存位姿数，程序内部下限为 3 |
| `max_outlier_fraction` | `0.25` | 鲁棒重算时最多允许剔除的样本比例 |
| `auto_capture` | `false` | 是否自动检测、保存新静止姿态并在覆盖充分后计算结果 |
| `auto_target_samples` | `20` | 自动计算的目标样本数；程序内部不会低于 `min_samples` |
| `auto_check_interval_sec` | `0.10` | 自动静止窗口检查间隔，单位为秒，程序内部下限为 `0.02` |
| `output_path` | 空 | 指定输出 JSON；若没有 `.json` 后缀则视为输出目录 |

常用直接命令行参数与 ROS 参数对应如下：

| 命令行参数 | ROS 参数 |
| --- | --- |
| `--camera` | `camera_type` |
| `--camera-serial` | `camera_serial` |
| `--rigid-id` | `rigid_id` |
| `--mocap-topic` | `mocap_topic` |
| `--mocap-position-scale` | `mocap_position_scale` |
| `--mocap-pose-direction` | `mocap_pose_direction` |
| `--min-samples` | `min_samples` |
| `--auto-capture` | `auto_capture=true` |
| `--auto-target-samples` | `auto_target_samples` |
| `--no-gui` | `show_image=false` |
| `--output` | `output_path` |

## 输出文件说明

脚本会运行 Tsai-Lenz、Park、Horaud、Andreff 和 Daniilidis 五种 OpenCV 手眼标定方法，并按固定标定板
在动捕世界中的位姿一致性选择残差最小的结果。若 `mocap_pose_direction=auto`，还会同时尝试两种动捕
位姿方向。

输出 JSON 的主要字段如下：

| 字段 | 说明 |
| --- | --- |
| `selected` | 最终选中的求解方法、位姿方向、变换和残差 |
| `selected.T_gripper_camera` | 相机光学坐标系到末端/动捕刚体坐标系的 `4×4` 齐次变换；主要输出 |
| `selected.T_rigid_camera` | 与 `T_gripper_camera` 相同，为兼容项目已有工具保留的字段 |
| `selected.quaternion_xyzw` | 与上述旋转矩阵对应的四元数，顺序为 `x, y, z, w` |
| `selected.translation_m` | 与上述变换对应的平移，单位为米 |
| `T_camera_gripper` | `T_gripper_camera` 的逆变换 |
| `T_camera_rigid` | 与 `T_camera_gripper` 相同，为兼容项目已有工具保留的字段 |
| `sample_count_collected` | 总采样数 |
| `sample_count_used` | 剔除离群样本后实际参与求解的样本数 |
| `rejected_sample_indices_zero_based` | 被剔除样本的零基索引 |
| `all_solver_results` | 所有成功求解候选项及其残差 |
| `sample_quality` | 每个样本的帧数、时间差、PnP 误差和角点数 |

`translation_rmse_mm` 和 `rotation_rmse_deg` 表示将各次观测还原到固定标定板后，标定板位姿在动捕世界
中的离散程度。数值越小越好，但它们是内部一致性指标，不能替代独立数据上的精度验证。还应检查：

- 不同求解方法的结果是否接近。
- 是否有大量样本被剔除。
- `sample_quality` 中是否存在明显偏大的时间差或 PnP 误差。
- 外参平移是否与相机和刚体的实际安装距离大致相符。

未指定 `--output`/`output_path` 时，每次成功计算都会按相机类型和计算时间生成新的输出目录，因此不会覆盖
以前的标定文件。更换相机、分辨率、镜头参数或相机与刚体的安装关系后，都应重新标定。目录中已有的
JSON 只对应生成它时的硬件和数据，不能直接视为其他设备的有效外参。

## 结果验证

建议使用未参与标定的独立数据验证外参。项目中的 `plot_auto_calib.py` 会读取
`selected.T_rigid_camera`，将相机测得的三维点转换到动捕世界坐标系。以下验证和修正命令均在仓库
根目录执行：

```bash
python3 src/mocap_bridge/scripts/plot_auto_calib.py \
  --dir src/mocap_bridge/scripts/data/<数据目录> \
  --handeye-calib src/mocap_bridge/scripts/detection/calib/<相机类型_时间戳>/handeye_calibration.json
```

当前 `plot_auto_calib.py` 使用刚体 `5`，并要求标定结果的
`selected.mocap_pose_direction` 为 `rigid_to_world`。因此在本项目默认链路中，建议标定时显式设置
`--mocap-pose-direction rigid_to_world`，或使用 ROS 参数
`-p mocap_pose_direction:=rigid_to_world`。

如果后续任务专门使用 RGB-D 球心，且独立验证发现稳定的系统偏差，可再使用
`scripts/refine_ball_extrinsic.py` 拟合“球心检测有效外参”。它不是基础手眼标定的替代品，输出也可能吸收
深度和球心算法的系统误差。至少需要 3 组、推荐 6 组以上不同方向和距离的数据，并应保留额外数据做独立
验证。为避免把时间偏移吸收到外参中，推荐每组记录时让球和相机都保持静止；不要用尚未校正时间戳的运动
轨迹做外参修正：

```bash
python3 src/mocap_bridge/scripts/refine_ball_extrinsic.py \
  --dirs <数据目录1> <数据目录2> <数据目录3> \
  --input src/mocap_bridge/scripts/detection/calib/<相机类型_时间戳>/handeye_calibration.json \
  --output src/mocap_bridge/scripts/detection/calib/handeye_ball_refined.json
```

每个输入目录必须包含 `center_raw.csv` 和 `mocap.csv`。
相机刚体 ID 默认从 `--input` 标定文件的 `rigid_id` 读取；需要覆盖时使用
`--rigid-id <ID>`。

## 常见问题

### 没有同步数据，或 `dt` 一直很大

- 检查 `/mocap_data` 是否持续发布，以及目标刚体是否处于跟踪状态。
- 确认相机时间和动捕 Header 时间来自同一台主机、同一时钟域。
- 本项目默认最大时间差为 `30 ms`；严格配置中的 `15 ms` 不适合低频或延迟较大的动捕链路。
- 如果外部动捕节点的 Header 时间戳不可用，可评估后设置
  `-p use_mocap_header_stamp:=false`，改用本节点的消息接收时间。

### 按 `s` 后提示位姿不静止

按 `s` 后程序会先输出“采样请求已收到”，并在 `stationary_wait_timeout_sec` 内用最新数据反复寻找连续静止
窗口；此时保持设备不动，出现“已保存姿态 N”才表示保存成功。超时后只输出一次最终离散量，避免反复按键
产生大量相同警告。

- 如果 mocap 与 PnP 的平移、旋转离散量同时以相近幅度变大，通常是相机/刚体仍在运动，或固定板发生晃动。
- 如果只有 mocap 偏大，检查刚体是否刚性安装、反光点跟踪是否跳变以及 `mocap_position_scale`。
- 如果只有 PnP 偏大，改善照明和对焦、增大标定板在画面中的尺寸，并检查打印尺寸和相机内参。
- 如需更多停稳时间，可设置 `-p stationary_wait_timeout_sec:=5.0`。只有在设备确实静止、并测得稳定噪声基线后，
  才应小幅放宽对应的 `stationary_*` 阈值；不应以大幅放宽阈值掩盖运动或同步问题。

### 提示位姿与上一样本过于相似

继续移动并旋转相机，增加相邻样本之间的平移或旋转差异。只在同一朝向下改变距离不能提供充分的旋转
激励。

### 提示旋转激励不足

增加绕至少两个不平行轴的旋转样本。程序要求样本集合中存在足够大的相对旋转，纯平移或只绕单一轴旋转
无法可靠求解手眼外参。

### 检测不到 ChArUco，或 PnP 误差过大

- 核对字典、方格数量和物理尺寸。
- 增大标定板在画面中的占比，改善照明和对焦，缩短曝光以减少运动模糊。
- 避免标定板弯曲、强反光、严重遮挡和极端倾斜角度。
- 严格配置要求至少 18 个角点且重投影 RMSE 不超过 `0.5 px`；可先使用默认的 8 个角点和
  `1.0 px` 排查检测链路。

### 相机启动失败

确认相机未被其他程序占用，并确认 D435 支持 `640×480@60 Hz`，或 ZED/ZED X 支持
`SVGA@120 Hz`。同时检查 `pyrealsense2`/`pyzed.sl` 是否安装、相机序列号是否正确，以及当前用户是否有
设备访问权限。脚本使用项目 `device` 目录中的相机封装；若具体 ZED 型号不支持默认规格，需要同步调整
`handeye.py` 与 `device/zed_camera.py` 中的相机配置。

### 快捷键无效

显示预览窗口时，可以先点击窗口使其获得焦点，再使用 `s/u/c/q`；也可以直接在交互式终端中按键。
使用 `--no-gui`/`show_image=false` 的手动模式时，必须保留交互式终端。完全无终端运行时应同时启用
`--auto-capture`/`auto_capture=true`，并用 `Ctrl+C` 中止异常流程。
