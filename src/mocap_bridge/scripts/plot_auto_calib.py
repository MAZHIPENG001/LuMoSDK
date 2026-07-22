import argparse
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R_scipy
from scipy.spatial.transform import Slerp
import json

"""
python3 plot_auto.py --dir <your_dir>
"""

def get_latest_data_dir(base_dir):
    """获取基础目录下名称排序最后的文件夹"""
    subdirs = [
        d
        for d in glob.glob(os.path.join(base_dir, "*"))
        if os.path.isdir(d) and not d.startswith(".")
    ]
    if not subdirs:
        raise FileNotFoundError(f"在 {base_dir} 路径下未找到任何数据文件夹！")
    latest_dir = max(subdirs, key=os.path.basename)
    return latest_dir
def set_axes_equal_3d(ax, points):
    """将三维坐标轴设置为等比例，避免轨迹显示变形。"""
    points = np.asarray(points, dtype=np.float64)
    axis_min = points.min(axis=0)
    axis_max = points.max(axis=0)
    axis_center = (axis_min + axis_max) / 2.0
    half_range = np.max(axis_max - axis_min) / 2.0
    if half_range <= 0.0:
        half_range = 1.0

    ax.set_xlim(axis_center[0] - half_range, axis_center[0] + half_range)
    ax.set_ylim(axis_center[1] - half_range, axis_center[1] + half_range)
    ax.set_zlim(axis_center[2] - half_range, axis_center[2] + half_range)
def main():
    # 配置参数解析
    base_dir = ("/home/ma/GithubDoc/LuMoSDK/src/mocap_bridge/scripts/data")
    parser = argparse.ArgumentParser(description="动捕与视觉数据对齐与可视化工具")
    parser.add_argument("--dir",type=str,default=None,help="数据文件夹路径。",)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_handeye_file = os.path.join(
        script_dir,
        "detection",
        "calib",
        "handeye_calibration.json",
    )
    parser.add_argument(
        "--handeye-calib",
        type=str,
        default=default_handeye_file,
        help="手眼标定结果 handeye_calibration.json 路径",
    )
    args = parser.parse_args()

    if args.dir:
        data_dir = args.dir
        print(f"-> 正在读取[用户指定]的文件夹: {data_dir}")
    else:
        data_dir = get_latest_data_dir(base_dir)
        print(f"-> 未指定目录，正在自动读取[最新]的文件夹: {data_dir}")

    center_path = os.path.join(data_dir, "center.csv")
    mocap_path = os.path.join(data_dir, "mocap.csv")
    if not os.path.exists(center_path) or not os.path.exists(mocap_path):
        print(f"错误: 文件夹 {data_dir} 内未同时包含 center.csv 和 mocap.csv")
        return

    # 加载和预处理数据
    center_df = pd.read_csv(center_path)
    mocap_df = pd.read_csv(mocap_path)

    # 时间戳(秒)
    center_df["time"] = (center_df["timestamp_sec"] + center_df["timestamp_nanosec"] * 1e-9)
    mocap_df["time"] = (mocap_df["timestamp_sec"] + mocap_df["timestamp_nanosec"] * 1e-9)
    # 按时间排序
    center_df.sort_values("time", inplace=True)
    mocap_df.sort_values("time", inplace=True)

    # Rigid 5 是无反光贴纸红球的球心；Rigid 4 是相机刚体;marker_id 1 是标准球
    # ball_gt = mocap_df[(mocap_df["rigid_id"] == 5) & (mocap_df["is_track"] == 1)].dropna(axis=1, how="all")
    # gt_keys = ["rx", "ry", "rz"] 
    ball_gt = mocap_df[mocap_df['marker_id'] == 1].dropna(axis=1, how='all')
    gt_keys = ["x", "y", "z"] 
    cam_pose = mocap_df[(mocap_df["rigid_id"] == 4) & (mocap_df["is_track"] == 1)].dropna(axis=1, how="all")

    if center_df.empty or ball_gt.empty or cam_pose.empty:
        print("错误: center、Rigid 5 球心或 Rigid 4 相机位姿数据为空")
        print(f"center_df.empty={center_df.empty}")
        print(f"ball_gt.empty={ball_gt.empty}")
        print(f"cam_pose.empty={cam_pose.empty}")
        return

    # 对齐时间戳
    center_df = center_df.drop_duplicates(subset=["time"]).set_index("time")
    ball_gt = ball_gt.drop_duplicates(subset=["time"]).set_index("time")
    cam_pose = cam_pose.drop_duplicates(subset=["time"]).set_index("time")
    if len(ball_gt) < 2 or len(cam_pose) < 2:
        print("错误: Rigid 5 或 Rigid 4 的有效位姿少于 2 帧，无法插值")
        return
    start_time = max(center_df.index[0], ball_gt.index[0], cam_pose.index[0])
    end_time = min(center_df.index[-1], ball_gt.index[-1], cam_pose.index[-1])
    if start_time >= end_time:
        print("错误: center、Rigid 5 与 Rigid 4 没有重叠的时间范围")
        return
    aligned_df = center_df[(center_df.index >= start_time) & (center_df.index <= end_time)].copy()
    if aligned_df.empty:
        print("错误: 重叠时间范围内没有视觉球心数据")
        return

    # 插值
    aligned_df["gt_x"] = np.interp(aligned_df.index, ball_gt.index, ball_gt[gt_keys[0]])
    aligned_df["gt_y"] = np.interp(aligned_df.index, ball_gt.index, ball_gt[gt_keys[1]])
    aligned_df["gt_z"] = np.interp(aligned_df.index, ball_gt.index, ball_gt[gt_keys[2]])
    
    # 相机和 Rigid4 固定连接，但整个组合体会运动。对每个视觉球心时间戳，
    # 实时插值 Rigid4 在动捕世界坐标系中的平移和旋转，不能使用固定参考位姿。
    aligned_df["cam_rx"] = np.interp(aligned_df.index, cam_pose.index, cam_pose["rx"])
    aligned_df["cam_ry"] = np.interp(aligned_df.index, cam_pose.index, cam_pose["ry"])
    aligned_df["cam_rz"] = np.interp(aligned_df.index, cam_pose.index, cam_pose["rz"])

    key_times = cam_pose.index.values
    key_rots = R_scipy.from_quat(cam_pose[["qx", "qy", "qz", "qw"]].values)
    slerp = Slerp(key_times, key_rots)
    interp_rots = slerp(aligned_df.index.values)
    cam_T = aligned_df[["cam_rx", "cam_ry", "cam_rz"]].values

    # ---------------------------------------------------------
    # 核心修改区：使用手眼标定结果计算世界坐标
    # ---------------------------------------------------------
    meas_x_m = aligned_df["x"].values
    meas_y_m = aligned_df["y"].values
    meas_z_m = aligned_df["z"].values

    # 1. 组装视觉原始坐标点云 (P_cam)，注意这里直接用正值，不需要手动加负号反转，
    #    因为正确的标定旋转矩阵会自动处理相机坐标系与动捕坐标系轴向不一致的问题。
    #    此处乘 1000 将米转换为毫米，以与动捕单位保持统一。
    P_cam = np.vstack([meas_x_m * 1000.0, 
                       meas_y_m * 1000.0, 
                       meas_z_m * 1000.0]).T

    # 2. 【填入标定结果】输入你使用标定脚本跑出的四元数和平移向量
    # 标定出来的平移是米(m)，请乘以 1000 转换为毫米(mm)
    # 读取手眼标定结果
    handeye_path = os.path.abspath(os.path.expanduser(args.handeye_calib))

    if not os.path.isfile(handeye_path):
        raise FileNotFoundError(f"未找到手眼标定文件: {handeye_path}")

    with open(handeye_path, "r", encoding="utf-8") as f:
        handeye_data = json.load(f)

    rigid_id = handeye_data.get("rigid_id")
    if rigid_id is not None and int(rigid_id) != 4:
        raise ValueError(
            f"标定文件对应 rigid_id={rigid_id}，"
            "但 plot_auto_calib.py 当前使用的是 rigid_id=4"
        )

    selected = handeye_data["selected"]

    if selected.get("mocap_pose_direction") != "rigid_to_world":
        raise ValueError(
            "标定文件的动捕位姿方向不是 rigid_to_world，"
            "不能直接用于当前坐标变换链"
        )

    # 单位：T_rigid_camera 的平移为米
    T_rigid_camera = np.asarray(
        selected["T_rigid_camera"],
        dtype=np.float64,
    )

    if T_rigid_camera.shape != (4, 4):
        raise ValueError(
            f"T_rigid_camera 尺寸错误: {T_rigid_camera.shape}"
        )

    if not np.allclose(
        T_rigid_camera[3],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-8,
    ):
        raise ValueError("T_rigid_camera 最后一行不是 [0, 0, 0, 1]")

    R_calib = T_rigid_camera[:3, :3]

    # plot_auto_calib.py 内部的动捕数据使用毫米
    t_calib = T_rigid_camera[:3, 3] * 1000.0

    if not np.allclose(
        R_calib.T @ R_calib,
        np.eye(3),
        atol=1e-4,
    ):
        raise ValueError("T_rigid_camera 中的旋转矩阵不正交")

    if not np.isclose(np.linalg.det(R_calib), 1.0, atol=1e-4):
        raise ValueError("T_rigid_camera 中的旋转矩阵行列式不为 1")

    print(f"-> 已加载手眼标定: {handeye_path}")
    print(
        f"   method={selected.get('method')}, "
        f"translation RMSE={selected.get('translation_rmse_mm')} mm, "
        f"rotation RMSE={selected.get('rotation_rmse_deg')} deg"
    )
    print("   T_rigid_camera =")
    print(T_rigid_camera)

    # 3. 将相机坐标系点 -> 转换到相机刚体坐标系 (P_gripper = R_calib * P_cam + t_calib)
    # 利用矩阵乘法：(N,3) @ (3,3).T = (N,3)
    P_gripper = (P_cam @ R_calib.T) + t_calib

    # 4. 将相机刚体坐标系点 -> 转换到动捕世界坐标系
    # P_world = R_mocap * (R_calib * P_cam + t_calib) + T_mocap
    # 视觉点和两个刚体已经处于同一个动捕世界坐标系，不应再叠加人工
    # bias_world。固定世界偏置会掩盖手眼标定或时间同步问题。
    P_world = interp_rots.apply(P_gripper) + cam_T
    # ---------------------------------------------------------

    aligned_df["meas_world_x"] = P_world[:, 0]
    aligned_df["meas_world_y"] = P_world[:, 1]
    aligned_df["meas_world_z"] = P_world[:, 2]

    aligned_df["err_x"] = aligned_df["meas_world_x"] - aligned_df["gt_x"]
    aligned_df["err_y"] = aligned_df["meas_world_y"] - aligned_df["gt_y"]
    aligned_df["err_z"] = aligned_df["meas_world_z"] - aligned_df["gt_z"]

    errors = aligned_df[["err_x", "err_y", "err_z"]].to_numpy()
    mean_error = errors.mean(axis=0)
    axis_rmse = np.sqrt(np.mean(errors ** 2, axis=0))
    rmse_3d = np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))
    print(
        "-> 误差统计 (mm):\n"
        f"   mean XYZ = {np.round(mean_error, 3)}\n"
        f"   RMSE XYZ = {np.round(axis_rmse, 3)}\n"
        f"   3D RMSE  = {rmse_3d:.3f}"
    )
    
    # ================= 可视化绘图 (保持不变) =================
    relative_time = (aligned_df.index - aligned_df.index[0]).to_numpy()
    plt.rcParams.update({"font.size": 10})
    dir_name = os.path.basename(data_dir)

    # 绘图1:误差
    plt.figure(figsize=(10, 4))
    plt.fill_between(relative_time, -10.0, 10.0, color="#0bdb1d", alpha=0.15, label="±10mm Bound")
    plt.plot(relative_time, aligned_df["err_x"].to_numpy(), label="Error X", color="r", alpha=0.8)
    plt.plot(relative_time, aligned_df["err_y"].to_numpy(), label="Error Y", color="g", alpha=0.8)
    plt.plot(relative_time, aligned_df["err_z"].to_numpy(), label="Error Z", color="b", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Error (mm)")
    plt.title(f"Plot 1: Error in World Frame [{dir_name}]")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 绘图2:对比图
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    coords = ["x", "y", "z"]
    colors = ["r", "g", "b"]
    for i, axis in enumerate(coords):
        gt_data = aligned_df[f"gt_{axis}"].to_numpy()
        axes[i].fill_between(relative_time, gt_data - 10.0, gt_data + 10.0, color="#0bdb1d", alpha=0.15, label="±10mm Bound")
        axes[i].plot(relative_time, gt_data, label="Ground Truth", color="black", linestyle="--")
        axes[i].plot(relative_time, aligned_df[f"meas_world_{axis}"].to_numpy(), label="Measurement", color=colors[i], alpha=0.7)
        axes[i].set_ylabel(f"{axis.upper()} Position (mm)")
        axes[i].legend()
        axes[i].grid(True)
    axes[2].set_xlabel("Time (s)")
    fig.suptitle(f"Plot 2: Ground Truth vs Measurement [{dir_name}]", fontsize=12)
    plt.tight_layout()

    # 绘图3:动捕中rigid4位置
    # fig3, axes3 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    # rigid4_xyz = [cam_T[:, 0], cam_T[:, 1], cam_T[:, 2]]
    # coords_rigid4 = ["x", "y", "z"]
    # colors_rigid4 = ["r", "g", "b"]

    # for i, axis in enumerate(coords_rigid4):
    #     axes3[i].plot(
    #         relative_time,
    #         rigid4_xyz[i],
    #         label=f"Rigid 4 {axis.upper()}",
    #         color=colors_rigid4[i],
    #         alpha=0.8,
    #     )
    #     axes3[i].set_ylabel(f"World {axis.upper()} (mm)")
    #     axes3[i].legend(loc="upper right")
    #     axes3[i].grid(True)

    # axes3[2].set_xlabel("Time (s)")
    # fig3.suptitle(
    #     f"Plot 3: Rigid 4 Position in World Frame [{dir_name}]",
    #     fontsize=12,
    # )
    # fig3.tight_layout()

    # 绘图4:相机检测位置信息
    fig4, axes4 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    coords_raw = ["x", "y", "z"]
    colors_raw = ["r", "g", "b"]
    meas_local_mm = [meas_x_m * 1000.0, meas_y_m * 1000.0, meas_z_m * 1000.0]
    for i, axis in enumerate(coords_raw):
        axes4[i].plot(relative_time, meas_local_mm[i], label=f"Measured {axis.upper()} (Local)", color=colors_raw[i], alpha=0.8)
        axes4[i].set_ylabel(f"Local {axis.upper()} (mm)")
        axes4[i].legend(loc="upper right")
        axes4[i].grid(True)
    axes4[2].set_xlabel("Time (s)")
    fig4.suptitle(f"Plot 4: Camera Raw Detection in Local Frame [{dir_name}]", fontsize=12)
    fig4.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
