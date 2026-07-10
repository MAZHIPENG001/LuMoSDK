import argparse
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R_scipy
from scipy.spatial.transform import Slerp

"""
python3 /home/ma/GithubDoc/LuMoSDK/src/mocap_bridge/scripts/plot_auto.py --dir 
python3 /home/ma/GithubDoc/LuMoSDK/src/mocap_bridge/scripts/plot_auto.py
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


def main():
    # 配置参数解析
    base_dir = ("/home/ma/GithubDoc/LuMoSDK/src/mocap_bridge/scripts/data")
    parser = argparse.ArgumentParser(description="动捕与视觉数据对齐与可视化工具")
    parser.add_argument("--dir",type=str,default=None,help="数据文件夹路径。",)
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

    """
        rigid_id==5:红球
        marker_id==1:反光球
        rigid_id==4:d435
    """
        
    ball_gt = mocap_df[(mocap_df["rigid_id"] == 5) & (mocap_df["is_track"] == 1)].dropna(axis=1, how="all")
    gt_keys = ["rx", "ry", "rz"] 
    # ball_gt = mocap_df[mocap_df['marker_id'] == 1].dropna(axis=1, how='all')
    # gt_keys = ["x", "y", "z"] 
    # 提取动捕 相机位姿数据(rigid 4)
    cam_pose = mocap_df[(mocap_df["rigid_id"] == 4) & (mocap_df["is_track"] == 1)].dropna(axis=1, how="all")

    # 时间戳
    center_df = center_df.drop_duplicates(subset=["time"]).set_index("time")
    ball_gt = ball_gt.drop_duplicates(subset=["time"]).set_index("time")
    cam_pose = cam_pose.drop_duplicates(subset=["time"]).set_index("time")
    start_time = max(center_df.index[0], ball_gt.index[0], cam_pose.index[0])
    end_time = min(center_df.index[-1], ball_gt.index[-1], cam_pose.index[-1])
    aligned_df = center_df[(center_df.index >= start_time) & (center_df.index <= end_time)].copy()

    # 根据选定的键值进行插值
    # 1.检测物体标准位置
    aligned_df["gt_x"] = np.interp(aligned_df.index, ball_gt.index, ball_gt[gt_keys[0]])
    aligned_df["gt_y"] = np.interp(aligned_df.index, ball_gt.index, ball_gt[gt_keys[1]])
    aligned_df["gt_z"] = np.interp(aligned_df.index, ball_gt.index, ball_gt[gt_keys[2]])
    # 2.相机
    aligned_df["cam_rx"] = np.interp(aligned_df.index, cam_pose.index, cam_pose["rx"])
    aligned_df["cam_ry"] = np.interp(aligned_df.index, cam_pose.index, cam_pose["ry"])
    aligned_df["cam_rz"] = np.interp(aligned_df.index, cam_pose.index, cam_pose["rz"])

    # 实时读取
    key_times = cam_pose.index.values
    key_rots = R_scipy.from_quat(cam_pose[["qx", "qy", "qz", "qw"]].values)
    slerp = Slerp(key_times, key_rots)
    interp_rots = slerp(aligned_df.index.values)
    
    # # 第一帧
    # aligned_df["cam_rx"] = cam_pose["rx"].iloc[0]
    # aligned_df["cam_ry"] = cam_pose["ry"].iloc[0]
    # aligned_df["cam_rz"] = cam_pose["rz"].iloc[0]
    # # 提取第一帧的旋转四元数
    # first_quat = cam_pose[["qx", "qy", "qz", "qw"]].iloc[0].values
    # # 构造与 aligned_df 长度相同的旋转对象，替代原本 slerp 的结果,将第一帧的四元数通过 np.tile 复制 N 份，以适配下方 P_world 批量 apply 的逻辑
    # interp_rots = R_scipy.from_quat(np.tile(first_quat, (len(aligned_df), 1)))

    # 坐标系转换计算
    meas_x_m = aligned_df["x"].values
    meas_y_m = aligned_df["y"].values
    meas_z_m = aligned_df["z"].values

    offset_x = 100
    offset_y = -25
    offset_z = 8
    offset_x = 159.574
    offset_y = 51.985
    offset_z = -51.035

    P_local = np.vstack([-meas_x_m * 1000.0 + offset_x, 
                         -meas_y_m * 1000.0 + offset_y, 
                         meas_z_m * 1000.0 + offset_z]).T

    cam_T = aligned_df[["cam_rx", "cam_ry", "cam_rz"]].values
    P_world = interp_rots.apply(P_local) + cam_T

    world_error_x = 0
    world_error_y = 0
    world_error_z = 0
    aligned_df["meas_world_x"] = P_world[:, 0] + world_error_x
    aligned_df["meas_world_y"] = P_world[:, 1] + world_error_y
    aligned_df["meas_world_z"] = P_world[:, 2] + world_error_z
    # aligned_df["meas_world_x"] = P_world[:, 0]
    # aligned_df["meas_world_y"] = P_world[:, 1]
    # aligned_df["meas_world_z"] = P_world[:, 2]

    aligned_df["err_x"] = aligned_df["meas_world_x"] - aligned_df["gt_x"]
    aligned_df["err_y"] = aligned_df["meas_world_y"] - aligned_df["gt_y"]
    aligned_df["err_z"] = aligned_df["meas_world_z"] - aligned_df["gt_z"]
    
    # ================= 可视化绘图 =================
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

    # # 绘图3:3d轨迹
    # fig = plt.figure(figsize=(9, 7))
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot(aligned_df["gt_x"].to_numpy(),
    #         aligned_df["gt_y"].to_numpy(),
    #         aligned_df["gt_z"].to_numpy(),
    #         label="Ground Truth Trajectory",color="black",linewidth=2,)
    # ax.plot(aligned_df["meas_world_x"].to_numpy(),
    #         aligned_df["meas_world_y"].to_numpy(),
    #         aligned_df["meas_world_z"].to_numpy(),
    #         label="Measured Trajectory",color="blue",alpha=0.6,linewidth=2,)
    # ax.set_xlabel("X (mm)")
    # ax.set_ylabel("Y (mm)")
    # ax.set_zlabel("Z (mm)")
    # ax.legend()
    # plt.title(f"Plot 3: 3D Trajectory of Football [{dir_name}]")
    
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

    # # 绘图5: 相机在世界坐标系中的移动和旋转关系图
    # fig5, axes5 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # # 5.1 相机平移 (Translation)
    # axes5[0].plot(relative_time, aligned_df["cam_rx"].to_numpy(), label="Cam Translation X", color="r", alpha=0.8)
    # axes5[0].plot(relative_time, aligned_df["cam_ry"].to_numpy(), label="Cam Translation Y", color="g", alpha=0.8)
    # axes5[0].plot(relative_time, aligned_df["cam_rz"].to_numpy(), label="Cam Translation Z", color="b", alpha=0.8)
    # axes5[0].set_ylabel("Translation (mm)")
    # axes5[0].set_title("Camera Translation in World Frame")
    # axes5[0].legend(loc="upper right")
    # axes5[0].grid(True)
    # # 5.2 相机旋转 (Rotation - 转换为欧拉角便于观察)
    # cam_euler = interp_rots.as_euler('xyz', degrees=True)
    # axes5[1].plot(relative_time, cam_euler[:, 0], label="Roll (X)", color="r", alpha=0.8)
    # axes5[1].plot(relative_time, cam_euler[:, 1], label="Pitch (Y)", color="g", alpha=0.8)
    # axes5[1].plot(relative_time, cam_euler[:, 2], label="Yaw (Z)", color="b", alpha=0.8)
    # axes5[1].set_ylabel("Rotation (Degrees)")
    # axes5[1].set_xlabel("Time (s)")
    # axes5[1].set_title("Camera Rotation in World Frame (Euler Angles)")
    # axes5[1].legend(loc="upper right")
    # axes5[1].grid(True)
    # fig5.suptitle(f"Plot 5: Camera Pose Relationship [{dir_name}]", fontsize=12)
    # fig5.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    main()