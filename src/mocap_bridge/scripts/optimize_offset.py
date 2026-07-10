import argparse
import glob
import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R_scipy
from scipy.spatial.transform import Slerp

def get_latest_data_dir(base_dir):
    """获取基础目录下名称排序最后的文件夹"""
    subdirs = [
        d for d in glob.glob(os.path.join(base_dir, "*"))
        if os.path.isdir(d) and not d.startswith(".")
    ]
    if not subdirs:
        raise FileNotFoundError(f"在 {base_dir} 路径下未找到任何数据文件夹！")
    latest_dir = max(subdirs, key=os.path.basename)
    return latest_dir

def load_and_align(data_dir):
    """加载数据，时间对齐，插值得到每一帧的测量值、真值和相机位姿"""
    center_path = os.path.join(data_dir, "center.csv")
    mocap_path = os.path.join(data_dir, "mocap.csv")
    if not os.path.exists(center_path) or not os.path.exists(mocap_path):
        raise FileNotFoundError(f"文件夹 {data_dir} 内未同时包含 center.csv 和 mocap.csv")

    center_df = pd.read_csv(center_path)
    mocap_df = pd.read_csv(mocap_path)

    center_df["time"] = center_df["timestamp_sec"] + center_df["timestamp_nanosec"] * 1e-9
    mocap_df["time"] = mocap_df["timestamp_sec"] + mocap_df["timestamp_nanosec"] * 1e-9
    center_df.sort_values("time", inplace=True)
    mocap_df.sort_values("time", inplace=True)

    # 提取球真值 (rigid_id==5 的位置，单位 mm)
    ball_gt = mocap_df[(mocap_df["rigid_id"] == 5) & (mocap_df["is_track"] == 1)].dropna(axis=1, how="all")
    gt_keys = ["rx", "ry", "rz"]   # 刚体位置

    # 提取相机刚体位姿 (rigid_id==4)
    cam_pose = mocap_df[(mocap_df["rigid_id"] == 4) & (mocap_df["is_track"] == 1)].dropna(axis=1, how="all")

    # 设置时间索引
    center_df = center_df.drop_duplicates(subset=["time"]).set_index("time")
    ball_gt = ball_gt.drop_duplicates(subset=["time"]).set_index("time")
    cam_pose = cam_pose.drop_duplicates(subset=["time"]).set_index("time")

    # 截取公共时间区间
    start_time = max(center_df.index[0], ball_gt.index[0], cam_pose.index[0])
    end_time = min(center_df.index[-1], ball_gt.index[-1], cam_pose.index[-1])
    aligned_df = center_df[(center_df.index >= start_time) & (center_df.index <= end_time)].copy()

    # 插值真值 (mm)
    aligned_df["gt_x"] = np.interp(aligned_df.index, ball_gt.index, ball_gt[gt_keys[0]])
    aligned_df["gt_y"] = np.interp(aligned_df.index, ball_gt.index, ball_gt[gt_keys[1]])
    aligned_df["gt_z"] = np.interp(aligned_df.index, ball_gt.index, ball_gt[gt_keys[2]])

    # 插值相机平移 (mm)
    aligned_df["cam_rx"] = np.interp(aligned_df.index, cam_pose.index, cam_pose["rx"])
    aligned_df["cam_ry"] = np.interp(aligned_df.index, cam_pose.index, cam_pose["ry"])
    aligned_df["cam_rz"] = np.interp(aligned_df.index, cam_pose.index, cam_pose["rz"])

    # 插值相机旋转 (四元数) —— 使用 Slerp
    key_times = cam_pose.index.values
    key_rots = R_scipy.from_quat(cam_pose[["qx", "qy", "qz", "qw"]].values)
    slerp = Slerp(key_times, key_rots)
    interp_rots = slerp(aligned_df.index.values)

    return aligned_df, interp_rots

def optimize_offset(aligned_df, interp_rots):
    """
    通过线性最小二乘法求解最优偏移 [ox, oy, oz]。
    假设相机坐标系 → 刚体坐标系： x_cam → -x, y_cam → -y, z_cam → z，然后平移 offset。
    公式： P_local = [-x*1000 + ox, -y*1000 + oy, z*1000 + oz]   (单位 mm)
          P_world = R * P_local + t
    目标： P_world ≈ gt
    因此： R * offset = gt - R * (-x*1000, -y*1000, z*1000) - t
    将所有帧堆叠成 A * offset = b
    """
    # 提取数据 (单位转换)
    x_m = aligned_df["x"].values          # 米
    y_m = aligned_df["y"].values
    z_m = aligned_df["z"].values
    gt_x = aligned_df["gt_x"].values      # mm
    gt_y = aligned_df["gt_y"].values
    gt_z = aligned_df["gt_z"].values
    t_x = aligned_df["cam_rx"].values     # mm
    t_y = aligned_df["cam_ry"].values
    t_z = aligned_df["cam_rz"].values
    # 旋转矩阵列表 (3x3)
    rots = interp_rots.as_matrix()        # shape (N, 3, 3)

    N = len(x_m)
    # 构建 m_i = [-1000*x_i, -1000*y_i, 1000*z_i]  (mm)
    m = np.column_stack([-1000*x_m, -1000*y_m, 1000*z_m])   # (N, 3)

    # 构建线性方程组 A * offset = b
    # 对于每一帧 i: R_i * offset = gt_i - R_i * m_i - t_i
    # 展开：R_i 的每一行乘以 offset 等于对应行的标量
    # 构造 A 为 (3N, 3), b 为 (3N,)
    A_list = []
    b_list = []
    for i in range(N):
        R = rots[i]          # 3x3
        t = np.array([t_x[i], t_y[i], t_z[i]])
        gt = np.array([gt_x[i], gt_y[i], gt_z[i]])
        # 右侧: gt - R @ m[i] - t
        rhs = gt - R @ m[i] - t
        # A 的块: R
        A_list.append(R)
        b_list.append(rhs)
    A = np.vstack(A_list)    # (3N, 3)
    b = np.concatenate(b_list)  # (3N,)

    # 最小二乘求解
    offset, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return offset, residuals

def main():
    base_dir = "/home/ma/GithubDoc/LuMoSDK/src/mocap_bridge/scripts/data"
    parser = argparse.ArgumentParser(description="优化相机到动捕刚体的偏移量")
    parser.add_argument("--dir", type=str, default=None, help="数据文件夹路径")
    args = parser.parse_args()

    if args.dir:
        data_dir = args.dir
        print(f"-> 使用用户指定文件夹: {data_dir}")
    else:
        data_dir = get_latest_data_dir(base_dir)
        print(f"-> 使用最新数据文件夹: {data_dir}")

    try:
        aligned_df, interp_rots = load_and_align(data_dir)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    offset, residuals = optimize_offset(aligned_df, interp_rots)
    ox, oy, oz = offset
    print(f"\n优化得到的偏移量 (单位: mm):")
    print(f"offset_x = {ox:.3f}")
    print(f"offset_y = {oy:.3f}")
    print(f"offset_z = {oz:.3f}")

    # 评估使用该偏移后的总体误差
    # 计算转换后的预测世界坐标
    x_m = aligned_df["x"].values
    y_m = aligned_df["y"].values
    z_m = aligned_df["z"].values
    m = np.column_stack([-1000*x_m, -1000*y_m, 1000*z_m])  # (N,3)
    P_local = m + offset   # (N,3)
    P_world = interp_rots.apply(P_local) + np.column_stack([aligned_df["cam_rx"], aligned_df["cam_ry"], aligned_df["cam_rz"]])
    gt = np.column_stack([aligned_df["gt_x"], aligned_df["gt_y"], aligned_df["gt_z"]])
    errors = P_world - gt
    rmse = np.sqrt(np.mean(errors**2))
    mean_err = np.mean(errors, axis=0)
    print(f"\n使用该偏移后的整体误差 (单位: mm):")
    print(f"RMSE: {rmse:.3f}")
    print(f"平均误差 (X, Y, Z): {mean_err[0]:.3f}, {mean_err[1]:.3f}, {mean_err[2]:.3f}")

    # 还可输出最大误差等
    max_err = np.max(np.abs(errors), axis=0)
    print(f"最大绝对误差 (X, Y, Z): {max_err[0]:.3f}, {max_err[1]:.3f}, {max_err[2]:.3f}")

if __name__ == "__main__":
    main()