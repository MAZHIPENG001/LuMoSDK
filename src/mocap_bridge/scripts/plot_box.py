import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_scipy
from scipy.spatial.transform import Slerp
import os

# ================= 1. 加载和预处理数据 =================
data_dir = '/home/ma/GithubDoc/LuMoSDK/src/mocap_bridge/scripts/data/20260702_113945'

# 读取 CSV 文件
center_df = pd.read_csv(os.path.join(data_dir, 'center.csv'))
mocap_df = pd.read_csv(os.path.join(data_dir, 'mocap.csv'))

# 合并秒和纳秒生成连续的时间戳 (秒为单位)
center_df['time'] = center_df['timestamp_sec'] + center_df['timestamp_nanosec'] * 1e-9
mocap_df['time'] = mocap_df['timestamp_sec'] + mocap_df['timestamp_nanosec'] * 1e-9

# 按时间排序
center_df.sort_values('time', inplace=True)
mocap_df.sort_values('time', inplace=True)

# 提取动捕中的 box数据(rigid 5) 和 相机位姿数据(rigid 4)
ball_gt = mocap_df[(mocap_df['rigid_id'] == 5) & (mocap_df['is_track'] == 1)].dropna(axis=1, how='all')
cam_pose = mocap_df[(mocap_df['rigid_id'] == 4) & (mocap_df['is_track'] == 1)].dropna(axis=1, how='all')

# 剔除重复的时间戳（为了后续插值安全）
center_df = center_df.drop_duplicates(subset=['time']).set_index('time')
ball_gt = ball_gt.drop_duplicates(subset=['time']).set_index('time')
cam_pose = cam_pose.drop_duplicates(subset=['time']).set_index('time')

# ================= 2. 时间戳对齐与插值 =================
# 找到有效的时间重叠区间
start_time = max(center_df.index[0], ball_gt.index[0], cam_pose.index[0])
end_time = min(center_df.index[-1], ball_gt.index[-1], cam_pose.index[-1])

# 截取重叠时间段的视觉数据
aligned_df = center_df[(center_df.index >= start_time) & (center_df.index <= end_time)].copy()

# 线性插值动捕足球真值
aligned_df['gt_x'] = np.interp(aligned_df.index, ball_gt.index, ball_gt['rx'])
aligned_df['gt_y'] = np.interp(aligned_df.index, ball_gt.index, ball_gt['ry'])
aligned_df['gt_z'] = np.interp(aligned_df.index, ball_gt.index, ball_gt['rz'])

# 线性插值相机平移 (T)
aligned_df['cam_rx'] = np.interp(aligned_df.index, cam_pose.index, cam_pose['rx'])
aligned_df['cam_ry'] = np.interp(aligned_df.index, cam_pose.index, cam_pose['ry'])
aligned_df['cam_rz'] = np.interp(aligned_df.index, cam_pose.index, cam_pose['rz'])

# 对相机旋转四元数进行 Slerp 球面插值
key_times = cam_pose.index.values
key_rots = R_scipy.from_quat(cam_pose[['qx', 'qy', 'qz', 'qw']].values)
slerp = Slerp(key_times, key_rots)
interp_rots = slerp(aligned_df.index.values)

# ================= 3. 坐标系转换计算 =================
# 获取相机测量的局部坐标并转换为毫米
meas_x_m = aligned_df['x'].values
meas_y_m = aligned_df['y'].values
meas_z_m = aligned_df['z'].values

# 根据坐标系定义：测量坐标系 -> 相机动捕刚体坐标系 (Z朝前, Y朝上, X朝左)
# 对应关系： X刚体 = -X测量, Y刚体 = -Y测量, Z刚体 = Z测量
P_local = np.vstack([-meas_x_m * 1000.0, 
                     -meas_y_m * 1000.0, 
                      meas_z_m * 1000.0]).T # 形状 (N, 3)

# 转换到世界坐标系: P_world = R * P_local + T
cam_T = aligned_df[['cam_rx', 'cam_ry', 'cam_rz']].values
P_world = interp_rots.apply(P_local) + cam_T

aligned_df['meas_world_x'] = P_world[:, 0]
aligned_df['meas_world_y'] = P_world[:, 1]
aligned_df['meas_world_z'] = P_world[:, 2]

# 计算误差 (测量值 - 真实值)
aligned_df['err_x'] = aligned_df['meas_world_x'] - aligned_df['gt_x']
aligned_df['err_y'] = aligned_df['meas_world_y'] - aligned_df['gt_y']
aligned_df['err_z'] = aligned_df['meas_world_z'] - aligned_df['gt_z']

# ================= 4. 可视化绘图 =================
relative_time = (aligned_df.index - aligned_df.index[0]).to_numpy()

# 设置全局字体大小等
plt.rcParams.update({'font.size': 10})

# 绘图1：真实值与测量值的X, Y, Z误差
plt.figure(figsize=(10, 5))
plt.plot(relative_time, aligned_df['err_x'].to_numpy(), label='Error X', alpha=0.8)
plt.plot(relative_time, aligned_df['err_y'].to_numpy(), label='Error Y', alpha=0.8)
plt.plot(relative_time, aligned_df['err_z'].to_numpy(), label='Error Z', alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Error (mm)')
plt.title('Plot 1: Measurement Error in World Frame (Measured - GroundTruth)')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()

# 绘图2：真实值与测量值 X, Y, Z 对比图
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
coords = ['x', 'y', 'z']
colors = ['r', 'g', 'b']
for i, axis in enumerate(coords):
    axes[i].plot(relative_time, aligned_df[f'gt_{axis}'].to_numpy(), label='Ground Truth', color='black', linestyle='--')
    axes[i].plot(relative_time, aligned_df[f'meas_world_{axis}'].to_numpy(), label='Measurement', color=colors[i], alpha=0.7)
    axes[i].set_ylabel(f'{axis.upper()} Position (mm)')
    axes[i].legend()
    axes[i].grid(True)
axes[2].set_xlabel('Time (s)')
fig.suptitle('Plot 2: Ground Truth vs Measurement Comparison', fontsize=14)
plt.tight_layout()
# plt.show()

# 绘图3：3D 位置对比图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# 3D 绘图同样需要全部转换为 numpy 数组
ax.plot(aligned_df['gt_x'].to_numpy(), aligned_df['gt_y'].to_numpy(), aligned_df['gt_z'].to_numpy(), 
        label='Ground Truth Trajectory', color='black', linewidth=2)
ax.plot(aligned_df['meas_world_x'].to_numpy(), aligned_df['meas_world_y'].to_numpy(), aligned_df['meas_world_z'].to_numpy(), 
        label='Measured Trajectory', color='blue', alpha=0.6, linewidth=2)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.legend()
plt.title('Plot 3: 3D Trajectory of Football (World Frame)')
plt.show()