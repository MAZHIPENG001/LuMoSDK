import os
import time
import csv
import matplotlib.pyplot as plt

"""
原始数据 (records_raw)
补偿后数据 (records_comp)

raw/trajectory.csv 和 compensated/trajectory.csv。表格包含以下四列：
Time(s): 相对运行时间（秒）  X(m): X轴坐标（米）   Y(m): Y轴坐标（米）   Z(m): Z轴坐标（米）

可视化图表
利用 Matplotlib 生成丰富的图表来进行对比和验证：
    独立视图（分别为原始数据和补偿后数据生成）：
        合并折线图 (trajectory_combined.png)：X、Y、Z 三轴数据在同一张图内随时间变化的折线图。
        分离折线图 (trajectory_separated.png)：上中下 3 个子图，分别独立展示 X、Y、Z 轴随时间的变化。
        3D 空间轨迹图 (trajectory_3d.png)：绘制在三维坐标系中的移动轨迹，并用绿色圆点标记起点，红色叉号标记终点。
    对比视图（保存在 comparison 目录，用于分析补偿效果）：
        三轴对比图 (comparison_separated.png)：3 个子图，将补偿前（虚线）和补偿后（实线）的 X、Y、Z 数据画在同一张图里进行直观比对。
        3D 轨迹重叠图 (comparison_3d.png)：将原始 3D 轨迹和补偿后的 3D 轨迹放在同一个三维空间中展示。
        补偿差值图 (compensation_delta.png)：记录每一帧数据被补偿的差量（$\Delta = \text{Compensated} - \text{Raw}$），展示 $\Delta X$、$\Delta Y$、$\Delta Z$ 随时间的变化。
"""
class TargetTracker:
    def __init__(self, base_output_dir="output"):
        """
        初始化目标追踪器，创建时间戳主目录以及补偿前、补偿后、对比图的子目录
        """
        self.start_time = time.time()

        # 生成时间戳文件夹名，例如：20260626_163025
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(base_output_dir, timestamp)

        # 定义子文件夹路径
        self.raw_dir = os.path.join(self.save_dir, "raw")
        self.comp_dir = os.path.join(self.save_dir, "compensated")
        self.compare_dir = os.path.join(self.save_dir, "comparison")  # 新增：对比图文件夹

        # 创建目录
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.comp_dir, exist_ok=True)
        os.makedirs(self.compare_dir, exist_ok=True)

        # 分别记录两组数据，格式为: [(time, x, y, z), ...]
        self.records_raw = []
        self.records_comp = []

    def update(self, raw_x, raw_y, raw_z, comp_x, comp_y, comp_z):
        """
        在同一个时间步下，同时更新并记录补偿前（表面）和补偿后（球心）的坐标
        """
        current_time = time.time() - self.start_time

        if raw_x is not None and raw_y is not None and raw_z is not None:
            self.records_raw.append((current_time, raw_x, raw_y, raw_z))

        if comp_x is not None and comp_y is not None and comp_z is not None:
            self.records_comp.append((current_time, comp_x, comp_y, comp_z))

    def _save_and_plot_stream(self, records, target_dir, stream_name):
        """
        内部辅助方法：处理单组数据的保存和绘图逻辑
        """
        if not records:
            print(f"⚠️ 没有记录到 {stream_name} 的位置数据，跳过生成。")
            return

        csv_path = os.path.join(target_dir, "trajectory.csv")
        img_combined_path = os.path.join(target_dir, "trajectory_combined.png")
        img_separated_path = os.path.join(target_dir, "trajectory_separated.png")
        img_3d_path = os.path.join(target_dir, "trajectory_3d.png")

        # --- 1. 保存原始数据到 CSV ---
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Time(s)", "X(m)", "Y(m)", "Z(m)"])
            writer.writerows(records)

        # 提取数据用于绘图
        times = [r[0] for r in records]
        xs = [r[1] for r in records]
        ys = [r[2] for r in records]
        zs = [r[3] for r in records]

        # --- 2. 绘制合并折线图 ---
        plt.figure(figsize=(10, 6))
        plt.plot(times, xs, label='X Axis', color='red', alpha=0.8)
        plt.plot(times, ys, label='Y Axis', color='green', alpha=0.8)
        plt.plot(times, zs, label='Z Axis', color='blue', alpha=0.8)
        plt.title(f"Target Position Over Time ({stream_name} - Combined)")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(img_combined_path, dpi=300, bbox_inches='tight')
        plt.close()

        # --- 3. 绘制 3x1 分离的折线图 ---
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        axs[0].plot(times, xs, label='X Axis', color='red')
        axs[0].set_ylabel("X (m)")
        axs[0].grid(True, linestyle='--', alpha=0.6)
        axs[0].legend(loc="upper right")
        axs[0].set_title(f"Target Position Separated ({stream_name})")

        axs[1].plot(times, ys, label='Y Axis', color='green')
        axs[1].set_ylabel("Y (m)")
        axs[1].grid(True, linestyle='--', alpha=0.6)
        axs[1].legend(loc="upper right")

        axs[2].plot(times, zs, label='Z Axis', color='blue')
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Z (m)")
        axs[2].grid(True, linestyle='--', alpha=0.6)
        axs[2].legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(img_separated_path, dpi=300, bbox_inches='tight')
        plt.close()

        # --- 4. 绘制 3D 空间轨迹图 ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(xs, ys, zs, label='Trajectory', color='purple', linewidth=2)
        if len(xs) > 0:
            ax.scatter(xs[0], ys[0], zs[0], color='green', s=60, label='Start', zorder=5)
            ax.scatter(xs[-1], ys[-1], zs[-1], color='red', marker='X', s=60, label='End', zorder=5)

        ax.set_title(f"3D Spatial Trajectory ({stream_name})")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()

        plt.savefig(img_3d_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _save_comparison_plots(self):
        """
        内部辅助方法：生成补偿前后的对比图
        """
        # 确保两组数据都不为空且长度一致（因为是一起记录的，通常是对齐的）
        if not self.records_raw or not self.records_comp:
            print("⚠️ 缺少足够的数据进行对比图生成，跳过。")
            return

        # 提取时间轴和坐标数据
        times = [r[0] for r in self.records_raw]

        raw_xs, raw_ys, raw_zs = [r[1] for r in self.records_raw], [r[2] for r in self.records_raw], [r[3] for r in
                                                                                                      self.records_raw]
        comp_xs, comp_ys, comp_zs = [r[1] for r in self.records_comp], [r[2] for r in self.records_comp], [r[3] for r in
                                                                                                           self.records_comp]

        # 路径定义
        img_sep_comp_path = os.path.join(self.compare_dir, "comparison_separated.png")
        img_3d_comp_path = os.path.join(self.compare_dir, "comparison_3d.png")
        img_delta_path = os.path.join(self.compare_dir, "compensation_delta.png")

        # --- 1. 3x1 分离轴对比图 ---
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        axs[0].plot(times, raw_xs, label='Raw (Surface)', color='lightcoral', linestyle='--')
        axs[0].plot(times, comp_xs, label='Compensated (Center)', color='red')
        axs[0].set_ylabel("X (m)")
        axs[0].legend(loc="upper right")
        axs[0].grid(True, linestyle='--', alpha=0.6)
        axs[0].set_title("Trajectory Comparison (Raw vs Compensated)")

        axs[1].plot(times, raw_ys, label='Raw (Surface)', color='lightgreen', linestyle='--')
        axs[1].plot(times, comp_ys, label='Compensated (Center)', color='green')
        axs[1].set_ylabel("Y (m)")
        axs[1].legend(loc="upper right")
        axs[1].grid(True, linestyle='--', alpha=0.6)

        axs[2].plot(times, raw_zs, label='Raw (Surface)', color='lightblue', linestyle='--')
        axs[2].plot(times, comp_zs, label='Compensated (Center)', color='blue')
        axs[2].set_ylabel("Z (m)")
        axs[2].set_xlabel("Time (s)")
        axs[2].legend(loc="upper right")
        axs[2].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(img_sep_comp_path, dpi=300, bbox_inches='tight')
        plt.close()

        # --- 2. 3D 轨迹重叠图 ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(raw_xs, raw_ys, raw_zs, label='Raw Trajectory', color='blue', alpha=0.5, linestyle='--')
        ax.plot(comp_xs, comp_ys, comp_zs, label='Compensated Trajectory', color='red', linewidth=2)

        ax.set_title("3D Trajectory Comparison")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        plt.savefig(img_3d_comp_path, dpi=300, bbox_inches='tight')
        plt.close()

        # --- 3. 补偿差值（Delta）图 ---
        # 计算每一帧补偿的差值：Delta = Compensated - Raw
        delta_xs = [c - r for c, r in zip(comp_xs, raw_xs)]
        delta_ys = [c - r for c, r in zip(comp_ys, raw_ys)]
        delta_zs = [c - r for c, r in zip(comp_zs, raw_zs)]

        plt.figure(figsize=(10, 6))
        plt.plot(times, delta_xs, label='ΔX', color='red', alpha=0.8)
        plt.plot(times, delta_ys, label='ΔY', color='green', alpha=0.8)
        plt.plot(times, delta_zs, label='ΔZ', color='blue', alpha=0.8)

        plt.title("Compensation Delta Over Time (Center - Surface)")
        plt.xlabel("Time (s)")
        plt.ylabel("Delta (m)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(img_delta_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_and_plot(self):
        """
        统一触发保存：分别处理 raw、compensated 数据，并生成对比图
        """
        print(f"\n正在生成轨迹图和数据文件，主目录: {self.save_dir}")

        # 1. 生成原始数据和图表
        self._save_and_plot_stream(self.records_raw, self.raw_dir, "Raw")

        # 2. 生成补偿后数据和图表
        self._save_and_plot_stream(self.records_comp, self.comp_dir, "Compensated")

        # 3. 生成综合对比图表
        self._save_comparison_plots()

        print("✅ 生成完毕！包含了独立数据和对比图表。")