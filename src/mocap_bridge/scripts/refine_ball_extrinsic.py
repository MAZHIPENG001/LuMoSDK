#!/usr/bin/env python3
"""Refine camera-to-Rigid-4 alignment from ball-center/mocap pairs.

This estimates an *effective* T_rigid_camera that absorbs repeatable RGB-D
ball-center bias as well as residual hand-eye bias.  Use several recordings
covering different camera-to-ball directions and distances, then validate the
result on a recording that was not used here.
"""

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp


def make_transform(rotation, translation_m):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation_m
    return transform


def invert_transform(transform):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return result


def add_time(dataframe):
    dataframe = dataframe.copy()
    dataframe["time"] = (
        dataframe["timestamp_sec"]
        + dataframe["timestamp_nanosec"] * 1e-9
    )
    return dataframe.sort_values("time").drop_duplicates("time")


def load_recording(data_dir):
    center_path = os.path.join(data_dir, "center_raw.csv")
    mocap_path = os.path.join(data_dir, "mocap.csv")
    if not os.path.isfile(center_path) or not os.path.isfile(mocap_path):
        raise FileNotFoundError(
            f"{data_dir} must contain center_raw.csv and mocap.csv"
        )

    center = add_time(pd.read_csv(center_path))
    mocap = add_time(pd.read_csv(mocap_path))
    marker = mocap[mocap["marker_id"] == 1].dropna(
        subset=["x", "y", "z"]
    )
    rigid = mocap[
        (mocap["rigid_id"] == 4) & (mocap["is_track"] == 1)
    ].dropna(subset=["rx", "ry", "rz", "qx", "qy", "qz", "qw"])

    if len(center) < 3 or len(marker) < 2 or len(rigid) < 2:
        raise ValueError(f"not enough valid data in {data_dir}")

    start = max(center["time"].iloc[0], marker["time"].iloc[0], rigid["time"].iloc[0])
    end = min(center["time"].iloc[-1], marker["time"].iloc[-1], rigid["time"].iloc[-1])
    center = center[(center["time"] >= start) & (center["time"] <= end)]
    if center.empty:
        raise ValueError(f"no overlapping timestamps in {data_dir}")

    times = center["time"].to_numpy()
    point_camera_mm = center[["x", "y", "z"]].to_numpy() * 1000.0
    marker_world_mm = np.column_stack(
        [np.interp(times, marker["time"], marker[key]) for key in ["x", "y", "z"]]
    )
    rigid_translation_mm = np.column_stack(
        [
            np.interp(times, rigid["time"], rigid[key])
            for key in ["rx", "ry", "rz"]
        ]
    )
    rigid_rotation = Slerp(
        rigid["time"].to_numpy(),
        Rotation.from_quat(rigid[["qx", "qy", "qz", "qw"]].to_numpy()),
    )(times)

    # The desired point in Rigid-4 coordinates follows directly from the
    # mocap world point and the measured Rigid-4 pose.
    point_rigid_mm = rigid_rotation.inv().apply(
        marker_world_mm - rigid_translation_mm
    )
    return {
        "name": os.path.basename(os.path.abspath(data_dir)),
        "data_dir": os.path.abspath(data_dir),
        "point_camera_mm": point_camera_mm,
        "point_rigid_mm": point_rigid_mm,
        "marker_world_mm": marker_world_mm,
        "rigid_translation_mm": rigid_translation_mm,
        "rigid_rotation": rigid_rotation,
    }


def weighted_rigid_fit(source, target, weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    source_mean = np.sum(source * weights[:, None], axis=0)
    target_mean = np.sum(target * weights[:, None], axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ (weights[:, None] * target_centered)
    left, singular_values, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    translation = target_mean - rotation @ source_mean
    return rotation, translation, singular_values


def refine_transform(recordings, max_samples_per_dir):
    source_parts = []
    target_parts = []
    weight_parts = []
    for recording in recordings:
        count = len(recording["point_camera_mm"])
        sample_count = min(count, max_samples_per_dir)
        indices = np.linspace(0, count - 1, sample_count, dtype=np.int64)
        source_parts.append(recording["point_camera_mm"][indices])
        target_parts.append(recording["point_rigid_mm"][indices])
        # Give each recording equal total weight regardless of duration.
        weight_parts.append(np.full(sample_count, 1.0 / sample_count))

    source = np.vstack(source_parts)
    target = np.vstack(target_parts)
    weights = np.concatenate(weight_parts)
    rotation, translation_mm, singular_values = weighted_rigid_fit(
        source, target, weights
    )
    return rotation, translation_mm, singular_values


def recording_metrics(recording, transform):
    rotation = transform[:3, :3]
    translation_mm = transform[:3, 3] * 1000.0
    point_rigid = (
        recording["point_camera_mm"] @ rotation.T + translation_mm
    )
    point_world = recording["rigid_rotation"].apply(point_rigid)
    point_world += recording["rigid_translation_mm"]
    error = point_world - recording["marker_world_mm"]
    mean = np.mean(error, axis=0)
    return {
        "sample_count": int(len(error)),
        "mean_error_mm": mean.tolist(),
        "mean_error_norm_mm": float(np.linalg.norm(mean)),
        "rmse_3d_mm": float(
            np.sqrt(np.mean(np.sum(error * error, axis=1)))
        ),
        "centered_rmse_3d_mm": float(
            np.sqrt(np.mean(np.sum((error - mean) ** 2, axis=1)))
        ),
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(
        script_dir, "detection", "calib", "handeye_calibration.json"
    )
    default_output = os.path.join(
        script_dir, "detection", "calib", "handeye_ball_refined.json"
    )
    parser = argparse.ArgumentParser(
        description="用视觉球心和 Marker 1 真值修正 camera -> Rigid 4 外参"
    )
    parser.add_argument("--dirs", nargs="+", required=True, help="标定数据目录")
    parser.add_argument("--input", default=default_input, help="原手眼标定 JSON")
    parser.add_argument("--output", default=default_output, help="候选标定输出 JSON")
    parser.add_argument("--max-samples-per-dir", type=int, default=200)
    args = parser.parse_args()

    if len(args.dirs) < 3:
        raise ValueError("至少需要 3 组、推荐 6 组以上不同球位置的数据")
    recordings = [load_recording(path) for path in args.dirs]
    with open(args.input, "r", encoding="utf-8") as input_file:
        original = json.load(input_file)
    old_transform = np.asarray(
        original["selected"]["T_rigid_camera"], dtype=np.float64
    )

    rotation, translation_mm, singular_values = refine_transform(
        recordings, max(10, args.max_samples_per_dir)
    )
    geometry_ratio = float(singular_values[-1] / singular_values[0])
    if geometry_ratio < 0.01:
        raise ValueError(
            "采样位置的三维分布退化，无法可靠估计旋转；"
            f"最小/最大奇异值比={geometry_ratio:.5f}，需要增加横向、纵向和深度变化"
        )

    new_transform = make_transform(rotation, translation_mm / 1000.0)
    old_metrics = {
        item["name"]: recording_metrics(item, old_transform)
        for item in recordings
    }
    new_metrics = {
        item["name"]: recording_metrics(item, new_transform)
        for item in recordings
    }

    output = copy.deepcopy(original)
    output["base_handeye_selected"] = copy.deepcopy(original["selected"])
    output["selected"] = copy.deepcopy(original["selected"])
    output["selected"].update(
        {
            "method": "ball-center-point-refinement",
            "T_rigid_camera": new_transform.tolist(),
            "quaternion_xyzw": Rotation.from_matrix(rotation).as_quat().tolist(),
            "translation_m": (translation_mm / 1000.0).tolist(),
            "translation_rmse_mm": float(
                np.mean([value["rmse_3d_mm"] for value in new_metrics.values()])
            ),
            "rotation_rmse_deg": None,
        }
    )
    output["T_camera_rigid"] = invert_transform(new_transform).tolist()
    output["ball_center_refinement"] = {
        "description": (
            "Effective camera optical -> Rigid 4 transform fitted from "
            "center_raw.csv and mocap Marker 1 correspondences"
        ),
        "source_directories": [item["data_dir"] for item in recordings],
        "max_samples_per_directory": args.max_samples_per_dir,
        "geometry_singular_values": singular_values.tolist(),
        "geometry_min_max_ratio": geometry_ratio,
        "old_metrics": old_metrics,
        "refined_metrics": new_metrics,
    }

    output_path = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(output, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    delta_rotation_deg = float(
        np.degrees(
            (
                Rotation.from_matrix(rotation)
                * Rotation.from_matrix(old_transform[:3, :3]).inv()
            ).magnitude()
        )
    )
    print(f"geometry min/max ratio: {geometry_ratio:.5f}")
    print(f"rotation correction: {delta_rotation_deg:.3f} deg")
    print(
        "translation correction (mm):",
        np.round(translation_mm - old_transform[:3, 3] * 1000.0, 3),
    )
    for name in new_metrics:
        print(
            f"{name}: steady error norm "
            f"{old_metrics[name]['mean_error_norm_mm']:.2f} -> "
            f"{new_metrics[name]['mean_error_norm_mm']:.2f} mm"
        )
    print("saved candidate calibration to:", output_path)


if __name__ == "__main__":
    main()
