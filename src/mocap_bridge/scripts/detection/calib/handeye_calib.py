#!/usr/bin/env python3
"""Calculate T_rigid_camera from synchronized ChArUco/mocap pose pairs.

Input is produced by ``charuco_mocap_collect.py``.  The board is assumed to be
stationary in the mocap world.  OpenCV's eye-in-hand convention is used:

    gripper -> rigid body 4 (camera bracket)
    base    -> mocap world
    target  -> ChArUco board
    camera  -> RealSense color optical frame

The main result, T_rigid_camera, maps points from the RealSense color optical
frame into the rigid-body coordinate frame.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover - depends on ROS installation
    raise SystemExit("PyYAML is required: sudo apt install python3-yaml") from exc


SCRIPT_DIR = Path(__file__).resolve().parent


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(
        np.clip((np.trace(np.asarray(rotation).reshape(3, 3)) - 1.0) * 0.5, -1.0, 1.0)
    )
    return math.degrees(math.acos(cosine))


def project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(matrix, dtype=np.float64).reshape(3, 3))
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation


def rotation_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a normalized [x, y, z, w] quaternion."""
    rotation = project_to_rotation(rotation)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = math.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])) * 2.0
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
            w = (rotation[2, 1] - rotation[1, 2]) / s
        elif index == 1:
            s = math.sqrt(max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])) * 2.0
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
            w = (rotation[0, 2] - rotation[2, 0]) / s
        else:
            s = math.sqrt(max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])) * 2.0
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s
            w = (rotation[1, 0] - rotation[0, 1]) / s
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


def quaternion_xyzw_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1.0e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def average_rotation(rotations: Sequence[np.ndarray]) -> np.ndarray:
    accumulator = np.zeros((4, 4), dtype=np.float64)
    for rotation in rotations:
        quaternion = rotation_to_quaternion_xyzw(rotation)
        accumulator += np.outer(quaternion, quaternion)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    mean_quaternion = eigenvectors[:, int(np.argmax(eigenvalues))]
    if mean_quaternion[3] < 0.0:
        mean_quaternion *= -1.0
    return quaternion_xyzw_to_rotation(mean_quaternion)


def validate_transform(value: Any, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} is not a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-6):
        raise ValueError(f"{name} has an invalid last row")
    rotation = project_to_rotation(transform[:3, :3])
    if np.linalg.norm(rotation - transform[:3, :3]) > 1.0e-3:
        raise ValueError(f"{name} contains an invalid rotation")
    result = transform.copy()
    result[:3, :3] = rotation
    return result


def finite_or_none(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_samples(
    input_path: Path,
    max_reprojection_error_px: float,
    max_sync_error_ms: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    with input_path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if int(document.get("schema_version", -1)) != 1:
        raise ValueError("unsupported or missing sample schema_version")
    if not bool(document.get("board_must_be_stationary_in_world", False)):
        raise ValueError("input does not declare a stationary board")

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for position, raw in enumerate(document.get("samples", [])):
        index = int(raw.get("index", position))
        reasons: List[str] = []
        reprojection = finite_or_none(raw.get("reprojection_error_px"))
        sync_error = finite_or_none(raw.get("sync_error_ms"))
        if reprojection is None or reprojection > max_reprojection_error_px:
            reasons.append("reprojection_error")
        if sync_error is None or sync_error > max_sync_error_ms:
            reasons.append("sync_error")
        try:
            transform_world_rigid = validate_transform(
                raw.get("T_world_rigid"), f"sample {index} T_world_rigid"
            )
            transform_camera_board = validate_transform(
                raw.get("T_camera_board"), f"sample {index} T_camera_board"
            )
        except ValueError as exc:
            reasons.append(str(exc))
            transform_world_rigid = np.eye(4)
            transform_camera_board = np.eye(4)

        item = {
            "index": index,
            "raw": raw,
            "T_world_rigid": transform_world_rigid,
            "T_camera_board": transform_camera_board,
        }
        if reasons:
            rejected.append({"index": index, "reasons": reasons})
        else:
            accepted.append(item)
    return document, accepted, rejected


def available_methods() -> Dict[str, int]:
    candidates = {
        "TSAI": "CALIB_HAND_EYE_TSAI",
        "PARK": "CALIB_HAND_EYE_PARK",
        "HORAUD": "CALIB_HAND_EYE_HORAUD",
        "ANDREFF": "CALIB_HAND_EYE_ANDREFF",
        "DANIILIDIS": "CALIB_HAND_EYE_DANIILIDIS",
    }
    return {
        name: int(getattr(cv2, attribute))
        for name, attribute in candidates.items()
        if hasattr(cv2, attribute)
    }


def calibrate_method(samples: Sequence[Dict[str, Any]], method: int) -> np.ndarray:
    rotations_gripper_to_base = [item["T_world_rigid"][:3, :3] for item in samples]
    translations_gripper_to_base = [
        item["T_world_rigid"][:3, 3].reshape(3, 1) for item in samples
    ]
    rotations_target_to_camera = [item["T_camera_board"][:3, :3] for item in samples]
    translations_target_to_camera = [
        item["T_camera_board"][:3, 3].reshape(3, 1) for item in samples
    ]
    rotation_camera_to_gripper, translation_camera_to_gripper = cv2.calibrateHandEye(
        rotations_gripper_to_base,
        translations_gripper_to_base,
        rotations_target_to_camera,
        translations_target_to_camera,
        method=method,
    )
    result = make_transform(
        project_to_rotation(rotation_camera_to_gripper),
        translation_camera_to_gripper,
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("method returned non-finite values")
    return result


def board_consistency(
    samples: Sequence[Dict[str, Any]], transform_rigid_camera: np.ndarray
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    world_board = np.asarray(
        [
            item["T_world_rigid"]
            @ transform_rigid_camera
            @ item["T_camera_board"]
            for item in samples
        ]
    )
    translations = world_board[:, :3, 3]
    center_translation = np.median(translations, axis=0)
    center_rotation = average_rotation(list(world_board[:, :3, :3]))
    translation_errors_mm = 1000.0 * np.linalg.norm(
        translations - center_translation, axis=1
    )
    rotation_errors_deg = np.asarray(
        [
            rotation_angle_deg(center_rotation.T @ transform[:3, :3])
            for transform in world_board
        ],
        dtype=np.float64,
    )
    # One degree is weighted like two millimetres only for ranking algorithms;
    # both physical residuals are also reported separately.
    score = float(
        np.median(translation_errors_mm) + 2.0 * np.median(rotation_errors_deg)
    )
    statistics = {
        "score": score,
        "translation_median_mm": float(np.median(translation_errors_mm)),
        "translation_rms_mm": float(np.sqrt(np.mean(translation_errors_mm**2))),
        "translation_max_mm": float(np.max(translation_errors_mm)),
        "rotation_median_deg": float(np.median(rotation_errors_deg)),
        "rotation_rms_deg": float(np.sqrt(np.mean(rotation_errors_deg**2))),
        "rotation_max_deg": float(np.max(rotation_errors_deg)),
    }
    center_transform = make_transform(center_rotation, center_translation)
    return statistics, translation_errors_mm, rotation_errors_deg, center_transform


def solve_all_methods(
    samples: Sequence[Dict[str, Any]], method_filter: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    methods = available_methods()
    if method_filter:
        if method_filter not in methods:
            raise ValueError(f"method {method_filter} is unavailable in this OpenCV build")
        methods = {method_filter: methods[method_filter]}
    for name, method in methods.items():
        try:
            transform = calibrate_method(samples, method)
            stats, translation_errors, rotation_errors, world_board = board_consistency(
                samples, transform
            )
            results[name] = {
                "T_rigid_camera": transform,
                "T_world_board": world_board,
                "statistics": stats,
                "translation_errors_mm": translation_errors,
                "rotation_errors_deg": rotation_errors,
            }
        except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
            print(f"警告：{name} 求解失败：{exc}")
    if not results:
        raise RuntimeError("all hand-eye methods failed")
    return results


def best_method(results: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    name = min(results, key=lambda item: results[item]["statistics"]["score"])
    return name, results[name]


def robust_threshold(values: np.ndarray, floor: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(float(floor), median + 3.5 * 1.4826 * mad)


def reject_geometric_outliers(
    samples: Sequence[Dict[str, Any]], result: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    translation_errors = np.asarray(result["translation_errors_mm"])
    rotation_errors = np.asarray(result["rotation_errors_deg"])
    translation_threshold = robust_threshold(translation_errors, 5.0)
    rotation_threshold = robust_threshold(rotation_errors, 1.0)
    keep_mask = (translation_errors <= translation_threshold) & (
        rotation_errors <= rotation_threshold
    )
    kept = [sample for sample, keep in zip(samples, keep_mask) if bool(keep)]
    rejected = [
        {
            "index": int(sample["index"]),
            "reasons": ["geometric_consistency"],
            "translation_error_mm": float(translation_error),
            "rotation_error_deg": float(rotation_error),
        }
        for sample, keep, translation_error, rotation_error in zip(
            samples, keep_mask, translation_errors, rotation_errors
        )
        if not bool(keep)
    ]
    thresholds = {
        "translation_mm": float(translation_threshold),
        "rotation_deg": float(rotation_threshold),
    }
    return kept, rejected, thresholds


def motion_diversity(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    translations: List[float] = []
    rotation_angles: List[float] = []
    rotation_axes: List[np.ndarray] = []
    for first in range(len(samples)):
        for second in range(first + 1, len(samples)):
            relative = invert_transform(samples[first]["T_world_rigid"]) @ samples[second][
                "T_world_rigid"
            ]
            translations.append(float(np.linalg.norm(relative[:3, 3])))
            rvec, _ = cv2.Rodrigues(relative[:3, :3])
            rvec = rvec.reshape(3)
            angle = float(np.linalg.norm(rvec))
            rotation_angles.append(math.degrees(angle))
            if angle > math.radians(3.0):
                rotation_axes.append(rvec / angle)

    singular_values = [0.0, 0.0, 0.0]
    axis_ratio = 0.0
    if len(rotation_axes) >= 3:
        _, values, _ = np.linalg.svd(np.asarray(rotation_axes), full_matrices=False)
        singular_values = [float(value) for value in values]
        if values[0] > 1.0e-12:
            axis_ratio = float(values[-1] / values[0])

    warnings: List[str] = []
    maximum_rotation = max(rotation_angles, default=0.0)
    maximum_translation = max(translations, default=0.0)
    if maximum_rotation < 20.0:
        warnings.append("rotation range is below 20 deg; rotate the bracket more")
    if axis_ratio < 0.08:
        warnings.append("rotation axes lack diversity; rotate about at least two axes")
    if maximum_translation < 0.10:
        warnings.append("translation baseline is below 0.10 m")
    return {
        "maximum_pair_translation_m": float(maximum_translation),
        "maximum_pair_rotation_deg": float(maximum_rotation),
        "rotation_axis_singular_values": singular_values,
        "rotation_axis_min_max_ratio": axis_ratio,
        "warnings": warnings,
    }


def matrix_list(transform: np.ndarray) -> List[List[float]]:
    return [[float(value) for value in row] for row in np.asarray(transform)]


def pose_document(transform: np.ndarray) -> Dict[str, Any]:
    quaternion = rotation_to_quaternion_xyzw(transform[:3, :3])
    return {
        "matrix": matrix_list(transform),
        "translation_m": [float(value) for value in transform[:3, 3]],
        "quaternion_xyzw": [float(value) for value in quaternion],
    }


def method_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    summary = dict(result["statistics"])
    summary["T_rigid_camera"] = matrix_list(result["T_rigid_camera"])
    return summary


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute rigid-body-to-camera hand-eye calibration"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "handeye_samples.json",
        help="sample JSON from charuco_mocap_collect.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "handeye_result.yaml",
        help="output YAML path",
    )
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--max-reprojection-error-px", type=float, default=1.0)
    parser.add_argument("--max-sync-error-ms", type=float, default=35.0)
    parser.add_argument(
        "--method",
        choices=["AUTO", "TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"],
        default="AUTO",
        help="AUTO compares every method and selects the lowest consistency residual",
    )
    parser.add_argument(
        "--no-outlier-rejection",
        action="store_true",
        help="do not reject samples inconsistent with a stationary board",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(argv)
    input_path = arguments.input.expanduser().resolve()
    output_path = arguments.output.expanduser().resolve()
    document, samples, rejected = load_samples(
        input_path,
        float(arguments.max_reprojection_error_px),
        float(arguments.max_sync_error_ms),
    )
    if len(samples) < int(arguments.min_samples):
        raise SystemExit(
            f"有效样本只有 {len(samples)} 组，至少需要 {arguments.min_samples} 组；"
            "建议实际采集 20--30 组"
        )

    method_filter = None if arguments.method == "AUTO" else str(arguments.method)
    initial_results = solve_all_methods(samples, method_filter)
    initial_name, initial_best = best_method(initial_results)
    outlier_thresholds: Optional[Dict[str, float]] = None

    final_samples = list(samples)
    geometric_rejected: List[Dict[str, Any]] = []
    if not arguments.no_outlier_rejection:
        candidate_samples, geometric_rejected, outlier_thresholds = reject_geometric_outliers(
            samples, initial_best
        )
        if len(candidate_samples) >= int(arguments.min_samples) and len(candidate_samples) < len(samples):
            final_samples = candidate_samples
        else:
            geometric_rejected = []

    final_results = solve_all_methods(final_samples, method_filter)
    selected_name, selected = best_method(final_results)
    transform_rigid_camera = selected["T_rigid_camera"]
    transform_camera_rigid = invert_transform(transform_rigid_camera)
    diversity = motion_diversity(final_samples)

    result_document: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "transform_convention": "p_A = T_A_B @ p_B",
        "frames": {
            "world": "mocap_world",
            "rigid": f"mocap rigid {document.get('mocap', {}).get('rigid_id', 4)}",
            "camera": "RealSense color optical frame",
            "board": "ChArUco board",
        },
        "selected_method": selected_name,
        "T_rigid_camera": pose_document(transform_rigid_camera),
        "T_camera_rigid": pose_document(transform_camera_rigid),
        "estimated_T_world_board": pose_document(selected["T_world_board"]),
        "residuals": dict(selected["statistics"]),
        "sample_count": {
            "recorded": len(document.get("samples", [])),
            "used": len(final_samples),
            "rejected": len(rejected) + len(geometric_rejected),
        },
        "used_sample_indices": [int(item["index"]) for item in final_samples],
        "rejected_samples": rejected + geometric_rejected,
        "outlier_thresholds": outlier_thresholds,
        "motion_diversity": diversity,
        "method_comparison": {
            name: method_summary(result) for name, result in final_results.items()
        },
        "input_configuration": {
            "board": document.get("board", {}),
            "camera": document.get("camera", {}),
            "mocap": document.get("mocap", {}),
            "max_reprojection_error_px": float(arguments.max_reprojection_error_px),
            "max_sync_error_ms": float(arguments.max_sync_error_ms),
        },
        "notes": [
            "T_rigid_camera maps a point from camera coordinates to rigid coordinates.",
            "The result is valid only if the ChArUco board stayed fixed during collection.",
            "The mocap rigid-body definition itself is part of this calibration.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            result_document,
            stream,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )

    stats = selected["statistics"]
    translation = transform_rigid_camera[:3, 3]
    quaternion = rotation_to_quaternion_xyzw(transform_rigid_camera[:3, :3])
    print(f"输入样本: {len(document.get('samples', []))}")
    print(f"有效/最终使用: {len(samples)}/{len(final_samples)}")
    print(f"初始/最终方法: {initial_name}/{selected_name}")
    print("\nT_rigid_camera =")
    print(np.array2string(transform_rigid_camera, precision=9, suppress_small=True))
    print(
        "translation [m]: "
        + " ".join(f"{value:.9f}" for value in translation)
    )
    print(
        "quaternion [x y z w]: "
        + " ".join(f"{value:.9f}" for value in quaternion)
    )
    print(
        f"固定板一致性: translation median/rms="
        f"{stats['translation_median_mm']:.3f}/{stats['translation_rms_mm']:.3f} mm, "
        f"rotation median/rms="
        f"{stats['rotation_median_deg']:.3f}/{stats['rotation_rms_deg']:.3f} deg"
    )
    for warning in diversity["warnings"]:
        print(f"警告：{warning}")
    print(f"\n结果已写入: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())