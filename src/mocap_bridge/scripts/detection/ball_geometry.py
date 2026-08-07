"""Recover a known-radius ball center from its image silhouette.

Stereo depth on a smooth, texture-poor ball can have a repeatable range bias.
The apparent silhouette of a sphere, however, also determines its 3-D center
when the physical radius and camera intrinsics are known.  This module uses
the segmentation contour's tangent rays, so it does not depend on the depth
map and remains usable while the ball is moving.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SilhouetteSphereResult:
    """Known-radius sphere estimate and contour quality diagnostics."""

    center: np.ndarray
    contour_rmse_m: float
    equivalent_radius_px: float
    mask_area_px: float
    circularity: float
    axis_ratio: float
    boundary_point_count: int


def _intrinsic_values(intrinsics):
    if intrinsics is None:
        raise ValueError("camera intrinsics are required")
    cx = getattr(intrinsics, "ppx", getattr(intrinsics, "cx", None))
    cy = getattr(intrinsics, "ppy", getattr(intrinsics, "cy", None))
    if cx is None or cy is None:
        raise ValueError("intrinsics must provide ppx/ppy or cx/cy")
    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0.0 or fy <= 0.0:
        raise ValueError("invalid camera intrinsics")
    return fx, fy, float(cx), float(cy)


def largest_component_mask(mask):
    """Return a filled mask containing only the largest external component."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("ball mask must be two-dimensional")
    binary = mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.zeros_like(mask, dtype=bool)
    contour = max(contours, key=cv2.contourArea)
    cleaned = np.zeros_like(binary)
    cv2.drawContours(cleaned, [contour], -1, 1, thickness=cv2.FILLED)
    return cleaned.astype(bool)


def make_inner_mask(mask, erosion_fraction=0.06):
    """Remove a scale-aware boundary band before sampling stereo depth."""
    mask = largest_component_mask(mask)
    area = int(np.count_nonzero(mask))
    if area == 0:
        return mask
    equivalent_radius_px = np.sqrt(area / np.pi)
    erosion_px = max(1, int(round(equivalent_radius_px * erosion_fraction)))
    kernel_size = 2 * erosion_px + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    # A very small detection should remain usable instead of being erased.
    if np.count_nonzero(eroded) < max(16, int(0.25 * area)):
        return mask
    return eroded.astype(bool)


def _tangent_residuals(center, unit_rays, radius):
    projections = unit_rays @ center
    distance_sq = np.dot(center, center) - projections * projections
    perpendicular_distance = np.sqrt(np.maximum(distance_sq, 0.0))
    return perpendicular_distance - radius


def _fit_robust_ellipse(boundary, max_iterations=3):
    """Fit an ellipse after rejecting localized mask-edge protrusions."""
    points = np.asarray(boundary, dtype=np.float32).reshape(-1, 2)
    if len(points) < 5:
        raise ValueError("at least five contour points are required")

    ellipse = cv2.fitEllipse(points.reshape(-1, 1, 2))
    for _ in range(max(0, int(max_iterations))):
        (center_u, center_v), axes, angle_deg = ellipse
        semi_axis_x = max(0.5 * float(axes[0]), 1e-6)
        semi_axis_y = max(0.5 * float(axes[1]), 1e-6)
        angle = np.radians(float(angle_deg))
        centered = points - np.array([center_u, center_v])
        local_x = (
            np.cos(angle) * centered[:, 0]
            + np.sin(angle) * centered[:, 1]
        )
        local_y = (
            -np.sin(angle) * centered[:, 0]
            + np.cos(angle) * centered[:, 1]
        )
        normalized_radius = np.sqrt(
            (local_x / semi_axis_x) ** 2
            + (local_y / semi_axis_y) ** 2
        )
        deviations = np.abs(normalized_radius - 1.0)
        median = float(np.median(deviations))
        mad = float(np.median(np.abs(deviations - median)))
        inlier_limit = max(0.025, median + 2.5 * 1.4826 * mad)
        inliers = deviations <= inlier_limit
        if np.count_nonzero(inliers) < max(5, len(points) // 2):
            break
        ellipse = cv2.fitEllipse(points[inliers].reshape(-1, 1, 2))
    return ellipse


def estimate_fixed_radius_sphere_from_mask(
    mask,
    intrinsics,
    radius,
    *,
    min_area_px=80,
    max_boundary_points=360,
    max_axis_ratio=1.55,
    min_circularity=0.45,
    contour_edge_offset_px=0.4,
    boundary_model="ellipse",
):
    """Estimate a sphere center from its segmentation silhouette.

    Every silhouette boundary pixel defines a camera ray tangent to the
    sphere.  The center-to-ray distance therefore equals ``radius``.  By
    default a sub-pixel ellipse is fitted to the binary contour before the
    tangent solve.  A projected sphere is exactly a conic, so this suppresses
    mask stair-stepping without introducing temporal filtering or delay.
    ``boundary_model="raw"`` retains the original direct-contour behavior.
    """
    mask = largest_component_mask(mask)
    radius = float(radius)
    if radius <= 0.0 or not np.isfinite(radius):
        raise ValueError("sphere radius must be finite and positive")

    binary = mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("ball mask is empty")
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < float(min_area_px):
        raise ValueError(
            f"ball silhouette is too small: {area:.1f} < {min_area_px} px"
        )

    boundary_model = str(boundary_model).strip().lower()
    if boundary_model not in {"ellipse", "raw"}:
        raise ValueError("boundary_model must be 'ellipse' or 'raw'")

    height, width = mask.shape
    raw_boundary = contour.reshape(-1, 2).astype(np.float64)
    if (
        np.any(raw_boundary[:, 0] <= 0)
        or np.any(raw_boundary[:, 0] >= width - 1)
        or np.any(raw_boundary[:, 1] <= 0)
        or np.any(raw_boundary[:, 1] >= height - 1)
    ):
        raise ValueError("ball silhouette is truncated by the image boundary")

    perimeter = float(cv2.arcLength(contour, closed=True))
    circularity = (
        4.0 * np.pi * area / (perimeter * perimeter)
        if perimeter > 0.0
        else 0.0
    )
    if circularity < float(min_circularity):
        raise ValueError(
            f"ball silhouette circularity is too low: {circularity:.3f}"
        )

    fitted_ellipse = None
    if len(raw_boundary) >= 5:
        fitted_ellipse = _fit_robust_ellipse(raw_boundary)
        _, ellipse_axes, _ = fitted_ellipse
        minor_axis, major_axis = sorted(map(float, ellipse_axes))
        axis_ratio = major_axis / max(minor_axis, 1e-9)
    else:
        axis_ratio = 1.0
    if axis_ratio > float(max_axis_ratio):
        raise ValueError(
            f"ball silhouette axis ratio is too large: {axis_ratio:.3f}"
        )

    moments = cv2.moments(contour)
    if moments["m00"] <= 0.0:
        raise ValueError("cannot calculate ball silhouette centroid")
    moment_center_u = moments["m10"] / moments["m00"]
    moment_center_v = moments["m01"] / moments["m00"]

    max_boundary_points = max(16, int(max_boundary_points))
    quality_boundary = raw_boundary
    if len(quality_boundary) > max_boundary_points:
        indices = np.linspace(
            0,
            len(quality_boundary) - 1,
            max_boundary_points,
            dtype=np.int64,
        )
        quality_boundary = quality_boundary[indices]

    # OpenCV contours run through foreground pixel centers, while the actual
    # binary silhouette edge lies between foreground and background pixels.
    # Move the samples by a sub-pixel amount toward the outside.  Without this
    # correction, a 20 px radius ball is made about 2% too small and its range
    # is consequently overestimated by the same order.
    contour_edge_offset_px = float(contour_edge_offset_px)
    quality_boundary = quality_boundary.copy()
    if contour_edge_offset_px:
        outward = quality_boundary - np.array(
            [moment_center_u, moment_center_v]
        )
        outward_norm = np.linalg.norm(outward, axis=1)
        valid_outward = outward_norm > 1e-9
        quality_boundary[valid_outward] += (
            contour_edge_offset_px
            * outward[valid_outward]
            / outward_norm[valid_outward, None]
        )

    if boundary_model == "ellipse":
        if fitted_ellipse is None:
            raise ValueError("at least five contour points are required")
        (center_u, center_v), ellipse_axes, ellipse_angle_deg = fitted_ellipse
        semi_axis_x = 0.5 * float(ellipse_axes[0]) + contour_edge_offset_px
        semi_axis_y = 0.5 * float(ellipse_axes[1]) + contour_edge_offset_px
        sample_count = min(
            max_boundary_points,
            max(64, len(quality_boundary)),
        )
        angles = np.linspace(
            0.0,
            2.0 * np.pi,
            sample_count,
            endpoint=False,
        )
        local_x = semi_axis_x * np.cos(angles)
        local_y = semi_axis_y * np.sin(angles)
        ellipse_angle = np.radians(float(ellipse_angle_deg))
        cos_angle = np.cos(ellipse_angle)
        sin_angle = np.sin(ellipse_angle)
        boundary = np.column_stack(
            (
                center_u + cos_angle * local_x - sin_angle * local_y,
                center_v + sin_angle * local_x + cos_angle * local_y,
            )
        )
    else:
        center_u = moment_center_u
        center_v = moment_center_v
        boundary = quality_boundary

    fx, fy, cx, cy = _intrinsic_values(intrinsics)

    def unit_rays_from_pixels(pixels):
        normalized = np.column_stack(
            (
                (pixels[:, 0] - cx) / fx,
                (pixels[:, 1] - cy) / fy,
                np.ones(len(pixels)),
            )
        )
        return normalized / np.linalg.norm(normalized, axis=1)[:, None]

    unit_rays = unit_rays_from_pixels(boundary)

    center_ray = np.array(
        [(center_u - cx) / fx, (center_v - cy) / fy, 1.0],
        dtype=np.float64,
    )
    center_ray /= np.linalg.norm(center_ray)

    # Mapping image area into the normalized pinhole plane gives a stable
    # angular-radius seed.  The nonlinear tangent solve below removes the
    # small-angle and off-axis approximation from the final result.
    normalized_radius = np.sqrt(area / (np.pi * fx * fy))
    if normalized_radius <= 1e-9:
        raise ValueError("invalid normalized silhouette radius")
    initial_range = radius * np.sqrt(1.0 + normalized_radius ** -2)
    initial_center = center_ray * initial_range

    solution = least_squares(
        _tangent_residuals,
        initial_center,
        args=(unit_rays, radius),
        bounds=(
            np.array([-np.inf, -np.inf, radius * 1.001]),
            np.array([np.inf, np.inf, 50.0]),
        ),
        loss="soft_l1",
        f_scale=max(0.001, 0.02 * radius),
        max_nfev=60,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    if not solution.success or not np.all(np.isfinite(solution.x)):
        raise ValueError(f"silhouette sphere optimizer failed: {solution.message}")

    quality_unit_rays = unit_rays_from_pixels(quality_boundary)
    residuals = _tangent_residuals(
        solution.x,
        quality_unit_rays,
        radius,
    )
    contour_rmse_m = float(np.sqrt(np.mean(residuals * residuals)))
    max_rmse_m = max(0.004, 0.08 * radius)
    if contour_rmse_m > max_rmse_m:
        raise ValueError(
            "silhouette sphere fit quality is insufficient: "
            f"RMSE={contour_rmse_m * 1000.0:.1f} mm"
        )

    equivalent_radius_px = float(np.sqrt(area / np.pi))
    return SilhouetteSphereResult(
        center=solution.x.copy(),
        contour_rmse_m=contour_rmse_m,
        equivalent_radius_px=equivalent_radius_px,
        mask_area_px=area,
        circularity=float(circularity),
        axis_ratio=float(axis_ratio),
        boundary_point_count=int(len(boundary)),
    )
