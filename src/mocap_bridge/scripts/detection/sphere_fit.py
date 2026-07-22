"""Robust fixed-radius sphere fitting for RGB-D ball detections."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SphereFitResult:
    """Result of fitting a known-radius sphere to measured surface points."""

    center: np.ndarray
    rmse_m: float
    median_abs_residual_m: float
    inlier_fraction: float
    point_count: int
    inlier_count: int


def _sphere_residuals(center, points, radius):
    return np.linalg.norm(points - center.reshape(1, 3), axis=1) - radius


def _sphere_jacobian(center, points, radius):
    del radius
    delta = center.reshape(1, 3) - points
    distance = np.linalg.norm(delta, axis=1)
    distance = np.maximum(distance, 1e-12)
    return delta / distance[:, None]


def _initial_center_from_points(points, radius):
    """Estimate a camera-facing center for use as an optimizer seed."""
    ranges = np.linalg.norm(points, axis=1)
    front_limit = np.percentile(ranges, 15.0)
    front_points = points[ranges <= front_limit]
    surface_point = np.median(front_points, axis=0)
    surface_range = np.linalg.norm(surface_point)
    if not np.isfinite(surface_range) or surface_range <= 1e-9:
        raise ValueError("cannot initialize sphere center from surface points")
    return surface_point + radius * surface_point / surface_range


def fit_fixed_radius_sphere(
    points,
    radius,
    initial_center=None,
    *,
    min_points=80,
    robust_scale_m=0.005,
    max_nfev=40,
):
    """Fit a sphere center while keeping its physical radius fixed.

    The input points must be expressed in the camera optical frame in metres.
    A robust first pass limits the influence of mask leakage and depth flying
    pixels.  A trimmed second pass then refines the center on geometric
    inliers.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    radius = float(radius)
    min_points = int(min_points)

    if radius <= 0.0 or not np.isfinite(radius):
        raise ValueError("sphere radius must be finite and positive")
    if len(points) < min_points:
        raise ValueError(
            f"not enough depth points for sphere fit: {len(points)} < {min_points}"
        )

    if initial_center is None:
        initial_center = _initial_center_from_points(points, radius)
    initial_center = np.asarray(initial_center, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(initial_center)) or initial_center[2] <= 0.0:
        raise ValueError("invalid initial sphere center")

    # Remove only points that are grossly incompatible with the seed.  The
    # threshold remains deliberately wide because the centroid-ray seed is an
    # approximation and can have lateral error for a partial mask.
    seed_residual = np.abs(_sphere_residuals(initial_center, points, radius))
    seed_limit = max(0.050, 0.75 * radius)
    fit_points = points[seed_residual <= seed_limit]
    if len(fit_points) < min_points:
        fit_points = points

    # Keep the solution near the physically meaningful camera-facing branch.
    # With only a shallow visible cap, an unconstrained sphere fit can otherwise
    # converge to a geometrically unrelated center.
    center_limit = max(radius, 0.050)
    lower = initial_center - center_limit
    upper = initial_center + center_limit
    lower[2] = max(lower[2], 1e-6)

    first = least_squares(
        _sphere_residuals,
        initial_center,
        jac=_sphere_jacobian,
        args=(fit_points, radius),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=float(robust_scale_m),
        max_nfev=int(max_nfev),
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )
    if not first.success or not np.all(np.isfinite(first.x)):
        raise ValueError(f"sphere optimizer failed: {first.message}")

    first_abs = np.abs(_sphere_residuals(first.x, fit_points, radius))
    median = float(np.median(first_abs))
    mad = float(np.median(np.abs(first_abs - median)))
    robust_sigma = 1.4826 * mad
    inlier_limit = max(0.008, median + 3.5 * robust_sigma)
    inliers = fit_points[first_abs <= inlier_limit]

    solution = first
    if len(inliers) >= min_points:
        solution = least_squares(
            _sphere_residuals,
            first.x,
            jac=_sphere_jacobian,
            args=(inliers, radius),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=float(robust_scale_m),
            max_nfev=int(max_nfev),
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
        if not solution.success or not np.all(np.isfinite(solution.x)):
            solution = first

    all_residual = np.abs(_sphere_residuals(solution.x, points, radius))
    final_inliers = all_residual <= inlier_limit
    if np.count_nonzero(final_inliers) < min_points:
        raise ValueError("sphere fit has too few geometric inliers")

    inlier_residual = all_residual[final_inliers]
    return SphereFitResult(
        center=solution.x.copy(),
        rmse_m=float(np.sqrt(np.mean(inlier_residual ** 2))),
        median_abs_residual_m=float(np.median(inlier_residual)),
        inlier_fraction=float(np.mean(final_inliers)),
        point_count=int(len(points)),
        inlier_count=int(np.count_nonzero(final_inliers)),
    )
