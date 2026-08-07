import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


DETECTION_DIR = Path(__file__).resolve().parents[1] / "scripts" / "detection"
sys.path.insert(0, str(DETECTION_DIR))

from ball_geometry import (  # noqa: E402
    estimate_fixed_radius_sphere_from_mask,
    largest_component_mask,
    make_inner_mask,
)


INTRINSICS = SimpleNamespace(
    fx=361.99130249,
    fy=361.99130249,
    ppx=478.683,
    ppy=299.224,
)
RADIUS_M = 0.115


def render_sphere_mask(center, height=600, width=960):
    rows, cols = np.indices((height, width))
    rays = np.stack(
        (
            (cols - INTRINSICS.ppx) / INTRINSICS.fx,
            (rows - INTRINSICS.ppy) / INTRINSICS.fy,
            np.ones((height, width)),
        ),
        axis=-1,
    )
    projection = rays @ center
    discriminant = (
        projection * projection
        - np.sum(rays * rays, axis=2)
        * (np.dot(center, center) - RADIUS_M * RADIUS_M)
    )
    return (discriminant >= 0.0) & (projection > 0.0)


def test_silhouette_recovers_known_radius_sphere_center():
    cases = (
        (np.array([-0.165, -0.225, 1.890]), 0.005),
        (np.array([0.0, 0.0, 1.0]), 0.005),
        # This is only about 13 px in radius, so raster quantization dominates.
        (np.array([0.5, -0.2, 3.0]), 0.030),
        (np.array([-0.4, 0.3, 1.5]), 0.005),
    )
    for expected, tolerance_m in cases:
        mask = render_sphere_mask(expected)
        result = estimate_fixed_radius_sphere_from_mask(
            mask, INTRINSICS, RADIUS_M
        )
        assert np.linalg.norm(result.center - expected) < tolerance_m


def test_mask_cleanup_removes_detached_pixels_and_erodes_only_depth_mask():
    mask = render_sphere_mask(np.array([-0.165, -0.225, 1.890]))
    mask[20:23, 20:23] = True
    cleaned = largest_component_mask(mask)
    inner = make_inner_mask(cleaned)

    assert not np.any(cleaned[20:23, 20:23])
    assert 0 < np.count_nonzero(inner) < np.count_nonzero(cleaned)


def test_truncated_silhouette_is_rejected():
    mask = np.zeros((100, 100), dtype=bool)
    mask[25:75, :30] = True
    try:
        estimate_fixed_radius_sphere_from_mask(mask, INTRINSICS, RADIUS_M)
    except ValueError as error:
        assert "truncated" in str(error)
    else:
        raise AssertionError("a truncated sphere silhouette must be rejected")
