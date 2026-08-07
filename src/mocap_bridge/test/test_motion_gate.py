import sys
from pathlib import Path

import numpy as np


DETECTION_DIR = Path(__file__).resolve().parents[1] / "scripts" / "detection"
sys.path.insert(0, str(DETECTION_DIR))

from motion_gate import BallMotionGate  # noqa: E402


def test_normal_motion_is_returned_unchanged():
    gate = BallMotionGate()
    for index in range(10):
        measurement = np.array([0.1 * index, 0.0, 2.0])
        decision = gate.update(measurement, index * 0.1)
        assert decision.accepted
        assert not decision.predicted
        np.testing.assert_allclose(decision.output_position, measurement)


def test_isolated_extreme_is_replaced_by_prediction_without_state_update():
    gate = BallMotionGate()
    gate.update(np.array([0.0, 0.0, 2.0]), 0.0)
    gate.update(np.array([0.1, 0.0, 2.0]), 0.1)

    extreme = gate.update(np.array([0.7, -1.5, 2.0]), 0.2)
    assert not extreme.accepted
    assert extreme.predicted
    assert extreme.apparent_speed_mps > 8.0
    np.testing.assert_allclose(
        extreme.output_position, np.array([0.16, 0.0, 2.0])
    )

    recovered = gate.update(np.array([0.2, 0.0, 2.0]), 0.2)
    # The duplicate timestamp is deliberately rejected.
    assert not recovered.accepted
    recovered = gate.update(np.array([0.3, 0.0, 2.0]), 0.3)
    assert recovered.accepted
    np.testing.assert_allclose(recovered.output_position, [0.3, 0.0, 2.0])


def test_prediction_stops_after_short_rejection_window():
    gate = BallMotionGate(max_prediction_sec=0.2)
    gate.update(np.array([0.0, 0.0, 2.0]), 0.0)
    short = gate.update(np.array([2.0, 0.0, 2.0]), 0.1)
    long = gate.update(np.array([2.0, 0.0, 2.0]), 0.3)

    assert short.predicted
    assert short.output_position is not None
    assert not long.predicted
    assert long.output_position is None


def test_detection_gap_reinitializes_gate():
    gate = BallMotionGate(reset_gap_sec=0.5)
    gate.update(np.array([0.0, 0.0, 2.0]), 0.0)
    decision = gate.update(np.array([2.0, 1.0, 3.0]), 0.6)

    assert decision.accepted
    assert decision.reason == "reinitialized"
