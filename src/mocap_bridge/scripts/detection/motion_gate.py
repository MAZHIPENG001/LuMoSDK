"""Online motion-consistency gate for published ball-center measurements."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotionGateDecision:
    """Decision for one candidate ball-center measurement."""

    accepted: bool
    output_position: np.ndarray | None
    predicted: bool
    reason: str
    apparent_speed_mps: float
    innovation_m: float


class BallMotionGate:
    """Reject isolated position jumps while preserving accepted measurements.

    Measurements inside the motion envelope are returned unchanged.  A
    rejected measurement never updates the tracker.  During a short rejection
    interval, ``output_position`` contains a constant-velocity prediction so
    the normal center topic can remain continuous; after that interval it is
    ``None`` and callers should stop publishing until a valid measurement or a
    genuine detection gap resets the tracker.
    """

    def __init__(
        self,
        *,
        max_speed_mps=8.0,
        max_innovation_m=1.0,
        max_prediction_sec=0.25,
        reset_gap_sec=0.5,
        velocity_smoothing=0.6,
    ):
        self.max_speed_mps = float(max_speed_mps)
        self.max_innovation_m = float(max_innovation_m)
        self.max_prediction_sec = float(max_prediction_sec)
        self.reset_gap_sec = float(reset_gap_sec)
        self.velocity_smoothing = float(velocity_smoothing)
        if self.max_speed_mps <= 0.0:
            raise ValueError("max_speed_mps must be positive")
        if self.max_innovation_m <= 0.0:
            raise ValueError("max_innovation_m must be positive")
        if self.max_prediction_sec < 0.0:
            raise ValueError("max_prediction_sec must not be negative")
        if self.reset_gap_sec <= 0.0:
            raise ValueError("reset_gap_sec must be positive")
        if not 0.0 <= self.velocity_smoothing <= 1.0:
            raise ValueError("velocity_smoothing must be in [0, 1]")
        self.reset()

    def reset(self):
        self.position = None
        self.velocity = np.zeros(3, dtype=np.float64)
        self.accepted_time = None
        self.observation_time = None

    @staticmethod
    def _validate(position, timestamp_sec):
        position = np.asarray(position, dtype=np.float64).reshape(3)
        timestamp_sec = float(timestamp_sec)
        if not np.all(np.isfinite(position)):
            raise ValueError("ball position must be finite")
        if not np.isfinite(timestamp_sec):
            raise ValueError("ball timestamp must be finite")
        return position, timestamp_sec

    def _accept(self, measurement, timestamp_sec, reason):
        if self.position is not None:
            dt = timestamp_sec - self.accepted_time
            if dt > 1e-6:
                measured_velocity = (measurement - self.position) / dt
                alpha = self.velocity_smoothing
                self.velocity = (
                    (1.0 - alpha) * self.velocity
                    + alpha * measured_velocity
                )
        self.position = measurement.copy()
        self.accepted_time = timestamp_sec
        self.observation_time = timestamp_sec
        return MotionGateDecision(
            accepted=True,
            output_position=measurement.copy(),
            predicted=False,
            reason=reason,
            apparent_speed_mps=0.0,
            innovation_m=0.0,
        )

    def update(self, measurement, timestamp_sec):
        measurement, timestamp_sec = self._validate(
            measurement, timestamp_sec
        )
        if self.position is None:
            return self._accept(measurement, timestamp_sec, "initialized")

        observation_gap = timestamp_sec - self.observation_time
        if observation_gap <= 0.0:
            return MotionGateDecision(
                accepted=False,
                output_position=None,
                predicted=False,
                reason="non-monotonic-timestamp",
                apparent_speed_mps=np.inf,
                innovation_m=np.inf,
            )
        if observation_gap > self.reset_gap_sec:
            self.reset()
            return self._accept(measurement, timestamp_sec, "reinitialized")
        self.observation_time = timestamp_sec

        dt = timestamp_sec - self.accepted_time
        if dt <= 0.0:
            return MotionGateDecision(
                accepted=False,
                output_position=None,
                predicted=False,
                reason="non-monotonic-accepted-time",
                apparent_speed_mps=np.inf,
                innovation_m=np.inf,
            )

        prediction = self.position + self.velocity * dt
        apparent_speed_mps = float(
            np.linalg.norm(measurement - self.position) / dt
        )
        innovation_m = float(np.linalg.norm(measurement - prediction))
        excessive_speed = apparent_speed_mps > self.max_speed_mps
        excessive_innovation = innovation_m > self.max_innovation_m

        if not excessive_speed and not excessive_innovation:
            decision = self._accept(measurement, timestamp_sec, "accepted")
            return MotionGateDecision(
                accepted=True,
                output_position=decision.output_position,
                predicted=False,
                reason="accepted",
                apparent_speed_mps=apparent_speed_mps,
                innovation_m=innovation_m,
            )

        reason_parts = []
        if excessive_speed:
            reason_parts.append("speed")
        if excessive_innovation:
            reason_parts.append("innovation")
        can_predict = dt <= self.max_prediction_sec
        return MotionGateDecision(
            accepted=False,
            output_position=prediction.copy() if can_predict else None,
            predicted=can_predict,
            reason="+".join(reason_parts),
            apparent_speed_mps=apparent_speed_mps,
            innovation_m=innovation_m,
        )
