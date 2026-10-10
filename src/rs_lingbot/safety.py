"""Position trajectory limiter. No MIT gains, CAN or model imports."""
import math
import numpy as np
from .contract import vector


class MotionFault(RuntimeError):
    pass


class TrajectoryLimiter:
    def __init__(self, cfg, state):
        self.lower, self.upper = np.array(cfg["lower"]), np.array(cfg["upper"])
        self.vmax, self.amax = np.array(cfg["max_velocity"]), np.array(cfg["max_acceleration"])
        self.tracking = np.array(cfg["max_tracking_error"])
        self.q = vector(state, 7, "initial state").copy()
        self.v = np.zeros(7)
        self.validate_target(self.q)

    def validate_target(self, target):
        q = vector(target, 7, "target")
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise MotionFault("target/state outside configured joint limits")
        return q

    def step(self, target, feedback, dt):
        target = None if target is None else self.validate_target(target)
        feedback = self.validate_target(feedback)
        if not math.isfinite(dt) or not 0 < dt <= 0.1:
            raise MotionFault("invalid control interval")
        if np.any(np.abs(self.q - feedback) > self.tracking):
            raise MotionFault("tracking error exceeds limit")
        # Reserve a full integration interval for braking at position limits.
        brake_hi = np.maximum(np.sqrt(2 * self.amax * (self.upper - self.q)) - self.amax * dt, 0)
        brake_lo = -np.maximum(np.sqrt(2 * self.amax * (self.q - self.lower)) - self.amax * dt, 0)
        low = np.maximum.reduce([-self.vmax, self.v - self.amax * dt, brake_lo,
                                 (self.lower - self.q) / dt,
                                 (feedback - self.tracking - self.q) / dt])
        high = np.minimum.reduce([self.vmax, self.v + self.amax * dt, brake_hi,
                                  (self.upper - self.q) / dt,
                                  (feedback + self.tracking - self.q) / dt])
        if np.any(low > high + 1e-10):
            raise MotionFault("no feasible bounded trajectory; stop and inspect tracking")
        error = np.zeros(7) if target is None else target - self.q
        stopping_speed = np.maximum(np.sqrt(2 * self.amax * np.abs(error)) - self.amax * dt, 0)
        desired = np.sign(error) * np.minimum(np.abs(error) / dt, stopping_speed)
        velocity = np.clip(desired, low, high)
        q = self.q + velocity * dt
        self.q, self.v = q, velocity
        return q.copy(), velocity.copy()


def validate_chunk(actions, cfg):
    contract = cfg["contract_object"]
    a = np.asarray(actions, dtype=np.float64)
    if (a.ndim != 2 or a.shape[1] != contract.dim or
            not 1 <= len(a) <= cfg["runtime"]["max_chunk_steps"] or not np.all(np.isfinite(a))):
        raise MotionFault("invalid action chunk shape, length or values")
    for i, side in enumerate(contract.arms):
        arm = cfg["arms"][side]
        part = a[:, i * 7:(i + 1) * 7]
        if np.any(part < arm["lower"]) or np.any(part > arm["upper"]):
            raise MotionFault(f"{side} policy target outside joint limits")
    return a
