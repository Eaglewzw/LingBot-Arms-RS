"""A compact, physical-unit wire contract, independent of the 55D model space."""
from dataclasses import dataclass
import math
import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")
SCHEMA = "rs-lingbot/1"


def vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return result


@dataclass(frozen=True)
class Contract:
    arms: tuple[str, ...]
    cameras: tuple[str, ...]
    action_dt_s: float
    embodiment: str = "b601_rs"
    rs_finetuned: bool = False

    def __post_init__(self):
        if self.arms not in (("left",), ("right",), ("left", "right")):
            raise ValueError("arms must be [left], [right], or [left, right] in that order")
        expected = {"camera_top", *(f"camera_wrist_{a}" for a in self.arms)}
        if set(self.cameras) != expected or len(self.cameras) != len(expected):
            raise ValueError(f"camera contract must contain exactly {sorted(expected)}")
        if not math.isfinite(self.action_dt_s) or not 0.005 <= self.action_dt_s <= 0.2:
            raise ValueError("action_dt_s must be in [0.005, 0.2]")
        if self.embodiment != "b601_rs" or type(self.rs_finetuned) is not bool:
            raise ValueError("invalid RS embodiment or finetuning declaration")

    @property
    def dim(self):
        return 7 * len(self.arms)

    def to_dict(self):
        return dict(schema=SCHEMA, arms=list(self.arms), cameras=list(self.cameras),
                    action_dt_s=self.action_dt_s, embodiment=self.embodiment,
                    rs_finetuned=self.rs_finetuned, joint_names=list(JOINTS),
                    joint_units="rad", gripper_units="open_fraction", action_type="absolute")

    @classmethod
    def from_dict(cls, data):
        for key, value in dict(schema=SCHEMA, joint_names=list(JOINTS), joint_units="rad",
                               gripper_units="open_fraction", action_type="absolute").items():
            if data.get(key) != value:
                raise ValueError(f"unsupported contract field {key}: {data.get(key)!r}")
        return cls(tuple(data["arms"]), tuple(data["cameras"]), float(data["action_dt_s"]),
                   data["embodiment"], data["rs_finetuned"])

    def compatible(self, other):
        a, b = self.to_dict(), other.to_dict()
        a.pop("rs_finetuned")
        b.pop("rs_finetuned")
        if a != b:
            raise ValueError("policy/robot contract mismatch (arms, camera layout, units or action period)")


def state_from_degrees(observation, closed_deg, open_deg):
    if not math.isfinite(open_deg - closed_deg) or open_deg <= closed_deg:
        raise ValueError("gripper endpoints must be measured and ordered")
    q = np.deg2rad([observation[f"{j}.pos"] for j in JOINTS])
    g = (float(observation["gripper.pos"]) - closed_deg) / (open_deg - closed_deg)
    return vector([*q, g], 7, "hardware observation")


def action_to_degrees(action, closed_deg, open_deg):
    a = vector(action, 7, "hardware action")
    if not 0 <= a[-1] <= 1:
        raise ValueError("gripper target outside [0, 1]")
    return {**{f"{j}.pos": float(np.rad2deg(q)) for j, q in zip(JOINTS, a[:6])},
            "gripper.pos": float(closed_deg + a[6] * (open_deg - closed_deg))}
