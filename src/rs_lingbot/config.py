from pathlib import Path
import math
import yaml
import numpy as np
from .contract import Contract, vector


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg["contract_object"] = Contract.from_dict(cfg["contract"])
    c = cfg["contract_object"]
    if set(cfg["arms"]) != set(c.arms):
        raise ValueError("arm configurations do not match contract")
    for side, arm in cfg["arms"].items():
        for key in ("lower", "upper", "max_velocity", "max_acceleration", "max_tracking_error"):
            arm[key] = vector(arm[key], 7, f"{side}.{key}")
        if np.any(arm["lower"] >= arm["upper"]):
            raise ValueError("invalid joint limits")
        for key in ("max_velocity", "max_acceleration", "max_tracking_error"):
            if np.any(arm[key] <= 0):
                raise ValueError(f"{key} must be positive")
        if arm["lower"][6] != 0 or arm["upper"][6] != 1:
            raise ValueError("gripper limits must be [0, 1]")
        if arm["gripper_open_deg"] <= arm["gripper_closed_deg"]:
            raise ValueError("gripper endpoints must be ordered")
        for key in ("gripper_open_deg", "gripper_closed_deg"):
            if not math.isfinite(arm[key]):
                raise ValueError("gripper endpoints must be finite")
    for key in ("control_hz", "max_tick_s", "max_observation_age_s", "max_camera_skew_s",
                "max_response_age_s", "policy_timeout_s", "policy_wait_timeout_s"):
        value = cfg["runtime"][key]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"runtime.{key} must be positive")
    runtime = cfg["runtime"]
    if runtime["max_tick_s"] < 1 / runtime["control_hz"]:
        raise ValueError("max_tick_s is smaller than the control period")
    for key in ("execute_steps", "max_chunk_steps"):
        if type(runtime[key]) is not int or not 1 <= runtime[key] <= 1000:
            raise ValueError(f"invalid {key}")
    if runtime["execute_steps"] > runtime["max_chunk_steps"]:
        raise ValueError("execute_steps exceeds max_chunk_steps")
    for key in ("stop_file", "log_dir"):
        cfg["runtime"][key] = str((path.parent / runtime[key]).resolve())
    if set(cfg["cameras"]) != set(c.cameras):
        raise ValueError("camera configuration does not match contract")
    return cfg
