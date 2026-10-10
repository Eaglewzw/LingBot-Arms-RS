"""Lazy adapters to the user's VR implementation; no duplicated motor protocol."""
from pathlib import Path
import numpy as np
from .contract import JOINTS, action_to_degrees, state_from_degrees


class MockArm:
    def __init__(self, cfg):
        self.state = np.array(cfg.get("mock_initial", [0, .8, .8, 0, 0, 0, .7]), dtype=float)
        self.commands = 0

    def connect(self):
        pass

    def read(self):
        return self.state.copy()

    def send(self, q, velocity):
        self.state = q.copy()
        self.commands += 1

    def close(self):
        pass


class RSArm:
    def __init__(self, cfg, *, execute, report_path):
        from lerobot_teleoperator_rebot_vr.hardware.rs import RSFollower, RSFollowerConfig
        from lerobot_teleoperator_rebot_vr.control.mit import MITCommandDispatcher
        from lerobot_teleoperator_rebot_vr.runtime.shutdown import ManagedFollower, ShutdownPolicy
        from lerobot_teleoperator_rebot_vr.hardware.profiles import RS

        self.cfg, self.execute = cfg, execute
        # Never allow configuration to widen the upstream mechanical bounds.
        lower, upper = np.array(cfg["lower"]), np.array(cfg["upper"])
        if np.any(lower[:6] < RS.lower_rad) or np.any(upper[:6] > RS.upper_rad):
            raise ValueError("configured limits exceed RS hardware profile")
        if execute and not cfg.get("gripper_endpoints_verified", False):
            raise ValueError("measure gripper endpoints and mark gripper_endpoints_verified first")
        span = cfg["gripper_open_deg"] - cfg["gripper_closed_deg"]
        limits = {name: (float(np.rad2deg(lo)), float(np.rad2deg(hi)))
                  for name, lo, hi in zip(JOINTS, lower[:6], upper[:6])}
        limits["gripper"] = (cfg["gripper_closed_deg"], cfg["gripper_open_deg"])
        relative = {name: float(np.rad2deg(error) / .8)
                    for name, error in zip(JOINTS, cfg["max_tracking_error"][:6])}
        relative["gripper"] = float(cfg["max_tracking_error"][6] * span / .8)
        settings = RSFollowerConfig(
            port=cfg["port"], id=cfg["robot_id"],
            calibration_dir=Path(cfg["calibration_dir"]).expanduser() if cfg.get("calibration_dir") else None,
            zero_confirmed=False, disable_torque_on_disconnect=True,
            joint_limits=limits, max_relative_target=relative,
            gripper_mit_kp=cfg["mit"]["gripper_kp"], gripper_mit_kd=cfg["mit"]["gripper_kd"])

        class StrictFollower(RSFollower):
            def _check_startup_positions(inner, observation):
                super()._check_startup_positions(observation)
                q = state_from_degrees(observation, cfg["gripper_closed_deg"], cfg["gripper_open_deg"])
                if np.any(q < lower) or np.any(q > upper):
                    raise ValueError("place RS in the verified starting workspace before enabling")

        self.raw = StrictFollower(settings)
        self.managed = ManagedFollower(self.raw, policy=ShutdownPolicy(), report_path=Path(report_path))
        # Construct gravity model and validate gains BEFORE opening CAN or enabling.
        self.dispatcher = MITCommandDispatcher(
            self.managed, kp=np.array(cfg["mit"]["kp"]), kd=np.array(cfg["mit"]["kd"]),
            torque_limit_nm=np.array(cfg["mit"]["torque_limit_nm"]),
            arm_velocity_limit_rad_s=np.array(cfg["max_velocity"][:6]),
            arm_acceleration_limit_rad_s2=np.array(cfg["max_acceleration"][:6]),
            position_lookahead_s=None, robot_model="b601_rs",
            gravity_scale=cfg["mit"]["gravity_scale"], gravity_ramp_s=1.5,
            gripper_velocity_limit_deg_s=float(cfg["max_velocity"][6] * span)) if execute else None

    def connect(self):
        if self.execute:
            self.managed.connect()
        else:
            self.raw.open()  # READ ONLY: no ensure_mode, enable, zeroing or disable.

    def probe(self):
        """Read startup state for model warmup without enabling any motor."""
        try:
            self.raw.open()
            obs = self.raw.get_observation()
            self.raw._check_startup_positions(obs)
            return state_from_degrees(obs, self.cfg["gripper_closed_deg"], self.cfg["gripper_open_deg"])
        finally:
            self.raw.close()

    def read(self):
        obs = self.dispatcher.get_observation() if self.execute else self.raw.get_observation()
        return state_from_degrees(obs, self.cfg["gripper_closed_deg"], self.cfg["gripper_open_deg"])

    def send(self, q, velocity):
        if not self.execute:
            return
        action = action_to_degrees(q, self.cfg["gripper_closed_deg"], self.cfg["gripper_open_deg"])
        self.dispatcher.set_arm_velocity(velocity[:6])
        actual = self.dispatcher.send_action(action, gripper_velocity_deg_s=float(
            velocity[6] * (self.cfg["gripper_open_deg"] - self.cfg["gripper_closed_deg"])))
        # A second controller changing our target invalidates the limiter history.
        if any(abs(actual[k] - v) > 1e-5 for k, v in action.items()):
            raise RuntimeError("MIT dispatcher clipped the command; stop to inspect limits/tracking")

    def close(self):
        if self.execute:
            # Uses the upstream bounded retry/report path, including partial connect.
            active = self.managed.needs_cleanup
            report = self.managed.disconnect()
            if active and (report.cleanup_errors or any(m.disable_sent < 1 for m in report.motors)):
                raise RuntimeError("RS shutdown reported cleanup/disable-send failures; inspect shutdown report")
        else:
            self.raw.close()
