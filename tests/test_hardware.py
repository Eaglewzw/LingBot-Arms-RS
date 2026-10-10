"""Fake the upstream driver boundaries; never import MotorBridge or touch CAN."""
from pathlib import Path
import sys
import types
import numpy as np
import pytest
from rs_lingbot.config import load_config
from rs_lingbot.hardware import RSArm


@pytest.fixture
def upstream(monkeypatch):
    calls = []
    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    class Raw:
        def __init__(self, config):
            self.config = config
        def open(self):
            calls.append("open")
        def close(self):
            calls.append("close")
        def _check_startup_positions(self, obs):
            pass
        def get_observation(self):
            calls.append("read")
            from rs_lingbot.contract import action_to_degrees
            return action_to_degrees([0, .8, .8, 0, 0, 0, .5], 0, 310.12)
    class Managed:
        def __init__(self, raw, **kwargs):
            self.raw = raw
        def connect(self):
            calls.append("enable")
        @property
        def needs_cleanup(self):
            return True
        def disconnect(self):
            calls.append("disable")
            return types.SimpleNamespace(cleanup_errors=[], motors=[types.SimpleNamespace(disable_sent=1)])
    class Dispatcher:
        def __init__(self, managed, **kwargs):
            calls.append("build_dispatcher")
            self.raw = managed.raw
            assert kwargs["position_lookahead_s"] is None
        def get_observation(self):
            return self.raw.get_observation()
        def set_arm_velocity(self, v):
            calls.append(("velocity", v.copy()))
        def send_action(self, action, **kwargs):
            calls.append(("command", action.copy()))
            return action
    base = "lerobot_teleoperator_rebot_vr"
    for name in (base, base + ".hardware", base + ".control", base + ".runtime"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    modules = {
        ".hardware.rs": dict(RSFollower=Raw, RSFollowerConfig=Config),
        ".runtime.shutdown": dict(ManagedFollower=Managed, ShutdownPolicy=lambda: None),
        ".control.mit": dict(MITCommandDispatcher=Dispatcher),
        ".hardware.profiles": dict(RS=types.SimpleNamespace(lower_rad=(-2.8, 0, 0, -1.57, -1.57, -3.14),
            upper_rad=(2.8, 3.14, 3.14, 1.57, 1.57, 3.14)))
    }
    for suffix, attrs in modules.items():
        module = types.ModuleType(base + suffix)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, base + suffix, module)
    return calls


def config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/single_right.yaml")["arms"]["right"]


def test_shadow_and_probe_never_send_enable_disable_or_motion(upstream, tmp_path):
    arm = RSArm(config(), execute=False, report_path=tmp_path / "shutdown.json")
    arm.probe()
    arm.connect()
    q = arm.read()
    arm.send(q, np.zeros(7))
    arm.close()
    assert upstream == ["open", "read", "close", "open", "read", "close"]


def test_live_uses_dispatcher_and_shutdown(upstream, tmp_path):
    cfg = config()
    cfg["gripper_endpoints_verified"] = True
    arm = RSArm(cfg, execute=True, report_path=tmp_path / "shutdown.json")
    assert upstream == ["build_dispatcher"]
    arm.probe()
    assert "enable" not in upstream
    arm.connect()
    q = arm.read()
    q[-1] = .6
    arm.send(q, np.zeros(7))
    arm.close()
    command = next(c[1] for c in upstream if isinstance(c, tuple) and c[0] == "command")
    assert command["gripper.pos"] == pytest.approx(186.072)
    assert upstream[-1] == "disable"


def test_unverified_gripper_blocks_live_before_driver_creation(upstream, tmp_path):
    with pytest.raises(ValueError, match="endpoints"):
        RSArm(config(), execute=True, report_path=tmp_path / "shutdown.json")
    assert upstream == []
