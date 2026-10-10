import json
from pathlib import Path
import threading
import time
import numpy as np
import pytest
from rs_lingbot.config import load_config
from rs_lingbot.contract import Contract, state_from_degrees, action_to_degrees
from rs_lingbot.safety import TrajectoryLimiter, MotionFault, validate_chunk
from rs_lingbot.runtime import ActionTimeline, run
from rs_lingbot.policy import HTTPPolicy, encode_image, decode_image, InferenceWorker
from rs_lingbot.server import PolicyService, make_server
from rs_lingbot.cameras import Cameras

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config(ROOT / "configs/single_right.yaml")
    cfg["runtime"]["log_dir"] = str(tmp_path / "runs")
    cfg["runtime"]["stop_file"] = str(tmp_path / "STOP")
    return cfg


def test_hardware_unit_roundtrip():
    q = np.array([.2, .8, 1.1, -.3, .4, -.9, .6])
    obs = action_to_degrees(q, 0, 310.12)
    assert obs["gripper.pos"] == pytest.approx(186.072)
    np.testing.assert_allclose(state_from_degrees(obs, 0, 310.12), q)


def test_contract_rejects_swapped_arms_and_units(cfg):
    data = cfg["contract_object"].to_dict()
    data["joint_units"] = "deg"
    with pytest.raises(ValueError):
        Contract.from_dict(data)
    with pytest.raises(ValueError):
        Contract(("right", "left"), ("camera_top", "camera_wrist_left", "camera_wrist_right"), .02)


def test_bounded_motion_reversal_braking_and_hold(cfg):
    arm = cfg["arms"]["right"]
    limiter = TrajectoryLimiter(arm, arm["mock_initial"])
    feedback = limiter.q.copy()
    for i in range(700):
        target = np.array(arm["mock_initial"])
        target[0] = .3 if i < 250 else -.3
        old_q, old_v = limiter.q.copy(), limiter.v.copy()
        q, v = limiter.step(None if i >= 500 else target, feedback, .02)
        assert np.max(np.abs(v) - arm["max_velocity"]) <= 1e-10
        assert np.max(np.abs(v - old_v) - arm["max_acceleration"] * .02) <= 1e-10
        np.testing.assert_allclose(q - old_q, v * .02, atol=1e-10)
        assert np.all(q >= arm["lower"]) and np.all(q <= arm["upper"])
        feedback = q
    np.testing.assert_allclose(limiter.v, 0, atol=1e-10)


def test_outlier_and_tracking_fault(cfg):
    arm = cfg["arms"]["right"]
    limiter = TrajectoryLimiter(arm, arm["mock_initial"])
    bad = limiter.q.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        limiter.step(bad, limiter.q, .02)
    feedback = limiter.q.copy()
    feedback[0] += .1
    with pytest.raises(MotionFault, match="tracking"):
        limiter.step(limiter.q, feedback, .02)
    actions = np.tile(limiter.q, (50, 1))
    actions[-1, 0] = 100
    with pytest.raises(MotionFault):
        validate_chunk(actions, cfg)


def test_action_latency_skips_past_and_never_repeats_expired(cfg):
    timeline = ActionTimeline(cfg)
    actions = np.tile(cfg["arms"]["right"]["mock_initial"], (50, 1))
    actions[:, 0] = np.arange(50) * .001
    timeline.accept({"captured_s": 100.}, actions, 100.105)
    assert timeline.target(100.105)[0] == pytest.approx(.005)
    assert timeline.target(101.) is None
    with pytest.raises(MotionFault, match="old"):
        timeline.accept({"captured_s": 100.}, actions, 101.)


def test_execute_gate_precedes_hardware_access(cfg, monkeypatch):
    monkeypatch.setattr("rs_lingbot.runtime.RSArm", lambda *a, **k: pytest.fail("hardware touched"))
    with pytest.raises(ValueError, match="execute"):
        run(cfg, mode="live")
    with pytest.raises(ValueError, match="actual model"):
        run(cfg, mode="shadow")
    Path(cfg["runtime"]["stop_file"]).touch()
    with pytest.raises(MotionFault, match="stop file"):
        run(cfg)


@pytest.mark.parametrize("filename", ["single_right.yaml", "dual_stack_three.yaml"])
def test_mock_control_loop_logs_complete_session(filename, tmp_path):
    cfg = load_config(ROOT / "configs" / filename)
    cfg["runtime"]["log_dir"] = str(tmp_path)
    cfg["runtime"]["stop_file"] = str(tmp_path / "STOP")
    output = run(cfg, duration=.5)
    result = json.loads((output / "result.json").read_text())
    assert result == dict(status="completed_duration", cleanup_errors=[])
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert any(x["event"] == "chunk" for x in events)
    ticks = [x for x in events if x["event"] == "tick"]
    assert len(ticks) >= 10
    assert all(not x["executed"] for x in ticks)
    assert all(len(x["command"]) == cfg["contract_object"].dim for x in ticks)


def test_http_model_contract_and_image_roundtrip(cfg):
    class Backend:
        contract = cfg["contract_object"]
        manifest = dict(model_id="test-only")

        def infer(self, state, images, task):
            assert task == "stack blocks"
            assert images["camera_top"][0, 0].tolist() == [255, 12, 34]
            return np.tile(state, (50, 1))

    server = make_server(PolicyService(Backend()), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = HTTPPolicy(f"http://127.0.0.1:{server.server_port}", cfg["contract_object"], 2)
        image = np.full((4, 5, 3), [255, 12, 34], dtype=np.uint8)
        obs = dict(request_id=3, state=np.array(cfg["arms"]["right"]["mock_initial"]),
                   task="stack blocks", images={k: image for k in cfg["contract_object"].cameras})
        result = np.asarray(client.infer(obs))
        assert result.shape == (50, 7)
        np.testing.assert_array_equal(decode_image(encode_image(image)), image)
        wrong = Contract(("left",), ("camera_top", "camera_wrist_left"), .02)
        with pytest.raises(ValueError, match="mismatch"):
            HTTPPolicy(f"http://127.0.0.1:{server.server_port}", wrong, 2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_async_failure_is_delivered_and_close_does_not_wait():
    class BadPolicy:
        def infer(self, obs):
            raise TimeoutError("test timeout")
    worker = InferenceWorker(BadPolicy())
    try:
        worker.submit(dict(request_id=1))
        deadline = time.monotonic() + 1
        result = None
        while result is None and time.monotonic() < deadline:
            result = worker.poll()
            time.sleep(.005)
        assert result[2] == "test timeout"
    finally:
        worker.close()


def test_camera_age_skew_and_disconnect_are_rejected():
    cameras = Cameras({"camera_top": {}, "camera_wrist_right": {}})
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    cameras.latest = {"camera_top": (10., image), "camera_wrist_right": (10.11, image)}
    with pytest.raises(MotionFault, match="unsynchronized"):
        cameras.snapshot(10.12, .2, .08)
    cameras.latest["camera_wrist_right"] = (10., image)
    with pytest.raises(MotionFault, match="stale"):
        cameras.snapshot(11., .2, .08)
    cameras.errors["camera_top"] = "disconnected"
    with pytest.raises(MotionFault, match="failure"):
        cameras.snapshot(10.01, .2, .08)


def test_dual_live_rejected_before_any_hardware(tmp_path, monkeypatch):
    cfg = load_config(ROOT / "configs/dual_stack_three.yaml")
    monkeypatch.setattr("rs_lingbot.runtime.RSArm", lambda *a, **k: pytest.fail("hardware touched"))
    with pytest.raises(ValueError, match="dual-arm"):
        run(cfg, mode="live", execute=True, policy_url="http://127.0.0.1:1")


def test_raw_checkpoint_cannot_enable_rs(cfg, monkeypatch):
    class DiagnosticPolicy:
        contract = cfg["contract_object"]
    monkeypatch.setattr("rs_lingbot.runtime.HTTPPolicy", lambda *a: DiagnosticPolicy())
    monkeypatch.setattr("rs_lingbot.runtime.RSArm", lambda *a, **k: pytest.fail("hardware touched"))
    with pytest.raises(ValueError, match="RS-finetuned"):
        run(cfg, mode="live", execute=True, policy_url="http://127.0.0.1:1")


def test_runtime_inference_failure_closes_all_arms(cfg, monkeypatch):
    from rs_lingbot.hardware import MockArm
    closed = []
    class TrackedArm(MockArm):
        def close(self):
            closed.append(True)
    class FailingPolicy:
        model_id = "fault-injection"
        def __init__(self, contract):
            self.contract = contract
            self.calls = 0
        def infer(self, obs):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("injected inference failure")
            return np.tile(obs["state"], (50, 1))
    monkeypatch.setattr("rs_lingbot.runtime.MockArm", TrackedArm)
    monkeypatch.setattr("rs_lingbot.runtime.MockPolicy", FailingPolicy)
    with pytest.raises(MotionFault, match="injected"):
        run(cfg, duration=.5)
    assert closed == [True]
    results = list(Path(cfg["runtime"]["log_dir"]).glob("*/result.json"))
    assert json.loads(results[0].read_text())["status"] == "failed"
