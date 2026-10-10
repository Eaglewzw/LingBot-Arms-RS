"""One control owner. Inference and camera I/O never run in the motor loop."""
from datetime import datetime
import json
from pathlib import Path
import signal
import time
import numpy as np
from .cameras import Cameras
from .hardware import MockArm, RSArm
from .policy import HTTPPolicy, MockPolicy, InferenceWorker
from .safety import MotionFault, TrajectoryLimiter, validate_chunk


class ActionTimeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.actions = None
        self.capture_s = None
        self.accepted_s = None

    def accept(self, observation, actions, now):
        if now - observation["captured_s"] > self.cfg["runtime"]["max_response_age_s"]:
            raise MotionFault("policy response is too old for its source observation")
        self.actions = validate_chunk(actions, self.cfg)
        self.capture_s = observation["captured_s"]
        self.accepted_s = now
        if self.target(now) is None:
            raise MotionFault("policy chunk expired before arrival")

    def target(self, now):
        if self.actions is None:
            return None
        index = int(max(0, now - self.capture_s) / self.cfg["contract_object"].action_dt_s)
        limit = min(len(self.actions), self.cfg["runtime"]["execute_steps"])
        return None if index >= limit else self.actions[index]

    def near_end(self, now):
        if self.actions is None:
            return True
        horizon = min(len(self.actions), self.cfg["runtime"]["execute_steps"])
        remaining = horizon * self.cfg["contract_object"].action_dt_s - (now - self.capture_s)
        return remaining <= self.cfg["runtime"]["max_response_age_s"] + .05


def run(cfg, *, mode="mock", execute=False, duration=10., policy_url=None):
    contract, rt = cfg["contract_object"], cfg["runtime"]
    if mode not in ("mock", "shadow", "live"):
        raise ValueError("unknown runtime mode")
    if mode == "live" and not execute:
        raise ValueError("live mode requires --execute; shadow is read-only")
    if execute and mode != "live":
        raise ValueError("--execute is only valid with --mode live")
    if mode == "live" and len(contract.arms) != 1:
        raise ValueError("dual-arm live execution is not commissioned; mock/shadow only")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    stop_file = Path(rt["stop_file"])
    if stop_file.exists():
        raise MotionFault(f"stop file exists: {stop_file}")
    if mode != "mock" and not policy_url:
        raise ValueError("hardware modes require an actual model server --policy-url")
    policy = HTTPPolicy(policy_url, contract, rt["policy_timeout_s"]) if policy_url else MockPolicy(contract)
    if mode == "live" and not policy.contract.rs_finetuned:
        raise ValueError("live RS requires an RS-finetuned deployment bundle; raw RoboTwin 18K is shadow-only")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir = Path(rt["log_dir"]) / stamp
    log_dir.mkdir(parents=True)
    (log_dir / "session.json").write_text(json.dumps(dict(mode=mode, contract=contract.to_dict(),
        model_id=policy.model_id, duration_s=duration), indent=2))
    (log_dir / "runtime_config.json").write_text(json.dumps(
        {k: v for k, v in cfg.items() if k != "contract_object"}, indent=2,
        default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value)))
    arms, worker = {}, None
    cameras = Cameras(cfg["cameras"], mock=mode == "mock")
    stop = False
    signal_handlers = {}

    def stop_now(signum, frame):
        nonlocal stop
        stop = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal_handlers[sig] = signal.signal(sig, stop_now)
    status = "failed"
    with (log_dir / "events.jsonl").open("w", buffering=1) as log:
        def record(event, **data):
            log.write(json.dumps(dict(event=event, monotonic_s=time.monotonic(), **data), allow_nan=False) + "\n")

        try:
            for side in contract.arms:
                arms[side] = (MockArm(cfg["arms"][side]) if mode == "mock" else
                    RSArm(cfg["arms"][side], execute=execute, report_path=log_dir / f"{side}_shutdown.json"))
            cameras.start()
            # Camera/model failures are resolved before any enable command.
            camera_deadline = time.monotonic() + 5
            while True:
                if stop or stop_file.exists():
                    raise MotionFault("stopped before startup")
                try:
                    images, captured = cameras.snapshot(time.monotonic(), rt["max_observation_age_s"], rt["max_camera_skew_s"])
                    break
                except MotionFault:
                    if time.monotonic() >= camera_deadline:
                        raise
                    time.sleep(.02)
            initial = {side: (arm.read() if mode == "mock" else arm.probe()) for side, arm in arms.items()}
            for side in contract.arms:
                TrajectoryLimiter(cfg["arms"][side], initial[side])
            warmup = dict(request_id=0, state=np.concatenate(list(initial.values())),
                          task=cfg["task"], images=images, captured_s=captured)
            validate_chunk(policy.infer(warmup), cfg)
            record("warmup_complete", motors_enabled=False)
            if stop or stop_file.exists():
                raise MotionFault("stopped before enable")
            for arm in arms.values():
                arm.connect()
            limiters = {side: TrajectoryLimiter(cfg["arms"][side], arm.read()) for side, arm in arms.items()}
            worker = InferenceWorker(policy)
            timeline = ActionTimeline(cfg)
            started = last = time.monotonic()
            last_available = started
            request_id = 1
            ticks = 0
            period = 1. / rt["control_hz"]
            deadline = started + period
            while not stop and time.monotonic() - started < duration:
                if stop_file.exists():
                    raise MotionFault("operator stop file detected")
                now = time.monotonic()
                dt = now - last
                if ticks == 0:
                    dt = period
                if dt > rt["max_tick_s"]:
                    raise MotionFault(f"control loop deadline missed: {dt:.3f}s")
                last = now
                read_started = time.monotonic()
                feedback = {side: arm.read() for side, arm in arms.items()}
                after_read = time.monotonic()
                if after_read - read_started > rt["max_observation_age_s"]:
                    raise MotionFault("joint feedback read exceeded freshness budget")
                images, camera_stamp = cameras.snapshot(after_read, rt["max_observation_age_s"], rt["max_camera_skew_s"])
                captured = min(read_started, camera_stamp)
                if after_read - captured > rt["max_observation_age_s"]:
                    raise MotionFault("observation is stale")
                result = worker.poll()
                if result is not None:
                    obs, actions, error, received = result
                    if error:
                        raise MotionFault(f"inference failed: {error}")
                    timeline.accept(obs, actions, after_read)
                    record("chunk", request_id=obs["request_id"], capture_s=obs["captured_s"],
                           latency_s=received - obs["captured_s"], actions=timeline.actions.tolist())
                if not worker.pending and timeline.near_end(after_read):
                    worker.submit(dict(request_id=request_id, state=np.concatenate(list(feedback.values())),
                                       images=images, captured_s=captured, task=cfg["task"]))
                    request_id += 1
                target = timeline.target(after_read)
                if target is not None:
                    last_available = after_read
                elif after_read - last_available > rt["policy_wait_timeout_s"]:
                    raise MotionFault("no fresh policy actions; policy wait timeout")
                # Validate every arm before sending any command. Shadow evaluates
                # only the next bounded step, never integrates imaginary motion.
                commands = {}
                for i, side in enumerate(contract.arms):
                    if mode == "shadow":
                        limiters[side] = TrajectoryLimiter(cfg["arms"][side], feedback[side])
                    part = None if target is None else target[i * 7:(i + 1) * 7]
                    commands[side] = limiters[side].step(part, feedback[side], dt)
                if stop or stop_file.exists():
                    raise MotionFault("stopped before command dispatch")
                if time.monotonic() - now > rt["max_tick_s"]:
                    raise MotionFault("control work exceeded deadline before dispatch")
                if time.monotonic() - captured > rt["max_observation_age_s"]:
                    raise MotionFault("observation expired before command dispatch")
                for side, (q, v) in commands.items():
                    arms[side].send(q, v)
                record("tick", state=np.concatenate(list(feedback.values())).tolist(),
                       command=np.concatenate([c[0] for c in commands.values()]).tolist(),
                       holding=target is None, executed=mode == "live")
                ticks += 1
                delay = deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                deadline += period
            status = "stopped" if stop else "completed_duration"
            record("finished", status=status, ticks=ticks)
        except BaseException as exc:
            record("fault", error=str(exc))
            raise
        finally:
            # No automatic home/zero trajectory on a perception or policy failure.
            # Upstream disconnect attempts disable; the arm must be supported.
            if worker:
                worker.close()
            cleanup_errors = []
            for side, arm in reversed(list(arms.items())):
                try:
                    arm.close()
                except Exception as exc:
                    cleanup_errors.append(f"{side}: {exc}")
            cameras.close()
            for sig, handler in signal_handlers.items():
                signal.signal(sig, handler)
            (log_dir / "result.json").write_text(json.dumps(dict(status=status, cleanup_errors=cleanup_errors), indent=2))
            if cleanup_errors:
                raise RuntimeError(f"hardware cleanup errors: {cleanup_errors}")
    return log_dir
