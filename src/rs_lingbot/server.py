"""Model-only HTTP service. No hardware imports or CAN access."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import numpy as np
from .bundle import inspect_checkpoint, check_norm
from .contract import Contract, vector
from .policy import MAX_BODY, decode_image


class LingBotBackend:
    def __init__(self, bundle, lingbot_repo, qwen_path, precision="bf16"):
        root = Path(bundle).resolve()
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.contract = Contract.from_dict(self.manifest["contract"])
        self.diagnostic = self.manifest["source"] == "robotwin-diagnostic"
        if self.manifest["source"] not in ("rs", "robotwin-diagnostic"):
            raise ValueError("unsupported bundle source")
        if self.diagnostic and self.contract.rs_finetuned:
            raise ValueError("diagnostic bundle cannot enable RS hardware")
        checkpoint = root / self.manifest["checkpoint"]
        inspect_checkpoint(checkpoint)
        norm = root / self.manifest["norm_stats"]
        check_norm(norm, 2 if self.diagnostic else len(self.contract.arms))
        sys.path.insert(0, str(Path(lingbot_repo).resolve()))
        os.environ["QWEN3VL_PATH"] = str(Path(qwen_path).resolve())
        from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
        from lingbotvla.data.vla_data.utils import FeatureTransform
        self.policy = LingbotVLAv2Server(str(checkpoint), robot_norm_path=str(norm),
            use_length=-1, chunk_ret=True, use_bf16=precision == "bf16",
            use_fp32=precision == "fp32", use_compile=False)
        # Explicit robot configuration, avoiding upstream reset's cwd-relative path.
        transform = FeatureTransform(str(root / self.manifest["robot_config"]),
            self.policy.data_config, self.policy.config, self.policy.processor,
            chunk_size=self.policy.config.chunk_size, norm_stats_path=str(norm))
        self.policy.vla.feature_transform = transform
        self.policy.action_key = transform.org_features["actions"]

    def infer(self, state, images, task):
        if self.diagnostic:
            # Deliberately diagnostic only: no claim that zero-filled absent Aloha
            # joints or RS geometry match RoboTwin training.
            raw = np.zeros(14, dtype=np.float32)
            for i, side in enumerate(self.contract.arms):
                start = 0 if side == "left" else 7
                raw[start:start + 7] = state[i * 7:(i + 1) * 7]
        else:
            raw = state.astype(np.float32)
        obs = {"observation.state": raw, "task": task,
               **{f"observation.images.{name}": value for name, value in images.items()}}
        result = np.asarray(self.policy.infer(obs)["action"], dtype=float)
        dim = 14 if self.diagnostic else self.contract.dim
        if result.ndim != 2 or result.shape[1] != dim or not np.all(np.isfinite(result)):
            raise ValueError("LingBot returned invalid action dimensions/values")
        if self.diagnostic:
            result = np.concatenate([result[:, (0 if side == "left" else 7):(7 if side == "left" else 14)]
                                     for side in self.contract.arms], axis=1)
        return result


class PolicyService:
    def __init__(self, backend):
        self.backend = backend
        self.lock = threading.Lock()

    def metadata(self):
        return dict(contract=self.backend.contract.to_dict(), model_id=self.backend.manifest["model_id"])

    def infer(self, payload):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("policy busy; use one controller per model server")
        try:
            c = self.backend.contract
            state = vector(payload["state"], c.dim, "policy state")
            if set(payload["images"]) != set(c.cameras):
                raise ValueError("camera keys differ from model contract")
            task = payload["task"]
            if not isinstance(task, str) or not 1 <= len(task) <= 2000:
                raise ValueError("task must be a nonempty instruction")
            if type(payload["request_id"]) is not int or payload["request_id"] < 0:
                raise ValueError("invalid request ID")
            images = {k: decode_image(v) for k, v in payload["images"].items()}
            start = time.monotonic()
            actions = np.asarray(self.backend.infer(state, images, task), dtype=float)
            if (actions.ndim != 2 or actions.shape[1] != c.dim or
                    not 1 <= len(actions) <= 1000 or not np.all(np.isfinite(actions))):
                raise ValueError("model returned invalid action chunk")
            return dict(request_id=payload["request_id"], actions=actions.tolist(),
                        infer_s=time.monotonic() - start)
        finally:
            self.lock.release()


def make_server(service, host="127.0.0.1", port=9331):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def reply(self, code, value):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/metadata":
                self.reply(200, service.metadata())
            elif self.path == "/healthz":
                self.reply(200, dict(ready=True))
            else:
                self.reply(404, dict(error="not found"))

        def do_POST(self):
            if self.path != "/infer":
                self.reply(404, dict(error="not found"))
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY:
                    raise ValueError("invalid request length")
                value = service.infer(json.loads(self.rfile.read(size)))
                self.reply(200, value)
            except Exception as exc:
                traceback.print_exc()
                self.reply(400, dict(error=str(exc)))
    return ThreadingHTTPServer((host, port), Handler)


def main():
    p = argparse.ArgumentParser(description="LingBot RS model service (no robot access)")
    p.add_argument("--bundle", required=True)
    p.add_argument("--lingbot-repo", required=True)
    p.add_argument("--qwen-path", required=True)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9331)
    a = p.parse_args()
    backend = LingBotBackend(a.bundle, a.lingbot_repo, a.qwen_path, a.precision)
    with make_server(PolicyService(backend), a.host, a.port) as server:
        print(f"Serving on {a.host}:{a.port}; rs_finetuned={backend.contract.rs_finetuned}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
