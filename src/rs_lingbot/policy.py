"""Bounded HTTP client and one in-flight asynchronous inference worker."""
import base64
import json
import queue
import threading
import time
import urllib.request
import numpy as np
from .contract import Contract

MAX_BODY = 32 * 1024 * 1024


def encode_image(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("images must be HWC RGB uint8")
    return dict(shape=list(image.shape), rgb=base64.b64encode(image.tobytes()).decode("ascii"))


def decode_image(data):
    shape = data["shape"]
    if (len(shape) != 3 or shape[2] != 3 or
            any(type(x) is not int or not 1 <= x <= 4096 for x in shape)):
        raise ValueError("invalid RGB image dimensions")
    raw = base64.b64decode(data["rgb"], validate=True)
    if len(raw) != int(np.prod(shape)):
        raise ValueError("invalid RGB byte count")
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()


class HTTPPolicy:
    def __init__(self, url, contract, timeout):
        if not url.startswith("http://"):
            raise ValueError("use a trusted local HTTP endpoint (SSH tunnel for remote inference)")
        self.url, self.timeout = url.rstrip("/"), timeout
        info = self._request("/metadata")
        self.contract = Contract.from_dict(info["contract"])
        contract.compatible(self.contract)
        self.model_id = info["model_id"]

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        if data is not None and len(data) > MAX_BODY:
            raise ValueError("request too large")
        request = urllib.request.Request(self.url + path, data=data,
                                        headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read(MAX_BODY + 1)
        if len(raw) > MAX_BODY:
            raise ValueError("response too large")
        return json.loads(raw)

    def infer(self, observation):
        payload = dict(request_id=observation["request_id"],
                       state=observation["state"].tolist(), task=observation["task"],
                       images={k: encode_image(v) for k, v in observation["images"].items()})
        result = self._request("/infer", payload)
        if result["request_id"] != observation["request_id"]:
            raise ValueError("policy response request ID mismatch")
        return result["actions"]


class MockPolicy:
    """A tiny gripper/joint oscillation for software testing, NOT a grasp policy."""
    model_id = "mock-motion-only"

    def __init__(self, contract):
        self.contract = contract

    def infer(self, obs):
        time.sleep(.01)
        state = obs["state"]
        actions = np.tile(state, (50, 1))
        for i in range(len(self.contract.arms)):
            actions[:, i * 7] += .01
            actions[:, i * 7 + 6] = np.clip(state[i * 7 + 6] - .05, .1, .9)
        return actions


class InferenceWorker:
    def __init__(self, policy):
        self.policy = policy
        self.requests = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.pending = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, observation):
        if self.pending:
            return False
        self.pending = True
        self.requests.put_nowait(observation)
        return True

    def poll(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.pending = False
        return result

    def _run(self):
        while not self.stop.is_set():
            try:
                obs = self.requests.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                actions, error = self.policy.infer(obs), None
            except Exception as exc:
                actions, error = None, str(exc)
            if not self.stop.is_set():
                self.results.put_nowait((obs, actions, error, time.monotonic()))

    def close(self):
        self.stop.set()
        # Worker owns no hardware. Never postpone a motor shutdown for network I/O.
        self.thread.join(timeout=0)
