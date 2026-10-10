import threading
import time
import numpy as np
from .safety import MotionFault


class Cameras:
    def __init__(self, configs, *, mock=False):
        self.configs, self.mock = configs, mock
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest, self.errors, self.threads = {}, {}, []

    def start(self):
        if self.mock:
            return
        for name, config in self.configs.items():
            t = threading.Thread(target=self._capture, args=(name, config), daemon=True)
            self.threads.append(t)
            t.start()

    def _capture(self, name, cfg):
        import cv2
        cap = None
        try:
            cap = cv2.VideoCapture(cfg["device"], cv2.CAP_V4L2)
            if not cap.isOpened():
                raise RuntimeError(f"cannot open camera {name}: {cfg['device']}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.get("width", 640))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.get("height", 480))
            cap.set(cv2.CAP_PROP_FPS, cfg.get("fps", 30))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            while not self.stop.is_set():
                ok, frame = cap.read()
                stamp = time.monotonic()
                if not ok:
                    raise RuntimeError(f"camera {name} read failed")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.latest[name] = (stamp, rgb)
        except Exception as exc:
            with self.lock:
                self.errors[name] = str(exc)
        finally:
            if cap is not None:
                cap.release()

    def snapshot(self, now, max_age, max_skew):
        if self.mock:
            return {name: np.zeros((48, 64, 3), dtype=np.uint8) for name in self.configs}, now
        with self.lock:
            if self.errors:
                raise MotionFault(f"camera failure: {self.errors}")
            frames = dict(self.latest)
        if set(frames) != set(self.configs):
            raise MotionFault("waiting for all cameras")
        stamps = [v[0] for v in frames.values()]
        if now - min(stamps) > max_age or max(stamps) - min(stamps) > max_skew:
            raise MotionFault("camera frames stale or unsynchronized")
        return {name: pair[1].copy() for name, pair in frames.items()}, min(stamps)

    def close(self):
        self.stop.set()
        for t in self.threads:
            t.join(timeout=.3)
