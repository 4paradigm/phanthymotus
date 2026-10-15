"""Opt-in robot model recognition using VOP boxes and a local CLIP encoder."""

from __future__ import annotations

import threading

import numpy as np

from plugins.image_input import BadInput, load_image_bytes
from plugins.robot_model_catalog import CANDIDATES, decide_model
from plugins.robot_model_proxy import RobotModelProxy


ROBOT_CLASSES = frozenset(("robot", "quadruped robot", "humanoid robot", "drone"))
TOOLS = [{
    "name": "robot_model", "type": "processor", "multiInstance": True,
    "description": "Identify visible robot models locally from a photo or camera frame",
    "inputSchema": {
        "type": "object", "required": ["action"],
        "properties": {
            "action": {"type": "string", "enum": ["start", "stop", "info", "recognize_by_photo", "recognize_by_url"]},
            "input_topic": {"type": "string", "description": "Camera image/jpeg topic"},
            "image_path": {"type": "string", "format": "file", "accept": "image/*", "uploadTo": "mcp"},
            "url": {"type": "string", "description": "http(s) image URL"},
        },
        "x-action-params": {
            "start": {"params": ["input_topic"]},
            "stop": {"params": []},
            "info": {"params": []},
            "recognize_by_photo": {"params": ["image_path"]},
            "recognize_by_url": {"params": ["url"]},
        },
    },
    "topic_in": [{"format": "image/jpeg", "desc": "camera frame"}],
    "topic_out": [{"format": "data/json", "desc": "robot model results"}],
}]


class RobotModelPlugin:
    def __init__(self, plugin_cfg: dict, detector, executor):
        self._cfg = dict(plugin_cfg)
        self._detector = detector
        self._executor = executor
        self._proxy = None
        self._proxy_lock = threading.Lock()

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == "robot_model" else name
        if action == "info":
            return {"name": "Robot Model", "model": "robot_model",
                    "state": "idle" if self._detector else "unavailable",
                    "model_loaded": self._proxy is not None,
                    "calibrated": self._cfg.get("min_score") is not None and
                                  self._cfg.get("min_margin") is not None,
                    "detail": "VOP detector is required" if self._detector is None else
                              "On-demand photo recognition is available",
                    "topic_in": [], "topic_out": []}
        if action == "start":
            if self._detector is None:
                return self._unavailable("VOP detector is disabled")
            if args.get("input_topic"):
                return {"ok": False, "reason": "unsupported_action",
                        "detail": "Camera streaming is not available yet"}
            return {"state": "idle", "detail": "Photo recognition is available on demand"}
        if action == "stop":
            return {"state": "idle"}
        if action in ("recognize_by_photo", "recognize_by_url"):
            return self._recognize_image(args)
        return None

    @staticmethod
    def _unavailable(detail: str) -> dict:
        return {"ok": False, "reason": "engine_unavailable", "detail": detail}

    def _get_proxy(self):
        with self._proxy_lock:
            if self._proxy is None:
                self._proxy = RobotModelProxy(
                    str(self._cfg.get("model_dir", "/models/robot_model")))
            return self._proxy

    def recognize_frame(self, frame: np.ndarray) -> dict:
        if self._detector is None:
            return self._unavailable("VOP detector is disabled")
        try:
            detections = self._detector.detect_frame(
                frame, float(self._cfg.get("detector_confidence", 0.3)))
        except Exception as error:
            return self._unavailable(f"VOP detection failed: {error}")

        height, width = frame.shape[:2]
        objects = []
        for detected in detections:
            if detected.get("name") not in ROBOT_CLASSES:
                continue
            x1, y1, x2, y2 = detected["bbox"]
            x1, x2 = max(0, int(x1)), min(width, int(x2))
            y1, y2 = max(0, int(y1)), min(height, int(y2))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            import cv2
            encoded, data = cv2.imencode(".jpg", frame[y1:y2, x1:x2])
            if not encoded:
                continue
            try:
                proxy = self._get_proxy()
                embedding = np.asarray(proxy.embed(data.tobytes()), dtype=np.float32)
                scores = {model_id: float(score) for (model_id, _), score in
                          zip(CANDIDATES, proxy.vectors @ embedding)}
                decision = decide_model(scores, self._cfg.get("min_score"),
                                        self._cfg.get("min_margin"))
            except Exception as error:
                return self._unavailable(f"Robot model encoder unavailable: {error}")
            objects.append({
                "bbox": [x1, y1, x2, y2],
                "detector_class": detected["name"],
                "detector_score": detected["confidence"],
                **decision,
            })
        return {"ok": True, "objects": objects}

    def _recognize_image(self, args: dict) -> dict:
        if self._detector is None:
            return self._unavailable("VOP detector is disabled")
        try:
            data, source = load_image_bytes(
                args, self._cfg, url_action="recognize_by_url")
        except BadInput as error:
            return error.as_result()
        import cv2
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return BadInput("could not decode that file as an image", source).as_result()
        return {"source": source, **self.recognize_frame(frame)}
