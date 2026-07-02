#!/usr/bin/env python3
"""
plugins/vop.py — VideoObjectPerceptionPlugin: YOLOv8-World open-vocabulary object detection.

Subscribes to image/jpeg topics, runs YOLOv8s-Worldv2 inference,
publishes detected objects with center-relative normalized coordinates.
Supports multi-instance (one instance per input topic).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

log = logging.getLogger(__name__)

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=2,
    durability=DurabilityPolicy.VOLATILE,
)

_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

_MODEL_URLS = {
    "yolov8s-worldv2": "https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public/yolov8s-worldv2.pt",
    "clip-vit-b-32": "https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public/ViT-B-32.pt",
}

_COCO_80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

TOOLS = [
    {
        "name": "vop",
        "type": "processor",
        "multiInstance": True,
        "description": "Video Object Perception — detect objects in camera feed using open-vocabulary YOLO",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "set_classes", "config"],
                    "description": "Action to perform"
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 image topic to subscribe (e.g. /hostname/camera/rgb, required for action=start)"
                },
                "classes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra object classes to add on top of COCO-80 base (for set_classes action)"
                },
            },
            "required": ["action"]
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "confidence": {"type": "number", "description": "Detection confidence threshold (0-1)", "default": 0.3, "scope": "instance"},
                "fps":        {"type": "integer", "description": "Max inference frames per second", "default": 5, "scope": "instance"},
                "classes":    {"type": "array", "items": {"type": "string"}, "description": "Extra object classes to add on top of COCO-80 base", "scope": "instance"},
            },
        },
        "topic_in":  [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [{"format": "data/json",  "desc": "detected objects with positions"}],
    }
]


# ── ROS2 Node (one per instance/topic) ────────────────────────────────────────

class _VOPNode(Node):
    """Per-topic YOLO inference node."""

    def __init__(self, input_topic: str, model, confidence: float, fps: float,
                 extra_classes: list[str], node_suffix: str):
        super().__init__(f"vop_{node_suffix}")
        self._input_topic = input_topic
        self._output_topic = f"{input_topic}/objects"
        self._model = model
        self._confidence = confidence
        self._fps = fps
        self._frame_interval = 1.0 / max(fps, 0.1)
        self._extra_classes = extra_classes
        self._classes = list(_COCO_80_CLASSES) + [c for c in extra_classes if c not in _COCO_80_CLASSES]

        self._pub = self.create_publisher(String, self._output_topic, _PUB_QOS)
        self._sub: Optional[object] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._last_inference_time = 0.0
        self._detect_count = 0

    def start(self) -> dict:
        if self._sub is not None:
            return {"state": "running", "input": self._input_topic, "output": self._output_topic}
        self._stop_event.clear()
        self._sub = self.create_subscription(
            CompressedImage, self._input_topic, self._image_cb, _LOW_LAT_QOS
        )
        self._worker = threading.Thread(target=self._inference_worker, daemon=True,
                                        name=f"vop_worker_{self._input_topic}")
        self._worker.start()
        log.info(f"[vop] started: {self._input_topic} → {self._output_topic}")
        return {"state": "running", "input": self._input_topic, "output": self._output_topic}

    def stop(self) -> dict:
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        self._worker = None
        log.info(f"[vop] stopped: {self._input_topic}")
        return {"state": "idle", "input": self._input_topic}

    def _image_cb(self, msg: CompressedImage):
        now = time.monotonic()
        if now - self._last_inference_time < self._frame_interval:
            return
        self._last_inference_time = now
        # Drop old frame if queue full (no backpressure)
        try:
            self._frame_queue.put_nowait(msg.data)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(msg.data)
            except queue.Full:
                pass

    def _inference_worker(self):
        import cv2
        while not self._stop_event.is_set():
            try:
                jpeg_bytes = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                frame = cv2.imdecode(
                    np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR
                )
                if frame is None:
                    continue
                results = self._model(frame, conf=self._confidence, verbose=False)
                objects = self._extract_objects(results[0], frame.shape)
                self._publish_objects(objects)
            except Exception as e:
                log.error(f"[vop] inference error: {e}", exc_info=True)

    def _extract_objects(self, result, shape) -> list:
        H, W = shape[:2]
        half_w, half_h = W / 2.0, H / 2.0
        objects = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            x_norm = round((cx - half_w) / half_w, 3)
            y_norm = round((cy - half_h) / half_h, 3)
            cls_id = int(box.cls[0])
            name = result.names[cls_id]
            conf = round(float(box.conf[0]), 2)
            objects.append({"name": name, "position": [x_norm, y_norm], "confidence": conf})
        return objects

    def _publish_objects(self, objects: list):
        self._detect_count += 1
        msg = String()
        msg.data = json.dumps({
            "timestamp": time.time(),
            "objects": objects,
        }, ensure_ascii=False)
        self._pub.publish(msg)


# ── Plugin class ──────────────────────────────────────────────────────────────

class VideoObjectPerceptionPlugin:
    PREFIX = "vop"

    def __init__(self, plugin_cfg: dict, namespace: str, executor):
        self._namespace = namespace
        self._executor = executor
        self._confidence = float(plugin_cfg.get("confidence", 0.3))
        self._fps = int(plugin_cfg.get("fps", 5))
        self._model_name = plugin_cfg.get("model", "yolov8s-worldv2")
        self._base_classes = list(_COCO_80_CLASSES)
        self._extra_classes: list[str] = plugin_cfg.get("classes") or []
        self._model = None  # lazy load
        self._model_lock = threading.Lock()
        self._nodes: dict[str, _VOPNode] = {}
        self._instance_configs: dict[str, dict] = {}  # per-instance config overrides

    def _get_all_classes(self) -> list[str]:
        """Merge base COCO-80 + global extra + all instance extra classes."""
        all_extra = set(self._extra_classes)
        for cfg in self._instance_configs.values():
            for c in cfg.get("classes") or []:
                all_extra.add(c)
        return self._base_classes + [c for c in sorted(all_extra) if c not in self._base_classes]

    def _sync_model_classes(self):
        """Re-sync model classes after config change."""
        if self._model is None:
            return
        classes = self._get_all_classes()
        # Move model to CPU for set_classes (CLIP tokenizer outputs CPU tensors)
        # then move back to GPU for inference
        try:
            self._model.model.cpu()
            self._model.set_classes(classes)
            if self._device and self._device != "cpu":
                self._model.model.to(self._device)
        except Exception as e:
            log.warning(f"[vop] set_classes failed: {e}, trying without device move")
            self._model.set_classes(classes)
        log.info(f"[vop] model classes synced: {len(classes)} total (+{len(classes) - len(self._base_classes)} extra)")

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return

            # Ensure YOLO_CONFIG_DIR points to /work so WEIGHTS_DIR = /work/weights
            # (CLIP weights are baked into image at /work/weights/clip/ViT-B-32.pt)
            os.environ.setdefault("YOLO_CONFIG_DIR", "/work")
            _model_dir = os.environ.get("YOLO_MODEL_DIR", "/models")
            os.makedirs(_model_dir, exist_ok=True)
            os.environ.setdefault("TORCH_HOME", _model_dir)

            # Fix broken system cv2 on Jetson (circular import in mat_wrapper)
            # and patch missing imshow for headless environments
            try:
                import cv2
                # Test if cv2 is functional
                _ = cv2.IMREAD_COLOR
            except (ImportError, AttributeError):
                import importlib.util, sys as _sys
                import glob as _glob
                # Find the .so directly
                _so_candidates = _glob.glob("/usr/lib/python*/dist-packages/cv2/python-*/cv2.cpython-*.so")
                if _so_candidates:
                    _spec = importlib.util.spec_from_file_location("cv2", _so_candidates[0])
                    _mod = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    _sys.modules["cv2"] = _mod
                    import cv2
                else:
                    import cv2  # let it fail naturally

            if not hasattr(cv2, 'imshow'):
                cv2.imshow = lambda *a, **k: None
                cv2.waitKey = lambda *a, **k: 0
                cv2.destroyAllWindows = lambda *a, **k: None

            from ultralytics import YOLO
            import torch

            # Determine device: prefer CUDA if available
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

            model_path = self._resolve_model_path()
            log.info(f"[vop] loading model: {model_path} (device={self._device})")
            self._model = YOLO(model_path)

            # Ensure CLIP weights are available locally before set_classes
            self._ensure_clip_weights()

            classes = self._get_all_classes()
            # set_classes on CPU (model loads on CPU by default), then move to GPU
            self._model.set_classes(classes)
            if self._device != "cpu":
                self._model.model.to(self._device)
            log.info(f"[vop] model loaded, {len(classes)} classes ({len(classes) - len(self._base_classes)} extra)")

    def _resolve_model_path(self) -> str:
        """Resolve model path: local file, config path, or download from COS."""
        # If config provides an absolute/relative path that exists, use it
        candidate = self._model_name if self._model_name.endswith(".pt") else f"{self._model_name}.pt"
        if os.path.isfile(candidate):
            return candidate

        # Check cache directory (mounted volume for persistence)
        cache_dir = os.environ.get("YOLO_MODEL_DIR", "/models")
        cached = os.path.join(cache_dir, os.path.basename(candidate))
        if os.path.isfile(cached):
            return cached

        # Download from COS mirror
        base_name = self._model_name.replace(".pt", "")
        url = _MODEL_URLS.get(base_name)
        if not url:
            # Fallback: let ultralytics handle download
            return candidate

        os.makedirs(cache_dir, exist_ok=True)
        log.info(f"[vop] downloading model from {url} → {cached}")
        urllib.request.urlretrieve(url, cached)
        log.info(f"[vop] download complete: {cached}")
        return cached

    def _ensure_clip_weights(self):
        """Ensure CLIP ViT-B-32 weights exist where ultralytics expects them.

        With YOLO_CONFIG_DIR=/work, ultralytics WEIGHTS_DIR = /work/weights,
        so clip.load(download_root=WEIGHTS_DIR/"clip") looks at /work/weights/clip/.
        The file is baked into the Docker image at build time.
        """
        clip_filename = "ViT-B-32.pt"
        target_path = "/work/weights/clip/" + clip_filename

        if os.path.isfile(target_path):
            return

        # Fallback: download from COS if not baked in (dev/local mode)
        url = _MODEL_URLS.get("clip-vit-b-32")
        if not url:
            return
        os.makedirs("/work/weights/clip", exist_ok=True)
        log.info(f"[vop] downloading CLIP weights from COS → {target_path}")
        urllib.request.urlretrieve(url, target_path)
        log.info(f"[vop] CLIP download complete: {target_path}")

    def _start_node(self, node_key: str, input_topic: str):
        """Create and start a VOPNode for the given topic."""
        icfg = self._instance_configs.get(node_key, {})
        confidence = float(icfg.get("confidence", self._confidence))
        fps = int(icfg.get("fps", self._fps))
        extra_classes = icfg.get("classes") or list(self._extra_classes)
        suffix = node_key.replace("/", "_").replace("-", "_").lstrip("_")
        node = _VOPNode(input_topic, self._model, confidence, fps,
                        extra_classes, node_suffix=suffix)
        self._executor.add_node(node)
        self._nodes[node_key] = node
        node.start()
        log.info(f"[vop] node started (background): {input_topic}")

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action", name)
        instance_id = args.get("instance_id", "")

        if action == "info":
            instances = {}
            for key, node in self._nodes.items():
                instances[key] = {
                    "input": node._input_topic,
                    "output": node._output_topic,
                    "confidence": node._confidence,
                    "fps": node._fps,
                    "extra_classes": node._extra_classes,
                    "detect_count": node._detect_count,
                }
            # Determine topic info: from running instance, args, or empty
            input_topic = args.get("input_topic", "")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            # If instance_id specified and running, use its topics
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                input_topic = node._input_topic
            # If no explicit topic but there are running instances, use first one
            elif not input_topic and self._nodes:
                first_node = next(iter(self._nodes.values()))
                input_topic = first_node._input_topic
            topics_in = [{"topic": input_topic, "format": "image/jpeg"}] if input_topic else []
            topics_out = [{"topic": f"{input_topic}/objects", "format": "data/json"}] if input_topic else []
            state = "running" if instances else "idle"
            return {
                "name": "VideoObjectPerception", "manufacture": "Embodied", "model": self._model_name,
                "state": state,
                "base_classes_count": len(self._base_classes),
                "total_classes": len(self._get_all_classes()),
                "instances": instances,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "YOLOv8-World open-vocabulary object detection",
            }

        elif action == "start":
            input_topic = args.get("input_topic")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if not input_topic:
                raise ValueError("input_topic is required")
            node_key = instance_id or input_topic
            if node_key not in self._nodes:
                if self._model is None:
                    # Model not loaded yet — start loading in background
                    def _bg_start():
                        self._ensure_model()
                        self._start_node(node_key, input_topic)
                    threading.Thread(target=_bg_start, daemon=True, name="vop_model_load").start()
                    return {"state": "loading", "input": input_topic, "output": f"{input_topic}/objects",
                            "message": "Model loading in background, will start automatically"}
                self._start_node(node_key, input_topic)
            return self._nodes[node_key].start()

        elif action == "stop":
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                result = node.stop()
                self._executor.remove_node(node)
                del self._nodes[instance_id]
                return result
            elif not instance_id and self._nodes:
                results = []
                for key in list(self._nodes.keys()):
                    node = self._nodes[key]
                    node.stop()
                    self._executor.remove_node(node)
                    del self._nodes[key]
                    results.append(key)
                return {"state": "idle", "stopped_instances": results}
            return {"state": "idle"}

        elif action == "set_classes":
            classes = args.get("classes")
            if not classes:
                raise ValueError("classes list is required")
            if instance_id:
                # Per-instance extra classes
                cfg = self._instance_configs.setdefault(instance_id, {})
                cfg["classes"] = classes
                # Update running node if exists
                if instance_id in self._nodes:
                    self._nodes[instance_id]._extra_classes = classes
                    self._nodes[instance_id]._classes = list(_COCO_80_CLASSES) + [c for c in classes if c not in _COCO_80_CLASSES]
            else:
                # Global extra classes
                self._extra_classes = classes
            # Sync model classes in background to avoid blocking HTTP (GPU model move is slow)
            threading.Thread(target=self._sync_model_classes, daemon=True, name="vop_sync_classes").start()
            return {"base_classes": len(self._base_classes), "extra_classes": classes, "total": len(self._get_all_classes())}

        elif action == "config":
            cfg = {k: v for k, v in args.items() if k not in ('action', 'instance_id') and v is not None and v != ''}
            if instance_id:
                self._instance_configs[instance_id] = cfg
                # If instance is running, restart with new config
                if instance_id in self._nodes:
                    node = self._nodes[instance_id]
                    input_topic = node._input_topic
                    node.stop()
                    self._executor.remove_node(node)
                    del self._nodes[instance_id]
                    # Will be re-created on next start with new config
                return {"status": "configured", "instance_id": instance_id, "config": cfg}
            else:
                # Update global defaults
                if "confidence" in cfg:
                    self._confidence = float(cfg["confidence"])
                if "fps" in cfg:
                    self._fps = int(cfg["fps"])
                if "classes" in cfg:
                    self._extra_classes = cfg["classes"]
                    # Sync model classes in background to avoid blocking HTTP
                    threading.Thread(target=self._sync_model_classes, daemon=True, name="vop_sync_classes").start()
                return {"status": "configured", "config": cfg}

        return None
