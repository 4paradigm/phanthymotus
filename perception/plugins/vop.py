#!/usr/bin/env python3
"""
plugins/vop.py — VideoObjectPerceptionPlugin: YOLOv8-World open-vocabulary object detection.

Subscribes to image/jpeg topics, runs YOLOv8s-Worldv2 inference,
publishes detected objects with center-relative normalized coordinates.
Supports multi-instance (one instance per input topic).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import logging
import os
import queue
import threading
import time
import urllib.request

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from plugins.vop_depth import DepthEstimator, extract_objects, render_preview, unavailable
from utils.cv2_compat import load_cv2

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
            "required": ["action"],
            "x-action-params": {
                "start": {"params": ["input_topic"], "description": "Start VOP on an image topic"},
                "stop": {"params": [], "description": "Stop and release this instance"},
                "info": {"params": ["input_topic"], "description": "Read state and output topics"},
                "set_classes": {"params": ["classes"], "description": "Add open-vocabulary classes"},
                "config": {"params": [], "description": "Configure this instance before starting"},
            }
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "confidence": {"type": "number", "description": "Detection confidence threshold (0-1)", "default": 0.3, "minimum": 0, "maximum": 1, "scope": "instance"},
                "fps":        {"type": "integer", "description": "Max inference frames per second", "default": 5, "minimum": 1, "maximum": 60, "scope": "instance"},
                "depth_enabled": {"type": "boolean", "default": False, "scope": "instance", "description": "物体轮廓内最近相对深度（未标定，不是米制避障距离）"},
                "classes":    {"type": "array", "items": {"type": "string"}, "description": "Extra object classes to add on top of COCO-80 base", "scope": "instance"},
            },
        },
        "topic_in":  [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [{"format": "data/json", "desc": "detected objects with positions and relative depth"},
                      {"format": "image/jpeg", "desc": "VOP 图像、分割轮廓与最近相对深度"}],
    }
]


# ── ROS2 Node (one per instance/topic) ────────────────────────────────────────

class _VOPNode(Node):
    """One cancellable worker per input; previews and JSON share a source frame."""

    def __init__(self, input_topic, plugin, config, node_suffix):
        super().__init__(f"vop_{node_suffix}")
        self._input_topic = input_topic
        self._output_topic = f"{input_topic}/objects"
        self._preview_topic = f"{input_topic}/objects/preview"
        self._plugin = plugin
        self._confidence = config["confidence"]
        self._fps = config["fps"]
        self._extra_classes = config["classes"]
        self._depth_enabled = config["depth_enabled"]
        self._pub = self.create_publisher(String, self._output_topic, _PUB_QOS)
        self._preview_pub = self.create_publisher(CompressedImage, self._preview_topic, _PUB_QOS)
        self._sub = None
        self._timer = None
        self._worker = None
        self._frame_queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._lifecycle_lock = threading.RLock()
        self._publish_lock = threading.RLock()
        self._state = "idle"
        self._error = None
        self._last_received = self._last_result = self._last_admitted = 0.0
        self._started = 0.0
        self._last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self._last_header = None
        self._source_received = 0.0
        self._detect_count = 0
        self._sequence = 0

    def start(self):
        with self._lifecycle_lock:
            if self._worker and self._worker.is_alive():
                if self._state == "stale" and not self._stop_event.is_set():
                    self._state, self._error = "waiting", None
                    self._started = time.monotonic()
                    self._last_received = self._last_result = 0.0
                return self.info()
            if self._stop_event.is_set():
                return {"state": "idle", "error": "Stopped instance; start again after stop completes"}
            self._state = "waiting"
            self._started = time.monotonic()
            self._sub = self.create_subscription(CompressedImage, self._input_topic,
                                                  self._image_cb, _LOW_LAT_QOS)
            self._timer = self.create_timer(1.0, self._watchdog)
            self._worker = threading.Thread(target=self._run, daemon=True,
                                            name=f"vop_{self.get_name()}")
            self._worker.start()
            return self.info()

    def stop(self):
        # Cancellation precedes the lifecycle lock so a concurrent start cannot win.
        with self._publish_lock:
            if not self._stop_event.is_set():
                try:
                    self._publish_status("stopped")
                except Exception:
                    log.exception("[vop] stop preview failed; continuing resource cleanup")
            self._stop_event.set()
            self._state = "stopping"
        with self._lifecycle_lock:
            if self._sub is not None:
                self.destroy_subscription(self._sub)
                self._sub = None
            if self._timer is not None:
                self.destroy_timer(self._timer)
                self._timer = None
            if self._worker:
                self._worker.join(timeout=3.0)
            if self._worker and self._worker.is_alive():
                return {"state": "stopping", "input": self._input_topic}
            self._state = "idle"
            return {"state": "idle", "input": self._input_topic}

    def info(self):
        now = time.monotonic()
        state = self._state
        error = self._error
        if state in ("waiting", "loading") or (state == "processing" and not self._last_result):
            state = "loading"
        elif state == "processing":
            state = "running"
        elif state in ("stale", "stopping"):
            error = error or ("No image input for 3 seconds" if state == "stale" else "Stop is still in progress")
            state = "error"
        return {"state": state, "phase": self._state, "error": error,
                "input": self._input_topic, "output": self._output_topic,
                "preview": self._preview_topic, "confidence": self._confidence,
                "fps": self._fps, "extra_classes": self._extra_classes,
                "depth_enabled": self._depth_enabled, "detect_count": self._detect_count,
                "result_age_s": now - self._last_result if self._last_result else None}

    def _image_cb(self, msg):
        if self._stop_event.is_set():
            return
        now = time.monotonic()
        self._last_received = now
        if now - self._last_admitted < 1.0 / self._fps:
            return
        self._last_admitted = now
        item = (msg, now)
        try:
            self._frame_queue.put_nowait(item)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(item)
            except queue.Full:
                pass

    def _watchdog(self):
        with self._publish_lock:
            if self._stop_event.is_set():
                return
            now = time.monotonic()
            if now - (self._last_received or self._started) > 3.0:
                self._state = "stale"
                self._publish_status("stale: no input")
            elif self._state == "error" or (not self._last_result and self._state in ("loading", "processing")):
                self._publish_status(self._state)

    def _publish_status(self, status):
        self._publish([], None, self._last_frame, self._last_header, status)

    def _publish(self, objects, masks, frame, header, status):
        with self._publish_lock:
            if self._stop_event.is_set():
                return
            stamp = None
            if header is not None:
                stamp = header.stamp.sec + header.stamp.nanosec / 1e9
            self._sequence += 1
            frame_age = time.monotonic() - self._source_received if self._source_received else None
            preview = render_preview(frame, objects, masks, status=status,
                                     depth_enabled=self._depth_enabled,
                                     source_stamp=stamp, sequence=self._sequence, frame_age_s=frame_age)
            data = {"timestamp": time.time(), "source_timestamp": stamp,
                    "frame_id": header.frame_id if header else "",
                    "sequence": self._sequence, "status": status, "error": self._error,
                    "frame_age_s": frame_age,
                    "objects": objects}
            message = String()
            message.data = json.dumps(data, ensure_ascii=False, allow_nan=False)
            image = CompressedImage()
            if header is not None:
                image.header = copy.deepcopy(header)
            image.format, image.data = "jpeg", preview
            self._pub.publish(message)
            self._preview_pub.publish(image)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                msg, received = self._frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if time.monotonic() - received > 3.0:
                continue
            try:
                cv2 = load_cv2()
                frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Invalid JPEG input")
                with self._publish_lock:
                    self._last_frame, self._last_header = frame, copy.deepcopy(msg.header)
                    self._source_received = received
                    self._state = "loading" if self._plugin._model is None else "processing"
                objects, masks, error = self._plugin._infer(frame, self._confidence,
                                                          self._depth_enabled, self._stop_event)
                with self._publish_lock:
                    if self._stop_event.is_set():
                        break
                    self._error = error
                    # Never republish a completed inference as live after input has disappeared.
                    if time.monotonic() - self._last_received > 3.0:
                        self._state = "stale"
                        self._publish_status("stale: no input")
                        continue
                    self._state = "error" if error else "running"
                    self._publish(objects, masks, frame, msg.header, "depth_unavailable" if error else "ok")
                    self._detect_count += 1
                    self._last_result = time.monotonic()
            except Exception as exc:
                log.exception("[vop] frame failed")
                with self._publish_lock:
                    self._error, self._state = str(exc), "error"
                    self._publish_status("error")


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
        self._model_lock = threading.RLock()
        self._nodes_lock = threading.RLock()
        self._depth_enabled = plugin_cfg.get("depth_enabled", False)
        self._depth = None
        self._device = "cpu"
        self._nodes: dict[str, _VOPNode] = {}
        self._instance_configs: dict[str, dict] = {}  # per-instance config overrides

    def _get_all_classes(self) -> list[str]:
        """Merge base COCO-80 + global extra + all instance extra classes."""
        all_extra = set(self._extra_classes)
        with self._nodes_lock:
            configs = list(self._instance_configs.values())
        for cfg in configs:
            for c in cfg.get("classes") or []:
                all_extra.add(c)
        return self._base_classes + [c for c in sorted(all_extra) if c not in self._base_classes]

    def _sync_model_classes(self):
        with self._model_lock:
            if self._model is None:
                return
            classes = self._get_all_classes()
            self._model.model.cpu()
            try:
                self._model.set_classes(classes)
            finally:
                self._model.model.to(self._device)

    def _infer(self, frame, confidence, depth_enabled, cancelled):
        # ponytail: shared models serialize instances; use per-device workers if throughput requires it.
        with self._model_lock:
            if cancelled.is_set():
                return [], None, None
            self._ensure_model()
            if cancelled.is_set():
                return [], None, None
            result = self._model(frame, conf=confidence, verbose=False, device=self._device)[0]
            objects = extract_objects(result, frame.shape)
            if not depth_enabled or not objects:
                return objects, None, None
            try:
                if self._depth is None:
                    directory = os.path.join(os.environ.get("YOLO_MODEL_DIR", "/models"), "vop-depth")
                    self._depth = DepthEstimator(directory, self._device)
                depths, masks = self._depth.estimate(frame, [obj["bbox_xyxy"] for obj in objects])
                for obj, depth in zip(objects, depths):
                    obj["obstacle_depth"] = depth
                return objects, masks, None
            except Exception as exc:
                for obj in objects:
                    obj["obstacle_depth"] = unavailable("error")
                return objects, None, str(exc)

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

            load_cv2()

            from ultralytics import YOLO
            import torch

            # Determine device: prefer CUDA if available
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

            model_path = self._resolve_model_path()
            log.info(f"[vop] loading model: {model_path} (device={self._device})")
            model = YOLO(model_path)

            # Ensure CLIP weights are available locally before set_classes
            self._ensure_clip_weights()

            classes = self._get_all_classes()
            # set_classes on CPU (model loads on CPU by default), then move to GPU
            model.set_classes(classes)
            if self._device != "cpu":
                model.model.to(self._device)
            self._model = model
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

    def _config_for(self, key):
        return {"confidence": self._confidence, "fps": self._fps,
                "classes": self._extra_classes, "depth_enabled": self._depth_enabled,
                **self._instance_configs.get(key, {})}

    @staticmethod
    def _validate_config(cfg):
        if "depth_enabled" in cfg and type(cfg["depth_enabled"]) is not bool:
            raise ValueError("depth_enabled must be boolean")
        for key, low, high in (("confidence", 0, 1), ("fps", 1, 60)):
            if key in cfg:
                value = cfg[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                    raise ValueError(f"{key} must be between {low} and {high}")
                if key == "fps" and int(value) != value:
                    raise ValueError("fps must be an integer")
        if "classes" in cfg and (not isinstance(cfg["classes"], list) or
                any(not isinstance(c, str) or not c.strip() or len(c) > 100 for c in cfg["classes"])):
            raise ValueError("classes must be a list of nonempty strings (max 100 characters)")

    def _stop_nodes(self, instance_id):
        with self._nodes_lock:
            selected = [(k, n) for k, n in self._nodes.items() if not instance_id or k == instance_id]
        results = []
        for key, node in selected:
            result = node.stop()
            results.append(result)
            if result["state"] == "idle":
                self._dispose_node(key, node)
            else:
                with self._nodes_lock:
                    if not getattr(node, "_cleanup_started", False):
                        node._cleanup_started = True
                        threading.Thread(target=self._finish_stop, args=(key, node), daemon=True,
                                         name=f"vop_cleanup_{node.get_name()}").start()
        if any(r["state"] == "stopping" for r in results):
            return {"state": "stopping"}
        return {"state": "idle"}

    def _finish_stop(self, key, node):
        node._worker.join()
        self._dispose_node(key, node)

    def _dispose_node(self, key, node):
        with self._nodes_lock:
            if self._nodes.get(key) is not node:
                return
            self._executor.remove_node(node)
            node.destroy_node()
            node._state = "idle"
            del self._nodes[key]
            # The shared depth models can be released once the final instance is gone.
            if not self._nodes and self._model_lock.acquire(blocking=False):
                try:
                    self._depth = None
                finally:
                    self._model_lock.release()

    def get_tools(self):
        return TOOLS

    def dispatch(self, name, args):
        action = args.get("action", name)
        instance_id = args.get("instance_id", "")
        input_topic = args.get("input_topic") or next(iter(args.get("input_topics") or []), "")
        if action == "info":
            with self._nodes_lock:
                instances = {key: node.info() for key, node in self._nodes.items()}
                selected = instances.get(instance_id) if instance_id else next(iter(instances.values()), None)
                if selected:
                    input_topic = selected["input"]
                state = selected["state"] if selected else "idle"
                return {"name": "VideoObjectPerception", "manufacture": "Embodied",
                        "model": self._model_name, "state": state,
                        "phase": selected["phase"] if selected else "idle",
                        "error": selected["error"] if selected else None,
                        "base_classes_count": len(self._base_classes),
                        "total_classes": len(self._get_all_classes()), "instances": instances,
                        "topic_in": [{"topic": input_topic, "format": "image/jpeg"}] if input_topic else [],
                        "topic_out": [
                            {"topic": f"{input_topic}/objects", "format": "data/json", "desc": "物体识别与相对深度"},
                            {"topic": f"{input_topic}/objects/preview", "format": "image/jpeg", "desc": "VOP 分割与最近点预览"},
                        ] if input_topic else [],
                        "desc": "Open-vocabulary objects; optional uncalibrated mask-min depth"}
        if action == "start":
            if not isinstance(input_topic, str) or not input_topic.startswith("/"):
                raise ValueError("input_topic must be an absolute ROS image topic")
            key = instance_id or input_topic
            with self._nodes_lock:
                node = self._nodes.get(key)
                if node is None:
                    if any(n._input_topic == input_topic for n in self._nodes.values()):
                        raise ValueError("This input already has a VOP instance; stop it before rebinding")
                    cfg = self._config_for(key)
                    self._validate_config(cfg)
                    suffix = hashlib.sha256(key.encode()).hexdigest()[:12]
                    node = _VOPNode(input_topic, self, cfg, suffix)
                    try:
                        self._executor.add_node(node)
                    except Exception:
                        node.destroy_node()
                        raise
                    self._nodes[key] = node
                elif node._input_topic != input_topic:
                    raise ValueError("Stop the instance before changing its input")
            return node.start()
        if action == "stop":
            return self._stop_nodes(instance_id)
        if action in ("config", "set_classes"):
            fields = ("confidence", "fps", "classes", "depth_enabled")
            cfg = {k: args[k] for k in fields if k in args}
            if action == "set_classes":
                if "classes" not in cfg:
                    raise ValueError("classes is required")
                cfg = {"classes": cfg["classes"]}
            self._validate_config(cfg)
            if action == "config":
                result = self._stop_nodes(instance_id)
                if result["state"] == "stopping":
                    return {"status": "error", "error": "Inference is stopping; retry config after stop completes"}
            with self._nodes_lock:
                if instance_id:
                    self._instance_configs.setdefault(instance_id, {}).update(cfg)
                else:
                    for key, value in cfg.items():
                        setattr(self, "_extra_classes" if key == "classes" else f"_{key}", value)
            if "classes" in cfg:
                self._sync_model_classes()
            return {"status": "configured", "instance_id": instance_id, "config": cfg}
        return None
