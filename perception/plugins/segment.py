"""Independent semantic segmentation cards share the YOLOE engine and a forward pass per camera."""
import hashlib
import json
import math
import re
import threading
import time

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from plugins.semantic_masks import decode_camera_jpeg, overlay, semantic_maps
from utils.qos import CAMERA_QOS

DEFAULT_CARDS = {"person": ["person"], "car": ["car"], "chair": ["chair"],
                 "traffic": ["bicycle", "car", "motorcycle", "bus", "truck"]}
FORMATS = [("mask", "image/png", "Lossless per-pixel class IDs, 0 = background"),
           ("classes", "data/json", "Class IDs, pixel counts, image size and capture stamp"),
           ("overlay", "image/jpeg", "Colour preview on the input image")]


def _ports(topic, name, key, preview):
    suffix = hashlib.sha256(key.encode()).hexdigest()[:16]
    root = f"{topic}/segment/{name}/i_{suffix}" if topic else ""
    return [{"topic": root + "/" + field if root and (field != "overlay" or preview) else "",
             "format": fmt, "desc": desc} for field, fmt, desc in FORMATS]


def _packet(header, fmt, data):
    import cv2
    ok, encoded = cv2.imencode("." + fmt, data)
    if not ok:
        raise ValueError(f"Could not encode segmentation {fmt}")
    msg = CompressedImage()
    msg.header = header
    msg.format = "jpeg" if fmt == "jpg" else fmt
    msg.data = encoded.tobytes()
    return msg


class _Camera(Node):
    def __init__(self, topic, executor, model_provider, confidence, fps):
        super().__init__("segment_" + hashlib.sha256(topic.encode()).hexdigest()[:16])
        self.executor_ref = executor
        self.topic = topic
        self.model_provider = model_provider
        self.confidence = confidence
        self.period = 1 / fps
        self.cv = threading.Condition()
        self.targets = {}
        self.latest = None
        self.closed = False
        self.ready = False
        self.error = ""
        self.load_failed = False
        self.vocabulary = None
        self.worker = threading.Thread(target=self._run, name="semantic-masks", daemon=True)
        try:
            self.create_subscription(CompressedImage, topic, self._receive, CAMERA_QOS)
            executor.add_node(self)
            self.worker.start()
        except Exception:
            executor.remove_node(self)
            self.destroy_node()
            raise

    def add(self, key, name, labels, ports):
        with self.cv:
            pubs = []
            try:
                for i, port in enumerate(ports):
                    msg_type = String if i == 1 else CompressedImage
                    pubs.append(self.create_publisher(msg_type, port["topic"], 2) if port["topic"] else None)
            except Exception:
                for pub in pubs:
                    if pub is not None:
                        self.destroy_publisher(pub)
                raise
            self.targets[key] = (name, labels, pubs)
            self.cv.notify_all()

    def remove(self, key):
        with self.cv:
            _, _, pubs = self.targets.pop(key)
            for pub in pubs:
                if pub is not None:
                    self.destroy_publisher(pub)
            return not self.targets

    def close(self):
        # Loading TensorRT can take minutes. A stopped worker can finish the shared
        # load but cannot touch a removed node or publish a late frame.
        with self.cv:
            self.closed = True
            self.latest = None
            self.cv.notify_all()
            self.executor_ref.remove_node(self)
            self.destroy_node()

    def _receive(self, msg):
        with self.cv:
            if not self.closed:
                self.latest = msg
                self.cv.notify()

    def _run(self):
        try:
            engine, names = self.model_provider()
            with self.cv:
                if self.closed:
                    return
                self.vocabulary = set(names)
                self.ready = True
        except Exception as exc:
            with self.cv:
                self.error = str(exc)
                self.load_failed = True
            return
        next_frame = 0.
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.closed or (self.latest is not None and self.targets))
                if self.closed:
                    return
                delay = next_frame - time.monotonic()
                if delay > 0:
                    self.cv.wait(timeout=delay)
                    continue
                msg, self.latest = self.latest, None
                targets = dict(self.targets)
            next_frame = time.monotonic() + self.period
            try:
                self._process(engine, names, targets, msg)
            except Exception as exc:
                with self.cv:
                    self.error = str(exc)

    def _process(self, engine, names, targets, msg):
        import numpy as np
        image = decode_camera_jpeg(bytes(msg.data))
        presets = {name: labels for name, labels, _ in targets.values() if set(labels) <= self.vocabulary}
        if not presets:
            return
        outputs, meta = engine.infer(image)
        masks = semantic_maps(outputs, meta, engine.input_size, names, presets, self.confidence)
        packets = {}
        for name, labels in presets.items():
            mask = masks[name]
            counts = np.bincount(mask.ravel(), minlength=len(labels) + 1)
            info = {"width": mask.shape[1], "height": mask.shape[0], "frame_id": msg.header.frame_id,
                    "stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec},
                    "classes": [{"id": i, "name": label, "pixels": int(counts[i])}
                                for i, label in enumerate(["background"] + labels)]}
            preview = any(n == name and pubs[2] is not None for n, _, pubs in targets.values())
            report = String()
            report.data = json.dumps(info)
            packets[name] = (_packet(msg.header, "png", mask), report,
                             _packet(msg.header, "jpg", overlay(image, mask)) if preview else None)
        with self.cv:
            if self.closed:
                return
            for key, target in targets.items():
                if self.targets.get(key) is not target or target[0] not in packets:
                    continue
                for pub, packet in zip(target[2], packets[target[0]]):
                    if pub is not None:
                        pub.publish(packet)
            self.error = ""


class SegmentationPlugin:
    PREFIX = "segment"

    def __init__(self, cfg, executor, model_provider, model_load_message):
        self.executor = executor
        self.provider = model_provider
        self.model_load_message = model_load_message
        self.confidence, self.fps = float(cfg.get("confidence", .30)), float(cfg.get("fps", 4))
        if not (0 < self.confidence <= 1 and 0 < self.fps <= 30 and math.isfinite(self.fps)):
            raise ValueError("segment confidence must be in (0,1] and fps in (0,30]")
        self.cards = cfg.get("cards", DEFAULT_CARDS)
        if not isinstance(self.cards, dict) or not self.cards:
            raise ValueError("segment.cards must be a mapping of card names to model labels")
        for name, labels in self.cards.items():
            if (not isinstance(name, str) or not re.fullmatch("[a-z][a-z0-9_]*", name)
                    or not isinstance(labels, list) or not 1 <= len(labels) <= 255
                    or any(not isinstance(v, str) or not v.strip() for v in labels) or len(set(labels)) != len(labels)):
                raise ValueError(f"Invalid semantic segmentation preset {name!r}")
        self.preview = cfg.get("publish_overlay", True)
        self.nodes, self.instances = {}, {}
        self.lock = threading.RLock()

    def get_tools(self):
        return [{"name": name, "type": "processor", "multiInstance": True,
                 "description": "逐像素语义分割 · " + ", ".join(labels) + "。输出类别掩膜 PNG、像素统计和摄像头叠加图。",
                 "inputSchema": {"type": "object", "required": ["action"], "properties": {
                     "action": {"type": "string", "enum": ["start", "stop", "info"]},
                     "input_topic": {"type": "string"}, "instance_id": {"type": "string"}}},
                 "topic_in": [{"format": "image/jpeg", "desc": "Camera JPEG"}],
                 "topic_out": _ports("", name, "", self.preview)} for name, labels in self.cards.items()]

    def _model(self):
        engine, names = self.provider()
        if not names:
            raise ValueError("Segmentation needs the frozen engine vocabulary")
        return engine, names

    def _remove(self, key):
        topic = self.instances.pop(key, "")
        if topic and self.nodes[topic].remove(key):
            self.nodes.pop(topic).close()

    def shutdown(self):
        with self.lock:
            for key in list(self.instances):
                self._remove(key)

    def dispatch(self, name, args):
        if name not in self.cards:
            return None
        action = args.get("action", "info")
        topic = args.get("input_topic") or ""
        if topic:
            from rclpy.validate_full_topic_name import validate_full_topic_name
            validate_full_topic_name(topic)
        key = (name, args.get("instance_id") or topic or "default")
        with self.lock:
            if action == "stop":
                if not args.get("instance_id") and not topic:
                    for entry in [k for k in self.instances if k[0] == name]:
                        self._remove(entry)
                else:
                    self._remove(key)
            elif action == "start":
                if not topic:
                    raise ValueError("Wire a camera JPEG to the segmentation card")
                if self.instances.get(key) != topic:
                    node = self.nodes.get(topic)
                    if node is None:
                        node = _Camera(topic, self.executor, self._model, self.confidence, self.fps)
                        self.nodes[topic] = node
                    try:
                        node.add(key, name, self.cards[name], _ports(topic, name, key[1], self.preview))
                    except Exception:
                        if not node.targets:
                            self.nodes.pop(topic).close()
                        raise
                    self._remove(key)
                    self.instances[key] = topic
            elif action != "info":
                raise ValueError("Unknown segmentation card action")
            active = self.nodes.get(self.instances.get(key))
            state = "idle"
            message = ""
            if active:
                with active.cv:
                    state = "error" if active.load_failed else ("running" if active.ready else "loading")
                    message = active.error
                    if state == "loading":
                        message = self.model_load_message() or "Preparing segmentation model..."
                    if active.ready:
                        missing = set(self.cards[name]) - active.vocabulary
                        if missing:
                            state, message = "error", "Classes absent from frozen YOLOE vocabulary: " + ", ".join(sorted(missing))
            wired = self.instances.get(key) or topic
            return {"state": state, "message": message, "classes": self.cards[name],
                    "topic_in": [{"topic": wired, "format": "image/jpeg", "desc": "Camera JPEG"}],
                    "topic_out": _ports(wired, name, key[1], self.preview)}
