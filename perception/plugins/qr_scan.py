"""Robot-independent, multi-instance JPEG QR perception card."""
from __future__ import annotations

import copy
import json
import math
import threading
import time
from datetime import datetime, timezone
from uuid import uuid4

from hashlib import sha256

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from utils.qr_decoder import QrDecoder, QrTracker
from utils.ros_lifecycle import dispose_node


class _QrStream:
    PREFIX = "qr_scan"

    def __init__(self, node, input_topic, output_topic, config=None, decoder=None):
        cfg = config or {}
        self.node = node
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.scan_hz = self._number(cfg, "scan_hz", 2, 0.2, 10)
        self.stale_s = self._number(cfg, "stale_after_s", 3, 0.5, 30)
        self.rearm_s = self._number(cfg, "rearm_after_s", 3, 0.5, 300)
        self.decoder = decoder
        self._lock = threading.RLock()
        self._lifecycle = threading.Lock()
        self._stop = threading.Event()
        self._thread = self._subscription = self._publisher = None
        self._frame = None
        self._sequence = 0
        self._tracker = QrTracker(self.rearm_s)
        self._session = uuid4().hex
        self._result = self._empty("idle")

    @staticmethod
    def _number(cfg, key, default, low, high):
        value = float(cfg.get(key, default))
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        return value

    def _empty(self, status, message=""):
        return {"status": status, "message": message, "session_id": self._session,
                "codes": [], "new_events": [], "frame_sequence": None,
                "frame_age_s": None, "frame_received_at": None,
                "processed_at": datetime.now(timezone.utc).isoformat()}

    def start(self):
        with self._lifecycle:
            if self._thread is not None and self._thread.is_alive():
                return
            if self.decoder is None:
                self.decoder = QrDecoder()
            from std_msgs.msg import String
            from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
            self._message_type = String
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            with self._lock:
                self._frame = None
                self._sequence = 0
                self._tracker = QrTracker(self.rearm_s)
                self._session = uuid4().hex
                self._result = self._empty("waiting_for_frame")
            self._stop.clear()
            try:
                # The node belongs to the perception executor.
                self._publisher = self.node.create_publisher(String, self.output_topic, qos)
                self._subscription = self.node.create_subscription(
                    CompressedImage, self.input_topic, self._on_frame, qos)
                self._thread = threading.Thread(target=self._run, name="perception-qr-scan", daemon=True)
                self._thread.start()
            except Exception:
                self._stop.set()
                self._destroy_io()
                raise

    def _destroy_io(self):
        if self._subscription is not None:
            self.node.destroy_subscription(self._subscription)
            self._subscription = None
        if self._publisher is not None:
            self._publisher = None

    def stop(self):
        with self._lifecycle:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=5)
                if self._thread.is_alive():
                    raise RuntimeError("QR worker has not stopped; retry stop before restarting")
            self._thread = None
            self._destroy_io()
            with self._lock:
                self._frame = None
                self._result = self._empty("idle")

    def _on_frame(self, message):
        if self._stop.is_set():
            return
        if str(message.format).lower() not in ("jpeg", "jpg", "image/jpeg"):
            self._reject_frame("Expected JPEG CompressedImage")
            return
        if not 0 < len(message.data) <= 5_000_000:
            self._reject_frame("Empty or oversized JPEG")
            return
        frame = {"data": bytes(message.data), "received_monotonic": time.monotonic(),
                 "received_at": datetime.now(timezone.utc).isoformat()}
        with self._lock:
            self._sequence += 1
            frame["frame_sequence"] = self._sequence
            self._frame = frame  # Keep only the newest image; never queue a backlog.

    def _reject_frame(self, message):
        with self._lock:
            self._sequence += 1
            self._frame = {"error": message, "frame_sequence": self._sequence,
                           "received_monotonic": time.monotonic(),
                           "received_at": datetime.now(timezone.utc).isoformat()}

    def process_frame(self, frame):
        age = max(0, time.monotonic() - frame["received_monotonic"])
        result = self._empty("no_qr")
        result.update(frame_sequence=frame["frame_sequence"], frame_age_s=age,
                      frame_received_at=frame["received_at"])
        if age > self.stale_s:
            result["status"] = "stale"
            return result
        try:
            if "error" in frame:
                raise ValueError(frame["error"])
            codes = self.decoder.decode(frame["data"])
            now = time.monotonic()
            result["frame_age_s"] = max(0, now - frame["received_monotonic"])
            if result["frame_age_s"] > self.stale_s:
                result["status"] = "stale"
                return result
            result.update(status="detected" if codes else "no_qr", codes=codes,
                          new_events=self._tracker.update(codes, now))
        except Exception as exc:
            result.update(status="decode_error", message=f"{type(exc).__name__}: {str(exc)[:160]}")
        return result

    def _run(self):
        last_sequence = None
        while not self._stop.is_set():
            started = time.monotonic()
            with self._lock:
                frame = self._frame
            try:
                if frame is None:
                    result = self._empty("waiting_for_frame")
                elif frame["frame_sequence"] != last_sequence:
                    result = self.process_frame(frame)
                    last_sequence = frame["frame_sequence"]
                elif started - frame["received_monotonic"] > self.stale_s:
                    result = self._empty("stale")
                    result.update(frame_sequence=frame["frame_sequence"],
                                  frame_received_at=frame["received_at"],
                                  frame_age_s=started - frame["received_monotonic"])
                else:
                    result = None
                if result is not None and not self._stop.is_set():
                    message = self._message_type()
                    message.data = json.dumps(result, ensure_ascii=False, allow_nan=False)
                    with self._lock:
                        self._result = result
                    self._publisher.publish(message)
            except Exception as exc:
                with self._lock:
                    self._result = self._empty("error", f"{type(exc).__name__}: {str(exc)[:160]}")
            self._stop.wait(max(0, 1 / self.scan_hz - (time.monotonic() - started)))

    def info(self):
        with self._lock:
            result = copy.deepcopy(self._result)
            frame = self._frame
            if frame is not None and result["frame_sequence"] == frame["frame_sequence"]:
                result["frame_age_s"] = max(0, time.monotonic() - frame["received_monotonic"])
                if result["frame_age_s"] > self.stale_s:
                    result.update(status="stale", codes=[], new_events=[])
        running = self._thread is not None and self._thread.is_alive() and not self._stop.is_set()
        return {"state": "running" if running else "idle", "input_topic": self.input_topic,
                "topic_in": [{"topic": self.input_topic, "format": "image/jpeg"}],
                "topic_out": [{"topic": self.output_topic, "format": "data/json"}], **result}


TOOLS = [{
    'name': 'qr_scan', 'type': 'processor', 'multiInstance': True,
    'description': '二维码读取：接入任意机器人的 JPEG 图像，输出内容、像素角点及去重事件；不执行二维码内容。',
    'inputSchema': {'type': 'object', 'properties': {
        'action': {'type': 'string', 'enum': ['start', 'stop', 'info', 'read', 'config']},
        'input_topic': {'type': 'string', 'description': '连接的 image/jpeg 图像话题'},
    }, 'required': ['action'], 'x-action-params': {
        'start': {'params': ['input_topic'], 'description': '启动已连接的相机输入'},
        'stop': {'params': [], 'description': '停止扫码'},
        'info': {'params': ['input_topic'], 'description': '查看状态及输入输出话题'},
        'read': {'params': ['input_topic'], 'description': '读取最新结果，不触发额外扫描'},
        'config': {'params': [], 'description': '更新本实例配置；运行中的实例需重新 start'},
    }},
    'configSchema': {'type': 'object', 'properties': {
        'scan_hz': {'type': 'number', 'minimum': 0.2, 'maximum': 10, 'default': 2, 'scope': 'instance'},
        'stale_after_s': {'type': 'number', 'minimum': 0.5, 'maximum': 30, 'default': 3, 'scope': 'instance'},
        'rearm_after_s': {'type': 'number', 'minimum': 0.5, 'maximum': 300, 'default': 3, 'scope': 'instance'},
    }},
    'topic_in': [{'format': 'image/jpeg'}], 'topic_out': [{'format': 'data/json'}],
}]


class QrScanPlugin:
    PREFIX = 'qr_scan'

    def __init__(self, config, executor):
        self._config = config or {}
        self._executor = executor
        self._streams = {}
        self._configs = {}
        self._lock = threading.RLock()

    def get_tools(self):
        return copy.deepcopy(TOOLS)

    @staticmethod
    def _output(key):
        token = sha256(key.encode()).hexdigest()[:16]
        return f'/perception/qr_scan/instance_{token}/result'

    def _stop(self, key):
        stream = self._streams.get(key)
        if stream is not None:
            stream.stop()
            dispose_node(self._executor, stream.node, label='qr_scan')
            del self._streams[key]

    def stop(self):
        with self._lock:
            for key in list(self._streams):
                self._stop(key)

    def dispatch(self, name, args):
        action = args.get('action', 'info') if name == self.PREFIX else name
        instance_id = str(args.get('instance_id') or '')
        topic = str(args.get('input_topic') or '')
        key = instance_id or topic
        with self._lock:
            if action == 'stop':
                if key:
                    self._stop(key)
                else:
                    self.stop()
                return {'state': 'idle'}
            if action == 'config':
                if not key:
                    return {'state': 'error', 'error': 'instance_id or input_topic required'}
                cfg = args.get('config', {})
                if not isinstance(cfg, dict):
                    return {'state': 'error', 'error': 'config must be an object'}
                merged = {**self._config, **self._configs.get(key, {}), **cfg}
                try:
                    # Validate before mutating a running instance.
                    _QrStream(None, topic, self._output(key), merged)
                except (TypeError, ValueError) as exc:
                    return {'state': 'error', 'error': str(exc)}
                self._stop(key)
                self._configs[key] = merged
                return {'state': 'idle', 'config': merged}
            if action == 'start':
                if not topic.startswith('/') or topic == '/' or any(c.isspace() for c in topic):
                    return {'state': 'error', 'error': 'Connect an absolute image/jpeg input_topic first'}
                cfg = {**self._config, **self._configs.get(key, {})}
                current = self._streams.get(key)
                if current is not None and current.input_topic != topic:
                    self._stop(key)
                    current = None
                if current is None:
                    node = None
                    try:
                        node = Node('qr_scan_' + sha256(key.encode()).hexdigest()[:16])
                        stream = _QrStream(node, topic, self._output(key), cfg)
                        self._executor.add_node(node)
                        self._streams[key] = stream
                        stream.start()
                    except Exception as exc:
                        if key in self._streams:
                            self._stop(key)
                        elif node is not None:
                            dispose_node(self._executor, node, label='qr_scan')
                        return {'state': 'error', 'error': str(exc)[:256]}
                return self._streams[key].info()
            if action in ('info', 'read'):
                if key:
                    if key in self._streams:
                        return self._streams[key].info()
                    return {'state': 'idle', 'status': 'idle', 'codes': [], 'new_events': [],
                            'input_topic': topic,
                            'topic_in': [{'topic': topic, 'format': 'image/jpeg'}] if topic else [],
                            'topic_out': [{'topic': self._output(key), 'format': 'data/json'}]}
                return {'state': 'running' if self._streams else 'idle',
                        'instances': {k: s.info() for k, s in self._streams.items()},
                        'topic_out': [{'topic': s.output_topic, 'format': 'data/json'}
                                      for s in self._streams.values()]}
            return {'state': 'error', 'error': f'Unknown action: {action}'}
