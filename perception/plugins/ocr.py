#!/usr/bin/env python3
"""
plugins/ocr.py — OCRPlugin: OCR 文字识别封装。

订阅 image/jpeg topic，持续进行 OCR 识别并发布结果到 ROS2 topic。
参考 asr.py 架构设计。
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from plugins.ocr_runtime import (
    DEFAULT_CROP_REFINEMENT_ENABLED,
    DEFAULT_CROP_REFINEMENT_MIN_GAIN,
    DEFAULT_CROP_REFINEMENT_MIN_SCORE,
    DEFAULT_CROP_REFINEMENT_PROFILES,
    DEFAULT_DET_BOX_THRESH,
    DEFAULT_DET_THRESH,
    DEFAULT_DET_UNCLIP_RATIO,
    DEFAULT_EMPTY_RESULT_RETRY_DET_BOX_THRESH,
    DEFAULT_EMPTY_RESULT_RETRY_DET_THRESH,
    DEFAULT_EMPTY_RESULT_RETRY_ENABLED,
    DEFAULT_MAX_SIDE_LEN,
    DEFAULT_REC_MIN_SCORE,
    RapidOCRAdapter,
    recognize_to_payload,
)

log = logging.getLogger(__name__)

DEFAULT_OCR_MODEL_DIR = "/models/ocr/ppocrv6-small-trt"
_ERROR_LOG_INTERVAL_SECONDS = 10.0

_CAMERA_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

_RESULT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

TOOLS = [
    {
        "name": "ocr",
        "type": "processor",
        "multiInstance": True,
        "description": "OCR — recognize text in camera feed via image topic subscription",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "config"],
                    "description": "Action to perform"
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 image topic to subscribe (e.g. /hostname/camera/rgb, required for action=start)"
                },
            },
            "required": ["action"]
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "description": "OCR 模型或 TensorRT engine 目录", "scope": "shared"},
                "device_id": {"type": "integer", "minimum": 0, "default": 0, "scope": "shared"},
                "use_angle_cls": {"type": "boolean", "default": True, "description": "启用 0/180 度文字方向分类", "scope": "shared"},
                "provider": {"type": "string", "enum": ["rapidocr"], "description": "OCR 运行时", "scope": "shared"},
                "language": {"type": "string", "description": "默认语言", "default": "zh", "scope": "instance"},
                "max_side_len": {"type": "integer", "minimum": 32, "default": DEFAULT_MAX_SIDE_LEN, "description": "检测输入最长边", "scope": "shared"},
                "det_thresh": {"type": "number", "minimum": 0, "maximum": 1, "default": DEFAULT_DET_THRESH, "description": "DB 文本像素阈值", "scope": "shared"},
                "det_box_thresh": {"type": "number", "minimum": 0, "maximum": 1, "default": DEFAULT_DET_BOX_THRESH, "description": "DB 文本框阈值", "scope": "shared"},
                "det_unclip_ratio": {"type": "number", "exclusiveMinimum": 0, "default": DEFAULT_DET_UNCLIP_RATIO, "description": "DB 文本框扩张比例", "scope": "shared"},
                "rec_min_score": {"type": "number", "minimum": 0, "maximum": 1, "default": DEFAULT_REC_MIN_SCORE, "description": "识别结果最低置信度", "scope": "shared"},
                "enable_preprocess": {"type": "boolean", "default": True, "description": "启用 OCR 图像预处理", "scope": "shared"},
                "crop_refinement": {
                    "type": "object",
                    "description": "TensorRT 低置信文本框二次裁剪识别",
                    "scope": "shared",
                    "default": {
                        "enabled": DEFAULT_CROP_REFINEMENT_ENABLED,
                        "min_score": DEFAULT_CROP_REFINEMENT_MIN_SCORE,
                        "min_gain": DEFAULT_CROP_REFINEMENT_MIN_GAIN,
                        "min_text_length": 2,
                        "profiles": list(DEFAULT_CROP_REFINEMENT_PROFILES),
                    },
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "min_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "min_gain": {"type": "number", "minimum": 0},
                        "min_text_length": {"type": "integer", "minimum": 1},
                        "profiles": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["prefix_65", "upper_center", "upper_tight"],
                            },
                        },
                    },
                },
                "empty_result_retry": {
                    "type": "object",
                    "description": "TensorRT 主流程空输出时复用检测图进行低阈值后处理",
                    "scope": "shared",
                    "default": {
                        "enabled": DEFAULT_EMPTY_RESULT_RETRY_ENABLED,
                        "det_thresh": DEFAULT_EMPTY_RESULT_RETRY_DET_THRESH,
                        "det_box_thresh": DEFAULT_EMPTY_RESULT_RETRY_DET_BOX_THRESH,
                    },
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "det_thresh": {"type": "number", "minimum": 0, "maximum": 1},
                        "det_box_thresh": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
                "min_interval_ms": {"type": "integer", "minimum": 0, "default": 0, "description": "帧处理最小间隔(ms)，限制 GPU 占用，0=不限", "scope": "instance"},
            },
            "required": ["provider"]
        },
        "topic_in":  [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [{"format": "data/json",  "desc": "OCR result with text boxes"}],
    }
]


# ── OCR Adapters ──────────────────────────────────────────────────────────────

def _ocr_output_topic(input_topic: str) -> str:
    return f"{input_topic}/ocr"


def _adapter_options(cfg: dict) -> dict:
    return {
        "model_dir": str(cfg.get("model_dir", DEFAULT_OCR_MODEL_DIR)),
        "device_id": int(cfg.get("device_id", 0)),
        "use_angle_cls": bool(cfg.get("use_angle_cls", True)),
        "max_side_len": int(cfg.get("max_side_len", DEFAULT_MAX_SIDE_LEN)),
        "rec_min_score": float(
            cfg.get("rec_min_score", DEFAULT_REC_MIN_SCORE)
        ),
        "enable_preprocess": bool(cfg.get("enable_preprocess", True)),
        "det_thresh": float(cfg.get("det_thresh", DEFAULT_DET_THRESH)),
        "det_box_thresh": float(
            cfg.get("det_box_thresh", DEFAULT_DET_BOX_THRESH)
        ),
        "det_unclip_ratio": float(
            cfg.get("det_unclip_ratio", DEFAULT_DET_UNCLIP_RATIO)
        ),
        "crop_refinement": dict(cfg.get("crop_refinement") or {}),
        "empty_result_retry": dict(cfg.get("empty_result_retry") or {}),
    }


def _adapter_signature(cfg: dict) -> tuple:
    provider = cfg.get('provider', 'rapidocr')
    return provider, _adapter_options(cfg)


def _build_ocr_adapter(cfg: dict) -> RapidOCRAdapter:
    """根据配置创建 OCR 适配器"""
    provider = cfg.get('provider', 'rapidocr')
    if provider != 'rapidocr':
        raise ValueError(f"unsupported OCR provider: {provider}")
    return RapidOCRAdapter(**_adapter_options(cfg))


# ── ROS2 Node (订阅模式) ───────────────────────────────────────────────────────

class _OCRNode(Node):
    """订阅 image/jpeg topic，持续进行 OCR 识别"""

    def __init__(self, input_topic: str, adapter: RapidOCRAdapter, language: str = "zh",
                 node_suffix: str = '', min_interval: float = 0.0):
        node_name = f"ocr_{node_suffix}" if node_suffix else "ocr"
        super().__init__(node_name)

        self._input_topic = input_topic
        self._output_topic = _ocr_output_topic(input_topic)
        self._adapter = adapter
        self._language = language
        # 帧处理最小间隔（秒）：限制 GPU 占用，0 = 不限
        self._min_interval = max(0.0, float(min_interval))
        self.state = "idle"

        self._sub = None
        self._pub = self.create_publisher(String, self._output_topic, _RESULT_QOS)

        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stop_event.set()
        self._generation = 0
        self._worker_threads: list[threading.Thread] = []
        self._last_error_log_at: float | None = None
        log.info(f"[ocr] node created: subscribing={self._input_topic}, publishing={self._output_topic}")

    def start(self) -> dict:
        if self.state == "running":
            return self._status_dict()

        if not self._adapter:
            raise RuntimeError("OCR adapter not configured")

        self._generation += 1
        generation = self._generation
        stop_event = threading.Event()
        frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = stop_event
        self._frame_queue = frame_queue
        if self._sub is None:
            self._sub = self.create_subscription(
                CompressedImage, self._input_topic, self._image_cb, _CAMERA_QOS
            )
        self.state = "running"
        self._worker_threads = [
            thread for thread in self._worker_threads if thread.is_alive()
        ]
        self._worker_thread = threading.Thread(
            target=self._worker,
            args=(generation, stop_event, frame_queue),
            daemon=True,
        )
        self._worker_threads.append(self._worker_thread)
        self._worker_thread.start()

        log.info(f"[ocr] started: {self._input_topic} → {self._output_topic}")
        return self._status_dict()

    def stop(self) -> dict:
        self.state = "idle"
        self._stop_event.set()
        while True:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        deadline = time.monotonic() + 3.0
        for thread in self._worker_threads:
            if thread.is_alive():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
        self._worker_threads = [
            thread for thread in self._worker_threads if thread.is_alive()
        ]
        if self._worker_threads:
            log.warning(
                "[ocr] %d worker(s) still stopping after timeout: %s",
                len(self._worker_threads),
                self._input_topic,
            )

        log.info(f"[ocr] stopped: {self._input_topic}")
        return {"state": "idle"}

    @property
    def worker_alive(self) -> bool:
        return any(thread.is_alive() for thread in self._worker_threads)

    def _image_cb(self, msg: CompressedImage):
        """接收图片帧，放入队列"""
        stop_event = self._stop_event
        frame_queue = self._frame_queue
        if self.state != "running" or stop_event.is_set():
            return
        image_data = bytes(msg.data)
        try:
            frame_queue.put_nowait((image_data, time.time()))
        except queue.Full:
            log.debug("[ocr] frame queue full, dropping old frame")
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                frame_queue.put_nowait((image_data, time.time()))
            except queue.Full:
                pass

    def _is_generation_active(
        self, generation: int, stop_event: threading.Event
    ) -> bool:
        return (
            self.state == "running"
            and self._generation == generation
            and self._stop_event is stop_event
            and not stop_event.is_set()
        )

    def _worker(
        self,
        generation: int,
        stop_event: threading.Event,
        frame_queue: queue.Queue,
    ):
        """后台工作线程：从队列取图片进行 OCR"""
        while not stop_event.is_set():
            try:
                image_bytes, ts = frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            t_start = time.time()
            payload = recognize_to_payload(
                self._adapter, image_bytes, self._language, ts
            )
            if not self._is_generation_active(generation, stop_event):
                continue
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self._pub.publish(msg)
            if "error" in payload:
                now = time.monotonic()
                if (
                    self._last_error_log_at is None
                    or now - self._last_error_log_at >= _ERROR_LOG_INTERVAL_SECONDS
                ):
                    log.error("[ocr] recognition error: %s", payload["error"])
                    self._last_error_log_at = now
                else:
                    log.debug("[ocr] recognition error: %s", payload["error"])
            else:
                log.debug(
                    "[ocr] published result to %s: %d items",
                    self._output_topic,
                    len(payload["items"]),
                )

            # 限帧：距上一帧开始不足 min_interval 则等待（降低 GPU 占用）
            if self._min_interval > 0:
                remaining = self._min_interval - (time.time() - t_start)
                if remaining > 0:
                    stop_event.wait(remaining)

    def _status_dict(self) -> dict:
        return {
            "state": self.state,
            "topic_in": [{"topic": self._input_topic, "format": "image/jpeg", "desc": "image input"}],
            "topic_out": [{"topic": self._output_topic, "format": "data/json", "desc": "OCR result"}],
        }


# ── Plugin ────────────────────────────────────────────────────────────────────

class OCRPlugin:
    PREFIX = "ocr"

    def __init__(self, plugin_cfg: dict, executor):
        self._plugin_cfg = dict(plugin_cfg)
        self._language = plugin_cfg.get('language', 'zh')
        self._nodes: dict[str, _OCRNode] = {}
        self._instance_configs: dict[str, dict] = {}
        self._retired_nodes: list[_OCRNode] = []
        self._executor = executor
        self._lifecycle_lock = threading.RLock()

        self._adapter: RapidOCRAdapter | None = None
        log.info(
            f"[ocr] plugin init: provider={plugin_cfg.get('provider')}, "
            f"language={self._language}"
        )

    def get_tools(self) -> list:
        return TOOLS

    def _ensure_adapter(self) -> RapidOCRAdapter:
        if self._adapter is None:
            self._adapter = _build_ocr_adapter(self._plugin_cfg)
        return self._adapter

    def _retire_node(self, node_key: str) -> dict:
        node = self._nodes.pop(node_key)
        result = node.stop()
        self._executor.remove_node(node)
        if node not in self._retired_nodes:
            self._retired_nodes.append(node)
        log.info(
            f"[ocr] node retired: {node_key} | nodes={len(self._nodes)} "
            f"retired={len(self._retired_nodes)}"
        )
        return result

    def _configure_node(self, node: _OCRNode, instance_id: str) -> None:
        cfg = {**self._plugin_cfg, **self._instance_configs.get(instance_id, {})}
        node._adapter = self._adapter
        node._language = cfg.get("language", self._language)
        node._min_interval = max(
            0.0, float(cfg.get("min_interval_ms", 0)) / 1000.0
        )

    @staticmethod
    def _unique_objects(values) -> list:
        unique = []
        seen = set()
        for value in values:
            if value is None:
                continue
            identity = id(value)
            if identity not in seen:
                seen.add(identity)
                unique.append(value)
        return unique

    def prepare_shutdown(self) -> None:
        with self._lifecycle_lock:
            nodes = self._unique_objects(
                [*self._nodes.values(), *self._retired_nodes]
            )
            for node in nodes:
                node.stop()

    def destroy_nodes(self) -> None:
        with self._lifecycle_lock:
            nodes = self._unique_objects(
                [*self._nodes.values(), *self._retired_nodes]
            )
            adapter = self._adapter
            for node in nodes:
                node.destroy_node()
            self._nodes.clear()
            self._retired_nodes.clear()
            self._adapter = None
            if adapter is not None:
                close = getattr(adapter, "close", None)
                if callable(close):
                    close()

    def dispatch(self, name: str, args: dict) -> dict | None:
        with self._lifecycle_lock:
            return self._dispatch_locked(name, args)

    def _dispatch_locked(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == "ocr" else name
        instance_id = args.get("instance_id", "")

        if action == "info":
            input_topic = args.get("input_topic", "")

            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                return {
                    "name": "OCR", "manufacture": "Embodied", "model": "ocr",
                    "state": node.state,
                    "topic_in": [{"topic": node._input_topic, "format": "image/jpeg", "desc": ""}],
                    "topic_out": [{"topic": node._output_topic, "format": "data/json", "desc": ""}],
                    "desc": "OCR service — extracts text from images",
                }

            if instance_id:
                inferred_out = f"{input_topic}/ocr" if input_topic else ""
                return {
                    "name": "OCR", "manufacture": "Embodied", "model": "ocr",
                    "state": "idle",
                    "topic_in": [{"topic": input_topic, "format": "image/jpeg", "desc": ""}] if input_topic else [],
                    "topic_out": [{"topic": inferred_out, "format": "data/json", "desc": ""}] if inferred_out else [],
                    "desc": "OCR service — extracts text from images",
                }

            # 聚合所有实例信息
            if self._nodes:
                topics_in = [{"topic": n._input_topic, "format": "image/jpeg", "desc": ""} for n in self._nodes.values()]
                topics_out = [{"topic": n._output_topic, "format": "data/json", "desc": ""} for n in self._nodes.values()]
                states = list(set(n.state for n in self._nodes.values()))
                state = "running" if "running" in states else states[0] if states else "idle"
            else:
                inferred_out = f"{input_topic}/ocr" if input_topic else ""
                topics_in = [{"topic": input_topic, "format": "image/jpeg", "desc": ""}]
                topics_out = [{"topic": inferred_out, "format": "data/json", "desc": ""}]
                state = "idle"

            return {
                "name": "OCR", "manufacture": "Embodied", "model": "ocr",
                "state": state,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "OCR service — extracts text from images",
            }

        elif action == "start":
            input_topic = args.get("input_topic")
            if not input_topic:
                raise ValueError("input_topic is required for start action")

            node_key = instance_id or input_topic

            existing = self._nodes.get(node_key)
            if existing is not None and existing._input_topic != input_topic:
                self._retire_node(node_key)

            adapter = self._ensure_adapter()
            if node_key not in self._nodes:
                cfg = {
                    **self._plugin_cfg,
                    **self._instance_configs.get(instance_id, {}),
                }

                node = _OCRNode(
                    input_topic,
                    adapter,
                    cfg.get("language", self._language),
                    node_suffix=node_key.replace('/', '_').replace('-', '_'),
                    min_interval=float(cfg.get('min_interval_ms', 0)) / 1000.0,
                )
                self._executor.add_node(node)
                self._nodes[node_key] = node
            elif self._nodes[node_key].state != "running":
                self._configure_node(self._nodes[node_key], instance_id)

            return self._nodes[node_key].start()

        elif action == "stop":
            if instance_id and instance_id in self._nodes:
                return self._nodes[instance_id].stop()
            elif not instance_id and self._nodes:
                for node in self._nodes.values():
                    node.stop()
                return {"state": "idle"}
            return {"state": "idle"}

        elif action == "config":
            cfg = {k: v for k, v in args.items() if k not in ('action', 'instance_id') and v is not None and v != ''}

            if instance_id:
                shared = set(cfg) - {"language", "min_interval_ms"}
                if shared:
                    raise ValueError(
                        "OCR inference settings are shared: "
                        + ", ".join(sorted(shared))
                    )
                previous = self._instance_configs.get(instance_id, {})
                self._instance_configs[instance_id] = {**previous, **cfg}
                if instance_id in self._nodes:
                    node = self._nodes[instance_id]
                    node.stop()
                    self._configure_node(node, instance_id)
                return {"status": "configured", "instance_id": instance_id}
            else:
                updated_cfg = {**self._plugin_cfg, **cfg}
                rebuild = (
                    _adapter_signature(updated_cfg)
                    != _adapter_signature(self._plugin_cfg)
                )
                previous_adapter = self._adapter
                replacement = (
                    None if rebuild else previous_adapter
                )
                for node in self._nodes.values():
                    node.stop()
                if rebuild and previous_adapter is not None and any(
                    node.worker_alive for node in self._nodes.values()
                ):
                    raise RuntimeError("OCR workers did not stop for reconfiguration")
                self._adapter = replacement
                self._plugin_cfg = updated_cfg
                self._language = updated_cfg.get('language', self._language)
                for node_key, node in self._nodes.items():
                    self._configure_node(node, node_key)
                if rebuild:
                    close = getattr(previous_adapter, "close", None)
                    if callable(close):
                        close()
                return {
                    "status": "configured",
                    "adapter_loaded": self._adapter is not None,
                    "reused": previous_adapter is not None and not rebuild,
                }

        return None
