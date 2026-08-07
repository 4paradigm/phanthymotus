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
from abc import ABC, abstractmethod
from typing import Optional

import rclpy
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
    normalize_rapidocr_output,
    recognize_to_payload,
)

log = logging.getLogger(__name__)


def _resource_snapshot() -> str:
    """轻量资源快照：RSS + 线程数（用于定位服务慢性死亡原因）"""
    rss_mb = -1.0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    rss_mb = int(line.split()[1]) / 1024.0
                    break
    except OSError:
        pass
    return f"rss={rss_mb:.0f}MB threads={threading.active_count()}"

DEFAULT_OCR_BACKEND = "tensorrt"
DEFAULT_OCR_FALLBACK_BACKEND = "mnn"
DEFAULT_OCR_MODEL_DIR = (
    "/models/ocr/ppocrv6-small-trt-jp511-trt8.5-orin-batch8-cls8"
)
DEFAULT_OCR_FALLBACK_MODEL_DIR = "/models/ocr/ppocrv6-small-mnn"
DEFAULT_OCR_ONNX_MODEL_DIR = "/models/ocr/ppocrv6-small-ort"
DEFAULT_OCR_DEVICE = "cuda"

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
                "backend": {"type": "string", "enum": ["mnn", "onnxruntime", "tensorrt"], "default": DEFAULT_OCR_BACKEND, "scope": "shared"},
                "fallback_backend": {"type": "string", "enum": ["", "mnn", "onnxruntime"], "default": DEFAULT_OCR_FALLBACK_BACKEND, "scope": "shared"},
                "model_dir": {"type": "string", "description": "OCR 模型或 TensorRT engine 目录", "scope": "shared"},
                "fallback_model_dir": {"type": "string", "description": "OCR 回退模型目录", "scope": "shared"},
                "device": {"type": "string", "enum": ["cpu", "cuda"], "default": DEFAULT_OCR_DEVICE, "scope": "shared"},
                "device_id": {"type": "integer", "minimum": 0, "default": 0, "scope": "shared"},
                "gpu_mem_mb": {"type": "integer", "minimum": 0, "default": 512, "scope": "shared"},
                "use_angle_cls": {"type": "boolean", "default": True, "description": "启用 0/180 度文字方向分类", "scope": "shared"},
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
                "min_interval_ms": {"type": "integer", "minimum": 0, "default": 0, "description": "帧处理最小间隔(ms)，限制 GPU 占用，0=不限", "scope": "shared"},
            },
        },
        "topic_in":  [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [{"format": "data/json",  "desc": "OCR result with text boxes"}],
    }
]


# ── OCR Adapters ──────────────────────────────────────────────────────────────

class OCRAdapter(ABC):
    """OCR 适配器抽象基类"""

    @abstractmethod
    def recognize(self, image_bytes: bytes, language: str = "zh") -> list:
        """识别图片中的文字，返回文本列表（每项包含 text 和 bbox）"""
        ...

    @staticmethod
    def format_result(results: list) -> str:
        """将结果列表格式化为纯文本字符串（用于兼容旧逻辑）"""
        return " ".join(item.get("text", "") for item in results if item.get("text"))


def _ocr_output_topic(input_topic: str) -> str:
    return f"{input_topic}/ocr"


def _freeze_config(value):
    if isinstance(value, dict):
        return tuple(
            sorted(
                (key, _freeze_config(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, list):
        return tuple(_freeze_config(item) for item in value)
    return value


def _adapter_signature(cfg: dict) -> tuple:
    backend = str(
        cfg.get('backend', DEFAULT_OCR_BACKEND)
    ).strip().lower()
    fallback_backend = str(cfg.get(
        'fallback_backend',
        DEFAULT_OCR_FALLBACK_BACKEND if backend == 'tensorrt' else '',
    )).strip().lower()
    default_model_dir = {
        'tensorrt': DEFAULT_OCR_MODEL_DIR,
        'mnn': DEFAULT_OCR_FALLBACK_MODEL_DIR,
        'onnxruntime': DEFAULT_OCR_ONNX_MODEL_DIR,
    }.get(backend, DEFAULT_OCR_MODEL_DIR)
    return (
        backend,
        fallback_backend,
        cfg.get('model_dir', default_model_dir),
        cfg.get(
            'fallback_model_dir',
            DEFAULT_OCR_FALLBACK_MODEL_DIR
            if fallback_backend == 'mnn' else '',
        ),
        str(cfg.get(
            'device', DEFAULT_OCR_DEVICE if backend == 'tensorrt' else 'cpu'
        )).strip().lower(),
        int(cfg.get('device_id', 0)),
        int(cfg.get('gpu_mem_mb', 512)),
        bool(cfg.get('use_angle_cls', True)),
        int(cfg.get('num_threads', 2)),
        int(cfg.get('max_side_len', DEFAULT_MAX_SIDE_LEN)),
        float(cfg.get('rec_min_score', DEFAULT_REC_MIN_SCORE)),
        bool(cfg.get('enable_preprocess', True)),
        float(cfg.get('det_thresh', DEFAULT_DET_THRESH)),
        float(cfg.get('det_box_thresh', DEFAULT_DET_BOX_THRESH)),
        float(cfg.get('det_unclip_ratio', DEFAULT_DET_UNCLIP_RATIO)),
        _freeze_config(cfg.get('large_image_strategy', {})),
        _freeze_config(cfg.get('crop_refinement', {})),
        _freeze_config(cfg.get('empty_result_retry', {})),
    )


def _build_ocr_adapter(cfg: dict) -> OCRAdapter:
    """Create the configured on-device OCR adapter."""
    backend = str(
        cfg.get('backend', DEFAULT_OCR_BACKEND)
    ).strip().lower()
    fallback_backend = str(cfg.get(
        'fallback_backend',
        DEFAULT_OCR_FALLBACK_BACKEND if backend == 'tensorrt' else '',
    )).strip().lower()
    default_model_dir = {
        'tensorrt': DEFAULT_OCR_MODEL_DIR,
        'mnn': DEFAULT_OCR_FALLBACK_MODEL_DIR,
        'onnxruntime': DEFAULT_OCR_ONNX_MODEL_DIR,
    }.get(backend, DEFAULT_OCR_MODEL_DIR)
    return RapidOCRAdapter(
        cfg.get('model_dir', default_model_dir),
        backend=backend,
        fallback_backend=fallback_backend,
        fallback_model_dir=cfg.get(
            'fallback_model_dir',
            DEFAULT_OCR_FALLBACK_MODEL_DIR
            if fallback_backend == 'mnn' else '',
        ),
        device=str(cfg.get(
            'device', DEFAULT_OCR_DEVICE if backend == 'tensorrt' else 'cpu'
        )).strip().lower(),
        device_id=int(cfg.get('device_id', 0)),
        gpu_mem_mb=int(cfg.get('gpu_mem_mb', 512)),
        use_angle_cls=bool(cfg.get('use_angle_cls', True)),
        num_threads=int(cfg.get('num_threads', 2)),
        max_side_len=int(cfg.get('max_side_len', DEFAULT_MAX_SIDE_LEN)),
        rec_min_score=float(cfg.get('rec_min_score', DEFAULT_REC_MIN_SCORE)),
        enable_preprocess=bool(cfg.get('enable_preprocess', True)),
        det_thresh=float(cfg.get('det_thresh', DEFAULT_DET_THRESH)),
        det_box_thresh=float(
            cfg.get('det_box_thresh', DEFAULT_DET_BOX_THRESH)
        ),
        det_unclip_ratio=float(
            cfg.get('det_unclip_ratio', DEFAULT_DET_UNCLIP_RATIO)
        ),
        large_image_strategy=dict(cfg.get('large_image_strategy') or {}),
        crop_refinement=dict(cfg.get('crop_refinement') or {}),
        empty_result_retry=dict(cfg.get('empty_result_retry') or {}),
    )


# ── ROS2 Node (订阅模式) ───────────────────────────────────────────────────────

class _OCRNode(Node):
    """订阅 image/jpeg topic，持续进行 OCR 识别"""

    def __init__(self, input_topic: str, adapter: OCRAdapter, language: str = "zh",
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
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stop_event.set()
        self._generation = 0
        self._worker_threads: list[threading.Thread] = []
        self._frame_count = 0  # 收到的图片帧计数

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

        log.info(f"[ocr] started: {self._input_topic} → {self._output_topic} | {_resource_snapshot()}")
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

        log.info(f"[ocr] stopped: {self._input_topic} | {_resource_snapshot()}")
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
        self._frame_count += 1
        image_data = bytes(msg.data)
        log.debug(f"[ocr] received image frame #{self._frame_count}: "
                  f"size={len(image_data)} bytes, format={msg.format}, "
                  f"topic={self._input_topic}")
        try:
            frame_queue.put_nowait((image_data, time.time()))
        except queue.Full:
            log.warning("[ocr] frame queue full, dropping old frame (queue_size=1)")
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
                log.error("[ocr] recognition error: %s", payload["error"])
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

            # 资源监控：每 20 帧打一次快照（泄漏/线程堆积排查用）
            if self._frame_count % 20 == 0:
                log.info(f"[ocr] monitor | {_resource_snapshot()}")

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
        self._instance_adapters: dict[str, tuple[tuple, OCRAdapter]] = {}
        self._retired_nodes: list[_OCRNode] = []
        self._executor = executor
        self._lifecycle_lock = threading.RLock()

        self._adapter: OCRAdapter | None = None
        log.info(
            f"[ocr] plugin registered: language={self._language}; "
            "runtime will load on first start"
        )

    def get_tools(self) -> list:
        return TOOLS

    def _retire_node(self, node_key: str) -> dict:
        node = self._nodes.pop(node_key)
        result = node.stop()
        self._executor.remove_node(node)
        if node not in self._retired_nodes:
            self._retired_nodes.append(node)
        log.info(
            f"[ocr] node retired: {node_key} | nodes={len(self._nodes)} "
            f"retired={len(self._retired_nodes)} "
            f"instance_adapters={len(self._instance_adapters)} | {_resource_snapshot()}"
        )
        return result

    def _default_adapter(self) -> OCRAdapter:
        if self._adapter is None:
            self._adapter = _build_ocr_adapter(self._plugin_cfg)
            log.info(f"[ocr] default runtime loaded | {_resource_snapshot()}")
        return self._adapter

    def _adapter_for_instance(self, instance_id: str) -> OCRAdapter:
        override = self._instance_configs.get(instance_id, {})
        cfg = {**self._plugin_cfg, **override}
        signature = _adapter_signature(cfg)
        if signature == _adapter_signature(self._plugin_cfg):
            return self._default_adapter()

        cached = self._instance_adapters.get(instance_id)
        if cached and cached[0] == signature:
            return cached[1]

        adapter = _build_ocr_adapter(cfg)
        self._instance_adapters[instance_id] = (signature, adapter)
        log.warning(
            f"[ocr] NEW adapter built for instance={instance_id} "
            f"(cache size={len(self._instance_adapters)}) | {_resource_snapshot()}"
        )
        return adapter

    def _configure_node(self, node: _OCRNode, instance_id: str) -> None:
        cfg = {**self._plugin_cfg, **self._instance_configs.get(instance_id, {})}
        node._adapter = self._adapter_for_instance(instance_id)
        node._language = cfg.get("language", self._language)
        node._min_interval = max(
            0.0, float(cfg.get("min_interval_ms", 0)) / 1000.0
        )

    @staticmethod
    def _unique_nodes(nodes) -> list[_OCRNode]:
        unique = []
        seen = set()
        for node in nodes:
            identity = id(node)
            if identity not in seen:
                seen.add(identity)
                unique.append(node)
        return unique

    def prepare_shutdown(self) -> None:
        with self._lifecycle_lock:
            nodes = self._unique_nodes(
                [*self._nodes.values(), *self._retired_nodes]
            )
            for node in nodes:
                node.stop()

    def destroy_nodes(self) -> None:
        with self._lifecycle_lock:
            nodes = self._unique_nodes(
                [*self._nodes.values(), *self._retired_nodes]
            )
            for node in nodes:
                node.destroy_node()
            self._nodes.clear()
            self._retired_nodes.clear()

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

            if node_key not in self._nodes:
                adapter = self._adapter_for_instance(instance_id)
                cfg = {
                    **self._plugin_cfg,
                    **self._instance_configs.get(instance_id, {}),
                }
                language = cfg.get("language", self._language)

                node = _OCRNode(
                    input_topic, adapter, language,
                    node_suffix=node_key.replace('/', '_').replace('-', '_'),
                    min_interval=float(
                        {**self._plugin_cfg,
                         **self._instance_configs.get(instance_id, {})}
                        .get('min_interval_ms', 0)
                    ) / 1000.0,
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
                previous = self._instance_configs.get(instance_id, {})
                previous_cfg = {**self._plugin_cfg, **previous}
                updated_override = {**previous, **cfg}
                updated_cfg = {**self._plugin_cfg, **updated_override}
                rebuild = (
                    _adapter_signature(updated_cfg)
                    != _adapter_signature(previous_cfg)
                )
                self._instance_configs[instance_id] = updated_override
                if rebuild:
                    self._instance_adapters.pop(instance_id, None)
                if instance_id in self._nodes:
                    node = self._nodes[instance_id]
                    node.stop()
                    node._adapter = None
                    node._language = updated_cfg.get("language", self._language)
                    node._min_interval = max(
                        0.0,
                        float(updated_cfg.get("min_interval_ms", 0)) / 1000.0,
                    )
                return {
                    "status": "configured",
                    "instance_id": instance_id,
                    "reused": not rebuild,
                }
            else:
                updated_cfg = {**self._plugin_cfg, **cfg}
                rebuild = (
                    _adapter_signature(updated_cfg)
                    != _adapter_signature(self._plugin_cfg)
                )
                if rebuild:
                    self._adapter = None
                    self._instance_adapters.clear()
                self._plugin_cfg = updated_cfg
                self._language = updated_cfg.get('language', self._language)
                for node in self._nodes.values():
                    node.stop()
                    node._adapter = None
                    node._language = self._language
                    node._min_interval = max(
                        0.0,
                        float(updated_cfg.get("min_interval_ms", 0)) / 1000.0,
                    )
                return {
                    "status": "configured",
                    "adapter_ok": True,
                    "adapter_loaded": self._adapter is not None,
                    "reused": not rebuild,
                }

        return None
