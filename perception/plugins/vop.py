#!/usr/bin/env python3
"""
plugins/vop.py — VideoObjectPerceptionPlugin: YOLOE-26 open-vocabulary object detection.

Subscribes to image/jpeg topics, runs a prebuilt TensorRT engine, publishes
detected objects with center-relative normalized coordinates. Supports
multi-instance (one instance per input topic).

Two things changed together here, and the second is a consequence of the first:

* **TensorRT directly, not PyTorch and not ultralytics.** The previous version
  loaded a `.pt` and ran ultralytics in eager mode, ~35.6 ms/frame on an Orin
  NX 8GB. Loading a TensorRT engine *through ultralytics* measured 37.3 ms — no
  better — because its Python pre/post-processing dominates whatever the
  backend. So the engine is driven through
  `utils.tensorrt_runtime.TensorRTEngine` with the letterbox and decode in
  `plugins/vision_runtime.py`. Engines are built offline and shipped as pinned
  bundles (`utils.model_downloader.ensure_vop_model`), as OCR ships its own.

* **The vocabulary is frozen.** Ultralytics bakes the open-vocabulary class list
  into the weights at export time; on an exported model `set_classes()` raises.
  So the runtime `set_classes` action and the per-instance `classes` config that
  this plugin used to offer cannot work against an engine. Both now fail with an
  explicit message naming the baked vocabulary rather than silently doing
  nothing — a card that quietly stops honouring its configured classes is far
  worse than one that says why.

The vocabulary is read out of the engine's own metadata, with the bundled
`vocab.json` as a fallback; it is never hardcoded here, so what the plugin
reports and what the engine can actually detect cannot drift apart.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from utils.ros_lifecycle import dispose_node

from plugins.image_input import BadInput, load_image_bytes

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

# Where a topic-less card publishes. Fixed, because with no input topic
# there is nothing to derive it from — same rule as tts's /perception/tts.
DEFAULT_OUTPUT_TOPIC = "/perception/vop"

# Instance key for the topic-less card, mirroring tts's `_default`.
_DEFAULT_INSTANCE = "_default"

DEFAULT_MODEL = "yoloe-26s-seg"

# Names that already exist in deployed canvas cards and yaml. A card saved when
# this plugin ran YOLOv8-World must keep loading after an upgrade: the engine it
# gets is the current one, but the name it asks for still resolves. Dropping the
# alias would turn every such card into `state: error` on its next restart.
_MODEL_ALIASES = {
    "yolov8s-worldv2": DEFAULT_MODEL,
    "yolov8s-world": DEFAULT_MODEL,
    "yoloe-26s": DEFAULT_MODEL,
}


def output_topic_for(input_topic: Optional[str]) -> str:
    """The one place the output topic is derived from the input.

    Built in three places before — the node, info, and the loading
    reply — and the loading reply forgot the topic-less case, so a
    card starting without a camera reported publishing to
    "None/objects".
    """
    return f"{input_topic}/objects" if input_topic else DEFAULT_OUTPUT_TOPIC


def canonical_model_name(name: str) -> str:
    """Map a configured model name onto the one model this build ships."""
    base = (name or "").strip()
    if base.endswith(".pt") or base.endswith(".engine"):
        base = base.rsplit(".", 1)[0]
    return _MODEL_ALIASES.get(base, base or DEFAULT_MODEL)

TOOLS = [
    {
        "name": "vop",
        "type": "processor",
        "multiInstance": True,
        # `set_classes` is deliberately absent from the enum: the engine's
        # vocabulary is frozen at export time, so advertising the action would
        # promise the agent a capability that always fails. dispatch() still
        # answers it — with an explanation — because deployed cards and older
        # conversations can still send it.
        "description": "Video Object Perception — detect objects in camera feed (fixed open-vocabulary set, see info)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "start", "stop", "info", "config",
                        "recognize_by_photo", "recognize_by_url",
                        "list_recognizable_objects",
                    ],
                    "description": "Action to perform"
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 image topic to subscribe (e.g. /hostname/camera/rgb). 可选：不填则卡片以按需模式启动，不订阅摄像头，只服务 recognize_by_photo / recognize_by_url"
                },
                # `format: file` makes the canvas render a file picker;
                # `uploadTo: mcp` posts it to /api/mcp/<id>/file/upload, which
                # streams the bytes to *this* service and returns the path they
                # landed on here — so the value this field receives is already a
                # path perception can open, with no shared mount.
                "image_path": {"type": "string", "format": "file", "accept": "image/*", "uploadTo": "mcp", "description": "图片文件。从卡片上传，或填一个容器可读的路径（如 /uploads/scene.jpg）。常见格式都支持，过大的图会本地缩放"},
                "url": {"type": "string", "description": "图片的 http(s) 地址，如 https://example.com/scene.jpg。下载后本地解码，格式限制同 image_path"},
                "confidence": {"type": "number", "description": "本次识别的置信度阈值（0-1）。不填则用卡片配置的值"},
            },
            "required": ["action"],
            "x-action-params": {
                "start":  {"params": ["input_topic"], "description": "启动。给 input_topic 则持续检测该摄像头话题；不给则以按需模式启动，只服务单张图片的识别"},
                "stop":   {"params": [], "description": "Stop detection"},
                "info":   {"params": ["input_topic"], "description": "Report state, topics and the frozen class list"},
                "config": {"params": [], "description": "Update confidence / fps"},
                "recognize_by_photo": {
                    "params": ["image_path", "confidence"],
                    "description": "认出一张图片里的物体 — 一次性识别，不需要摄像头也不需要先 start。返回每个物体的名称、画面中的相对位置与置信度",
                },
                "recognize_by_url": {
                    "params": ["url", "confidence"],
                    "description": "认出图片 URL 里的物体 — 与 recognize_by_photo 相同，只是图片来自 http(s) 而非本地文件",
                },
                "list_recognizable_objects": {
                    "params": [],
                    "description": "列出这个引擎能认出的全部物体类别。类别在导出 engine 时固定，运行时不可更改 — 先查这个，就知道某样东西问不问得出来",
                },
            },
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "confidence": {"type": "number", "description": "Detection confidence threshold (0-1)", "default": 0.3, "scope": "instance"},
                "fps":        {"type": "integer", "description": "Max inference frames per second", "default": 5, "scope": "instance"},
            },
        },
        "topic_in":  [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [{"format": "data/json",  "desc": "detected objects with positions"}],
    }
]


# ── ROS2 Node (one per instance/topic) ────────────────────────────────────────

class _VOPNode(Node):
    """Per-topic YOLO inference node."""

    def __init__(self, input_topic: Optional[str], model, confidence: float, fps: float,
                 node_suffix: str, vocabulary: Optional[list] = None):
        super().__init__(f"vop_{node_suffix}" if node_suffix else "vop")
        # Topic-less is a supported mode, as in plugins/tts.py: a card driven
        # only by recognize_by_photo has no camera to subscribe to, but still
        # wants somewhere to publish so the canvas shows the flow. The fallback
        # output is fixed because there is no input topic to derive it from.
        self._input_topic = input_topic or ''
        self._output_topic = output_topic_for(input_topic)
        self._model = model
        self._vocabulary = list(vocabulary or [])
        self._confidence = confidence
        self._fps = fps
        self._frame_interval = 1.0 / max(fps, 0.1)

        self._pub = self.create_publisher(String, self._output_topic, _PUB_QOS)
        self._sub: Optional[object] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._last_inference_time = 0.0
        self._detect_count = 0
        self._running = False
        # Serializes start/stop on this node. The plugin calls both from HTTP
        # handler threads, and the canvas routinely does config→start→stop→start
        # within seconds; without this two starts can both pass the running
        # check and the second overwrites the first's subscription, orphaning a
        # live worker nothing can stop. Same rule as plugins/ocr.py.
        self._lifecycle_lock = threading.RLock()

    def request_stop(self) -> None:
        """Signal the worker to wind down without taking the lifecycle lock.

        stop() has to be able to cancel a start() that is still in progress, so
        the cancellation flag must be settable while that start holds the lock.
        """
        self._stop_event.set()

    def _status(self) -> dict:
        return {
            "state": "running" if self._running else "idle",
            "input": self._input_topic,
            "output": self._output_topic,
            "mode": "stream" if self._input_topic else "on_demand",
        }

    def start(self) -> dict:
        with self._lifecycle_lock:
            if self._running:
                return self._status()
            self._stop_event.clear()
            if self._input_topic and self._sub is None:
                self._sub = self.create_subscription(
                    CompressedImage, self._input_topic, self._image_cb, _LOW_LAT_QOS
                )
                self._worker = threading.Thread(target=self._inference_worker, daemon=True,
                                                name=f"vop_worker_{self._input_topic}")
                self._worker.start()
            # Without a topic there is nothing to subscribe to and no frames to
            # consume, so no worker is spawned; the node exists to own the
            # publisher that one-shot results go out on.
            self._running = True
            log.info(f"[vop] started: {self._input_topic or '(no topic, on-demand)'} "
                     f"→ {self._output_topic}")
            return self._status()

    def stop(self) -> dict:
        """Stop this node's worker. Does NOT touch the subscription.

        Destroying a subscription while the node is still registered with the
        executor races the executor's own wait list and kills the spin thread
        with `InvalidHandle: cannot use Destroyable because destruction was
        requested` — which takes every subscription in the process with it,
        silently, because nothing catches it. The subscription is torn down by
        destroy_node() in utils.ros_lifecycle.dispose_node, after the node has
        been removed from the executor. Same order plugins/ocr.py uses.
        """
        # Flag first, lock second: a start() holding the lock will see the flag
        # as soon as it releases, instead of this call queueing behind it.
        self._stop_event.set()
        with self._lifecycle_lock:
            if self._worker and self._worker.is_alive():
                self._worker.join(timeout=3.0)
            self._worker = None
            self._running = False
            log.info(f"[vop] stopped: {self._input_topic or '(no topic)'}")
            return self._status()

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
        from plugins.vision_runtime import decode_detections

        while not self._stop_event.is_set():
            try:
                jpeg_bytes = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            started = time.time()
            try:
                frame = cv2.imdecode(
                    np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR
                )
                if frame is None:
                    continue
                outputs, meta = self._model.infer(frame)
                boxes, scores, classes = decode_detections(
                    outputs, meta, self._confidence
                )
                objects = self._extract_objects(boxes, scores, classes, frame.shape)
                self._publish_objects(objects, started)
            except Exception as e:
                log.error(f"[vop] inference error: {e}", exc_info=True)

    def _class_name(self, cls_id: int) -> str:
        """Resolve a class id through the engine's frozen vocabulary.

        vocab.json is written by the exporter from the same list, in the same
        order, that was baked into the weights. An id outside it stays numeric
        rather than becoming a confidently wrong word.
        """
        if 0 <= cls_id < len(self._vocabulary):
            return self._vocabulary[cls_id]
        return str(cls_id)

    def _extract_objects(self, boxes, scores, classes, shape) -> list:
        H, W = shape[:2]
        half_w, half_h = W / 2.0, H / 2.0
        objects = []
        for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, classes):
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            objects.append({
                "name": self._class_name(int(cls_id)),
                "position": [
                    round(float((cx - half_w) / half_w), 3),
                    round(float((cy - half_h) / half_h), 3),
                ],
                "confidence": round(float(score), 2),
            })
        return objects

    def publish_objects(self, objects: list, started: Optional[float] = None):
        """Publish a detection payload. Used by the stream worker and by
        the one-shot photo actions, so both emit the same thing."""
        self._publish_objects(objects, started)

    def _publish_objects(self, objects: list, started: Optional[float] = None):
        self._detect_count += 1
        # `count` and `latency_ms` follow plugins/face.py: a consumer should not
        # have to len() the list to know whether anything was seen, and latency
        # is the number an operator actually watches. `timestamp` keeps its name
        # rather than becoming face's `ts` — renaming it would break every
        # existing reader of {topic}/objects for no gain.
        payload = {
            "timestamp": time.time(),
            "count": len(objects),
            "objects": objects,
        }
        if started is not None:
            # Measured from the start of processing this frame, not from its
            # arrival: queue wait is a function of the fps cap, not of how long
            # detection takes, and mixing them makes the number unreadable.
            payload["latency_ms"] = int((time.time() - started) * 1000)
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self._pub.publish(msg)


# ── Plugin class ──────────────────────────────────────────────────────────────

class VideoObjectPerceptionPlugin:
    PREFIX = "vop"

    def __init__(self, plugin_cfg: dict, namespace: str, executor):
        self._namespace = namespace
        self._executor = executor
        # Kept whole for the image-source helpers, which read image_roots and
        # max_image_bytes straight from it (see plugins/image_input.py).
        self._plugin_cfg = dict(plugin_cfg or {})
        self._confidence = float(plugin_cfg.get("confidence", 0.3))
        self._fps = int(plugin_cfg.get("fps", 5))
        self._model_name = canonical_model_name(plugin_cfg.get("model", DEFAULT_MODEL))
        self._vocabulary: list[str] = []      # filled from the bundle's vocab.json
        self._model = None  # lazy load
        self._model_loading = False
        self._model_load_error = None
        # The downloader's progress line while the engine bundle is being
        # fetched; None once it is in. Read by the "loading" replies below, so a
        # cold start reports how far along it is instead of a fixed sentence.
        self._model_load_status = None
        self._model_lock = threading.Lock()
        self._nodes: dict[str, _VOPNode] = {}
        self._instance_configs: dict[str, dict] = {}  # per-instance config overrides
        # Guards _nodes and _instance_configs. Every dispatch() runs on its own
        # ThreadingHTTPServer thread, so an unguarded read-modify-write of
        # _nodes can leave a started node unreachable — see
        # perception/README.md § "Plugin Concurrency". Never held across
        # node.start()/stop() or a model load.
        self._nodes_lock = threading.RLock()
        # A configured `classes` list cannot be honoured any more (the engine's
        # vocabulary is frozen). Remember that it was asked for, so info() and
        # the next dispatch can say so instead of pretending it took effect.
        self._rejected_classes: list[str] = list(plugin_cfg.get("classes") or [])
        if self._rejected_classes:
            log.warning(
                "[vop] ignoring configured classes %s — this build runs a "
                "TensorRT engine with a frozen vocabulary; see info action",
                self._rejected_classes,
            )
        # The engine is loaded lazily on the first start, but the card has to be
        # able to say what this build can detect before anyone starts it —
        # otherwise a freshly deployed robot shows a vop card with no classes
        # and no way to find out what it would find. Fetching just vocab.json
        # (~2 KB, pinned) in the background gives an honest answer within
        # seconds without blocking dispatch or touching the GPU.
        threading.Thread(target=self._prefetch_vocabulary, daemon=True,
                         name="vop_vocab_prefetch").start()

    def _prefetch_vocabulary(self):
        """Populate _vocabulary from vocab.json alone, without loading the engine.

        Never fatal: a robot with no route to COS keeps a working vop (the
        engine download at start has its own error path) and simply reports an
        unknown vocabulary until then.
        """
        try:
            from utils.model_downloader import (
                VOP_MODEL_BUNDLES, ensure_verified_bundle, require_models_subpath,
                select_bundle_family,
            )

            model_dir = require_models_subpath(
                os.environ.get("VOP_MODEL_DIR", "/models/vop")
            )
            key = select_bundle_family(VOP_MODEL_BUNDLES)
            entry = VOP_MODEL_BUNDLES[key]
            vocab_only = {n: m for n, m in entry["files"].items() if n == "vocab.json"}
            if not vocab_only:
                return
            paths = ensure_verified_bundle(
                f"vop/{key}/vocab", model_dir, entry["base_url"], vocab_only
            )
            vocab = self._read_vocab(paths.get("vocab.json"))
            if vocab and not self._vocabulary:
                self._vocabulary = vocab
                log.info(f"[vop] vocabulary available before load: {len(vocab)} classes")
        except Exception as exc:
            log.warning(f"[vop] could not prefetch vocabulary ({exc}); "
                        "info will report it as unknown until the engine loads")

    def _frozen_vocab_error(self, requested) -> str:
        """The one explanation both rejection paths give."""
        if self._vocabulary:
            head = ", ".join(self._vocabulary[:12])
            more = f" (+{len(self._vocabulary) - 12} more)" if len(self._vocabulary) > 12 else ""
            covers = f"The engine detects {len(self._vocabulary)} classes: {head}{more}. "
        else:
            # Never say "0 classes" — it reads as "this detects nothing", which
            # is wrong and alarming. The list simply is not known yet.
            covers = ("The engine's class list is not loaded yet, so it cannot be "
                      "listed here; start the card, or check the vop card's info. ")
        return (
            f"This build runs a prebuilt TensorRT engine whose open-vocabulary "
            f"class list was frozen when the engine was exported, so classes "
            f"cannot be changed at runtime. Requested: {list(requested)}. "
            f"{covers}"
            f"To detect something outside that list, rebuild the engine with "
            f"tools/export_vision_engines.py and republish the bundle."
        )

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return

            # No YOLO_CONFIG_DIR / TORCH_HOME setup any more: ultralytics is not
            # in this path at all — the engine is run directly through
            # utils.tensorrt_runtime. Only the cv2 repair below still matters,
            # because the decode and letterbox use it.

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

            from plugins.vision_runtime import VisionEngineSession

            engine_path, vocab = self._resolve_engine()
            log.info(f"[vop] loading engine: {engine_path}")
            self._model = VisionEngineSession(engine_path)
            # The engine's own metadata wins over the bundled vocab.json: it was
            # written by the export that baked the classes into the weights, so
            # it cannot be stale or out of order. vocab.json only covers an
            # engine built without names.
            self._vocabulary = self._model.class_names() or vocab
            if not self._vocabulary:
                log.warning("[vop] engine carries no class names and no vocab.json "
                            "was readable — detections will be labelled by index")
            log.info(f"[vop] engine loaded: {self._model_name} "
                     f"input={self._model.input_size}, "
                     f"{len(self._vocabulary)} classes")

    def _resolve_engine(self) -> tuple[str, list[str]]:
        """Return (engine path, frozen vocabulary) for the configured model.

        A path given in config wins, so a dev box can point at a locally built
        engine; otherwise the pinned bundle for this machine's TensorRT is
        fetched. There is no `.pt` fallback: silently dropping back to PyTorch
        would cost 8x per frame and look like nothing was wrong.
        """
        configured = (self._model_name or "").strip()
        if configured.endswith(".engine") and os.path.isfile(configured):
            return configured, self._read_vocab(
                os.path.join(os.path.dirname(configured), "vocab.json")
            )

        from utils.model_downloader import ensure_vop_model
        from utils.model_progress import fetch_status

        model_dir = os.environ.get("VOP_MODEL_DIR", "/models/vop")
        progress_cb, _ = fetch_status(
            lambda text: setattr(self, "_model_load_status", text), "yoloe-26s-seg")
        paths = ensure_vop_model(model_dir, progress_cb=progress_cb)
        engine = next(p for name, p in paths.items() if name.endswith(".engine"))
        return engine, self._read_vocab(paths.get("vocab.json"))

    @staticmethod
    def _read_vocab(path: Optional[str]) -> list[str]:
        """Read the class list the engine was exported with.

        Missing or unreadable is not fatal: the engine still detects exactly
        what it was built for, and the only thing lost is this plugin's ability
        to name those classes in info() and in rejection messages. Detection
        results carry their own names from the engine metadata.
        """
        if not path or not os.path.isfile(path):
            log.warning("[vop] no vocab.json beside the engine; class list unknown")
            return []
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as exc:
            log.warning(f"[vop] unreadable vocab.json ({exc}); class list unknown")
            return []
        names = data.get("classes") if isinstance(data, dict) else data
        return [str(n) for n in names] if isinstance(names, list) else []

    # ── one-shot recognition ─────────────────────────────────────────────

    def _require_engine(self):
        """Return a loaded engine, loading it on demand.

        A photo question is useful without any instance running — an operator
        asks "what is in this picture" before pointing a camera anywhere — so
        it triggers the same single-flight load a `start` would and waits for
        it, rather than reporting `loading` and making the caller poll. Each
        tools/call already has its own thread (ThreadingHTTPServer), so
        blocking here blocks nothing else. Same rule as plugins/face.py.
        """
        self._ensure_model()
        return self._model

    def _recognize_image(self, args: dict, url_action: str) -> dict:
        """Decode one image and run the detector over it once."""
        cfg = dict(self._plugin_cfg)
        try:
            data, source = load_image_bytes(args, cfg, url_action=url_action)
        except BadInput as error:
            return error.as_result()

        import cv2

        started = time.time()
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return BadInput(
                "could not decode that file as an image — check it is a real "
                "picture and not, say, HTML returned by a redirect", source,
            ).as_result()

        try:
            model = self._require_engine()
        except Exception as error:  # noqa: BLE001 — surfaced to the caller
            log.error(f"[vop] engine load failed during recognize: {error}", exc_info=True)
            return {"ok": False, "reason": "engine_unavailable", "detail": str(error)}

        confidence = args.get("confidence")
        confidence = float(confidence) if confidence not in (None, "") else self._confidence

        from plugins.vision_runtime import decode_detections

        outputs, meta = model.infer(frame)
        boxes, scores, classes = decode_detections(outputs, meta, confidence)

        # Same shape the stream publishes, so a consumer written against
        # {topic}/objects needs no second parser — plus the pixel box, which a
        # caller who cannot see the frame has no other way to recover.
        height, width = frame.shape[:2]
        half_w, half_h = width / 2.0, height / 2.0
        objects = []
        for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, classes):
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            objects.append({
                "name": self._class_name_from_vocab(int(cls_id)),
                "position": [
                    round(float((cx - half_w) / half_w), 3),
                    round(float((cy - half_h) / half_h), 3),
                ],
                "confidence": round(float(score), 2),
                "bbox": [round(float(v), 1) for v in (x1, y1, x2, y2)],
            })

        # Echo onto the card's output topic when one is running, so a topic-less
        # card wired into the canvas actually shows data flowing — which is the
        # only reason it is startable without a camera. Purely additive: the
        # answer goes back through MCP regardless, and a card that was never
        # started publishes nothing.
        #
        # Deliberately not reported back. Which topic the echo went out on is
        # not something the caller asked about, and every field in this reply is
        # re-read by the model on every turn. Same call as visual_depth's.
        self._publish_one_shot(args.get("instance_id", ""), objects, started)

        result = {
            "ok": True,
            "source": source,
            "image_size": [width, height],
            "confidence_threshold": confidence,
            "count": len(objects),
            # Same field the stream publishes, so the MCP reply and
            # {topic}/objects cannot disagree about how long this took.
            "latency_ms": int((time.time() - started) * 1000),
            "objects": objects,
        }
        return result

    def _publish_one_shot(self, instance_id: str, objects: list,
                          started: Optional[float] = None) -> Optional[str]:
        """Publish a one-shot result on the named instance, or the default one."""
        with self._nodes_lock:
            node = self._nodes.get(instance_id) if instance_id else None
            if node is None:
                node = self._nodes.get(_DEFAULT_INSTANCE)
            if node is None and len(self._nodes) == 1:
                node = next(iter(self._nodes.values()))
        if node is None:
            return None
        try:
            node.publish_objects(objects, started)
            return node._output_topic
        except Exception as error:  # noqa: BLE001 — never fail the answer on this
            log.warning(f"[vop] could not echo one-shot result: {error}")
            return None

    def _class_name_from_vocab(self, cls_id: int) -> str:
        if 0 <= cls_id < len(self._vocabulary):
            return self._vocabulary[cls_id]
        return str(cls_id)

    def _start_node(self, node_key: str, input_topic: str):
        """Create and start a VOPNode for the given topic.

        Registers the node before starting it so a concurrent stop can always
        find and cancel it; the lock covers only the registration, never the
        start itself, so a stop is not queued behind a start it is trying to
        abort (perception/README.md § "Plugin Concurrency").
        """
        with self._nodes_lock:
            if node_key in self._nodes:
                return
            icfg = self._instance_configs.get(node_key, {})
            confidence = float(icfg.get("confidence", self._confidence))
            fps = int(icfg.get("fps", self._fps))
            suffix = node_key.replace("/", "_").replace("-", "_").lstrip("_")
            input_for_node = input_topic or None
            node = _VOPNode(input_for_node, self._model, confidence, fps,
                            node_suffix=suffix, vocabulary=self._vocabulary)
            self._executor.add_node(node)
            self._nodes[node_key] = node
        node.start()
        log.info(f"[vop] node started (background): {input_topic}")

    def _retire_node(self, node_key: str) -> Optional[dict]:
        """Stop, unregister and destroy one node. Returns its stop() result."""
        with self._nodes_lock:
            node = self._nodes.pop(node_key, None)
        if node is None:
            return None
        node.request_stop()
        result = node.stop()
        # remove-then-destroy, via the shared helper: the node has to leave the
        # executor before any of its handles are destroyed, or the spin thread
        # dies on InvalidHandle. It must also be destroyed and not merely
        # removed, or the publisher and the ROS node name leak and the next
        # start on the same topic trips "Publisher already registered".
        dispose_node(self._executor, node, label=f"vop/{node_key}")
        return result

    def get_tools(self) -> list:
        """TOOLS, with the frozen vocabulary folded into the description.

        The dashboard renders a tool's `description` but ignores the rest of an
        info payload — `canvas.js` keeps `topic_out` and drops everything else.
        So `info`'s `classes` list, however complete, is invisible to an
        operator, and removing the `classes` config field took away the last
        place the UI showed anything about what vop can detect. The description
        is the one channel that reaches the card, so the summary goes there.

        Built per call rather than baked into TOOLS because the vocabulary
        arrives from the background prefetch after import; the registration
        heartbeat re-reads tools, so the card picks it up shortly after start.
        """
        if not self._vocabulary:
            return TOOLS
        sample = ", ".join(self._vocabulary[:8])
        tools = [dict(t) for t in TOOLS]
        tools[0]["description"] = (
            f"Video Object Perception — detects a FIXED set of "
            f"{len(self._vocabulary)} classes ({sample}, ...). The class list is "
            f"frozen into the TensorRT engine and cannot be changed at runtime; "
            f"call info for the full list."
        )
        return tools

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action", name)
        instance_id = args.get("instance_id", "")

        if action == "info":
            # Report loading/error state
            if self._model_loading:
                return {
                    "name": "VideoObjectPerception", "manufacture": "Embodied", "model": self._model_name,
                    "state": "loading",
                    "desc": "Loading YOLO model...",
                }
            if self._model_load_error:
                return {
                    "name": "VideoObjectPerception", "manufacture": "Embodied", "model": self._model_name,
                    "state": "error",
                    "desc": f"Model load failed: {self._model_load_error}",
                }
            with self._nodes_lock:
                nodes = dict(self._nodes)
            instances = {}
            for key, node in nodes.items():
                instances[key] = {
                    "input": node._input_topic,
                    "output": node._output_topic,
                    "confidence": node._confidence,
                    "fps": node._fps,
                    "detect_count": node._detect_count,
                }
            # Determine topic info: from running instance, args, or empty
            input_topic = args.get("input_topic", "")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            # If instance_id specified and running, use its topics
            if instance_id and instance_id in nodes:
                input_topic = nodes[instance_id]._input_topic
            # If no explicit topic but there are running instances, use first one
            elif not input_topic and nodes:
                input_topic = next(iter(nodes.values()))._input_topic
            topics_in = [{"topic": input_topic, "format": "image/jpeg"}] if input_topic else []
            topics_out = ([{"topic": output_topic_for(input_topic), "format": "data/json"}]
                          if (input_topic or nodes) else [])
            state = "running" if instances else "idle"
            info = {
                "name": "VideoObjectPerception", "manufacture": "Embodied", "model": self._model_name,
                "state": state,
                "vocabulary_frozen": True,
                # `None` rather than 0 while the list is still unknown: a card
                # that says "0 classes" reads as "detects nothing", which is
                # both wrong and the first thing an operator would report as a
                # bug. The list is fetched in the background at startup.
                "total_classes": len(self._vocabulary) if self._vocabulary else None,
                "classes": self._vocabulary,
                "classes_loaded": bool(self._vocabulary),
                "instances": instances,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "YOLOE-26 open-vocabulary object detection (TensorRT, fixed class list)",
            }
            # Surface a `classes` that yaml asked for and this build cannot
            # honour. Without this the card looks perfectly healthy while
            # silently detecting a different set than its config states.
            if self._rejected_classes:
                info["ignored_config_classes"] = self._rejected_classes
                info["warning"] = self._frozen_vocab_error(self._rejected_classes)
            return info

        elif action == "start":
            input_topic = args.get("input_topic")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            # No topic is a supported mode, as in plugins/tts.py: the card comes
            # up on-demand, loads the engine and owns a publisher, and answers
            # recognize_by_photo / recognize_by_url. It just has nothing to
            # subscribe to, so it consumes no frames.
            node_key = instance_id or input_topic or _DEFAULT_INSTANCE
            with self._nodes_lock:
                running = self._nodes.get(node_key)
            if running is None:
                if self._model is None:
                    if self._model_loading:
                        return {"state": "loading",
                                "message": (self._model_load_status
                                            or "Model is still loading, please wait...")}
                    if self._model_load_error:
                        return {"state": "error", "message": f"Model failed to load: {self._model_load_error}"}
                    # Model not loaded yet — start loading in background
                    def _bg_start():
                        self._model_loading = True
                        self._model_load_error = None
                        self._model_load_status = None
                        try:
                            self._ensure_model()
                            self._model_loading = False
                            self._model_load_status = None
                            self._start_node(node_key, input_topic)
                        except Exception as e:
                            self._model_loading = False
                            self._model_load_error = str(e)
                            log.error(f"[vop] model load failed: {e}", exc_info=True)
                    threading.Thread(target=_bg_start, daemon=True, name="vop_model_load").start()
                    return {"state": "loading", "input": input_topic or "",
                            "output": output_topic_for(input_topic),
                            "message": "Model loading in background, will start automatically"}
                self._start_node(node_key, input_topic)
                with self._nodes_lock:
                    running = self._nodes.get(node_key)
                if running is None:
                    # A concurrent stop retired it between start and lookup.
                    return {"state": "idle", "input": input_topic}
            return running.start()

        elif action == "stop":
            if instance_id:
                result = self._retire_node(instance_id)
                return result if result is not None else {"state": "idle"}
            with self._nodes_lock:
                keys = list(self._nodes.keys())
            results = [key for key in keys if self._retire_node(key) is not None]
            return {"state": "idle", "stopped_instances": results} if results else {"state": "idle"}

        elif action == "recognize_by_photo":
            return self._recognize_image(args, url_action="recognize_by_url")

        elif action == "recognize_by_url":
            return self._recognize_image(args, url_action="recognize_by_url")

        elif action == "list_recognizable_objects":
            # Answers "can I ask about X" without loading the engine: the
            # vocabulary is prefetched at startup from vocab.json alone.
            if not self._vocabulary:
                return {
                    "ok": False,
                    "reason": "vocabulary_unavailable",
                    "detail": "the class list has not been fetched yet — it is "
                              "loaded in the background at startup, and comes "
                              "with the engine bundle. Retry shortly, or start "
                              "the card to force the download.",
                }
            return {
                "ok": True,
                "model": self._model_name,
                "frozen": True,
                "count": len(self._vocabulary),
                "objects": list(self._vocabulary),
                "note": "This list is baked into the TensorRT engine at export "
                        "time and cannot be changed at runtime. Anything not "
                        "listed here will never be detected, however it is "
                        "phrased.",
            }

        elif action == "set_classes":
            # Kept reachable although it is no longer advertised in the tool
            # schema: deployed cards and in-flight conversations can still send
            # it, and an explicit refusal is worth far more than "unknown
            # action" or a success that changes nothing.
            raise ValueError(self._frozen_vocab_error(args.get("classes") or []))

        elif action == "config":
            cfg = {k: v for k, v in args.items() if k not in ('action', 'instance_id') and v is not None and v != ''}
            # An old card still carrying `classes` reaches here. Refuse the
            # whole config call rather than applying confidence/fps and
            # dropping classes on the floor: a half-applied config is the
            # failure mode this is meant to prevent.
            if cfg.get("classes"):
                self._rejected_classes = list(cfg["classes"])
                raise ValueError(self._frozen_vocab_error(cfg["classes"]))
            if instance_id:
                with self._nodes_lock:
                    self._instance_configs[instance_id] = cfg
                    running = instance_id in self._nodes
                # If instance is running, retire it; the next start picks up
                # the new config.
                if running:
                    self._retire_node(instance_id)
                return {"status": "configured", "instance_id": instance_id, "config": cfg}
            else:
                # Update global defaults
                if "confidence" in cfg:
                    self._confidence = float(cfg["confidence"])
                if "fps" in cfg:
                    self._fps = int(cfg["fps"])
                return {"status": "configured", "config": cfg}

        return None
