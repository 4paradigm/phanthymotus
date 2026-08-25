#!/usr/bin/env python3
"""
plugins/ocr.py — OCRPlugin: OCR 文字识别封装。

订阅 image/jpeg topic，持续进行 OCR 识别并发布结果到 ROS2 topic。
参考 asr.py 架构设计。
"""

from __future__ import annotations

import json
import logging
import threading
import time

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from utils.latest_frame import LatestFrame
from utils.log_sampling import SampledLogGate, escape_log_text
from utils.qos import CAMERA_QOS
from utils.ros_lifecycle import dispose_node

from plugins.ocr_runtime import (
    DEFAULT_DET_BOX_THRESH,
    DEFAULT_DET_THRESH,
    DEFAULT_DET_UNCLIP_RATIO,
    DEFAULT_MAX_SIDE_LEN,
    DEFAULT_REC_MIN_SCORE,
    RapidOCRAdapter,
    recognize_to_payload,
)

log = logging.getLogger(__name__)

DEFAULT_OCR_MODEL_DIR = "/models/ocr/ppocrv6-small-trt"
_ERROR_LOG_INTERVAL_SECONDS = 10.0

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
        # Deliberately minimal: only what an operator meaningfully decides.
        # Expert knobs (model_dir, device_id, DB thresholds, crop refinement,
        # empty-result retry, ...) stay config.yaml-only — the dispatch below
        # still honors them, they are just not advertised to the config UI.
        # language is also yaml-only: the TensorRT pipeline runs one bilingual
        # zh/en model, so the value only annotates the published payload and
        # offering it in the UI would suggest a recognition switch that does
        # not exist.
        "configSchema": {
            "type": "object",
            "properties": {
                "min_interval_ms": {"type": "integer", "minimum": 0, "default": 0, "description": "帧处理最小间隔(ms)，限制 GPU 占用，0=不限", "scope": "instance"},
            },
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
    options = _adapter_options(cfg)
    from utils.model_downloader import ensure_ocr_model
    ensure_ocr_model(options["model_dir"])
    return RapidOCRAdapter(**options)


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

        # Latest frame wins: the camera callback overwrites, the worker pops.
        self._frames: LatestFrame = LatestFrame()
        self._frames.close()
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stop_event.set()
        self._generation = 0
        self._worker_threads: list[threading.Thread] = []
        # Serializes start/stop/retire on this node. The plugin calls these
        # outside its _state_lock, so without a per-node lock two concurrent
        # starts could both pass the running check and overwrite each other's
        # stop event / frame slot (obstacle's node has carried this lock from
        # the start; same pattern here).
        self._node_lock = threading.RLock()
        self._retired = False
        self._log_gate = SampledLogGate(every=100)
        self._last_error_log_at: float | None = None
        log.info(f"[ocr] node created: subscribing={self._input_topic}, publishing={self._output_topic}")

    def start(self) -> dict:
        with self._node_lock:
            return self._start_locked()

    def _start_locked(self) -> dict:
        if self._retired:
            # A concurrent stop retired (destroyed) this node between the
            # plugin registering it and this call; report idle without
            # touching the destroyed rclpy handle.
            return self._status_dict()
        if self.state == "running":
            return self._status_dict()

        if not self._adapter:
            raise RuntimeError("OCR adapter not configured")

        self._generation += 1
        generation = self._generation
        stop_event = threading.Event()
        frames: LatestFrame = LatestFrame()
        self._stop_event = stop_event
        self._frames = frames
        if self._sub is None:
            self._sub = self.create_subscription(
                CompressedImage, self._input_topic, self._image_cb, CAMERA_QOS
            )
        self.state = "running"
        self._worker_threads = [
            thread for thread in self._worker_threads if thread.is_alive()
        ]
        self._worker_thread = threading.Thread(
            target=self._worker,
            args=(generation, stop_event, frames),
            daemon=True,
        )
        self._worker_threads.append(self._worker_thread)
        self._worker_thread.start()

        log.info(f"[ocr] started: {self._input_topic} → {self._output_topic}")
        return self._status_dict()

    def stop(self) -> dict:
        with self._node_lock:
            self.state = "idle"
            self._stop_event.set()
            self._frames.close()  # drops the pending frame and wakes the worker
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

    def retire(self) -> dict:
        """Stop the node and mark it dead-for-good before it is destroyed.

        Set the flag and stop under one lock acquisition, so a start()
        waiting on the lock cannot run between them and revive a worker on a
        node that is about to be destroyed.
        """
        with self._node_lock:
            self._retired = True
            return self.stop()

    @property
    def worker_alive(self) -> bool:
        return any(thread.is_alive() for thread in self._worker_threads)

    def _image_cb(self, msg: CompressedImage):
        """接收图片帧：只保留最新一帧"""
        stop_event = self._stop_event
        frames = self._frames
        if self.state != "running" or stop_event.is_set():
            return
        frames.push((bytes(msg.data), time.time()))

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
        frames: LatestFrame,
    ):
        """后台工作线程：取最新一帧进行 OCR"""
        while not stop_event.is_set():
            frame = frames.pop(timeout=1.0)
            if frame is None:
                continue
            image_bytes, ts = frame

            t_start = time.time()
            payload = recognize_to_payload(
                self._adapter, image_bytes, self._language, ts
            )
            if not self._is_generation_active(generation, stop_event):
                continue
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self._pub.publish(msg)
            # Hot path: transitions log unthrottled, steady state is sampled
            # (first + every 100th) so camera-rate output cannot flood the
            # container logs even at DEBUG.
            outcome = "error" if "error" in payload else "ok"
            should_log, transition, occurrence = self._log_gate.check(outcome)
            if outcome == "error":
                now = time.monotonic()
                if transition or (
                    self._last_error_log_at is None
                    or now - self._last_error_log_at >= _ERROR_LOG_INTERVAL_SECONDS
                ):
                    log.error("[ocr] recognition error (occurrence %d): %s",
                              occurrence, escape_log_text(payload["error"]))
                    self._last_error_log_at = now
                elif should_log:
                    log.debug("[ocr] recognition error (occurrence %d): %s",
                              occurrence, escape_log_text(payload["error"]))
            elif should_log:
                log.debug(
                    "[ocr] published result to %s: %d items (frame %d%s)",
                    self._output_topic,
                    len(payload["items"]),
                    occurrence,
                    ", recovered" if transition and occurrence == 1 else "",
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
    """OCR MCP plugin with a non-blocking start/stop/load state machine.

    dispatch() only mutates bookkeeping under a short-lived lock; the slow
    work (model download + three TensorRT engines) runs in one background
    loader shared by every instance:

        idle --start--> loading --ok--> ready/running
                          |               ^
                          +----fail--> error (next start retries)

    * first start records the instance as pending, spawns the single loader
      and immediately returns {"state": "loading"};
    * concurrent starts only add pending instances — one download, one
      engine build, N instances;
    * info never blocks and never loads anything;
    * stop during loading cancels the pending instance; stop of a live
      instance disposes its node (executor.remove_node + destroy_node);
    * config bumps a generation token so a stale loader can never install
      an adapter built from an outdated configuration.
    """

    PREFIX = "ocr"

    def __init__(self, plugin_cfg: dict, executor):
        self._plugin_cfg = dict(plugin_cfg)
        self._language = plugin_cfg.get('language', 'zh')
        self._executor = executor

        # All fields below are guarded by _state_lock. The lock is only ever
        # held for dict/flag updates — never while downloading, building
        # engines, or joining workers — so info/start/stop stay responsive.
        self._state_lock = threading.Lock()
        self._nodes: dict[str, _OCRNode] = {}
        self._instance_configs: dict[str, dict] = {}
        self._pending_starts: dict[str, str] = {}   # node_key -> input_topic
        self._adapter: RapidOCRAdapter | None = None
        self._adapter_state = "idle"                # idle|loading|ready|error
        self._load_error: str | None = None
        self._load_generation = 0

        log.info(
            f"[ocr] plugin init: provider={plugin_cfg.get('provider')}, "
            f"language={self._language}"
        )

    def get_tools(self) -> list:
        return TOOLS

    # ── background loader (single-flight) ────────────────────────────────

    def _spawn_loader_locked(self) -> None:
        """Start the one background adapter loader. Caller holds the lock."""
        self._adapter_state = "loading"
        self._load_error = None
        generation = self._load_generation
        cfg = dict(self._plugin_cfg)
        thread = threading.Thread(
            target=self._loader, args=(generation, cfg),
            name="ocr-adapter-loader", daemon=True,
        )
        thread.start()

    def _loader(self, generation: int, cfg: dict) -> None:
        try:
            adapter = _build_ocr_adapter(cfg)
        except Exception as error:  # noqa: BLE001 - surfaced via state/info
            log.exception("[ocr] adapter load failed")
            with self._state_lock:
                if generation == self._load_generation:
                    self._adapter_state = "error"
                    self._load_error = str(error)
            return

        with self._state_lock:
            if generation != self._load_generation:
                stale = adapter
            else:
                self._adapter = adapter
                self._adapter_state = "ready"
                stale = None
        if stale is not None:
            # A config change superseded this load; never install the result.
            _close_quietly(stale)
            return

        # Bring up every instance that is still pending. Nodes are created
        # outside the lock; each one is committed only if its start was not
        # cancelled in the meantime.
        while True:
            with self._state_lock:
                if generation != self._load_generation or not self._pending_starts:
                    return
                node_key, input_topic = next(iter(self._pending_starts.items()))
            # README lifecycle rule: register under the lock *before*
            # starting, so a concurrent stop (or a failure at any later
            # point) can always locate the node — a started-but-unregistered
            # worker would be unreachable and leak.
            try:
                node = self._create_node(node_key, input_topic, adapter)
            except Exception as error:  # noqa: BLE001 - keep serving others
                log.error("[ocr] failed to build instance %r on %r: %s",
                          node_key, input_topic, escape_log_text(error))
                with self._state_lock:
                    if self._pending_starts.get(node_key) == input_topic:
                        del self._pending_starts[node_key]
                continue
            registered = False
            with self._state_lock:
                still_wanted = (
                    generation == self._load_generation
                    and self._pending_starts.get(node_key) == input_topic
                )
                if still_wanted:
                    try:
                        self._executor.add_node(node)
                    except Exception as error:  # noqa: BLE001
                        log.error("[ocr] failed to register instance %r: %s",
                                  node_key, escape_log_text(error))
                        del self._pending_starts[node_key]
                    else:
                        self._nodes[node_key] = node
                        del self._pending_starts[node_key]
                        registered = True
            if not registered:
                try:
                    node.destroy_node()   # never started, never registered
                except Exception:  # noqa: BLE001
                    pass
                continue
            try:
                node.start()
            except Exception as error:  # noqa: BLE001
                log.error("[ocr] failed to start instance %r: %s",
                          node_key, escape_log_text(error))
                with self._state_lock:
                    if self._nodes.get(node_key) is node:
                        del self._nodes[node_key]
                self._dispose(node_key, node)
                continue
            with self._state_lock:
                still_ours = self._nodes.get(node_key) is node
            if not still_ours:
                # A concurrent stop retired the node while start() ran;
                # its dispose already handled teardown — just make sure the
                # worker this start spawned is gone too.
                node.stop()

    def _create_node(self, node_key: str, input_topic: str, adapter) -> "_OCRNode":
        cfg = {**self._plugin_cfg, **self._instance_configs.get(node_key, {})}
        return _OCRNode(
            input_topic,
            adapter,
            cfg.get("language", self._language),
            node_suffix=node_key.replace('/', '_').replace('-', '_'),
            min_interval=float(cfg.get('min_interval_ms', 0)) / 1000.0,
        )

    def _dispose(self, node_key: str, node: "_OCRNode") -> None:
        """Stop and fully destroy one node (worker, executor entry, handle)."""
        try:
            # retire (not plain stop): marks the node dead under its lock so
            # an in-flight start that lost the race reports idle instead of
            # creating a subscription/worker on the destroyed handle.
            node.retire()
        finally:
            dispose_node(self._executor, node, label=f"ocr/{node_key}")
        log.info(f"[ocr] node disposed: {node_key}")

    # ── per-instance status (lock held) ───────────────────────────────────

    def _instance_state_locked(self, node_key: str) -> str:
        node = self._nodes.get(node_key)
        if node is not None:
            return node.state
        if node_key in self._pending_starts:
            return "error" if self._adapter_state == "error" else "loading"
        return "idle"

    # ── MCP dispatch ──────────────────────────────────────────────────────

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == "ocr" else name
        instance_id = args.get("instance_id", "")

        if action == "info":
            return self._do_info(instance_id, args.get("input_topic", ""))
        if action == "start":
            return self._do_start(instance_id, args)
        if action == "stop":
            return self._do_stop(instance_id)
        if action == "config":
            return self._do_config(instance_id, args)
        return None

    _DESC = "OCR service — extracts text from images"

    def _desc_locked(self, state: str) -> str:
        """Dashboard-facing description, mirroring the ASR plugin (#113): the
        static blurb normally, a reason while loading or after a failure."""
        if state == "loading":
            return "Loading OCR model and TensorRT engines..."
        if state == "error" and self._load_error:
            return f"Model load failed: {self._load_error}"
        return self._DESC

    def _do_info(self, instance_id: str, input_topic: str) -> dict:
        base = {"name": "OCR", "manufacture": "Embodied", "model": "ocr"}
        with self._state_lock:
            if instance_id:
                node = self._nodes.get(instance_id)
                topic = (
                    node._input_topic if node is not None
                    else self._pending_starts.get(instance_id, input_topic)
                )
                out = f"{topic}/ocr" if topic else ""
                state = self._instance_state_locked(instance_id)
                result = {
                    **base,
                    "state": state,
                    "desc": self._desc_locked(state),
                    "topic_in": [{"topic": topic, "format": "image/jpeg", "desc": ""}] if topic else [],
                    "topic_out": [{"topic": out, "format": "data/json", "desc": ""}] if out else [],
                }
                if state == "error" and self._load_error:
                    result["error"] = self._load_error
                return result

            keys = list(self._nodes) + [
                k for k in self._pending_starts if k not in self._nodes
            ]
            instances = {k: {"state": self._instance_state_locked(k)} for k in keys}
            topics_in, topics_out = [], []
            for key in keys:
                node = self._nodes.get(key)
                topic = node._input_topic if node else self._pending_starts[key]
                topics_in.append({"topic": topic, "format": "image/jpeg", "desc": ""})
                topics_out.append({"topic": f"{topic}/ocr", "format": "data/json", "desc": ""})
            states = {entry["state"] for entry in instances.values()}
            if "loading" in states or self._adapter_state == "loading":
                state = "loading"
            elif "running" in states:
                state = "running"
            elif "error" in states or self._adapter_state == "error":
                state = "error"
            else:
                state = "idle"
            if not keys and input_topic:
                topics_in = [{"topic": input_topic, "format": "image/jpeg", "desc": ""}]
                topics_out = [{"topic": f"{input_topic}/ocr", "format": "data/json", "desc": ""}]
            result = {
                **base,
                "state": state,
                "desc": self._desc_locked(state),
                "topic_in": topics_in,
                "topic_out": topics_out,
            }
            if instances:
                result["instances"] = instances
            if self._load_error and state == "error":
                result["error"] = self._load_error
            return result

    def _do_start(self, instance_id: str, args: dict) -> dict:
        input_topic = args.get("input_topic")
        if not input_topic:
            raise ValueError("input_topic is required for start action")
        node_key = instance_id or input_topic

        retired = None
        with self._state_lock:
            existing = self._nodes.get(node_key)
            if existing is not None and existing._input_topic != input_topic:
                # Topics are fixed at node construction; rebind the key.
                retired = self._nodes.pop(node_key)
                existing = None
        if retired is not None:
            self._dispose(node_key, retired)

        start_node = None
        with self._state_lock:
            existing = self._nodes.get(node_key)
            if existing is not None:
                start_node = existing        # idempotent re-start
            elif self._adapter_state == "ready":
                # Claim the key before leaving the lock: a concurrent stop
                # must always find the instance in _pending_starts or _nodes,
                # never in an invisible in-between state (the orphan-node
                # window PR #113 documents for the ASR plugin).
                self._pending_starts[node_key] = input_topic
            else:
                self._pending_starts[node_key] = input_topic
                if self._adapter_state in ("idle", "error"):
                    self._spawn_loader_locked()
                return {
                    "state": "loading",
                    "input": input_topic,
                    "output": _ocr_output_topic(input_topic),
                }
            adapter = self._adapter
            generation = self._load_generation

        if start_node is not None:
            return start_node.start()

        node = self._create_node(node_key, input_topic, adapter)
        # README lifecycle rule: register before start, so a concurrent stop
        # (or an add_node/start failure) can always locate the node instead of
        # leaking a started worker that nothing tracks.
        with self._state_lock:
            claimed = self._pending_starts.get(node_key) == input_topic
            current = self._nodes.get(node_key)
            fresh = generation == self._load_generation
            registered = False
            if claimed and current is None and fresh:
                try:
                    self._executor.add_node(node)
                except Exception as error:  # noqa: BLE001
                    log.error("[ocr] failed to register instance %r: %s",
                              node_key, escape_log_text(error))
                    del self._pending_starts[node_key]
                else:
                    self._nodes[node_key] = node
                    del self._pending_starts[node_key]
                    registered = True
            elif claimed and not fresh:
                # A config change invalidated the adapter mid-start. Keep the
                # pending claim: the loader spawned by config brings this
                # instance up on the new adapter.
                pass
            elif claimed:
                del self._pending_starts[node_key]
        if not registered:
            try:
                node.destroy_node()   # never started, never registered
            except Exception:  # noqa: BLE001
                pass
            if current is not None:
                return current.start()
            if claimed and not fresh:
                return {"state": "loading", "input": input_topic,
                        "output": _ocr_output_topic(input_topic)}
            return {"state": "idle", "input": input_topic,
                    "output": _ocr_output_topic(input_topic)}
        try:
            result = node.start()
        except Exception:
            with self._state_lock:
                if self._nodes.get(node_key) is node:
                    del self._nodes[node_key]
            self._dispose(node_key, node)
            raise
        with self._state_lock:
            still_ours = self._nodes.get(node_key) is node
        if not still_ours:
            # A concurrent stop retired the node while start() ran; its
            # dispose handled teardown — stop the worker this start spawned.
            node.stop()
            return {"state": "idle", "input": input_topic,
                    "output": _ocr_output_topic(input_topic)}
        return result

    def _do_stop(self, instance_id: str) -> dict:
        to_dispose: list[tuple[str, _OCRNode]] = []
        with self._state_lock:
            if instance_id:
                self._pending_starts.pop(instance_id, None)
                node = self._nodes.pop(instance_id, None)
                if node is not None:
                    to_dispose.append((instance_id, node))
            else:
                self._pending_starts.clear()
                to_dispose.extend(self._nodes.items())
                self._nodes = {}
        for node_key, node in to_dispose:
            self._dispose(node_key, node)
        return {"state": "idle"}

    def _do_config(self, instance_id: str, args: dict) -> dict:
        cfg = {
            k: v for k, v in args.items()
            if k not in ('action', 'instance_id') and v is not None and v != ''
        }

        if instance_id:
            shared = set(cfg) - {"language", "min_interval_ms"}
            if shared:
                raise ValueError(
                    "OCR inference settings are shared: " + ", ".join(sorted(shared))
                )
            retired = None
            with self._state_lock:
                previous = self._instance_configs.get(instance_id, {})
                self._instance_configs[instance_id] = {**previous, **cfg}
                retired = self._nodes.pop(instance_id, None)
            if retired is not None:
                # The instance goes back to idle; the next start picks up the
                # merged per-instance configuration.
                self._dispose(instance_id, retired)
            return {"status": "configured", "instance_id": instance_id}

        with self._state_lock:
            updated_cfg = {**self._plugin_cfg, **cfg}
            rebuild = (
                _adapter_signature(updated_cfg)
                != _adapter_signature(self._plugin_cfg)
            )
            self._plugin_cfg = updated_cfg
            self._language = updated_cfg.get('language', self._language)
            if not rebuild:
                # Lightweight fields only — apply to live nodes in place.
                for node_key, node in self._nodes.items():
                    inst = self._instance_configs.get(node_key, {})
                    merged = {**updated_cfg, **inst}
                    node._language = merged.get("language", self._language)
                    node._min_interval = max(
                        0.0, float(merged.get("min_interval_ms", 0)) / 1000.0
                    )
                return {"status": "configured", "adapter_loaded": self._adapter is not None, "reused": self._adapter is not None}

            # Model-affecting change: invalidate any in-flight load, drop the
            # cached adapter and retire nodes built on it. Pending starts stay
            # pending and are served by a fresh loader for the new config.
            self._load_generation += 1
            stale_adapter = self._adapter
            self._adapter = None
            disposed = list(self._nodes.items())
            self._nodes = {}
            if self._pending_starts:
                self._spawn_loader_locked()
            else:
                self._adapter_state = "idle"
                self._load_error = None
        for node_key, node in disposed:
            self._dispose(node_key, node)
        if stale_adapter is not None:
            _close_quietly(stale_adapter)
        return {"status": "configured", "adapter_loaded": False, "reused": False}


def _close_quietly(adapter) -> None:
    close = getattr(adapter, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - best-effort release
            log.warning("[ocr] adapter close failed", exc_info=True)
