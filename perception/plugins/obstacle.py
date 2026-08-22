#!/usr/bin/env python3
"""
plugins/obstacle.py — ObstacleDistancePlugin: obstacle distance estimation.

Subscribes to image/jpeg topics, estimates obstacle distance from camera,
publishes distance results to ROS2 topic.
Supports multi-instance (one instance per input topic).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from copy import deepcopy

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from utils.latest_frame import LatestFrame
from utils.log_sampling import SampledLogGate, escape_log_text
from utils.qos import CAMERA_QOS
from utils.ros_lifecycle import dispose_node

from .obstacle_distance_core.contracts import SceneDomain

log = logging.getLogger(__name__)


_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

TOOLS = [
    {
        "name": "obstacle",
        "type": "processor",
        "multiInstance": True,
        "description": "Obstacle Distance Estimation — estimate distance to obstacles from camera feed",
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
        # provider (single valid value), model_dir, ROI/percentile/calibration
        # tuning and the per-scene expert blocks stay config.yaml-only — the
        # dispatch below still honors them, they are just not advertised to
        # the config UI.
        "configSchema": {
            "type": "object",
            "properties": {
                "fixed_scene": {"type": "string", "enum": ["indoor", "vehicle"], "default": "indoor", "description": "部署场景（固定场景模式下生效）：indoor=室内 ZipDepth，vehicle=车载 YOLO 深度", "scope": "shared"},
                "decision_threshold_m": {"type": "number", "exclusiveMinimum": 0, "default": 2.0, "description": "近障判定距离(米)：估算距离小于该值时 near_obstacle=true", "scope": "shared"},
            },
        },
        "topic_in":  [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [{"format": "data/json",  "desc": "obstacle distance estimation result"}],
    }
]


# ── Distance Estimation Pipeline ──────────────────────────────────────────────


class LocalDistanceAdapter:
    """本地深度 + 分割管线（基于 obstacle_distance_core 估算器）。

    加载深度与分割后端，由 ObstacleDistanceEstimator
    计算最近障碍物距离。
    模型模式初始化失败时直接抛错，禁止用随机值伪装成有效预测。
    """

    def __init__(self, cfg: dict):
        from .obstacle_distance_core.estimator import ObstacleDistanceEstimator
        from .obstacle_distance_core.zipdepth_tensorrt_backends import (
            create_backends,
        )
        from utils.model_downloader import ensure_obstacle_models

        self._cfg = deepcopy(cfg or {})
        scene_mode = self._cfg.get("scene_mode", "fixed")
        environment_scene = os.environ.get("OBSTACLE_FIXED_SCENE")
        self._resolution_scene_map: dict[tuple[int, int], str] = {}
        if environment_scene or scene_mode == "fixed":
            configured_scene = (
                environment_scene
                or self._cfg.get("fixed_scene")
                or self._cfg.get("scene_hint")
            )
            try:
                self._scene_hint = SceneDomain(
                    str(configured_scene).strip().lower()
                ).value
            except Exception:
                raise ValueError(
                    "fixed_scene must be indoor or vehicle"
                ) from None
            self._scene_mode = "fixed"
        elif scene_mode == "resolution":
            configured_map = self._cfg.get("resolution_scene_map")
            if not isinstance(configured_map, dict) or not configured_map:
                raise ValueError(
                    "resolution_scene_map must be a nonempty mapping"
                )
            for resolution, configured_scene in configured_map.items():
                try:
                    width_text, height_text = str(resolution).lower().split("x")
                    width = int(width_text)
                    height = int(height_text)
                    scene = SceneDomain(
                        str(configured_scene).strip().lower()
                    ).value
                except Exception:
                    raise ValueError(
                        "resolution_scene_map entries must use WIDTHxHEIGHT "
                        "and indoor or vehicle"
                    ) from None
                if width <= 0 or height <= 0:
                    raise ValueError(
                        "resolution_scene_map dimensions must be positive"
                    )
                key = (width, height)
                if key in self._resolution_scene_map:
                    raise ValueError(
                        "resolution_scene_map contains duplicate dimensions"
                    )
                self._resolution_scene_map[key] = scene
            self._scene_hint = None
            self._scene_mode = "resolution"
            # Unknown resolutions must produce missing_scene rather than silently
            # falling back to a stale fixed_scene value.
            self._cfg["fixed_scene"] = None
        else:
            raise ValueError(
                "obstacle ROS input has no scene metadata; scene_mode must be "
                "fixed or resolution"
            )
        engine_paths = ensure_obstacle_models(
            self._cfg.get("model_dir", "/models/obstacle/zipdepth-int8")
        )
        for key, filename in (
            ("indoor_depth_engine", "zipdepth-base-npu-512x384-int8.engine"),
            ("vehicle_depth_engine", "yolo26n-depth-int8.engine"),
            ("segmentation_engine", "yolo26n-seg-int8.engine"),
        ):
            self._cfg.setdefault(key, engine_paths[filename])
        depth_backend, segmentation_backend = create_backends(self._cfg)
        self._backends = (depth_backend, segmentation_backend)
        self._estimator = ObstacleDistanceEstimator(
            depth_backend,
            segmentation_backend,
            self._cfg,
        )

    def close(self) -> None:
        """Release the TensorRT engines held by the backends."""
        backends = getattr(self, "_backends", ())
        self._backends = ()
        for backend in backends:
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    log.debug("[obstacle] backend close failed", exc_info=True)

    def estimate(self, image_bytes: bytes) -> dict:
        """估算最近障碍物距离。"""
        scene_hint = self._scene_hint
        estimator_image_bytes = image_bytes
        if self._scene_mode == "resolution":
            try:
                import cv2
                import numpy as np

                image = cv2.imdecode(
                    np.frombuffer(image_bytes, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
            except Exception:
                image = None
            if image is None or image.size == 0:
                # Preserve the estimator's structured invalid_image fallback.
                estimator_image_bytes = b""
            else:
                height, width = image.shape[:2]
                scene_hint = self._resolution_scene_map.get((width, height))
        result = self._estimator.estimate(
            estimator_image_bytes,
            scene_hint=scene_hint,
        )
        return {
            "pred_distance": result.distance_m,
            "distance_m": result.distance_m,
            "near_obstacle": result.near_obstacle,
            "scene": result.scene,
            "status": result.status,
            "error_code": result.error_code,
            "fallback": result.fallback,
            "approximate_geometry": result.approximate_geometry,
            "latency_ms": result.latency_ms,
        }


def _build_distance_adapter(cfg: dict) -> LocalDistanceAdapter:
    if cfg.get("provider", "local") != "local":
        raise ValueError("obstacle provider must be local")
    return LocalDistanceAdapter(cfg)


# ── ROS2 Node (one per instance/topic) ────────────────────────────────────────

class _ObstacleNode(Node):
    """Per-topic obstacle distance estimation node."""

    def __init__(self, input_topic: str, node_suffix: str):
        super().__init__(f"obstacle_{node_suffix}")
        self._input_topic = input_topic
        self._output_topic = f"{input_topic}/obstacle"
        self._pub = self.create_publisher(String, self._output_topic, _PUB_QOS)
        self._sub: object | None = None
        # Latest frame wins: the camera callback overwrites, the worker pops.
        # A FIFO here would make the estimator report stale frames whenever
        # inference falls behind the camera.
        self._frames: LatestFrame = LatestFrame()
        self._frames.close()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()
        self._detect_count = 0
        self._logged_first_frame = False
        self._log_gate = SampledLogGate(every=100)
        self.state = "idle"

    def start(self, adapter: LocalDistanceAdapter) -> dict:
        with self._lifecycle_lock:
            if self.state == "running":
                return {"state": "running", "input": self._input_topic, "output": self._output_topic}
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("previous obstacle worker is still stopping")
            if self._sub is None:
                self._sub = self.create_subscription(
                    CompressedImage, self._input_topic, self._image_cb, CAMERA_QOS
                )
            # Use a distinct event and frame slot for every worker generation.
            # If an old inference ever outlives stop()'s join timeout, a later
            # start must not clear that worker's stop signal and revive it.
            stop_event = threading.Event()
            frames: LatestFrame = LatestFrame()
            self._stop_event = stop_event
            self._frames = frames
            self._worker = threading.Thread(
                target=self._inference_worker,
                args=(stop_event, frames, adapter),
                daemon=True,
                name=f"obstacle_worker_{self._input_topic}",
            )
            self._worker.start()
            self.state = "running"
        log.info(f"[obstacle] started: {self._input_topic} -> {self._output_topic}")
        return {"state": "running", "input": self._input_topic, "output": self._output_topic}

    def stop(self) -> dict:
        # stop() only pauses the inference worker; the node itself stays
        # registered until the plugin retires it (see ObstacleDistancePlugin.
        # _dispose_node), so a later start on the same topic reuses it.
        with self._lifecycle_lock:
            self.state = "idle"
            self._stop_event.set()
            self._frames.close()  # drops the pending frame and wakes the worker
            worker = self._worker
            if worker and worker.is_alive():
                worker.join(timeout=3.0)
            if worker is None or not worker.is_alive():
                self._worker = None
            else:
                log.warning(
                    "[obstacle] worker still stopping after timeout: %s",
                    self._input_topic,
                )
        log.info(f"[obstacle] stopped: {self._input_topic}")
        return {"state": "idle", "input": self._input_topic}

    def _image_cb(self, msg: CompressedImage):
        if self.state != "running":
            return
        try:
            image_bytes = bytes(msg.data)
        except Exception:
            log.warning(
                "[obstacle] received image frame with invalid data on %s",
                self._input_topic,
            )
            return
        if not self._logged_first_frame:
            # First frame only: msg.format is externally supplied — escape and
            # cap it, and never put it in a per-frame log.
            self._logged_first_frame = True
            log.info(
                "[obstacle] first image frame on %s: size=%d bytes, format=%.32r",
                self._input_topic, len(image_bytes), str(msg.format),
            )
        # Latest frame wins (no backpressure, no history).
        self._frames.push(image_bytes)

    def _inference_worker(
        self,
        stop_event: threading.Event,
        frames: LatestFrame,
        adapter: LocalDistanceAdapter,
    ):
        while not stop_event.is_set():
            jpeg_bytes = frames.pop(timeout=1.0)
            if jpeg_bytes is None:
                continue
            if stop_event.is_set():
                break
            try:
                result = adapter.estimate(jpeg_bytes)
                # Hot path: log state transitions unthrottled, sample steady
                # state (first + every 100th) so camera-rate output cannot
                # flood the container logs.
                outcome = "fallback" if result.get("fallback") else "ok"
                should_log, transition, occurrence = self._log_gate.check(outcome)
                if should_log:
                    emit = log.info if transition else log.debug
                    emit(
                        "[obstacle] %s scene=%s error_code=%s latency_ms=%.1f "
                        "distance_m=%s (frame %d)",
                        outcome,
                        result.get("scene"),
                        result.get("error_code"),
                        float(result.get("latency_ms", 0.0)),
                        result.get("pred_distance"),
                        occurrence,
                    )
                if not stop_event.is_set():
                    self._publish_result(result)
            except Exception as e:
                outcome = f"error:{type(e).__name__}"
                should_log, transition, occurrence = self._log_gate.check(outcome)
                if transition:
                    # Full traceback only when the error class first appears.
                    log.error("[obstacle] inference error: %s",
                              escape_log_text(e), exc_info=True)
                elif should_log:
                    log.error("[obstacle] inference error (occurrence %d): %s",
                              occurrence, escape_log_text(e))

    def _publish_result(self, result: dict):
        self._detect_count += 1
        msg = String()
        # Publish the whole structured result the adapter produced. A bare
        # distance cannot be told apart from the estimator's fallback (
        # fallback_distance_m, 3.0 by default), and consumers also need the
        # near_obstacle/scene/status/error_code/latency fields this plugin
        # advertises as data/json. pred_distance is still present, so existing
        # subscribers keep working.
        msg.data = json.dumps(result, ensure_ascii=False)
        self._pub.publish(msg)


# ── Plugin class ──────────────────────────────────────────────────────────────

class ObstacleDistancePlugin:
    """Obstacle-distance MCP plugin with the same non-blocking
    start/stop/load state machine as the OCR plugin:

        idle --start--> loading --ok--> ready/running
                          |               ^
                          +----fail--> error (next start retries)

    * first start records the instance as pending, spawns one background
      loader and immediately returns {"state": "loading"};
    * concurrent starts only add pending instances — the engines are
      downloaded and initialised once (single-flight);
    * info is a pure query and never blocks on the loader;
    * stop during loading cancels the pending instance; stop of a live
      instance disposes its node (executor.remove_node + destroy_node);
    * config bumps a load generation so a stale loader can never install
      an adapter built from an outdated configuration.

    One shared adapter serves every instance without per-instance
    configuration; instances that do carry per-instance configuration get
    their own cached adapter, built by the same background loader (the
    verified-bundle download is deduplicated by the downloader's file lock,
    so extra instances never re-download).
    """

    PREFIX = "obstacle"

    def __init__(self, plugin_cfg: dict, executor):
        self._executor = executor
        self._plugin_cfg = deepcopy(plugin_cfg or {})
        self._provider = self._plugin_cfg.get("provider", "local")

        # Guarded by _state_lock, held for bookkeeping only — never while
        # building engines or joining workers.
        self._state_lock = threading.Lock()
        self._nodes: dict[str, _ObstacleNode] = {}
        self._instance_configs: dict[str, dict] = {}
        self._instance_adapters: dict[str, tuple[dict, LocalDistanceAdapter]] = {}
        self._pending_starts: dict[str, str] = {}   # node_key -> input_topic
        self._adapter: LocalDistanceAdapter | None = None
        self._adapter_state = "idle"                # idle|loading|ready|error
        self._load_error: str | None = None
        self._load_generation = 0
        self._loader_thread: threading.Thread | None = None

        log.info("[obstacle] plugin registered: provider=%s", self._provider)

    def get_tools(self) -> list:
        return TOOLS

    # ── helpers ───────────────────────────────────────────────────────────

    def _effective_cfg_locked(self, node_key: str) -> dict:
        merged = deepcopy(self._plugin_cfg)
        merged.update(self._instance_configs.get(node_key, {}))
        return merged

    def _dispose(self, node_key: str, node: "_ObstacleNode") -> None:
        """Stop worker and fully destroy the node. Never called under lock."""
        try:
            node.stop()
        finally:
            dispose_node(self._executor, node, label=f"obstacle/{node_key}")
        log.info(f"[obstacle] node disposed: {node_key}")

    @staticmethod
    def _close_adapter(adapter) -> None:
        if adapter is None:
            return
        close = getattr(adapter, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                log.debug("[obstacle] adapter close failed", exc_info=True)

    # ── background loader (single-flight) ────────────────────────────────

    def _spawn_loader_locked(self) -> None:
        """Ensure the one background loader/bring-up thread is running.

        Caller holds the lock. Single-flight: never two loaders. The loader
        clears self._loader_thread under the same lock right before exiting,
        so a pending entry added here is always observed by a live loader.
        """
        thread = self._loader_thread
        if thread is not None and thread.is_alive():
            return
        if self._adapter is None or self._adapter_state == "error":
            self._adapter_state = "loading"
            self._load_error = None
        generation = self._load_generation
        cfg = deepcopy(self._plugin_cfg)
        thread = threading.Thread(
            target=self._loader, args=(generation, cfg),
            name="obstacle-adapter-loader", daemon=True,
        )
        self._loader_thread = thread
        thread.start()

    def _finish_loader_locked(self, generation: int) -> None:
        """Loader exit handshake. Caller holds the lock.

        Clears the thread handle; if a config change replaced the generation
        while this loader ran and instances are still pending, respawn a
        loader for the new configuration so no pending start is stranded.
        """
        self._loader_thread = None
        if generation != self._load_generation and self._pending_starts:
            self._spawn_loader_locked()

    def _adapter_for_key(self, node_key: str, shared: LocalDistanceAdapter):
        """Return the adapter serving node_key, building the per-instance one
        on demand. Runs in the loader thread — never under the lock.

        The build happens outside the lock, so a per-instance config can
        change mid-build. The result is committed only if the effective
        config is still the one it was built for; otherwise it is closed and
        the build retried, so a node never starts on a stale adapter and a
        replaced cache entry never leaks its engine.
        """
        while True:
            with self._state_lock:
                if not self._instance_configs.get(node_key):
                    return shared
                effective = self._effective_cfg_locked(node_key)
                cached = self._instance_adapters.get(node_key)
                if cached is not None and cached[0] == effective:
                    return cached[1]
            adapter = _build_distance_adapter(effective)
            stale = None
            with self._state_lock:
                still_wanted = (
                    bool(self._instance_configs.get(node_key))
                    and self._effective_cfg_locked(node_key) == effective
                )
                if still_wanted:
                    previous = self._instance_adapters.get(node_key)
                    if previous is not None and previous[1] is not adapter:
                        stale = previous[1]
                    self._instance_adapters[node_key] = (effective, adapter)
            if still_wanted:
                if stale is not None:
                    self._close_adapter(stale)
                return adapter
            # Config changed (or the per-instance config was removed) while
            # building: discard this result and re-evaluate.
            self._close_adapter(adapter)

    def _loader(self, generation: int, cfg: dict) -> None:
        with self._state_lock:
            adapter = self._adapter if generation == self._load_generation else None
        if adapter is None:
            try:
                adapter = _build_distance_adapter(cfg)
            except Exception as error:  # noqa: BLE001 - surfaced via info
                log.exception("[obstacle] adapter load failed")
                with self._state_lock:
                    if generation == self._load_generation:
                        self._adapter_state = "error"
                        self._load_error = str(error)
                    self._finish_loader_locked(generation)
                return

            with self._state_lock:
                if generation != self._load_generation:
                    stale = adapter
                else:
                    self._adapter = adapter
                    self._adapter_state = "ready"
                    stale = None
            if stale is not None:
                self._close_adapter(stale)
                with self._state_lock:
                    self._finish_loader_locked(generation)
                return

        # Bring up every instance that is still pending.
        while True:
            with self._state_lock:
                if generation != self._load_generation or not self._pending_starts:
                    self._finish_loader_locked(generation)
                    return
                node_key, input_topic = next(iter(self._pending_starts.items()))
            # README lifecycle rule: register under the lock *before*
            # starting, so a concurrent stop (or a failure at any later
            # point) can always locate the node — a started-but-unregistered
            # worker would be unreachable and leak.
            try:
                node_adapter = self._adapter_for_key(node_key, adapter)
                suffix = node_key.replace("/", "_").replace("-", "_").lstrip("_")
                node = _ObstacleNode(input_topic, suffix)
            except Exception as error:  # noqa: BLE001 - keep serving others
                log.error("[obstacle] failed to build instance %r on %r: %s",
                          node_key, input_topic, escape_log_text(error))
                with self._state_lock:
                    if self._pending_starts.get(node_key) == input_topic:
                        del self._pending_starts[node_key]
                continue
            registered = False
            with self._state_lock:
                entry_valid = (
                    generation == self._load_generation
                    and self._pending_starts.get(node_key) == input_topic
                )
                still_wanted = entry_valid and node_key not in self._nodes
                if still_wanted:
                    try:
                        self._executor.add_node(node)
                    except Exception as error:  # noqa: BLE001
                        log.error("[obstacle] failed to register instance %r: %s",
                                  node_key, escape_log_text(error))
                        del self._pending_starts[node_key]
                    else:
                        self._nodes[node_key] = node
                        del self._pending_starts[node_key]
                        registered = True
                elif entry_valid:
                    # A concurrent start already owns a live node for this
                    # key; drop the entry so the loop can progress.
                    del self._pending_starts[node_key]
            if not registered:
                try:
                    node.destroy_node()   # never started, never registered
                except Exception:  # noqa: BLE001
                    pass
                continue
            try:
                node.start(node_adapter)
            except Exception as error:  # noqa: BLE001
                log.error("[obstacle] failed to start instance %r: %s",
                          node_key, escape_log_text(error))
                with self._state_lock:
                    if self._nodes.get(node_key) is node:
                        del self._nodes[node_key]
                self._dispose(node_key, node)
                continue
            with self._state_lock:
                still_ours = self._nodes.get(node_key) is node
            if not still_ours:
                # A concurrent stop retired the node while start() ran; its
                # dispose handled teardown — stop the worker start spawned.
                node.stop()

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
        action = args.get("action", name)
        instance_id = args.get("instance_id", "")

        if action == "info":
            return self._do_info(instance_id, args)
        if action == "start":
            return self._do_start(instance_id, args)
        if action == "stop":
            return self._do_stop(instance_id)
        if action == "config":
            return self._do_config(instance_id, args)
        return None

    _DESC = "Obstacle distance estimation from camera feed"

    def _desc_for(self, state: str, load_error: str | None) -> str:
        """Dashboard-facing description, mirroring the ASR plugin (#113): the
        static blurb normally, a reason while loading or after a failure."""
        if state == "loading":
            return "Loading obstacle models and TensorRT engines..."
        if state == "error" and load_error:
            return f"Model load failed: {load_error}"
        return self._DESC

    def _do_info(self, instance_id: str, args: dict) -> dict:
        input_topic = args.get("input_topic", "")
        if not input_topic:
            topics_list = args.get("input_topics") or []
            if topics_list:
                input_topic = topics_list[0]

        with self._state_lock:
            instances: dict[str, dict] = {}
            for key, node in self._nodes.items():
                instances[key] = {
                    "input": node._input_topic,
                    "output": node._output_topic,
                    "detect_count": node._detect_count,
                    "state": node.state,
                }
            for key, topic in self._pending_starts.items():
                if key in instances:
                    continue
                state = self._instance_state_locked(key)
                entry = {
                    "input": topic,
                    "output": f"{topic}/obstacle",
                    "detect_count": 0,
                    "state": state,
                }
                if state == "error" and self._load_error:
                    entry["error"] = self._load_error
                instances[key] = entry

            if instance_id and instance_id in instances:
                input_topic = instances[instance_id]["input"]
            elif not input_topic and instances:
                input_topic = next(iter(instances.values()))["input"]

            states = {entry["state"] for entry in instances.values()}
            if "loading" in states or self._adapter_state == "loading":
                state = "loading"
            elif "running" in states:
                state = "running"
            elif "error" in states or self._adapter_state == "error":
                state = "error"
            else:
                state = "idle"
            load_error = self._load_error

        topics_in = [{"topic": input_topic, "format": "image/jpeg"}] if input_topic else []
        topics_out = [{"topic": f"{input_topic}/obstacle", "format": "data/json"}] if input_topic else []
        result = {
            "name": "ObstacleDistance", "manufacture": "Embodied", "model": "obstacle",
            "state": state,
            "instances": instances,
            "topic_in": topics_in,
            "topic_out": topics_out,
            "desc": self._desc_for(state, load_error),
        }
        if state == "error" and load_error:
            result["error"] = load_error
        return result

    def _do_start(self, instance_id: str, args: dict) -> dict:
        input_topic = args.get("input_topic")
        if not input_topic:
            topics_list = args.get("input_topics") or []
            if topics_list:
                input_topic = topics_list[0]
        if not input_topic:
            raise ValueError("input_topic is required")
        node_key = instance_id or input_topic

        retired = None
        with self._state_lock:
            previous = self._nodes.get(node_key)
            if previous is not None and previous._input_topic != input_topic:
                # Topics are fixed at node construction; rebind the key.
                retired = self._nodes.pop(node_key)
        if retired is not None:
            self._dispose(node_key, retired)

        start_node = None
        with self._state_lock:
            existing = self._nodes.get(node_key)
            instance_cfg = self._instance_configs.get(node_key)
            cached = self._instance_adapters.get(node_key)
            adapter_available = self._adapter_state == "ready" and (
                not instance_cfg
                or (cached is not None and cached[0] == self._effective_cfg_locked(node_key))
            )
            if existing is not None:
                start_node = existing        # idempotent re-start
                shared = self._adapter
            elif adapter_available:
                # Claim the key before leaving the lock: a concurrent stop
                # must always find the instance in _pending_starts or
                # _nodes, never in an invisible in-between state.
                self._pending_starts[node_key] = input_topic
                shared = self._adapter
                generation = self._load_generation
            else:
                # Cold start, error retry, or an instance whose per-instance
                # adapter is not built yet — all go through the background
                # loader so start never blocks on engine initialisation.
                self._pending_starts[node_key] = input_topic
                self._spawn_loader_locked()
                return {
                    "state": "loading",
                    "input": input_topic,
                    "output": f"{input_topic}/obstacle",
                }

        node_adapter = self._adapter_for_key(node_key, shared)
        if start_node is not None:
            return start_node.start(node_adapter)

        suffix_source = node_key if retired is None else f"{node_key}_{input_topic}"
        suffix = suffix_source.replace("/", "_").replace("-", "_").lstrip("_")
        node = _ObstacleNode(input_topic, suffix)
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
                    log.error("[obstacle] failed to register instance %r: %s",
                              node_key, escape_log_text(error))
                    del self._pending_starts[node_key]
                else:
                    self._nodes[node_key] = node
                    del self._pending_starts[node_key]
                    registered = True
            elif claimed and not fresh:
                # A config change invalidated the adapter mid-start; keep the
                # claim so the loader brings this instance up on the new one.
                pass
            elif claimed:
                del self._pending_starts[node_key]
        if not registered:
            try:
                node.destroy_node()   # never started, never registered
            except Exception:  # noqa: BLE001
                pass
            if current is not None:
                return current.start(node_adapter)
            if claimed and not fresh:
                return {"state": "loading", "input": input_topic,
                        "output": f"{input_topic}/obstacle"}
            return {"state": "idle", "input": input_topic,
                    "output": f"{input_topic}/obstacle"}
        try:
            result = node.start(node_adapter)
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
                    "output": f"{input_topic}/obstacle"}
        return result

    def _do_stop(self, instance_id: str) -> dict:
        to_dispose: list[tuple[str, _ObstacleNode]] = []
        with self._state_lock:
            if instance_id:
                self._pending_starts.pop(instance_id, None)
                node = self._nodes.pop(instance_id, None)
                if node is not None:
                    to_dispose.append((instance_id, node))
                stopped = [instance_id]
            else:
                self._pending_starts.clear()
                stopped = list(self._nodes)
                to_dispose.extend(self._nodes.items())
                self._nodes = {}
        for node_key, node in to_dispose:
            self._dispose(node_key, node)
        if instance_id:
            return {"state": "idle"}
        return {"state": "idle", "stopped_instances": stopped}

    def _do_config(self, instance_id: str, args: dict) -> dict:
        cfg = {
            k: v for k, v in args.items()
            if k not in ('action', 'instance_id') and v is not None and v != ''
        }

        if instance_id:
            retired = None
            stale_adapter = None
            with self._state_lock:
                previous_cfg = self._instance_configs.get(instance_id, {})
                merged_cfg = deepcopy(previous_cfg)
                merged_cfg.update(cfg)
                self._instance_configs[instance_id] = merged_cfg
                retired = self._nodes.pop(instance_id, None)
                if merged_cfg != previous_cfg:
                    dropped = self._instance_adapters.pop(instance_id, None)
                    if dropped is not None:
                        stale_adapter = dropped[1]
            if retired is not None:
                # The instance goes back to idle; the next start rebuilds it
                # with the merged per-instance configuration.
                self._dispose(instance_id, retired)
            self._close_adapter(stale_adapter)
            return {
                "status": "configured",
                "instance_id": instance_id,
                "config": merged_cfg,
            }

        with self._state_lock:
            merged_cfg = deepcopy(self._plugin_cfg)
            merged_cfg.update(cfg)
            if merged_cfg == self._plugin_cfg:
                log.debug("[obstacle] configuration unchanged")
                return {"status": "configured", "config": cfg}

            self._plugin_cfg = merged_cfg
            self._provider = merged_cfg.get("provider", "local")
            # Invalidate any in-flight load and every cached adapter; nodes
            # built on the old adapters are retired. Pending starts stay
            # pending and are served by a fresh loader for the new config.
            self._load_generation += 1
            stale_adapters = [self._adapter]
            stale_adapters.extend(
                adapter for _cfg, adapter in self._instance_adapters.values()
            )
            self._adapter = None
            self._instance_adapters = {}
            disposed = list(self._nodes.items())
            self._nodes = {}
            if self._pending_starts:
                self._spawn_loader_locked()
            else:
                self._adapter_state = "idle"
                self._load_error = None
        for node_key, node in disposed:
            self._dispose(node_key, node)
        for adapter in stale_adapters:
            self._close_adapter(adapter)
        log.info("[obstacle] configuration updated")
        return {"status": "configured", "config": cfg}
