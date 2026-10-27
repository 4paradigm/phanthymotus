"""Generic object measurement from VOP boxes and metric depth.

The card is a **processor**, not an actuator: it reads what perception already
publishes, turns a detection box into metres, and publishes a JSON observation.
It never commands a robot.

Why this is not "another navigation card"
-----------------------------------------

Navigation asks "is the way ahead blocked?". Measurement asks "how wide is this
thing?". They both use depth, but their failure modes are different: a stale
navigation corridor can stop a chassis; a stale measurement is merely a number
that must not be trusted. The card therefore publishes `fresh`/quality fields
and never turns an unmeasured dimension into zero.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass

from . import camera as camera_mod
from . import geometry
from . import view as view_mod

log = logging.getLogger(__name__)

SCHEMA = "motus.measure/1"
PREFIX = "measure"
DEFAULT_TOPIC = "/perception/measure/state"
TOPIC_FORMAT = "data/json"

# VOP's output suffixes. A hardware depth topic may be `/camera/depth` or a
# namespaced head camera; the same suffix order is used by navi, so a canvas
# wired for navigation does not need a second naming convention here.
_OBJECTS_SUFFIX = "/objects"
_DEPTH_SUFFIXES = ("/visual_depth", "/camera/head/depth", "/camera/depth", "/depth")


def _sensor_qos():
    """BEST_EFFORT, matching VOP and every depth producer in this project.

    A RELIABLE subscriber paired with a BEST_EFFORT publisher silently receives
    nothing in ROS2. The failure looks like a broken camera, not a QoS mismatch.
    """
    from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                           ReliabilityPolicy)

    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      history=HistoryPolicy.KEEP_LAST,
                      depth=2,
                      durability=DurabilityPolicy.VOLATILE)


@dataclass
class Config:
    rate_hz: float = 5.0
    max_objects: int = 8
    min_confidence: float = 0.3
    depth_percentile: float = geometry.DEPTH_PERCENTILE
    max_depth_m: float = 20.0
    max_obs_age_ms: int = 500
    stable_frames: int = 5
    stable_min: int = 3
    track_timeout_ms: int = 2000
    match_distance_norm: float = geometry.DEFAULT_MATCH_DISTANCE
    history_size: int = 200
    record_interval_ms: int = 1000
    record_min_delta_m: float = 0.01
    only_stable: bool = True
    summary_max_objects: int = 5
    half_fov_rad: float | None = None
    half_fov_v_rad: float | None = None

    @classmethod
    def from_dict(cls, values: dict | None) -> "Config":
        values = values or {}
        config = cls()
        for key in ("rate_hz", "min_confidence", "depth_percentile",
                    "max_depth_m", "match_distance_norm", "record_min_delta_m",
                    "half_fov_rad", "half_fov_v_rad"):
            if key in values and values[key] is not None:
                setattr(config, key, float(values[key]))
        for key in ("max_objects", "max_obs_age_ms", "stable_frames",
                    "stable_min", "track_timeout_ms", "history_size",
                    "record_interval_ms", "summary_max_objects"):
            if key in values and values[key] is not None:
                setattr(config, key, int(values[key]))
        if "only_stable" in values:
            config.only_stable = bool(values["only_stable"])
        config.validate()
        return config

    def validate(self) -> None:
        if not 0.5 <= self.rate_hz <= 30:
            raise ValueError("measure.rate_hz must be in [0.5, 30]")
        if not 1 <= self.max_objects <= 50:
            raise ValueError("measure.max_objects must be in [1, 50]")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("measure.min_confidence must be in [0, 1]")
        if not 1 <= self.depth_percentile <= 99:
            raise ValueError("measure.depth_percentile must be in [1, 99]")
        if not 0 < self.max_depth_m <= 100:
            raise ValueError("measure.max_depth_m must be in (0, 100]")
        if not 50 <= self.max_obs_age_ms <= 5000:
            raise ValueError("measure.max_obs_age_ms must be in [50, 5000]")
        if not 1 <= self.stable_frames <= 50:
            raise ValueError("measure.stable_frames must be in [1, 50]")
        if not 1 <= self.stable_min <= self.stable_frames:
            raise ValueError("measure.stable_min must be in [1, stable_frames]")
        if not 0 <= self.track_timeout_ms <= 60000:
            raise ValueError("measure.track_timeout_ms must be in [0, 60000]")
        if not 0 <= self.match_distance_norm <= 1:
            raise ValueError("measure.match_distance_norm must be in [0, 1]")
        if not 1 <= self.history_size <= 5000:
            raise ValueError("measure.history_size must be in [1, 5000]")
        if not 0 <= self.record_interval_ms <= 60000:
            raise ValueError("measure.record_interval_ms must be in [0, 60000]")
        if not 0 <= self.record_min_delta_m <= 10:
            raise ValueError("measure.record_min_delta_m must be in [0, 10]")
        if not 1 <= self.summary_max_objects <= 100:
            raise ValueError("measure.summary_max_objects must be in [1, 100]")
        for name in ("half_fov_rad", "half_fov_v_rad"):
            angle = getattr(self, name)
            if angle is not None and not 0 < angle < math.pi / 2:
                raise ValueError(f"measure.{name} must be in (0, π/2)")

    def as_dict(self) -> dict:
        return {
            "rate_hz": self.rate_hz,
            "max_objects": self.max_objects,
            "min_confidence": self.min_confidence,
            "depth_percentile": self.depth_percentile,
            "max_depth_m": self.max_depth_m,
            "max_obs_age_ms": self.max_obs_age_ms,
            "stable_frames": self.stable_frames,
            "stable_min": self.stable_min,
            "track_timeout_ms": self.track_timeout_ms,
            "match_distance_norm": self.match_distance_norm,
            "history_size": self.history_size,
            "record_interval_ms": self.record_interval_ms,
            "record_min_delta_m": self.record_min_delta_m,
            "only_stable": self.only_stable,
            "summary_max_objects": self.summary_max_objects,
        }


def _cm(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number * 100.0, 1)


def _round(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 2)


def _compact_camera(camera) -> dict:
    return {
        "width": camera.width,
        "height": camera.height,
        "fx": round(camera.fx, 1) if camera.fx else None,
        "fy": round(camera.fy, 1) if camera.fy else None,
        "source": camera.source,
    }


class MeasurePlugin:
    PREFIX = PREFIX

    def __init__(self, plugin_cfg: dict, executor, namespace: str = ""):
        self._cfg = dict(plugin_cfg or {})
        self._executor = executor
        self._namespace = (namespace or "").strip("/")
        self._topic = str(self._cfg.get("topic") or DEFAULT_TOPIC)
        self._view_topic = str(self._cfg.get("view_topic") or
                               (self._topic.rsplit("/", 1)[0] + "/view"))
        self._view_hz = float(self._cfg.get("view_hz", 5.0))
        if not 0 <= self._view_hz <= 15:
            raise ValueError("measure.view_hz must be in [0, 15]")
        self._camera_topic_cfg = str(self._cfg.get("camera_topic") or "").strip()
        self._camera_topic = ""
        self._image_jpeg = None
        self._image_ms = 0
        self._view_publisher = None
        self._view_queue = None
        self._view_thread = None
        self._view_stop = threading.Event()
        self._view_next_at = 0.0
        self._view_failed = False
        self._config = Config.from_dict(self._cfg)

        self._lock = threading.RLock()
        self._obs_lock = threading.RLock()
        self._running = False
        self._node = None
        self._publisher = None
        self._timer = None
        self._binding = {}
        self._objects = None
        self._objects_ms = 0
        self._depth = None
        self._depth_ms = 0
        self._objects_frame = (geometry.WIDTH, geometry.HEIGHT)
        self._depth_frame = (geometry.WIDTH, geometry.HEIGHT)
        self._camera = camera_mod.CameraModel()
        self._camera_notes: list[str] = []
        self._published = 0
        self._last = None
        self._last_error = ""
        self._stabilizer = self._new_stabilizer()
        self._session = self._new_session()

    # ── tool ─────────────────────────────────────────────────────────────────

    def get_tools(self) -> list:
        return [{
            "name": PREFIX,
            "type": "processor",
            "multiInstance": False,
            "description": (
                "通用物体测量：接 VOP 检测结果和一路米制深度图，输出每个目标的"
                "距离、可见宽度和可见高度，并在可选的 image/jpeg 输出口直接画出"
                "bbox 和尺寸标签。宽度/高度是当前视角下的可见外廓，不是不可见背面"
                "的完整三维尺寸；后者需要分割和点云/多视角。"
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["start", "stop", "info", "config",
                                 "read", "history", "reset_session"],
                    },
                    "target": {
                        "type": "string",
                        "description": "read 的可选名字过滤，例如 laptop / person",
                    },
                },
                "required": ["action"],
                # start/stop are canvas lifecycle, not LLM functions. config is
                # a gear action; only info is useful to an agent.
                "x-action-params": {
                    "info": {"params": [], "description": "返回最近一次测量、本次会话摘要和降级信息"},
                    "history": {"params": [], "description": "返回本次会话最近的记录（有界、按变化/时间节流）"},
                    "reset_session": {"params": [], "description": "清空本次会话的汇总和历史，不停止测量"},
                    "read": {"params": ["target"],
                             "description": "读取当前帧的测量，可按名字过滤；不读历史"},
                    "config": {"params": [], "description": "更新测量阈值"},
                },
            },
            "topic_in": [
                {"format": "data/json",
                 "desc": "vop 检测结果（必需，含 bbox）"},
                {"format": "image/depth-zlib",
                 "desc": "与检测同一镜头的米制深度图（必需）"},
            ],
            "topic_out": ([{
                "topic": self._topic,
                "format": TOPIC_FORMAT,
                "desc": "每个目标的距离/可见宽高、质量与降级信息",
            }] + ([{
                "topic": self._view_topic,
                "format": "image/jpeg",
                "desc": "camera 画面 + 测量 bbox 和 W×H 标签；只给人看",
            }] if self._view_hz > 0 else [])),
            "configSchema": {
                "type": "object",
                "properties": {
                    "rate_hz": {"type": "number", "default": 5.0,
                                "description": "发布频率，Hz", "scope": "instance"},
                    "max_objects": {"type": "integer", "default": 8,
                                    "description": "每帧最多测量几个目标", "scope": "instance"},
                    "min_confidence": {"type": "number", "default": 0.3,
                                       "description": "低于此置信度的检测不测", "scope": "instance"},
                    "depth_percentile": {"type": "number", "default": 25,
                                         "description": "bbox 内深度的分位数，越小越偏向最近表面",
                                         "scope": "instance"},
                    "max_depth_m": {"type": "number", "default": 20.0,
                                    "description": "超过这个距离不输出尺寸", "scope": "instance"},
                    "stable_frames": {"type": "integer", "default": 5,
                                      "description": "平滑窗口帧数", "scope": "instance"},
                    "stable_min": {"type": "integer", "default": 3,
                                   "description": "至少多少帧才标记 stable", "scope": "instance"},
                    "track_timeout_ms": {"type": "integer", "default": 2000,
                                         "description": "目标丢失后保留平滑结果的时间",
                                         "scope": "instance"},
                    "history_size": {"type": "integer", "default": 200,
                                     "description": "history 动作保留的最近记录条数",
                                     "scope": "instance"},
                    "record_interval_ms": {"type": "integer", "default": 1000,
                                           "description": "同一目标最少多久记一条历史；0=每帧变化都记",
                                           "scope": "instance"},
                    "record_min_delta_m": {"type": "number", "default": 0.01,
                                           "description": "宽/高变化超过这个值才额外记一条",
                                           "scope": "instance"},
                    "only_stable": {"type": "boolean", "default": True,
                                    "description": "只把 stable 的测量计入会话摘要和历史",
                                    "scope": "instance"},
                    "summary_max_objects": {"type": "integer", "default": 5,
                                            "description": "会话摘要最多保留几个对象",
                                            "scope": "instance"},
                },
                "required": [],
            },
        }]

    def dispatch(self, name: str, args: dict | None) -> dict | None:
        args = args or {}
        action = args.get("action") or name
        if action == "start":
            return self._start(args)
        if action == "stop":
            return self._stop()
        if action == "info":
            return self._info()
        if action == "read":
            return self._read(args)
        if action == "history":
            return self._history()
        if action == "reset_session":
            return self._reset_session()
        if action == "config":
            return self._config_action(args)
        return None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Bundle lifecycle is inert: a container restart must not start measuring.

        The canvas starts the project and supplies the input topics.
        """

    def stop(self) -> None:
        self._stop()

    def _start(self, args: dict) -> dict:
        if self._executor is None:
            return self._error("没有 ROS 上下文，无法订阅 VOP/深度")

        inputs = list(args.get("input_topics") or [])
        single = args.get("input_topic")
        if single and single not in inputs:
            inputs.insert(0, single)
        if not inputs:
            return self._error("缺少 input_topic —— 请把 VOP 和深度图连到本卡片")

        with self._lock:
            if self._running:
                return self._error("already running")
            self._running = True
            self._published = 0
            self._last = None
            self._last_error = ""
            self._objects = None
            self._objects_ms = 0
            self._depth = None
            self._depth_ms = 0
            self._image_jpeg = None
            self._image_ms = 0
            self._camera_topic = ""
            self._view_failed = False
            self._binding = {}
            self._camera_notes = []
            self._stabilizer.reset()
            self._session = self._new_session()
            self._session.reset(int(time.time() * 1000))

        try:
            self._open_publisher()
            binding, problem = self._bind_inputs(
                self._node, inputs, args.get("camera_info"),
                args.get("input_bindings"))
        except Exception as error:                            # noqa: BLE001
            self._unwind()
            return self._error(f"启动失败：{error}")
        if problem:
            self._unwind()
            return self._error(problem)

        camera_problem = self._adopt_camera(args.get("camera_info"), binding)
        if camera_problem:
            self._unwind()
            return self._error(camera_problem)

        self._binding = binding
        log.info("measure started: topic=%s %.1f Hz inputs=%s",
                 self._topic, self._config.rate_hz, binding)
        return {
            "state": "running",
            "topic": self._topic,
            "rate_hz": self._config.rate_hz,
            "inputs": binding,
            "degraded": self._degradations(),
        }

    def _stop(self) -> dict:
        with self._lock:
            if not self._running:
                return {"state": "idle"}
            self._running = False
            node, timer = self._node, self._timer
            self._node = None
            self._timer = None
            self._publisher = None
            self._view_publisher = None
        self._stop_view_thread()
        if timer is not None and node is not None:
            try:
                node.destroy_timer(timer)
            except Exception:                                 # noqa: BLE001
                pass
        if node is not None:
            try:
                if self._executor is not None:
                    self._executor.remove_node(node)
            except Exception:                                 # noqa: BLE001
                pass
            try:
                node.destroy_node()
            except Exception:                                 # noqa: BLE001
                pass
        return {"state": "idle"}

    def _unwind(self) -> None:
        self._stop()

    # ── wiring ───────────────────────────────────────────────────────────────

    def _open_publisher(self) -> None:
        from rclpy.node import Node
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String

        node = Node(f"measure_{abs(hash(self._topic)) % 100000}")
        self._publisher = node.create_publisher(String, self._topic, 10)
        if self._view_hz > 0:
            self._view_publisher = node.create_publisher(
                CompressedImage, self._view_topic, 2)
            self._view_queue = queue.Queue(maxsize=1)
            self._view_stop = threading.Event()
            self._view_thread = threading.Thread(
                target=self._view_worker, daemon=True, name="measure_view")
            self._view_thread.start()
        self._timer = node.create_timer(1.0 / max(self._config.rate_hz, 0.1),
                                        self._tick)
        self._executor.add_node(node)
        self._node = node

    def _bind_inputs(self, node, topics, declarations=None, input_bindings=None):
        bound = {"objects": "", "depth": "", "unknown": []}
        for topic in topics:
            role = self._role_of(topic)
            if not role:
                bound["unknown"].append(topic)
                continue
            if not bound[role]:
                bound[role] = topic

        missing = []
        if not bound["objects"]:
            missing.append("VOP 检测结果")
        if not bound["depth"]:
            missing.append("深度图")
        if missing:
            return bound, ("测量需要的输入没有连上：" + "、".join(missing)
                           + "。请把 VOP 和深度图接到本卡片。")

        depth_format = self._declared_format(input_bindings, bound["depth"])
        if depth_format and depth_format != "image/depth-zlib":
            return bound, (f"深度输入格式是 {depth_format!r}，本卡片需要 "
                           "image/depth-zlib（640x480 uint16 毫米，zlib 压缩）")

        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String

        qos = _sensor_qos()
        node.create_subscription(String, bound["objects"], self._on_objects, qos)
        node.create_subscription(CompressedImage, bound["depth"], self._on_depth, qos)
        if self._view_hz > 0:
            self._bind_camera(node, bound["depth"], qos)
        return bound, ""

    def _bind_camera(self, node, depth_topic: str, qos) -> None:
        """Subscribe to the colour stream that belongs to the depth camera.

        The canvas cannot fan one camera node out to two cards, so this is a
        direct ROS subscription rather than a fourth canvas input. It is
        derived from the depth topic (same camera, sibling colour port) and can
        be overridden with `camera_topic` in config.yaml.
        """
        from sensor_msgs.msg import CompressedImage

        topic = self._camera_topic_cfg
        if not topic:
            candidates = self._camera_candidates(depth_topic)
            graph = dict(node.get_topic_names_and_types())
            for candidate in candidates:
                types = graph.get(candidate) or []
                if any(name.endswith("CompressedImage") for name in types):
                    topic = candidate
                    break
            if not topic and candidates:
                # The producer may not have appeared yet. Subscribe to the most
                # likely sibling; a late publisher is fine, a wrong one yields
                # no frames and the overlay simply stays absent.
                topic = candidates[0]
        if not topic:
            self._camera_notes.append(
                "没有从深度话题推导出相机彩色话题，叠加图关闭；"
                "可在 config.yaml 设置 measure.camera_topic")
            return
        try:
            node.create_subscription(CompressedImage, topic, self._on_camera, qos)
        except Exception as error:                            # noqa: BLE001
            self._camera_notes.append(f"相机叠加订阅失败（{topic}）：{error}")
            return
        self._camera_topic = topic

    @staticmethod
    def _camera_candidates(depth_topic: str) -> list[str]:
        if depth_topic.endswith("/visual_depth"):
            base = depth_topic[:-len("/visual_depth")]
            return [base, base + "/color", base + "/rgb"]
        if depth_topic.endswith("/camera/head/depth"):
            base = depth_topic[:-len("/depth")]
            return [base + "/color", base + "/rgb", base]
        if depth_topic.endswith("/camera/depth"):
            base = depth_topic[:-len("/depth")]
            return [base + "/color", base + "/rgb"]
        if depth_topic.endswith("/depth"):
            base = depth_topic[:-len("/depth")]
            return [base + "/color", base + "/rgb", base]
        return []

    def _on_camera(self, message) -> None:
        data = bytes(getattr(message, "data", b"") or b"")
        if not data:
            return
        with self._obs_lock:
            self._image_jpeg = data
            self._image_ms = int(time.time() * 1000)

    def _stop_view_thread(self) -> None:
        self._view_stop.set()
        thread = self._view_thread
        self._view_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._view_publisher = None
        self._view_queue = None

    def _queue_view(self, payload: dict) -> None:
        if (self._view_queue is None or self._view_hz <= 0
                or not self._running):
            return
        now = time.monotonic()
        if now < self._view_next_at:
            return
        self._view_next_at = now + 1.0 / self._view_hz
        with self._obs_lock:
            jpeg = self._image_jpeg
            image_ms = self._image_ms
        if not jpeg or int(time.time() * 1000) - image_ms > self._config.max_obs_age_ms:
            return
        snapshot = {"jpeg": jpeg, "objects": payload.get("objects") or []}
        try:
            self._view_queue.put_nowait(snapshot)
        except queue.Full:
            try:
                self._view_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._view_queue.put_nowait(snapshot)
            except queue.Full:
                pass

    def _view_worker(self) -> None:
        while not self._view_stop.is_set():
            try:
                snapshot = self._view_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                jpeg = view_mod.render(snapshot["jpeg"], snapshot["objects"])
            except Exception as error:                        # noqa: BLE001
                if not self._view_failed:
                    self._view_failed = True
                    log.warning("measure overlay disabled: %s", error)
                continue
            publisher = self._view_publisher
            if publisher is None:
                continue
            try:
                from sensor_msgs.msg import CompressedImage

                message = CompressedImage()
                message.format = "jpeg"
                message.data = jpeg
                publisher.publish(message)
            except Exception as error:                        # noqa: BLE001
                if not self._view_failed:
                    self._view_failed = True
                    log.warning("measure overlay publish failed: %s", error)

    def _adopt_camera(self, declarations, binding: dict) -> str:
        depth_topic = binding.get("depth")
        objects_topic = binding.get("objects")
        depth_decl = camera_mod.for_topic(declarations, depth_topic) if depth_topic else {}
        objects_decl = camera_mod.for_topic(declarations, objects_topic) if objects_topic else {}

        depth_id = camera_mod.camera_id(depth_decl)
        objects_id = camera_mod.camera_id(objects_decl)
        if depth_id and objects_id and depth_id != objects_id:
            return (f"检测结果来自 {objects_id}，深度图来自 {depth_id}。"
                    "两个不同的镜头不能用同一个相机模型换算尺寸；"
                    "请把 VOP 和深度接到同一颗相机上")

        object_size = self._declared_size(objects_decl)
        declared_depth_size = self._declared_size(depth_decl)
        # `image/depth-zlib` is a fixed 640x480 uint16 wire format. The camera
        # declaration can describe a 1280x720 RGB stream; using that size to
        # decode a 640x480 depth buffer would reject every frame. Keep the
        # wire size for decode, and use the declaration's geometry for metres.
        depth_size = (geometry.WIDTH, geometry.HEIGHT)
        if object_size is None:
            object_size = depth_size
            self._camera_notes.append(
                "VOP 没有声明相机分辨率，bbox 按 640x480 深度图归一化 —— "
                "如果 VOP 使用另一个分辨率，宽高会偏")
        if declared_depth_size and declared_depth_size != depth_size:
            self._camera_notes.append(
                f"相机声明是 {declared_depth_size[0]}x{declared_depth_size[1]}，"
                "但 image/depth-zlib 按 640x480 契约解码；尺寸仍按声明的内参换算")

        camera_size = (declared_depth_size or object_size or depth_size)
        camera = camera_mod.from_declaration(
            depth_decl or objects_decl,
            fallback_width=camera_size[0],
            fallback_height=camera_size[1],
            fallback_half_fov=self._config.half_fov_rad,
            fallback_half_fov_v=self._config.half_fov_v_rad,
        )
        self._objects_frame = object_size
        self._depth_frame = depth_size
        self._camera = camera
        self._camera_notes.extend(camera.notes)
        if not camera.has_width:
            self._camera_notes.append(
                "没有可用的水平相机几何 —— 只输出距离，visible_width_m 为 null")
        if not camera.has_height:
            self._camera_notes.append(
                "没有可用的垂直相机几何 —— 只输出距离，visible_height_m 为 null")
        return ""

    @staticmethod
    def _declared_size(declaration: dict):
        if not isinstance(declaration, dict):
            return None
        try:
            width = int(declaration.get("width") or 0)
            height = int(declaration.get("height") or 0)
        except (TypeError, ValueError):
            return None
        return (width, height) if width > 0 and height > 0 else None

    @staticmethod
    def _declared_format(bindings, topic: str) -> str:
        if isinstance(bindings, dict):
            entry = bindings.get(topic)
            if isinstance(entry, str):
                return entry
            if isinstance(entry, dict):
                return str(entry.get("format") or "")
            entries = bindings.get("bindings") or bindings.get("inputs") or ()
        else:
            entries = bindings or ()
        for entry in entries:
            if isinstance(entry, dict) and entry.get("topic") == topic:
                return str(entry.get("format") or "")
        return ""

    @staticmethod
    def _role_of(topic: str) -> str:
        if topic.endswith(_OBJECTS_SUFFIX):
            return "objects"
        for suffix in _DEPTH_SUFFIXES:
            if topic.endswith(suffix):
                return "depth"
        return ""

    # ── observations ─────────────────────────────────────────────────────────

    def _on_objects(self, message) -> None:
        try:
            payload = json.loads(message.data)
        except Exception:                                     # noqa: BLE001
            return
        if not isinstance(payload, dict):
            return
        with self._obs_lock:
            self._objects = payload
            self._objects_ms = int(time.time() * 1000)

    def _on_depth(self, message) -> None:
        data = bytes(getattr(message, "data", b"") or b"")
        if not data:
            return
        try:
            depth = geometry.decode(data, *self._depth_frame)
        except geometry.DepthError as error:
            with self._lock:
                self._last_error = f"深度图解码失败：{error}"
            return
        with self._obs_lock:
            self._depth = depth
            self._depth_ms = int(time.time() * 1000)

    def _observation(self):
        now = int(time.time() * 1000)
        max_age = self._config.max_obs_age_ms
        with self._obs_lock:
            objects = (self._objects
                       if now - self._objects_ms <= max_age else None)
            depth = self._depth if now - self._depth_ms <= max_age else None
        return objects, depth

    # ── loop ─────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        publisher = self._publisher
        if publisher is None or not self._running:
            return
        try:
            payload = self._payload()
        except Exception as error:                            # noqa: BLE001
            with self._lock:
                self._last_error = f"{type(error).__name__}: {error}"
            log.warning("measure tick failed: %s", error)
            return

        from std_msgs.msg import String

        message = String()
        message.data = json.dumps(payload, ensure_ascii=False)
        publisher.publish(message)
        with self._lock:
            self._published += 1
            self._last = payload
        self._queue_view(payload)

    def _payload(self) -> dict:
        now_ms = int(time.time() * 1000)
        objects, depth = self._observation()
        if objects is None or depth is None:
            missing = []
            if objects is None:
                missing.append("VOP")
            if depth is None:
                missing.append("depth")
            payload = {
                "schema": SCHEMA,
                "ts_ms": now_ms,
                "fresh": False,
                "reason": "stale: " + "+".join(missing),
                "objects": [],
            }
            with self._lock:
                self._last = payload
            return payload

        measurements = geometry.measure_objects(
            objects.get("objects"),
            depth,
            self._camera,
            self._objects_frame,
            min_confidence=self._config.min_confidence,
            max_objects=self._config.max_objects,
            depth_percentile=self._config.depth_percentile,
            max_depth_m=self._config.max_depth_m,
        )
        smoothed = self._stabilizer.update(measurements, time.monotonic())
        self._session.record(smoothed, now_ms)
        objects_out = self._compact_objects(smoothed)
        payload = {
            "schema": SCHEMA,
            "ts_ms": now_ms,
            "fresh": True,
            "count": len(objects_out),
            "objects": objects_out,
        }
        degraded = self._degradations()
        if degraded:
            payload["degraded"] = degraded
        with self._lock:
            self._last = payload
        return payload

    @staticmethod
    def _compact_objects(records: list[dict]) -> list[dict]:
        out = []
        for record in records:
            # Stabilizer keeps a vanished track for track_timeout_ms. Those
            # records are predictions, not current observations; including them
            # made `count` reach 20+ on a frame with only a few detections.
            if record.get("observed") is False:
                continue
            bbox = record.get("bbox_norm")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            width = _cm(record.get("visible_width_m"))
            height = _cm(record.get("visible_height_m"))
            distance = _cm(record.get("distance_m"))
            if width is None and height is None and distance is None:
                continue
            name = str(record.get("name") or "")
            item = {
                "name": name,
                "box": [round(float(value), 3) for value in bbox],
                "conf": _round(record.get("confidence")),
            }
            if width is not None:
                item["visible_w_cm"] = width
            if height is not None:
                item["visible_h_cm"] = height
            if distance is not None:
                item["distance_cm"] = distance
            item["stable"] = bool(record.get("stable"))
            item["n"] = int(record.get("samples") or 0)
            if record.get("clipped"):
                item["clipped"] = True
            label_w = f"{width:.1f}" if width is not None else "?"
            label_h = f"{height:.1f}" if height is not None else "?"
            label_d = f"{distance / 100.0:.2f}" if distance is not None else "?"
            item["label"] = f"{name} {label_w}x{label_h}cm @{label_d}m"
            out.append(item)
        return out

    def _new_stabilizer(self) -> geometry.Stabilizer:
        return geometry.Stabilizer(
            window=self._config.stable_frames,
            min_samples=self._config.stable_min,
            timeout_s=self._config.track_timeout_ms / 1000.0,
            match_distance_norm=self._config.match_distance_norm,
        )

    def _new_session(self) -> geometry.Session:
        return geometry.Session(
            history_size=self._config.history_size,
            record_interval_ms=self._config.record_interval_ms,
            record_min_delta_m=self._config.record_min_delta_m,
            only_stable=self._config.only_stable,
            max_objects=self._config.summary_max_objects,
        )

    def _read(self, args: dict) -> dict:
        """The current frame's compact objects, never the session history."""
        target = str(args.get("target") or "").strip().lower()
        with self._lock:
            payload = self._last
            running = self._running
        objects = list((payload or {}).get("objects") or [])
        if target:
            objects = [obj for obj in objects
                       if target in str(obj.get("name") or "").lower()
                       or target in str(obj.get("label") or "").lower()]
        return {
            "state": "running" if running else "idle",
            "fresh": bool((payload or {}).get("fresh")),
            "ts_ms": (payload or {}).get("ts_ms"),
            "objects": objects,
        }

    def _history(self) -> dict:
        now_ms = int(time.time() * 1000)
        with self._lock:
            return {
                "state": "running" if self._running else "idle",
                "session": self._session.summary(now_ms),
                "records": self._session.history(),
            }

    def _reset_session(self) -> dict:
        with self._lock:
            self._session.reset(int(time.time() * 1000))
            self._last = None
        return {"state": "reset", "session": self._session.summary(int(time.time() * 1000))}

    # ── info/config ──────────────────────────────────────────────────────────

    def _info(self) -> dict:
        with self._lock:
            out = {
                "state": "running" if self._running else "idle",
                "topic_out": ([{"topic": self._topic, "format": TOPIC_FORMAT}]
                              + ([{"topic": self._view_topic, "format": "image/jpeg"}]
                                 if self._view_hz > 0 else [])),
                "rate_hz": self._config.rate_hz,
                "camera_topic": self._camera_topic,
                "view_hz": self._view_hz,
                "camera": _compact_camera(self._camera),
                "session": self._session.summary(int(time.time() * 1000)),
                "history_count": len(self._session.history()),
            }
            degraded = self._degradations()
            if degraded:
                out["degraded"] = degraded
            if self._last_error:
                out["error"] = self._last_error
            return out

    def _config_action(self, args: dict) -> dict:
        allowed = {
            "rate_hz": float,
            "max_objects": int,
            "min_confidence": float,
            "depth_percentile": float,
            "max_depth_m": float,
            "max_obs_age_ms": int,
            "stable_frames": int,
            "stable_min": int,
            "track_timeout_ms": int,
            "match_distance_norm": float,
            "history_size": int,
            "record_interval_ms": int,
            "record_min_delta_m": float,
            "only_stable": bool,
            "summary_max_objects": int,
        }
        changed = {}
        previous = dict(self._config.__dict__)
        previous_rate = self._config.rate_hz
        previous_window = (self._config.stable_frames,
                           self._config.stable_min,
                           self._config.track_timeout_ms,
                           self._config.match_distance_norm)
        for key, cast in allowed.items():
            if key not in args or args[key] is None:
                continue
            try:
                value = cast(args[key])
            except (TypeError, ValueError):
                self._config.__dict__.update(previous)
                return {"status": "error", "error": f"{key} 类型不正确"}
            setattr(self._config, key, value)
            changed[key] = value
        try:
            self._config.validate()
        except ValueError as error:
            self._config.__dict__.update(previous)
            return {"status": "error", "error": str(error)}
        if changed:
            self._cfg.update(changed)
            if previous_window != (self._config.stable_frames,
                                   self._config.stable_min,
                                   self._config.track_timeout_ms,
                                   self._config.match_distance_norm):
                self._stabilizer = self._new_stabilizer()
            if self._session_config_changed(changed):
                self._session = self._new_session()
                self._session.reset(int(time.time() * 1000))
            if self._running and previous_rate != self._config.rate_hz:
                self._restart_timer()
        return {"status": "configured", "config": self._config.as_dict(),
                "changed": changed}

    @staticmethod
    def _session_config_changed(changed: dict) -> bool:
        return bool({"history_size", "record_interval_ms", "record_min_delta_m",
                     "only_stable", "summary_max_objects"} & set(changed))

    def _restart_timer(self) -> None:
        node = self._node
        if node is None:
            return
        try:
            if self._timer is not None:
                node.destroy_timer(self._timer)
            self._timer = node.create_timer(1.0 / max(self._config.rate_hz, 0.1),
                                            self._tick)
        except Exception as error:                            # noqa: BLE001
            self._last_error = f"重启定时器失败：{error}"

    def _degradations(self) -> list[str]:
        if not self._running:
            return []
        return list(dict.fromkeys(self._camera_notes))

    def _error(self, message: str) -> dict:
        with self._lock:
            self._last_error = message
        log.warning("measure: %s", message)
        return {"state": "error", "message": message}
