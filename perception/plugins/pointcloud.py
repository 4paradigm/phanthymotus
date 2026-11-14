#!/usr/bin/env python3
"""Depth-only pointcloud card: one Z16/zlib input, display and optional metric output.

Camera parameters come from upstream camera_info. Display output retains its
approximate field-of-view fallback; metric output requires measured calibration.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import queue
import threading
import time
from typing import Optional

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField
from std_msgs.msg import String, UInt8MultiArray

from plugins.pointcloud_math import (
    MAX_POINTS,
    backproject_depth, decimation_stride,
    encode_packet, intrinsics_from_hfov,
    pinhole_of,
    summarize_cloud,
)
from utils.ros_lifecycle import dispose_node

log = logging.getLogger(__name__)

DEFAULT_CLOUD_TOPIC = "/perception/pointcloud"
DEFAULT_SUMMARY_TOPIC = "/perception/pointcloud_summary"
_DEFAULT_INSTANCE = "_default"

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


def output_topics_for(input_topic: Optional[str]) -> tuple[str, str]:
    """The one place the two output topics are derived from the input.

    跟随工具命名（{input}/pointcloud），summary 是兄弟而非子路径，
    同 visual_depth 的教训：两者互不为前缀，下游前缀匹配不会混淆。
    """
    if input_topic:
        return f"{input_topic}/pointcloud", f"{input_topic}/pointcloud_summary"
    return DEFAULT_CLOUD_TOPIC, DEFAULT_SUMMARY_TOPIC


TOOLS = [
    {
        "name": "pointcloud",
        "type": "processor",
        "multiInstance": True,
        "description": (
            "点云 — 一路深度图（image/depth-z16 或 image/depth-zlib）反投影为 3D 点云。"
            "输出点云包（sensor/pointcloud）与摘要（data/json）"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "config"],
                    "description": "Action to perform",
                },
                "input_topic": {
                    "type": "string",
                    "description": "要订阅的一路深度图话题",
                },
                "input_topics": {
                    "type": "array", "items": {"type": "string"},
                    "minItems": 1, "maxItems": 1,
                    "description": "一路深度图话题；与 input_topic 同时提供时必须一致",
                },
            },
            "required": ["action"],
            "x-input-metadata-consumer": True,
            "x-action-params": {
                "start": {
                    "params": ["input_topic", "input_topics"],
                    "description": "启动。通过 input_topic 或 input_topics 连接一路深度图",
                },
                "stop": {"params": [], "description": "停止点云输出"},
                "info": {"params": [], "description": "查看状态、模式与输出话题"},
                "config": {"params": [], "description": "更新实例配置（fps / max_points / 深度范围等），变更后需重启实例生效"},
            },
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "fps": {"type": "integer", "description": "Max output frames per second", "default": 2, "scope": "instance"},
                "max_points": {"type": "integer", "description": "每帧最多点数（上限 40000，由渲染器决定）", "default": 20000, "scope": "instance"},
                "min_depth_m": {"type": "number", "description": "最近有效距离（米）", "default": 0.1, "scope": "instance"},
                "max_depth_m": {"type": "number", "description": "最远有效距离（米）", "default": 10.0, "scope": "instance"},
                "metric_cloud": {"type": "boolean", "default": False, "scope": "instance",
                                 "description": "启用计算输出；要求实测深度、采集时间、物理坐标系和对应的实测内参"},
            },
        },
        "topic_in": [
            {"format": "image/depth-z16", "desc": "16-bit 深度图输入（毫米）"},
            {"format": "image/depth-zlib", "desc": "zlib 压缩深度图输入"},
        ],
        "topic_out": [
            {"format": "sensor/pointcloud", "desc": "XYZ 点云二进制包（point_step=12, float32 LE）"},
            {"format": "data/json", "desc": "点数/范围/三区最近障碍摘要"},
            {"format": "sensor/pointcloud2", "desc": "可选：带采集时间和物理坐标系的米制计算点云"},
        ],
    }
]


# review 第 6 轮：畸形 depth-zlib 帧以前每回调刷一条 warning，2 fps 的坏流
# 一天十几万行。第 1 条与之后每第 100 条采样记录，解出一帧好数据即复位
# —— 连续失败是状态，散落的失败是噪声。模块级计数：日志洪峰按进程算。
_zlib_fail_count = 0


def _decode_depth_message(msg) -> tuple[Optional[np.ndarray], str]:
    """z16 Image 或 depth-zlib CompressedImage → (深度图[米], 编码名)。

    只接受 16UC1/mono16 的 uint16 毫米深度帧；32FC1 等浮点深度帧（米制）
    点云卡片不支持，直接返回 None 拒绝。返回 (None, 编码名) 表示该帧不可用。
    zlib 帧解出来是 640x480 uint16 毫米 —— 与 visual_depth 的发布格式一致，
    解码端共用同一约定。
    """
    global _zlib_fail_count

    if isinstance(msg, Image):
        if msg.encoding not in ("16UC1", "mono16"):
            return None, msg.encoding
        needed = int(msg.height) * int(msg.step)
        if msg.width <= 0 or msg.height <= 0 or msg.step < msg.width * 2 or len(msg.data) < needed:
            return None, msg.encoding
        dtype = np.dtype((">" if msg.is_bigendian else "<") + "u2")
        depth = np.ndarray((msg.height, msg.width), dtype=dtype,
                           buffer=bytes(msg.data[:needed]), strides=(msg.step, 2))
        return depth.astype(np.float32) * 0.001, msg.encoding

    # CompressedImage must identify depth, not a JPEG/PNG image.
    encoding = str(getattr(msg, "format", "") or "").lower()
    if "depth" not in encoding or "zlib" not in encoding:
        return None, encoding
    data = bytes(msg.data)
    try:
        raw = zlib_decompress(data)
    except Exception as error:
        _zlib_fail_count += 1
        if _zlib_fail_count == 1 or _zlib_fail_count % 100 == 0:
            log.warning(f"[pointcloud] depth-zlib decompress failed"
                        f"（第 {_zlib_fail_count} 帧）: {str(error)[:200]}")
        return None, getattr(msg, "format", "depth-zlib")
    if len(raw) != 640 * 480 * 2:
        return None, "depth-zlib"
    _zlib_fail_count = 0
    depth = np.frombuffer(raw[:640 * 480 * 2], dtype="<u2").reshape(480, 640).astype(np.float32) * 0.001
    return depth, "depth-zlib"


def zlib_decompress(data: bytes) -> bytes:
    """有界解压：深度帧是 640x480 uint16 = 614400 B，加余量封顶 1 MiB。

    review 第 18 轮 Issue 2：zlib.decompress 无上限——几百 KB 的压缩包
    能声明几十 MB 的解压产物，在回调里把 perception 进程直接 OOM。
    decompressobj.max_length 挡住超限，unconsumed_tail 非空即"还有没解
    完的"= 超限，连同超出部分一起拒掉。"""
    import zlib
    cap = 1 << 20
    d = zlib.decompressobj()
    raw = d.decompress(data, cap)
    if d.unconsumed_tail or len(raw) > cap or not d.eof:
        raise ValueError("decompressed output exceeds the 1 MiB depth-frame cap")
    return raw


class _PointCloudNode(Node):
    """One depth subscription and bounded worker per instance."""

    def __init__(self, input_topic: str, calibration,
                 fps: float, max_points: int, min_depth_m: float,
                 max_depth_m: float,
                 node_suffix: str):
        super().__init__(f"pointcloud_{node_suffix}" if node_suffix else "pointcloud")
        self._mode = "depth"
        self._input_topic = input_topic or ''
        self._calibration = calibration or {}
        self._fps = fps
        self._frame_interval = 1.0 / max(fps, 0.1)
        self._max_points = max_points
        self._min_depth_m = min_depth_m
        self._max_depth_m = max_depth_m
        self._cloud_topic, self._summary_topic = output_topics_for(input_topic)
        self._cloud_pub = self.create_publisher(UInt8MultiArray, self._cloud_topic, _PUB_QOS)
        self._summary_pub = self.create_publisher(String, self._summary_topic, _PUB_QOS)
        self._metric_cloud_enabled = False
        self._input_format = None
        self._metric_cloud_topic = (input_topic or "/perception") + "/pointcloud_metric"
        self._metric_cloud_pub = None
        self._metric_metadata_topic = self._metric_cloud_topic + "/metadata"
        self._metric_metadata_source_topic = None
        self._metric_metadata_pub = None
        self._metric_metadata_lock = threading.Lock()
        self._metric_pending_clouds = {}
        self._metric_pending_metadata = {}
        self._metric_lock = threading.Lock()
        self._metric_published = 0
        self._metric_skipped = {}
        self._metric_last_skip = None

        self._subs: list = []
        self._input_discovery_timer = None
        self._input_discovery_errors = 0
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._worker_error_count = 0
        self._last_emit_time = 0.0
        self._frame_count = 0
        self._running = False
        self._lifecycle_lock = threading.RLock()

    # ── 生命周期（同 visual_depth：request_stop 不拿锁；stop 只收 worker）──

    def request_stop(self) -> None:
        self._stop_event.set()

    def _input_ros_type(self):
        """Select one ROS transport type, never two subscriptions on one topic.

        Camera declarations take priority. For older producers that supply no
        format, discover actual publisher endpoint types rather than guessing
        from the topic name or creating incompatible DDS subscriptions.
        """
        declared = {
            "image/depth-z16": "sensor_msgs/msg/Image",
            "image/depth-zlib": "sensor_msgs/msg/CompressedImage",
        }
        if self._input_format:
            if self._input_format not in declared:
                raise ValueError(f"unsupported pointcloud input format: {self._input_format}")
            return declared[self._input_format]
        endpoints = self.get_publishers_info_by_topic(self._input_topic)
        types = {entry.topic_type for entry in endpoints}
        supported = set(declared.values())
        if types - supported:
            raise ValueError(f"unsupported pointcloud publisher types on {self._input_topic}: {sorted(types)}")
        if len(types) > 1:
            raise ValueError(f"conflicting ROS publisher types on {self._input_topic}: {sorted(types)}")
        return next(iter(types)) if types else None

    def _attach_input_subscription(self):
        msg_type = self._input_ros_type()
        if msg_type is None:
            return False  # Publisher may appear after canvas startup.
        raw = msg_type == "sensor_msgs/msg/Image"
        self._input_format = "image/depth-z16" if raw else "image/depth-zlib"
        callback = self._depth_image_cb if raw else self._depth_comp_cb
        self._subs.append(self.create_subscription(
            Image if raw else CompressedImage, self._input_topic, callback, _LOW_LAT_QOS))
        log.info("[pointcloud] bound %s to %s", self._input_topic, msg_type)
        return True

    def _poll_input_subscription(self):
        # Legacy untyped publishers may come online after the card starts.
        with self._lifecycle_lock:
            if self._stop_event.is_set():
                return
            try:
                attached = self._attach_input_subscription()
            except Exception as error:
                self._input_discovery_errors += 1
                if self._input_discovery_errors == 1 or self._input_discovery_errors % 120 == 0:
                    log.error("[pointcloud] input binding failed: %s", error,
                              exc_info=self._input_discovery_errors == 1)
                return
            if attached:
                self._input_discovery_errors = 0
                timer, self._input_discovery_timer = self._input_discovery_timer, None
                if timer is not None:
                    self.destroy_timer(timer)

    def start(self) -> dict:
        with self._lifecycle_lock:
            if self._running:
                return self._state("running")
            self._stop_event.clear()
            if not self._attach_input_subscription():
                self._input_discovery_timer = self.create_timer(0.5, self._poll_input_subscription)
            if self._metric_metadata_source_topic:
                self._subs.append(self.create_subscription(
                    String, self._metric_metadata_source_topic, self._metric_metadata_cb, _PUB_QOS))
            if self._subs or self._input_discovery_timer is not None:
                self._worker = threading.Thread(target=self._worker_loop, daemon=True,
                                                name=f"pointcloud_worker_{self._input_topic}")
                self._worker.start()
            self._running = True
            log.info(f"[pointcloud] started mode={self._mode} "
                     f"in={self._input_topic or '(none)'}"
                     f"→ {self._cloud_topic}, {self._summary_topic}")
            return self._state("running")

    def stop(self) -> dict:
        # Worker only — subscriptions are destroyed by dispose_node after the
        # node leaves the executor (InvalidHandle race otherwise; see vop).
        self._stop_event.set()
        with self._lifecycle_lock:
            timer, self._input_discovery_timer = self._input_discovery_timer, None
            if timer is not None:
                self.destroy_timer(timer)
            if self._worker and self._worker.is_alive():
                self._worker.join(timeout=3.0)
                if self._worker.is_alive():
                    log.warning('[pointcloud] worker still stopping after timeout')
            self._worker = None
            self._running = False
            with self._metric_metadata_lock:
                self._metric_pending_clouds.clear()
                self._metric_pending_metadata.clear()
            return self._state("idle")

    def _state(self, state: str) -> dict:
        return {
            "state": state,
            "mode": self._mode,
            "input": self._input_topic or None,
            "cloud_topic": self._cloud_topic,
            "summary_topic": self._summary_topic,
        }

    # ── 订阅回调（只做节流和解码入队，重活都在 worker）─────────────────────

    def _throttled(self) -> bool:
        now = time.monotonic()
        if now - self._last_emit_time < self._frame_interval:
            return True
        self._last_emit_time = now
        return False

    @staticmethod
    def _drop_stale(q: queue.Queue, item) -> None:
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    def _depth_comp_cb(self, msg: CompressedImage):
        if self._throttled():
            return
        self._drop_stale(self._frame_queue, ("depth_zlib", msg))

    def _depth_image_cb(self, msg: Image):
        if self._throttled():
            return
        self._drop_stale(self._frame_queue, ("depth_z16", msg))


    # ── worker：按模式出云 ─────────────────────────────────────────────────

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                _, payload = self._frame_queue.get(timeout=1.0)
                if self._stop_event.is_set():
                    break
                self._emit_depth(payload)
            except queue.Empty:
                continue
            except Exception as error:  # noqa: BLE001 — keep the worker alive
                # review 第 6 轮：以前每帧一条完整 traceback —— 坏标定/坏帧
                # 把容器日志淹掉。第 1 条带 traceback，之后每第 100 条采样
                # 且只记有界摘要；一帧处理成功即复位。
                self._worker_error_count += 1
                if self._worker_error_count == 1:
                    log.error(f"[pointcloud] worker error: {error}", exc_info=True)
                elif self._worker_error_count % 100 == 0:
                    log.error(f"[pointcloud] worker error"
                              f"（第 {self._worker_error_count} 帧，连续失败）: "
                              f"{str(error)[:200]}")
            else:
                if self._worker_error_count:
                    self._worker_error_count = 0
                    log.info("[pointcloud] worker recovered; error stream cleared")

    # A: 深度图 → 反投影
    def _emit_depth(self, msg):
        depth_m, enc = _decode_depth_message(msg)
        if depth_m is None:
            if self._metric_cloud_enabled:
                self._metric_skip('invalid_depth')
            return
        h, w = depth_m.shape[:2]
        fx, fy, cx, cy = self._pinhole_for(w, h)
        stride = decimation_stride(w, h, self._max_points)
        xyz = backproject_depth(depth_m, fx, fy, cx, cy, stride,
                                 self._min_depth_m, self._max_depth_m)
        self._publish(xyz)
        if self._metric_cloud_enabled and ((enc in ("16UC1", "mono16") and self._input_format == "image/depth-z16") or
                (enc == "depth-zlib" and self._input_format == "image/depth-zlib")):
            self._publish_metric(xyz, msg, w, h)
        elif self._metric_cloud_enabled:
            self._metric_skip('unsupported_encoding')

    def _metric_skip(self, reason):
        with self._metric_lock:
            count = self._metric_skipped.get(reason, 0) + 1
            self._metric_skipped[reason] = count
            self._metric_last_skip = reason
        if count == 1 or count % 100 == 0:
            log.warning('[pointcloud] metric cloud skipped: %s (%d)', reason, count)

    def _metric_status(self):
        with self._metric_lock:
            return {'published': self._metric_published, 'skipped': dict(self._metric_skipped),
                    'last_skip_reason': self._metric_last_skip}

    def _publish_metric(self, xyz, source, width, height):
        if self._stop_event.is_set():
            return
        # An opt-in asserts that this source's stamp is capture time. A header
        # alone cannot distinguish capture time from publication time.
        calibration = self._calibration
        header = getattr(source, "header", None)
        stamp = getattr(header, "stamp", None)
        if not xyz.size:
            self._metric_skip('empty_cloud')
            return
        if (not isinstance(calibration, dict) or calibration.get('width') != width or
                calibration.get('height') != height or
                not all(k in calibration for k in ('fx', 'fy', 'cx', 'cy'))):
            self._metric_skip('calibration_mismatch')
            return
        if not getattr(header, 'frame_id', ''):
            self._metric_skip('missing_frame')
            return
        sec, nanosec = getattr(stamp, 'sec', None), getattr(stamp, 'nanosec', None)
        if type(sec) is not int or sec <= 0 or type(nanosec) is not int or not 0 <= nanosec < 1_000_000_000:
            self._metric_skip('invalid_capture_stamp')
            return
        try:
            intrinsics = [float(calibration[k]) for k in ("fx", "fy", "cx", "cy")]
            if not all(np.isfinite(intrinsics)) or min(intrinsics[:2]) <= 0:
                self._metric_skip('invalid_intrinsics')
                return
        except (TypeError, ValueError):
            self._metric_skip('invalid_intrinsics')
            return
        cloud = PointCloud2()
        cloud.header = header
        cloud.height, cloud.width = 1, len(xyz)
        cloud.fields = [PointField(name=name, offset=i * 4, datatype=PointField.FLOAT32, count=1)
                        for i, name in enumerate(("x", "y", "z"))]
        cloud.is_bigendian = False
        cloud.point_step, cloud.row_step = 12, 12 * len(xyz)
        cloud.is_dense = True
        cloud.data = np.ascontiguousarray(xyz, dtype="<f4").tobytes()
        self._metric_cloud_pub.publish(cloud)
        if self._metric_metadata_pub is not None:
            key = (sec * 1_000_000_000 + nanosec, header.frame_id,
                   hashlib.sha256(bytes(source.data)).hexdigest())
            self._pair_metric_metadata(key, cloud_digest=hashlib.sha256(cloud.data).hexdigest())
        with self._metric_lock:
            self._metric_published += 1

    def _metric_metadata_cb(self, msg):
        if self._stop_event.is_set():
            return
        try:
            packet = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        if not isinstance(packet, dict) or packet.get('schema') != 'motus.slam_frame/1':
            return
        stamp, frame, digest = (packet.get(k) for k in ('stamp_ns', 'frame_id', 'data_sha256'))
        if (type(stamp) is not int or stamp <= 0 or not isinstance(frame, str) or not frame
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            return
        self._pair_metric_metadata((stamp, frame, digest), packet=packet)

    def _pair_metric_metadata(self, key, *, packet=None, cloud_digest=None):
        # A reset can reuse stamp/frame. Bind evidence to the exact source
        # payload, then replace its digest with that of the projected cloud.
        with self._metric_metadata_lock:
            if self._stop_event.is_set():
                return
            if packet is not None:
                self._metric_pending_metadata.setdefault(key, packet)
            if cloud_digest is not None:
                self._metric_pending_clouds.setdefault(key, cloud_digest)
            if key in self._metric_pending_metadata and key in self._metric_pending_clouds:
                output = dict(self._metric_pending_metadata.pop(key))
                output['data_sha256'] = self._metric_pending_clouds.pop(key)
                msg = String()
                msg.data = json.dumps(output)
                self._metric_metadata_pub.publish(msg)
            for pending in (self._metric_pending_metadata, self._metric_pending_clouds):
                while len(pending) > 10:
                    pending.pop(next(iter(pending)))

    def _pinhole_for(self, w: int, h: int) -> tuple:
        """(fx, fy, cx, cy)：驱动实测参数 → 上游 camera_info 的 K（跨分辨率
        按 camera_info._rescale_K 缩放）→ hfov 90° 兜底。

        review 指出：上游明明声明了真实内参，反投影却按 90° hfov 硬算，
        横向坐标系统性偏差。K 是像素单位、分辨率变了必须缩放。
        """
        if self._calibration:
            return pinhole_of(self._calibration, w, h)
        upstream = getattr(self, "_upstream_intrinsics", None)
        if upstream:
            K, src_w, src_h = upstream
            from plugins.camera_info import _rescale_K

            scaled = _rescale_K(K, (src_w, src_h), (w, h)) if (w, h) != (src_w, src_h) else list(K)
            if scaled:
                return (float(scaled[0]), float(scaled[4]),
                        float(scaled[2]), float(scaled[5]))
        fov = getattr(self, '_upstream_fov', None)
        if fov:
            fx = (w / 2.) / np.tan(fov[0])
            fy = (h / 2.) / np.tan(fov[1]) if fov[1] is not None else fx
            return float(fx), float(fy), w / 2., h / 2.
        if not getattr(self, '_warned_fov_fallback', False):
            log.warning('[pointcloud] no usable intrinsics or field of view; display uses approximate 90-degree fallback')
            self._warned_fov_fallback = True
        return intrinsics_from_hfov(w, h, 90.0)

    def _publish(self, xyz: np.ndarray):
        if self._stop_event.is_set() or xyz.shape[0] == 0:
            return
        self._frame_count += 1
        msg = UInt8MultiArray()
        msg.data = encode_packet(xyz)
        self._cloud_pub.publish(msg)

        summary = summarize_cloud(xyz, self._min_depth_m, self._max_depth_m)
        summary["timestamp"] = time.time()
        summary["mode"] = self._mode
        smsg = String()
        smsg.data = json.dumps(summary, ensure_ascii=False)
        self._summary_pub.publish(smsg)


class PointCloudPerceptionPlugin:
    PREFIX = "pointcloud"
    ALIASES = ()

    def __init__(self, plugin_cfg: dict, namespace: str, executor):
        self._namespace = namespace
        self._executor = executor
        self._plugin_cfg = dict(plugin_cfg or {})
        self._fps = int(plugin_cfg.get("fps", 2))
        self._max_points = max(100, min(MAX_POINTS, int(plugin_cfg.get("max_points", 20000))))
        self._min_depth_m = float(plugin_cfg.get("min_depth_m", 0.1))
        self._max_depth_m = float(plugin_cfg.get("max_depth_m", 10.0))
        self._nodes: dict[str, _PointCloudNode] = {}
        self._instance_configs: dict[str, dict] = {}
        self._nodes_lock = threading.RLock()
        self._upstream_camera: dict[str, dict] = {}
        self._upstream_input_metadata: dict[str, dict] = {}

    def _resolve_input(self, args: dict) -> str:
        # Keep saved single-input auto/depth calls, but never infer depth from RGB.
        if args.get("mode") not in (None, "", "auto", "depth"):
            raise ValueError("点云仅支持一路深度输入，mode 仅兼容 depth/auto")
        topics = args.get("input_topics")
        if topics is not None and (not isinstance(topics, list) or len(topics) != 1 or
                not isinstance(topics[0], str) or not topics[0].strip()):
            raise ValueError("input_topics 必须只包含一路深度图话题")
        topic = args.get("input_topic")
        if topic is not None and (not isinstance(topic, str) or not topic.strip()):
            raise ValueError("input_topic 必须是非空深度图话题")
        if topics and topic and topic != topics[0]:
            raise ValueError("input_topic 与 input_topics 必须是同一路深度图话题")
        topic = topic or (topics[0] if topics else None)
        if not topic:
            raise ValueError("需要一路深度输入：input_topic 或 input_topics")
        from plugins.camera_info import for_topic
        declarations = [for_topic(args.get("camera_info"), topic)]
        metadata = args.get("input_metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get(topic), dict):
            declarations.append(metadata[topic])
        for declaration in declarations:
            if declaration.get("format") not in (None, "image/depth-z16", "image/depth-zlib"):
                raise ValueError("点云仅支持 image/depth-z16 或 image/depth-zlib 深度输入")
        return topic

    def _camera_info(self, instance_id, input_topic, nodes,
                     cloud_topic: str, summary_topic: str) -> tuple:
        """`(declarations, note)`：本卡两个输出端口的 motus.camera/1 声明。

        同 visual_depth：点云不裁剪不缩放（抽稀只丢点不丢视场），width/
        height 原样继承，K 与 half_fov_rad 都不变。note 区分"上游没声明"
        与"本卡弄丢了"——只有前者该找相机卡片的人。
        """
        from plugins.camera_info import inherit

        key = instance_id if instance_id in (nodes or {}) else None
        if key is None:
            key = next(iter(nodes), None) if nodes else (input_topic or _DEFAULT_INSTANCE)
        with self._nodes_lock:
            upstream = self._upstream_camera.get(key) or {}
        if not upstream:
            return [], ("上游相机没有声明 camera_info —— 下游拿不到视场角，"
                        "横向坐标只能按 90° 兜底反投影。相机卡片补上声明即可，见 "
                        "phanthymotus-driver/README_dev.md 的 Camera Parameters")
        # 点云端口：视场与上游一致，只是格式变了；摘要描述的是同一片点云。
        out = inherit(upstream, topic=cloud_topic, fmt="sensor/pointcloud",
                      stage="perception/pointcloud")
        out += inherit(upstream, topic=summary_topic, fmt="data/json",
                       stage="perception/pointcloud")
        return out, ""

    def _start_node(self, node_key: str, input_topic: str, calibration):
        with self._nodes_lock:
            if node_key in self._nodes:
                return
            icfg = self._instance_configs.get(node_key, {})
            effective = self._effective_config(icfg)
            node = _PointCloudNode(
                input_topic, calibration,
                fps=effective['fps'], max_points=effective['max_points'],
                min_depth_m=effective['min_depth_m'], max_depth_m=effective['max_depth_m'],
                node_suffix=node_key.replace("/", "_").replace("-", "_").lstrip("_"),
            )
            node._applied_config = effective
            node._metric_cloud_enabled = effective['metric_cloud']
            if node._metric_cloud_enabled:
                node._metric_cloud_pub = node.create_publisher(PointCloud2, node._metric_cloud_topic, _PUB_QOS)
            with self._nodes_lock:
                upstream = self._upstream_camera.get(node_key) or {}
            declaration = self._upstream_input_metadata.get(node_key) or {}
            node._input_format = declaration.get("format") or upstream.get("format")
            metadata_topic = declaration.get("frame_metadata_topic")
            if node._metric_cloud_enabled and isinstance(metadata_topic, str) and metadata_topic:
                node._metric_metadata_source_topic = metadata_topic
                node._metric_metadata_pub = node.create_publisher(
                    String, node._metric_metadata_topic, _PUB_QOS)
            K = upstream.get("K")
            if isinstance(K, (list, tuple)) and len(K) == 9:
                try:
                    matrix = np.asarray(K, dtype=float).reshape(3, 3)
                    dimensions = (upstream.get('width'), upstream.get('height'))
                    valid = (np.isfinite(matrix).all() and matrix[0, 0] > 0 and matrix[1, 1] > 0
                             and abs(matrix[0, 1]) < 1e-9 and abs(matrix[1, 0]) < 1e-9
                             and np.allclose(matrix[2], [0, 0, 1])
                             and all(type(v) is int and v > 0 for v in dimensions))
                    if valid:
                        node._upstream_intrinsics = (matrix.reshape(-1).tolist(), *dimensions)
                except (TypeError, ValueError):
                    pass
            half_h, half_v = upstream.get('half_fov_rad'), upstream.get('half_fov_v_rad')
            if (isinstance(half_h, (float, int)) and not isinstance(half_h, bool)
                    and np.isfinite(half_h) and 0 < half_h < np.pi / 2
                    and (half_v is None or (isinstance(half_v, (float, int))
                         and not isinstance(half_v, bool) and np.isfinite(half_v)
                         and 0 < half_v < np.pi / 2))):
                node._upstream_fov = (half_h, half_v)
            node._calibration_source = (
                'declared-K' if getattr(node, '_upstream_intrinsics', None) else
                'declared-fov-approximate' if getattr(node, '_upstream_fov', None) else
                '90-degree-fallback')
            self._executor.add_node(node)
            try:
                node.start()
            except Exception:
                dispose_node(self._executor, node, label=f'pointcloud/{node_key}')
                raise
            self._nodes[node_key] = node
        log.info("[pointcloud] depth node started: %s", input_topic)

    def _retire_node(self, node_key: str) -> Optional[dict]:
        with self._nodes_lock:
            node = self._nodes.pop(node_key, None)
        if node is None:
            return None
        with self._nodes_lock:
            # 声明随节点走：留着会让重新接线的卡片拿一个已经不喂它的相机
            # 的光学参数回答 info()（同 visual_depth 的注释）。
            self._upstream_camera.pop(node_key, None)
            self._upstream_input_metadata.pop(node_key, None)
        node.request_stop()
        result = node.stop()
        dispose_node(self._executor, node, label=f"pointcloud/{node_key}")
        return result

    def shutdown(self) -> None:
        """Retire workers and subscriptions before ROS teardown."""
        with self._nodes_lock:
            for key in list(self._nodes):
                self._retire_node(key)

    def get_tools(self) -> list:
        return TOOLS

    def _effective_config(self, cfg: dict) -> dict:
        return {
            'fps': int(cfg.get('fps', self._fps)),
            'max_points': max(100, min(MAX_POINTS, int(cfg.get('max_points', self._max_points)))),
            'min_depth_m': float(cfg.get('min_depth_m', self._min_depth_m)),
            'max_depth_m': float(cfg.get('max_depth_m', self._max_depth_m)),
            'metric_cloud': cfg.get('metric_cloud', self._plugin_cfg.get('metric_cloud', False)) is True,
        }

    def _validate_ranges(self, cfg: dict) -> Optional[str]:
        """fps / min_depth_m / max_depth_m 的联合校验（review 第 17 轮
        Issue 3）。单值合法、组合非法（min >= max）以前照收——反投影
        的 z 截断窗口为空，卡片 running 却一帧云不出；fps<=0 则被静默钳到
        0.1 fps。校验组合时未给出的一侧取现值，允许分两步把范围从 (0.1, 10)
        改到 (5, 20) 这类操作。"""
        fps = cfg.get("fps")
        min_d = cfg.get("min_depth_m")
        max_d = cfg.get("max_depth_m")
        try:
            if fps is not None:
                fps = float(fps)
                if not np.isfinite(fps) or fps <= 0:
                    raise ValueError(
                        f"fps 无效（fps={cfg.get('fps')!r}）：需要正的有限值，"
                        "否则帧间隔被静默钳到 0.1 fps")
            if min_d is not None:
                min_d = float(min_d)
            if max_d is not None:
                max_d = float(max_d)
        except (TypeError, ValueError) as error:
            return f"配置数值异常：{error}"
        lo = min_d if min_d is not None else self._min_depth_m
        hi = max_d if max_d is not None else self._max_depth_m
        if not (np.isfinite(lo) and np.isfinite(hi)) or not (0.0 <= lo < hi <= 1000.0):
            return (f"深度范围无效（min_depth_m={cfg.get('min_depth_m')!r}, "
                    f"max_depth_m={cfg.get('max_depth_m')!r}）：需要 0 <= min < max "
                    "<= 1000（米），否则 z 截断窗口为空、云一帧不出")
        return None

    def dispatch(self, name: str, args: dict) -> Optional[dict]:
        action = args.get("action", name)
        instance_id = args.get("instance_id", "")

        if action == "info":
            with self._nodes_lock:
                nodes = dict(self._nodes)
            instances = {
                key: {
                    "mode": node._mode,
                    "input": node._input_topic or None,
                    "cloud_topic": node._cloud_topic,
                    "summary_topic": node._summary_topic,
                    "frame_count": node._frame_count,
                    # 标定来源：声明层（驱动型号级出厂标定）还是用户配置 ——
                    # 出云错了排查方向完全不同。
                    "calibration_source": getattr(node, "_calibration_source", None),
                    "metric_cloud": node._metric_status() if node._metric_cloud_enabled else None,
                }
                for key, node in nodes.items()
            }
            topics_in = []
            selected = [nodes[instance_id]] if instance_id in nodes else nodes.values()
            for node in selected:
                if all(t["topic"] != node._input_topic for t in topics_in):
                    topics_in.append({"topic": node._input_topic,
                                      "format": node._input_format or "image/depth-z16"})
            # 每实例一对输出话题；多实例时不再用 output_topics_for(None)
            # 的默认 /perception/... 话题把下游引向不存在的话题。
            if instance_id and instance_id in nodes:
                node = nodes[instance_id]
                topics_out = [
                    {"topic": node._cloud_topic, "format": "sensor/pointcloud"},
                    {"topic": node._summary_topic, "format": "data/json"},
                ]
                if node._metric_cloud_enabled:
                    topics_out.append({"topic": node._metric_cloud_topic, "format": "sensor/pointcloud2"})
            else:
                seen: set[str] = set()
                topics_out: list[dict[str, str]] = []
                for node in nodes.values():
                    outputs = [(node._cloud_topic, "sensor/pointcloud"),
                               (node._summary_topic, "data/json")]
                    if node._metric_cloud_enabled:
                        outputs.append((node._metric_cloud_topic, "sensor/pointcloud2"))
                    for topic, fmt in outputs:
                        if topic and topic not in seen:
                            seen.add(topic)
                            topics_out.append({"topic": topic, "format": fmt})
            topics_out = topics_out if topics_in or nodes else []
            # 供 _camera_info 的光学参数继承：有实例就用该实例的话题，
            # 没有就用第一个节点的话题，绝不回退到默认话题。
            if instance_id and instance_id in nodes:
                sample = nodes[instance_id]
            else:
                sample = next(iter(nodes.values())) if nodes else None
            cloud_topic = sample._cloud_topic if sample else DEFAULT_CLOUD_TOPIC
            summary_topic = sample._summary_topic if sample else DEFAULT_SUMMARY_TOPIC
            info = {
                "name": "PointCloudPerception", "manufacture": "Embodied",
                "model": "depth",
                "state": "running" if instances else "idle",
                "instances": instances,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "3D point cloud from one depth map",
            }
            # 光学参数往下传（同 visual_depth 的契约）：声明了才带
            # camera_info，没声明就带 note 说明是谁的问题。
            camera_out, camera_note = self._camera_info(
                instance_id,
                args.get("input_topic") or (args.get("input_topics") or [""])[0] or "",
                nodes, cloud_topic, summary_topic)
            from plugins.camera_info import inherit
            metadata_out = []
            for key, metric_node in nodes.items():
                if not metric_node._metric_cloud_enabled or (instance_id in nodes and key != instance_id):
                    continue
                with self._nodes_lock:
                    upstream = self._upstream_camera.get(key)
                    declaration = copy.deepcopy(self._upstream_input_metadata.get(key))
                if declaration:
                    declaration.update(topic=metric_node._metric_cloud_topic, kind="cloud",
                                       format="sensor/pointcloud2", length_unit="m",
                                       frame_metadata_topic=metric_node._metric_metadata_topic)
                    metadata_out.append(declaration)
                c = metric_node._calibration
                declaration = inherit(upstream, topic=metric_node._metric_cloud_topic,
                    fmt="sensor/pointcloud2", stage="perception/pointcloud",
                    width=c['width'], height=c['height'])
                if declaration:
                    declaration[0]['K'] = [float(c['fx']), 0., float(c['cx']), 0.,
                                           float(c['fy']), float(c['cy']), 0., 0., 1.]
                    camera_out.extend(declaration)
            if metadata_out:
                info["input_metadata"] = metadata_out
            if camera_out:
                info["camera_info"] = camera_out
            if camera_note:
                info["camera_info_note"] = camera_note
            return info

        elif action == "start":
            try:
                input_topic = self._resolve_input(args)
                calibration = None
            except ValueError as error:
                return {"state": "error", "message": str(error)}
            node_key = instance_id or input_topic or _DEFAULT_INSTANCE
            icfg = self._instance_configs.get(node_key, {})
            metric_enabled = icfg.get("metric_cloud", self._plugin_cfg.get("metric_cloud", False)) is True
            if metric_enabled:
                # 沿用现有深度类型：毫米制 Z16 或固定 640x480 的 zlib。
                # 仅驱动声明的实测针孔内参可开启计算输出，估计深度仍只供显示。
                from plugins.camera_info import for_topic as _camera_for_topic
                declared = _camera_for_topic(args.get("camera_info"), input_topic)
                K = declared.get("K")
                vendor = declared.get("vendor") or {}
                fmt = declared.get("format")
                try:
                    matrix = np.asarray(K, dtype=float).reshape(3, 3)
                    distortion = np.asarray(declared.get('D') or [], dtype=float)
                    supported_K = (np.isfinite(matrix).all() and
                                   np.allclose(matrix[2], [0., 0., 1.]) and
                                   abs(matrix[0, 1]) < 1e-9 and abs(matrix[1, 0]) < 1e-9 and
                                   np.isfinite(distortion).all() and np.all(distortion == 0))
                except (TypeError, ValueError):
                    supported_K = False
                if not (declared.get("schema") == "motus.camera/1"
                        and fmt in ("image/depth-z16", "image/depth-zlib")
                        and declared.get("source") == "derived-from-K"
                        and declared.get("distortion_model") == "pinhole"
                        and isinstance(vendor, dict)
                        and vendor.get("depth_unit") == "mm"
                        and vendor.get("depth_source") == "measured"
                        and vendor.get("timestamp_semantics") == "source_capture_mapped"
                        and isinstance(K, (list, tuple)) and len(K) == 9 and supported_K
                        and (fmt != "image/depth-zlib" or
                             (declared.get("width") == 640 and declared.get("height") == 480))):
                    return {"state": "error", "message": "metric_cloud requires measured millimetre depth, driver pinhole intrinsics and capture timestamps"}
                calibration = dict(fx=K[0], fy=K[4], cx=K[2], cy=K[5],
                                   width=declared.get("width"), height=declared.get("height"))
            if metric_enabled:
                if not isinstance(calibration, dict):
                    return {"state": "error", "message": "metric_cloud requires depth mode and measured intrinsics"}
                try:
                    values = [float(calibration[k]) for k in ("fx", "fy", "cx", "cy")]
                    dimensions = [calibration[k] for k in ("width", "height")]
                    if (not all(np.isfinite(values)) or min(values[:2]) <= 0 or
                            any(type(v) is not int or v <= 0 for v in dimensions)):
                        raise ValueError
                except (KeyError, TypeError, ValueError):
                    return {"state": "error", "message": "metric_cloud requires finite measured fx/fy/cx/cy and image width/height"}
            declarations = args.get("input_metadata")
            if 'input_metadata' in args and (not isinstance(declarations, dict) or
                    (input_topic in declarations and
                     (not isinstance(declarations[input_topic], dict) or not declarations[input_topic]))):
                return {'state': 'error', 'message': 'input_metadata must map input topics to nonempty declaration objects'}
            declaration = declarations.get(input_topic) if isinstance(declarations, dict) else None
            declaration = copy.deepcopy(declaration) if isinstance(declaration, dict) else {}
            if metric_enabled and declaration.get('format') not in (None, fmt):
                return {'state': 'error', 'message': 'input_metadata format conflicts with driver camera_info'}
            with self._nodes_lock:
                existing = self._nodes.get(node_key)
                if existing and (existing._input_topic != (input_topic or '') or
                                 declaration != self._upstream_input_metadata.get(node_key, {})):
                    return {'state': 'error', 'message': 'stop the pointcloud instance before changing its inputs, mode or metadata'}
                self._upstream_input_metadata[node_key] = declaration
            if input_topic:
                from plugins.camera_info import for_topic as _camera_for_topic

                declarations = args.get("camera_info")
                with self._nodes_lock:
                    self._upstream_camera[node_key] = _camera_for_topic(
                        declarations, input_topic)
            with self._nodes_lock:
                running = self._nodes.get(node_key)
            if running is None:
                try:
                    self._start_node(node_key, input_topic, calibration)
                except Exception as error:
                    log.error("[pointcloud] unable to bind ROS input topic %s: %s",
                              input_topic, error, exc_info=True)
                    return {'state': 'error', 'message': str(error)}
                with self._nodes_lock:
                    running = self._nodes.get(node_key)
                if running is None:
                    return {"state": "idle", "input": input_topic}
            return running.start()

        elif action == "stop":
            with self._nodes_lock:
                if instance_id:
                    result = self._retire_node(instance_id)
                    return result if result is not None else {"state": "idle"}
                results = [key for key in list(self._nodes) if self._retire_node(key) is not None]
            return {"state": "idle", "stopped_instances": results} if results else {"state": "idle"}

        elif action == "config":
            # 仅接收当前卡片支持的配置；旧画布保存的手动标定键直接忽略。
            cfg = {k: v for k, v in args.items()
                   if k in TOOLS[0]['configSchema']['properties'] and v is not None and v != ""}
            with self._nodes_lock:
                previous = self._instance_configs.get(instance_id, {}) if instance_id else {}
                merged = {**previous, **cfg}
                error = self._validate_ranges(merged)
                try:
                    effective = self._effective_config(merged)
                    error = error or self._validate_ranges(effective)
                except (TypeError, ValueError, OverflowError) as exc:
                    error = f"配置数值异常：{exc}"
                if error is not None:
                    return {"state": "error", "message": error}
                if instance_id:
                    self._instance_configs[instance_id] = merged
                    running = self._nodes.get(instance_id)
                    # 首次显式保存默认值、重复保存或等效的限幅值均保留运行实例。
                    changed = running is not None and effective != running._applied_config
            if instance_id:
                if changed:
                    self._retire_node(instance_id)
                return {"status": "configured", "instance_id": instance_id, "config": merged}
            if "fps" in cfg:
                self._fps = int(cfg["fps"])
            if "max_points" in cfg:
                self._max_points = max(100, min(MAX_POINTS, int(cfg["max_points"])))
            if "min_depth_m" in cfg:
                self._min_depth_m = float(cfg["min_depth_m"])
            if "max_depth_m" in cfg:
                self._max_depth_m = float(cfg["max_depth_m"])
            if "metric_cloud" in cfg:
                self._plugin_cfg["metric_cloud"] = cfg["metric_cloud"]
            return {"status": "configured", "config": cfg}

        return None
