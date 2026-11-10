#!/usr/bin/env python3
"""
plugins/pointcloud.py — PointCloudPerceptionPlugin: 3D 点云，三种输入任选其一。

一个工具，按输入形态走三条路：

    A. 深度图        image/depth-z16 (Image) / image/depth-zlib (CompressedImage)
                     → 针孔模型直接反投影。
    B. 单目 RGB      image/jpeg (CompressedImage)
                     → 复用 visual_depth 的 YOLO26-depth TensorRT 引擎估计深度，
                       再走 A 的反投影。
    C. 双目左右两路  image/jpeg × 2（left=第一路, right=第二路）
                     → SGBM 视差 + Q 矩阵 reprojectImageTo3D
                       （同 phanthymotus-driver Go1 的 pointcloud_stream.cc）。

输出两路（跟随工具命名且互不为前后缀 —— 见 output_topics_for）：

    {input}/pointcloud          sensor/pointcloud  UInt8MultiArray 二进制包
    {input}/pointcloud_summary  data/json           点数/范围/最近障碍摘要

相机参数仅来自驱动的 camera_info 声明，不接受卡片手动标定。
A/B 显示可用 K、视场角或 90° 兜底；计算输出必须有实测深度和对应内参。
C 由两条输入的 stereo_peer + K/D 自动拼接双目参数，缺失时拒绝启动。

纯数学在 plugins/pointcloud_math.py（不依赖 cv2 / rclpy，本地可测）。
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
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
    backproject_depth, calibration_tier, decimate_cloud, decimation_stride,
    encode_packet, filter_cloud, intrinsics_from_hfov,
    pinhole_of, Q_from_baseline,
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

# 双目两帧允许的最大时间差（秒）——超过就丢掉旧帧
_STEREO_SYNC_S = 0.05


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
            "点云 — 把相机的画面变成 3D 点云并持续输出。三种输入任选其一："
            "深度图（image/depth-z16 或 image/depth-zlib）直接反投影；"
            "单目 RGB（image/jpeg）用单目深度模型估计后反投影；"
            "双目左右两路（image/jpeg × 2）做立体匹配。"
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
                    "description": "单目/深度图模式：要订阅的话题。双目模式可留空，由 input_topics 提供左右两路",
                },
                "input_topics": {
                    "type": "array", "items": {"type": "string"},
                    "description": "双目模式的左右话题列表：第一路=左目，第二路=右目。单输入时忽略此参数",
                },
                "input_metadata": {
                    "type": "object",
                    "description": "Core-provided capture provenance for connected input topics",
                },
                "mode": {
                    "type": "string",
                    "enum": ["auto", "depth", "mono", "stereo"],
                    "description": "输入模式。auto（默认）：2 路输入 → 双目 stereo；1 路输入按声明或 ROS 发布者类型选择订阅，首帧确定 depth（z16/zlib 深度图）或 mono（jpeg）",
                },
            },
            "required": ["action"],
            "x-input-metadata-consumer": True,
            "x-action-params": {
                "start": {
                    "params": ["input_topic", "input_topics", "mode"],
                    "description": "启动。给 input_topic（单输入）或 input_topics（双目左右两路）。mode 留空则按输入与消息类型自动判定",
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
            {"format": "image/jpeg", "desc": "mono / stereo（左右两路）输入"},
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
    import cv2

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

    # CompressedImage: image/depth-zlib
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


def _decode_jpeg(data: bytes) -> Optional[np.ndarray]:
    """JPEG 字节 → BGR 帧。解码失败返回 None。"""
    import cv2
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return None if frame is None else frame


class _PointCloudNode(Node):
    """One node per instance: 1 路（depth/mono）或 2 路（stereo）订阅。"""

    def __init__(self, mode: str, input_topic: Optional[str],
                 input_topic_right: Optional[str], calibration,
                 fps: float, max_points: int, min_depth_m: float,
                 max_depth_m: float,
                 node_suffix: str):
        super().__init__(f"pointcloud_{node_suffix}" if node_suffix else "pointcloud")
        self._mode = mode
        self._input_topic = input_topic or ''
        self._input_topic_right = input_topic_right or ''
        self._calibration = calibration or {}
        self._fps = fps
        self._frame_interval = 1.0 / max(fps, 0.1)
        self._max_points = max_points
        self._min_depth_m = min_depth_m
        self._max_depth_m = max_depth_m
        self._cloud_topic, self._summary_topic = output_topics_for(input_topic)
        # raw 档 rectify 映射缓存（_rectify_pair）：blob 固定，映射只随
        # 分辨率变 —— 每帧重算 stereoRectify/initUndistortRectifyMap 是
        # review 指出的白烧 CPU。worker 线程独占，无需加锁。
        self._rectify_cache: dict = {}

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
        self._stereo_queue: queue.Queue = queue.Queue(maxsize=1)
        # 双目左右帧按拍摄时刻（header.stamp，缺省退回到达时刻）配对：
        # {stamp: jpeg 字节}（编码态，配对成功才解码），旧于 sync 窗口的直接丢
        self._left_latest: Optional[tuple[float, bytes]] = None
        self._left_lock = threading.Lock()
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
            "image/jpeg": "sensor_msgs/msg/CompressedImage",
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
        if self._mode == "depth":
            callback = self._depth_image_cb if raw else self._depth_comp_cb
        else:
            callback = self._auto_image_cb if raw else self._auto_comp_cb
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
            if self._mode == "stereo":
                if self._input_topic and self._input_topic_right:
                    self._subs.append(self.create_subscription(
                        CompressedImage, self._input_topic, self._left_cb, _LOW_LAT_QOS))
                    self._subs.append(self.create_subscription(
                        CompressedImage, self._input_topic_right, self._right_cb, _LOW_LAT_QOS))
            elif self._input_topic:
                if self._mode in ("depth", "auto"):
                    if not self._attach_input_subscription():
                        # Don't install a guessed fallback subscription; bind
                        # when a publisher advertises its real ROS type.
                        self._input_discovery_timer = self.create_timer(
                            0.5, self._poll_input_subscription)
                else:  # mono
                    self._subs.append(self.create_subscription(
                        CompressedImage, self._input_topic, self._jpeg_cb, _LOW_LAT_QOS))
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
                     f"{'+' + self._input_topic_right if self._input_topic_right else ''} "
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
                    log.warning('[pointcloud] worker still stopping after timeout; model load may be non-cancellable')
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
            "input_right": self._input_topic_right or None,
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

    def _jpeg_cb(self, msg: CompressedImage):
        if self._throttled():
            return
        self._drop_stale(self._frame_queue, ("jpeg", bytes(msg.data)))

    def _depth_comp_cb(self, msg: CompressedImage):
        if self._throttled():
            return
        self._drop_stale(self._frame_queue, ("depth_zlib", msg))

    def _depth_image_cb(self, msg: Image):
        if self._throttled():
            return
        self._drop_stale(self._frame_queue, ("depth_z16", msg))

    def _auto_image_cb(self, msg: Image):
        # Image 消息只会是深度（z16）；RGB 走 CompressedImage。这里不改
        # _mode：mode 的定形只发生在 worker 里（按队列里的 kind），否则
        # worker 的懒加载分支永远轮不到（review issue：冷启动 jpeg 帧
        # 全部静默丢）。
        if self._throttled():
            return
        self._drop_stale(self._frame_queue, ("depth_z16", msg))

    def _auto_comp_cb(self, msg: CompressedImage):
        if self._throttled():
            return
        # review 第 9 轮：小写归一化后再匹配。栈内 visual_depth 发的格式串
        # 是 "16UC1; compressedDepth zlib"（visual_depth.py），不含小写
        # "depth" —— 旧实现把它判成 JPEG，触发 TRT 引擎下载，再在 JPEG
        # 解码上失败，一帧云都不出。
        fmt = str(getattr(msg, "format", "") or "").lower()
        if "depth" in fmt:
            self._drop_stale(self._frame_queue, ("depth_zlib", msg))
        else:
            self._drop_stale(self._frame_queue, ("jpeg", bytes(msg.data)))

    def _left_cb(self, msg: CompressedImage):
        # review 第 16 轮：配对必须用拍摄时刻（header.stamp）而非回调到达
        # 时刻。调度/传输抖动下"右帧晚到"会把旧右帧配新左帧——SGBM 出错
        # 视差、发布几何无效的云；同时刻拍的两帧也会因到达差超窗被误丢。
        # stamp 缺失（老驱动不写 header）才退回到达时间。stereo 单位是
        # 两条消息 stamp 之差，同一时钟域即可（ROS stamp 是 epoch 秒）。
        # review 第 19 轮建议：这里存编码字节而不是解码帧——左路 30 Hz
        # 全量解码只为被下一次到达覆盖，出云 2 fps 时真正配上的才 2 对，
        # Jetson 上每秒白白解码 ~28 帧。解码挪到 _right_cb 配对成功之后。
        with self._left_lock:
            self._left_latest = (
                self._frame_stamp(msg, time.monotonic()), bytes(msg.data))

    def _right_cb(self, msg: CompressedImage):
        # review 第 16 轮建议：节流在解码前——出云 2 fps 时右路 30 Hz 全量
        # JPEG 解码纯属浪费（Jetson CPU 上是双路 60 解码/s 换 2 次配对）。
        # 左路存编码字节、配对成功才解码（见 _left_cb），两边都只对配上的
        # 对付解码钱。
        if self._throttled():
            return
        stamp = self._frame_stamp(msg, time.monotonic())
        with self._left_lock:
            left = self._left_latest
            self._left_latest = None
        if left is None or abs(stamp - left[0]) > _STEREO_SYNC_S:
            return
        left_frame = _decode_jpeg(left[1])
        right_frame = _decode_jpeg(bytes(msg.data))
        if left_frame is None or right_frame is None:
            return
        self._drop_stale(self._stereo_queue, (left_frame, right_frame))

    @staticmethod
    def _frame_stamp(msg, arrival: float) -> float:
        """帧的"拍摄时刻"：header.stamp（sec + nanosec，epoch 秒）；发布端
        没写 stamp 时退回回调到达时刻。左右帧混用两种来源也能配对——
        差值恒 > 50 ms 的跨域组合本来就该被 sync 窗口拒绝。"""
        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None) if header is not None else None
        if stamp is None:
            return arrival
        # 同域配对用差值，无需换算 epoch；但要防"发布端写了 stamp 却恒 0"
        # 的老驱动——0 也是"没写"，退回到达时间（对齐 ASR 的判定习惯）。
        try:
            sec = getattr(stamp, "sec", 0) or 0
            nanosec = getattr(stamp, "nanosec", 0) or 0
        except (TypeError, ValueError):
            return arrival
        if sec == 0 and nanosec == 0:
            return arrival
        return float(sec) + float(nanosec) * 1e-9

    # ── worker：按模式出云 ─────────────────────────────────────────────────

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                if self._mode == "stereo":
                    left, right = self._stereo_queue.get(timeout=1.0)
                    self._emit_stereo(left, right)
                else:
                    kind, payload = self._frame_queue.get(timeout=1.0)
                    if self._mode == "auto":
                        # mode 只在这里定形（回调不动 _mode），保证懒加载
                        # 分支一定轮得到
                        self._mode = "mono" if kind == "jpeg" else "depth"
                    if kind == "jpeg":
                        self._maybe_load_mono_model()
                        if self._stop_event.is_set():
                            break
                        self._emit_mono(payload)
                    else:
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

    # B: RGB → 单目深度引擎 → 反投影
    def _emit_mono(self, jpeg_bytes: bytes):
        frame = _decode_jpeg(jpeg_bytes)
        if frame is None:
            return
        model = self._model_for_mono()
        if model is None:
            return
        from plugins.vision_runtime import decode_depth

        outputs, meta = model.infer(frame)
        depth_m = decode_depth(outputs, meta)
        h, w = depth_m.shape[:2]
        fx, fy, cx, cy = self._pinhole_for(w, h)
        stride = decimation_stride(w, h, self._max_points)
        xyz = backproject_depth(depth_m, fx, fy, cx, cy, stride,
                                self._min_depth_m, self._max_depth_m)
        self._publish(xyz)

    def _pinhole_for(self, w: int, h: int) -> tuple:
        """(fx, fy, cx, cy)：驱动实测参数 → 上游 camera_info 的 K（跨分辨率
        按 camera_info._rescale_K 缩放）→ hfov 90° 兜底。

        review 指出：上游明明声明了真实内参，反投影却按 90° hfov 硬算，
        横向坐标系统性偏差。K 是像素单位、分辨率变了必须缩放。
        """
        if self._calibration:
            if calibration_tier(self._calibration) != 'pinhole':
                raise ValueError('depth/mono requires pinhole calibration')
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

    def _model_for_mono(self):
        # 由插件在 start 时注入，见 PointCloudPerceptionPlugin._start_node；
        # auto 模式引擎后台加载时 _mono_model 还是 None，插件留了取回钩子。
        model = getattr(self, "_mono_model", None)
        if model is None:
            getter = getattr(self, "_plugin_model", None)
            if callable(getter):
                model = getter()
        return model

    def _maybe_load_mono_model(self):
        """worker：本帧是 jpeg（mono 链路）而引擎未就绪 → 加载再出云。

        加载可能要下载模型（分钟级），阻塞 worker 是有意的：期间 _model_loading
        置位让 info 报 loading，新帧被 _drop_stale 挤掉（maxsize=1），不堆积。
        深度输入永不走这里。"""
        if self._model_for_mono() is not None:
            return
        loader = getattr(self, "_ensure_mono_model", None)
        if callable(loader):
            loader()

    # C: 双目 → 视差 → Q 反投影
    def _emit_stereo(self, left: np.ndarray, right: np.ndarray):
        import cv2

        blob = self._calibration
        tier = calibration_tier(blob)
        if tier not in ("raw", "rectified"):
            # 缺少标定时不生成点云。
            return
        fx, cx, cy, Tx, left_u, right_u = _rectify_pair(
            left, right, blob, getattr(self, "_rectify_cache", None))
        # SGBM 参数照 Go1 经验取：块匹配对小基线更稳，但 SGBM 质量更好。
        block = 5
        sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,   # 64 的倍数；小基线双目 64 足够
            blockSize=block,
            P1=8 * 3 * block * block,
            P2=32 * 3 * block * block,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=2,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        disp = sgbm.compute(left_u, right_u).astype(np.float32) / 16.0
        # SGBM 视差 d ≥ 0，reprojectImageTo3D 的 W = Q[3,2]·d 必须为正
        # 才能得到正深度 → Tx 必须为负。Q_from_baseline 统一取负号。
        Q = Q_from_baseline(fx, cx, cy, Tx)
        xyz = cv2.reprojectImageTo3D(disp, Q, True).reshape(-1, 3).astype(np.float32)
        xyz = filter_cloud(xyz, self._min_depth_m, self._max_depth_m)
        # review 第 11 轮：这里以前用 2-D 的 decimation_stride(w, h)（图像
        # 网格步长，sqrt(w*h/max_points)）去抽已展平的 (N,3) 云 —— 步长
        # 差 sqrt(N) 倍，640x480 密集云实际发出 ~76800 点，是 max_points
        # 的近 4 倍，还超渲染器 40000 上限。抽稀必须按点数本身算。
        xyz = decimate_cloud(xyz, decimation_stride(1, len(xyz), self._max_points))
        self._publish(xyz)

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


def _raw_stereo_params(blob: dict) -> tuple:
    """解析 raw 档 blob → (K1, D1, K2, D2, R, T)。解析不了就 ValueError。

    start 的预检与 _rectify_pair 的缓存 miss 共用这一份 —— 两处各写一遍
    必然漂移（review 第 6 轮：start 只查档位不查数值，坏 blob 拖到首帧
    才在 worker 里炸，卡片显示 running 却一云不出）。"""
    try:
        K1 = np.asarray(blob.get("K1") or blob.get("K_left"), dtype=np.float64).reshape(3, 3)
        K2 = np.asarray(blob.get("K2") or blob.get("K_right"), dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError) as error:
        raise ValueError(f"raw 标定的 K1/K2 不是 9 元素的 3x3 矩阵：{error}")
    D1 = np.asarray(blob.get("D1") or blob.get("D_left") or [0, 0, 0, 0], dtype=np.float64).ravel()
    D2 = np.asarray(blob.get("D2") or blob.get("D_right") or [0, 0, 0, 0], dtype=np.float64).ravel()
    T = np.asarray(blob.get("T") or [-blob.get("Tx", 0.02443), 0, 0], dtype=np.float64).ravel()
    # review 第 21 轮 Issue：T 的长度以前不查 —— {"T": [[-0.05]]} 过得了
    # start 预检（T[0] 有限非零），worker 里 T.reshape(3, 1) 每帧炸，卡片
    # running 却一云不出。平移向量必须恰为 3 元，当场报清楚。
    if T.size != 3:
        raise ValueError(f"raw 标定的 T 不是 3 元素平移向量（{T.size} 个元素）")
    # blob 里带标定出来的 R（stereoCalibrate 的第 6 个返回值）。老 blob 没有
    # R 字段时退单位阵 —— R=I 是"两相机已共线"的假设，比拒收宽容，且与
    # 旧行为兼容（此前恒传 I，等价于所有老 blob 都当共线用）。
    R = np.asarray(blob.get("R") if blob.get("R") is not None else np.eye(3),
                   dtype=np.float64)
    if R.size != 9:
        raise ValueError(f"raw 标定的 R 不是 3x3（{R.size} 个元素）")
    R = R.reshape(3, 3)
    # review 第 10 轮建议：K 已查有限值，D/R 没查 —— 畸变系数/旋转矩阵
    # 里一个 NaN 就让 stereoRectify 的输出全 NaN，映射全废，云一帧不出
    # 且无任何报错。start 预检与 worker 共用这里，当场报清楚。
    for name, arr in (("K1", K1), ("K2", K2), ("D1", D1), ("D2", D2),
                      ("R", R), ("T", T)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"raw 标定的 {name} 含非有限值（NaN/inf）")
    return K1, D1, K2, D2, R, T


def _stereo_blob_from_declarations(left: dict, right: dict) -> tuple:
    """两条输入的 motus.camera/1 声明 → (raw 档 blob, None) 或 (None, 原因)。

    驱动层的型号级出厂标定走这里（review 第 20 轮后的设计：标定不该让
    使用者做）。左目声明的 `stereo_peer` 携带 {id, R, T}，右目声明提供
    自己的 K/D —— 左右各自的内参与畸变本来就描述各自的镜头，peer 只
    补"两镜头之间"的刚体关系。

    成套才用，缺一件整体退回（协议规则：unknown is null, never a guess
    —— 拼半套标定与错误标定在下游不可区分）。peer 用 `id` 匹配而不是
    topic：经过处理器后右图的 topic 已经不是相机原始 topic，`id` 在
    inherit() 里是刻意保活的；匹配不上 = 左右接反或不是一对，拒收并
    说明，这正好是"接反了"唯一能被发现的地点。

    声明缺失分两档：两条输入都没有声明 → (None, "")，调用方落回
    capture-only（老契约：无标定 start 合法）；**带了声明但差件套**
    → (None, 原因)，调用方按错误处理 —— 相机卡片明明声明了双目标定
    却用不了，静默降级会把"驱动数据坏了"藏成"用户没标定"。
    """
    from plugins.camera_info import camera_id

    if not (isinstance(left, dict) and left) and not (isinstance(right, dict) and right):
        return None, ""
    peer = left.get("stereo_peer") if isinstance(left, dict) else None
    if not isinstance(peer, dict):
        return None, "左目声明的 stereo_peer 缺失或不是对象"
    if not peer.get("id"):
        return None, "stereo_peer 缺 id（peer 的相机标识，非话题名）"
    K1, D1 = left.get("K"), left.get("D")
    K2, D2 = right.get("K"), right.get("D") if isinstance(right, dict) else None
    if not (isinstance(K1, (list, tuple)) and len(K1) == 9):
        return None, "左目声明没有可用的 K（9 元素内参矩阵）"
    if not (isinstance(K2, (list, tuple)) and len(K2) == 9):
        return None, "右目声明没有可用的 K（9 元素内参矩阵）"
    if camera_id(right) != str(peer["id"]):
        return None, (f"双目输入与声明不配对：左目声明的 peer 是 {peer['id']!r}，"
                      f"右目声明的 id 是 {camera_id(right)!r} —— 左右接反了？")
    blob = {
        "K1": [list(map(float, K1[0:3])), list(map(float, K1[3:6])), list(map(float, K1[6:9]))],
        "K2": [list(map(float, K2[0:3])), list(map(float, K2[3:6])), list(map(float, K2[6:9]))],
        "D1": [float(v) for v in D1] if isinstance(D1, (list, tuple)) and D1 else [0, 0, 0, 0],
        "D2": [float(v) for v in D2] if isinstance(D2, (list, tuple)) and D2 else [0, 0, 0, 0],
        "R": peer["R"], "T": peer["T"],
        "source": peer.get("source", "unknown"),
    }
    # review 第 21 轮建议：peer 的 R/T 只查了非 None，畸形值（如 1 元素的
    # T）会一路带进 worker 才炸。拼完当场用与配置 blob 同一份解析器全量
    # 预检（长度/有限值/NaN 全查）—— 坏驱动数据在 start 就报清楚。
    try:
        _raw_stereo_params(blob)
    except ValueError as error:
        return None, f"声明的 stereo_peer 标定数据畸形：{error}"
    return blob, None


def _rectified_params(blob: dict) -> tuple:
    """解析 rectified 档 blob → (fx, cx, cy, Tx)，全部有限、fx>0、Tx≠0。

    start 的预检与 _rectify_pair 共用这一份（同 _raw_stereo_params 的道理：
    两处各写一遍必然漂移）。Q 反投影要四个数都能用：fx=0 → 深度恒 0，
    云被 filter_cloud 剔光；Tx=0 → Q[3,2] 除零；cx/cy 非有限 → 点云坐标
    全 NaN。缺省规则与 _rectify_pair 一致：cx/cy 缺省取 fx。"""
    try:
        fx = float(blob["fx"])
        cx = float(blob.get("cx", blob["fx"]))
        cy = float(blob.get("cy", blob["fx"]))
        tx = float(blob["Tx"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"rectified 标定数值异常（需要有限的 fx/cx/cy/Tx）：{error}")
    if not np.isfinite(fx) or fx <= 0:
        raise ValueError(
            f"rectified 标定的焦距无效（fx={blob.get('fx')!r}）："
            "需要正的有限值（像素），否则反投影深度恒 0、一帧云不出")
    if not (np.isfinite(cx) and np.isfinite(cy)):
        raise ValueError(f"rectified 标定的主点无效（cx={cx!r}, cy={cy!r}）")
    if not np.isfinite(tx) or abs(tx) < 1e-6:
        raise ValueError(
            f"rectified 标定的基线 Tx 无效（Tx={blob.get('Tx')!r}）："
            "需要非零有限值（米），否则 Q 反投影每帧除零")
    return fx, cx, cy, tx


def _rectify_pair(left: np.ndarray, right: np.ndarray, blob: dict,
                  cache: Optional[dict] = None):
    """按标定分档把左右帧变校正灰度图。返回 (fx, cx, cy, Tx, L, R)。

    * rectified 档：帧本来就共线，直接灰度化。
    * raw 档：用 blob 里的 K1/K2/D1/D2/R/T 调 cv2.stereoRectify 生成映射。
      blob 在节点生命周期内不变、映射只随分辨率变：传 cache dict 时按
      帧尺寸复用上次的映射与投影参数（review 指出 docstring 声称缓存、
      代码却每帧重算 stereoRectify + initUndistortRectifyMap）。
    """
    import cv2

    tier = calibration_tier(blob)
    if tier == "rectified":
        # 与 start 预检共用 _rectified_params（review 第 10 轮：以前这里
        # 只查 Tx，fx=0/缺失的 blob start 照收、worker 里静默不出云）。
        fx, cx, cy, tx = _rectified_params(blob)
        L = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        R = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        return fx, cx, cy, -abs(tx), L, R

    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    size = (left.shape[1], left.shape[0])
    entry = cache.get("maps") if (cache is not None and cache.get("size") == size) else None
    if entry is None:
        K1, D1, K2, D2, R, T = _raw_stereo_params(blob)
        # review 第 20 轮 Issue：blob 里 T[0] 的符号有两种约定——OpenCV
        # 的 stereoCalibrate 给负（标准左右 rig），部分厂商出厂标定给正。
        # stereoRectify 原样收 T：正 T 下校正后 x_R > x_L、真实视差为负
        # （d = -fx·T[0]/Z），而 SGBM 硬编码 minDisparity=0、Q 的
        # W=Q[3,2]·d 要 d≥0 才有正深度 —— 正 T 的云要么全空要么深度
        # 错。进 rectify 前统一翻成 OpenCV 负号约定（R 不动：纯平移
        # rig 的等价重表达对左右互换就是 T 取负）。
        if float(T[0]) > 0:
            T = -T
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K1, D1, K2, D2, size, R, T.reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        map1, map2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
        map3, map4 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1)
        fx = float(P1[0, 0]); cx = float(P1[0, 2]); cy = float(P1[1, 2])
        # 基线取外参 T[0]，不是 P1[0,3]：stereoRectify 以左目为参考系，平移
        # 放在 P2[0,3]，P1[0,3] 正常为 0 —— 从它推导基线恒得 0，Q 反投影
        # 除零（review 指出的 raw 档无法发布的根因）。Q_from_baseline 只取
        # 长度，统一归一到负号。
        Tx = float(T[0]) if np.any(T) else float(P2[0, 3] / -P2[0, 0])
        if not np.isfinite(Tx) or abs(Tx) < 1e-6:
            # review 建议：坏标定在 start 时报清楚，而不是 worker 每帧一个
            # 千篇一律的 ZeroDivisionError。基线为 0/非有限意味着 blob 里既没
            # 有可用的 T 也没有 Tx，Q 反投影必然除零。
            raise ValueError(f"stereo 标定基线异常（Tx={Tx}）：需要 blob 的 T[0]/Tx，且不能为 0")
        entry = (map1, map2, map3, map4, fx, cx, cy, -abs(Tx))
        if cache is not None:
            cache["size"], cache["maps"] = size, entry
    map1, map2, map3, map4, fx, cx, cy, Tx = entry
    L = cv2.remap(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), map1, map2, cv2.INTER_LINEAR)
    R = cv2.remap(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), map3, map4, cv2.INTER_LINEAR)
    return fx, cx, cy, Tx, L, R


class PointCloudPerceptionPlugin:
    PREFIX = "pointcloud"
    ALIASES = ("stereo_pointcloud",)

    def __init__(self, plugin_cfg: dict, namespace: str, executor):
        self._namespace = namespace
        self._executor = executor
        self._plugin_cfg = dict(plugin_cfg or {})
        self._fps = int(plugin_cfg.get("fps", 2))
        self._max_points = max(100, min(MAX_POINTS, int(plugin_cfg.get("max_points", 20000))))
        self._min_depth_m = float(plugin_cfg.get("min_depth_m", 0.1))
        self._max_depth_m = float(plugin_cfg.get("max_depth_m", 10.0))
        self._model = None
        self._model_loading = False
        self._model_load_error = None
        self._model_load_status = None
        self._model_lock = threading.Lock()
        # mono 共享加载的并发 start 登记（review 第 13 轮）：加载在跑时
        # 后续 start 登记到这里，加载落地后由共享线程统一补齐节点。
        self._model_pending_starts: list[tuple] = []
        self._pending_lock = threading.Lock()
        self._nodes: dict[str, _PointCloudNode] = {}
        self._instance_configs: dict[str, dict] = {}
        self._nodes_lock = threading.RLock()
        # 上游相机的 motus.camera/1 声明（start 时经 for_topic 按输入话题
        # 挑出）：A/B 模式的内参兜底（_pinhole_for）与 info() 的向下继承
        # 都用它。空声明也记录 —— 重启不带声明时替换旧值，不留过期的镜头。
        # _upstream_camera_right 只在 stereo 两路时记录（右目自己的声明）。
        self._upstream_camera: dict[str, dict] = {}
        self._upstream_camera_right: dict[str, dict] = {}
        self._upstream_input_metadata: dict[str, dict] = {}

    # ── lazy engine（mono 模式，与 visual_depth 同一机制与下载器）──────────

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
            from plugins.vision_runtime import VisionEngineSession
            from utils.model_downloader import ensure_depth_model
            from utils.model_progress import fetch_status

            model_dir = os.environ.get("DEPTH_MODEL_DIR", "/models/depth")
            progress_cb, _ = fetch_status(
                lambda text: setattr(self, "_model_load_status", text), "yolo26n-depth")
            paths = ensure_depth_model(model_dir, progress_cb=progress_cb)
            engine = next(p for name, p in paths.items() if name.endswith(".engine"))
            log.info(f"[pointcloud] loading mono depth engine: {engine}")
            self._model = VisionEngineSession(engine)
            log.info(f"[pointcloud] mono depth engine loaded, input={self._model.input_size}")

    def _ensure_model_bg(self):
        """auto 节点用：worker 线程里同步加载引擎（首帧定成 jpeg 时调用）。

        阻塞 worker 是有意的：加载可能要下载模型（分钟级），期间新帧被
        _drop_stale 挤掉（maxsize=1）不堆积。深度输入永不走这里。"""
        with self._model_lock:
            if self._model is not None or self._model_loading:
                return
        self._load_model_sync()

    def _load_model_sync(self) -> bool:
        """置位加载状态 → 同步加载 → 失败时留 error。返回是否由本线程执行。

        显式 mono 的 _bg_start 与 auto 的 _ensure_model_bg 共用这一份，
        _model_loading/_model_load_error 在两条路径上都可见（review 指出
        原先 auto 加载不置位，info 把下载中的卡片报成 idle）。"""
        with self._model_lock:
            if self._model is not None:
                return False  # 已就绪，不必加载
            if self._model_loading:
                return False  # 已有加载在跑
            self._model_loading = True
            self._model_load_error = None
            self._model_load_status = None
        try:
            self._ensure_model()
        except Exception as error:  # noqa: BLE001 — 记下来给 info/dispatch 报
            self._model_load_error = str(error)
            log.error(f"[pointcloud] engine load failed: {error}", exc_info=True)
            return False
        finally:
            self._model_loading = False
            self._model_load_status = None
        return True

    def _bg_start(self):
        """共享加载 + 启动全部登记过的 start 请求（review 第 13/14 轮）。

        赢下 `_load_model_sync()` 的线程承担两件事：落地加载，然后把
        `_model_pending_starts` 里**还活着**的登记（含自己）用 _start_node
        补齐；重复 node_key 被 _start_node 去重，双实例并发 start 因此不会
        丢一条。输家在加载还在跑或已失败时不能清队，否则赢家的收获被自己的
        早退掏空；stop 时清走的条目活着进不了 _nodes，也就不会再被启动。

        任何线程只要看到加载已落地（本线程 _load_model_sync 成功，或本线程
        之外 _model is not None 且无 error），就清走队列——清队幂等，落节点
        靠 node_key 去重，不会重复。队列由 _bg_start 独占 drained（stop 摘
        走条目与 _bg_start 读队列互斥——pending_lock 串行，strip 进不了
        drain，drain 也看不到 strip），drained 条目一律落成节点。"""
        loaded = self._load_model_sync() or (
            self._model is not None and not self._model_load_error)
        if not loaded:
            return  # 输家：加载失败或还在跑（本线程没赢）
        with self._pending_lock:
            starts = self._model_pending_starts
            self._model_pending_starts = []
            # Serialize draining and node creation with stop, so a drained
            # request cannot recreate a node after its stop has returned.
            for entry in starts:
                self._start_node(*entry)

    # ── node 生命周期 ──────────────────────────────────────────────────────

    def _resolve_mode(self, args: dict,
                      instance_id: str = "") -> tuple[str, Optional[str], Optional[str], Optional[dict], str]:
        """→ (mode, left_topic, right_topic, calibration, declared_note)。auto 按输入判定。

        单话题 auto 不猜话题名（"/camera/image_raw" 也可能载着 z16 深度）：
        按 explicit "depth" 之外一律走 "auto" 节点 —— start 时同时订阅
        Image + CompressedImage，回调按消息真实类型分派（见 _auto_*_cb），
        首帧确定 depth / mono。
        """
        topics_list = list(args.get("input_topics") or [])
        input_topic = args.get("input_topic") or (topics_list[0] if topics_list else "")
        right_topic = topics_list[1] if len(topics_list) > 1 else None
        mode = args.get("mode") or "auto"
        # review 建议：不认识的 mode 以前会静默当成 mono 订阅（start 的
        # else 分支只订 CompressedImage），depth 拼错成 detph 就永远收不到
        # z16 帧。这里报清楚而不是猜。input_topics 超过 2 路同理 ——
        # 第三路会被静默忽略，双目只认前两路。
        if mode not in ("auto", "depth", "mono", "stereo"):
            raise ValueError(f"不支持的 mode：{mode!r}（可选 auto / depth / mono / stereo）")
        if len(topics_list) > 2:
            raise ValueError(f"input_topics 最多 2 路（左、右），收到 {len(topics_list)} 路")
        # 旧画布残留的手动标定不再参与计算，参数统一由驱动维护。
        calibration = None
        if mode == "auto":
            mode = "stereo" if right_topic else "auto"
        declared_blob, declared_note = None, ""
        if mode == "stereo" and right_topic:
            from plugins.camera_info import for_topic as _camera_for_topic

            declared_blob, declared_note = _stereo_blob_from_declarations(
                _camera_for_topic(args.get("camera_info"), input_topic),
                _camera_for_topic(args.get("camera_info"), right_topic))
            if declared_blob is not None:
                calibration = declared_blob
        return mode, input_topic or None, right_topic, calibration, declared_note

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

    def _loading_camera_info(self, args: dict, instance_id: str) -> dict:
        """`{"camera_info": [...]}`：loading 期间 info() 也要带声明（同
        visual_depth / vop 的教训：早退分支不能把声明丢在地上）。"""
        topic = args.get("input_topic") or ""
        if not topic:
            topics = args.get("input_topics") or []
            topic = topics[0] if topics else ""
        cloud_topic, summary_topic = output_topics_for(topic)
        declared, _ = self._camera_info(instance_id, topic, {},
                                        cloud_topic, summary_topic)
        return {"camera_info": declared} if declared else {}

    def _start_node(self, node_key: str, mode: str, input_topic: Optional[str],
                    right_topic: Optional[str], calibration):
        with self._nodes_lock:
            if node_key in self._nodes:
                return
            icfg = self._instance_configs.get(node_key, {})
            effective = self._effective_config(icfg)
            node = _PointCloudNode(
                mode, input_topic, right_topic,
                calibration,
                fps=effective['fps'], max_points=effective['max_points'],
                min_depth_m=effective['min_depth_m'], max_depth_m=effective['max_depth_m'],
                node_suffix=node_key.replace("/", "_").replace("-", "_").lstrip("_"),
            )
            node._applied_config = effective
            node._metric_cloud_enabled = effective['metric_cloud']
            if node._metric_cloud_enabled:
                node._metric_cloud_pub = node.create_publisher(PointCloud2, node._metric_cloud_topic, _PUB_QOS)
            if mode == "mono":
                node._mono_model = self._model
            elif mode == "auto":
                # auto 可能落到 mono：引擎已就绪就注入，否则首帧判定为 jpeg 后
                # 由 _ensure_mono_model 后台加载（深度输入则永不加载）。
                node._mono_model = self._model
                node._plugin_model = lambda: self._model
                node._ensure_mono_model = self._ensure_model_bg
            with self._nodes_lock:
                upstream = self._upstream_camera.get(node_key) or {}
            declaration = self._upstream_input_metadata.get(node_key) or {}
            node._input_format = declaration.get("format") or upstream.get("format")
            metadata_topic = declaration.get("frame_metadata_topic")
            if node._metric_cloud_enabled and isinstance(metadata_topic, str) and metadata_topic:
                node._metric_metadata_source_topic = metadata_topic
                node._metric_metadata_pub = node.create_publisher(
                    String, node._metric_metadata_topic, _PUB_QOS)
            # A/B 模式的反投影内参兜底用上游 K（见 _pinhole_for）；
            # stereo 的 K 是双目标定的，不掺上游单目声明。
            if mode in ("depth", "mono", "auto"):
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
            if mode == "stereo" and calibration:
                node._calibration_source = calibration.get("source")
            self._executor.add_node(node)
            try:
                node.start()
            except Exception:
                dispose_node(self._executor, node, label=f'pointcloud/{node_key}')
                raise
            self._nodes[node_key] = node
        log.info(f"[pointcloud] node started (background): mode={mode} "
                 f"{input_topic or '(none)'}{' +' + right_topic if right_topic else ''}")

    def _retire_node(self, node_key: str) -> Optional[dict]:
        with self._nodes_lock:
            node = self._nodes.pop(node_key, None)
        if node is None:
            return None
        with self._nodes_lock:
            # 声明随节点走：留着会让重新接线的卡片拿一个已经不喂它的相机
            # 的光学参数回答 info()（同 visual_depth 的注释）。
            self._upstream_camera.pop(node_key, None)
            self._upstream_camera_right.pop(node_key, None)
            self._upstream_input_metadata.pop(node_key, None)
        node.request_stop()
        result = node.stop()
        dispose_node(self._executor, node, label=f"pointcloud/{node_key}")
        return result

    def shutdown(self) -> None:
        """Retire all nodes before ROS/interpreter teardown (review 第 17 轮
        Issue 2)。

        PerceptionBundle.shutdown() 只调用带 shutdown() 的插件；没有这一
        份的话，SIGTERM 一路走到 executor.shutdown()/rclpy.shutdown() 时
        本插件的订阅、发布和 worker 线程还全活着——worker 还可能对着已
        关闭的 ROS context 发布。清 pending 队列：等共享模型下载的 mono
        start 不该在关停中途被 _bg_start 拉起。模型本身没有需要 close()
        的 GPU 资源（TRT 引擎随进程走），不比照 visual_depth 关 backend。"""
        with self._pending_lock:
            self._model_pending_starts = []
            with self._nodes_lock:
                keys = list(self._nodes)
            for key in keys:
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
        Issue 3）。单值合法、组合非法（min >= max）以前照收——filter_cloud
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
            # 下载契约（同 visual_depth 的 info）：加载中报 loading + 进度，
            # 失败报 error —— 否则 mono 加载期间卡片显示 idle，运维看不到
            # 一次可能长达数分钟的模型下载。
            if self._model_loading:
                return {"name": "PointCloudPerception", "manufacture": "Embodied",
                        "model": "depth|stereo", "state": "loading",
                        "desc": self._model_load_status or "Loading depth engine...",
                        "instances": {}, "topic_in": [], "topic_out": [],
                        **self._loading_camera_info(args, instance_id)}
            if self._model_load_error:
                return {"name": "PointCloudPerception", "manufacture": "Embodied",
                        "model": "depth|stereo", "state": "error",
                        "desc": f"Engine load failed: {self._model_load_error}",
                        "instances": {}, "topic_in": [], "topic_out": []}
            with self._nodes_lock:
                nodes = dict(self._nodes)
            instances = {
                key: {
                    "mode": node._mode,
                    "input": node._input_topic or None,
                    "input_right": node._input_topic_right or None,
                    "cloud_topic": node._cloud_topic,
                    "summary_topic": node._summary_topic,
                    "frame_count": node._frame_count,
                    # 标定来源：声明层（驱动型号级出厂标定）还是用户配置 ——
                    # 出云错了排查方向完全不同。
                    "calibration_source": getattr(node, "_calibration_source", None),
                    "capture_only": node._mode == 'stereo' and calibration_tier(node._calibration) not in ('raw', 'rectified'),
                    "metric_cloud": node._metric_status() if node._metric_cloud_enabled else None,
                }
                for key, node in nodes.items()
            }
            # agent-core 的 _dropped_inputs 按这里报告的 topic 判定：
            # 双目必须把两路都报出来，否则卡片以"忽略了一路输入"失败。
            topics_in: list = []
            if instance_id and instance_id in nodes:
                node = nodes[instance_id]
                if node._input_topic:
                    topics_in.append({"topic": node._input_topic,
                                      "format": node._input_format or ("image/depth-z16" if node._metric_cloud_enabled else "image/jpeg")})
                if node._input_topic_right:
                    topics_in.append({"topic": node._input_topic_right, "format": "image/jpeg"})
            else:
                for node in nodes.values():
                    if node._input_topic and all(t["topic"] != node._input_topic for t in topics_in):
                        topics_in.append({"topic": node._input_topic,
                                          "format": node._input_format or ("image/depth-z16" if node._metric_cloud_enabled else
                                                    "image/depth-z16" if node._mode == "depth" else "image/jpeg")})
                    if node._input_topic_right and all(t["topic"] != node._input_topic_right for t in topics_in):
                        topics_in.append({"topic": node._input_topic_right, "format": "image/jpeg"})
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
                "model": "depth|stereo",
                "state": "running" if instances else "idle",
                "instances": instances,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "3D point cloud from depth map / mono RGB / stereo pair",
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
                mode, input_topic, right_topic, calibration, declared_note = \
                    self._resolve_mode(args, instance_id)
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
                if mode == "auto":
                    mode = "depth"
            if mode == "stereo":
                if not (input_topic and right_topic):
                    return {"state": "error",
                            "message": "stereo 模式需要左右两路输入（input_topics[0]=左, [1]=右）"}
                if not calibration or calibration_tier(calibration) not in ("raw", "rectified"):
                    if declared_note:
                        # 声明层想兜底但差了件套：报清楚缺什么，而不是静默
                        # 落 capture-only 让人以为是没标定。
                        return {"state": "error",
                                "message": f"驱动声明的 stereo 标定不完整：{declared_note}"}
                    return {"state": "error", "message": "stereo 模式需要驱动声明完整相机参数"}
                elif calibration_tier(calibration) == "rectified":
                    # review 第 6/10 轮：以前只查 Tx（Tx=0 → Q[3,2] 每帧除零）。
                    # 但 Q 的其余输入同样致命：fx=0 → Z=fx·Tx/w 恒 0，云被
                    # filter_cloud 全剔光，卡片 running 却一帧不出；fx 缺失/
                    # 非数值则在 _rectify_pair 里每帧抛异常，同样静默。启动时
                    # 用与 _rectify_pair 同一份解析器全量预检。
                    try:
                        fx, cx, cy, tx = _rectified_params(calibration)
                    except ValueError as error:
                        return {"state": "error", "message": str(error)}
                else:
                    # review 第 6 轮：raw 档以前 start 只查档位不查内容，
                    # K 缺失/畸形拖到首帧才在 worker 里炸 —— 卡片显示
                    # running 却一云不出。启动时用与 _rectify_pair 同一份
                    # 解析器预检（含基线非零有限），坏配置当场报清楚。
                    try:
                        K1, _D1, K2, _D2, _R, T = _raw_stereo_params(calibration)
                        tx = float(T[0]) if T.size else 0.0
                        if not np.isfinite(tx) or abs(tx) < 1e-6:
                            raise ValueError(
                                f"基线异常（Tx={tx}）：T/Tx 缺失或为 0，Q 反投影必然除零")
                        if not (np.all(np.isfinite(K1)) and np.all(np.isfinite(K2))):
                            raise ValueError("K1/K2 含非有限值（NaN/inf）")
                    except ValueError as error:
                        return {"state": "error",
                                "message": f"raw 标定无法使用：{error}"}
            elif not input_topic:
                # review 第 8 轮：这条检查以前只挡 mode == "auto"。显式
                # mode="depth"/"mono" 不带 input_topic 会一路走到
                # _start_node —— start() 不建任何订阅却把自己标成 running，
                # 从此卡片永远 idle 却报 running，且没有任何报错可查。
                # stereo 已在上面要求双路，这里覆盖其余全部模式。
                return {"state": "error",
                        "message": "需要一路输入：input_topic（深度图/单目）或 input_topics（双目）"}
            if metric_enabled:
                if mode != "depth" or not isinstance(calibration, dict):
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
                                 existing._input_topic_right != (right_topic or '') or
                                 (mode != 'auto' and existing._mode != mode) or
                                 declaration != self._upstream_input_metadata.get(node_key, {})):
                    return {'state': 'error', 'message': 'stop the pointcloud instance before changing its inputs, mode or metadata'}
                self._upstream_input_metadata[node_key] = declaration
            # 先记录再起节点（同 visual_depth）：loading 期间 info() 也能回答；
            # 空声明同样记录 —— 重启不再带声明时要替换旧值而不是留过期镜头。
            # stereo 两路各记一份：声明层标定（_stereo_blob_from_declarations）
            # 与 info() 的向下继承都要用右目的。
            if input_topic:
                from plugins.camera_info import for_topic as _camera_for_topic

                declarations = args.get("camera_info")
                with self._nodes_lock:
                    self._upstream_camera[node_key] = _camera_for_topic(
                        declarations, input_topic)
                    if right_topic:
                        self._upstream_camera_right[node_key] = _camera_for_topic(
                            declarations, right_topic)
            with self._nodes_lock:
                running = self._nodes.get(node_key)
            if running is None:
                # mono 一定需要引擎：后台加载完再起节点。auto 首帧才定
                # depth/mono，立即起节点、首帧是 jpeg 时再懒加载（纯深度
                # 相机永远不碰 TRT）。
                needs_model = (mode == "mono" and self._model is None)
                if needs_model:
                    if self._model_loading:
                        # 加载在跑时先登记（含本条），落地后由赢下的
                        # _bg_start 统一启动：双实例并发 start 才不会有一条
                        # 停在"加载完成但只有 winner 启动 node"。
                        with self._pending_lock:
                            self._model_pending_starts.append(
                                (node_key, mode, input_topic, right_topic, calibration))
                        return {"state": "loading",
                                "message": (self._model_load_status or
                                            "Engine is still loading, please wait...")}
                    if self._model_load_error:
                        return {"state": "error",
                                "message": f"Engine failed to load: {self._model_load_error}"}
                    # 还没有加载在跑：先登记（含本条）再启动后台线程——顺序不能
                    # 反，否则线程可能在本条登记前就完成加载并清走队列。
                    with self._pending_lock:
                        self._model_pending_starts.append(
                            (node_key, mode, input_topic, right_topic, calibration))
                    threading.Thread(target=self._bg_start, daemon=True,
                                     name="pointcloud_model_load").start()
                    return {"state": "loading", "mode": mode, "input": input_topic,
                            "message": "Engine loading in background, will start automatically"}
                try:
                    self._start_node(node_key, mode, input_topic, right_topic, calibration)
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
            if instance_id:
                # review 第 14 轮：stop 必须取消该卡片 pending 的 mono
                # 启动——它可能还在等共享模型下载，否则 _bg_start 落完
                # 加载后会把已停掉卡片按旧 topic 重新拉起。节点尚未落地
                # （还在 pending）时 retire_node 返回 None，pending 条目
                # 仍在——同一次过滤两条路径都覆盖，不必分两次筛
                # （review 第 17 轮建议：去掉重复的第二次过滤）。
                with self._pending_lock:
                    self._model_pending_starts = [
                        e for e in self._model_pending_starts
                        if e[0] != instance_id]
                    result = self._retire_node(instance_id)
                return result if result is not None else {"state": "idle"}
            # review 第 15 轮：全停（不带 instance_id）同样要清 pending 队列
            # ——否则还在等共享模型下载的显式 mono 实例会在加载落地后被
            # _bg_start 按"已停掉"的状态重新拉起（回复却已是 idle）。
            with self._pending_lock:
                self._model_pending_starts = []
                with self._nodes_lock:
                    keys = list(self._nodes.keys())
                results = [key for key in keys if self._retire_node(key) is not None]
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
