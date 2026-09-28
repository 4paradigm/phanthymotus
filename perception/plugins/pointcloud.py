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

标定：A/B 只需 {fx, fy, cx, cy} 或 hfov（缺省 90°；上游 camera_info 声明
了 K 时优先用 K，见 _pinhole_for）；C 需 {fx, cx, cy, Tx}（已校正）或完整
stereoCalibrate blob（未校正，自动 rectify）。`calibrate` action 面向 C：
现场棋盘格采样（同一对帧不算新姿态）→ stereoCalibrate → 默认保留完整
raw blob（运行时 rectify；只有明确 pre_rectified=true 才落快捷档 ——
rectify 后的 Δy 对任何标定都 ≈ 0，不能拿来分档）→ 落盘
/models/stereo_calib/ 并返回可粘贴回配置的 blob。

纯数学在 plugins/pointcloud_math.py（不依赖 cv2 / rclpy，本地可测）。
"""

from __future__ import annotations

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
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String, UInt8MultiArray

from plugins.pointcloud_math import (
    MAX_POINTS, CALIB_DIR,
    backproject_depth, calibration_tier, decimate_cloud, decimation_stride,
    default_calib_name, encode_packet, filter_cloud, intrinsics_from_hfov,
    load_calibration_blob, pinhole_of, Q_from_baseline, sanitize_calib_name,
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

# calibrate action 的采样节奏：默认收 15 对，每对之间停一拍让手挪棋盘格
_CALIB_PAIRS_TARGET = 15


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
                    "enum": ["start", "stop", "info", "config", "calibrate"],
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
                "mode": {
                    "type": "string",
                    "enum": ["auto", "depth", "mono", "stereo"],
                    "description": "输入模式。auto（默认）：2 路输入 → 双目 stereo；1 路输入同时订阅 Image/CompressedImage 并按首帧消息类型自动选 depth（z16/zlib 深度图）或 mono（jpeg）",
                },
                "calibration": {
                    "type": "string",
                    "description": "标定：JSON 对象或 /models 下的标定文件路径。深度图/单目要 {fx,fy,cx,cy} 或 {hfov}（缺省 90°）；双目要 {fx,cx,cy,Tx}（已校正）或完整 stereoCalibrate blob（未校正，自动 rectify）。双目没标定也可以先 start（只采集不出云），再用 calibrate 现场标定后把 blob 填回来重启",
                },
                "board_w": {"type": "integer", "description": "calibrate 用：棋盘格内角点列数（默认 9）"},
                "board_h": {"type": "integer", "description": "calibrate 用：棋盘格内角点行数（默认 6）"},
                "square_m": {"type": "number", "description": "calibrate 用：棋盘格格距（米，默认 0.025）"},
                "pairs": {"type": "integer", "description": "calibrate 用：要采集的棋盘格姿态数（默认 15）"},
                "name": {"type": "string", "description": "calibrate 用：标定结果文件名（默认按时间戳），落在 /models/stereo_calib/ 下。只能含字母/数字/点/下划线/连字符"},
                "pre_rectified": {
                    "type": "boolean",
                    "description": (
                        "calibrate 用：默认 false。只有当上游保证左右两路已经过立体校正"
                        "（行对齐、无畸变）才设 true —— 此时只保留 Q 反投影四要素（运行时跳过 rectify）。"
                        "默认保留完整标定 blob，运行时逐帧 rectify"
                    ),
                },
            },
            "required": ["action"],
            "x-action-params": {
                "start": {
                    "params": ["input_topic", "input_topics", "mode", "calibration"],
                    "description": "启动。给 input_topic（单输入）或 input_topics（双目左右两路）。mode 留空则按输入与消息类型自动判定",
                },
                "stop": {"params": [], "description": "停止点云输出"},
                "info": {"params": [], "description": "查看状态、模式与输出话题"},
                "config": {"params": ["calibration"], "description": "更新实例配置（calibration / fps / max_points 等），需重启实例生效"},
                "calibrate": {
                    "params": ["board_w", "board_h", "square_m", "pairs", "name", "pre_rectified"],
                    "description": (
                        "双目标定：把棋盘格举到双目前，保持左右同帧可见，"
                        "每换一个姿态调用一次直到采满 pairs（默认 15）对"
                        "（还是同一对帧会被拒，挪动棋盘出新帧再调）。"
                        "完成后返回 rms / 角点数 / 标定 blob（可直接粘贴进卡片配置），"
                        "并落盘到 /models/stereo_calib/。默认保留完整 blob（运行时 rectify）；"
                        "只有上游明确保证已 pre-rectified 才用快捷档。"
                        "需要卡片以双目模式 start 过（无标定也可以 start，只采集不出云）"
                    ),
                },
            },
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "fps": {"type": "integer", "description": "Max output frames per second", "default": 2, "scope": "instance"},
                "max_points": {"type": "integer", "description": "每帧最多点数（上限 40000，由渲染器决定）", "default": 20000, "scope": "instance"},
                "min_depth_m": {"type": "number", "description": "最近有效距离（米）", "default": 0.1, "scope": "instance"},
                "max_depth_m": {"type": "number", "description": "最远有效距离（米）", "default": 10.0, "scope": "instance"},
                "calibration": {"type": "string", "description": "标定 JSON 或 /models 下的标定文件路径", "scope": "instance"},
                "cal_a": {"type": "number", "description": "mono 模式站点标定指数 a（d^a），同 visual_depth", "default": 1.0, "scope": "instance"},
                "cal_b": {"type": "number", "description": "mono 模式站点标定偏移 b（乘 e^b），同 visual_depth", "default": 0.0, "scope": "instance"},
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
        ],
    }
]


# review 第 6 轮：畸形 depth-zlib 帧以前每回调刷一条 warning，2 fps 的坏流
# 一天十几万行。第 1 条与之后每第 100 条采样记录，解出一帧好数据即复位
# —— 连续失败是状态，散落的失败是噪声。模块级计数：日志洪峰按进程算。
_zlib_fail_count = 0


def _decode_depth_message(msg) -> tuple[Optional[np.ndarray], str]:
    """z16 Image 或 depth-zlib CompressedImage → (深度图[米], 编码名)。

    返回 (None, 编码名) 表示该帧不可用。zlib 帧解出来是 640x480 uint16 毫米
    —— 与 visual_depth 的发布格式一致，解码端共用同一约定。
    """
    import cv2

    global _zlib_fail_count

    if isinstance(msg, Image):
        if msg.encoding not in ("16UC1", "mono16"):
            return None, msg.encoding
        needed = int(msg.height) * int(msg.step)
        if msg.width <= 0 or msg.height <= 0 or msg.step < msg.width * 2 or len(msg.data) < needed:
            return None, msg.encoding
        dtype = np.dtype(">u2" if msg.is_bigendian else "<u2")
        depth = np.frombuffer(msg.data[:needed], dtype=dtype).reshape(msg.height, msg.step // 2)
        depth = depth[:, :msg.width].astype(np.float32) * 0.001
        return depth, msg.encoding

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
    if len(raw) < 640 * 480 * 2:
        return None, "depth-zlib"
    _zlib_fail_count = 0
    depth = np.frombuffer(raw[:640 * 480 * 2], dtype="<u2").reshape(480, 640).astype(np.float32) * 0.001
    return depth, "depth-zlib"


def zlib_decompress(data: bytes) -> bytes:
    import zlib
    return zlib.decompress(data)


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
                 max_depth_m: float, cal_a: float, cal_b: float,
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
        self._cal_a = cal_a
        self._cal_b = cal_b
        self._cloud_topic, self._summary_topic = output_topics_for(input_topic)
        # raw 档 rectify 映射缓存（_rectify_pair）：blob 固定，映射只随
        # 分辨率变 —— 每帧重算 stereoRectify/initUndistortRectifyMap 是
        # review 指出的白烧 CPU。worker 线程独占，无需加锁。
        self._rectify_cache: dict = {}

        self._cloud_pub = self.create_publisher(UInt8MultiArray, self._cloud_topic, _PUB_QOS)
        self._summary_pub = self.create_publisher(String, self._summary_topic, _PUB_QOS)

        self._subs: list = []
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stereo_queue: queue.Queue = queue.Queue(maxsize=1)
        # 双目左右帧按到达配对：{stamp: frame}，旧于 sync 窗口的直接丢
        self._left_latest: Optional[tuple[float, np.ndarray]] = None
        self._left_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._worker_error_count = 0
        self._last_emit_time = 0.0
        self._frame_count = 0
        self._running = False
        # calibrate 用：最近一对左右帧（解好色的 BGR），棋盘格检测在
        # calibrate 线程里做，不在回调里做。_stereo_pair_seq 随每对新帧
        # 自增 —— calibrate 靠它识别"还是同一帧"（review 指出的重复采样）。
        self._last_stereo_pair: Optional[tuple] = None
        self._stereo_pair_seq = 0
        self._lifecycle_lock = threading.RLock()

    # ── 生命周期（同 visual_depth：request_stop 不拿锁；stop 只收 worker）──

    def request_stop(self) -> None:
        self._stop_event.set()

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
                if self._mode == "depth":
                    # z16 用 Image，zlib 用 CompressedImage；两路都订上，
                    # 哪个来数据用哪个（摄像头驱动只会发其中一种）。
                    self._subs.append(self.create_subscription(
                        Image, self._input_topic, self._depth_image_cb, _LOW_LAT_QOS))
                    self._subs.append(self.create_subscription(
                        CompressedImage, self._input_topic, self._depth_comp_cb, _LOW_LAT_QOS))
                elif self._mode == "auto":
                    # 话题名不说明载的是什么（"/camera/image_raw" 也可能是
                    # z16 深度）：Image + CompressedImage 都订上，回调按消息
                    # 真实类型分派 —— Image→depth-z16；CompressedImage.format
                    # 含 depth→depth-zlib，否则当 jpeg 走 mono。
                    self._subs.append(self.create_subscription(
                        Image, self._input_topic, self._auto_image_cb, _LOW_LAT_QOS))
                    self._subs.append(self.create_subscription(
                        CompressedImage, self._input_topic, self._auto_comp_cb, _LOW_LAT_QOS))
                else:  # mono
                    self._subs.append(self.create_subscription(
                        CompressedImage, self._input_topic, self._jpeg_cb, _LOW_LAT_QOS))
            if self._subs:
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
            if self._worker and self._worker.is_alive():
                self._worker.join(timeout=3.0)
            self._worker = None
            self._running = False
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
        fmt = str(getattr(msg, "format", "") or "")
        if "depth" in fmt:
            self._drop_stale(self._frame_queue, ("depth_zlib", msg))
        else:
            self._drop_stale(self._frame_queue, ("jpeg", bytes(msg.data)))

    def _left_cb(self, msg: CompressedImage):
        frame = _decode_jpeg(bytes(msg.data))
        if frame is None:
            return
        # 只存最新左帧；配对在 _right_cb 里做（右帧到达时找时间最近的左帧）。
        with self._left_lock:
            self._left_latest = (time.monotonic(), frame)

    def _right_cb(self, msg: CompressedImage):
        frame = _decode_jpeg(bytes(msg.data))
        if frame is None:
            return
        stamp = time.monotonic()
        if self._throttled():
            return
        with self._left_lock:
            left = self._left_latest
            if left is not None and stamp - left[0] <= _STEREO_SYNC_S:
                self._drop_stale(self._stereo_queue, (left[1], frame))
            self._left_latest = None

    # ── worker：按模式出云 ─────────────────────────────────────────────────

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                if self._mode == "stereo":
                    left, right = self._stereo_queue.get(timeout=1.0)
                    self._last_stereo_pair = (left, right)
                    self._stereo_pair_seq += 1
                    self._emit_stereo(left, right)
                else:
                    kind, payload = self._frame_queue.get(timeout=1.0)
                    if self._mode == "auto":
                        # mode 只在这里定形（回调不动 _mode），保证懒加载
                        # 分支一定轮得到
                        self._mode = "mono" if kind == "jpeg" else "depth"
                    if kind == "jpeg":
                        self._maybe_load_mono_model()
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
            return
        h, w = depth_m.shape[:2]
        fx, fy, cx, cy = self._pinhole_for(w, h)
        stride = decimation_stride(w, h, self._max_points)
        xyz = backproject_depth(depth_m, fx, fy, cx, cy, stride,
                                 self._min_depth_m, self._max_depth_m)
        self._publish(xyz)

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
        # 站点标定（同 visual_depth：metres**a * e^b）
        if not (self._cal_a == 1.0 and self._cal_b == 0.0):
            depth_m = np.nan_to_num(
                np.power(np.maximum(depth_m, 0.0), self._cal_a) * float(np.exp(self._cal_b)),
                nan=0.0, posinf=0.0, neginf=0.0)
        h, w = depth_m.shape[:2]
        fx, fy, cx, cy = self._pinhole_for(w, h)
        stride = decimation_stride(w, h, self._max_points)
        xyz = backproject_depth(depth_m, fx, fy, cx, cy, stride,
                                self._min_depth_m, self._max_depth_m)
        self._publish(xyz)

    def _pinhole_for(self, w: int, h: int) -> tuple:
        """(fx, fy, cx, cy)：实例标定 → 上游 camera_info 的 K（跨分辨率
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
            # 无标定 stereo 是合法的"标定采集"状态（start 允许，见 dispatch）：
            # 只配对存帧供 calibrate 用，不出云。
            return
        fx, cx, cy, Tx, left_u, right_u = _rectify_pair(
            left, right, blob, getattr(self, "_rectify_cache", None))
        h, w = left_u.shape[:2]
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
        stride = decimation_stride(w, h, self._max_points)
        xyz = decimate_cloud(xyz, stride)
        self._publish(xyz)

    def _publish(self, xyz: np.ndarray):
        if xyz.shape[0] == 0:
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
    # blob 里带标定出来的 R（stereoCalibrate 的第 6 个返回值）。老 blob 没有
    # R 字段时退单位阵 —— R=I 是"两相机已共线"的假设，比拒收宽容，且与
    # 旧行为兼容（此前恒传 I，等价于所有老 blob 都当共线用）。
    R = np.asarray(blob.get("R") if blob.get("R") is not None else np.eye(3),
                   dtype=np.float64)
    if R.size != 9:
        raise ValueError(f"raw 标定的 R 不是 3x3（{R.size} 个元素）")
    R = R.reshape(3, 3)
    return K1, D1, K2, D2, R, T


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
        try:
            tx = float(blob["Tx"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"stereo 标定基线异常（Tx={blob.get('Tx')!r}）：{error}")
        if not np.isfinite(tx) or abs(tx) < 1e-6:
            # 与 raw 档同一契约：Q_from_baseline 里 -1/Tx，Tx=0 每帧除零
            # （review 指出 start 只查档位不查数值）。start 会提前拒绝，
            # 这里是 worker 侧的兜底。
            raise ValueError(f"stereo 标定基线异常（Tx={tx}）：rectified 档的 Tx 必须非零")
        L = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        R = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        return float(blob["fx"]), float(blob.get("cx", blob["fx"])), \
            float(blob.get("cy", blob["fx"])), -abs(tx), L, R

    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    size = (left.shape[1], left.shape[0])
    entry = cache.get("maps") if (cache is not None and cache.get("size") == size) else None
    if entry is None:
        K1, D1, K2, D2, R, T = _raw_stereo_params(blob)
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
        self._calibration = load_calibration_blob(plugin_cfg.get("calibration"))
        # mono 模式的引擎与站点标定（复用 visual_depth 的下载器与机制）
        self._cal_a = float(plugin_cfg.get("cal_a", 1.0))
        self._cal_b = float(plugin_cfg.get("cal_b", 0.0))
        self._model = None
        self._model_loading = False
        self._model_load_error = None
        self._model_load_status = None
        self._model_lock = threading.Lock()
        self._nodes: dict[str, _PointCloudNode] = {}
        self._instance_configs: dict[str, dict] = {}
        self._nodes_lock = threading.RLock()
        # 上游相机的 motus.camera/1 声明（start 时经 for_topic 按输入话题
        # 挑出）：A/B 模式的内参兜底（_pinhole_for）与 info() 的向下继承
        # 都用它。空声明也记录 —— 重启不带声明时替换旧值，不留过期的镜头。
        self._upstream_camera: dict[str, dict] = {}
        # calibrate 的采集状态：{(instance_id): {"pairs": [...], "target": N}}
        self._cal_sessions: dict[str, dict] = {}
        self._cal_lock = threading.Lock()

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

    # ── node 生命周期 ──────────────────────────────────────────────────────

    def _resolve_mode(self, args: dict) -> tuple[str, Optional[str], Optional[str], Optional[dict]]:
        """→ (mode, left_topic, right_topic, calibration)。auto 按输入判定。

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
        calibration = load_calibration_blob(args.get("calibration")) or self._calibration
        if mode == "auto":
            mode = "stereo" if right_topic else "auto"
        return mode, input_topic or None, right_topic, calibration

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
            max_points = max(100, min(MAX_POINTS, int(icfg.get("max_points", self._max_points))))
            node = _PointCloudNode(
                mode, input_topic, right_topic,
                calibration or load_calibration_blob(icfg.get("calibration")),
                fps=int(icfg.get("fps", self._fps)),
                max_points=max_points,
                min_depth_m=float(icfg.get("min_depth_m", self._min_depth_m)),
                max_depth_m=float(icfg.get("max_depth_m", self._max_depth_m)),
                cal_a=float(icfg.get("cal_a", self._cal_a)),
                cal_b=float(icfg.get("cal_b", self._cal_b)),
                node_suffix=node_key.replace("/", "_").replace("-", "_").lstrip("_"),
            )
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
            # A/B 模式的反投影内参兜底用上游 K（见 _pinhole_for）；
            # stereo 的 K 是双目标定的，不掺上游单目声明。
            if mode in ("depth", "mono", "auto"):
                K = upstream.get("K")
                if isinstance(K, (list, tuple)) and len(K) == 9:
                    node._upstream_intrinsics = (
                        [float(v) for v in K],
                        upstream.get("width"), upstream.get("height"))
            self._executor.add_node(node)
            self._nodes[node_key] = node
        node.start()
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
        node.request_stop()
        result = node.stop()
        dispose_node(self._executor, node, label=f"pointcloud/{node_key}")
        with self._cal_lock:
            self._cal_sessions.pop(node_key, None)
        return result

    # ── calibrate（stereo）────────────────────────────────────────────────

    def _stereo_pair_for_calibration(self, instance_id: str) -> tuple:
        """取标定用的 (pair, seq)。seq 随每对新帧自增，calibrate 靠它
        区分"新姿态"与"同一帧重复调用"（review 指出的重复采样）。"""
        with self._nodes_lock:
            node = self._nodes.get(instance_id) if instance_id else None
            if node is None:
                node = self._nodes.get(_DEFAULT_INSTANCE)
            if node is None and len(self._nodes) == 1:
                node = next(iter(self._nodes.values()))
        if node is None:
            raise ValueError("no stereo instance running — start the card on both cameras first")
        if node._mode != "stereo":
            raise ValueError(f"instance is in {node._mode} mode, calibrate needs stereo")
        pair = node._last_stereo_pair
        if pair is None:
            raise ValueError("no stereo frame received yet — check both cameras are publishing")
        return pair, node._stereo_pair_seq

    def _calibrate(self, args: dict, instance_id: str) -> dict:
        import cv2

        board_w = int(args.get("board_w", 9))
        board_h = int(args.get("board_h", 6))
        square = float(args.get("square_m", 0.025))
        target = int(args.get("pairs", _CALIB_PAIRS_TARGET))
        name = args.get("name") or default_calib_name()
        name, name_error = sanitize_calib_name(name)
        if name is None:
            # 文件名带路径分隔符会写出 CALIB_DIR 之外（review 指出的路径
            # 穿越），拒绝整个请求而不是换个名字替它落盘。
            return {"ok": False, "reason": "bad_name", "detail": name_error}

        # 图比棋盘还小时 findChessboardCorners 内部 adaptiveThreshold 断言
        # 失败直接抛 cv2.error（镜像里 8×6 帧 + 9×6 棋盘就是这么炸的），
        # 提前按尺寸拒绝并给出能看懂的回复。
        (left, right), pair_seq = self._stereo_pair_for_calibration(instance_id)
        if left.shape[0] < board_h or left.shape[1] < board_w:
            return {"ok": False, "reason": "no_checkerboard",
                    "detail": f"帧只有 {left.shape[1]}x{left.shape[0]}，"
                              f"放不下 {board_w}x{board_h} 的棋盘格"}
        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        pattern_size = (board_w, board_h)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        try:
            found_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size, flags)
            found_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size, flags)
        except cv2.error as error:
            # 图能容纳棋盘但检测仍可能断言失败（过小/纯色帧）—— 按文档
            # 契约回 no_checkerboard，而不是让 MCP 请求抛异常。
            return {"ok": False, "reason": "no_checkerboard", "detail": str(error)}
        if not (found_l and found_r):
            return {"ok": False, "reason": "no_checkerboard",
                    "detail": f"两张图里没找到 {board_w}x{board_h} 的棋盘格。"
                              "确认左右同帧都能看到完整棋盘、光线充足、棋盘平整不反光"}
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), crit)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), crit)

        with self._cal_lock:
            session = self._cal_sessions.setdefault(
                instance_id or _DEFAULT_INSTANCE, {"pairs": [], "target": target})
            session["target"] = target
            # 同一对帧（seq 未变）不允许计为新姿态：连打 15 次 calibrate
            # 不挪棋盘会采到 15 份相同观测，stereoCalibrate 拟合出貌似
            # 成功的退化标定。等下一对帧（seq 变了）才继续收。
            last_seq = session.get("pair_seq")
            if last_seq is not None and last_seq == pair_seq:
                return {
                    "ok": False, "reason": "same_pose",
                    "pairs_used": len(session["pairs"]), "pairs_target": target,
                    "message": "还是同一对帧（没检测到新画面）。挪动棋盘格换个姿态，"
                               "确认左右相机出新帧后再调用一次",
                }
            session["pair_seq"] = pair_seq
            session["pairs"].append((corners_l, corners_r, gray_l.shape[::-1]))
            pairs = list(session["pairs"])
        if len(pairs) < target:
            return {
                "ok": True, "state": "collecting",
                "pairs_used": len(pairs), "pairs_target": target,
                "message": f"已采 {len(pairs)}/{target} 对。换一个棋盘姿态/角度再调用一次 calibrate",
            }

        objp = np.zeros((board_w * board_h, 3), dtype=np.float64)
        objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
        objp *= square
        obj_points = [objp] * len(pairs)
        img_points_l = [p[0] for p in pairs]
        img_points_r = [p[1] for p in pairs]
        size = pairs[0][2]

        # 不用 CALIB_FIX_INTRINSIC（把 K 钉死就拟合不出外参）：
        # 让 stereoCalibrate 同时拟合左右 K/D 与外参，K 用 hfov 估计做初值。
        w_img, h_img = size
        fx_seed = (w_img / 2.0) / np.tan(np.radians(90.0) / 2.0)  # 90° hfov → w/2
        K1 = np.array([[fx_seed, 0, w_img / 2], [0, fx_seed, h_img / 2], [0, 0, 1]], dtype=np.float64)
        K2 = K1.copy()
        D1 = np.zeros(4); D2 = np.zeros(4)
        rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            obj_points, img_points_l, img_points_r,
            K1, D1, K2, D2, size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5))
        # 分档只认显式声明：stereoRectify 本来就把两图都旋转到共线行，
        # rectify 之后量的 Δy 对任何标定都 ≈ 0，拿它分档必然永远落
        # rectified 档、把刚算出来的 raw blob 丢掉（review 指出的回归：
        # 运行时会跳过去畸变，SGBM 吃的是畸变图）。默认保留完整 raw
        # blob（运行时 rectify）；只有调用方明确 pre_rectified=true
        # （上游保证两路已做立体校正）才落快捷档。
        if args.get("pre_rectified"):
            # 已校正：只需要 Q 矩阵的四要素。objp 以米建，T 即米，无需 mm 换算。
            # T 是 (3,1)：float(T[0]) 撞 numpy 弃用警告，用 ravel()[0]。
            blob = {"fx": float(K1[0, 0]), "cx": float(K1[0, 2]),
                    "cy": float(K1[1, 2]),
                    "Tx": abs(float(np.asarray(T).ravel()[0])),
                    "rms": round(float(rms), 4), "pairs_used": len(pairs)}
            tier = "rectified"
        else:
            blob = {
                "K1": K1.tolist(), "D1": D1.tolist(),
                "K2": K2.tolist(), "D2": D2.tolist(),
                "R": R.tolist(), "T": T.tolist(),
                "board": {"w": board_w, "h": board_h, "square_m": square},
                "rms": round(float(rms), 4),
                "pairs_used": len(pairs),
            }
            tier = "raw"

        try:
            os.makedirs(CALIB_DIR, exist_ok=True)
            path = os.path.join(CALIB_DIR, f"{name}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(blob, fh, ensure_ascii=False, indent=2)
            saved_to = path
        except Exception as error:  # noqa: BLE001 — non-fatal
            log.warning(f"[pointcloud] could not persist calibration: {error}")
            saved_to = None

        with self._cal_lock:
            self._cal_sessions.pop(instance_id or _DEFAULT_INSTANCE, None)
        return {
            "ok": True, "state": "done", "tier": tier,
            "rms": round(float(rms), 4), "pairs_used": len(pairs),
            "calibration": blob,
            "saved_to": saved_to,
            "message": (
                f"标定完成（{tier}）。把 calibration 字段粘贴进卡片配置"
                + (f"，或直接引用文件 {saved_to}" if saved_to else "（落盘失败，请手动保存 blob）")
            ),
        }

    # ── dispatch ──────────────────────────────────────────────────────────

    def get_tools(self) -> list:
        return TOOLS

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
                }
                for key, node in nodes.items()
            }
            # agent-core 的 _dropped_inputs 按这里报告的 topic 判定：
            # 双目必须把两路都报出来，否则卡片以"忽略了一路输入"失败。
            topics_in: list = []
            if instance_id and instance_id in nodes:
                node = nodes[instance_id]
                if node._input_topic:
                    topics_in.append({"topic": node._input_topic, "format": "image/jpeg"})
                if node._input_topic_right:
                    topics_in.append({"topic": node._input_topic_right, "format": "image/jpeg"})
            else:
                for node in nodes.values():
                    if node._input_topic and all(t["topic"] != node._input_topic for t in topics_in):
                        topics_in.append({"topic": node._input_topic,
                                          "format": "image/depth-z16" if node._mode == "depth" else "image/jpeg"})
                    if node._input_topic_right and all(t["topic"] != node._input_topic_right for t in topics_in):
                        topics_in.append({"topic": node._input_topic_right, "format": "image/jpeg"})
            cloud_topic, summary_topic = output_topics_for(
                next(iter(nodes.values()))._input_topic if len(nodes) == 1 else None)
            topics_out = ([
                {"topic": cloud_topic, "format": "sensor/pointcloud"},
                {"topic": summary_topic, "format": "data/json"},
            ] if topics_in or nodes else [])
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
                instance_id, (args.get("input_topic") or
                              (args.get("input_topics") or [""])[0] or ""),
                nodes, cloud_topic, summary_topic)
            if camera_out:
                info["camera_info"] = camera_out
            if camera_note:
                info["camera_info_note"] = camera_note
            return info

        elif action == "start":
            try:
                mode, input_topic, right_topic, calibration = self._resolve_mode(args)
            except ValueError as error:
                return {"state": "error", "message": str(error)}
            if mode == "stereo":
                if not (input_topic and right_topic):
                    return {"state": "error",
                            "message": "stereo 模式需要左右两路输入（input_topics[0]=左, [1]=右）"}
                if not calibration or calibration_tier(calibration) not in ("raw", "rectified"):
                    # review 指出的死锁：calibrate 需要运行中的 stereo 节点，
                    # 而 stereo 又要求先有标定 —— 新卡片永远走不进第一次标定。
                    # 放行无标定启动：节点照常订阅配对存帧（_emit_stereo 的
                    # unknown 档不出云），calibrate 采完把 blob 填回配置重启。
                    log.info("[pointcloud] stereo start without calibration: capture-only")
                elif calibration_tier(calibration) == "rectified":
                    # review 指出：rectified 档 Tx=0 以前 start 照收，worker
                    # 里 Q_from_baseline 的 -1/Tx 每帧除零。启动时就拒绝。
                    tx = calibration.get("Tx")
                    try:
                        tx = float(tx)
                    except (TypeError, ValueError):
                        tx = None
                    if tx is None or not np.isfinite(tx) or abs(tx) < 1e-6:
                        return {"state": "error",
                                "message": f"rectified 标定的基线 Tx 无效（Tx={calibration.get('Tx')!r}）："
                                           "需要非零有限值（米），否则 Q 反投影每帧除零"}
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
            elif mode == "auto" and not input_topic:
                return {"state": "error",
                        "message": "需要一路输入：input_topic（深度图/单目）或 input_topics（双目）"}
            node_key = instance_id or input_topic or _DEFAULT_INSTANCE
            # 先记录再起节点（同 visual_depth）：loading 期间 info() 也能回答；
            # 空声明同样记录 —— 重启不再带声明时要替换旧值而不是留过期镜头。
            if input_topic:
                from plugins.camera_info import for_topic as _camera_for_topic

                with self._nodes_lock:
                    self._upstream_camera[node_key] = _camera_for_topic(
                        args.get("camera_info"), input_topic)
            with self._nodes_lock:
                running = self._nodes.get(node_key)
            if running is None:
                # mono 一定需要引擎：后台加载完再起节点。auto 首帧才定
                # depth/mono，立即起节点、首帧是 jpeg 时再懒加载（纯深度
                # 相机永远不碰 TRT）。
                needs_model = (mode == "mono" and self._model is None)
                if needs_model:
                    if self._model_loading:
                        return {"state": "loading",
                                "message": (self._model_load_status or "Engine is still loading, please wait...")}
                    if self._model_load_error:
                        return {"state": "error", "message": f"Engine failed to load: {self._model_load_error}"}

                    def _bg_start():
                        if not self._load_model_sync():
                            return  # 已有加载在跑或已失败，等它 / 由它报
                        if self._model_load_error is None:
                            self._start_node(node_key, mode, input_topic, right_topic, calibration)

                    threading.Thread(target=_bg_start, daemon=True, name="pointcloud_model_load").start()
                    return {"state": "loading", "mode": mode, "input": input_topic,
                            "message": "Engine loading in background, will start automatically"}
                self._start_node(node_key, mode, input_topic, right_topic, calibration)
                with self._nodes_lock:
                    running = self._nodes.get(node_key)
                if running is None:
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

        elif action == "config":
            cfg = {k: v for k, v in args.items()
                   if k not in ("action", "instance_id") and v is not None and v != ""}
            if instance_id:
                with self._nodes_lock:
                    self._instance_configs[instance_id] = cfg
                    running = instance_id in self._nodes
                if running:
                    self._retire_node(instance_id)
                return {"status": "configured", "instance_id": instance_id, "config": cfg}
            if "fps" in cfg:
                self._fps = int(cfg["fps"])
            if "max_points" in cfg:
                self._max_points = max(100, min(MAX_POINTS, int(cfg["max_points"])))
            if "min_depth_m" in cfg:
                self._min_depth_m = float(cfg["min_depth_m"])
            if "max_depth_m" in cfg:
                self._max_depth_m = float(cfg["max_depth_m"])
            if "calibration" in cfg:
                self._calibration = load_calibration_blob(cfg["calibration"])
            if any(k in cfg for k in ("cal_a", "cal_b")):
                self._cal_a = float(cfg.get("cal_a", self._cal_a))
                self._cal_b = float(cfg.get("cal_b", self._cal_b))
            return {"status": "configured", "config": cfg}

        elif action == "calibrate":
            try:
                return self._calibrate(args, instance_id)
            except ValueError as error:
                return {"ok": False, "reason": "bad_input", "detail": str(error)}
            except AttributeError as error:
                # fake cv2（host 测试）没实现棋盘 API；真环境不会走到
                return {"ok": False, "reason": "cv2_unavailable", "detail": str(error)}

        return None
