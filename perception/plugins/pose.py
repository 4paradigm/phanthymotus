#!/usr/bin/env python3
"""
plugins/pose.py — PosePerceptionPlugin: COCO-17 human keypoints + action labels.

Subscribes to an `image/jpeg` topic, runs a prebuilt TensorRT pose engine
through `utils.tensorrt_runtime` (letterbox and decode in
`plugins/vision_runtime.py`, exactly as vop and visual_depth do), and publishes
what each person in frame is doing. Multi-instance, one instance per input
topic.

**Three output topics, not one, and that is the design.** agent-core copies the
*whole* message of a subscribed topic into the event bus as event text
(`agent-core/src/topic_subscriber.py`), so every byte published on the topic
wired to `decision_core` is a byte of LLM context on every frame. 17 keypoints x
3 floats x N people at 5 fps does not fit in that budget, and the skeleton is
useless to a text model anyway — but it is exactly what the dashboard wants to
draw. So the fat payload goes somewhere the LLM does not subscribe:

    {input}/poses                data/json      lean: action + bbox + centre
    {input}/poses/skeleton       sensor/pose2d   full keypoints, for the renderer
    {input}/poses/overlay_img    image/jpeg      skeleton drawn on the frame (opt-in)

A dashboard renderer only ever sees one topic — `detail-panel.js` opens a single
`/ws/bus/{topic}` for the selected port — so an overlay cannot be assembled in
the browser from two of them. Hence the third topic, which costs a draw and a
JPEG encode per frame and is therefore off by default.

The engine loads lazily on the first `start`, like visual_depth: on an 8 GB Orin
already running vop, depth, OCR, ASR and TTS, memory rather than GPU time is the
binding constraint, so an enabled-but-unwired card must cost nothing.

Action labels come from `plugins/pose_action.py` — pure numpy, no second model
and no second inference runtime. That is not a shortcut: a second ONNX Runtime
in this process shares one provider bridge with the first and SIGSEGVs the whole
of perception on jp5.11 (see plugins/kokoro_worker.py).
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
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from utils.ros_lifecycle import dispose_node

from plugins.image_input import BadInput, load_image_bytes
from plugins.pose_action import (
    ACTION_LABELS_ZH,
    DEFAULT_THRESHOLDS,
    POSTURE_LABELS_ZH,
    TEMPORAL_ACTIVITIES,
    PoseFrame,
    PoseTracker,
    action_catalogue,
)
from plugins.pose_stgcn import (
    BACKENDS,
    DEFAULT_MIN_SCORE,
    DEFAULT_WINDOW_S,
    action_vocabulary,
    build_backend,
)
from plugins.vision_runtime import COCO_KEYPOINTS, COCO_SKELETON, N_KEYPOINTS

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

# Where a topic-less card publishes. Fixed, because with no input topic there is
# nothing to derive it from — same rule as vop's /perception/vop.
DEFAULT_OUTPUT_TOPIC = "/perception/pose"

# Instance key for the topic-less card, mirroring vop's and tts's.
_DEFAULT_INSTANCE = "_default"

# `s` rather than `n`, chosen deliberately: keypoint precision is not a cosmetic
# property here, it is the input to the action rules. A noisy wrist breaks the
# reversal count that distinguishes waving from reaching, and a noisy hip breaks
# the drop ratio that distinguishes a fall from lying down — so the 57.2 → 63.0
# mAP(pose) step buys fewer misjudged actions, not prettier skeletons. Costs
# ~12 MB more resident and ~8 ms more per frame than `n` (see README).
DEFAULT_MODEL = "yolo26s-pose"

KEYPOINT_LEVELS = ("off", "compact", "full")

# What the card offers. `stgcn` is deliberately NOT here, although
# `build_backend` can still construct it for tests and deliberate experiments.
#
# Offering it would be offering a foot-gun: alone it has no `standing` and no
# `sitting` class, because NTU-60 is built from actions and a motionless person
# is not one — so every stationary person comes back `unknown`. Worse, a static
# clip does not make it abstain: fed 100 identical frames of a real person lying
# on pavement it returns NTU's "play with phone/tablet" at **0.997** with an
# entropy of 0.03. It is not unsure, it is confidently wrong, because "no action"
# is not an answer the label space contains.
#
# Same call as vop's `classes` config, which was removed rather than left
# available-but-broken.
ACTION_BACKENDS = ("hybrid", "rules")

# Cards saved before `stgcn` was withdrawn. Migrated rather than refused, the
# way plugins/asr.py migrates the removed `kws` trigger mode: a deployed card
# must keep working after an upgrade, and `hybrid` is what its owner wanted
# anyway — the learned labels, plus the postures that backend cannot produce.
WITHDRAWN_BACKENDS = {"stgcn": "hybrid"}

# Below this, a temporal backend is being starved. ST-GCN++ classifies a clip,
# and at 5 fps a 2.5 s window is 13 real frames resampled up to the engine's 48
# — mostly interpolation. The card does not refuse (a thin window still beats
# no actions at all) but it says so in `info`, because a quietly starved model
# looks like a wrong model.
MIN_FPS_FOR_TEMPORAL_BACKEND = 12


def output_topic_for(input_topic: Optional[str]) -> str:
    """The one place the lean output topic is derived from the input.

    vop learnt this the hard way: built in three places, and the third forgot
    the topic-less case and reported publishing to "None/objects".
    """
    return f"{input_topic}/poses" if input_topic else DEFAULT_OUTPUT_TOPIC


def skeleton_topic_for(input_topic: Optional[str]) -> str:
    return f"{output_topic_for(input_topic)}/skeleton"


def overlay_topic_for(input_topic: Optional[str]) -> str:
    return f"{output_topic_for(input_topic)}/overlay_img"


def _keypoint_level(value) -> str:
    text = str(value or "").strip().lower()
    return text if text in KEYPOINT_LEVELS else "off"


def _compact_keypoints(keypoints: np.ndarray) -> list:
    """[x, y] rounded to whole pixels, for a consumer that wants the pose but
    not three decimal places of it."""
    return [[int(round(float(x))), int(round(float(y)))]
            for x, y, _v in keypoints]


def _full_keypoints(keypoints: np.ndarray) -> list:
    return [[round(float(x), 1), round(float(y), 1), round(float(v), 2)]
            for x, y, v in keypoints]


TOOLS = [
    {
        "name": "pose",
        "type": "processor",
        "multiInstance": True,
        "description": (
            "Human pose and action — COCO-17 keypoints per person plus what they "
            "are doing (standing / sitting / waving / walking / fallen / ...). "
            "Call list_actions for the full label set and what each one means."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "start", "stop", "info", "config",
                        "recognize_by_photo", "recognize_by_url",
                        "list_actions",
                    ],
                    "description": "Action to perform",
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 image topic to subscribe (e.g. /hostname/camera/rgb). 可选：不填则卡片以按需模式启动，不订阅摄像头，只服务 recognize_by_photo / recognize_by_url",
                },
                # Same upload mechanism as vop/face: `uploadTo: mcp` posts the
                # file to this service and hands back a path *this* container can
                # open, so no shared mount is needed.
                "image_path": {"type": "string", "format": "file", "accept": "image/*", "uploadTo": "mcp", "description": "图片文件。从卡片上传，或填一个容器可读的路径（如 /uploads/scene.jpg）。常见格式都支持，过大的图会本地缩放"},
                "url": {"type": "string", "description": "图片的 http(s) 地址。下载后本地解码，格式限制同 image_path"},
                "confidence": {"type": "number", "description": "本次识别的人体检测置信度阈值（0-1）。不填则用卡片配置的值"},
            },
            "required": ["action"],
            "x-action-params": {
                "start": {"params": ["input_topic"], "description": "启动。给 input_topic 则持续检测该摄像头话题；不给则以按需模式启动，只服务单张图片"},
                "stop": {"params": [], "description": "Stop detection"},
                "info": {"params": ["input_topic"], "description": "Report state, topics and the action label set"},
                "config": {"params": [], "description": "Update confidence / fps / payload detail / action thresholds"},
                "recognize_by_photo": {
                    "params": ["image_path", "confidence"],
                    "description": "看一张图片里的人在做什么 —— 一次性识别，不需要摄像头也不需要先 start。返回每个人的关键点、画面中的相对位置与姿态标签。注意：挥手、走动、跌倒这类需要时间的动作单张图片判不出来，回复里会列出来",
                },
                "recognize_by_url": {
                    "params": ["url", "confidence"],
                    "description": "同 recognize_by_photo，只是图片来自 http(s)",
                },
                "list_actions": {
                    "params": [],
                    "description": "列出全部动作标签、中文名，以及哪些是「事件」（跌倒）而不是「姿态」。先查这个就知道某个动作问不问得出来",
                },
            },
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "人体检测置信度阈值", "default": 0.4, "scope": "instance"},
                "fps": {"type": "integer", "minimum": 1, "description": "每秒最多推理几帧。12 而不是 5：几何规则逐帧就能判，但骨架动作模型判的是一段视频 —— 2.5s 窗口在 5 fps 下只有 13 帧真实数据，要重采样到 engine 的 48 帧，大部分是插值。低于 12 时 info 会给出 action_fps_note", "default": 12, "scope": "instance"},
                "kpt_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "单个关键点的可见性阈值。低于此值的关节既不参与动作判定也不绘制 —— 比把它当成 (0,0) 画出来强", "default": 0.3, "scope": "instance"},
                "max_persons": {"type": "integer", "minimum": 1, "description": "单帧最多处理几个人（按检测置信度取前 N 个）", "default": 5, "scope": "instance"},
                # Governs the LEAN topic only. The skeleton topic always carries
                # full keypoints — it is not in the LLM's context, so there is
                # nothing to save there.
                "publish_keypoints": {"type": "string", "enum": list(KEYPOINT_LEVELS), "description": "data/json 那条流里要不要带关键点：off（默认）只发动作和位置；compact 带 17 个整数像素点；full 带 17×(x,y,可见性)。off→full 每人每帧约 60 B → 900 B，而这条流的每个字节都是每帧的 LLM 上下文字节。画骨架用的是 /skeleton 那条，不受这里影响", "default": "off", "scope": "instance"},
                "publish_bbox": {"type": "boolean", "description": "lean 流里带上像素框 [x1,y1,x2,y2]", "default": True, "scope": "instance"},
                "publish_overlay": {"type": "boolean", "description": "另发一条把骨架画在原始画面上的 JPEG（{topic}/poses/overlay_img）。每帧多一次绘制+编码，外加一条跑 JPEG 的话题，所以默认关闭；要录给人看时再开", "default": False, "scope": "instance"},
                "action_window_s": {"type": "number", "minimum": 0.2, "description": "动作判定回看多少秒。挥手频率和步频都是在这个窗口里数出来的", "default": 1.5, "scope": "instance"},
                "action_backend": {"type": "string", "enum": list(ACTION_BACKENDS), "description": "动作分类后端。hybrid（默认）= 姿态走几何规则、跌倒/挥手/指向走 ST-GCN++ 骨架动作模型，两者各做擅长的；rules = 只用几何规则，不加载第二个 engine —— 这也是取不到 engine 时自动退到的模式，手动选它主要用于在真机上区分「模型判错」和「几何判错」。只用模型的 stgcn 模式已撤下：NTU-60 里没有「站立」「坐」这两个状态类，静止的人不但报不出来，还会拿到一个自信的错答案", "default": "hybrid", "scope": "instance"},
                "activity_interval_s": {"type": "number", "minimum": 0.0, "description": "How often the action model runs, in seconds. Its window is 2.5 s, so two runs one frame apart share 97% of their input and cost 20 ms each; with three people in frame, running it every frame measured 98.9 ms per frame against 41 ms throttled. The geometry still runs every frame, so posture stays frame-rate. 0 disables the throttle.", "default": 0.35, "scope": "instance"},
                "label_hold": {"type": "integer", "minimum": 1, "description": "标签迟滞：新动作要连续赢多少帧才换。每个阈值都是悬崖，实测在边界上原始答案会逐帧翻（模型得分在 0.40 附近摆动时 9 次比较全翻）。一个每秒跳十几次的标签比一个稳定的错标签更糟 —— 下游没法用、人读不了。代价是每次真实变化也要晚这么多帧（12 fps 下 3 帧 = 250 ms）。原始答案在 evidence.raw_action 里", "default": 3, "scope": "instance"},
                "action_min_score": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "骨架动作模型的得分阈值，低于它不报。跌倒单独用更高的 0.75 —— 实测该 engine 对纯噪声会给出 A43「跌倒」0.62，而误报跌倒的代价是机器人丢下手上的事去问人有没有受伤。调之前先看 info 里的实际得分，那是区分「模型判错」和「阈值定错」的唯一办法", "default": DEFAULT_MIN_SCORE, "scope": "instance"},
                # Exposed because they are NOT constants: the same fall measures
                # differently depending on where the camera is mounted.
                "fall_drop_ratio": {"type": "number", "minimum": 0.05, "maximum": 1.0, "description": "判定跌倒所需的髋部下降幅度，按站立身高的比例。和机位强相关 —— 相机离地 0.4 m 和 1.2 m 量同一次跌倒得到的数不一样，务必在真机上调", "default": DEFAULT_THRESHOLDS["fall_drop_ratio"], "scope": "instance"},
                "fall_drop_window_s": {"type": "number", "minimum": 0.1, "description": "上面那个下降必须在多少秒内完成。慢慢躺下不算跌倒", "default": DEFAULT_THRESHOLDS["fall_drop_window_s"], "scope": "instance"},
                "fall_settle_s": {"type": "number", "minimum": 0.2, "description": "落地后保持水平多久才报跌倒。弯腰捡东西也是短暂水平的", "default": DEFAULT_THRESHOLDS["fall_settle_s"], "scope": "instance"},
            },
        },
        "topic_in": [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [
            {"format": "data/json", "desc": "per-person action labels and positions"},
            {"format": "sensor/pose2d", "desc": "COCO-17 keypoints for the skeleton renderer"},
            {"format": "image/jpeg", "desc": "skeleton drawn on the frame (publish_overlay)"},
        ],
    }
]


# ── ROS2 Node (one per instance/topic) ────────────────────────────────────────

class _PoseNode(Node):
    """Per-topic pose inference node."""

    def __init__(self, input_topic: Optional[str], model, confidence: float,
                 fps: float, node_suffix: str, *, classifier: PoseActionClassifier,
                 kpt_confidence: float = 0.3, max_persons: int = 5,
                 publish_keypoints: str = "off", publish_bbox: bool = True,
                 publish_overlay: bool = False, label_hold: int = 3,
                 activity_interval_s: float = 0.35):
        super().__init__(f"pose_{node_suffix}" if node_suffix else "pose")
        # Topic-less is a supported mode, as in vop and tts: a card driven only
        # by recognize_by_photo has no camera, but still wants somewhere to
        # publish so the canvas shows the flow.
        self._input_topic = input_topic or ''
        self._output_topic = output_topic_for(input_topic)
        self._skeleton_topic = skeleton_topic_for(input_topic)
        self._overlay_topic = overlay_topic_for(input_topic)
        self._model = model
        self._classifier = classifier
        self._confidence = confidence
        self._kpt_confidence = kpt_confidence
        self._max_persons = max(1, int(max_persons))
        self._fps = fps
        self._publish_keypoints = _keypoint_level(publish_keypoints)
        self._publish_bbox = bool(publish_bbox)
        self._publish_overlay = bool(publish_overlay)
        self._frame_interval = 1.0 / max(fps, 0.1)

        self._activity_interval_s = float(activity_interval_s)
        # Most recent raw model output, so `info` can show what it actually
        # scored. "The activity field is empty" is otherwise impossible to tell
        # from "the model was never asked" or "nothing cleared the threshold".
        self._last_prediction: dict = {}
        self._tracker = PoseTracker(history_s=classifier.history_s,
                                   min_conf=kpt_confidence,
                                   label_hold=label_hold)

        self._pub = self.create_publisher(String, self._output_topic, _PUB_QOS)
        self._pub_skeleton = self.create_publisher(String, self._skeleton_topic,
                                                   _PUB_QOS)
        # Created only when asked for: an idle publisher is a port the canvas
        # shows as wired and silent, and `publish_overlay` already says why.
        self._pub_overlay = (
            self.create_publisher(CompressedImage, self._overlay_topic, _PUB_QOS)
            if self._publish_overlay else None)

        self._sub: Optional[object] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._last_inference_time = 0.0
        self._detect_count = 0
        self._person_count = 0
        self._last_actions: dict = {}
        self._running = False
        # Serializes start/stop on this node. dispatch() runs on a
        # ThreadingHTTPServer thread and the canvas routinely does
        # config→start→stop→start within seconds; without this two starts can
        # both pass the running check and the second orphans the first's worker.
        # perception/README.md § "Plugin Concurrency".
        self._lifecycle_lock = threading.RLock()

    def request_stop(self) -> None:
        """Signal the worker to wind down without taking the lifecycle lock.

        stop() must be able to cancel a start() that is still in progress, so
        the cancellation flag has to be settable while that start holds the lock.
        """
        self._stop_event.set()

    def _status(self) -> dict:
        return {
            "state": "running" if self._running else "idle",
            "input": self._input_topic,
            "output": self._output_topic,
            "skeleton_output": self._skeleton_topic,
            "overlay_output": self._overlay_topic if self._publish_overlay else None,
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
                self._worker = threading.Thread(
                    target=self._inference_worker, daemon=True,
                    name=f"pose_worker_{self._input_topic}")
                self._worker.start()
            # Without a topic there is nothing to subscribe to and no frames to
            # consume, so no worker; the node exists to own the publishers that
            # one-shot results go out on.
            self._running = True
            log.info(f"[pose] started: {self._input_topic or '(no topic, on-demand)'} "
                     f"→ {self._output_topic}")
            return self._status()

    def stop(self) -> dict:
        """Stop this node's worker. Does NOT touch the subscription.

        Destroying a subscription while the node is still registered with the
        executor races the executor's wait list and kills the spin thread with
        `InvalidHandle`, which takes every subscription in the process with it,
        silently. Teardown is dispose_node()'s job, after removal from the
        executor. Same order as vop and ocr.
        """
        # Flag first, lock second, so a start() holding the lock sees the flag
        # as soon as it releases rather than this call queueing behind it.
        self._stop_event.set()
        with self._lifecycle_lock:
            if self._worker and self._worker.is_alive():
                self._worker.join(timeout=3.0)
            self._worker = None
            self._running = False
            self._tracker.reset()
            log.info(f"[pose] stopped: {self._input_topic or '(no topic)'}")
            return self._status()

    def _image_cb(self, msg: CompressedImage):
        now = time.monotonic()
        if now - self._last_inference_time < self._frame_interval:
            return
        self._last_inference_time = now
        # Drop the old frame if the queue is full (no backpressure).
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
        from plugins.vision_runtime import decode_poses

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
                boxes, scores, keypoints = decode_poses(
                    outputs, meta, self._confidence
                )
                persons = self._describe(boxes, scores, keypoints, frame, started)
                self._publish(persons, frame, started)
            except Exception as e:
                log.error(f"[pose] inference error: {e}", exc_info=True)

    # ── payload construction ─────────────────────────────────────────────

    def _describe(self, boxes, scores, keypoints, frame, now_wall: float) -> list:
        """Associate, classify, and build one record per person.

        The tracker is fed in confidence order and capped at `max_persons`, so
        the people dropped on a crowded frame are the least certain ones rather
        than whichever happened to be decoded first.
        """
        order = np.argsort(np.asarray(scores, dtype=np.float32))[::-1]
        order = order[:self._max_persons]
        boxes = [boxes[i] for i in order]
        kept_scores = [float(scores[i]) for i in order]
        kept_keypoints = [keypoints[i] for i in order]

        height, width = frame.shape[:2] if hasattr(frame, "shape") else (0, 0)
        # The tracker's clock is the frame's, not the wall's: every temporal
        # threshold in pose_action is in seconds of video. The frame size goes
        # with it because a learned backend normalises the skeleton by the
        # frame, and cannot derive that from the keypoints.
        tracks = self._tracker.update(boxes, kept_keypoints, now_wall,
                                      image_size=(width, height))
        half_w, half_h = max(width / 2.0, 1.0), max(height / 2.0, 1.0)

        persons = []
        for track, box, score, kpts in zip(tracks, boxes, kept_scores,
                                           kept_keypoints):
            # The learned backend is throttled, the geometry is not. Its
            # window is 2.5 s, so two runs one frame apart share 97% of their
            # input and cost 20 ms each; measured on Orin 6 with three people,
            # re-running it every frame put the card at 98.9 ms per frame —
            # a 10 fps ceiling on a 12 fps stream. The geometry beside it costs
            # 0.68 ms and does run every frame, so posture stays frame-rate.
            #
            # Staggered by track id so two people do not both pay on the same
            # frame, which would show up as a periodic stutter rather than as a
            # higher average.
            due = (now_wall - track.activity_t) >= self._activity_interval_s
            if self._activity_interval_s > 0 and not due:
                stagger = (track.id % 3) * self._activity_interval_s / 3.0
                due = (now_wall - track.activity_t) >= (
                    self._activity_interval_s + stagger)
            verdict = dict(self._classifier.classify(
                list(track.history), want_activity=due))
            if due:
                track.activity = verdict.get("activity")
                track.activity_t = now_wall
                self._last_prediction = verdict.get("evidence") or {}
            elif track.activity is not None:
                verdict["activity"] = track.activity
            # Hysteresis, per person, per channel. Every threshold here is a
            # cliff and at the boundaries the raw answer flips on every frame —
            # a label alternating twelve times a second is worse than a stable
            # wrong one, because nothing downstream can act on it. The two
            # channels are stabilised independently: a posture settling must not
            # hold back an activity, or vice versa.
            posture, pending = track.posture_stabiliser.update(
                verdict.get("posture") or "unknown")
            if posture != (verdict.get("posture") or "unknown"):
                verdict["evidence"] = {**(verdict.get("evidence") or {}),
                                       "raw_posture": verdict.get("posture")}
                verdict["posture"] = None if posture == "unknown" else posture
                verdict["posture_confidence"] = 0.0
            if pending:
                verdict["evidence"] = {**(verdict.get("evidence") or {}),
                                       "pending_posture": pending}

            activity = verdict.get("activity")
            name, _ = track.activity_stabiliser.update(
                activity["name"] if activity else "none")
            if activity and name != activity["name"]:
                verdict["evidence"] = {**(verdict.get("evidence") or {}),
                                       "raw_activity": activity["name"]}
                verdict["activity"] = None
            elif not activity and name != "none":
                # The stabiliser is still holding the previous activity; do not
                # resurrect it with a stale score, just say nothing this frame.
                pass

            cx = (float(box[0]) + float(box[2])) / 2.0
            cy = (float(box[1]) + float(box[3])) / 2.0
            persons.append({
                "track": track,
                "id": track.id,
                "score": round(score, 2),
                "box": [float(v) for v in box],
                "keypoints": kpts,
                "position": [round((cx - half_w) / half_w, 3),
                             round((cy - half_h) / half_h, 3)],
                "verdict": verdict,
            })
        self._last_actions = {
            p["id"]: (p["verdict"].get("activity") or {}).get("name")
                     or p["verdict"].get("posture") or "unknown"
            for p in persons
        }
        return persons

    def _lean_record(self, person: dict) -> dict:
        """Two channels, because two different questions are answered.

        `posture` is the shape the body is in — a state, readable from one
        frame, and everybody has one. `activity` is what they are doing — a
        process, needs motion, and may legitimately be absent. Neither is
        derived from the other and there is no third field flattening them: a
        single `action` picked by priority across both put `standing` and
        `reading` in one slot whose vocabulary was the union of everything, and
        a consumer could not rely on it coming from a known set.
        """
        verdict = person["verdict"]
        record = {
            "id": person["id"],
            "position": person["position"],
            "posture": verdict.get("posture"),
            "posture_confidence": verdict.get("posture_confidence", 0.0),
        }
        activity = verdict.get("activity")
        if activity:
            record["activity"] = activity["name"]
            record["activity_zh"] = activity["name_zh"]
            record["activity_score"] = activity["score"]
        if verdict.get("point_direction"):
            record["point_direction"] = verdict["point_direction"]
        # Evidence rides along only for the one activity a robot acts on:
        # "they fell" without the numbers behind it is not something an operator
        # can check.
        if activity and activity["name"] == "falling down":
            record["evidence"] = verdict.get("evidence", {})
        if self._publish_bbox:
            record["bbox"] = [round(v, 1) for v in person["box"]]
        if self._publish_keypoints == "compact":
            record["keypoints"] = _compact_keypoints(person["keypoints"])
        elif self._publish_keypoints == "full":
            record["keypoints"] = _full_keypoints(person["keypoints"])
        return record

    def _skeleton_record(self, person: dict) -> dict:
        verdict = person["verdict"]
        return {
            "id": person["id"],
            "score": person["score"],
            "bbox": [round(v, 1) for v in person["box"]],
            "posture": verdict.get("posture"),
            "activity": (verdict["activity"]["name"]
                         if verdict.get("activity") else None),
            "keypoints": _full_keypoints(person["keypoints"]),
        }

    def publish_persons(self, persons: list, frame=None,
                        started: Optional[float] = None) -> None:
        """Publish one frame's worth of results. Used by the stream worker and
        by the one-shot photo actions, so both emit the same thing."""
        self._publish(persons, frame, started)

    def _publish(self, persons: list, frame=None,
                 started: Optional[float] = None) -> None:
        self._detect_count += 1
        self._person_count = len(persons)
        payload = {
            "timestamp": time.time(),
            "count": len(persons),
            "persons": [self._lean_record(p) for p in persons],
        }
        if started is not None:
            # Measured from the start of processing this frame, not its arrival:
            # queue wait is a function of the fps cap, not of how long inference
            # takes, and mixing them makes the number unreadable. Same as vop.
            payload["latency_ms"] = int((time.time() - started) * 1000)
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False)
        self._pub.publish(message)

        height, width = ((frame.shape[0], frame.shape[1])
                         if frame is not None and hasattr(frame, "shape")
                         else (0, 0))
        skeleton = {
            "timestamp": payload["timestamp"],
            "count": len(persons),
            # The renderer has no other way to know what the coordinates are
            # relative to — keypoints are in source-frame pixels.
            "image_size": [int(width), int(height)],
            "keypoint_names": list(COCO_KEYPOINTS),
            "skeleton": [list(edge) for edge in COCO_SKELETON],
            "persons": [self._skeleton_record(p) for p in persons],
        }
        skeleton_message = String()
        skeleton_message.data = json.dumps(skeleton, ensure_ascii=False)
        self._pub_skeleton.publish(skeleton_message)

        if self._pub_overlay is not None and frame is not None:
            self._publish_overlay_frame(persons, frame)

    def _publish_overlay_frame(self, persons: list, frame) -> None:
        try:
            import cv2

            canvas = frame.copy()
            draw_skeleton(canvas, persons, self._kpt_confidence)
            ok, buffer = cv2.imencode(".jpg", canvas,
                                      [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                return
            message = CompressedImage()
            message.header.stamp = self.get_clock().now().to_msg()
            message.format = "jpeg"
            message.data = buffer.tobytes()
            self._pub_overlay.publish(message)
        except Exception as error:  # noqa: BLE001 — never lose the real result
            log.warning(f"[pose] overlay publish failed: {error}")


# Colour per track id, so two people keep two colours between frames. BGR,
# because this is drawn with cv2.
_TRACK_COLOURS = (
    (255, 176, 77), (119, 215, 125), (120, 120, 240), (220, 190, 110),
    (200, 130, 220), (110, 210, 220),
)
_ALERT_COLOUR = (87, 119, 215)      # var(--orange) in BGR

#: Labels drawn in the alert colour. Same set the web renderer uses, and the
#: same reason: somebody on the floor is worth looking at whether or not
#: anything called it a fall.
_ALERT_LABELS = ("falling down", "lying")


def is_alerting(verdict: dict) -> bool:
    """Is this person on the ground?

    Decided from the channels, never from `overlay_label`'s output: that is a
    compound string once both channels are present ("lying · falling down"), so
    testing it for membership in a label set silently stops matching.
    """
    activity = verdict.get("activity")
    if isinstance(activity, dict):
        activity = activity.get("name")
    return verdict.get("posture") in _ALERT_LABELS or activity in _ALERT_LABELS


def overlay_label(verdict: dict) -> str:
    """The one line drawn over a person.

    Both channels when both are there — `upright · hand waving` — because they
    answer different questions and showing only one hides the other. Posture
    first: it is the thing that is always available, so the label does not
    change shape when an activity comes and goes.

    This read `verdict["action"]` until that field was removed, and then
    silently labelled everybody `unknown`: the drawing tests passed because they
    fed a hand-built verdict that still had the old key, which is exactly how a
    renderer keeps drawing after its data has moved.
    """
    activity = verdict.get("activity")
    if isinstance(activity, dict):
        activity = activity.get("name")
    posture = verdict.get("posture")
    if posture and activity:
        return f"{posture} · {activity}"
    return posture or activity or "unknown"


def draw_skeleton(canvas, persons: list, kpt_confidence: float) -> None:
    """Draw bones, joints and the action label onto a BGR frame, in place.

    A joint below `kpt_confidence` is skipped rather than drawn: the engine
    emits a coordinate for every one of the 17 whether it saw it or not, so
    drawing them all puts a limb through whatever (0, 0) happens to be.
    """
    import cv2

    for person in persons:
        keypoints = np.asarray(person["keypoints"], dtype=np.float32)
        verdict = person.get("verdict") or {}
        label = overlay_label(verdict)
        colour = (_ALERT_COLOUR if is_alerting(verdict)
                  else _TRACK_COLOURS[int(person.get("id", 0)) % len(_TRACK_COLOURS)])
        visible = keypoints[:, 2] >= kpt_confidence
        for a, b in COCO_SKELETON:
            if a < len(visible) and b < len(visible) and visible[a] and visible[b]:
                cv2.line(canvas,
                         (int(keypoints[a, 0]), int(keypoints[a, 1])),
                         (int(keypoints[b, 0]), int(keypoints[b, 1])),
                         colour, 2, cv2.LINE_AA)
        for index in range(min(len(visible), N_KEYPOINTS)):
            if visible[index]:
                cv2.circle(canvas, (int(keypoints[index, 0]),
                                    int(keypoints[index, 1])), 3, colour, -1,
                           cv2.LINE_AA)
        box = person.get("box") or person.get("bbox")
        if box:
            cv2.putText(canvas, f"#{person.get('id', '?')} {label}",
                        (int(box[0]), max(int(box[1]) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


# ── Plugin class ──────────────────────────────────────────────────────────────

# Config keys that belong to the action rules rather than to the detector.
_THRESHOLD_KEYS = ("kpt_confidence", "fall_drop_ratio", "fall_drop_window_s",
                   "fall_settle_s")


class PosePerceptionPlugin:
    PREFIX = "pose"

    def __init__(self, plugin_cfg: dict, namespace: str, executor):
        self._namespace = namespace
        self._executor = executor
        # Kept whole for the image-source helpers, which read image_roots and
        # max_image_bytes straight from it (plugins/image_input.py).
        self._plugin_cfg = dict(plugin_cfg or {})
        self._confidence = float(plugin_cfg.get("confidence", 0.4))
        self._fps = int(plugin_cfg.get("fps", 12))
        self._kpt_confidence = float(
            plugin_cfg.get("kpt_confidence", DEFAULT_THRESHOLDS["kpt_confidence"]))
        self._max_persons = int(plugin_cfg.get("max_persons", 5))
        self._publish_keypoints = _keypoint_level(plugin_cfg.get("publish_keypoints"))
        self._publish_bbox = bool(plugin_cfg.get("publish_bbox", True))
        self._publish_overlay = bool(plugin_cfg.get("publish_overlay", False))
        self._action_window_s = float(plugin_cfg.get("action_window_s", 1.5))
        self._backend_migrated: Optional[str] = None
        self._action_backend = self._migrate_backend(
            str(plugin_cfg.get("action_backend", "hybrid")))
        self._action_min_score = float(
            plugin_cfg.get("action_min_score", DEFAULT_MIN_SCORE))
        self._label_hold = int(plugin_cfg.get("label_hold", 3))
        self._activity_interval_s = float(
            plugin_cfg.get("activity_interval_s", 0.35))
        # Why a temporal backend may be unavailable, kept so `info` can say it
        # instead of the card looking like it chose the geometry on purpose.
        self._backend_fallback: Optional[str] = None
        self._model_name = str(plugin_cfg.get("model", DEFAULT_MODEL) or DEFAULT_MODEL)

        self._model = None  # lazy: see the module docstring on memory
        self._model_loading = False
        self._model_load_error = None
        # The downloader's progress line while the engine bundle is fetched, None
        # once it is in. A cold fetch runs for minutes, and a card with no
        # progress is indistinguishable from a hung one.
        self._model_load_status = None
        self._model_lock = threading.Lock()

        self._nodes: dict[str, _PoseNode] = {}
        self._instance_configs: dict[str, dict] = {}
        # What the camera feeding each instance said about its optics. `position`
        # here is a normalised lateral offset, so without a field of view it is
        # dimensionless to everyone downstream; passing the declaration on is
        # what lets a consumer turn it into an angle. Guarded by _nodes_lock.
        self._upstream_camera: dict[str, dict] = {}
        # Guards _nodes, _instance_configs and _upstream_camera. Every dispatch()
        # runs on its own ThreadingHTTPServer thread, so an unguarded
        # read-modify-write of _nodes can leave a started node unreachable —
        # perception/README.md § "Plugin Concurrency". Never held across
        # node.start()/stop() or a model load.
        self._nodes_lock = threading.RLock()

    # ── config → objects ─────────────────────────────────────────────────

    def _merged_config(self, node_key: str) -> dict:
        icfg = self._instance_configs.get(node_key, {})
        merged = {
            "confidence": float(icfg.get("confidence", self._confidence)),
            "fps": int(icfg.get("fps", self._fps)),
            "kpt_confidence": float(icfg.get("kpt_confidence", self._kpt_confidence)),
            "max_persons": int(icfg.get("max_persons", self._max_persons)),
            "publish_keypoints": (_keypoint_level(icfg["publish_keypoints"])
                                  if "publish_keypoints" in icfg
                                  else self._publish_keypoints),
            "publish_bbox": bool(icfg.get("publish_bbox", self._publish_bbox)),
            "publish_overlay": bool(icfg.get("publish_overlay", self._publish_overlay)),
            "action_window_s": float(icfg.get("action_window_s",
                                              self._action_window_s)),
            "action_backend": self._migrate_backend(
                str(icfg.get("action_backend", self._action_backend))),
            "action_min_score": float(icfg.get("action_min_score",
                                               self._action_min_score)),
            "label_hold": int(icfg.get("label_hold", self._label_hold)),
            "activity_interval_s": float(icfg.get(
                "activity_interval_s", self._activity_interval_s)),
        }
        for key in _THRESHOLD_KEYS:
            if key in icfg and icfg[key] is not None:
                merged[key] = icfg[key]
            elif key in self._plugin_cfg and self._plugin_cfg[key] is not None:
                merged[key] = self._plugin_cfg[key]
        return merged

    def _classifier_for(self, merged: dict):
        """Build the configured action backend.

        Falls back to `rules` when a temporal backend cannot be had — there is
        no published action engine yet, and a robot with no route to COS is the
        other case. The fallback is *recorded* rather than silent: a card
        running geometry while its config says `hybrid` is exactly the kind of
        divergence that gets diagnosed as "the model is bad".
        """
        thresholds = {key: merged[key] for key in _THRESHOLD_KEYS if key in merged}
        thresholds.setdefault("kpt_confidence", merged["kpt_confidence"])
        name = merged["action_backend"]
        self._backend_fallback = None
        if name == "rules":
            return build_backend("rules", thresholds=thresholds,
                                 action_window_s=merged["action_window_s"])
        try:
            return build_backend(
                name,
                thresholds=thresholds,
                action_window_s=merged["action_window_s"],
                window_s=max(merged["action_window_s"], DEFAULT_WINDOW_S),
                min_score=merged["action_min_score"],
            )
        except Exception as error:  # noqa: BLE001 — the card must still work
            self._backend_fallback = (
                f"action_backend={name!r} unavailable ({error}); "
                f"running the geometry rules instead")
            log.warning("[pose] %s", self._backend_fallback)
            return build_backend("rules", thresholds=thresholds,
                                 action_window_s=merged["action_window_s"])

    # ── engine ───────────────────────────────────────────────────────────

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return

            # Fix the broken system cv2 on some Jetson BSPs (circular import in
            # mat_wrapper) and stub imshow for headless containers. ultralytics
            # is not in this path at all — the engine runs through
            # utils.tensorrt_runtime — but the letterbox and decode use cv2.
            # Lifted from plugins/vop.py; same images, same breakage.
            try:
                import cv2
                _ = cv2.IMREAD_COLOR
            except (ImportError, AttributeError):
                import importlib.util
                import sys as _sys
                import glob as _glob
                candidates = _glob.glob(
                    "/usr/lib/python*/dist-packages/cv2/python-*/cv2.cpython-*.so")
                if candidates:
                    spec = importlib.util.spec_from_file_location("cv2", candidates[0])
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    _sys.modules["cv2"] = module
                    import cv2
                else:
                    import cv2  # let it fail naturally

            if not hasattr(cv2, 'imshow'):
                cv2.imshow = lambda *a, **k: None
                cv2.waitKey = lambda *a, **k: 0
                cv2.destroyAllWindows = lambda *a, **k: None

            from plugins.vision_runtime import VisionEngineSession

            engine_path = self._resolve_engine()
            log.info(f"[pose] loading engine: {engine_path}")
            self._model = VisionEngineSession(engine_path)
            log.info(f"[pose] engine loaded: {self._model_name} "
                     f"input={self._model.input_size}, "
                     f"{N_KEYPOINTS} keypoints")

    def _resolve_engine(self) -> str:
        """Path to the pose engine for this machine's TensorRT.

        A path in config wins, so a dev box can point at a locally built engine;
        otherwise the pinned bundle is fetched. There is no `.pt` fallback —
        dropping back to PyTorch silently would cost ~8x per frame and look like
        nothing was wrong (measured for vop on an Orin NX).
        """
        configured = (self._model_name or "").strip()
        if configured.endswith(".engine") and os.path.isfile(configured):
            return configured

        from utils.model_downloader import ensure_pose_model
        from utils.model_progress import fetch_status

        model_dir = os.environ.get("POSE_MODEL_DIR", "/models/pose")
        progress_cb, _ = fetch_status(
            lambda text: setattr(self, "_model_load_status", text), DEFAULT_MODEL)
        paths = ensure_pose_model(model_dir, progress_cb=progress_cb)
        return next(p for name, p in paths.items() if name.endswith(".engine"))

    def _require_engine(self):
        """Return a loaded engine, loading it on demand.

        A photo question is useful with no instance running — "what is this
        person doing" comes before pointing a camera anywhere — so it triggers
        the same single-flight load a `start` would and waits, rather than
        reporting `loading` and making the caller poll. Each tools/call has its
        own thread, so blocking here blocks nothing else. Same rule as vop/face.
        """
        self._ensure_model()
        return self._model

    # ── one-shot recognition ─────────────────────────────────────────────

    def _recognize_image(self, args: dict, url_action: str) -> dict:
        """Decode one image and read the people in it once."""
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
            log.error(f"[pose] engine load failed during recognize: {error}",
                      exc_info=True)
            return {"ok": False, "reason": "engine_unavailable", "detail": str(error)}

        confidence = args.get("confidence")
        confidence = (float(confidence) if confidence not in (None, "")
                      else self._confidence)

        from plugins.vision_runtime import decode_poses

        merged = self._merged_config(args.get("instance_id", "") or _DEFAULT_INSTANCE)
        classifier = self._classifier_for(merged)
        outputs, meta = model.infer(frame)
        boxes, scores, keypoints = decode_poses(outputs, meta, confidence)

        height, width = frame.shape[:2]
        half_w, half_h = max(width / 2.0, 1.0), max(height / 2.0, 1.0)
        order = np.argsort(np.asarray(scores, dtype=np.float32))[::-1]
        order = order[:merged["max_persons"]]

        persons = []
        for rank, index in enumerate(order, start=1):
            box = boxes[index]
            kpts = keypoints[index]
            frame_features = PoseFrame(0.0, box, kpts, merged["kpt_confidence"])
            verdict = classifier.classify_frame(frame_features)
            cx = (float(box[0]) + float(box[2])) / 2.0
            cy = (float(box[1]) + float(box[3])) / 2.0
            persons.append({
                "id": rank,
                "score": round(float(scores[index]), 2),
                "box": [float(v) for v in box],
                "keypoints": kpts,
                "position": [round((cx - half_w) / half_w, 3),
                             round((cy - half_h) / half_h, 3)],
                "verdict": verdict,
            })

        # Echo onto the card's topics when an instance is running, so a
        # topic-less card wired into the canvas shows data flowing — the only
        # reason it is startable without a camera. Purely additive, and
        # deliberately not reported back: which topic the echo went out on is
        # not what the caller asked, and every field here is re-read by the
        # model on every turn. Same call as vop's and visual_depth's.
        self._publish_one_shot(args.get("instance_id", ""), persons, frame, started)

        # The full keypoints always go back in this reply, unlike the stream:
        # it is asked for once and read once, so trimming it saves nothing, and
        # a caller who cannot see the frame has no other way to get them.
        return {
            "ok": True,
            "source": source,
            "image_size": [width, height],
            "confidence_threshold": confidence,
            "count": len(persons),
            "latency_ms": int((time.time() - started) * 1000),
            "keypoint_names": list(COCO_KEYPOINTS),
            # Said plainly rather than left for the caller to notice: a single
            # image cannot support waving, walking, turning or a fall, so
            # answering "raising_hand" to "is she waving" would be answering a
            # narrower question than was asked.
            "temporal": False,
            "unavailable_activities": list(TEMPORAL_ACTIVITIES),
            "note": ("单张图片只能判姿势和手臂动作；挥手、走动、转身、跌倒需要"
                     "连续画面，请把卡片连上摄像头后看 {topic}/poses"),
            "persons": [
                {
                    "id": person["id"],
                    "score": person["score"],
                    "bbox": [round(v, 1) for v in person["box"]],
                    "position": person["position"],
                    "posture": person["verdict"].get("posture"),
                    "posture_confidence": person["verdict"].get(
                        "posture_confidence", 0.0),
                    **({"activity": person["verdict"]["activity"]["name"],
                        "activity_zh": person["verdict"]["activity"]["name_zh"],
                        "activity_score": person["verdict"]["activity"]["score"]}
                       if person["verdict"].get("activity") else {}),
                    "evidence": person["verdict"].get("evidence", {}),
                    **({"point_direction": person["verdict"]["point_direction"]}
                       if person["verdict"].get("point_direction") else {}),
                    "keypoints": _full_keypoints(person["keypoints"]),
                }
                for person in persons
            ],
        }

    def _publish_one_shot(self, instance_id: str, persons: list, frame,
                          started: Optional[float] = None) -> Optional[str]:
        with self._nodes_lock:
            node = self._nodes.get(instance_id) if instance_id else None
            if node is None:
                node = self._nodes.get(_DEFAULT_INSTANCE)
            if node is None and len(self._nodes) == 1:
                node = next(iter(self._nodes.values()))
        if node is None:
            return None
        try:
            node.publish_persons(persons, frame, started)
            return node._output_topic
        except Exception as error:  # noqa: BLE001 — never fail the answer on this
            log.warning(f"[pose] could not echo one-shot result: {error}")
            return None

    # ── node lifecycle ───────────────────────────────────────────────────

    def _start_node(self, node_key: str, input_topic: str):
        """Create and start a _PoseNode for the given topic.

        Registers the node before starting it so a concurrent stop can always
        find and cancel it; the lock covers only the registration, never the
        start, so a stop is not queued behind the start it is trying to abort
        (perception/README.md § "Plugin Concurrency").
        """
        with self._nodes_lock:
            if node_key in self._nodes:
                return
            merged = self._merged_config(node_key)
            suffix = node_key.replace("/", "_").replace("-", "_").lstrip("_")
            node = _PoseNode(
                input_topic or None, self._model, merged["confidence"],
                merged["fps"], node_suffix=suffix,
                classifier=self._classifier_for(merged),
                kpt_confidence=merged["kpt_confidence"],
                max_persons=merged["max_persons"],
                publish_keypoints=merged["publish_keypoints"],
                publish_bbox=merged["publish_bbox"],
                publish_overlay=merged["publish_overlay"],
                label_hold=merged["label_hold"],
                activity_interval_s=merged["activity_interval_s"],
            )
            self._executor.add_node(node)
            self._nodes[node_key] = node
        node.start()
        log.info(f"[pose] node started (background): {input_topic}")

    def _retire_node(self, node_key: str) -> Optional[dict]:
        """Stop, unregister and destroy one node. Returns its stop() result."""
        with self._nodes_lock:
            node = self._nodes.pop(node_key, None)
            # Goes with the node: a re-wired card answering info() with the
            # optics of a camera it is no longer fed by is worse than answering
            # with nothing, because downstream cannot tell the difference.
            self._upstream_camera.pop(node_key, None)
        if node is None:
            return None
        node.request_stop()
        result = node.stop()
        # remove-then-destroy via the shared helper: the node must leave the
        # executor before any handle is destroyed, or the spin thread dies on
        # InvalidHandle and takes every subscription in the process with it. It
        # must be destroyed and not merely removed, or the publishers and the
        # ROS node name leak and the next start trips "Publisher already
        # registered".
        dispose_node(self._executor, node, label=f"pose/{node_key}")
        return result

    def _loading_camera_info(self, args: dict, instance_id: str) -> dict:
        """`{"camera_info": [...]}` for an `info()` answered while loading.

        Recorded at `start`, which happens before the engine begins loading, so
        it is available here. vop's version of this reply dropped it, and a
        consumer that started while the card was still loading got no camera
        declaration at all with nothing saying so — reproducible only on a cold
        container, which is why it took a robot to find.
        """
        from plugins.camera_info import inherit

        topic = args.get("input_topic") or ""
        if not topic:
            topics = args.get("input_topics") or []
            topic = topics[0] if topics else ""
        key = instance_id or topic or _DEFAULT_INSTANCE
        with self._nodes_lock:
            upstream = self._upstream_camera.get(key) or {}
        declared = inherit(upstream, topic=output_topic_for(topic),
                           fmt="data/json", stage="perception/pose")
        return {"camera_info": declared} if declared else {}

    # ── MCP surface ──────────────────────────────────────────────────────

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action", name)
        instance_id = args.get("instance_id", "")

        if action == "info":
            return self._info(args, instance_id)

        elif action == "start":
            return self._start(args, instance_id)

        elif action == "stop":
            if instance_id:
                result = self._retire_node(instance_id)
                return result if result is not None else {"state": "idle"}
            with self._nodes_lock:
                keys = list(self._nodes.keys())
            stopped = [key for key in keys if self._retire_node(key) is not None]
            return ({"state": "idle", "stopped_instances": stopped} if stopped
                    else {"state": "idle"})

        elif action in ("recognize_by_photo", "recognize_by_url"):
            return self._recognize_image(args, url_action="recognize_by_url")

        elif action == "list_actions":
            effective = self._effective_backend()
            return {
                "ok": True,
                "backend": effective,
                # Two vocabularies, because two questions are being answered.
                "postures": sorted(POSTURE_LABELS_ZH),
                "activities": (action_vocabulary() if effective != "rules" else []),
                "backend_note": (
                    "hybrid：posture（身体是什么姿势）来自关键点几何，单帧可判；"
                    "activity（人在做什么）来自 ST-GCN++，用它自己的 NTU-60 词表，"
                    "需要画面里有运动。两者答的不是同一个问题 —— NTU-60 的 60 个类"
                    "全是「某人正在做某事」，没有「静止」这个答案，所以站着不动的人"
                    "只有 posture；而几何没有「挥手长什么样」的先验。"
                    "跌倒是唯一的告警类，要模型得分过 0.75 且几何同意身体不直立"
                    if effective == "hybrid" else
                    "rules：只用关键点几何，不加载动作模型。没有 activity 这一路"),
                "actions": action_catalogue(),
                "note": ("姿态标签来自关键点几何，事件标签（跌倒）判的是「转换」"
                         "而不是终态 —— 躺在地上和躺在沙发上是同一个终态。"
                         "跌倒阈值与机位强相关，必须在真机上按相机高度和俯仰角调。"
                         "单目正面视角下「跌倒」和「蹲下再趴下」几乎不可分，"
                         "侧视角可靠得多。"),
                "limitations": {
                    "needs_stream": list(TEMPORAL_ACTIVITIES),
                    "occluded_lower_body": ("看不到髋/膝时姿态一律报 unknown —— "
                                            "坐在桌子后面和站在桌子后面的躯干轴"
                                            "完全一样，猜一个比不猜更糟"),
                    "identity": ("track id 只在人不离开画面期间有效，不是认人。"
                                 "要认人用 face_recognition 卡片"),
                },
            }

        elif action == "config":
            return self._config(args, instance_id)

        return None

    def _info(self, args: dict, instance_id: str) -> dict:
        base = {"name": "PosePerception", "manufacture": "Embodied",
                "model": self._model_name}
        if self._model_loading:
            return {
                **base,
                "state": "loading",
                "desc": self._model_load_status or "Loading pose engine...",
                # The declaration does not wait for the engine — see
                # _loading_camera_info.
                **self._loading_camera_info(args, instance_id),
            }
        if self._model_load_error:
            return {**base, "state": "error",
                    "desc": f"Engine load failed: {self._model_load_error}"}

        with self._nodes_lock:
            nodes = dict(self._nodes)
        instances = {
            key: {
                "input": node._input_topic,
                "output": node._output_topic,
                "skeleton_output": node._skeleton_topic,
                "overlay_output": (node._overlay_topic
                                   if node._publish_overlay else None),
                "confidence": node._confidence,
                "fps": node._fps,
                "kpt_confidence": node._kpt_confidence,
                "max_persons": node._max_persons,
                "publish_keypoints": node._publish_keypoints,
                "publish_bbox": node._publish_bbox,
                "publish_overlay": node._publish_overlay,
                "detect_count": node._detect_count,
                "persons_last_frame": node._person_count,
                "last_actions": dict(node._last_actions),
            }
            for key, node in nodes.items()
        }

        input_topic = args.get("input_topic", "")
        if not input_topic:
            topics_list = args.get("input_topics") or []
            if topics_list:
                input_topic = topics_list[0]
        if instance_id and instance_id in nodes:
            input_topic = nodes[instance_id]._input_topic
        elif not input_topic and nodes:
            input_topic = next(iter(nodes.values()))._input_topic

        topics_in = ([{"topic": input_topic, "format": "image/jpeg"}]
                     if input_topic else [])
        topics_out = []
        if input_topic or nodes:
            topics_out = [
                {"topic": output_topic_for(input_topic), "format": "data/json"},
                {"topic": skeleton_topic_for(input_topic), "format": "sensor/pose2d"},
            ]
            if any(node._publish_overlay for node in nodes.values()) or (
                    not nodes and self._publish_overlay):
                topics_out.append({"topic": overlay_topic_for(input_topic),
                                   "format": "image/jpeg"})

        info = {
            **base,
            "state": "running" if instances else "idle",
            "keypoints": N_KEYPOINTS,
            "keypoint_names": list(COCO_KEYPOINTS),
            "action_backend": self._action_backend,
            "action_backend_effective": self._effective_backend(),
            "postures": sorted(POSTURE_LABELS_ZH),
            "instances": instances,
            "topic_in": topics_in,
            "topic_out": topics_out,
            "desc": ("COCO-17 human keypoints + action labels (TensorRT); "
                     "lean JSON for the agent, full skeleton for the dashboard"),
        }

        # Why the configured backend is not the one running, when that is the
        # case. Without this the card looks like it chose the geometry.
        if self._backend_fallback:
            info["action_backend_note"] = self._backend_fallback
        if self._backend_migrated:
            info["action_backend_migrated"] = self._backend_migrated

        # A backend can construct fine and then fail on every inference — the
        # engine is fetched lazily, and there is no published action engine yet.
        # Reported, because otherwise the card says `hybrid` while answering
        # from geometry and the model gets blamed for the geometry's mistakes.
        engine_errors = {
            key: node._classifier.last_error
            for key, node in nodes.items()
            if getattr(node, "_classifier", None) is not None
            and node._classifier.last_error
        }
        if engine_errors:
            info["action_engine_error"] = engine_errors

        # What the model last actually scored. Without this, an empty activity
        # field is indistinguishable between "the model was never asked" (no
        # motion), "nothing cleared the threshold", and "the engine is broken".
        predictions = {key: node._last_prediction
                       for key, node in nodes.items() if node._last_prediction}
        if predictions:
            info["last_prediction"] = predictions

        # A temporal backend classifies a *clip*, so it is starved by a low fps
        # in a way the geometry is not. Said here rather than enforced: a thin
        # window still beats no actions, but a quietly starved model looks like
        # a wrong model, and that is weeks of misdirected tuning.
        if self._action_backend != "rules":
            configured_fps = [node._fps for node in nodes.values()] or [self._fps]
            if min(configured_fps) < MIN_FPS_FOR_TEMPORAL_BACKEND:
                info["action_fps_note"] = (
                    f"fps={min(configured_fps)} 对 action_backend="
                    f"{self._action_backend!r} 偏低：模型判的是一段视频，"
                    f"{self._action_window_s}s 窗口在这个帧率下只有约 "
                    f"{int(min(configured_fps) * self._action_window_s)} 帧真实数据，"
                    f"要被重采样到 engine 的 48 帧 —— 大部分是插值出来的。"
                    f"挥手/跌倒这类动作建议 fps >= "
                    f"{MIN_FPS_FOR_TEMPORAL_BACKEND}")

        # Pass the camera's optics on. Nothing about the picture's geometry
        # changes here — this card reads the frame and publishes text — so the
        # declaration goes through with only `pipeline` extended, and in
        # particular width/height and K are left exactly as they came. A
        # consumer needs the field of view to turn `position` into an angle at
        # all, and the `id` on this port is what catches camera A's people being
        # paired with camera B's distances.
        if topics_out:
            from plugins.camera_info import inherit
            key = instance_id if instance_id in nodes else (
                next(iter(nodes), None) if nodes
                else (instance_id or input_topic or _DEFAULT_INSTANCE))
            with self._nodes_lock:
                upstream = self._upstream_camera.get(key) or {}
            declared = inherit(upstream, topic=topics_out[0]["topic"],
                               fmt="data/json", stage="perception/pose")
            if declared:
                info["camera_info"] = declared
            elif input_topic:
                info["camera_info_note"] = (
                    "上游相机没有声明 camera_info —— 本卡片报的 position 是归一化"
                    "横向偏移，下游拿不到视场角就没法换算成角度。相机卡片补上声明"
                    "即可，见 phanthymotus-driver/README_dev.md 的 Camera Parameters")
        return info

    def _migrate_backend(self, name: str) -> str:
        """Carry a withdrawn backend name onto its replacement, loudly.

        A card saved while `stgcn` was offered keeps working — refusing it would
        turn an upgrade into a broken card — and `hybrid` is what its owner
        wanted anyway: the learned labels, plus the postures that backend cannot
        produce at all.
        """
        replacement = WITHDRAWN_BACKENDS.get(name)
        if replacement is None:
            return name
        self._backend_migrated = (
            f"action_backend={name!r} has been withdrawn and this card now runs "
            f"{replacement!r}. Alone it has no `standing` or `sitting` class — "
            f"NTU-60 is built from actions and a motionless person is not one — "
            f"and a static clip does not make it abstain: a real person lying on "
            f"pavement came back as \"play with phone/tablet\" at 0.997.")
        log.warning("[pose] %s", self._backend_migrated)
        return replacement

    def _effective_backend(self) -> str:
        """The backend actually running, which is not always the configured one."""
        return "rules" if self._backend_fallback else self._action_backend

    def _start(self, args: dict, instance_id: str) -> dict:
        input_topic = args.get("input_topic")
        if not input_topic:
            topics_list = args.get("input_topics") or []
            if topics_list:
                input_topic = topics_list[0]
        # No topic is a supported mode, as in vop and tts: the card comes up
        # on-demand, loads the engine, owns its publishers and answers
        # recognize_by_photo. It simply has nothing to subscribe to.
        node_key = instance_id or input_topic or _DEFAULT_INSTANCE
        # Recorded before the node starts, and recorded even when empty, so a
        # restart that no longer carries a declaration replaces the old entry
        # rather than leaving one claiming a camera that is no longer wired.
        if input_topic:
            from plugins.camera_info import for_topic as _camera_for_topic
            with self._nodes_lock:
                self._upstream_camera[node_key] = _camera_for_topic(
                    args.get("camera_info"), input_topic)

        with self._nodes_lock:
            running = self._nodes.get(node_key)
        if running is not None:
            return running.start()

        if self._model is None:
            if self._model_loading:
                return {"state": "loading",
                        "message": (self._model_load_status
                                    or "Engine is still loading, please wait...")}
            if self._model_load_error:
                return {"state": "error",
                        "message": f"Engine failed to load: {self._model_load_error}"}

            def _bg_start():
                self._model_loading = True
                self._model_load_error = None
                self._model_load_status = None
                try:
                    self._ensure_model()
                    self._model_loading = False
                    self._model_load_status = None
                    self._start_node(node_key, input_topic)
                except Exception as error:  # noqa: BLE001
                    self._model_loading = False
                    self._model_load_error = str(error)
                    log.error(f"[pose] engine load failed: {error}", exc_info=True)

            threading.Thread(target=_bg_start, daemon=True,
                             name="pose_model_load").start()
            return {"state": "loading", "input": input_topic or "",
                    "output": output_topic_for(input_topic),
                    "message": "Engine loading in background, will start automatically"}

        self._start_node(node_key, input_topic)
        with self._nodes_lock:
            running = self._nodes.get(node_key)
        if running is None:
            # A concurrent stop retired it between start and lookup.
            return {"state": "idle", "input": input_topic}
        return running.start()

    def _config(self, args: dict, instance_id: str) -> dict:
        cfg = {k: v for k, v in args.items()
               if k not in ('action', 'instance_id') and v is not None and v != ''}
        if instance_id:
            with self._nodes_lock:
                self._instance_configs[instance_id] = cfg
                running = instance_id in self._nodes
            # A running instance is retired; the next start picks the new config
            # up. Rebuilding thresholds under a live worker would change what
            # the rules mean halfway through a window.
            if running:
                self._retire_node(instance_id)
            return {"status": "configured", "instance_id": instance_id, "config": cfg}

        if "confidence" in cfg:
            self._confidence = float(cfg["confidence"])
        if "fps" in cfg:
            self._fps = int(cfg["fps"])
        if "kpt_confidence" in cfg:
            self._kpt_confidence = float(cfg["kpt_confidence"])
        if "max_persons" in cfg:
            self._max_persons = int(cfg["max_persons"])
        if "publish_keypoints" in cfg:
            self._publish_keypoints = _keypoint_level(cfg["publish_keypoints"])
        if "publish_bbox" in cfg:
            self._publish_bbox = bool(cfg["publish_bbox"])
        if "publish_overlay" in cfg:
            self._publish_overlay = bool(cfg["publish_overlay"])
        if "action_window_s" in cfg:
            self._action_window_s = float(cfg["action_window_s"])
        if "action_backend" in cfg:
            self._action_backend = self._migrate_backend(str(cfg["action_backend"]))
        if "action_min_score" in cfg:
            self._action_min_score = float(cfg["action_min_score"])
        if "label_hold" in cfg:
            self._label_hold = int(cfg["label_hold"])
        if "activity_interval_s" in cfg:
            self._activity_interval_s = float(cfg["activity_interval_s"])
        for key in _THRESHOLD_KEYS:
            if key in cfg:
                self._plugin_cfg[key] = cfg[key]
        return {"status": "configured", "config": cfg}
