#!/usr/bin/env python3
"""
plugins/face.py — FaceRecognitionPlugin: 人脸注册与持续识别。

订阅 image/jpeg topic，持续识别画面中的人脸并发布身份到 ROS2 topic；
已注册的人输出 id 与 profile，未注册的人输出稳定的 unknown-{id}。

Lifecycle, locking and the start/stop/config state machine are copied from
`plugins/ocr.py`, which is the reference implementation of the rules in
`perception/README.md` § "Plugin Concurrency". The parts that look redundant
(claiming a node key before leaving the lock, registering with the executor
*before* `start()`, bumping a generation on a model-affecting config change)
are each there because omitting them orphaned a live node in production.

What this plugin adds over OCR:

* **A 3-second rolling frame window** per instance. `register_by_stream`
  analyses every frame in it rather than one grab, so a blink, a turned head or
  one motion-blurred frame does not decide the enrolment.
* **A persistent identity database** (`plugins/face_db.py`), shared by all
  instances of the plugin, holding embeddings, names, free-form profiles and a
  visit log.
* **Structured failure reasons.** Enrolment fails for mundane physical reasons
  — nobody in frame, nobody sharp enough, too many candidates to guess a
  subject — and the caller (an LLM or an operator) can only react if it is told
  which. These return `{"ok": false, "reason": ...}` rather than raising, and
  the batch path returns one such record per file.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections import deque
from typing import Any

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from utils.latest_frame import LatestFrame
from utils.log_sampling import SampledLogGate, escape_log_text
from utils.qos import CAMERA_QOS
from utils.ros_lifecycle import dispose_node

from plugins.face_db import (
    DEFAULT_DB_DIR,
    DEFAULT_VISIT_CHECKPOINT_S,
    DEFAULT_MAX_SAMPLES_PER_PERSON,
    DEFAULT_UNKNOWN_CAPACITY,
    DEFAULT_VISIT_GAP_S,
    DEFAULT_VISIT_LOG_MAX,
    FaceDB,
    is_unknown_id,
)
from plugins.face_runtime import (
    DEFAULT_BLUR_MIN,
    DEFAULT_MAX_IMAGE_PIXELS,
    DEFAULT_MAX_IMAGE_SIDE,
    DEFAULT_DET_SIZE,
    DEFAULT_DET_THRESH,
    DEFAULT_FACE_MODEL_DIR,
    DEFAULT_MIN_FACE_PX,
    DEFAULT_NMS_THRESH,
    DetectedFace,
    FaceAnalyzer,
)

log = logging.getLogger(__name__)

_ERROR_LOG_INTERVAL_SECONDS = 10.0
_SEEN_FLUSH_INTERVAL_SECONDS = 300.0

DEFAULT_MATCH_THRESHOLD = 0.35
DEFAULT_SUBJECT_DOMINANCE = 1.6
DEFAULT_MAX_FACES = 8
DEFAULT_DETECT_FPS = 1.0
DEFAULT_ENROLL_WINDOW_S = 3.0
DEFAULT_ENROLL_WINDOW_MAX_FRAMES = 60
DEFAULT_ENROLL_MAX_ANALYZED = 8
DEFAULT_MAX_BATCH = 200
# Only a transfer/memory guard now, not a policy limit: oversized *images* are
# downscaled locally (see FaceAnalyzer.decode_image) rather than rejected, so
# this only has to be larger than any real photo. 64 MB covers a 60 MP
# uncompressed-ish PNG; the pixel cap is what actually protects memory.
DEFAULT_MAX_IMAGE_BYTES = 64 * 1024 * 1024
# /models/uploads is where the file-intake endpoint writes (see
# utils/file_intake.py and the `file_intake` block in config.yaml); /models is
# already listed, which covers it.
DEFAULT_IMAGE_ROOTS = ("/models", "/tmp", "/work")


def detect_interval(cfg: dict) -> float:
    """Seconds to leave between detections, from `detect_fps`.

    Expressed as a frequency because that is what an operator reasons about
    ("look once a second"), and it stays meaningful when the camera's own rate
    changes — a minimum interval in ms does not. Fractional values are the
    point: 0.2 means once every five seconds.

    `detect_fps: 0` means "every frame the camera delivers". `min_interval_ms`
    is still honoured when `detect_fps` is absent, so a canvas saved by an
    earlier build of this plugin keeps working (same courtesy
    `normalize_device` extends to the pre-`device` ASR config).
    """
    if cfg.get("detect_fps") is None and cfg.get("min_interval_ms") is not None:
        legacy = max(0.0, float(cfg["min_interval_ms"])) / 1000.0
        log.info("[face] using legacy min_interval_ms=%s as %.3fs between "
                 "detections; set detect_fps instead",
                 cfg["min_interval_ms"], legacy)
        return legacy
    fps = float(cfg.get("detect_fps", DEFAULT_DETECT_FPS))
    if fps <= 0:
        return 0.0
    return 1.0 / fps


# What the two decoders between them can read (cv2, then Pillow). Suffixes are
# only used to *find* images in a corpus directory — the decode itself does not
# care about the name, so this errs wide.
_IMAGE_SUFFIXES = (
    ".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".webp", ".tif", ".tiff",
    ".ppm", ".pgm", ".pbm", ".gif", ".ico", ".jfif",
)

_RESULT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)


TOOLS = [
    {
        "name": "face_recognition",
        "type": "processor",
        "multiInstance": True,
        "description": (
            "Face recognition — identify people in a camera feed, and register "
            "new people from a photo, the live stream, or a batch package"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "start", "stop", "info", "config",
                        "register_by_photo", "register_by_url",
                        "register_by_stream", "register_by_corpus",
                        "recognize_by_photo", "recognize_by_stream",
                        "list_persons", "get_person", "update_person", "forget",
                        "list_visits",
                    ],
                    "description": "Action to perform",
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 image topic to subscribe (e.g. /hostname/camera/rgb, required for action=start)",
                },
                # `format: file` makes the canvas render a file picker;
                # `uploadTo: mcp` sends it to POST /api/mcp/<id>/file/upload,
                # which streams the bytes to *this* service and returns the path
                # they landed on here. So the value this field receives is
                # already a path perception can open — no shared mount, and no
                # container recreation to make one appear.
                "image_path": {"type": "string", "format": "file", "accept": "image/*", "uploadTo": "mcp", "description": "图片文件。从卡片上传，或填一个容器可读的路径（如 /uploads/alice.jpg）。常见格式都支持（jpg/png/bmp/webp/tiff/gif...），过大的图会本地缩放，不需要预处理"},
                "url":        {"type": "string", "description": "图片的 http(s) 地址，如 https://example.com/alice.jpg。下载后本地解码缩放，格式限制同 image_path"},
                "name":       {"type": "string", "description": "姓名（结构化），如 \"小王\"。有 name 才算已注册；每次识别都会随 id 一起输出"},
                "profile":    {"type": "object", "description": "非结构化画像对象，如 {\"gender\":\"male\",\"team\":\"运营部\",\"note\":\"常穿蓝色外套\"}。键名自定，随 name 一起在每帧输出；传字符串会被存成 {\"note\":\"...\"}"},
                "profile_delete": {"type": "array", "items": {"type": "string"}, "description": "要删除的 profile 键名列表，如 [\"team\",\"note\"]"},
                "merge":      {"type": "boolean", "description": "profile 合并进已有对象（默认 true），false 为整体替换"},
                "since":      {"type": "string", "description": "起始时间，epoch 秒或 ISO-8601，如 \"2026-09-07T15:00\" 或 1788780000。按时间重叠筛选：15:00 前到、15:20 走的人，查 15:00-15:05 也会返回"},
                "until":      {"type": "string", "description": "结束时间，格式同 since。留空表示至今"},
                "person_id":  {"type": "string", "description": "已存在的人员 id，形如 p-3（已命名）或 unknown-7（陌生人）"},
                "person_ids": {"type": "array", "items": {"type": "string"}, "description": "批量删除的 id 列表，如 [\"p-1\",\"p-2\",\"unknown-7\"]；也接受逗号或空格分隔的字符串 \"p-1, unknown-7\"。一次提交完成，不是逐个删。返回 forgotten(已删数量+id列表) 与 missing(不存在的 id)，部分成功是正常结果"},
                "window_s":   {"type": "number", "description": "回看最近多少秒的画面。register_by_stream 默认 3.0，recognize_by_stream 默认 1.0，上限为 enroll_window_s"},
                "package":    {"type": "string", "description": "图片包：目录 / .zip / .tar.gz 的路径或 URL。包内可放 manifest.json 指定每张图的 name 与 profile，如 [{\"file\":\"alice.jpg\",\"name\":\"Alice\",\"person\":\"alice\"}]；没有 manifest 则用同名 .json/.txt 侧文件，再退回文件名"},
                "named":      {"type": "string", "enum": ["all", "named", "unknown"], "description": "过滤范围，默认 all。用于 action=forget 时，'unknown' 表示清空所有陌生人条目"},
                "query":      {"type": "string", "description": "在 id、name、profile 上做子串匹配，如 \"运营部\""},
                "limit":      {"type": "integer", "description": "Page size (default 100)"},
                "offset":     {"type": "integer", "description": "Page offset"},
            },
            "required": ["action"],
            "x-action-params": {
                "start":  {"params": ["input_topic"], "description": "Start recognising faces on an image topic"},
                "stop":   {"params": [], "description": "Stop recognition"},
                "info":   {"params": ["input_topic"], "description": "Report state, topics and database statistics"},
                "config": {"params": [], "description": "Update configuration"},
                "register_by_photo": {
                    "params": ["image_path", "name", "profile"],
                    "description": "Register a person from one photo. Fails with a reason when there is no face, no clear face, or no obvious subject",
                },
                "register_by_url": {
                    "params": ["url", "name", "profile"],
                    "description": "从图片 URL 注册一个人。过大或非常见格式的图片会在本地缩放/转码，而不是被拒绝",
                },
                "register_by_stream": {
                    "params": ["name", "profile", "window_s"],
                    "description": "Register the person currently in front of the camera, using the last few seconds of the live stream",
                },
                "register_by_corpus": {
                    "params": ["package"],
                    "description": "Register many people from a package of photos; returns a per-photo result saying which succeeded and why the others did not",
                },
                "recognize_by_photo": {
                    "params": ["image_path", "url"],
                    "description": "认出照片里的人 — 只读，不写库：返回每张人脸的 id/name/profile 与相似度，不会登记陌生人",
                },
                "recognize_by_stream": {
                    "params": ["window_s"],
                    "description": "认出摄像头前的人 — 只读，不写库：返回当前画面里每个人的 id/name/profile 与相似度",
                },
                "list_persons":  {"params": ["named", "query", "limit", "offset"], "description": "List registered people with their name and profile"},
                "get_person":    {"params": ["person_id"], "description": "Read one person's full record"},
                "update_person": {"params": ["person_id", "name", "profile", "profile_delete", "merge"], "description": "Edit a person's name or profile; setting a name on an unknown-N id names that identity, keeping the id"},
                "forget":        {"params": ["person_id", "person_ids", "named"], "description": "删除人员：单个 person_id、批量 person_ids，或 named='unknown' 清空所有陌生人。id 退役后永不复用"},
                "list_visits":   {"params": ["person_id", "since", "until", "limit", "offset"], "description": "访问记录：查询某段时间内出现过的人。一次连续出现算一条记录，含首末时间与出现次数"},
            },
        },
        # Deliberately minimal, as in plugins/ocr.py: only what an operator
        # meaningfully decides. Expert knobs (model_dir, db_dir, device,
        # det_size, det_thresh, nms_thresh, num_threads, enroll_window_s,
        # max_batch, image_roots, ...) stay config.yaml-only — dispatch still
        # honours them, they are just not advertised to the config UI.
        "configSchema": {
            "type": "object",
            "properties": {
                "device":            {"type": "string", "enum": ["auto", "cpu", "gpu"], "default": "auto", "description": "推理设备。auto=有 GPU 用 GPU，没有则用 CPU"},
                "detect_fps":        {"type": "number", "minimum": 0, "default": DEFAULT_DETECT_FPS, "description": "检测频率，每秒 x 次，支持小数（如 0.5 = 每 2 秒一次）；0=每帧都检测", "scope": "instance"},
                "match_threshold":   {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": DEFAULT_MATCH_THRESHOLD, "description": "余弦相似度阈值，越高越严格（越不容易认错人，但越容易认不出）"},
                "min_face_px":       {"type": "integer", "minimum": 16, "default": DEFAULT_MIN_FACE_PX, "description": "最小人脸边长(px)，小于此值不做识别"},
                "blur_min":          {"type": "number", "minimum": 0.0, "default": DEFAULT_BLUR_MIN, "description": "清晰度下限(拉普拉斯方差)，低于此值视为模糊人脸"},
                "max_faces":         {"type": "integer", "minimum": 1, "default": DEFAULT_MAX_FACES, "description": "单帧最多处理的人脸数"},
                "unknown_capacity":  {"type": "integer", "minimum": 0, "default": DEFAULT_UNKNOWN_CAPACITY, "description": "陌生人(unknown-N)数量上限，超出时淘汰最久未见的；已注册人员不受影响"},
            },
        },
        "topic_in":  [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [{"format": "data/json",  "desc": "recognised identities per frame"}],
    }
]


# ── failure reasons ───────────────────────────────────────────────────────────

REASON_NO_FACE = "no_face"
REASON_LOW_QUALITY = "low_quality"
REASON_AMBIGUOUS = "ambiguous_subject"
REASON_NO_FRAMES = "no_frames"
REASON_BAD_INPUT = "bad_input"

# Which reason to report when frames in a window disagree. Ordered by what the
# operator has to change: move people out of shot, then get closer / hold
# still, then point the camera at somebody at all. Reporting "no clear face"
# for a window that mostly contained a crowd sends them to fix the wrong thing.
_REASON_PRECEDENCE = (REASON_AMBIGUOUS, REASON_LOW_QUALITY, REASON_NO_FACE)


class _BadInput(Exception):
    """An image or package could not be loaded. Carries the caller-facing detail."""

    def __init__(self, detail: str, source: str = ""):
        super().__init__(detail)
        self.detail = detail
        self.source = source

    def as_result(self) -> dict:
        result = {"ok": False, "reason": REASON_BAD_INPUT, "detail": self.detail}
        if self.source:
            result["source"] = self.source
        return result


def _face_output_topic(input_topic: str) -> str:
    return f"{input_topic}/face"


# ── configuration ─────────────────────────────────────────────────────────────

def _analyzer_options(cfg: dict) -> dict:
    det_size = cfg.get("det_size") or DEFAULT_DET_SIZE
    if isinstance(det_size, (list, tuple)) and len(det_size) == 2:
        det_size = (int(det_size[0]), int(det_size[1]))
    else:
        det_size = DEFAULT_DET_SIZE
    return {
        "model_dir": str(cfg.get("model_dir", DEFAULT_FACE_MODEL_DIR)),
        "device": str(cfg.get("device", "cpu")),
        "det_size": det_size,
        "det_thresh": float(cfg.get("det_thresh", DEFAULT_DET_THRESH)),
        "nms_thresh": float(cfg.get("nms_thresh", DEFAULT_NMS_THRESH)),
        "num_threads": int(cfg.get("num_threads", 2)),
        "warmup": bool(cfg.get("warmup", True)),
    }


def _db_options(cfg: dict) -> dict:
    return {
        "db_dir": str(cfg.get("db_dir", DEFAULT_DB_DIR)),
        "unknown_capacity": int(
            cfg.get("unknown_capacity", DEFAULT_UNKNOWN_CAPACITY)
        ),
        "max_samples_per_person": int(
            cfg.get("max_samples_per_person", DEFAULT_MAX_SAMPLES_PER_PERSON)
        ),
        "visit_gap_s": float(cfg.get("visit_gap_s", DEFAULT_VISIT_GAP_S)),
        "visit_log_max": int(cfg.get("visit_log_max", DEFAULT_VISIT_LOG_MAX)),
        "visit_checkpoint_s": float(
            cfg.get("visit_checkpoint_s", DEFAULT_VISIT_CHECKPOINT_S)
        ),
    }


def _decode_options(cfg: dict) -> dict:
    """How incoming photos are normalised before detection."""
    return {
        "max_side": int(cfg.get("max_image_side", DEFAULT_MAX_IMAGE_SIDE)),
        "max_pixels": int(cfg.get("max_image_pixels", DEFAULT_MAX_IMAGE_PIXELS)),
    }


def _gates(cfg: dict) -> dict:
    """The quality/ambiguity thresholds, resolved once per call."""
    return {
        "det_thresh": float(cfg.get("det_thresh", DEFAULT_DET_THRESH)),
        "min_face_px": float(cfg.get("min_face_px", DEFAULT_MIN_FACE_PX)),
        "blur_min": float(cfg.get("blur_min", DEFAULT_BLUR_MIN)),
        "dominance": float(cfg.get("subject_dominance", DEFAULT_SUBJECT_DOMINANCE)),
        "match_threshold": float(cfg.get("match_threshold", DEFAULT_MATCH_THRESHOLD)),
        "max_faces": int(cfg.get("max_faces", DEFAULT_MAX_FACES)),
    }


def _engine_signature(cfg: dict) -> tuple:
    """What a config change must rebuild the engine for."""
    options = _analyzer_options(cfg)
    db_options = _db_options(cfg)
    # unknown_capacity is applied in place (set_unknown_capacity), so it must
    # not force a rebuild — changing it from the card would otherwise drop
    # every running instance and reload both models.
    db_options.pop("unknown_capacity", None)
    return (
        tuple(sorted((key, str(value)) for key, value in options.items())),
        tuple(sorted((key, str(value)) for key, value in db_options.items())),
    )


class _FaceEngine:
    """The analyzer and the identity database, loaded as one unit.

    They are built together because both can fail slowly (a model download, a
    corrupt database) and the plugin's single-flight loader has exactly one
    slot; a half-loaded engine would let recognition start against a database
    that never opened.
    """

    def __init__(self, analyzer: FaceAnalyzer, db: FaceDB):
        self.analyzer = analyzer
        self.db = db

    def close(self) -> None:
        try:
            self.db.flush()
        except Exception:  # noqa: BLE001 - closing must not raise
            log.warning("[face] failed to flush the database on close",
                        exc_info=True)
        self.analyzer.close()


def _build_engine(cfg: dict) -> _FaceEngine:
    analyzer = FaceAnalyzer(**_analyzer_options(cfg))
    db = FaceDB(**_db_options(cfg))
    return _FaceEngine(analyzer, db)


def _close_quietly(engine) -> None:
    close = getattr(engine, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - best-effort release
            log.warning("[face] engine close failed", exc_info=True)


# ── subject selection (pure, unit-tested) ─────────────────────────────────────

def _subject_weight(face: DetectedFace, shape: tuple[int, int]) -> float:
    """Area, discounted for being off-centre.

    Enrolment needs "the person being shown to the camera", which is bigger
    *and* more central than a bystander. Pure area picks the bystander who
    happens to stand closer to the lens edge; pure centrality picks a distant
    face framed dead-on. The 40% centre discount is a deliberate compromise and
    the number that `subject_dominance` is compared against.
    """
    height, width = shape[0], shape[1]
    if width <= 0 or height <= 0:
        return face.area
    centre_x, centre_y = width / 2.0, height / 2.0
    face_x = (face.bbox[0] + face.bbox[2]) / 2.0
    face_y = (face.bbox[1] + face.bbox[3]) / 2.0
    half_diagonal = (width ** 2 + height ** 2) ** 0.5 / 2.0
    distance = ((face_x - centre_x) ** 2 + (face_y - centre_y) ** 2) ** 0.5
    offness = min(1.0, distance / half_diagonal) if half_diagonal else 0.0
    return face.area * (1.0 - 0.4 * offness)


def _candidate_summary(face: DetectedFace, shape: tuple[int, int]) -> dict:
    return {
        "bbox": face.bbox_xywh(),
        "det_score": round(face.det_score, 4),
        "min_side_px": int(face.min_side),
        "blur": round(face.blur, 2),
        "weight": round(_subject_weight(face, shape), 1),
    }


def select_subject(
    faces: list[DetectedFace], shape: tuple[int, int], gates: dict
) -> tuple[DetectedFace | None, dict | None]:
    """Pick the single person an enrolment is about.

    Returns `(face, None)` on success or `(None, failure)` where `failure` is
    the caller-facing record explaining which physical condition was not met.
    """
    if not faces:
        return None, {
            "ok": False,
            "reason": REASON_NO_FACE,
            "detail": "no face detected",
            "faces": 0,
        }

    passing = [
        face for face in faces
        if face.det_score >= gates["det_thresh"]
        and face.min_side >= gates["min_face_px"]
        and face.blur >= gates["blur_min"]
    ]
    if not passing:
        best = max(faces, key=lambda face: _subject_weight(face, shape))
        reasons = []
        if best.det_score < gates["det_thresh"]:
            reasons.append(
                f"detector score {best.det_score:.2f} < {gates['det_thresh']:.2f}"
            )
        if best.min_side < gates["min_face_px"]:
            reasons.append(
                f"face {int(best.min_side)} px < {int(gates['min_face_px'])} px "
                "(move closer)"
            )
        if best.blur < gates["blur_min"]:
            reasons.append(
                f"sharpness {best.blur:.1f} < {gates['blur_min']:.1f} (hold still)"
            )
        return None, {
            "ok": False,
            "reason": REASON_LOW_QUALITY,
            "detail": "no clear face: " + "; ".join(reasons),
            "faces": len(faces),
            "candidates": [_candidate_summary(face, shape) for face in faces[:5]],
        }

    passing.sort(key=lambda face: _subject_weight(face, shape), reverse=True)
    if len(passing) >= 2:
        primary = _subject_weight(passing[0], shape)
        runner_up = _subject_weight(passing[1], shape)
        ratio = primary / runner_up if runner_up > 0 else float("inf")
        if ratio < gates["dominance"]:
            return None, {
                "ok": False,
                "reason": REASON_AMBIGUOUS,
                "detail": (
                    f"{len(passing)} faces pass the quality gate; the largest is "
                    f"only {ratio:.2f}x the runner-up (need "
                    f"{gates['dominance']:.2f}x). Have one person face the camera, "
                    "or register from a photo instead."
                ),
                "faces": len(passing),
                "candidates": [
                    _candidate_summary(face, shape) for face in passing[:5]
                ],
            }
    return passing[0], None


def _worst_reason(failures: list[dict]) -> dict:
    """Pick the failure to report when several frames failed differently."""
    for reason in _REASON_PRECEDENCE:
        for failure in failures:
            if failure.get("reason") == reason:
                counts = {
                    name: sum(1 for f in failures if f.get("reason") == name)
                    for name in _REASON_PRECEDENCE
                }
                return {
                    **failure,
                    "frames_examined": len(failures),
                    "frame_reasons": {
                        name: count for name, count in counts.items() if count
                    },
                }
    return {
        "ok": False,
        "reason": REASON_NO_FACE,
        "detail": "no face detected",
        "frames_examined": len(failures),
    }


# ── image sources ─────────────────────────────────────────────────────────────

def _load_image_bytes(args: dict, cfg: dict) -> tuple[bytes, str]:
    """Read image bytes from `url` or `image_path`.

    **Deliberately no base64 input.** It was there, and an LLM failed on it
    twice in production: a 43 800-character string is not something a model can
    carry through its own context reliably, and what arrived was truncated, so
    the decoder correctly refused it. Both remaining channels move a *reference*
    instead of the bytes.

    The byte ceiling here is a transfer/memory guard, not a policy limit: an
    image that is merely *large* is downscaled and converted locally by
    `FaceAnalyzer.decode_image`, because "your photo is 24 MB" or "we only take
    JPEG" is a limitation of ours rather than a property of their photo.

    `url` is its own action (`register_by_url`) rather than a parameter
    smuggled into the photo path, so the capability is visible on the card. Note
    it does let a caller make this container issue an outbound request —
    acceptable for a deliberate, named action, which is why it is not folded
    into the generic input.
    """
    max_bytes = int(cfg.get("max_image_bytes", DEFAULT_MAX_IMAGE_BYTES))

    if args.get("image_b64"):
        # Say what to do instead, rather than silently ignoring the argument:
        # the model that reaches for base64 has the file in hand already.
        raise _BadInput(
            "image_b64 is no longer accepted — a long base64 string does not "
            "survive being carried through an LLM's context. Upload the file "
            "through POST /api/mcp/<mcp_id>/file/upload and pass the path it "
            "returns as image_path, or use register_by_url.",
            "image_b64",
        )

    url = args.get("url") or args.get("image_url")
    if url:
        return _fetch_url(str(url), max_bytes), str(url)

    path = args.get("image_path")
    if path:
        return _read_local(str(path), cfg, max_bytes), str(path)

    raise _BadInput("one of image_path or url is required")


def _image_roots(cfg: dict) -> tuple[str, ...]:
    roots = cfg.get("image_roots") or DEFAULT_IMAGE_ROOTS
    return tuple(os.path.realpath(str(root)) for root in roots)


def _check_under_roots(path: str, cfg: dict) -> str:
    """Confine caller-supplied paths to the configured roots.

    The MCP server has no authentication and runs as root in the container, so
    an unrestricted path would let any LAN caller probe the filesystem by
    asking whether a file decodes as an image. Symlinks are resolved first —
    a link inside a root pointing outside it would otherwise pass.
    """
    resolved = os.path.realpath(path)
    roots = _image_roots(cfg)
    if any(resolved == root or resolved.startswith(root + os.sep) for root in roots):
        return resolved
    # A caller that names a plausible-but-invisible path is almost always
    # another container's filesystem — agent-core's /work and /tmp are its own,
    # which is exactly how the first LLM attempt failed. Say where to put it.
    raise _BadInput(
        f"path must be under one of {', '.join(roots)}: got {path!r}. "
        "If you are writing the file from another container (e.g. agent-core), "
        "If you are calling from another container, upload the file through "
        "POST /api/mcp/<mcp_id>/file/upload — the reply carries a path this "
        "container can open — or use register_by_url.",
        path,
    )


def _read_local(path: str, cfg: dict, max_bytes: int) -> bytes:
    resolved = _check_under_roots(path, cfg)
    try:
        if os.path.isdir(resolved):
            raise _BadInput(f"{path!r} is a directory, not an image", path)
        size = os.path.getsize(resolved)
        if size > max_bytes:
            raise _BadInput(
                    f"file is {size} bytes, over the {max_bytes} byte transfer cap "
                "(raise max_image_bytes if this is a real photo)", path
            )
        with open(resolved, "rb") as handle:
            return handle.read()
    except OSError as error:
        raise _BadInput(f"cannot read {path!r}: {error}", path) from error


def _fetch_url(url: str, max_bytes: int) -> bytes:
    if not url.lower().startswith(("http://", "https://")):
        raise _BadInput(f"only http(s) URLs are supported: {url!r}", url)
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = response.read(max_bytes + 1)
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise _BadInput(f"cannot fetch {url!r}: {error}", url) from error
    if len(data) > max_bytes:
        raise _BadInput(f"download exceeds the {max_bytes} byte limit", url)
    if not data:
        raise _BadInput(f"{url!r} returned no data", url)
    return data


# ── package extraction ────────────────────────────────────────────────────────

def _safe_member_name(name: str) -> str | None:
    """Reject archive members that could escape the extraction directory.

    Same policy as `model_downloader._check_bundle_relpath`: no absolute paths,
    no `..`, no drive letters, and nothing that is not a plain relative path.
    Returns the normalised name, or None if it must be skipped.
    """
    if not name or name.endswith("/"):
        return None
    normalised = os.path.normpath(name.replace("\\", "/"))
    if (
        os.path.isabs(normalised)
        or normalised.startswith("..")
        or os.sep + ".." + os.sep in os.sep + normalised + os.sep
    ):
        return None
    return normalised


def _extract_package(package: str, cfg: dict, destination: str) -> str:
    """Materialise a package as a directory of files. Returns that directory."""
    if package.lower().startswith(("http://", "https://")):
        max_bytes = int(cfg.get("max_package_bytes", 512 * 1024 * 1024))
        payload = _fetch_url(package, max_bytes)
        archive = os.path.join(destination, "package.bin")
        with open(archive, "wb") as handle:
            handle.write(payload)
        source = archive
    else:
        source = _check_under_roots(package, cfg)

    if os.path.isdir(source):
        return source

    payload = os.path.join(destination, "payload")
    os.makedirs(payload, exist_ok=True)
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                name = _safe_member_name(info.filename)
                if name is None:
                    log.warning("[face] skipping unsafe archive member %s",
                                escape_log_text(info.filename))
                    continue
                target = os.path.join(payload, name)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with archive.open(info) as src, open(target, "wb") as dst:
                    dst.write(src.read())
        return payload

    try:
        with tarfile.open(source) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    # Skips symlinks and devices as well as directories — a
                    # symlink member is the classic tar escape.
                    continue
                name = _safe_member_name(member.name)
                if name is None:
                    log.warning("[face] skipping unsafe archive member %s",
                                escape_log_text(member.name))
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                target = os.path.join(payload, name)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with extracted, open(target, "wb") as dst:
                    dst.write(extracted.read())
        return payload
    except tarfile.TarError as error:
        raise _BadInput(
            f"{package!r} is neither a directory, a zip, nor a tar archive: {error}",
            package,
        ) from error


def _read_manifest(directory: str) -> dict[str, dict]:
    """Read `manifest.json`, accepting either supported shape.

    A list of records (`[{"file": ..., "profile": ..., "person": ...}]`) or a
    flat mapping of filename to profile text. Both appear in the wild because
    the second is what somebody writes by hand.
    """
    path = os.path.join(directory, "manifest.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError) as error:
        raise _BadInput(f"manifest.json is unreadable: {error}", path) from error

    entries: dict[str, dict] = {}
    if isinstance(raw, dict):
        for filename, value in raw.items():
            if isinstance(value, dict):
                entries[str(filename)] = dict(value)
            else:
                entries[str(filename)] = {"profile": str(value)}
    elif isinstance(raw, list):
        for record in raw:
            if not isinstance(record, dict):
                continue
            filename = record.get("file") or record.get("filename")
            if not filename:
                continue
            entries[str(filename)] = {
                key: value for key, value in record.items()
                if key not in ("file", "filename")
            }
    else:
        raise _BadInput("manifest.json must be an object or an array", path)
    return entries


def _sidecar_identity(image_path: str) -> tuple[str, dict]:
    """Per-image fallback: `alice.json`, then `alice.txt`, else the file stem.

    Returns `(name, profile)`. A sidecar may carry `name` and a `profile`
    object; `profile` as a plain string is read as the name, which is the shape
    a hand-written sidecar tends to have.
    """
    stem, _ = os.path.splitext(image_path)
    json_path = f"{stem}.json"
    if os.path.isfile(json_path):
        try:
            with open(json_path, encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                name = data.get("name")
                profile = data.get("profile")
                if name is None and isinstance(profile, str):
                    name, profile = profile, None
                return str(name or ""), (profile if isinstance(profile, dict) else {})
            return str(data), {}
        except (OSError, ValueError):
            log.warning("[face] ignoring unreadable sidecar %s",
                        escape_log_text(json_path))
    text_path = f"{stem}.txt"
    if os.path.isfile(text_path):
        try:
            with open(text_path, encoding="utf-8") as handle:
                return handle.read().strip(), {}
        except OSError:
            pass
    return os.path.basename(stem), {}


# ── ROS2 Node ─────────────────────────────────────────────────────────────────

class _FaceNode(Node):
    """订阅 image/jpeg topic，持续识别人脸身份并发布结果。"""

    def __init__(
        self,
        input_topic: str,
        engine: _FaceEngine,
        cfg: dict,
        node_suffix: str = "",
        detect_interval_s: float = 1.0,
    ):
        node_name = f"face_{node_suffix}" if node_suffix else "face"
        super().__init__(node_name)

        self._input_topic = input_topic
        self._output_topic = _face_output_topic(input_topic)
        self._engine = engine
        self._cfg = dict(cfg)
        # Seconds between detections; 0 = every frame. See detect_interval().
        self._detect_interval = max(0.0, float(detect_interval_s))
        self.state = "idle"

        self._sub = None
        self._pub = self.create_publisher(String, self._output_topic, _RESULT_QOS)

        self._frames: LatestFrame = LatestFrame()
        self._frames.close()
        # Rolling enrolment window, kept beside LatestFrame rather than in place
        # of it: the worker still wants "newest frame, drop the rest", while
        # register_by_stream wants the last few seconds. Bounded by age and
        # by count, so a fast camera cannot grow it without limit.
        self._window: deque[tuple[bytes, float]] = deque(
            maxlen=max(1, int(cfg.get(
                "enroll_window_max_frames", DEFAULT_ENROLL_WINDOW_MAX_FRAMES
            )))
        )
        self._window_lock = threading.Lock()
        self._window_seconds = float(
            cfg.get("enroll_window_s", DEFAULT_ENROLL_WINDOW_S)
        )

        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stop_event.set()
        self._generation = 0
        self._worker_threads: list[threading.Thread] = []
        self._node_lock = threading.RLock()
        self._retired = False
        self._log_gate = SampledLogGate(every=100)
        self._last_error_log_at: float | None = None
        self._last_flush_at = time.monotonic()
        log.info("[face] node created: subscribing=%s, publishing=%s",
                 self._input_topic, self._output_topic)

    # ── lifecycle (mirrors _OCRNode) ──────────────────────────────────────

    def start(self) -> dict:
        with self._node_lock:
            return self._start_locked()

    def _start_locked(self) -> dict:
        if self._retired:
            return self._status_dict()
        if self.state == "running":
            return self._status_dict()
        if not self._engine:
            raise RuntimeError("face engine not configured")

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

        log.info("[face] started: %s → %s", self._input_topic, self._output_topic)
        return self._status_dict()

    def stop(self) -> dict:
        with self._node_lock:
            self.state = "idle"
            self._stop_event.set()
            self._frames.close()
            deadline = time.monotonic() + 3.0
            for thread in self._worker_threads:
                if thread.is_alive():
                    thread.join(timeout=max(0.0, deadline - time.monotonic()))
            self._worker_threads = [
                thread for thread in self._worker_threads if thread.is_alive()
            ]
            if self._worker_threads:
                log.warning("[face] %d worker(s) still stopping after timeout: %s",
                            len(self._worker_threads), self._input_topic)
            with self._window_lock:
                self._window.clear()
            # force: an instance stopping means everyone currently in frame has
            # left as far as this camera is concerned, so close their visits
            # rather than losing them.
            self._flush_seen(force_close=True)
            log.info("[face] stopped: %s", self._input_topic)
            return {"state": "idle"}

    def retire(self) -> dict:
        with self._node_lock:
            self._retired = True
            return self.stop()

    @property
    def worker_alive(self) -> bool:
        return any(thread.is_alive() for thread in self._worker_threads)

    # ── frame intake ──────────────────────────────────────────────────────

    def _image_cb(self, msg: CompressedImage):
        stop_event = self._stop_event
        frames = self._frames
        if self.state != "running" or stop_event.is_set():
            return
        data = bytes(msg.data)
        now = time.time()
        frames.push((data, now))
        with self._window_lock:
            self._window.append((data, now))
            cutoff = now - self._window_seconds
            while self._window and self._window[0][1] < cutoff:
                self._window.popleft()

    def recent_frames(self, window_s: float | None = None) -> list[tuple[bytes, float]]:
        """Frames captured within the last `window_s` seconds, oldest first."""
        limit = self._window_seconds if window_s is None else min(
            float(window_s), self._window_seconds
        )
        cutoff = time.time() - max(0.0, limit)
        with self._window_lock:
            return [item for item in self._window if item[1] >= cutoff]

    # ── recognition worker ────────────────────────────────────────────────

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
        while not stop_event.is_set():
            frame = frames.pop(timeout=1.0)
            if frame is None:
                continue
            image_bytes, timestamp = frame

            started = time.time()
            payload = self._recognise_to_payload(image_bytes, timestamp)
            if not self._is_generation_active(generation, stop_event):
                continue
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self._pub.publish(msg)
            self._log_payload(payload)

            # Closing a visit is in-memory bookkeeping plus one append, so it
            # can run every cycle; only the persons.json rewrite is throttled.
            try:
                self._engine.db.close_stale_visits()
                # Throttled inside the DB to visit_checkpoint_s, so calling it
                # every cycle costs a lock and a timestamp compare.
                self._engine.db.checkpoint_open_visits()
            except Exception:  # noqa: BLE001
                log.warning("[face] visit bookkeeping failed", exc_info=True)
            if time.monotonic() - self._last_flush_at >= _SEEN_FLUSH_INTERVAL_SECONDS:
                self._flush_seen()

            if self._detect_interval > 0:
                remaining = self._detect_interval - (time.time() - started)
                if remaining > 0:
                    stop_event.wait(remaining)

    def _recognise_to_payload(self, image_bytes: bytes, timestamp: float) -> dict:
        started = time.time()
        # No `topic` field: a subscriber already knows which topic it read
        # this from, and OCR's build_ocr_payload does not carry one either.
        payload: dict[str, Any] = {
            "ts": timestamp,
            "count": 0,
            "faces": [],
        }
        try:
            analyzer = self._engine.analyzer
            database = self._engine.db
            gates = _gates(self._cfg)
            image = analyzer.decode_image(
                image_bytes, **_decode_options(self._cfg)
            )
            if image is None:
                payload["error"] = "undecodable frame"
                return payload
            shape = image.shape[:2]
            faces = analyzer.detect(image, max_faces=gates["max_faces"])
            entries = []
            for face in faces:
                analyzer.prepare(image, face)
                usable = (
                    face.det_score >= gates["det_thresh"]
                    and face.min_side >= gates["min_face_px"]
                    and face.blur >= gates["blur_min"]
                )
                entry = {
                    "bbox": face.bbox_xywh(),
                    "det_score": round(face.det_score, 4),
                    "blur": round(face.blur, 2),
                    "min_side_px": int(face.min_side),
                }
                if not usable:
                    # Reported, but neither matched nor enrolled. Matching a
                    # blurred 30 px face is a coin flip, and auto-enrolling it
                    # would spend an unknown-N slot on a smear that never
                    # matches anything again. "There is a face here and I
                    # cannot identify it" is the honest answer.
                    entry.update({
                        "person_id": None,
                        "name": "",
                        "known": False,
                        "quality": "low",
                        "reason": REASON_LOW_QUALITY,
                    })
                    entries.append(entry)
                    continue

                embedding = analyzer.embed(face.aligned)
                person_id, score = database.match(
                    embedding, gates["match_threshold"]
                )
                if person_id is None:
                    record = database.enroll_unknown(embedding)
                    person_id = record["id"]
                else:
                    record = database.get_person(person_id)
                # Sightings drive last_seen_at and the visit log; neither
                # timestamp goes in the payload — "when was this person around"
                # is a visit-log question (list_visits), not a per-frame field.
                database.record_sighting(person_id, timestamp, self._input_topic)
                entry.update({
                    "person_id": person_id,
                    "name": record["name"],
                    "profile": record["profile"],
                    "known": record["named"],
                    "score": round(float(score), 4),
                    "quality": "ok",
                })
                entries.append(entry)
            payload["faces"] = entries
            payload["count"] = len(entries)
        except Exception as error:  # noqa: BLE001 - reported in the payload
            payload["error"] = str(error)
        payload["latency_ms"] = int((time.time() - started) * 1000)
        return payload

    def _log_payload(self, payload: dict) -> None:
        """Transitions unthrottled, steady state sampled — as in plugins/ocr.py."""
        outcome = "error" if "error" in payload else "ok"
        should_log, transition, occurrence = self._log_gate.check(outcome)
        if outcome == "error":
            now = time.monotonic()
            if transition or (
                self._last_error_log_at is None
                or now - self._last_error_log_at >= _ERROR_LOG_INTERVAL_SECONDS
            ):
                log.error("[face] recognition error (occurrence %d): %s",
                          occurrence, escape_log_text(payload["error"]))
                self._last_error_log_at = now
            elif should_log:
                log.debug("[face] recognition error (occurrence %d): %s",
                          occurrence, escape_log_text(payload["error"]))
        elif should_log:
            log.debug("[face] published to %s: %d face(s) (frame %d%s)",
                      self._output_topic, payload["count"], occurrence,
                      ", recovered" if transition and occurrence == 1 else "")

    def _flush_seen(self, force_close: bool = False) -> None:
        """Persist `last_seen_at` and close any visit whose subject has left."""
        self._last_flush_at = time.monotonic()
        try:
            self._engine.db.close_stale_visits(force=force_close)
            self._engine.db.flush()
        except Exception:  # noqa: BLE001 - never kill the worker over this
            log.warning("[face] failed to persist sighting timestamps",
                        exc_info=True)

    def apply_config(self, cfg: dict) -> None:
        """Apply lightweight fields to a running node, as OCR's config does."""
        self._cfg = dict(cfg)
        self._detect_interval = detect_interval(cfg)
        self._window_seconds = float(
            cfg.get("enroll_window_s", DEFAULT_ENROLL_WINDOW_S)
        )

    def _status_dict(self) -> dict:
        return {
            "state": self.state,
            "topic_in": [{"topic": self._input_topic, "format": "image/jpeg", "desc": "image input"}],
            "topic_out": [{"topic": self._output_topic, "format": "data/json", "desc": "recognised identities"}],
        }


# ── Plugin ────────────────────────────────────────────────────────────────────

class FaceRecognitionPlugin:
    """Face recognition MCP plugin with a non-blocking start/stop/load machine.

    The state machine, the single-flight loader and every locking rule are
    `plugins/ocr.py`'s; see that docstring. The engine here is the analyzer plus
    the identity database.
    """

    PREFIX = "face_recognition"

    def __init__(self, plugin_cfg: dict, executor):
        self._plugin_cfg = dict(plugin_cfg)
        self._executor = executor

        self._state_lock = threading.Lock()
        self._nodes: dict[str, _FaceNode] = {}
        self._instance_configs: dict[str, dict] = {}
        self._pending_starts: dict[str, str] = {}
        self._engine: _FaceEngine | None = None
        self._engine_state = "idle"          # idle|loading|ready|error
        self._load_error: str | None = None
        self._load_generation = 0

        log.info("[face] plugin init: device=%s, model_dir=%s, db_dir=%s",
                 plugin_cfg.get("device", "cpu"),
                 plugin_cfg.get("model_dir", DEFAULT_FACE_MODEL_DIR),
                 plugin_cfg.get("db_dir", DEFAULT_DB_DIR))

    def get_tools(self) -> list:
        return TOOLS

    # ── background loader (single-flight) ────────────────────────────────

    def _spawn_loader_locked(self) -> None:
        self._engine_state = "loading"
        self._load_error = None
        generation = self._load_generation
        cfg = dict(self._plugin_cfg)
        threading.Thread(
            target=self._loader, args=(generation, cfg),
            name="face-engine-loader", daemon=True,
        ).start()

    def _loader(self, generation: int, cfg: dict) -> None:
        try:
            engine = _build_engine(cfg)
        except Exception as error:  # noqa: BLE001 - surfaced via state/info
            log.exception("[face] engine load failed")
            with self._state_lock:
                if generation == self._load_generation:
                    self._engine_state = "error"
                    self._load_error = str(error)
            return

        with self._state_lock:
            if generation != self._load_generation:
                stale = engine
            else:
                self._engine = engine
                self._engine_state = "ready"
                stale = None
        if stale is not None:
            _close_quietly(stale)
            return

        while True:
            with self._state_lock:
                if generation != self._load_generation or not self._pending_starts:
                    return
                node_key, input_topic = next(iter(self._pending_starts.items()))
            try:
                node = self._create_node(node_key, input_topic, engine)
            except Exception as error:  # noqa: BLE001 - keep serving others
                log.error("[face] failed to build instance %r on %r: %s",
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
                        log.error("[face] failed to register instance %r: %s",
                                  node_key, escape_log_text(error))
                        del self._pending_starts[node_key]
                    else:
                        self._nodes[node_key] = node
                        del self._pending_starts[node_key]
                        registered = True
            if not registered:
                try:
                    node.destroy_node()
                except Exception:  # noqa: BLE001
                    pass
                continue
            try:
                node.start()
            except Exception as error:  # noqa: BLE001
                log.error("[face] failed to start instance %r: %s",
                          node_key, escape_log_text(error))
                with self._state_lock:
                    if self._nodes.get(node_key) is node:
                        del self._nodes[node_key]
                self._dispose(node_key, node)
                continue
            with self._state_lock:
                still_ours = self._nodes.get(node_key) is node
            if not still_ours:
                node.stop()

    def _merged_cfg(self, node_key: str) -> dict:
        return {**self._plugin_cfg, **self._instance_configs.get(node_key, {})}

    def _create_node(
        self, node_key: str, input_topic: str, engine: _FaceEngine
    ) -> _FaceNode:
        cfg = self._merged_cfg(node_key)
        return _FaceNode(
            input_topic,
            engine,
            cfg,
            node_suffix=node_key.replace("/", "_").replace("-", "_"),
            detect_interval_s=detect_interval(cfg),
        )

    def _dispose(self, node_key: str, node: _FaceNode) -> None:
        try:
            node.retire()
        finally:
            dispose_node(self._executor, node, label=f"face/{node_key}")
        log.info("[face] node disposed: %s", node_key)

    def _instance_state_locked(self, node_key: str) -> str:
        node = self._nodes.get(node_key)
        if node is not None:
            return node.state
        if node_key in self._pending_starts:
            return "error" if self._engine_state == "error" else "loading"
        return "idle"

    # ── MCP dispatch ──────────────────────────────────────────────────────

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == self.PREFIX else name
        instance_id = args.get("instance_id", "")

        if action == "info":
            return self._do_info(instance_id, args.get("input_topic", ""))
        if action == "start":
            return self._do_start(instance_id, args)
        if action == "stop":
            return self._do_stop(instance_id)
        if action == "config":
            return self._do_config(instance_id, args)
        if action == "register_by_photo":
            return self._do_register_by_photo(args)
        if action == "register_by_url":
            return self._do_register_by_photo(args)
        if action == "register_by_stream":
            return self._do_register_by_stream(instance_id, args)
        if action == "register_by_corpus":
            return self._do_register_by_corpus(args)
        if action == "recognize_by_photo":
            return self._do_recognize_by_photo(args)
        if action == "recognize_by_stream":
            return self._do_recognize_by_stream(instance_id, args)
        if action == "list_persons":
            return self._do_list_persons(args)
        if action == "get_person":
            return self._do_get_person(args)
        if action == "update_person":
            return self._do_update_person(args)
        if action == "forget":
            return self._do_forget(args)
        if action == "list_visits":
            return self._do_list_visits(args)
        return None

    _DESC = "Face recognition — identifies registered people in the camera feed"

    def _desc_locked(self, state: str) -> str:
        if state == "loading":
            return "Loading face detection and recognition models..."
        if state == "error" and self._load_error:
            return f"Model load failed: {self._load_error}"
        return self._DESC

    # ── info / start / stop / config ──────────────────────────────────────

    def _do_info(self, instance_id: str, input_topic: str) -> dict:
        base = {"name": "FaceRecognition", "manufacture": "Embodied",
                "model": "scrfd+arcface"}
        with self._state_lock:
            engine = self._engine
            if instance_id:
                node = self._nodes.get(instance_id)
                topic = (
                    node._input_topic if node is not None
                    else self._pending_starts.get(instance_id, input_topic)
                )
                out = _face_output_topic(topic) if topic else ""
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
                return self._with_db_stats(result, engine)

            keys = list(self._nodes) + [
                key for key in self._pending_starts if key not in self._nodes
            ]
            instances = {key: {"state": self._instance_state_locked(key)} for key in keys}
            topics_in, topics_out = [], []
            for key in keys:
                node = self._nodes.get(key)
                topic = node._input_topic if node else self._pending_starts[key]
                topics_in.append({"topic": topic, "format": "image/jpeg", "desc": ""})
                topics_out.append({"topic": _face_output_topic(topic), "format": "data/json", "desc": ""})
            states = {entry["state"] for entry in instances.values()}
            if "loading" in states or self._engine_state == "loading":
                state = "loading"
            elif "running" in states:
                state = "running"
            elif "error" in states or self._engine_state == "error":
                state = "error"
            else:
                state = "idle"
            if not keys and input_topic:
                topics_in = [{"topic": input_topic, "format": "image/jpeg", "desc": ""}]
                topics_out = [{"topic": _face_output_topic(input_topic), "format": "data/json", "desc": ""}]
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
            return self._with_db_stats(result, engine)

    def _with_db_stats(self, result: dict, engine: _FaceEngine | None) -> dict:
        """Attach roster counts. Never fails info — it is the diagnostic path."""
        if engine is None:
            return result
        try:
            result["database"] = engine.db.stats()
        except Exception:  # noqa: BLE001
            log.warning("[face] could not read database stats", exc_info=True)
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
                retired = self._nodes.pop(node_key)
        if retired is not None:
            self._dispose(node_key, retired)

        with self._state_lock:
            existing = self._nodes.get(node_key)
            if existing is not None:
                start_node = existing            # idempotent re-start
            else:
                start_node = None
                # Claim the key before leaving the lock so a concurrent stop
                # always finds the instance in _pending_starts or _nodes, never
                # in an invisible in-between state.
                self._pending_starts[node_key] = input_topic
                if self._engine_state != "ready":
                    if self._engine_state in ("idle", "error"):
                        self._spawn_loader_locked()
                    return {
                        "state": "loading",
                        "input": input_topic,
                        "output": _face_output_topic(input_topic),
                    }
            engine = self._engine
            generation = self._load_generation

        if start_node is not None:
            return start_node.start()

        node = self._create_node(node_key, input_topic, engine)
        with self._state_lock:
            claimed = self._pending_starts.get(node_key) == input_topic
            current = self._nodes.get(node_key)
            fresh = generation == self._load_generation
            registered = False
            if claimed and current is None and fresh:
                try:
                    self._executor.add_node(node)
                except Exception as error:  # noqa: BLE001
                    log.error("[face] failed to register instance %r: %s",
                              node_key, escape_log_text(error))
                    del self._pending_starts[node_key]
                else:
                    self._nodes[node_key] = node
                    del self._pending_starts[node_key]
                    registered = True
            elif claimed and not fresh:
                # A config change invalidated the engine mid-start; leave the
                # claim so the loader it spawned brings this instance up.
                pass
            elif claimed:
                del self._pending_starts[node_key]
        if not registered:
            try:
                node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
            if current is not None:
                return current.start()
            if claimed and not fresh:
                return {"state": "loading", "input": input_topic,
                        "output": _face_output_topic(input_topic)}
            return {"state": "idle", "input": input_topic,
                    "output": _face_output_topic(input_topic)}
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
            node.stop()
            return {"state": "idle", "input": input_topic,
                    "output": _face_output_topic(input_topic)}
        return result

    def _do_stop(self, instance_id: str) -> dict:
        to_dispose: list[tuple[str, _FaceNode]] = []
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

    # Per-instance because one camera may want a different cadence from
    # another; everything else (thresholds, the model, the database) is
    # shared by every instance of the plugin.
    _INSTANCE_SCOPED = ("detect_fps", "min_interval_ms")

    def _do_config(self, instance_id: str, args: dict) -> dict:
        cfg = {
            key: value for key, value in args.items()
            if key not in ("action", "instance_id")
            and value is not None and value != ""
        }

        if instance_id:
            shared = set(cfg) - set(self._INSTANCE_SCOPED)
            if shared:
                raise ValueError(
                    "face recognition settings are shared: "
                    + ", ".join(sorted(shared))
                )
            with self._state_lock:
                previous = self._instance_configs.get(instance_id, {})
                self._instance_configs[instance_id] = {**previous, **cfg}
                node = self._nodes.get(instance_id)
                merged = self._merged_cfg(instance_id)
            if node is not None:
                # detect_fps is applied in place; no reason to retire a
                # live subscription for a frame-rate change.
                node.apply_config(merged)
            return {"status": "configured", "instance_id": instance_id}

        evicted = 0
        with self._state_lock:
            updated = {**self._plugin_cfg, **cfg}
            rebuild = _engine_signature(updated) != _engine_signature(self._plugin_cfg)
            capacity_changed = (
                int(updated.get("unknown_capacity", DEFAULT_UNKNOWN_CAPACITY))
                != int(self._plugin_cfg.get("unknown_capacity", DEFAULT_UNKNOWN_CAPACITY))
            )
            self._plugin_cfg = updated
            engine = self._engine
            if not rebuild:
                for node_key, node in self._nodes.items():
                    node.apply_config({
                        **updated, **self._instance_configs.get(node_key, {})
                    })
                nodes_live = self._engine is not None
            else:
                self._load_generation += 1
                stale_engine = self._engine
                self._engine = None
                disposed = list(self._nodes.items())
                self._nodes = {}
                if self._pending_starts:
                    self._spawn_loader_locked()
                else:
                    self._engine_state = "idle"
                    self._load_error = None

        if not rebuild:
            if capacity_changed and engine is not None:
                evicted = engine.db.set_unknown_capacity(
                    int(self._plugin_cfg.get("unknown_capacity",
                                             DEFAULT_UNKNOWN_CAPACITY))
                )
            result = {"status": "configured", "engine_loaded": nodes_live,
                      "reused": nodes_live}
            if capacity_changed:
                result["unknown_evicted"] = evicted
                result["unknown_capacity"] = int(
                    self._plugin_cfg.get("unknown_capacity",
                                         DEFAULT_UNKNOWN_CAPACITY)
                )
            return result

        for node_key, node in disposed:
            self._dispose(node_key, node)
        if stale_engine is not None:
            _close_quietly(stale_engine)
        return {"status": "configured", "engine_loaded": False, "reused": False}

    # ── engine access for the action paths ────────────────────────────────

    def _require_engine(self) -> _FaceEngine:
        """Return a ready engine, loading it on demand.

        The registration and roster actions are useful without any instance
        running — an operator enrols people from photos before pointing a
        camera anywhere — so they trigger the same single-flight load a `start`
        would, and wait for it rather than reporting `loading`. A tools/call
        already has its own thread (ThreadingHTTPServer), so blocking here
        blocks nothing else.
        """
        with self._state_lock:
            if self._engine is not None:
                return self._engine
            if self._engine_state in ("idle", "error"):
                self._spawn_loader_locked()
            generation = self._load_generation

        deadline = time.monotonic() + float(
            self._plugin_cfg.get("load_timeout_s", 180.0)
        )
        while time.monotonic() < deadline:
            time.sleep(0.2)
            with self._state_lock:
                if generation != self._load_generation:
                    generation = self._load_generation
                    continue
                if self._engine is not None:
                    return self._engine
                if self._engine_state == "error":
                    raise RuntimeError(
                        self._load_error or "face engine failed to load"
                    )
        raise RuntimeError("face engine is still loading; try again shortly")

    # ── registration ──────────────────────────────────────────────────────

    def _analyze_subject(
        self, engine: _FaceEngine, image_bytes: bytes, gates: dict
    ) -> tuple[np.ndarray | None, dict | None]:
        """One image → the subject's embedding, or the failure record."""
        image = engine.analyzer.decode_image(
            image_bytes, **_decode_options(self._plugin_cfg)
        )
        if image is None:
            return None, {
                "ok": False, "reason": REASON_BAD_INPUT,
                "detail": (
                    "image could not be decoded (cv2 and Pillow both refused it; "
                    "HEIC/HEIF is the common format neither reads), or it exceeds "
                    "max_image_pixels"
                ),
            }
        shape = image.shape[:2]
        faces = engine.analyzer.detect(image, max_faces=gates["max_faces"])
        for face in faces:
            engine.analyzer.prepare(image, face)
        subject, failure = select_subject(faces, shape, gates)
        if subject is None:
            return None, failure
        return engine.analyzer.embed(subject.aligned), None

    def _commit_enrolment(
        self,
        engine: _FaceEngine,
        embeddings: list[np.ndarray],
        name: str,
        profile: Any,
        person_id: str | None,
        gates: dict,
    ) -> dict:
        """Attach embeddings to the right identity, creating one if needed."""
        database = engine.db

        if person_id:
            try:
                existing = database.get_person(person_id)
            except KeyError:
                return {
                    "ok": False, "reason": REASON_BAD_INPUT,
                    "detail": f"no such person: {person_id!r}",
                }
            was_unknown = not existing["named"]
            record = database.add_samples(person_id, embeddings)
            if name or profile is not None:
                record = database.update_person(
                    person_id, name=name or None, profile=profile
                )
            return {
                "ok": True,
                "person_id": person_id,
                "name": record["name"],
                "profile": record["profile"],
                "samples": record["samples"],
                "promoted": was_unknown and bool(name),
            }

        # No id given: does this face already exist under some identity?
        matched, score = database.match(embeddings[0], gates["match_threshold"])
        if matched is not None:
            record = database.add_samples(matched, embeddings)
            promoted = False
            if is_unknown_id(matched) or not record["named"]:
                # The face was already being tracked anonymously; naming it now
                # keeps that id, so earlier sightings stay attributable.
                record = database.update_person(matched, name=name, profile=profile)
                promoted = True
            elif profile is not None:
                record = database.update_person(matched, profile=profile)
            return {
                "ok": True,
                "person_id": matched,
                "name": record["name"],
                "profile": record["profile"],
                "samples": record["samples"],
                "merged": True,
                "promoted": promoted,
                "score_to_existing": round(float(score), 4),
            }

        record = database.add(name, embeddings, named=True, profile=profile)
        return {
            "ok": True,
            "person_id": record["id"],
            "name": record["name"],
            "profile": record["profile"],
            "samples": record["samples"],
            "merged": False,
            "promoted": False,
        }

    def _do_register_by_photo(self, args: dict) -> dict:
        engine = self._require_engine()
        cfg = dict(self._plugin_cfg)
        gates = _gates(cfg)
        try:
            data, source = _load_image_bytes(args, cfg)
        except _BadInput as error:
            return error.as_result()

        embedding, failure = self._analyze_subject(engine, data, gates)
        if embedding is None:
            return {**failure, "source": source}
        # No person_id input: which identity a photo belongs to is decided by
        # matching, not by the caller. A face that matches an existing person
        # becomes another sample of them; one that matches an unknown-N promotes
        # that entry in place. Naming an identity after the fact is
        # `update_person`, and grouping several photos under one person is the
        # batch manifest's `person` key.
        result = self._commit_enrolment(
            engine, [embedding], str(args.get("name") or ""),
            args.get("profile"), None, gates,
        )
        if result.get("ok"):
            log.info("[face] registered %s from %s: %s", result["person_id"],
                     escape_log_text(source), escape_log_text(result["name"]))
        return {**result, "source": source}

    def _do_register_by_stream(self, instance_id: str, args: dict) -> dict:
        cfg = dict(self._plugin_cfg)
        gates = _gates(cfg)

        node, failure = self._pick_instance(instance_id)
        if node is None:
            return failure

        window = min(
            float(args.get("window_s") or cfg.get(
                "enroll_window_s", DEFAULT_ENROLL_WINDOW_S)),
            float(cfg.get("enroll_window_s", DEFAULT_ENROLL_WINDOW_S)),
        )
        frames = node.recent_frames(window)
        if not frames:
            return {
                "ok": False, "reason": REASON_NO_FRAMES,
                "detail": f"no frames received in the last {window:.1f}s",
                "instance_id": instance_id or node._input_topic,
            }

        engine = self._require_engine()
        # Analyse newest-first and cap the count: the window can hold 60 frames
        # and running the full pair of models over all of them would take
        # seconds for no extra accuracy.
        budget = max(1, int(cfg.get("enroll_max_analyzed", DEFAULT_ENROLL_MAX_ANALYZED)))
        selected = list(reversed(frames))[:budget]

        embeddings: list[np.ndarray] = []
        failures: list[dict] = []
        for image_bytes, _ts in selected:
            embedding, failure = self._analyze_subject(engine, image_bytes, gates)
            if embedding is None:
                failures.append(failure)
            else:
                embeddings.append(embedding)

        if not embeddings:
            return {
                **_worst_reason(failures),
                "instance_id": instance_id or node._input_topic,
                "window_s": round(window, 2),
            }

        # Every accepted frame must show the *same* person. Two people taking
        # turns being the dominant face would otherwise be enrolled as one
        # identity whose embeddings match neither of them well.
        anchor = embeddings[0]
        agreeing = [
            embedding for embedding in embeddings
            if float(np.dot(anchor, embedding)) >= gates["match_threshold"]
        ]
        if len(agreeing) * 2 < len(embeddings):
            return {
                "ok": False,
                "reason": REASON_AMBIGUOUS,
                "detail": (
                    f"the last {window:.1f}s did not show one stable subject "
                    f"({len(agreeing)} of {len(embeddings)} usable frames agree). "
                    "Have a single person hold still in front of the camera."
                ),
                "frames_examined": len(selected),
                "instance_id": instance_id or node._input_topic,
            }

        result = self._commit_enrolment(
            engine, agreeing, str(args.get("name") or ""),
            args.get("profile"), None, gates,
        )
        if result.get("ok"):
            log.info("[face] registered %s from the live stream (%d/%d frames): %s",
                     result["person_id"], len(agreeing), len(selected),
                     escape_log_text(result["name"]))
        return {
            **result,
            "instance_id": instance_id or node._input_topic,
            "frames_used": len(agreeing),
            "frames_examined": len(selected),
            "window_s": round(window, 2),
        }

    def _do_register_by_corpus(self, args: dict) -> dict:
        package = str(args.get("package") or "").strip()
        if not package:
            return {"ok": False, "reason": REASON_BAD_INPUT,
                    "detail": "package is required"}

        engine = self._require_engine()
        cfg = dict(self._plugin_cfg)
        gates = _gates(cfg)
        max_batch = int(cfg.get("max_batch", DEFAULT_MAX_BATCH))

        with tempfile.TemporaryDirectory(prefix="face-batch-") as staging:
            try:
                directory = _extract_package(package, cfg, staging)
                manifest = _read_manifest(directory)
            except _BadInput as error:
                return error.as_result()

            images: list[str] = []
            for root, _dirs, files in os.walk(directory):
                for filename in sorted(files):
                    if filename.lower().endswith(_IMAGE_SUFFIXES):
                        images.append(os.path.join(root, filename))
            images.sort()

            if not images:
                return {"ok": False, "reason": REASON_BAD_INPUT,
                        "detail": "package contains no images", "source": package}
            if len(images) > max_batch:
                return {
                    "ok": False, "reason": REASON_BAD_INPUT,
                    "detail": (
                        f"package has {len(images)} images, over the max_batch "
                        f"limit of {max_batch}. Split it, or raise max_batch in "
                        "config.yaml."
                    ),
                    "source": package,
                }

            results: list[dict] = []
            # person key → the id its first successful photo created, so several
            # photos of one person become several samples of one identity.
            groups: dict[str, str] = {}
            for image_path in images:
                relative = os.path.relpath(image_path, directory)
                entry = manifest.get(relative) or manifest.get(
                    os.path.basename(image_path)
                ) or {}
                if entry:
                    # `name` is the structured label; a manifest that still puts
                    # a plain string in `profile` is read as the name.
                    raw_profile = entry.get("profile")
                    name = entry.get("name")
                    if name is None and isinstance(raw_profile, str):
                        name, raw_profile = raw_profile, None
                    name = str(name or "")
                    profile = raw_profile if isinstance(raw_profile, dict) else None
                else:
                    name, sidecar_profile = _sidecar_identity(image_path)
                    profile = sidecar_profile or None
                group = str(entry.get("person") or entry.get("id") or "").strip()
                person_id = groups.get(group) if group else None

                try:
                    with open(image_path, "rb") as handle:
                        data = handle.read()
                except OSError as error:
                    results.append({"file": relative, "ok": False,
                                    "reason": REASON_BAD_INPUT,
                                    "detail": f"cannot read: {error}"})
                    continue

                embedding, failure = self._analyze_subject(engine, data, gates)
                if embedding is None:
                    results.append({"file": relative, "name": name, **failure})
                    continue
                try:
                    outcome = self._commit_enrolment(
                        engine, [embedding], name, profile, person_id, gates
                    )
                except Exception as error:  # noqa: BLE001 - one bad photo must
                    # not abandon the rest of a 200-image batch half-registered
                    log.error("[face] batch entry %s failed: %s",
                              escape_log_text(relative), escape_log_text(error))
                    results.append({"file": relative, "ok": False,
                                    "reason": REASON_BAD_INPUT,
                                    "detail": str(error)})
                    continue
                if outcome.get("ok") and group:
                    groups.setdefault(group, outcome["person_id"])
                results.append({"file": relative, **outcome})

        registered = sum(1 for item in results if item.get("ok"))
        failed = len(results) - registered
        log.info("[face] batch %s: %d registered, %d failed",
                 escape_log_text(package), registered, failed)
        return {
            "ok": True,
            "source": package,
            "total": len(results),
            "registered": registered,
            "failed": failed,
            "results": results,
        }

    def _pick_instance(self, instance_id: str):
        """Resolve which running instance a stream action should read from.

        Returns `(node, None)` or `(None, failure)`. With exactly one instance
        the id is optional — the common case is one camera — but with several
        it is required rather than guessed, because reading the wrong camera
        would silently answer about the wrong room.
        """
        with self._state_lock:
            if instance_id:
                node = self._nodes.get(instance_id)
            elif len(self._nodes) == 1:
                node = next(iter(self._nodes.values()))
            else:
                node = None
                if len(self._nodes) > 1:
                    return None, {
                        "ok": False, "reason": REASON_BAD_INPUT,
                        "detail": (
                            f"{len(self._nodes)} instances are running; pass "
                            "instance_id to say which camera to use"
                        ),
                        "instances": sorted(self._nodes),
                    }
        if node is None:
            return None, {
                "ok": False, "reason": REASON_NO_FRAMES,
                "detail": (
                    "no running instance to read from — start the card on a "
                    "camera topic first"
                ),
            }
        return node, None

    # ── recognition on demand (read-only) ────────────────────────────────

    def _identify(
        self, engine: _FaceEngine, image_bytes: bytes, gates: dict
    ) -> dict | list[dict]:
        """Every face in one image, matched against the database.

        **Read-only.** Unlike the continuous stream, this neither auto-enrols a
        stranger as `unknown-N` nor records a sighting: "who is this" is a
        question, and answering it should not mutate the roster or the visit
        log. It also does not apply `subject_dominance` — that gate exists
        because *enrolment* must resolve to exactly one person, whereas a query
        can simply report everyone it sees.
        """
        image = engine.analyzer.decode_image(
            image_bytes, **_decode_options(self._plugin_cfg)
        )
        if image is None:
            return {
                "ok": False, "reason": REASON_BAD_INPUT,
                "detail": (
                    "image could not be decoded (cv2 and Pillow both refused it; "
                    "HEIC/HEIF is the common format neither reads), or it exceeds "
                    "max_image_pixels"
                ),
            }
        faces = engine.analyzer.detect(image, max_faces=gates["max_faces"])
        results = []
        for face in faces:
            engine.analyzer.prepare(image, face)
            entry = {
                "bbox": face.bbox_xywh(),
                "det_score": round(face.det_score, 4),
                "blur": round(face.blur, 2),
                "min_side_px": int(face.min_side),
            }
            if not (
                face.det_score >= gates["det_thresh"]
                and face.min_side >= gates["min_face_px"]
                and face.blur >= gates["blur_min"]
            ):
                entry.update({
                    "person_id": None, "name": "", "known": False,
                    "quality": "low", "reason": REASON_LOW_QUALITY,
                })
                results.append(entry)
                continue
            embedding = engine.analyzer.embed(face.aligned)
            person_id, score = engine.db.match(
                embedding, gates["match_threshold"]
            )
            if person_id is None:
                # `best_score` is what an operator needs to decide whether
                # match_threshold is too strict, so report it rather than just
                # saying no.
                entry.update({
                    "person_id": None, "name": "", "known": False,
                    "quality": "ok", "best_score": round(float(score), 4),
                })
            else:
                record = engine.db.get_person(person_id)
                entry.update({
                    "person_id": person_id,
                    "name": record["name"],
                    "profile": record["profile"],
                    "known": record["named"],
                    "score": round(float(score), 4),
                    "quality": "ok",
                })
            results.append(entry)
        return results

    def _do_recognize_by_photo(self, args: dict) -> dict:
        engine = self._require_engine()
        cfg = dict(self._plugin_cfg)
        gates = _gates(cfg)
        try:
            data, source = _load_image_bytes(args, cfg)
        except _BadInput as error:
            return error.as_result()
        outcome = self._identify(engine, data, gates)
        if isinstance(outcome, dict):
            return {**outcome, "source": source}
        return {
            "ok": True,
            "source": source,
            "count": len(outcome),
            "faces": outcome,
        }

    def _do_recognize_by_stream(self, instance_id: str, args: dict) -> dict:
        cfg = dict(self._plugin_cfg)
        gates = _gates(cfg)
        node, failure = self._pick_instance(instance_id)
        if node is None:
            return failure

        # Default 1 s, not the 3 s enrolment window: this answers "who is in
        # front of me now". Several frames rather than one because a single
        # blurred frame would otherwise report nobody; each person is reported
        # once, at their best score across the window.
        window = min(
            float(args.get("window_s") or 1.0),
            float(cfg.get("enroll_window_s", DEFAULT_ENROLL_WINDOW_S)),
        )
        frames = node.recent_frames(window)
        if not frames:
            return {
                "ok": False, "reason": REASON_NO_FRAMES,
                "detail": f"no frames received in the last {window:.1f}s",
                "instance_id": instance_id or node._input_topic,
            }
        engine = self._require_engine()
        budget = max(1, int(cfg.get("enroll_max_analyzed", DEFAULT_ENROLL_MAX_ANALYZED)))
        selected = list(reversed(frames))[:budget]

        best: dict[str, dict] = {}
        unidentified: list[dict] = []
        for image_bytes, _ts in selected:
            outcome = self._identify(engine, image_bytes, gates)
            if isinstance(outcome, dict):
                continue                     # undecodable frame; try the next
            for entry in outcome:
                person_id = entry.get("person_id")
                if person_id is None:
                    unidentified.append(entry)
                    continue
                previous = best.get(person_id)
                if previous is None or entry["score"] > previous["score"]:
                    best[person_id] = entry

        faces = sorted(best.values(), key=lambda e: e["score"], reverse=True)
        if not faces and unidentified:
            # Nobody recognised, but there were faces — report the best-looking
            # one so the answer is "someone I do not know" rather than "nobody".
            faces = [max(unidentified, key=lambda e: e.get("best_score", -2.0))]
        return {
            "ok": True,
            "instance_id": instance_id or node._input_topic,
            "count": len(faces),
            "faces": faces,
            "frames_examined": len(selected),
            "window_s": round(window, 2),
        }

    # ── roster CRUD ───────────────────────────────────────────────────────

    def _do_list_persons(self, args: dict) -> dict:
        engine = self._require_engine()
        return {
            "ok": True,
            **engine.db.list_persons(
                named=str(args.get("named") or "all"),
                query=str(args.get("query") or ""),
                limit=int(args.get("limit") or 100),
                offset=int(args.get("offset") or 0),
            ),
        }

    def _do_get_person(self, args: dict) -> dict:
        person_id = str(args.get("person_id") or "").strip()
        if not person_id:
            raise ValueError("person_id is required")
        engine = self._require_engine()
        try:
            return {"ok": True, "person": engine.db.get_person(person_id)}
        except KeyError:
            return {"ok": False, "reason": REASON_BAD_INPUT,
                    "detail": f"no such person: {person_id!r}"}

    def _do_update_person(self, args: dict) -> dict:
        person_id = str(args.get("person_id") or "").strip()
        if not person_id:
            raise ValueError("person_id is required")
        engine = self._require_engine()
        try:
            record = engine.db.update_person(
                person_id,
                name=args.get("name"),
                profile=args.get("profile"),
                profile_delete=args.get("profile_delete") or [],
                merge=bool(args.get("merge", True)),
            )
        except KeyError:
            return {"ok": False, "reason": REASON_BAD_INPUT,
                    "detail": f"no such person: {person_id!r}"}
        except ValueError as error:
            return {"ok": False, "reason": REASON_BAD_INPUT, "detail": str(error)}
        return {"ok": True, "person": record}

    def _do_list_visits(self, args: dict) -> dict:
        """访问记录查询 — who was around, and when."""
        engine = self._require_engine()
        try:
            return {
                "ok": True,
                **engine.db.list_visits(
                    person_id=str(args.get("person_id") or ""),
                    since=args.get("since"),
                    until=args.get("until"),
                    limit=int(args.get("limit") or 100),
                    offset=int(args.get("offset") or 0),
                ),
            }
        except ValueError as error:      # unparseable since/until
            return {"ok": False, "reason": REASON_BAD_INPUT, "detail": str(error)}

    def _do_forget(self, args: dict) -> dict:
        """Delete one person, a list of them, or every anonymous entry.

        The list form is not just ergonomics: `forget_many` commits once, where
        N single calls rewrote the whole database N times.
        """
        engine = self._require_engine()
        scope = str(args.get("named") or "").strip().lower()

        raw_ids = args.get("person_ids")
        if isinstance(raw_ids, str):
            # An LLM (or a form field) will send "p-1, p-2" as one string.
            raw_ids = [part for part in re.split(r"[,\s]+", raw_ids) if part]
        ids = [str(pid).strip() for pid in (raw_ids or []) if str(pid).strip()]

        single = str(args.get("person_id") or "").strip()
        if single:
            ids.append(single)

        if scope == "unknown" and not ids:
            removed = engine.db.forget_unknowns()
            return {"ok": True, "forgotten": removed, "scope": "unknown"}

        if not ids:
            raise ValueError(
                "person_id, person_ids or named='unknown' is required"
            )

        outcome = engine.db.forget_many(ids)
        forgotten, missing = outcome["forgotten"], outcome["missing"]
        result = {
            # Partially-successful is the normal case with a list, so `ok`
            # reports "the request was processed", and the two lists say what
            # actually happened to each id.
            "ok": bool(forgotten) or not missing,
            "forgotten": len(forgotten),
            "person_ids": forgotten,
        }
        if missing:
            result["missing"] = missing
            result["detail"] = (
                f"{len(missing)} id(s) did not exist: " + ", ".join(missing)
            )
            if not forgotten:
                result["reason"] = REASON_BAD_INPUT
        return result


__all__ = [
    "DEFAULT_MATCH_THRESHOLD",
    "REASON_AMBIGUOUS",
    "REASON_BAD_INPUT",
    "REASON_LOW_QUALITY",
    "REASON_NO_FACE",
    "REASON_NO_FRAMES",
    "FaceRecognitionPlugin",
    "TOOLS",
    "select_subject",
]
