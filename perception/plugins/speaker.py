#!/usr/bin/env python3
"""
plugins/speaker.py — SpeakerRecognitionPlugin: 声纹注册与识别。

Lifecycle, locking and the start/stop/config state machine are `plugins/face.py`'s,
which took them from `plugins/ocr.py` — the reference implementation of the rules in
`perception/README.md` § "Plugin Concurrency". The parts that look redundant
(claiming a node key before leaving the lock, registering with the executor *before*
`start()`, bumping a generation on a model-affecting config change) are each there
because omitting them orphaned a live node in production.

## How this is *not* face recognition

**The identity reaches the LLM through ASR, not through this card's topic.** The
embedding is computed on the VAD segment ASR already cut — the same PCM buffer, the
same tuple, in a thread beside `transcribe()` — so there is no timestamp join and no
alignment error to get wrong. `identify_pcm()` is that entry point and it needs only
the engine, not a running node, so it works whether or not this card is started.
This card's own `start` subscribes to an audio topic with its own VAD, for the
canvas visualisation and for the case where ASR is not running at all.

**No file or URL enrolment.** Not an omission — voiceprints carry channel. The
microphone, the sampling chain, the room and the distance are all baked into the
embedding, so a phone recording enrolled against a far-field array mic matches
systematically low, and no threshold recovers it (lower it enough to admit the phone
and it admits strangers too). Photos have no equivalent of this, which is why face's
`register_by_photo` / `_by_url` / `_by_corpus` are not mirrored here. Having no such
entry point is the design.

**Enrolment is retroactive.** The dominant path is not "upload a sample of 小王", it
is: the LLM has just seen `speaker_id: p-7, speaker_name: ""` in an ASR payload, the
person says "我是小王", and that existing identity gets a name. So `name_speaker`
takes an optional `speaker_id` and defaults to *whoever spoke most recently*, which
`_RecentRing` remembers — including the embedding, so an identity can be created on
the spot when `auto_enroll` is off.

**`get_speaker` returns playable audio.** A face identity is built from a photo you
already recognised; a voiceprint identity is built from an anonymous utterance, and
the only way to answer "who is this" is to listen. That is this plugin's equivalent
of face's corpus flow, and it needs no file entry point.

## Input is quantised and long clips are checked for a speaker change

Both in `plugins/speaker_runtime.py`, with the measurements that forced them. The
short version: unquantised input lengths make ONNX Runtime's arena grow without
bound, and a VAD segment is not guaranteed to hold one voice — measured p05 0.197
cosine between a long segment and its own leading window on real office audio.
The check compares the clip's two **ends**, so it sees past the truncation cap;
a clip that fails it reports `multi_speaker` and **withholds the identity** rather
than publishing a blend of two people as one person.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import wave
from collections import deque
from typing import Any

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String

from utils.log_sampling import escape_log_text
from utils.ros_lifecycle import dispose_node

from plugins.identity_db import (
    DEFAULT_MAX_SAMPLES_PER_PERSON,
    DEFAULT_UNKNOWN_CAPACITY,
    DEFAULT_VISIT_CHECKPOINT_S,
    DEFAULT_VISIT_GAP_S,
    DEFAULT_VISIT_LOG_MAX,
    IdentityDB,
    parse_time,
)
from plugins.speaker_runtime import (
    DEFAULT_SPEAKER_MODEL,
    DEFAULT_SPEAKER_MODEL_DIR,
    MAX_S,
    SAMPLE_RATE,
    SPEAKER_MODELS,
    SpeakerEmbedder,
)

log = logging.getLogger(__name__)

DEFAULT_DB_DIR = "/models/speaker_db"
DEFAULT_SAMPLE_DIR = "/models/speaker_db/samples"

# Cosine threshold between L2-normalised CAM++ embeddings. 0.6 is sherpa's own
# example default and a STARTING POINT, not a measured value — face's 0.35 is
# ArcFace's scale and does not transfer. Raise it if the robot confuses people,
# lower it if it fails to recognise them, and record what you saw. Phase 0 could
# not pin this down: the VAD segments on hand are unlabelled, so only the dynamic
# range was visible (unlabelled pairwise cosine p50 0.28-0.48, p95 0.79-0.92).
DEFAULT_MATCH_THRESHOLD = 0.6

# Below this much speech an embedding is not worth matching. Speaker verification
# degrades sharply under ~1.5 s, while ASR's VAD emits anything over 500 ms
# (plugins/asr.py:1106) — so "嗯", "好", "停" would otherwise produce a garbage
# embedding and a confident-looking wrong identity. Also a STARTING POINT.
DEFAULT_MIN_SPEECH_S = 1.5

# Above this, check the clip for a speaker change (speaker_runtime.embed_windowed).
DEFAULT_SPLIT_ABOVE_S = 4.0
# Cosine below which a long clip is called multi-speaker. Starting point.
DEFAULT_COHERENCE_MIN = 0.8

# Off by default, and this is not timidity. perception has no acoustic echo
# cancellation anywhere, so the robot's own TTS reaches the microphone: ASR turns
# that into junk text that gets filtered, but speaker recognition would enrol the
# robot's own voice as a person and then "recognise" it every time it speaks. TVs,
# speakers and corridor passers-by do the same. Turn it on once the review flow
# (list_speakers → get_speaker → listen → name_speaker, plus forget named=unknown)
# has been exercised on a real robot and the contamination rate is known.
DEFAULT_AUTO_ENROLL = False

# On by default, and it is the cheapest accuracy win available. A named person's
# voiceprint drifts with distance, a cold, emotion and background noise far more
# than a face does with lighting, and IdentityDB scores a person by their *best*
# sample rather than a centroid — so extra samples strictly help. Bounded by
# max_samples_per_person. Runs off the critical path; see _maybe_add_sample.
DEFAULT_AUTO_ADD_SAMPLES = True
# A sample is only worth keeping if it is clearly the same person and clearly
# long enough; otherwise auto-accumulation slowly poisons the identity.
DEFAULT_SAMPLE_MIN_SIMILARITY = 0.75
DEFAULT_SAMPLE_MIN_SPEECH_S = 2.0

DEFAULT_RECENT_RING = 32
DEFAULT_RECENT_WINDOW_S = 120.0

# Standalone-mode VAD (this card started on an audio topic directly). ASR's
# settings are deliberately not reused: when ASR feeds us we inherit its
# segmentation whether we like it or not, and when it does not, a longer silence
# gap produces fewer, longer, more usable segments for a voiceprint.
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_VAD_SILENCE_MS = 600
DEFAULT_VAD_MAX_SPEECH_S = 20.0

AUDIO_FORMAT = "audio/pcm-16k"
_AUDIO_FORMATS = frozenset((AUDIO_FORMAT, "pcm_16k_16bit_mono"))

_AUDIO_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST, depth=50,
    durability=DurabilityPolicy.VOLATILE,
)
_RESULT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST, depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

# ── failure reasons ───────────────────────────────────────────────────────────

REASON_TOO_SHORT = "too_short"
REASON_MULTI_SPEAKER = "multi_speaker"
REASON_NO_RECENT = "no_recent_speaker"
REASON_BAD_INPUT = "bad_input"
# 「库里一个人都没有」和「比过了但都不够像」必须分开报。两者的处置相反：前者要去
# 注册一个人，后者要么调低 match_threshold 要么这个人确实不认识。合成一个数字
# （之前是都报 similarity 0.0）会把人送去调阈值，而阈值跟空库毫无关系。
REASON_NO_SPEAKERS = "no_speakers_enrolled"
REASON_NO_MATCH = "no_match"


TOOLS = [
    {
        "name": "speaker_recognition",
        "type": "processor",
        "multiInstance": True,
        "description": (
            "Speaker recognition — tell who is talking from their voice, and give "
            "a name to a voice the robot has already heard"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "start", "stop", "info", "config",
                        "name_speaker",
                        "list_speakers", "get_speaker", "forget", "list_heard",
                    ],
                    "description": "Action to perform",
                },
                "input_topic": {"type": "string", "description": "ROS2 音频 topic（如 /hostname/mic/audio，action=start 时必填）。只有独立模式需要——身份随 ASR 的输出一起走，不需要启动这张卡"},
                "name":       {"type": "string", "description": "姓名（结构化），如 \"小王\"。有 name 才算已注册；每次识别都会随 id 一起输出"},
                "profile":    {"type": "object", "description": "非结构化画像对象，如 {\"team\":\"运营部\",\"note\":\"说话很快\"}。键名自定；传字符串会被存成 {\"note\":\"...\"}"},
                "profile_delete": {"type": "array", "items": {"type": "string"}, "description": "要删除的 profile 键名列表"},
                "merge":      {"type": "boolean", "description": "profile 合并进已有对象（默认 true），false 为整体替换"},
                "speaker_id": {"type": "string", "description": "已存在的声纹 id，形如 p-3。**留空表示「最近说话的那个人」** —— 这是主用法：刚听到一句话、对方说了自己是谁，直接命名，不需要先查 id"},
                "speaker_ids": {"type": "array", "items": {"type": "string"}, "description": "批量删除的 id 列表；也接受逗号或空格分隔的字符串。返回 forgotten 与 missing，部分成功是正常结果"},
                "named":      {"type": "string", "enum": ["all", "named", "unknown"], "description": "过滤范围，默认 all。action=forget 时 'unknown' 表示清空所有未命名声纹 —— 回收被电视/机器人自己的声音污染的条目就用它"},
                "query":      {"type": "string", "description": "在 id、name、profile 上做子串匹配"},
                "since":      {"type": "string", "description": "起始时间，epoch 秒或 ISO-8601。按时间重叠筛选"},
                "until":      {"type": "string", "description": "结束时间，格式同 since。留空表示至今"},
                "limit":      {"type": "integer", "description": "Page size (default 100)"},
                "offset":     {"type": "integer", "description": "Page offset"},
            },
            "required": ["action"],
            "x-action-params": {
                "start":  {"params": ["input_topic"], "description": "独立模式：自己订阅音频 topic 并发布「谁在说话」。ASR 在跑时不需要这个 —— 身份已经随 ASR 的输出走了"},
                "stop":   {"params": [], "description": "Stop the standalone listener"},
                "info":   {"params": ["input_topic"], "description": "Report state, topics and database statistics"},
                "config": {"params": [], "description": "Update configuration"},
                "name_speaker": {
                    "params": ["name", "profile", "speaker_id"],
                    "description": "给一个声纹命名。speaker_id 留空 = 最近说话的那个人（主用法）。声纹已存在则 id 不变，之前所有出现记录都归到这个名字下",
                },
                "list_speakers": {
                    "params": ["named", "query", "limit", "offset"],
                    "description": "列出声纹，默认未命名优先、最近听到的在前 —— 先命名经常碰到的那几个，长尾不用管。「总共说过多少次」看 list_heard",
                },
                "get_speaker": {
                    "params": ["speaker_id"],
                    "description": "读一个声纹的完整记录，含 sample_audio_path —— 一段可播放的代表音频。声纹是从匿名话语建的，判断「这是谁」只能靠听",
                },
                "forget": {
                    "params": ["speaker_id", "speaker_ids", "named"],
                    "description": "删除声纹：单个、批量，或 named='unknown' 清空所有未命名。id 退役后永不复用",
                },
                "list_heard": {
                    "params": ["speaker_id", "since", "until", "limit", "offset"],
                    "description": "出现记录：谁在某段时间内说过话。一次连续说话算一条，含首末时间与次数",
                },
            },
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "model":            {"type": "string", "enum": sorted(SPEAKER_MODELS), "default": DEFAULT_SPEAKER_MODEL, "description": "声纹模型：" + "；".join(f"{name} = {spec['description']}" for name, spec in sorted(SPEAKER_MODELS.items())) + "。切换模型会使已存声纹失效——不同网络的 embedding 不可比较，数据库会拒绝加载，已注册的人需重新录入"},
                "device":           {"type": "string", "enum": ["cpu", "gpu"], "default": "cpu", "description": "推理设备。cpu 是默认且通常是对的：实测 gpu 单跑快 1.62 倍，但 ASR 在 cpu 时声纹已完全藏在它后面（净增 2-7ms），ASR 在 gpu 时两个 CUDA session 争同一块 GPU，搬上去只省 3ms 还多占 478MB"},
                "match_threshold":  {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": DEFAULT_MATCH_THRESHOLD, "description": "余弦相似度阈值，越高越严格。0.6 是起点不是实测值（face 的 0.35 是 ArcFace 的尺度，不能照搬）"},
                "min_speech_s":     {"type": "number", "minimum": 0.0, "default": DEFAULT_MIN_SPEECH_S, "description": "最短可用语音时长(秒)。低于此值只报听到了、不给身份——声纹在 1.5 秒以下急剧退化，而 VAD 只要 0.5 秒就出段"},
                "auto_enroll":      {"type": "boolean", "default": DEFAULT_AUTO_ENROLL, "description": "把没见过的声纹自动登记成未命名条目(p-N)。默认关：perception 没有回声消除，机器人自己的 TTS、电视、路人都会被登记进来。开之前先确认污染率"},
                "auto_add_samples": {"type": "boolean", "default": DEFAULT_AUTO_ADD_SAMPLES, "description": "已命名的人每说一句质量够好的话就补一条样本。声纹随距离/感冒/情绪漂移很大，而打分取样本最大值而非平均，所以补样本只会更准"},
                "unknown_capacity": {"type": "integer", "minimum": 0, "default": DEFAULT_UNKNOWN_CAPACITY, "description": "未命名声纹数量上限，超出时淘汰最久未听到的；已命名的不受影响"},
            },
        },
        "topic_in":  [{"format": AUDIO_FORMAT, "desc": "16 kHz mono PCM16 audio"}],
        "topic_out": [{"format": "data/json",  "desc": "who is speaking"}],
    }
]


def _speaker_output_topic(input_topic: str) -> str:
    return f"{input_topic}/speaker"


def _analyzer_options(cfg: dict) -> dict:
    return {
        "model": str(cfg.get("model", DEFAULT_SPEAKER_MODEL)),
        "model_dir": str(cfg.get("model_dir", "") or ""),
        "device": str(cfg.get("device", "cpu")),
        "num_threads": int(cfg.get("num_threads", 2)),
        "warmup": bool(cfg.get("warmup", True)),
    }


def _db_options(cfg: dict) -> dict:
    return {
        "db_dir": str(cfg.get("db_dir", DEFAULT_DB_DIR)),
        "model": str(cfg.get("model", DEFAULT_SPEAKER_MODEL)),
        "label": "speaker_db",
        "unknown_capacity": int(cfg.get("unknown_capacity", DEFAULT_UNKNOWN_CAPACITY)),
        "max_samples_per_person": int(
            cfg.get("max_samples_per_person", DEFAULT_MAX_SAMPLES_PER_PERSON)),
        "visit_gap_s": float(cfg.get("visit_gap_s", DEFAULT_VISIT_GAP_S)),
        "visit_log_max": int(cfg.get("visit_log_max", DEFAULT_VISIT_LOG_MAX)),
        "visit_checkpoint_s": float(
            cfg.get("visit_checkpoint_s", DEFAULT_VISIT_CHECKPOINT_S)),
    }


def _gates(cfg: dict) -> dict:
    return {
        "match_threshold": float(cfg.get("match_threshold", DEFAULT_MATCH_THRESHOLD)),
        "min_speech_s": float(cfg.get("min_speech_s", DEFAULT_MIN_SPEECH_S)),
        "split_above_s": float(cfg.get("split_above_s", DEFAULT_SPLIT_ABOVE_S)),
        "coherence_min": float(cfg.get("coherence_min", DEFAULT_COHERENCE_MIN)),
        "auto_enroll": bool(cfg.get("auto_enroll", DEFAULT_AUTO_ENROLL)),
        "auto_add_samples": bool(
            cfg.get("auto_add_samples", DEFAULT_AUTO_ADD_SAMPLES)),
        "sample_min_similarity": float(
            cfg.get("sample_min_similarity", DEFAULT_SAMPLE_MIN_SIMILARITY)),
        "sample_min_speech_s": float(
            cfg.get("sample_min_speech_s", DEFAULT_SAMPLE_MIN_SPEECH_S)),
    }


def _engine_signature(cfg: dict) -> tuple:
    """What a config change must rebuild the engine for.

    `unknown_capacity` is applied in place (set_unknown_capacity), so it must not
    force a rebuild — changing it from the card would otherwise drop every running
    instance and reload the model. Same exemption face makes.
    """
    db_options = _db_options(cfg)
    db_options.pop("unknown_capacity", None)
    return (
        tuple(sorted((k, str(v)) for k, v in _analyzer_options(cfg).items())),
        tuple(sorted((k, str(v)) for k, v in db_options.items())),
    )


class _SpeakerEngine:
    """The extractor and the identity database, loaded together."""

    __slots__ = ("embedder", "db", "sample_dir")

    def __init__(self, embedder: SpeakerEmbedder, db: IdentityDB, sample_dir: str):
        self.embedder = embedder
        self.db = db
        self.sample_dir = sample_dir

    def close(self) -> None:
        try:
            self.db.flush()
        except Exception:  # noqa: BLE001 - best-effort
            log.warning("[speaker] db flush on close failed", exc_info=True)
        try:
            self.embedder.close()
        except Exception:  # noqa: BLE001
            pass


def _build_engine(cfg: dict, on_status=None) -> _SpeakerEngine:
    """Extractor first, then the database — the dimension comes from the model.

    Deliberately in this order: `IdentityDB` needs `dim`, and the only honest
    source for it is `extractor.dim`. Hardcoding 192 here would let a model swap
    produce a database whose declared width disagrees with its contents, which is
    the exact failure IdentityDB refuses to let through.
    """
    embedder = SpeakerEmbedder(**_analyzer_options(cfg), on_status=on_status)
    options = _db_options(cfg)
    sample_dir = str(cfg.get("sample_dir", "")) or os.path.join(
        options["db_dir"], "samples")
    db = IdentityDB(dim=embedder.dim, **options)
    return _SpeakerEngine(embedder, db, sample_dir)


def _close_quietly(engine) -> None:
    close = getattr(engine, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            log.warning("[speaker] engine close failed", exc_info=True)


class _RecentRing:
    """The last few identifications, so `name_speaker` can mean "that person".

    Holds the **embedding**, not just the id: with `auto_enroll` off there is no
    `p-N` yet for a voice nobody has named, and "我是小王" has to be able to create
    the identity from what was just heard. Bounded by count and by age — a stale
    entry would let a name land on somebody who spoke an hour ago.
    """

    def __init__(self, size: int = DEFAULT_RECENT_RING,
                 window_s: float = DEFAULT_RECENT_WINDOW_S):
        self._items: deque[dict] = deque(maxlen=max(1, int(size)))
        self._window = max(1.0, float(window_s))
        self._lock = threading.Lock()

    def add(self, entry: dict) -> None:
        with self._lock:
            self._items.append(entry)

    def latest(self, now: float | None = None) -> dict | None:
        now = time.time() if now is None else now
        with self._lock:
            for entry in reversed(self._items):
                if now - entry["ts"] <= self._window:
                    return dict(entry)
        return None

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [dict(item) for item in self._items]


class _SpeakerNode(Node):
    """独立模式：自己订阅音频、自己做 VAD、发布「谁在说话」。

    Not the path identities reach the LLM by — that is `identify_pcm()`, called
    from `plugins/asr.py` on the segment ASR already cut. This exists for the
    canvas visualisation and for a robot with no ASR card running, and it carries
    its own VAD because there is nothing else to segment the stream. Starting it
    on a topic ASR is also on means two silero instances; that is an explicit
    choice the operator makes by wiring it, not something that happens silently.
    """

    def __init__(self, input_topic: str, engine: _SpeakerEngine, cfg: dict,
                 identify, node_suffix: str = ""):
        super().__init__(f"speaker_{node_suffix}" if node_suffix else "speaker")
        self._input_topic = input_topic
        self._output_topic = _speaker_output_topic(input_topic)
        self._engine = engine
        self._cfg = dict(cfg)
        self._identify = identify
        self.state = "idle"

        self._sub = None
        self._pub = self.create_publisher(String, self._output_topic, _RESULT_QOS)
        self._node_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._stop_event.set()
        self._worker: threading.Thread | None = None
        self._pcm_lock = threading.Lock()
        self._pcm_chunks: deque[bytes] = deque()
        self._retired = False
        self._stats = {"chunks": 0, "segments": 0, "errors": 0}
        log.info("[speaker] node created: subscribing=%s, publishing=%s",
                 input_topic, self._output_topic)

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> dict:
        with self._node_lock:
            if self._retired or self.state == "running":
                return self._status_dict()
            from audio_msgs.msg import AudioChunk

            self._stop_event = threading.Event()
            with self._pcm_lock:
                self._pcm_chunks.clear()
            self._sub = self.create_subscription(
                AudioChunk, self._input_topic, self._audio_cb, _AUDIO_QOS)
            self._worker = threading.Thread(
                target=self._vad_loop, args=(self._stop_event,),
                name="speaker-vad", daemon=True)
            self._worker.start()
            self.state = "running"
            return self._status_dict()

    def request_stop(self) -> None:
        """Signal cancellation without taking the lifecycle lock.

        Rule 4 of § "Plugin Concurrency": a `stop` that queues behind a `start`
        can no longer cancel it.
        """
        self._stop_event.set()

    def stop(self) -> dict:
        self.request_stop()
        with self._node_lock:
            if self._sub is not None:
                try:
                    self.destroy_subscription(self._sub)
                except Exception:  # noqa: BLE001
                    log.warning("[speaker] destroy_subscription failed", exc_info=True)
                self._sub = None
            worker, self._worker = self._worker, None
            self.state = "idle"
        if worker is not None and worker.is_alive():
            worker.join(timeout=3.0)
        return self._status_dict()

    def retire(self) -> dict:
        self._retired = True
        return self.stop()

    # ── audio ─────────────────────────────────────────────────────────────

    def _audio_cb(self, message: Any) -> None:
        if self._stop_event.is_set():
            return
        if getattr(message, "format", "") not in _AUDIO_FORMATS:
            return
        try:
            pcm = bytes(message.data)
        except (TypeError, ValueError):
            return
        if not pcm or len(pcm) % 2:
            return
        self._stats["chunks"] += 1
        with self._pcm_lock:
            # Bounded: a wedged worker must not grow this without limit. 60 s of
            # 16 kHz PCM16 at ~32 ms a chunk is comfortably above any real
            # backlog, and dropping the oldest is right for live audio.
            self._pcm_chunks.append(pcm)
            while len(self._pcm_chunks) > 2000:
                self._pcm_chunks.popleft()

    def _drain(self) -> list[bytes]:
        with self._pcm_lock:
            chunks = list(self._pcm_chunks)
            self._pcm_chunks.clear()
        return chunks

    def _vad_loop(self, stop_event: threading.Event) -> None:
        """sherpa-onnx silero VAD, in a thread rather than a child process.

        ASR runs its VAD in a `multiprocessing.Process`; that is not copied here
        because the reason for it does not apply — there is no second ONNX Runtime
        to isolate (see plugins/speaker_runtime.py), and silero infers one
        512-sample window at a time.
        """
        try:
            import sherpa_onnx

            from utils.model_downloader import ensure_model

            vad_dir = "/models/sherpa-onnx/vad"
            ensure_model("vad", vad_dir)
            config = sherpa_onnx.VadModelConfig(
                silero_vad=sherpa_onnx.SileroVadModelConfig(
                    model=os.path.join(vad_dir, "silero_vad.onnx"),
                    threshold=float(self._cfg.get("vad_threshold",
                                                  DEFAULT_VAD_THRESHOLD)),
                    min_silence_duration=float(
                        self._cfg.get("vad_silence_ms", DEFAULT_VAD_SILENCE_MS)) / 1000.0,
                    min_speech_duration=0.25,
                    max_speech_duration=float(
                        self._cfg.get("vad_max_speech_s", DEFAULT_VAD_MAX_SPEECH_S)),
                ),
                sample_rate=SAMPLE_RATE,
                num_threads=1,
                provider="cpu",
            )
            vad = sherpa_onnx.VoiceActivityDetector(
                config, buffer_size_in_seconds=float(
                    self._cfg.get("vad_max_speech_s", DEFAULT_VAD_MAX_SPEECH_S)) + 5.0)
        except Exception:
            log.exception("[speaker] standalone VAD could not start")
            self.state = "error"
            return

        while not stop_event.is_set():
            chunks = self._drain()
            if not chunks:
                time.sleep(0.05)
                continue
            try:
                samples = SpeakerEmbedder.pcm16_to_float(b"".join(chunks))
                vad.accept_waveform(samples)
                while not vad.empty():
                    if stop_event.is_set():
                        return
                    segment = np.asarray(vad.front.samples, dtype=np.float32)
                    vad.pop()
                    self._publish(segment)
            except Exception:
                self._stats["errors"] += 1
                log.warning("[speaker] VAD loop error", exc_info=True)
                time.sleep(0.2)

    def _publish(self, samples: np.ndarray) -> None:
        self._stats["segments"] += 1
        payload = self._identify(samples, source=self._input_topic)
        payload["topic"] = self._input_topic
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False)
        self._pub.publish(message)

    def _status_dict(self) -> dict:
        return {
            "state": self.state,
            "topic_in": [{"topic": self._input_topic, "format": AUDIO_FORMAT, "desc": ""}],
            "topic_out": [{"topic": self._output_topic, "format": "data/json", "desc": ""}],
            "statistics": dict(self._stats),
        }


class SpeakerRecognitionPlugin:
    """Speaker recognition MCP plugin.

    The state machine, the single-flight loader and every locking rule are
    `plugins/face.py`'s; see that docstring and `plugins/ocr.py`'s. What is new
    here is `identify_pcm`, the in-process entry point ASR calls, and the fact
    that it works with **no node started** — the engine is plugin-level, so
    identities flow through ASR whether or not this card is on a canvas.
    """

    PREFIX = "speaker_recognition"
    # Short alias so a canvas card or an LLM can say `speaker_*`. Resolved by
    # main.py's longest-prefix match, which prefers PREFIX over an alias.
    ALIASES = ("speaker",)

    def __init__(self, plugin_cfg: dict, executor):
        self._plugin_cfg = dict(plugin_cfg)
        self._executor = executor

        self._state_lock = threading.Lock()
        self._nodes: dict[str, _SpeakerNode] = {}
        self._instance_configs: dict[str, dict] = {}
        self._pending_starts: dict[str, str] = {}
        self._engine: _SpeakerEngine | None = None
        self._engine_state = "idle"          # idle|loading|ready|error
        self._load_error: str | None = None
        self._load_status: str | None = None
        self._load_generation = 0

        self._recent = _RecentRing(
            size=int(plugin_cfg.get("recent_ring", DEFAULT_RECENT_RING)),
            window_s=float(plugin_cfg.get("recent_window_s", DEFAULT_RECENT_WINDOW_S)),
        )
        log.info("[speaker] plugin init: model=%s device=%s db_dir=%s "
                 "auto_enroll=%s auto_add_samples=%s",
                 plugin_cfg.get("model", DEFAULT_SPEAKER_MODEL),
                 plugin_cfg.get("device", "cpu"),
                 plugin_cfg.get("db_dir", DEFAULT_DB_DIR),
                 plugin_cfg.get("auto_enroll", DEFAULT_AUTO_ENROLL),
                 plugin_cfg.get("auto_add_samples", DEFAULT_AUTO_ADD_SAMPLES))

    def get_tools(self) -> list:
        return TOOLS

    # ── background loader (single-flight) ────────────────────────────────

    def _spawn_loader_locked(self) -> None:
        self._engine_state = "loading"
        self._load_error = None
        self._load_status = None
        generation = self._load_generation
        cfg = dict(self._plugin_cfg)
        threading.Thread(
            target=self._loader, args=(generation, cfg),
            name="speaker-engine-loader", daemon=True,
        ).start()

    def _loader(self, generation: int, cfg: dict) -> None:
        try:
            engine = _build_engine(
                cfg, on_status=lambda text: setattr(self, "_load_status", text))
        except Exception as error:  # noqa: BLE001 - surfaced via state/info
            log.exception("[speaker] engine load failed")
            with self._state_lock:
                if generation == self._load_generation:
                    self._engine_state = "error"
                    self._load_error = str(error)
                    self._load_status = None
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
                log.error("[speaker] failed to build instance %r on %r: %s",
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
                        log.error("[speaker] failed to register instance %r: %s",
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
                log.error("[speaker] failed to start instance %r: %s",
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

    def _create_node(self, node_key: str, input_topic: str,
                     engine: _SpeakerEngine) -> _SpeakerNode:
        return _SpeakerNode(
            input_topic, engine, self._merged_cfg(node_key),
            identify=self.identify_samples,
            node_suffix=node_key.replace("/", "_").replace("-", "_"),
        )

    def _dispose(self, node_key: str, node: _SpeakerNode) -> None:
        try:
            node.retire()
        finally:
            dispose_node(self._executor, node, label=f"speaker/{node_key}")
        log.info("[speaker] node disposed: %s", node_key)

    def _instance_state_locked(self, node_key: str) -> str:
        node = self._nodes.get(node_key)
        if node is not None:
            return node.state
        if node_key in self._pending_starts:
            return "error" if self._engine_state == "error" else "loading"
        return "idle"

    def _require_engine(self) -> _SpeakerEngine:
        """Block until the engine is up, loading it if nobody has yet.

        The management actions are useful with no node running at all — naming a
        voice, listing the roster, clearing contaminated unknowns — so they
        trigger the same single-flight load a `start` would and wait for it. A
        tools/call already has its own thread (ThreadingHTTPServer).
        """
        with self._state_lock:
            if self._engine is not None:
                return self._engine
            if self._engine_state in ("idle", "error"):
                self._spawn_loader_locked()
            generation = self._load_generation

        deadline = time.monotonic() + float(
            self._plugin_cfg.get("load_timeout_s", 180.0))
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
                        self._load_error or "speaker engine failed to load")
        raise RuntimeError("speaker engine is still loading; try again shortly")

    def engine_if_ready(self) -> _SpeakerEngine | None:
        """The engine, or None — never loads, never blocks.

        What `plugins/asr.py` uses. ASR must not block its worker loop on a model
        download, and an utterance with no identity is a complete, publishable
        result; an utterance delayed by 90 s of downloading is not.
        """
        with self._state_lock:
            if self._engine is None and self._engine_state == "idle":
                self._spawn_loader_locked()
            return self._engine

    # ── the identification path ───────────────────────────────────────────

    def identify_pcm(self, pcm: bytes, sample_rate: int = SAMPLE_RATE,
                     source: str = "") -> dict:
        """int16 PCM → who said it. The entry point `plugins/asr.py` calls.

        Returns a payload that is always safe to merge into an ASR result, and
        **never raises** — a failure here must not cost the transcript. Keys that
        would carry no information are omitted rather than set to null/empty:
        `profile` when there is none, `name` when the voice is unnamed,
        `similarity` when nothing was matched. An LLM paying for every key in its
        context should not be charged for `"profile": {}`.
        """
        return self.identify_samples(
            SpeakerEmbedder.pcm16_to_float(pcm), sample_rate=sample_rate,
            source=source)

    def identify_samples(self, samples: np.ndarray,
                         sample_rate: int = SAMPLE_RATE,
                         source: str = "") -> dict:
        try:
            return self._identify(samples, sample_rate, source)
        except Exception as error:  # noqa: BLE001 - never cost the transcript
            log.warning("[speaker] identify failed: %s", escape_log_text(error),
                        exc_info=True)
            return {"speaker_error": str(error)}

    def _identify(self, samples: np.ndarray, sample_rate: int,
                  source: str) -> dict:
        engine = self.engine_if_ready()
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        rate = int(sample_rate or SAMPLE_RATE)
        duration = len(samples) / float(rate)
        payload: dict[str, Any] = {"speech_duration_s": round(duration, 2)}
        if engine is None:
            # Loading, or failed. Say so once in the payload rather than
            # silently looking like "nobody recognised".
            payload["speaker_state"] = self._engine_state
            return payload

        cfg = self._plugin_cfg
        gates = _gates(cfg)

        # Gate 1: too short to be worth matching. Reported, not guessed at.
        if duration < gates["min_speech_s"]:
            payload["speaker_reason"] = REASON_TOO_SHORT
            return payload

        embedding, coherence = engine.embedder.embed_windowed(
            samples, sample_rate=rate, split_above_s=gates["split_above_s"])

        # Gate 2: the clip's two ends describe different people. Withholding the
        # identity is the point — a blend of two people published as one person
        # is worse than no identity, because it looks correct.
        if coherence is not None:
            payload["coherence"] = round(float(coherence), 3)
            if coherence < gates["coherence_min"]:
                payload["multi_speaker"] = True
                payload["speaker_reason"] = REASON_MULTI_SPEAKER
                return payload

        person_id, score = engine.db.match(embedding, gates["match_threshold"])
        record = None
        if person_id is None:
            if gates["auto_enroll"]:
                record = engine.db.enroll_unknown(embedding)
                person_id = record["id"]
            else:
                # `match` returns -1.0 as best_score on an empty database, and
                # that distinction is the whole point: "nobody is enrolled" and
                # "compared against the roster and nothing was close enough"
                # need opposite responses, and a single number cannot say which.
                #
                # `speaker_best_similarity` is a *different key* from
                # `speaker_similarity` rather than the same one with a null id:
                # `speaker_similarity` means "this is the matched person's
                # score", so reusing it for a miss invites reading a near-miss
                # as an identification.
                if score < 0:
                    payload["speaker_reason"] = REASON_NO_SPEAKERS
                else:
                    payload["speaker_reason"] = REASON_NO_MATCH
                    payload["speaker_best_similarity"] = round(float(score), 4)
                # Remembered either way: "我是小王" must be able to create the
                # identity from what was just heard.
                self._remember(embedding, None, duration, samples, rate, source)
                return payload
        else:
            record = engine.db.get_person(person_id)

        engine.db.record_sighting(person_id, time.time(), source)
        payload["speaker_id"] = person_id
        payload["speaker_similarity"] = round(float(score), 4)
        if record.get("name"):
            payload["speaker_name"] = record["name"]
        if record.get("profile"):
            payload["speaker_profile"] = record["profile"]
        self._remember(embedding, person_id, duration, samples, rate, source)
        self._maybe_add_sample(engine, person_id, record, embedding, score,
                               duration, gates)
        return payload

    def _remember(self, embedding, person_id, duration, samples, rate,
                  source) -> None:
        self._recent.add({
            "ts": time.time(),
            "embedding": embedding,
            "speaker_id": person_id,
            "duration_s": duration,
            "source": source,
            # The audio itself, so name_speaker can keep a playable sample for a
            # voice that had no identity at the time it was heard.
            "samples": samples,
            "sample_rate": rate,
        })

    def _maybe_add_sample(self, engine, person_id, record, embedding, score,
                          duration, gates) -> None:
        """Accumulate a sample for a named person — off the critical path.

        On a thread, because `add_samples` goes through `_save_locked`, which
        rewrites the whole embeddings `.npy` and `persons.json`. That is real
        eMMC IO, and `plugins/asr.py` joins this plugin's work before it
        publishes: doing the write inline would put a growing disk cost on every
        utterance. The sample is not urgent — missing one is invisible.
        """
        if not gates["auto_add_samples"] or not record.get("named"):
            return
        if score < gates["sample_min_similarity"]:
            return
        if duration < gates["sample_min_speech_s"]:
            return
        if engine.db.samples_of(person_id) >= engine.db.max_samples:
            return

        def write() -> None:
            try:
                engine.db.add_samples(person_id, [embedding])
            except Exception:  # noqa: BLE001 - best-effort enrichment
                log.debug("[speaker] auto sample for %s failed", person_id,
                          exc_info=True)

        threading.Thread(target=write, name="speaker-sample",
                         daemon=True).start()

    # ── representative audio ──────────────────────────────────────────────

    def _save_sample_audio(self, engine: _SpeakerEngine, person_id: str,
                           samples: np.ndarray, rate: int) -> str:
        """Keep one playable clip per identity, outside the VAD rolling cache.

        `/models/vad_segments/` is capped at 1000 files and rolls, so a clip left
        there is gone within a day. `get_speaker` has to be able to play
        *something* months later — it is the only way to answer "who is this" for
        an identity that was created from an anonymous utterance.
        """
        os.makedirs(engine.sample_dir, exist_ok=True)
        path = os.path.join(engine.sample_dir, f"{person_id}.wav")
        clipped = np.asarray(samples, dtype=np.float32).reshape(-1)[
            : int(rate * MAX_S)]
        pcm = (np.clip(clipped, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        tmp = f"{path}.tmp"
        with wave.open(tmp, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(int(rate))
            handle.writeframes(pcm)
        os.replace(tmp, path)
        return path

    def _sample_path_if_present(self, engine: _SpeakerEngine,
                                person_id: str) -> str:
        path = os.path.join(engine.sample_dir, f"{person_id}.wav")
        return path if os.path.exists(path) else ""

    # ── MCP dispatch ──────────────────────────────────────────────────────

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name in (self.PREFIX,) + self.ALIASES else name
        instance_id = args.get("instance_id", "")

        if action == "info":
            return self._do_info(instance_id, args.get("input_topic", ""))
        if action == "start":
            return self._do_start(instance_id, args)
        if action == "stop":
            return self._do_stop(instance_id)
        if action == "config":
            return self._do_config(instance_id, args)
        if action == "name_speaker":
            return self._do_name_speaker(args)
        if action == "list_speakers":
            return self._do_list_speakers(args)
        if action == "get_speaker":
            return self._do_get_speaker(args)
        if action == "forget":
            return self._do_forget(args)
        if action == "list_heard":
            return self._do_list_heard(args)
        return None

    _DESC = "Speaker recognition — tells who is talking from their voice"

    def _desc_locked(self, state: str) -> str:
        if state == "loading":
            return self._load_status or "Loading the speaker embedding model..."
        if state == "error" and self._load_error:
            return f"Model load failed: {self._load_error}"
        return self._DESC

    # ── info / start / stop / config ──────────────────────────────────────

    def _with_db_stats(self, result: dict, engine: _SpeakerEngine | None) -> dict:
        """Attach roster counts. Never fails info — it is the diagnostic path."""
        if engine is None:
            return result
        try:
            result["database"] = engine.db.stats()
            result["embedding_dim"] = engine.embedder.dim
            result["device"] = engine.embedder.device
        except Exception:  # noqa: BLE001
            log.warning("[speaker] could not read database stats", exc_info=True)
        return result

    def _do_info(self, instance_id: str, input_topic: str) -> dict:
        base = {"name": "SpeakerRecognition", "manufacture": "Embodied",
                "model": str(self._plugin_cfg.get("model", DEFAULT_SPEAKER_MODEL))}
        with self._state_lock:
            engine = self._engine
            keys = list(self._nodes) + [
                key for key in self._pending_starts if key not in self._nodes]
            if instance_id:
                node = self._nodes.get(instance_id)
                topic = (node._input_topic if node is not None
                         else self._pending_starts.get(instance_id, input_topic))
                state = self._instance_state_locked(instance_id)
                result = {
                    **base, "state": state, "desc": self._desc_locked(state),
                    "topic_in": ([{"topic": topic, "format": AUDIO_FORMAT, "desc": ""}]
                                 if topic else []),
                    "topic_out": ([{"topic": _speaker_output_topic(topic),
                                    "format": "data/json", "desc": ""}]
                                  if topic else []),
                }
                if state == "error" and self._load_error:
                    result["error"] = self._load_error
                return self._with_db_stats(result, engine)

            instances = {k: {"state": self._instance_state_locked(k)} for k in keys}
            topics_in, topics_out = [], []
            for key in keys:
                node = self._nodes.get(key)
                topic = node._input_topic if node else self._pending_starts[key]
                topics_in.append({"topic": topic, "format": AUDIO_FORMAT, "desc": ""})
                topics_out.append({"topic": _speaker_output_topic(topic),
                                   "format": "data/json", "desc": ""})
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
                topics_in = [{"topic": input_topic, "format": AUDIO_FORMAT, "desc": ""}]
                topics_out = [{"topic": _speaker_output_topic(input_topic),
                               "format": "data/json", "desc": ""}]
            result = {**base, "state": state, "desc": self._desc_locked(state),
                      "topic_in": topics_in, "topic_out": topics_out}
            if instances:
                result["instances"] = instances
            if self._load_error and state == "error":
                result["error"] = self._load_error
            return self._with_db_stats(result, engine)

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
                # always finds the instance in _pending_starts or _nodes.
                self._pending_starts[node_key] = input_topic
                if self._engine_state != "ready":
                    if self._engine_state in ("idle", "error"):
                        self._spawn_loader_locked()
                    return {"state": "loading", "input": input_topic,
                            "output": _speaker_output_topic(input_topic)}
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
                    log.error("[speaker] failed to register instance %r: %s",
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
                        "output": _speaker_output_topic(input_topic)}
            return {"state": "idle", "input": input_topic,
                    "output": _speaker_output_topic(input_topic)}
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
                    "output": _speaker_output_topic(input_topic)}
        return result

    def _do_stop(self, instance_id: str) -> dict:
        to_dispose: list[tuple[str, _SpeakerNode]] = []
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
        # Signal every node before disposing any: rule 4 of § "Plugin
        # Concurrency" — stop must be able to cancel a start already in flight.
        for _, node in to_dispose:
            node.request_stop()
        for node_key, node in to_dispose:
            self._dispose(node_key, node)
        return {"state": "idle"}

    # Per-instance because one microphone may want a different VAD cadence from
    # another; the model, the thresholds and the database are shared.
    _INSTANCE_SCOPED = ("vad_threshold", "vad_silence_ms", "vad_max_speech_s")

    def _do_config(self, instance_id: str, args: dict) -> dict:
        cfg = {k: v for k, v in args.items()
               if k not in ("action", "instance_id") and v is not None and v != ""}

        if instance_id:
            shared = set(cfg) - set(self._INSTANCE_SCOPED)
            if shared:
                raise ValueError(
                    f"{sorted(shared)} are shared by every instance of this "
                    f"plugin; send them without instance_id"
                )
            with self._state_lock:
                merged = {**self._instance_configs.get(instance_id, {}), **cfg}
                self._instance_configs[instance_id] = merged
                node = self._nodes.get(instance_id)
            if node is not None:
                node._cfg.update(cfg)
            return {"state": self._instance_state_locked(instance_id),
                    "config": dict(cfg)}

        with self._state_lock:
            before = _engine_signature(self._plugin_cfg)
            candidate = {**self._plugin_cfg, **cfg}
            after = _engine_signature(candidate)
            self._plugin_cfg = candidate
            engine = self._engine

            if before == after:
                # unknown_capacity applies in place rather than rebuilding.
                capacity = cfg.get("unknown_capacity")
                state = self._engine_state
            else:
                capacity = None
                self._load_generation += 1
                stale, self._engine = self._engine, None
                nodes, self._nodes = self._nodes, {}
                for key, node in nodes.items():
                    self._pending_starts[key] = node._input_topic
                self._spawn_loader_locked()
                state = "loading"

        if before == after:
            if capacity is not None and engine is not None:
                try:
                    engine.db.set_unknown_capacity(int(capacity))
                except Exception as error:  # noqa: BLE001
                    log.warning("[speaker] set_unknown_capacity failed: %s",
                                escape_log_text(error))
            return {"state": state, "config": dict(cfg)}

        for key, node in nodes.items():
            node.request_stop()
        for key, node in nodes.items():
            self._dispose(key, node)
        if stale is not None:
            _close_quietly(stale)
        return {"state": "loading", "config": dict(cfg)}

    # ── naming ────────────────────────────────────────────────────────────

    def _do_name_speaker(self, args: dict) -> dict:
        """Bind a name to a voiceprint. The main registration path.

        `speaker_id` omitted means *whoever spoke most recently*, which is how
        this is almost always used: the LLM has just seen an unnamed id in an ASR
        payload and the person has said who they are. Three cases, and all three
        end with the same shape of record:

        * the voice already has an id → name it, keep the id, keep its history;
        * the voice was heard but never enrolled (`auto_enroll` off) → create the
          identity now from the remembered embedding;
        * an explicit `speaker_id` → name that one, whenever it was heard.
        """
        name = str(args.get("name") or "").strip()
        if not name:
            return {"ok": False, "reason": REASON_BAD_INPUT,
                    "detail": "name is required"}
        profile = args.get("profile")
        merge = args.get("merge", True)
        engine = self._require_engine()
        speaker_id = str(args.get("speaker_id") or "").strip()

        if speaker_id:
            try:
                record = engine.db.update_person(
                    speaker_id, name=name, profile=profile, merge=bool(merge))
            except KeyError:
                return {"ok": False, "reason": REASON_BAD_INPUT,
                        "detail": f"no such speaker {speaker_id}"}
            return self._named_result(engine, record, created=False)

        entry = self._recent.latest()
        if entry is None:
            return {
                "ok": False, "reason": REASON_NO_RECENT,
                "detail": (
                    "nobody has spoken recently — say a sentence first, or pass "
                    "speaker_id to name a voice from list_speakers"
                ),
            }

        if entry["speaker_id"]:
            record = engine.db.update_person(
                entry["speaker_id"], name=name, profile=profile, merge=bool(merge))
            created = False
        else:
            record = engine.db.add(name, [entry["embedding"]], profile=profile)
            created = True
        try:
            self._save_sample_audio(engine, record["id"], entry["samples"],
                                    entry["sample_rate"])
        except Exception:  # noqa: BLE001 - the name is the deliverable
            log.warning("[speaker] could not keep a sample clip for %s",
                        record["id"], exc_info=True)
        return self._named_result(engine, record, created=created,
                                  heard_s=round(entry["duration_s"], 2))

    def _named_result(self, engine, record: dict, created: bool,
                      heard_s: float | None = None) -> dict:
        result = {"ok": True, "speaker_id": record["id"], "name": record["name"],
                  "created": created, "samples": record.get("samples", 0)}
        if record.get("profile"):
            result["profile"] = record["profile"]
        if heard_s is not None:
            result["heard_s"] = heard_s
        path = self._sample_path_if_present(engine, record["id"])
        if path:
            result["sample_audio_path"] = path
        return result

    # ── reads ─────────────────────────────────────────────────────────────

    def _do_list_speakers(self, args: dict) -> dict:
        engine = self._require_engine()
        page = engine.db.list_persons(
            named=str(args.get("named", "all") or "all"),
            query=str(args.get("query", "") or ""),
            limit=int(args.get("limit", 100) or 100),
            offset=int(args.get("offset", 0) or 0),
            # Unnamed first, most recently heard first. Not "most often heard"
            # — that is not stored, and list_heard answers it properly. An
            # auto-enrolled voice is never edited, so the default `updated`
            # tiebreak would be its creation order, i.e. arbitrary.
            order="recent",
        )
        for entry in page.get("persons", []):
            path = self._sample_path_if_present(engine, entry["id"])
            if path:
                entry["sample_audio_path"] = path
            if not entry.get("profile"):
                entry.pop("profile", None)
            if not entry.get("name"):
                entry.pop("name", None)
        return page

    def _do_get_speaker(self, args: dict) -> dict:
        speaker_id = str(args.get("speaker_id") or "").strip()
        if not speaker_id:
            return {"ok": False, "reason": REASON_BAD_INPUT,
                    "detail": "speaker_id is required"}
        engine = self._require_engine()
        try:
            record = engine.db.get_person(speaker_id)
        except KeyError:
            return {"ok": False, "reason": REASON_BAD_INPUT,
                    "detail": f"no such speaker {speaker_id}"}
        if not record.get("profile"):
            record.pop("profile", None)
        if not record.get("name"):
            record.pop("name", None)
        path = self._sample_path_if_present(engine, speaker_id)
        if path:
            record["sample_audio_path"] = path
        else:
            # Say why rather than omitting silently: an operator trying to
            # identify this voice needs to know listening is not an option.
            record["sample_audio_note"] = (
                "没有留存音频 —— 这条声纹建立于保留样本之前，或样本文件已被删除")
        return record

    def _do_forget(self, args: dict) -> dict:
        engine = self._require_engine()
        named = str(args.get("named", "") or "").strip()
        ids = args.get("speaker_ids")
        if isinstance(ids, str):
            ids = [part for part in ids.replace(",", " ").split() if part]

        if named == "unknown" and not ids and not args.get("speaker_id"):
            before = [entry["id"] for entry in engine.db.list_persons(
                named="unknown", limit=100000).get("persons", [])]
            count = engine.db.forget_unknowns()
            for pid in before:
                self._drop_sample_audio(engine, pid)
            return {"ok": True, "forgotten": count, "named": "unknown"}

        if ids:
            result = engine.db.forget_many(ids)
            for pid in result.get("forgotten", []) or []:
                self._drop_sample_audio(engine, pid)
            return {"ok": True, **result}

        speaker_id = str(args.get("speaker_id") or "").strip()
        if not speaker_id:
            return {"ok": False, "reason": REASON_BAD_INPUT,
                    "detail": "pass speaker_id, speaker_ids, or named='unknown'"}
        if not engine.db.forget(speaker_id):
            return {"ok": False, "reason": REASON_BAD_INPUT,
                    "detail": f"no such speaker {speaker_id}"}
        self._drop_sample_audio(engine, speaker_id)
        return {"ok": True, "forgotten": [speaker_id]}

    def _drop_sample_audio(self, engine: _SpeakerEngine, person_id: str) -> None:
        """A forgotten identity must not leave its recording behind.

        The clip is somebody's voice, and `forget` is how the operator revokes
        it. Leaving the wav would mean "deleted" did not delete.
        """
        path = os.path.join(engine.sample_dir, f"{person_id}.wav")
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError:
            log.warning("[speaker] could not remove sample clip %s", path,
                        exc_info=True)

    def _do_list_heard(self, args: dict) -> dict:
        engine = self._require_engine()
        return engine.db.list_visits(
            person_id=str(args.get("speaker_id", "") or ""),
            since=parse_time(args.get("since")),
            until=parse_time(args.get("until")),
            limit=int(args.get("limit", 100) or 100),
            offset=int(args.get("offset", 0) or 0),
        )


__all__ = [
    "DEFAULT_AUTO_ADD_SAMPLES",
    "DEFAULT_AUTO_ENROLL",
    "DEFAULT_DB_DIR",
    "DEFAULT_MATCH_THRESHOLD",
    "DEFAULT_MIN_SPEECH_S",
    "REASON_MULTI_SPEAKER",
    "REASON_NO_MATCH",
    "REASON_NO_RECENT",
    "REASON_NO_SPEAKERS",
    "REASON_TOO_SHORT",
    "SpeakerRecognitionPlugin",
    "TOOLS",
]
