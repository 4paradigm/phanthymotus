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

**The identity reaches the LLM through ASR, and this card has no topics at all.**
The embedding is computed on the VAD segment ASR already cut — the same PCM buffer,
the same tuple, in a thread beside `transcribe()` — so there is no timestamp join and
no alignment error to get wrong. `identify_pcm()` is that entry point.

So this is an `actuator` card with no `topic_in` and no `topic_out`, which is the
shape 115 of phanthymotus-driver's 277 tools already have. Drawing ports on it would
invite wiring audio in, and that is the one thing not to do: it would mean a second
VAD, duplicate enrolment and duplicate sightings, with nothing saying so. What the
card is *for* on a canvas is being commanded — name, list, listen, forget.

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

from utils.log_sampling import escape_log_text

from plugins.identity_db import (
    DEFAULT_MAX_SAMPLES_PER_PERSON,
    DEFAULT_UNKNOWN_CAPACITY,
    DEFAULT_VISIT_CHECKPOINT_S,
    DEFAULT_VISIT_GAP_S,
    DEFAULT_VISIT_LOG_MAX,
    IdentityDB,
    dim_on_disk,
    parse_time,
)
from plugins.speaker_runtime import (
    DEFAULT_SPEAKER_MODEL,
    MODEL_OFF,
    DEFAULT_SPEAKER_MODEL_DIR,
    MAX_S,
    SAMPLE_RATE,
    SPEAKER_MODELS,
    SpeakerEmbedder,
    is_off,
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

# **On** by default, same as face. The point of the `p-N` format is that an
# identity is stable *before* it has a name: without an id, a voice nobody has
# named carries no information at all, so "the same person as the previous turn"
# — the one thing an agent can actually act on in a multi-person conversation —
# is inexpressible. `name_speaker(speaker_id=...)` also needs ids to exist;
# without them the only possible flow is naming whoever spoke most recently.
#
# This was `False` at first, out of a worry that stands up badly: perception has
# no acoustic echo cancellation, so the robot's own TTS reaches the microphone
# and would be enrolled as a person. It would — but a persistent noise source
# **matches itself**, so it becomes one or two stable entries, not a flood that
# fills the roster. TVs and corridor passers-by are the same. And the
# contamination is recoverable and visible: `list_speakers` puts unnamed first,
# `get_speaker` hands back a clip to listen to, `forget named=unknown` clears
# them, and `unknown_capacity` evicts the least recently heard anyway.
#
# So the trade is: a handful of junk entries that can be listened to and deleted,
# against silently losing speaker *distinction* in the default configuration.
DEFAULT_AUTO_ENROLL = True

# Creating an identity needs a longer clip than matching one, and the asymmetry is
# deliberate — `plugins/face.py` makes exactly the same distinction for exactly
# the same reason ("auto-enrolling it would spend a person slot on a smear that
# never matches anything again"). A 1.6 s clip is long enough to risk a match
# against an existing voiceprint; it is not long enough to become one, because a
# marginal embedding then occupies a slot and never matches anything again.
DEFAULT_ENROLL_MIN_SPEECH_S = 2.0

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
# auto_enroll 开着、但这一段短到不值得为它建一个身份。和 no_match 分开报，因为
# 「说长一点就会有 id」和「这个人确实不认识」是两件不同的事。
REASON_TOO_SHORT_TO_ENROLL = "too_short_to_enroll"
# model 被设成 off：什么都没加载，所以连「不够像」都谈不上。
REASON_DISABLED = "speaker_recognition_off"


TOOLS = [
    {
        "name": "speaker_recognition",
        # actuator，而且**不声明任何 topic**。声纹不是流水线上的一级：身份是在 ASR
        # 已经切好的那个 VAD 段上算出来的，随 ASR 的输出一起走（见模块文档）。给这
        # 张卡画输入输出口会暗示「要把音频连进来」，而那恰恰是不该做的事 —— 连进来
        # 等于再跑一份 VAD、重复登记、重复记出现记录。
        #
        # 这张卡在画布上的作用是**对它下命令**：命名、查名单、听样本、删人。
        # phanthymotus-driver 里 277 个工具有 115 个同样不声明 topic，几乎全是
        # actuator —— 这是这个项目里「一个你调用的东西」既有的形状。
        "type": "actuator",
        "description": (
            "Speaker recognition — tell who is talking from their voice, and give "
            "a name to a voice the robot has already heard. 身份随 ASR 的输出一起"
            "走，这张卡不需要连线"
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
                "start":  {"params": [], "description": "开启声纹识别：预加载模型（否则第一句话要多等 1-2 秒），并允许给语音附加身份。不需要输入 topic"},
                "stop":   {"params": [], "description": "停止给语音附加身份。ASR 继续正常转写，只是输出里不再带 speaker_* 字段 —— 也是一个「别再给声音建档」的开关"},
                "info":   {"params": [], "description": "Report state and database statistics"},
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
                "model":            {"type": "string", "enum": [MODEL_OFF] + sorted(SPEAKER_MODELS), "default": DEFAULT_SPEAKER_MODEL, "description": "声纹模型。" + "；".join(f"{name} = {spec['description']}" for name, spec in sorted(SPEAKER_MODELS.items())) + f"。选 `{MODEL_OFF}` 则**完全不加载**：权重不进内存（cpu 约 84 MB / gpu 约 478 MB）、不占算力，ASR 的输出也不再带 speaker_* 字段——和卡片上的 stop 不同，那个只是停止归因、引擎仍留在内存里。关闭状态下名单、改名、删除仍然可用。切换模型会使已存声纹失效：不同网络的 embedding 不可比较，数据库会拒绝加载，已注册的人需重新录入"},
                "device":           {"type": "string", "enum": ["cpu", "gpu"], "default": "cpu", "description": "推理设备。cpu 是默认且通常是对的：实测 gpu 单跑快 1.62 倍，但 ASR 在 cpu 时声纹已完全藏在它后面（净增 2-7ms），ASR 在 gpu 时两个 CUDA session 争同一块 GPU，搬上去只省 3ms 还多占 478MB"},
                "match_threshold":  {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": DEFAULT_MATCH_THRESHOLD, "description": "余弦相似度阈值，越高越严格。0.6 是起点不是实测值（face 的 0.35 是 ArcFace 的尺度，不能照搬）"},
                "min_speech_s":     {"type": "number", "minimum": 0.0, "default": DEFAULT_MIN_SPEECH_S, "description": "最短可用语音时长(秒)。低于此值只报听到了、不给身份——声纹在 1.5 秒以下急剧退化，而 VAD 只要 0.5 秒就出段"},
                "auto_enroll":      {"type": "boolean", "default": DEFAULT_AUTO_ENROLL, "description": "把没见过的声纹自动登记成未命名条目(p-N)。默认开，和 face 一致：没有 id 的话「刚才那个人又说话了」就表达不出来，而那是多人对话里唯一能用的信息。代价是机器人自己的 TTS、电视、路人也会被登记（perception 没有回声消除）—— 用 list_speakers 看、get_speaker 听、forget named=unknown 清"},
                "enroll_min_speech_s": {"type": "number", "minimum": 0.0, "default": DEFAULT_ENROLL_MIN_SPEECH_S, "description": "自动登记一个**新**声纹所需的最短时长(秒)，比 min_speech_s 严：1.6 秒够冒险匹配一次，但不够成为一条声纹——勉强的 embedding 会占住一个槽位且再也匹配不上任何东西"},
                "auto_add_samples": {"type": "boolean", "default": DEFAULT_AUTO_ADD_SAMPLES, "description": "已命名的人每说一句质量够好的话就补一条样本。声纹随距离/感冒/情绪漂移很大，而打分取样本最大值而非平均，所以补样本只会更准"},
                "unknown_capacity": {"type": "integer", "minimum": 0, "default": DEFAULT_UNKNOWN_CAPACITY, "description": "未命名声纹数量上限，超出时淘汰最久未听到的；已命名的不受影响"},
            },
        },
        # 没有 topic_in / topic_out —— 见上面 type 处的注释。
    }
]




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
        "enroll_min_speech_s": float(
            cfg.get("enroll_min_speech_s", DEFAULT_ENROLL_MIN_SPEECH_S)),
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
        if self.db is not None:
            try:
                self.db.flush()
            except Exception:  # noqa: BLE001 - best-effort
                log.warning("[speaker] db flush on close failed", exc_info=True)
        if self.embedder is not None:
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
    options = _db_options(cfg)
    sample_dir = str(cfg.get("sample_dir", "")) or os.path.join(
        options["db_dir"], "samples")

    if is_off(options["model"]):
        # Nothing is loaded: no weights in memory, no CUDA context, no inference.
        # The roster is still opened when its width can be read off disk, so the
        # management actions keep working — somebody who switched this off for
        # privacy reasons wants to *clear* what it collected, and making them
        # switch it back on first would be backwards.
        dim = dim_on_disk(options["db_dir"])
        if dim is None:
            return _SpeakerEngine(None, None, sample_dir)
        options = {**options, "model": ""}   # 不比对模型名：这里没有模型
        return assemble_engine(None, sample_dir, dim=dim, **options)

    embedder = SpeakerEmbedder(**_analyzer_options(cfg), on_status=on_status)
    return assemble_engine(embedder, sample_dir, **options)


def assemble_engine(embedder, sample_dir: str, dim: int | None = None,
                    **db_options) -> _SpeakerEngine:
    """Tie an embedder, a database and the kept clips into one engine.

    Separate from `_build_engine` only so the tests can build an engine around a
    fake embedder **through the same wiring**. A test helper that constructed
    `IdentityDB` itself would silently skip `on_evict`, and the first thing that
    went unnoticed would be exactly what this hook exists to prevent — which is
    what happened while writing it.
    """
    engine_box: dict = {}

    def on_evict(ids) -> None:
        # Capacity eviction happens deep inside IdentityDB, which has no idea
        # this plugin keeps a wav per voiceprint. `forget` deletes those files;
        # without this hook eviction would not, and `samples/` would only ever
        # grow. The engine does not exist yet when IdentityDB is constructed, so
        # it is read out of the box at call time.
        engine = engine_box.get("engine")
        if engine is None:
            return
        for person_id in ids:
            _drop_sample_file(engine.sample_dir, person_id)

    db = IdentityDB(dim=embedder.dim if embedder is not None else int(dim),
                    on_evict=on_evict, **db_options)
    engine = _SpeakerEngine(embedder, db, sample_dir)
    engine_box["engine"] = engine
    return engine


def _drop_sample_file(sample_dir: str, person_id: str) -> None:
    """Remove one identity's kept clip. Shared by `forget` and by eviction.

    A forgotten or evicted identity must not leave its recording behind: the
    clip is somebody's voice, and `forget` is how an operator revokes it —
    leaving the wav would mean "deleted" did not delete.
    """
    path = os.path.join(sample_dir, f"{person_id}.wav")
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        log.warning("[speaker] could not remove sample clip %s", path,
                    exc_info=True)


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


class SpeakerRecognitionPlugin:
    """Speaker recognition MCP plugin.

    The single-flight loader and its locking are `plugins/face.py`'s; see that
    docstring and `plugins/ocr.py`'s. What is *not* here is the rest of it: no
    ROS node, no per-instance bookkeeping, no topics. This plugin owns an engine
    and a set of commands, and `identify_pcm` — the in-process entry point ASR
    calls on the segment it already cut.

    An earlier version also ran a standalone mode: its own subscription, its own
    silero VAD, its own `<topic>/speaker` output. It was removed. Its two stated
    purposes were canvas visualisation (the identity is already in ASR's payload)
    and working when ASR is off (which is when nobody is listening anyway), and
    against that it cost a second VAD plus a real footgun — started on the topic
    ASR was on, it double-segmented, double-enrolled and double-logged every
    sighting, with nothing saying so.

    `start`/`stop` are kept and mean something: the framework sends them to every
    card on the canvas (and perception, unlike the driver bundles, has no
    `common/lifecycle`-style shim, so declining them fails the card and rolls the
    whole project back). `start` pre-loads the model so the first utterance does
    not pay for it; `stop` turns identity attribution off — which doubles as the
    switch for "stop building voiceprints of people".
    """

    PREFIX = "speaker_recognition"
    # Short alias so a canvas card or an LLM can say `speaker_*`. Resolved by
    # main.py's longest-prefix match, which prefers PREFIX over an alias.
    ALIASES = ("speaker",)

    def __init__(self, plugin_cfg: dict, executor):
        self._plugin_cfg = dict(plugin_cfg)
        self._executor = executor

        self._state_lock = threading.Lock()
        # No node, no instances: this card has no stream of its own.
        # `_enabled` is what `start`/`stop` move, and what `identify_pcm`
        # checks — see the class docstring.
        self._enabled = bool(plugin_cfg.get("enabled_on_start", True))
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

    def _require_engine(self) -> _SpeakerEngine:
        """Block until the engine is up, loading it if nobody has yet.

        Every command here is useful with the card never started — naming a
        voice, listing the roster, clearing contaminated unknowns — so they
        trigger the same single-flight load `start` would and wait for it. A
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

    def _require_db(self) -> _SpeakerEngine:
        """`_require_engine`, plus the guarantee that there is a roster to touch.

        With `model: off` and no database on disk there is nothing to list or
        delete. The error has to say both halves: "no such speaker p-1" alone
        reads as "it was deleted", when the truth is that nothing is open.
        """
        engine = self._require_engine()
        if engine.db is None:
            raise RuntimeError(
                "声纹识别已关闭（model: off），而磁盘上也没有已有的声纹库，"
                "所以没有名单可以操作。要重新开始识别，把 model 设回一个模型。"
            )
        return engine

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
        """float32 waveform → who said it. `identify_pcm` is the usual entry."""
        try:
            return self._identify(samples, sample_rate, source)
        except Exception as error:  # noqa: BLE001 - never cost the transcript
            log.warning("[speaker] identify failed: %s", escape_log_text(error),
                        exc_info=True)
            return {"speaker_error": str(error)}

    def _identify(self, samples: np.ndarray, sample_rate: int,
                  source: str) -> dict:
        # `stop` on the card lands here, and returning an empty dict is the whole
        # effect: ASR keeps transcribing and its payload simply carries no
        # `speaker_*` fields. Checked before the engine is touched so a stopped
        # card also stops *creating* voiceprints, which is the half of this that
        # matters for somebody turning it off deliberately.
        if not self._enabled:
            return {}
        engine = self.engine_if_ready()
        if engine is not None and engine.embedder is None:
            # model: off —— 权重根本没加载。返回空而不是一个 reason：ASR 的 payload
            # 不该因为一个被关掉的功能而多出字段。
            return {}
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
        created = False
        if person_id is None:
            # Enrolment needs a longer clip than matching — see
            # DEFAULT_ENROLL_MIN_SPEECH_S.
            long_enough = duration >= gates["enroll_min_speech_s"]
            if gates["auto_enroll"] and long_enough:
                record = engine.db.enroll_unknown(embedding)
                person_id = record["id"]
                created = True
            elif gates["auto_enroll"]:
                payload["speaker_reason"] = REASON_TOO_SHORT_TO_ENROLL
                if score >= 0:
                    payload["speaker_best_similarity"] = round(float(score), 4)
                self._remember(embedding, None, duration, samples, rate, source)
                return payload
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
        if created:
            # A just-created identity was not *matched*, so it has no similarity.
            # `score` here is `match`'s best-against-everybody-else — `-1.0` on an
            # empty roster, otherwise a number that failed the threshold. Either
            # way, reporting it as `speaker_similarity` reads as "p-3 matched at
            # 0.49" when p-3 was created *because* nothing matched. Measured on
            # Orin 5: the first ever utterance published `speaker_similarity: -1.0`.
            payload["speaker_new"] = True
            if score >= 0:
                payload["speaker_best_similarity"] = round(float(score), 4)
        else:
            payload["speaker_similarity"] = round(float(score), 4)
        if record.get("name"):
            payload["speaker_name"] = record["name"]
        if record.get("profile"):
            payload["speaker_profile"] = record["profile"]
        self._remember(embedding, person_id, duration, samples, rate, source)
        # Keep one playable clip per identity, written on the first sighting that
        # finds none. This is what makes the review flow possible at all: an
        # auto-enrolled `p-N` has no name, and the only way to answer "who is
        # this" is to listen — so `list_speakers` → `get_speaker` → listen →
        # `name_speaker` needs something to play. It used to be written only when
        # naming the most recent speaker, which meant every auto-enrolled
        # identity was unlistenable and the flow was dead on arrival.
        #
        # Also self-healing: an identity enrolled before this existed, or named
        # from the roster rather than from the live ring, gets its clip the next
        # time it speaks.
        self._maybe_keep_sample_audio(engine, person_id, samples, rate)
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

    def _maybe_keep_sample_audio(self, engine: _SpeakerEngine, person_id: str,
                                 samples: np.ndarray, rate: int) -> None:
        """Write this identity's representative clip if it has none yet.

        On a thread and never on the critical path: `plugins/asr.py` joins this
        plugin's work before it publishes, and a 192 kB wav write onto eMMC does
        not belong in that window. Missing one clip is invisible; the next
        sighting writes it.

        Bounded by the roster: one file per identity, 6 s each (~192 kB), and
        unnamed identities are capped by `unknown_capacity` — 50 × 192 kB ≈
        9.6 MB — with eviction now deleting the file too (see `_drop_sample_file`).
        """
        if self._sample_path_if_present(engine, person_id):
            return

        def write() -> None:
            # Re-check inside the thread: two sightings can race here, and the
            # loser would rewrite a file the winner just wrote.
            if self._sample_path_if_present(engine, person_id):
                return
            try:
                self._save_sample_audio(engine, person_id, samples, rate)
            except Exception:  # noqa: BLE001 - best-effort enrichment
                log.debug("[speaker] could not keep a clip for %s", person_id,
                          exc_info=True)

        threading.Thread(target=write, name="speaker-clip",
                         daemon=True).start()

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

        if action == "info":
            return self._do_info()
        if action == "start":
            return self._do_start()
        if action == "stop":
            return self._do_stop()
        if action == "config":
            return self._do_config(args)
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
        if state == "idle":
            if is_off(str(self._plugin_cfg.get("model", DEFAULT_SPEAKER_MODEL))):
                return ("已关闭（model: off）—— 模型未加载，不占内存/显存/算力；"
                        "ASR 的输出不附带说话人身份。名单与删除仍然可用")
            return "已关闭 —— ASR 的输出不再附带说话人身份"
        return self._DESC

    # ── info / start / stop / config ──────────────────────────────────────

    def _card_state_locked(self) -> str:
        """One state for the card, derived from the engine and the switch.

        There is no node to ask, so this is the whole state machine:

            _enabled=False            -> idle     (stopped; ASR adds no identity)
            engine loading            -> loading
            engine failed             -> error
            engine ready              -> running

        `idle` wins over the engine's own state on purpose: a stopped card must
        read as stopped even though the model is still in memory, because what
        the operator turned off is the *attribution*, not the allocation.
        """
        if not self._enabled:
            return "idle"
        if is_off(str(self._plugin_cfg.get("model", DEFAULT_SPEAKER_MODEL))):
            # 关闭不是「错误」也不是「在加载」—— 它就是停着，而且是被要求停着的。
            return "idle"
        if self._engine_state == "loading":
            return "loading"
        if self._engine_state == "error":
            return "error"
        if self._engine is not None:
            return "running"
        return "loading"

    def _with_db_stats(self, result: dict, engine: _SpeakerEngine | None) -> dict:
        """Attach roster counts. Never fails info — it is the diagnostic path.

        Both halves are optional and for different reasons: with `model: off`
        there is no embedder (so no dim, no device), and with `off` plus nothing
        on disk there is no database either. Reaching through either one blindly
        logged a traceback on **every** `info` while switched off — caught, so
        info still answered, but a stack trace per poll is exactly the noise that
        buries a real one.
        """
        if engine is None:
            return result
        try:
            if engine.db is not None:
                result["database"] = engine.db.stats()
            if engine.embedder is not None:
                result["embedding_dim"] = engine.embedder.dim
                result["device"] = engine.embedder.device
        except Exception:  # noqa: BLE001
            log.warning("[speaker] could not read database stats", exc_info=True)
        return result

    def _do_info(self) -> dict:
        with self._state_lock:
            engine = self._engine
            state = self._card_state_locked()
            result = {
                "name": "SpeakerRecognition", "manufacture": "Embodied",
                "model": str(self._plugin_cfg.get("model", DEFAULT_SPEAKER_MODEL)),
                "state": state,
                "desc": self._desc_locked(state),
            }
            if state == "error" and self._load_error:
                result["error"] = self._load_error
        return self._with_db_stats(result, engine)

    def _do_start(self) -> dict:
        """Arm the card and pre-load the model.

        Pre-loading is the only real work: without it the first utterance pays
        ~1-2 s for the model load inside the window ASR joins on, and that shows
        up as a slow first reply with nothing to explain it.
        """
        with self._state_lock:
            self._enabled = True
            off = is_off(str(self._plugin_cfg.get("model", DEFAULT_SPEAKER_MODEL)))
            if (not off and self._engine is None
                    and self._engine_state in ("idle", "error")):
                self._spawn_loader_locked()
            state = self._card_state_locked()
            result = {"state": state, "desc": self._desc_locked(state)}
            engine = self._engine
        return self._with_db_stats(result, engine)

    def _do_stop(self) -> dict:
        """Stop attributing speech to people.

        The engine stays loaded: reloading it on the next `start` would cost
        seconds, and nothing else is holding that memory for anybody. What stops
        is `identify_pcm` — ASR keeps transcribing, its payload simply carries no
        `speaker_*` fields, and no new voiceprints are created. That makes this
        the switch for "stop building voiceprints of people", which is worth
        having as one deliberate action rather than three config flips.
        """
        with self._state_lock:
            self._enabled = False
        return {"state": "idle", "desc": self._desc_locked("idle")}

    def _do_config(self, args: dict) -> dict:
        cfg = {k: v for k, v in args.items()
               if k not in ("action", "instance_id") and v is not None and v != ""}

        with self._state_lock:
            before = _engine_signature(self._plugin_cfg)
            candidate = {**self._plugin_cfg, **cfg}
            after = _engine_signature(candidate)
            self._plugin_cfg = candidate
            engine = self._engine

            if before == after:
                # unknown_capacity applies in place rather than rebuilding.
                capacity = cfg.get("unknown_capacity")
                state = self._card_state_locked()
                stale = None
            else:
                capacity = None
                self._load_generation += 1
                stale, self._engine = self._engine, None
                self._spawn_loader_locked()
                state = "loading"

        if stale is None:
            if capacity is not None and engine is not None:
                try:
                    engine.db.set_unknown_capacity(int(capacity))
                except Exception as error:  # noqa: BLE001
                    log.warning("[speaker] set_unknown_capacity failed: %s",
                                escape_log_text(error))
            return {"state": state, "config": dict(cfg)}

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
        engine = self._require_db()
        speaker_id = str(args.get("speaker_id") or "").strip()

        if speaker_id:
            try:
                record = engine.db.update_person(
                    speaker_id, name=name, profile=profile, merge=bool(merge))
            except KeyError:
                return {"ok": False, "reason": REASON_BAD_INPUT,
                        "detail": f"no such speaker {speaker_id}"}
            # If this id is also the voice in the live ring, back-fill its clip
            # now rather than waiting for the next sighting — somebody naming
            # from the roster will want to confirm by listening immediately.
            entry = self._recent.latest()
            if entry is not None and entry.get("speaker_id") == speaker_id:
                self._maybe_keep_sample_audio(
                    engine, speaker_id, entry["samples"], entry["sample_rate"])
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
        engine = self._require_db()
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
        engine = self._require_db()
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
        engine = self._require_db()
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
        _drop_sample_file(engine.sample_dir, person_id)

    def _do_list_heard(self, args: dict) -> dict:
        engine = self._require_db()
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
    "DEFAULT_ENROLL_MIN_SPEECH_S",
    "DEFAULT_MIN_SPEECH_S",
    "REASON_DISABLED",
    "REASON_MULTI_SPEAKER",
    "REASON_NO_MATCH",
    "REASON_NO_RECENT",
    "REASON_NO_SPEAKERS",
    "REASON_TOO_SHORT",
    "REASON_TOO_SHORT_TO_ENROLL",
    "SpeakerRecognitionPlugin",
    "TOOLS",
    "assemble_engine",
]
