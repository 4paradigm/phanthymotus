"""
tests/test_speaker_plugin.py — SpeakerRecognitionPlugin 的 dispatch 与识别语义。

The engine is faked at exactly one boundary (`SpeakerEmbedder.embed`) and the
database is real: `IdentityDB` on tmp_path, 4 dims. So the id allocation, the
matching, the visit log and the atomic writes are the production code, and only
the 28 MB of weights are not.

What this file is really guarding:

* **Empty keys do not appear.** `"speaker_profile": {}` and `"speaker_name": ""`
  are tokens an LLM pays for and information it does not get. Face emits them
  unconditionally; this plugin must not.
* **A multi-speaker clip withholds the identity.** Publishing a blend of two
  people as one person is worse than publishing nothing, because it looks right.
* **`name_speaker` works in all three of its cases**, including the one where
  `auto_enroll` is off and the voice has no id yet — which is the default
  configuration, so it is the normal case, not the edge case.
* **`forget` deletes the audio too.** The clip is somebody's voice and forget is
  how it is revoked; leaving the wav would mean "deleted" did not delete.

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from vision_stubs import PERCEPTION_ROOT  # noqa: F401  (ROS stubs + sys.path)

import plugins.identity_db as identity_db_module  # noqa: E402
import plugins.speaker as speaker_module  # noqa: E402
from plugins.identity_db import IdentityDB  # noqa: E402
from plugins.speaker import (  # noqa: E402
    REASON_BAD_INPUT,
    REASON_MULTI_SPEAKER,
    REASON_NO_RECENT,
    REASON_TOO_SHORT,
    SpeakerRecognitionPlugin,
    _SpeakerEngine,
)
from plugins.speaker_runtime import SAMPLE_RATE  # noqa: E402

DIM = 4
VOICE_A = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
VOICE_B = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
BLEND = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)


@pytest.fixture(autouse=True)
def _allow_tmp_db(monkeypatch):
    monkeypatch.setattr(
        identity_db_module, "require_models_subpath",
        lambda path, root="/models": str(path),
    )


class FakeEmbedder:
    """Maps a waveform to one of three fixed unit vectors, by its first sample.

    `0.5 → VOICE_A`, `-0.5 → VOICE_B`, anything else → BLEND. A test then writes
    `clip(3.0, 0.5)` to mean "three seconds of speaker A", which keeps every test
    below about the plugin's decisions rather than about signal processing.
    """

    dim = DIM
    device = "cpu"
    model = "fake"

    def __init__(self):
        self.calls = 0

    def _vector(self, samples):
        # 按出现过的 level 集合判断，不是首样本：换人段的前 4 秒全是 A，而整段
        # 同时含 A 和别的东西 —— 只看首样本的话两者得到同一个向量，换人就检测
        # 不出来了。真实 embedding 对整段的内容敏感，不只对开头。
        levels = set(np.unique(np.round(
            np.asarray(samples, dtype=np.float32).reshape(-1), 3)).tolist())
        levels.discard(0.0)        # condition() 补的零不属于任何人
        if levels == {0.5}:
            return VOICE_A
        if levels == {-0.5}:
            return VOICE_B
        return BLEND

    def embed(self, samples, sample_rate=SAMPLE_RATE):
        self.calls += 1
        return self._vector(samples).copy()

    def embed_windowed(self, samples, sample_rate=SAMPLE_RATE,
                       split_above_s=4.0):
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        duration = len(samples) / float(sample_rate)
        if duration <= split_above_s:
            return self.embed(samples, sample_rate), None
        window = int(sample_rate * split_above_s)
        lead = self.embed(samples[:window], sample_rate)
        whole = self.embed(samples, sample_rate)
        return lead, float(np.dot(lead, whole))

    def close(self):
        pass


def clip(seconds: float, level: float = 0.5,
         flip_after: float | None = None) -> np.ndarray:
    out = np.full(int(SAMPLE_RATE * seconds), level, dtype=np.float32)
    if flip_after is not None:
        # 后半段写成 0.0，于是整段的首样本仍是 level 但 FakeEmbedder 对
        # samples[:window] 和整段会给出不同的向量 —— 换人的结构。
        out[int(SAMPLE_RATE * flip_after):] = 0.123
    return out


def make_plugin(tmp_path, **cfg) -> SpeakerRecognitionPlugin:
    plugin = SpeakerRecognitionPlugin({"db_dir": str(tmp_path), **cfg},
                                      executor=None)
    embedder = FakeEmbedder()
    db = IdentityDB(db_dir=str(tmp_path), dim=DIM, label="speaker_db")
    plugin._engine = _SpeakerEngine(embedder, db,
                                    os.path.join(str(tmp_path), "samples"))
    plugin._engine_state = "ready"
    return plugin


# ── 识别：门与空值 ───────────────────────────────────────────────────────────

def test_short_clip_is_reported_not_guessed(tmp_path):
    plugin = make_plugin(tmp_path)
    payload = plugin.identify_samples(clip(0.8))
    assert payload["speaker_reason"] == REASON_TOO_SHORT
    assert "speaker_id" not in payload
    assert payload["speech_duration_s"] == pytest.approx(0.8, abs=0.01)
    # 没有跑 embedding —— 太短的段连算都不该算
    assert plugin._engine.embedder.calls == 0


def test_unknown_voice_without_auto_enroll_gives_no_id(tmp_path):
    plugin = make_plugin(tmp_path, auto_enroll=False)
    payload = plugin.identify_samples(clip(3.0))
    assert "speaker_id" not in payload
    # 但要被记住，否则「我是小王」就无从下手
    assert plugin._recent.latest() is not None


def test_unknown_voice_with_auto_enroll_creates_an_unnamed_identity(tmp_path):
    plugin = make_plugin(tmp_path, auto_enroll=True)
    payload = plugin.identify_samples(clip(3.0))
    assert payload["speaker_id"] == "p-1"
    assert "speaker_name" not in payload, "未命名的人不该带一个空 name"
    assert "speaker_profile" not in payload, "没有 profile 就不该出现这个 key"
    assert plugin._engine.db.get_person("p-1")["named"] is False


def test_named_voice_comes_back_with_name_and_profile(tmp_path):
    plugin = make_plugin(tmp_path)
    plugin._engine.db.add("小王", [VOICE_A], profile={"team": "运营部"})
    payload = plugin.identify_samples(clip(3.0))
    assert payload["speaker_id"] == "p-1"
    assert payload["speaker_name"] == "小王"
    assert payload["speaker_profile"] == {"team": "运营部"}
    assert payload["speaker_similarity"] == pytest.approx(1.0, abs=1e-4)


def test_named_voice_without_profile_omits_the_key(tmp_path):
    """这是本次要求的那一条：profile 空就整个 key 不出现。"""
    plugin = make_plugin(tmp_path)
    plugin._engine.db.add("小李", [VOICE_A])
    payload = plugin.identify_samples(clip(3.0))
    assert payload["speaker_name"] == "小李"
    assert "speaker_profile" not in payload


def test_a_different_voice_is_not_matched(tmp_path):
    plugin = make_plugin(tmp_path, auto_enroll=False)
    plugin._engine.db.add("小王", [VOICE_A])
    payload = plugin.identify_samples(clip(3.0, level=-0.5))   # VOICE_B
    assert "speaker_id" not in payload
    assert payload["speaker_similarity"] == pytest.approx(0.0, abs=1e-6)


# ── 多说话人 ─────────────────────────────────────────────────────────────────

def test_multi_speaker_clip_withholds_the_identity(tmp_path):
    plugin = make_plugin(tmp_path, auto_enroll=True)
    plugin._engine.db.add("小王", [VOICE_A])
    payload = plugin.identify_samples(clip(6.0, flip_after=4.0))
    assert payload["multi_speaker"] is True
    assert payload["speaker_reason"] == REASON_MULTI_SPEAKER
    assert "speaker_id" not in payload, (
        "两个人的混合必须不给身份 —— 给了就是一个看起来正确的错答案"
    )
    # 也不能趁机登记一个新的「人」
    assert plugin._engine.db.stats()["persons"] == 1


def test_coherent_long_clip_still_identifies(tmp_path):
    plugin = make_plugin(tmp_path)
    plugin._engine.db.add("小王", [VOICE_A])
    payload = plugin.identify_samples(clip(6.0))
    assert payload["speaker_id"] == "p-1"
    assert payload["coherence"] == pytest.approx(1.0, abs=1e-4)
    assert "multi_speaker" not in payload


def test_identify_never_raises(tmp_path):
    """识别失败不能让 ASR 丢掉这句话的转写。"""
    plugin = make_plugin(tmp_path)

    def boom(*_a, **_k):
        raise RuntimeError("extractor exploded")

    plugin._engine.embedder.embed_windowed = boom
    payload = plugin.identify_samples(clip(3.0))
    assert "speaker_error" in payload
    assert "speaker_id" not in payload


def test_engine_not_ready_says_so_rather_than_looking_unrecognised(tmp_path):
    plugin = SpeakerRecognitionPlugin({"db_dir": str(tmp_path)}, executor=None)
    plugin._engine_state = "loading"
    payload = plugin.identify_samples(clip(3.0))
    assert payload["speaker_state"] == "loading"
    assert "speaker_reason" not in payload


# ── name_speaker 的三条分支 ──────────────────────────────────────────────────

def test_name_the_most_recent_voice_creates_the_identity(tmp_path):
    """auto_enroll 关着时的主路径：刚听到一句，对方说「我是小王」。"""
    plugin = make_plugin(tmp_path, auto_enroll=False)
    plugin.identify_samples(clip(3.0))
    result = plugin.dispatch("speaker_recognition",
                             {"action": "name_speaker", "name": "小王"})
    assert result["ok"] is True
    assert result["created"] is True
    assert result["speaker_id"] == "p-1"
    # 同一个声音下次就该被认出来
    payload = plugin.identify_samples(clip(3.0))
    assert payload["speaker_name"] == "小王"


def test_name_the_most_recent_voice_promotes_an_existing_unnamed_id(tmp_path):
    plugin = make_plugin(tmp_path, auto_enroll=True)
    first = plugin.identify_samples(clip(3.0))
    result = plugin.dispatch("speaker_recognition",
                             {"action": "name_speaker", "name": "小王",
                              "profile": {"team": "运营部"}})
    assert result["created"] is False
    assert result["speaker_id"] == first["speaker_id"], (
        "命名不能换 id —— 之前所有出现记录都挂在那个 id 上"
    )
    assert result["profile"] == {"team": "运营部"}


def test_name_an_explicit_id(tmp_path):
    plugin = make_plugin(tmp_path)
    record = plugin._engine.db.enroll_unknown(VOICE_B)
    result = plugin.dispatch("speaker_recognition",
                             {"action": "name_speaker", "name": "小李",
                              "speaker_id": record["id"]})
    assert result["ok"] is True and result["created"] is False
    assert plugin._engine.db.get_person(record["id"])["name"] == "小李"


def test_name_with_nothing_recent_explains_what_to_do(tmp_path):
    plugin = make_plugin(tmp_path)
    result = plugin.dispatch("speaker_recognition",
                             {"action": "name_speaker", "name": "小王"})
    assert result["ok"] is False
    assert result["reason"] == REASON_NO_RECENT
    assert "list_speakers" in result["detail"]


def test_name_requires_a_name(tmp_path):
    plugin = make_plugin(tmp_path)
    result = plugin.dispatch("speaker_recognition",
                             {"action": "name_speaker", "name": "  "})
    assert result["ok"] is False and result["reason"] == REASON_BAD_INPUT


def test_stale_recent_entry_is_not_namable(tmp_path):
    """两小时前说话的人不该被现在的「我是小王」命中。"""
    plugin = make_plugin(tmp_path, recent_window_s=60.0)
    plugin._recent = speaker_module._RecentRing(size=4, window_s=60.0)
    plugin.identify_samples(clip(3.0))
    for item in plugin._recent._items:
        item["ts"] -= 7200
    result = plugin.dispatch("speaker_recognition",
                             {"action": "name_speaker", "name": "小王"})
    assert result["reason"] == REASON_NO_RECENT


def test_naming_keeps_a_playable_clip(tmp_path):
    plugin = make_plugin(tmp_path, auto_enroll=False)
    plugin.identify_samples(clip(3.0))
    result = plugin.dispatch("speaker_recognition",
                             {"action": "name_speaker", "name": "小王"})
    path = result["sample_audio_path"]
    assert os.path.exists(path)
    import wave
    with wave.open(path, "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == SAMPLE_RATE
        assert handle.getnframes() > 0


# ── 读与删 ───────────────────────────────────────────────────────────────────

def test_get_speaker_without_audio_says_why(tmp_path):
    plugin = make_plugin(tmp_path)
    plugin._engine.db.add("小王", [VOICE_A])
    record = plugin.dispatch("speaker_recognition",
                             {"action": "get_speaker", "speaker_id": "p-1"})
    assert "sample_audio_path" not in record
    assert "sample_audio_note" in record, (
        "听不了要明说 —— 想认出这条声纹的人需要知道『听』不是一个选项"
    )


def test_get_speaker_omits_empty_fields(tmp_path):
    plugin = make_plugin(tmp_path)
    plugin._engine.db.enroll_unknown(VOICE_A)
    record = plugin.dispatch("speaker_recognition",
                             {"action": "get_speaker", "speaker_id": "p-1"})
    assert "name" not in record and "profile" not in record


def test_get_speaker_rejects_a_missing_id(tmp_path):
    plugin = make_plugin(tmp_path)
    result = plugin.dispatch("speaker_recognition",
                             {"action": "get_speaker", "speaker_id": "p-99"})
    assert result["ok"] is False and "p-99" in result["detail"]


def test_list_speakers_puts_unnamed_first(tmp_path):
    plugin = make_plugin(tmp_path)
    plugin._engine.db.add("小王", [VOICE_A])
    plugin._engine.db.enroll_unknown(VOICE_B)
    page = plugin.dispatch("speaker_recognition", {"action": "list_speakers"})
    assert [entry["id"] for entry in page["persons"]] == ["p-2", "p-1"]
    assert "name" not in page["persons"][0]      # 未命名的不带空 name


def test_forget_removes_the_audio_too(tmp_path):
    plugin = make_plugin(tmp_path, auto_enroll=False)
    plugin.identify_samples(clip(3.0))
    named = plugin.dispatch("speaker_recognition",
                            {"action": "name_speaker", "name": "小王"})
    path = named["sample_audio_path"]
    assert os.path.exists(path)
    result = plugin.dispatch("speaker_recognition",
                             {"action": "forget", "speaker_id": "p-1"})
    assert result["ok"] is True
    assert not os.path.exists(path), (
        "声纹删了而录音留着，等于『删除』没有删除"
    )


def test_forget_unknown_clears_contamination(tmp_path):
    """电视/机器人自己的声音被登记进来之后的回收路径。"""
    plugin = make_plugin(tmp_path, auto_enroll=True)
    plugin._engine.db.add("小王", [VOICE_A])
    plugin._engine.db.enroll_unknown(VOICE_B)
    plugin._engine.db.enroll_unknown(BLEND)
    result = plugin.dispatch("speaker_recognition",
                             {"action": "forget", "named": "unknown"})
    assert result["forgotten"] == 2
    assert plugin._engine.db.stats()["persons"] == 1


def test_forget_many_reports_partial_success(tmp_path):
    plugin = make_plugin(tmp_path)
    plugin._engine.db.add("小王", [VOICE_A])
    result = plugin.dispatch("speaker_recognition",
                             {"action": "forget", "speaker_ids": "p-1, p-7"})
    assert result["forgotten"] == ["p-1"]
    assert result["missing"] == ["p-7"]


def test_forget_needs_a_target(tmp_path):
    plugin = make_plugin(tmp_path)
    result = plugin.dispatch("speaker_recognition", {"action": "forget"})
    assert result["ok"] is False and result["reason"] == REASON_BAD_INPUT


def test_list_heard_returns_visits(tmp_path):
    plugin = make_plugin(tmp_path, auto_enroll=True)
    plugin.identify_samples(clip(3.0))
    page = plugin.dispatch("speaker_recognition", {"action": "list_heard"})
    assert page["total"] >= 1


# ── 工具声明 ─────────────────────────────────────────────────────────────────

def test_no_file_or_url_entry_point_exists():
    """声纹只能从机器人自己的麦克风注册 —— 这条约束编码在工具面里。

    手机录音注册 + 远场麦识别会系统性偏低，而调阈值补不回来。工具面里没有这个
    入口，就不会有人用错的素材注册然后得出「声纹不准」的结论。
    """
    actions = set(speaker_module.TOOLS[0]["inputSchema"]["properties"]
                  ["action"]["enum"])
    for forbidden in ("register_by_photo", "register_by_audio", "register_by_url",
                      "register_by_corpus", "recognize_by_audio"):
        assert forbidden not in actions
    assert "name_speaker" in actions


def test_every_action_has_x_action_params_and_a_handler(tmp_path):
    schema = speaker_module.TOOLS[0]["inputSchema"]
    actions = schema["properties"]["action"]["enum"]
    declared = schema["x-action-params"]
    assert set(actions) == set(declared), "enum 与 x-action-params 必须一一对应"
    plugin = make_plugin(tmp_path)
    for action in actions:
        if action in ("start", "stop", "config"):
            continue        # 需要 executor / ROS，由 face 的同构路径覆盖
        assert plugin.dispatch("speaker_recognition", {"action": action}) is not None, \
            f"{action} 没有 handler"


def test_unknown_action_returns_none(tmp_path):
    plugin = make_plugin(tmp_path)
    assert plugin.dispatch("speaker_recognition", {"action": "nope"}) is None


def test_config_schema_defaults_match_the_module_constants():
    """卡片上显示的默认值必须和代码真正用的一致，否则 UI 在骗人。"""
    props = speaker_module.TOOLS[0]["configSchema"]["properties"]
    assert props["match_threshold"]["default"] == speaker_module.DEFAULT_MATCH_THRESHOLD
    assert props["min_speech_s"]["default"] == speaker_module.DEFAULT_MIN_SPEECH_S
    assert props["auto_enroll"]["default"] is False
    assert props["auto_add_samples"]["default"] is True
    assert props["device"]["default"] == "cpu"
