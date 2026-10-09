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
from plugins.identity_db import IdentityDB  # noqa: F401,E402
from plugins.speaker import (  # noqa: E402
    REASON_BAD_INPUT,
    REASON_MULTI_SPEAKER,
    REASON_NO_MATCH,
    REASON_NO_RECENT,
    REASON_NO_SPEAKERS,
    REASON_TOO_SHORT,
    REASON_TOO_SHORT_TO_ENROLL,
    SpeakerRecognitionPlugin,
    _db_options,
    _SpeakerEngine,  # noqa: F401  (kept for readers tracing the engine shape)
    assemble_engine,
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
    """A plugin with a fake embedder but the **real** engine wiring.

    `assemble_engine` rather than constructing IdentityDB here: a helper that
    built the database itself would skip `on_evict`, and the two eviction tests
    below would pass while production leaked a wav per evicted voiceprint.
    That is not hypothetical — it is what this helper did first.
    """
    plugin = SpeakerRecognitionPlugin({"db_dir": str(tmp_path), **cfg},
                                      executor=None)
    db_options = {
        key: value for key, value in _db_options({"db_dir": str(tmp_path), **cfg}).items()
        if key != "model"
    }
    plugin._engine = assemble_engine(
        FakeEmbedder(), os.path.join(str(tmp_path), "samples"), **db_options)
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


def test_auto_enroll_is_on_by_default(tmp_path):
    """默认必须给 id。没有 id 的话「刚才那个人又说话了」就表达不出来，而那是
    多人对话里唯一真正能让 agent 行动的那条信息。"""
    plugin = make_plugin(tmp_path)            # 不传 auto_enroll
    payload = plugin.identify_samples(clip(3.0))
    assert payload["speaker_id"] == "p-1"
    assert "speaker_name" not in payload      # 有 id 不等于有名字
    assert plugin._engine.db.get_person("p-1")["named"] is False
    # 同一个声音第二次必须是同一个 id —— 这才叫「能区分说话人」
    again = plugin.identify_samples(clip(3.0))
    assert again["speaker_id"] == "p-1"


def test_a_new_identity_reports_no_similarity(tmp_path):
    """刚建的身份没有「相似度」可报 —— 它不是被匹配出来的，是被创建出来的。

    实机上第一句话发出的是 `speaker_similarity: -1.0`（空库哨兵值直接漏出去），
    而库非空时漏出去的是「和别人比的最高分、没过阈值」—— 读起来像「p-3 以 0.49
    匹配上了」，而真相是「没匹配上任何人，所以建了 p-3」。
    """
    plugin = make_plugin(tmp_path)
    payload = plugin.identify_samples(clip(3.0))
    assert payload["speaker_new"] is True
    assert "speaker_similarity" not in payload
    assert "speaker_best_similarity" not in payload, "空库没有「差一点」可言"
    assert payload["speaker_id"] == "p-1"


def test_a_new_identity_reports_the_closest_existing_voice(tmp_path):
    """库非空时，新建身份要报出它和最接近的那个人差多少 —— 调阈值要用。"""
    plugin = make_plugin(tmp_path)
    plugin._engine.db.add("小王", [VOICE_A])
    payload = plugin.identify_samples(clip(3.0, level=-0.5))   # VOICE_B
    assert payload["speaker_new"] is True
    assert payload["speaker_best_similarity"] == pytest.approx(0.0, abs=1e-6)
    assert "speaker_similarity" not in payload


def test_a_matched_identity_is_not_marked_new(tmp_path):
    plugin = make_plugin(tmp_path)
    plugin._engine.db.add("小王", [VOICE_A])
    payload = plugin.identify_samples(clip(3.0))
    assert "speaker_new" not in payload
    assert payload["speaker_similarity"] == pytest.approx(1.0, abs=1e-4)


def test_similarity_is_never_negative_in_any_payload(tmp_path):
    """-1.0 是 match() 的空库哨兵，不是一个相似度。它不该出现在任何 payload 里。"""
    plugin = make_plugin(tmp_path)
    payloads = [plugin.identify_samples(clip(3.0)),
                plugin.identify_samples(clip(3.0)),
                plugin.identify_samples(clip(3.0, level=-0.5))]
    for payload in payloads:
        for key in ("speaker_similarity", "speaker_best_similarity"):
            if key in payload:
                assert payload[key] >= 0.0, f"{key}={payload[key]}"


def test_a_second_voice_gets_a_second_id(tmp_path):
    plugin = make_plugin(tmp_path)
    first = plugin.identify_samples(clip(3.0))
    second = plugin.identify_samples(clip(3.0, level=-0.5))
    assert first["speaker_id"] == "p-1"
    assert second["speaker_id"] == "p-2"
    assert first["speaker_new"] is True and second["speaker_new"] is True
    assert plugin._engine.db.stats()["persons"] == 2


def test_enrolment_needs_a_longer_clip_than_matching(tmp_path):
    """创建身份的门槛比匹配严 —— 勉强的 embedding 会永久占住一个槽位。

    face 做了完全相同的区分（见 face.py 里 usable 为 False 那一段的注释）。
    """
    plugin = make_plugin(tmp_path, min_speech_s=1.5, enroll_min_speech_s=2.0)
    payload = plugin.identify_samples(clip(1.7))
    assert payload["speaker_reason"] == REASON_TOO_SHORT_TO_ENROLL
    assert "speaker_id" not in payload
    assert plugin._engine.db.stats()["persons"] == 0, "没有浪费槽位"
    # 但仍然被记住，所以「我是小王」还能把它扶成一个身份
    assert plugin._recent.latest() is not None
    named = plugin.dispatch("speaker_recognition",
                            {"action": "name_speaker", "name": "小王"})
    assert named["ok"] is True and named["created"] is True


def test_a_marginal_clip_can_still_match_an_existing_voice(tmp_path):
    """短到不够建身份，但够匹配已有的 —— 这正是两个门槛不同的意义。"""
    plugin = make_plugin(tmp_path, min_speech_s=1.5, enroll_min_speech_s=2.0)
    plugin._engine.db.add("小王", [VOICE_A])
    payload = plugin.identify_samples(clip(1.7))
    assert payload["speaker_id"] == "p-1"
    assert payload["speaker_name"] == "小王"


def test_marginal_clip_also_reports_how_close_it_came(tmp_path):
    plugin = make_plugin(tmp_path, min_speech_s=1.5, enroll_min_speech_s=2.0,
                         match_threshold=0.9)
    halfway = (VOICE_A + BLEND) / np.linalg.norm(VOICE_A + BLEND)
    plugin._engine.db.add("小王", [halfway])
    payload = plugin.identify_samples(clip(1.7))
    assert payload["speaker_reason"] == REASON_TOO_SHORT_TO_ENROLL
    assert payload["speaker_best_similarity"] == pytest.approx(0.7071, abs=1e-3)


def test_empty_roster_says_nobody_is_enrolled(tmp_path):
    """空库和「比过了都不够像」必须分开报 —— 两者的处置相反。

    之前两种都报 `speaker_similarity: 0.0`，而 0.0 读起来像「和库里的人比过、
    一个都不像」，会把人送去调 match_threshold —— 而阈值跟空库毫无关系，该做的
    是去注册一个人。
    """
    plugin = make_plugin(tmp_path, auto_enroll=False)
    payload = plugin.identify_samples(clip(3.0))
    assert payload["speaker_reason"] == REASON_NO_SPEAKERS
    assert "speaker_id" not in payload
    assert "speaker_similarity" not in payload, "没有可比的人，就没有相似度可报"
    assert "speaker_best_similarity" not in payload
    # 但要被记住，否则「我是小王」就无从下手
    assert plugin._recent.latest() is not None


def test_a_near_miss_reports_how_close_it_came(tmp_path):
    """库非空但都不够像：报出最高分，这是调阈值唯一需要的那个数。"""
    plugin = make_plugin(tmp_path, auto_enroll=False, match_threshold=0.9)
    plugin._engine.db.add("小王", [VOICE_A])
    payload = plugin.identify_samples(clip(3.0, level=-0.5))   # VOICE_B
    assert payload["speaker_reason"] == REASON_NO_MATCH
    assert payload["speaker_best_similarity"] == pytest.approx(0.0, abs=1e-6)
    assert "speaker_id" not in payload
    assert "speaker_similarity" not in payload, (
        "speaker_similarity 只在认出人时出现 —— 复用它会让「差一点」被读成「认出了」"
    )


def test_a_near_miss_reports_the_real_score_not_a_placeholder(tmp_path):
    """差一点的那个分必须是真实值。

    这是 `match_threshold` 唯一能靠的调参依据：看到 0.707 才知道把阈值从 0.9 降到
    0.65 就能认出，而看到一个占位的 0.0 只会让人以为完全不像。
    """
    plugin = make_plugin(tmp_path, auto_enroll=False, match_threshold=0.9)
    # 一个「半像」的声纹：和 VOICE_A 的余弦是 1/sqrt(2) ≈ 0.707
    halfway = (VOICE_A + BLEND) / np.linalg.norm(VOICE_A + BLEND)
    plugin._engine.db.add("小王", [halfway])
    payload = plugin.identify_samples(clip(3.0))               # VOICE_A
    assert payload["speaker_reason"] == REASON_NO_MATCH
    assert payload["speaker_best_similarity"] == pytest.approx(0.7071, abs=1e-3)
    assert "speaker_id" not in payload


def test_unknown_voice_with_auto_enroll_creates_an_unnamed_identity(tmp_path):
    plugin = make_plugin(tmp_path, auto_enroll=True)  # 默认值，显式写明以防默认再变
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
    assert payload["speaker_reason"] == REASON_NO_MATCH


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
        # 现在每个 action 都能在没有 ROS 的情况下调 —— 这张卡没有节点了
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
    assert props["auto_enroll"]["default"] is True
    assert props["enroll_min_speech_s"]["default"] == \
        speaker_module.DEFAULT_ENROLL_MIN_SPEECH_S
    assert (speaker_module.DEFAULT_ENROLL_MIN_SPEECH_S
            > speaker_module.DEFAULT_MIN_SPEECH_S), (
        "建身份的门槛必须严于匹配的门槛"
    )
    assert props["auto_add_samples"]["default"] is True
    assert props["device"]["default"] == "cpu"


# ── 代表音频：自动登记的身份也要能听 ────────────────────────────────────────

def _wait_for(predicate, timeout: float = 3.0) -> bool:
    """等后台线程把文件写出来。写盘故意不在关键路径上，所以这里要等。"""
    import time as _time
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if predicate():
            return True
        _time.sleep(0.02)
    return False


def sample_path(plugin, person_id: str) -> str:
    return os.path.join(plugin._engine.sample_dir, f"{person_id}.wav")


def test_auto_enrolled_identity_gets_a_playable_clip(tmp_path):
    """这是 auto_enroll 能用的前提。

    自动登记的 p-N 没有名字，而判断「这是谁」唯一的办法是听。之前音频只在
    「命名最近说话的那个」这条分支里写，于是每一个自动登记的身份都听不了，
    list_speakers → get_speaker → 听 → name_speaker 这套回收流程是死的。
    """
    plugin = make_plugin(tmp_path)
    payload = plugin.identify_samples(clip(3.0))
    assert payload["speaker_id"] == "p-1"
    assert _wait_for(lambda: os.path.exists(sample_path(plugin, "p-1"))), \
        "自动登记的身份没有留音频 —— 那就只能靠「最近说话的那个」命名"
    record = plugin.dispatch("speaker_recognition",
                             {"action": "get_speaker", "speaker_id": "p-1"})
    assert record["sample_audio_path"] == sample_path(plugin, "p-1")
    assert "sample_audio_note" not in record


def test_the_clip_is_a_real_wav_capped_at_max_s(tmp_path):
    import wave
    plugin = make_plugin(tmp_path)
    plugin.identify_samples(clip(30.0))
    assert _wait_for(lambda: os.path.exists(sample_path(plugin, "p-1")))
    with wave.open(sample_path(plugin, "p-1"), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == SAMPLE_RATE
        # MAX_S 上限：一段 30 秒的音频不能留成 30 秒的文件
        assert handle.getnframes() <= int(SAMPLE_RATE * 6) + 1


def test_a_second_sighting_does_not_rewrite_the_clip(tmp_path):
    """已有音频就不再写 —— 否则每次听到这个人都是一次 eMMC 写入。"""
    plugin = make_plugin(tmp_path)
    plugin.identify_samples(clip(3.0))
    assert _wait_for(lambda: os.path.exists(sample_path(plugin, "p-1")))
    first_mtime = os.stat(sample_path(plugin, "p-1")).st_mtime_ns
    plugin.identify_samples(clip(3.0))
    import time as _time
    _time.sleep(0.3)
    assert os.stat(sample_path(plugin, "p-1")).st_mtime_ns == first_mtime


def test_a_legacy_identity_with_no_clip_heals_on_the_next_sighting(tmp_path):
    """这个改动之前建的身份、或从名单里命名的身份，下次说话时补上音频。"""
    plugin = make_plugin(tmp_path)
    plugin._engine.db.add("小王", [VOICE_A])            # 直接入库，没有音频
    assert not os.path.exists(sample_path(plugin, "p-1"))
    plugin.identify_samples(clip(3.0))
    assert _wait_for(lambda: os.path.exists(sample_path(plugin, "p-1")))


def test_naming_from_the_roster_backfills_the_clip(tmp_path):
    """用 speaker_id 从名单里命名时，如果它正是刚说话的那个，立刻补音频。

    从名单里命名的人会想马上听一下确认，而不是等下一次开口。
    """
    plugin = make_plugin(tmp_path)
    plugin.identify_samples(clip(3.0))
    assert _wait_for(lambda: os.path.exists(sample_path(plugin, "p-1")))
    os.unlink(sample_path(plugin, "p-1"))               # 模拟音频缺失
    result = plugin.dispatch("speaker_recognition",
                             {"action": "name_speaker", "name": "小王",
                              "speaker_id": "p-1"})
    assert result["ok"] is True
    assert _wait_for(lambda: os.path.exists(sample_path(plugin, "p-1")))


# ── 容量淘汰必须带走音频 ─────────────────────────────────────────────────────

def test_capacity_eviction_removes_the_clip(tmp_path):
    """淘汰以前只返回一个计数，所以插件不知道哪些 id 走了，wav 永远留在盘上。

    `forget` 会删，淘汰不会 —— samples/ 于是只增不减。
    """
    plugin = make_plugin(tmp_path, unknown_capacity=1)
    plugin.identify_samples(clip(3.0))                  # p-1
    assert _wait_for(lambda: os.path.exists(sample_path(plugin, "p-1")))
    plugin.identify_samples(clip(3.0, level=-0.5))      # p-2，把 p-1 挤掉
    assert plugin._engine.db.stats()["persons"] == 1
    assert _wait_for(lambda: not os.path.exists(sample_path(plugin, "p-1"))), \
        "被淘汰的声纹把录音留在了盘上"


def test_lowering_the_capacity_also_removes_clips(tmp_path):
    plugin = make_plugin(tmp_path, unknown_capacity=10)
    plugin.identify_samples(clip(3.0))
    plugin.identify_samples(clip(3.0, level=-0.5))
    assert _wait_for(lambda: os.path.exists(sample_path(plugin, "p-1"))
                     and os.path.exists(sample_path(plugin, "p-2")))
    evicted = plugin._engine.db.set_unknown_capacity(1)
    assert evicted == 1
    assert _wait_for(lambda: not os.path.exists(sample_path(plugin, "p-1")))
    assert os.path.exists(sample_path(plugin, "p-2")), "留下的那个不能被删"


def test_named_identities_are_never_evicted(tmp_path):
    """容量是用来约束自动登记的，不是用来过期有人特意注册过的人。"""
    plugin = make_plugin(tmp_path, unknown_capacity=1)
    plugin._engine.db.add("小王", [VOICE_A])
    plugin.identify_samples(clip(3.0, level=-0.5))      # 未命名的 p-2
    plugin.identify_samples(clip(3.0, level=0.123))     # 未命名的 p-3，挤掉 p-2
    assert plugin._engine.db.get_person("p-1")["name"] == "小王"


# ── 卡片形状：actuator、无输入输出 ──────────────────────────────────────────

def test_the_card_declares_no_topics():
    """给这张卡画输入输出口会暗示「把音频连进来」，而那恰恰是不该做的事。

    身份是在 ASR 已经切好的 VAD 段上算的，随 ASR 的输出走。把音频也连进这张卡
    等于再跑一份 VAD、重复登记、重复记出现记录，而且没有任何东西会说出来。
    """
    tool = speaker_module.TOOLS[0]
    assert "topic_in" not in tool
    assert "topic_out" not in tool
    assert tool["type"] == "actuator", (
        "phanthymotus-driver 里 115/277 个工具同样不声明 topic，几乎全是 actuator"
        " —— 这是这个项目里「一个你调用的东西」既有的形状"
    )
    assert "multiInstance" not in tool, "没有流就没有实例"
    assert "input_topic" not in tool["inputSchema"]["properties"]


def test_start_and_stop_are_still_declared():
    """框架会对画布上每张卡无条件发 start/stop。

    perception 没有 phanthymotus-driver 的 common/lifecycle 兜底，拒绝这两个动作
    会让卡片失败，而启动序列是严格的 —— 一张卡会把整个项目回滚。
    """
    actions = speaker_module.TOOLS[0]["inputSchema"]["properties"]["action"]["enum"]
    assert "start" in actions and "stop" in actions


# ── start / stop 是真开关 ────────────────────────────────────────────────────

def test_stop_turns_off_identity_attribution(tmp_path):
    plugin = make_plugin(tmp_path)
    assert plugin.identify_samples(clip(3.0))["speaker_id"] == "p-1"
    assert plugin.dispatch("speaker_recognition", {"action": "stop"})["state"] == "idle"
    assert plugin.identify_samples(clip(3.0)) == {}, (
        "停了之后 ASR 的 payload 不该再带任何 speaker_* 字段"
    )


def test_stop_also_stops_creating_voiceprints(tmp_path):
    """这是「别再给人建声纹档」那一半，对特意关掉它的人才是重点。"""
    plugin = make_plugin(tmp_path)
    plugin.dispatch("speaker_recognition", {"action": "stop"})
    plugin.identify_samples(clip(3.0))
    assert plugin._engine.db.stats()["persons"] == 0


def test_start_arms_it_again(tmp_path):
    plugin = make_plugin(tmp_path)
    plugin.dispatch("speaker_recognition", {"action": "stop"})
    result = plugin.dispatch("speaker_recognition", {"action": "start"})
    assert result["state"] == "running"
    assert plugin.identify_samples(clip(3.0))["speaker_id"] == "p-1"


def test_start_needs_no_input_topic(tmp_path):
    """以前 start 不给 input_topic 会抛 —— 而框架正是不给的。"""
    plugin = make_plugin(tmp_path)
    assert plugin.dispatch("speaker_recognition", {"action": "start"})["state"] == "running"


def test_stopping_does_not_unload_the_engine(tmp_path):
    """重新 start 要马上可用，而那块内存也没别人在等。"""
    plugin = make_plugin(tmp_path)
    engine = plugin._engine
    plugin.dispatch("speaker_recognition", {"action": "stop"})
    assert plugin._engine is engine


def test_a_stopped_card_reads_as_stopped(tmp_path):
    """模型还在内存里，但操作员关掉的是「归因」，所以必须显示为 idle。"""
    plugin = make_plugin(tmp_path)
    plugin.dispatch("speaker_recognition", {"action": "stop"})
    info = plugin.dispatch("speaker_recognition", {"action": "info"})
    assert info["state"] == "idle"
    assert "已关闭" in info["desc"]


def test_management_actions_work_while_stopped(tmp_path):
    """停了也要能查名单、改名、删人 —— 库的生命周期独立于这个开关。"""
    plugin = make_plugin(tmp_path)
    plugin.identify_samples(clip(3.0))
    plugin.dispatch("speaker_recognition", {"action": "stop"})
    page = plugin.dispatch("speaker_recognition", {"action": "list_speakers"})
    assert page["total"] == 1
    assert plugin.dispatch("speaker_recognition",
                           {"action": "forget", "speaker_id": "p-1"})["ok"] is True


def test_info_reports_the_database_even_before_start(tmp_path):
    plugin = make_plugin(tmp_path)
    info = plugin.dispatch("speaker_recognition", {"action": "info"})
    assert info["state"] == "running"        # engine 已就绪（测试里直接装好了）
    assert info["embedding_dim"] == DIM
    assert "database" in info
