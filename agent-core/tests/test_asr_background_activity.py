"""
test_asr_background_activity.py — 旁人说的话要在活动流里看得见。

主 agent 听到的那句话在活动流里是一条 `trigger`。**旁边听到的那句话此前没有任何
痕迹**：它进 bg buffer，而 bg subagent 可能被 1 秒节流窗口合掉、或被
`_bg_buffer_has_substance` 判为「没有内容」而根本不 spawn —— 于是那句话只在 DDS 上
存在过，看活动流的人无从知道机器人听见了什么。

这个文件守住四件事，每件都是一种「不报错但是错」：

* **背景 ASR 进活动流，而别的 P=0 事件不进。** P=0 里绝大多数是电量/温度/IMU，每秒
  都在来；全推进去等于把活动流冲掉，而它的用处恰恰是能一眼看完。
* **两种判据都算。** `background` 标记由 perception 写进 JSON，topic 名由画布连线
  决定，两者实现上独立；只认一个，另一端改动时会**静默**停止显示。
* **只带有信息的字段。** 空的 speaker/情绪不该占一行日志的宽度。
* **推送失败不能影响事件分流。** 显示是附带收益。

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_asr_background_activity.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import collector  # noqa: E402


def bg_event(**fields) -> dict:
    payload = {"text": "今天天气不错", "background": True, "priority": 0,
               "log_type": True, **fields}
    return {"source": "dds:/remote_control/mic/asr_background",
            "text": json.dumps(payload, ensure_ascii=False)}


# ── 哪些事件进活动流 ─────────────────────────────────────────────────────────

def test_background_speech_is_shown():
    entry = collector._asr_background_entry(bg_event())
    assert entry is not None
    assert entry["text"] == "今天天气不错"


def test_the_background_flag_alone_is_enough():
    """topic 名换了（画布连线改了）也要照常显示。"""
    ev = bg_event()
    ev["source"] = "dds:/some/other/topic"
    assert collector._asr_background_entry(ev) is not None


def test_the_topic_name_alone_is_enough():
    """perception 不写 background 标记了也要照常显示。"""
    ev = {"source": "dds:/remote_control/mic/asr_background",
          "text": json.dumps({"text": "旁边有人在说话", "priority": 0})}
    assert collector._asr_background_entry(ev)["text"] == "旁边有人在说话"


def test_a_gauge_reading_is_not_shown():
    """电量/温度/IMU 每秒都在来 —— 它们进活动流就是把活动流冲掉。"""
    ev = {"source": "dds:/battery", "text": json.dumps({"soc": 87.5})}
    assert collector._asr_background_entry(ev) is None


def test_the_foreground_utterance_is_not_duplicated():
    """主 agent 那句已经是一条 trigger 了，不能再出一条。"""
    ev = {"source": "dds:/remote_control/mic/asr",
          "text": json.dumps({"text": "小范小范，你好", "priority": 1,
                              "kws_triggered": True})}
    assert collector._asr_background_entry(ev) is None


def test_an_empty_transcript_is_not_shown():
    assert collector._asr_background_entry(bg_event(text="   ")) is None
    assert collector._asr_background_entry(bg_event(text="")) is None


def test_malformed_text_does_not_raise():
    for text in ("{not json", "", "就是一句纯文本"):
        ev = {"source": "dds:/remote_control/mic/asr_background", "text": text}
        entry = collector._asr_background_entry(ev)
        # 纯文本的那条要显示出来，坏 JSON 的不要抛
        if text == "就是一句纯文本":
            assert entry["text"] == text
        else:
            assert entry is None


# ── 带哪些字段 ───────────────────────────────────────────────────────────────

def test_informative_fields_come_along():
    entry = collector._asr_background_entry(bg_event(
        speaker_id="p-4", speaker_name="云强", lang="zh",
        emotion="HAPPY", audio_event="Laughter"))
    assert entry["speaker_name"] == "云强"
    assert entry["speaker_id"] == "p-4"
    assert entry["lang"] == "zh"
    assert entry["emotion"] == "HAPPY"
    assert entry["audio_event"] == "Laughter"


def test_noise_fields_are_left_out():
    """相似度/时间戳是排查信息，一行日志放不下；payload 原样在 raw_input_info 里。"""
    entry = collector._asr_background_entry(bg_event(
        speaker_similarity=0.81, coherence=0.95, audio_start_ts=1.0,
        audio_duration_ms=1992, spans=[{"span": "asr_transcribe"}]))
    assert set(entry) == {"text", "source"}


def test_empty_values_do_not_take_up_a_row():
    entry = collector._asr_background_entry(bg_event(
        speaker_name="", speaker_id=None, lang="zh"))
    assert "speaker_name" not in entry
    assert "speaker_id" not in entry
    assert entry["lang"] == "zh"


# ── 推送 ─────────────────────────────────────────────────────────────────────

def test_push_sends_one_event(monkeypatch):
    sent = []

    async def fake_push(event):
        sent.append(event)

    monkeypatch.setattr('api.motus_stream.push_event', fake_push,
                        raising=False)
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        collector._push_asr_background(bg_event(lang="zh")))
    assert len(sent) == 1
    assert sent[0]["type"] == "asr_background"
    assert sent[0]["payload"]["text"] == "今天天气不错"


def test_a_failing_push_is_swallowed(monkeypatch):
    """活动流是附带收益 —— 它坏了不能影响事件分流。"""
    async def boom(event):
        raise RuntimeError("socket gone")

    monkeypatch.setattr('api.motus_stream.push_event', boom, raising=False)
    loop = asyncio.get_event_loop_policy().new_event_loop()
    loop.run_until_complete(collector._push_asr_background(bg_event()))


def test_push_skips_everything_else(monkeypatch):
    sent = []

    async def fake_push(event):
        sent.append(event)

    monkeypatch.setattr('api.motus_stream.push_event', fake_push,
                        raising=False)
    loop = asyncio.get_event_loop_policy().new_event_loop()
    loop.run_until_complete(collector._push_asr_background(
        {"source": "dds:/battery", "text": json.dumps({"soc": 87.5})}))
    assert sent == []
