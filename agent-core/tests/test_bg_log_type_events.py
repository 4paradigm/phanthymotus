"""
test_bg_log_type_events.py — P=0 管道对「日志型」事件不能只留最后一条。

这条管道原本只服务仪表读数：SOC、温度、电压、IMU。对它们「最新值取代旧值」是正确的
—— 一秒内来两条电量，你只想要后一条。两个机制实现了这件事：

* `_bg_buffer_add` 在 1 秒节流窗口内**替换**同 source 的最后一条；
* `_format_bg_batch` 只渲染 `evs[-1]`，附一句「共 N 条，显示最新」。

把 KWS 未触发的话语送进来之后，这两条叠起来的效果是：间隔小于 1 秒的第二句话覆盖掉
第一句，而就算攒下了 N 条，送到 bg subagent 面前的也只有最后一条 —— 一场 30 秒的旁人
对话到它那里是最后半句话。对话不是读数，是日志，每条都带独立信息。

所以事件自己声明 `log_type: true`，collector 据此改成追加 + 全量渲染。下面每一条用例
都对应一种「发得出去、看起来工作、而数据静默丢失」的方式。

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_bg_log_type_events.py
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import collector  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_bg():
    collector._bg_buffer.clear()
    collector._bg_last_accepted.clear()
    yield
    collector._bg_buffer.clear()
    collector._bg_last_accepted.clear()


def speech(text: str, ts: float, speaker: str = "") -> dict:
    payload = {"text": text, "overheard": True, "priority": 0, "log_type": True}
    if speaker:
        payload["speaker_name"] = speaker
    return {"source": "dds:/mic/audio/asr_overheard", "ts": ts,
            "text": json.dumps(payload, ensure_ascii=False)}


def gauge(value: float, ts: float, source: str = "dds:/battery") -> dict:
    return {"source": source, "ts": ts,
            "text": json.dumps({"soc": value})}


# ── 分类 ─────────────────────────────────────────────────────────────────────

def test_log_type_is_read_from_the_json_text():
    assert collector._is_log_type(speech("你好", 100.0)) is True
    assert collector._is_log_type(gauge(80.0, 100.0)) is False


def test_log_type_is_also_read_from_the_payload_dict():
    assert collector._is_log_type(
        {"source": "x", "ts": 1.0, "payload": {"log_type": True}}) is True


def test_malformed_and_plain_text_events_are_not_log_type():
    """判定不能靠猜 source 名字，也不能被坏 JSON 搞崩。"""
    assert collector._is_log_type({"source": "x", "ts": 1.0, "text": "{oops"}) is False
    assert collector._is_log_type({"source": "x", "ts": 1.0, "text": "hello"}) is False
    assert collector._is_log_type({"source": "x", "ts": 1.0}) is False


# ── 节流：仪表型替换，日志型追加 ─────────────────────────────────────────────

def test_gauge_readings_within_the_window_still_replace():
    """既有行为不能变：一秒内的两条电量只保留后一条。"""
    collector._bg_buffer_add(gauge(80.0, 100.0))
    collector._bg_buffer_add(gauge(79.0, 100.5))
    assert len(collector._bg_buffer) == 1
    assert json.loads(collector._bg_buffer[0]["text"])["soc"] == 79.0


def test_two_utterances_inside_the_throttle_window_are_both_kept():
    """这就是那个 bug：接话间隔小于 1 秒时第二句会覆盖第一句。"""
    collector._bg_buffer_add(speech("明天的会改到三点", 100.0))
    collector._bg_buffer_add(speech("好，我通知大家", 100.3))
    assert len(collector._bg_buffer) == 2, (
        "两句话在 1 秒内说完就丢掉一句 —— 对话不是仪表读数"
    )
    texts = [json.loads(e["text"])["text"] for e in collector._bg_buffer]
    assert texts == ["明天的会改到三点", "好，我通知大家"]


def test_a_long_conversation_keeps_every_turn():
    for i in range(10):
        collector._bg_buffer_add(speech(f"第{i}句", 100.0 + i * 0.2))
    assert len(collector._bg_buffer) == 10


def test_mixed_sources_do_not_interfere():
    collector._bg_buffer_add(gauge(80.0, 100.0))
    collector._bg_buffer_add(speech("一句话", 100.1))
    collector._bg_buffer_add(gauge(79.0, 100.2))
    kinds = [collector._is_log_type(e) for e in collector._bg_buffer]
    assert kinds == [False, True], "仪表型被替换，日志型保留"


# ── FIFO 上限：日志型优先保留 ────────────────────────────────────────────────

def test_fifo_drops_gauges_before_utterances(monkeypatch):
    monkeypatch.setattr(collector.config, "main",
                        {"event": {"llm": {"collector_max_window": 3}}})
    collector._bg_buffer_add(gauge(80.0, 100.0, source="dds:/a"))
    collector._bg_buffer_add(gauge(70.0, 102.0, source="dds:/b"))
    collector._bg_buffer_add(speech("要留下的话", 104.0))
    collector._bg_buffer_add(speech("也要留下", 106.0))
    assert len(collector._bg_buffer) == 3
    kept = [json.loads(e["text"]) for e in collector._bg_buffer]
    assert [k.get("text") for k in kept if "text" in k] == ["要留下的话", "也要留下"], (
        "满了要先扔旧的仪表读数 —— 它的旧值本来就被下一条覆盖了，而丢一句话不可逆"
    )


def test_fifo_eventually_drops_the_oldest_utterance(monkeypatch):
    """全是日志型且超限时，还是得扔 —— 但扔最老的，而不是拒绝新的。"""
    monkeypatch.setattr(collector.config, "main",
                        {"event": {"llm": {"collector_max_window": 2}}})
    for i in range(4):
        collector._bg_buffer_add(speech(f"第{i}句", 100.0 + i * 2))
    texts = [json.loads(e["text"])["text"] for e in collector._bg_buffer]
    assert texts == ["第2句", "第3句"]


# ── 渲染：日志型给全部 ───────────────────────────────────────────────────────

def test_formatting_a_conversation_shows_every_turn():
    batch = [speech("明天的会改到三点", 100.0, speaker="小王"),
             speech("好，我通知大家", 100.4, speaker="小李"),
             speech("别忘了会议室换了", 101.0, speaker="小王")]
    rendered = collector._format_bg_batch(batch)
    for text in ("明天的会改到三点", "好，我通知大家", "别忘了会议室换了"):
        assert text in rendered, f"{text!r} 被渲染掉了 —— subagent 看不到就等于没发生"
    assert 'count="3"' in rendered
    # 说话人也要在，否则「是谁说的」丢了
    assert "小王" in rendered and "小李" in rendered


def test_formatting_a_gauge_still_shows_only_the_latest():
    batch = [gauge(80.0, 100.0), gauge(79.0, 101.0)]
    rendered = collector._format_bg_batch(batch)
    assert "79.0" in rendered
    assert "显示最新" in rendered


def test_a_very_long_conversation_says_what_it_dropped():
    """截断可以，静默截断不行 —— 那读起来像「这段时间只说了这么多」。"""
    batch = [speech(f"第{i}句", 100.0 + i) for i in range(40)]
    rendered = collector._format_bg_batch(batch)
    assert "已省略" in rendered
    assert 'count="40"' in rendered
    assert "第39句" in rendered, "最新的必须在"


def test_each_rendered_turn_carries_its_own_timestamp():
    batch = [speech("早些说的", 100.0), speech("后来说的", 160.0)]
    rendered = collector._format_bg_batch(batch)
    import prompt
    assert prompt.format_ts(100.0) in rendered
    assert prompt.format_ts(160.0) in rendered


# ── spawn 门槛 ───────────────────────────────────────────────────────────────

def test_an_utterance_with_no_numbers_still_counts_as_substance():
    """没认出说话人、没带时长的一句话，也不能被判为「没内容」。

    原本的判据是「JSON 里有没有数值字段」。一句话能通过是**偶然**的（恰好带了
    speaker_similarity 之类），而没有任何数字的一句话会被丢掉。
    """
    event = {"source": "dds:/mic/audio/asr_overheard", "ts": 100.0,
             "text": json.dumps({"text": "走吧", "log_type": True,
                                 "overheard": True})}
    assert collector._bg_buffer_has_substance([event]) is True


def test_empty_text_is_still_not_substance():
    event = {"source": "dds:/mic/audio/asr_overheard", "ts": 100.0,
             "text": json.dumps({"text": "", "log_type": True})}
    # text 字段为空但 JSON 非空 —— 这里判的是事件的 text，不是 payload 里的
    assert collector._bg_buffer_has_substance(
        [{"source": "x", "ts": 1.0, "text": "  "}]) is False
    assert collector._bg_buffer_has_substance([event]) is True


def test_gauge_substance_rules_unchanged():
    assert collector._bg_buffer_has_substance([gauge(80.0, 100.0)]) is True
    assert collector._bg_buffer_has_substance(
        [{"source": "x", "ts": 1.0, "text": '{"name":"idle"}'}]) is False


# ── 路由 ─────────────────────────────────────────────────────────────────────

def test_overheard_speech_routes_to_bg_despite_asr_in_the_source():
    """`priority` 字段优先于 source 匹配 —— 这是整条路由的全部机制。"""
    assert collector._extract_priority(speech("随便说的", 100.0)) == 0
    # 对照：普通 ASR 事件（priority 1）仍然打断主 agent
    normal = {"source": "dds:/mic/audio/asr", "ts": 100.0,
              "text": json.dumps({"text": "小范小范，过来", "priority": 1})}
    assert collector._extract_priority(normal) == 1


def test_an_asr_source_without_a_priority_field_still_defaults_to_one():
    """没有 priority 字段时按 source 匹配，既有行为不能变。"""
    assert collector._extract_priority(
        {"source": "dds:/mic/audio/asr", "ts": 1.0, "text": "plain text"}) == 1


# ── bg prompt ────────────────────────────────────────────────────────────────

def test_bg_goal_prompt_has_a_speech_branch():
    """prompt 写死「传感器数据」时，一句听到的话落进去最可能的结果是静默 finish。"""
    source = open(os.path.join(os.path.dirname(__file__), '..', 'src',
                               'collector.py'), encoding='utf-8').read()
    goal_region = source[source.index("'[bg] 后台监控"):]
    goal_region = goal_region[:goal_region.index("priority=P_LOW")]
    assert "听到的话" in goal_region
    assert "subagent_finish" in goal_region
    # 必须说清楚「不要因为有人说话就上报」和「上报要带原话和是谁说的」
    assert "不要回应" in goal_region
    assert "是谁说的" in goal_region
