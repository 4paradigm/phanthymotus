"""
tests/test_asr_speaker_hook.py — ASR 与声纹的集成点。

Four properties, each of which is a way the integration could silently be wrong:

* **pre-roll is stripped before embedding, but not before transcription.**
  `vad_pre_roll_ms` defaults to 500, and a 1.5 s segment with 500 ms of leading
  silence is a third contaminated. If this breaks, tuning an *ASR* parameter
  changes *speaker* accuracy and nobody will think to look there.
* **identify runs before the KWS gate.** `trigger_mode` defaults to `asr_kws`,
  where an utterance without the wake word never publishes. Hanging identify off
  the publish path only would mean the visit log holds nothing but people who
  said the wake word — i.e. "who was in the room" would be empty.
* **the embedding is concurrent with transcription, and bounded.** A plain
  `join()` would turn a path that cannot currently fail into one that can: one
  stuck `compute()` and ASR publishes nothing, ever.
* **no handle, or a broken one, costs nothing.** The transcript must survive
  speaker recognition being disabled, still loading, or raising.

The ASR node is built with stubbed ROS (vision_stubs) and a fake adapter; nothing
here loads a model or starts a VAD process.

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import threading
import time

import pytest

from vision_stubs import PERCEPTION_ROOT  # noqa: F401  (ROS stubs + sys.path)

import plugins.asr as asr_module  # noqa: E402


class FakeSpeaker:
    """Records what it was handed, so the test can assert on the PCM itself."""

    def __init__(self, fields=None, delay=0.0, raises=False):
        self.seen: list[bytes] = []
        self.sources: list[str] = []
        self._fields = fields if fields is not None else {"speaker_id": "p-1"}
        self._delay = delay
        self._raises = raises
        self.calls = 0

    def identify_pcm(self, pcm, sample_rate=16000, source=""):
        self.calls += 1
        self.seen.append(bytes(pcm))
        self.sources.append(source)
        if self._delay:
            time.sleep(self._delay)
        if self._raises:
            raise RuntimeError("boom")
        return dict(self._fields)


def make_node(speaker=None) -> asr_module._ASRNode:
    return asr_module._ASRNode(
        "/mic/audio", adapter=None, language="zh-CN", kws_cfg={},
        node_suffix="t", speaker=speaker,
    )


def pcm(n_samples: int, value: int = 1000) -> bytes:
    import struct
    return struct.pack(f"<{n_samples}h", *([value] * n_samples))


# ── pre-roll ─────────────────────────────────────────────────────────────────

def test_pre_roll_is_stripped_before_embedding():
    speaker = FakeSpeaker()
    node = make_node(speaker)
    utterance = pcm(16000)                      # 1 s
    pre_roll_bytes = 2 * 8000                   # 前 0.5 s 是 pre-roll
    node._collect_identity(node._start_identify(utterance, pre_roll_bytes))
    assert speaker.seen[0] == utterance[pre_roll_bytes:]
    assert len(speaker.seen[0]) == 2 * 8000


def test_zero_pre_roll_passes_the_whole_buffer():
    speaker = FakeSpeaker()
    node = make_node(speaker)
    utterance = pcm(16000)
    node._collect_identity(node._start_identify(utterance, 0))
    assert speaker.seen[0] == utterance


def test_absurd_pre_roll_does_not_empty_the_buffer():
    """pre_roll >= 整段时不能剥成空的 —— 那会让声纹对一段静音求 embedding。"""
    speaker = FakeSpeaker()
    node = make_node(speaker)
    utterance = pcm(1000)
    node._collect_identity(node._start_identify(utterance, 999999))
    assert speaker.seen[0] == utterance


def test_source_topic_is_passed_through():
    """出现记录要知道是哪只麦克风听到的。"""
    speaker = FakeSpeaker()
    node = make_node(speaker)
    node._collect_identity(node._start_identify(pcm(16000), 0))
    assert speaker.sources == ["/mic/audio"]


# ── 并发与边界 ───────────────────────────────────────────────────────────────

def test_identify_runs_concurrently_with_the_caller():
    """_start_identify 必须立刻返回，不能自己把活干完。"""
    speaker = FakeSpeaker(delay=0.3)
    node = make_node(speaker)
    started = time.monotonic()
    handle = node._start_identify(pcm(16000), 0)
    assert time.monotonic() - started < 0.1, "开线程的那一步不该阻塞"
    fields = node._collect_identity(handle)
    assert fields == {"speaker_id": "p-1"}


def test_a_stuck_identify_does_not_block_publishing(monkeypatch):
    """裸 join 会把一条今天不可能失败的路径变成可能停摆的。"""
    monkeypatch.setattr(asr_module._ASRNode, "_IDENTIFY_TIMEOUT_S", 0.15)
    speaker = FakeSpeaker(delay=5.0)
    node = make_node(speaker)
    started = time.monotonic()
    fields = node._collect_identity(node._start_identify(pcm(16000), 0))
    elapsed = time.monotonic() - started
    assert fields == {}, "超时必须降级为「不带身份」，而不是等下去"
    assert elapsed < 1.0, f"等了 {elapsed:.2f}s，timeout 没生效"


def test_an_exception_in_the_thread_costs_only_the_identity():
    speaker = FakeSpeaker(raises=True)
    node = make_node(speaker)
    assert node._collect_identity(node._start_identify(pcm(16000), 0)) == {}


def test_no_speaker_handle_is_free():
    node = make_node(None)
    assert node._start_identify(pcm(16000), 0) is None
    assert node._collect_identity(None) == {}


def test_duplicate_duration_key_is_dropped():
    """payload 里已经有 audio_duration_ms，两个 key 表达同一件事只会让 LLM 困惑。"""
    speaker = FakeSpeaker(fields={"speaker_id": "p-1", "speech_duration_s": 2.4})
    node = make_node(speaker)
    fields = node._collect_identity(node._start_identify(pcm(16000), 0))
    assert "speech_duration_s" not in fields
    assert fields["speaker_id"] == "p-1"


def test_empty_fields_merge_to_nothing():
    """声纹说不出身份时，payload 不该多出任何 key。"""
    speaker = FakeSpeaker(fields={})
    node = make_node(speaker)
    assert node._collect_identity(node._start_identify(pcm(16000), 0)) == {}


def test_identify_thread_is_a_daemon():
    """卡住的那个线程不能让进程退不出去。"""
    speaker = FakeSpeaker(delay=0.2)
    node = make_node(speaker)
    thread, _box = node._start_identify(pcm(16000), 0)
    assert isinstance(thread, threading.Thread)
    assert thread.daemon is True
    thread.join(timeout=2.0)


# ── VAD worker 的契约 ────────────────────────────────────────────────────────

def test_worker_tuple_without_pre_roll_still_parses():
    """旧三元组（热拷贝的文件、滚动重启）要降级成「不知道 pre-roll」，不能抛。

    消费侧是 `int(item[3]) if len(item) > 3 else 0`；这里把两种形状都走一遍，
    确认取值逻辑本身不依赖长度。
    """
    for item in [(b"\x00\x01", 1.0, 2.0), (b"\x00\x01", 1.0, 2.0, 64)]:
        utterance, start_ts, end_ts = item[:3]
        pre_roll = int(item[3]) if len(item) > 3 else 0
        assert utterance == b"\x00\x01"
        assert (start_ts, end_ts) == (1.0, 2.0)
        assert pre_roll in (0, 64)


def test_identify_is_called_before_the_kws_gate_in_source():
    """源码顺序的守卫：identify 必须在 asr_kws 的 continue 之前。

    不是风格问题。trigger_mode 默认 asr_kws，没有唤醒词的句子永不 publish；
    identify 若排在门之后，出现记录里就只剩说过唤醒词的人，而「谁在房间里待过」
    正是要对齐 face 的那个核心能力。
    """
    source = (PERCEPTION_ROOT / "plugins" / "asr.py").read_text()
    start_at = source.index("_start_identify(utterance")
    collect_at = source.index("_collect_identity(_identity)")
    gate_at = source.index("if not matched:")
    assert start_at < collect_at < gate_at, (
        "identify 跑到 KWS 门后面去了 —— 未触发的话语将不再被记录"
    )


# ── 未触发的话语 → 独立 topic ────────────────────────────────────────────────

def test_overheard_payload_routes_to_the_background():
    """`priority: 0` 和 `log_type: true` 两个都必需，缺一个就静默丢数据。

    priority 决定它进后台而不是打断主 agent（collector._extract_priority 里 JSON
    的 priority 优先于按 source 匹配）。log_type 决定 bg 管道对它追加而不是 1 秒
    内替换、只渲染最后一条 —— 没有它这条 topic 能发出去、看起来工作，而一场
    30 秒的旁人对话到 subagent 面前只剩最后半句。
    """
    import json as _json
    node = make_node(None)
    node._publish_overheard("明天的会改到三点", 100.0, 102.0, pcm(32000),
                            {"speaker_id": "p-3", "speaker_name": "小王"})
    published = node._overheard_pub.messages
    assert len(published) == 1
    payload = _json.loads(published[0])
    assert payload["priority"] == 0
    assert payload["log_type"] is True
    assert payload["overheard"] is True
    assert payload["text"] == "明天的会改到三点"
    assert payload["speaker_name"] == "小王"


def test_overheard_goes_to_its_own_topic():
    """独立 topic，不是同一条带 priority 0 —— 否则两种话在 ring buffer、
    raw_input_info 和画布上混在一起，而且共用一个节流桶。"""
    node = make_node(None)
    assert node._overheard_topic == "/mic/audio/asr_overheard"
    assert node._overheard_topic != node._output_topic


def test_overheard_without_an_identity_still_publishes():
    import json as _json
    node = make_node(None)
    node._publish_overheard("随便说的", 1.0, 2.0, pcm(16000), {})
    payload = _json.loads(node._overheard_pub.messages[0])
    assert "speaker_id" not in payload
    assert payload["text"] == "随便说的"


def test_overheard_publish_never_raises():
    """这条路径是附带收益，不能让它影响 ASR 的主输出。"""
    node = make_node(None)

    def boom(_msg):
        raise RuntimeError("publisher gone")

    node._overheard_pub.publish = boom
    node._publish_overheard("x", 1.0, 2.0, b"\x00\x01", {})      # 不抛即通过


def test_status_reports_both_output_topics():
    node = make_node(None)
    outs = [entry["topic"] for entry in node._status_dict()["topic_out"]]
    assert node._output_topic in outs
    assert node._overheard_topic in outs


def test_overheard_is_published_from_the_unmatched_branch():
    """源码守卫：未匹配唤醒词的分支必须在 continue 之前发出去。

    这是一行挪动就能静默破坏的 —— 挪到 continue 之后就永远不执行，而测试和日志
    都不会有任何异样。
    """
    source = (PERCEPTION_ROOT / "plugins" / "asr.py").read_text()
    branch = source[source.index("if not matched:"):]
    branch = branch[:branch.index("# Extract text after keyword")]
    assert "_publish_overheard" in branch
    assert branch.index("_publish_overheard") < branch.index("continue")


# ── publish_overheard 开关 ───────────────────────────────────────────────────

def make_node_with(publish_overheard: bool) -> asr_module._ASRNode:
    return asr_module._ASRNode(
        "/mic/audio", adapter=None, language="zh-CN", kws_cfg={},
        node_suffix="t", speaker=None, publish_overheard=publish_overheard,
    )


def test_publish_overheard_defaults_on():
    """默认开。关着而操作员想要，表现为「连好线什么都看不到且不知道为什么」。"""
    schema = asr_module.TOOLS[0]["configSchema"]["properties"]
    assert schema["publish_overheard"]["default"] is True
    assert make_node_with(True)._publish_overheard_enabled is True


def test_disabling_it_restores_the_old_behaviour():
    """关掉就是以前的行为：未触发的句子直接丢弃，一个字都不发。"""
    node = make_node_with(False)
    node._publish_overheard("随便说的", 1.0, 2.0, pcm(16000), {})
    assert node._overheard_pub.messages == []


def test_the_publisher_exists_even_when_disabled():
    """publisher 总是建出来，否则改配置得重启节点才生效。"""
    node = make_node_with(False)
    assert node._overheard_pub is not None
    assert node._overheard_topic == "/mic/audio/asr_overheard"


def test_the_flag_can_be_flipped_live():
    node = make_node_with(False)
    node._publish_overheard("第一句", 1.0, 2.0, pcm(16000), {})
    node._publish_overheard_enabled = True          # 模拟 config 下发
    node._publish_overheard("第二句", 3.0, 4.0, pcm(16000), {})
    import json as _json
    assert len(node._overheard_pub.messages) == 1
    assert _json.loads(node._overheard_pub.messages[0])["text"] == "第二句"


def test_config_applies_the_flag_to_running_nodes():
    """_sync_node_cfg 必须带上这个字段，否则 config 看着生效其实没下发到节点。"""
    source = (PERCEPTION_ROOT / "plugins" / "asr.py").read_text()
    assert "node._publish_overheard_enabled = self._publish_overheard_enabled" in source
    assert "if 'publish_overheard' in cfg:" in source
