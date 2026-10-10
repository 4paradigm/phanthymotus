"""
tests/test_asr_audio_tags.py — SenseVoice 顺带给出的情绪 / 音频事件 / 语种。

SenseVoice-small 的解码结果里 `emotion` / `event` / `lang` 三个字段**本来就已经
算出来了**（sherpa 把这几个 tag 从 text 里剥掉的同时填进了 result），以前
`return text` 把它们一起扔了。这个文件守住把它们捡回来之后的四件事，每一件都是
一种「不报错但是错」：

* **没信息的取值必须被压掉。** `NEUTRAL` / `EMO_UNKNOWN` / `Speech` 不是信息。
  每句话都挂一个 `"emotion": "NEUTRAL"`，LLM 要为它付 token，还会把「模型没判断
  出情绪」读成「这个人语气平淡」—— 那是两件不同的事。
* **tag 走返回值，不走适配器上的状态。** `ASRPlugin._adapter` 是被所有 node 共享
  的**一个**实例，每个 node 有自己的 worker 线程。挂在实例上的情绪会在两个麦克风
  同时转写时串到另一句话上，而那不会报错也不会有日志。
* **两条发布路径都要带。** 前台（`<topic>/asr`）和旁人对话（`/asr_background`）
  任意一条漏掉，都会表现为「情绪时有时无」。
* **非 SenseVoice 的模型下这些 key 干脆不出现**，而不是出现空值。
* **少一个字段不能让机器人听不见。** 这三个字段来自 sherpa 的 C++ binding，换版
  本少一个就直接取属性会让整条 ASR 路径抛异常。

ROS 用 vision_stubs 打桩，不加载任何模型、不起 VAD 进程。

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import json
import queue
import struct
import threading
import time

import pytest

from vision_stubs import PERCEPTION_ROOT  # noqa: F401  (ROS stubs + sys.path)

import plugins.asr as asr_module  # noqa: E402


# ── tag 解析 ─────────────────────────────────────────────────────────────────

def test_strip_tag_unwraps_the_sensevoice_shape():
    assert asr_module._strip_tag("<|HAPPY|>") == "HAPPY"
    assert asr_module._strip_tag("<|zh|>") == "zh"


def test_strip_tag_passes_through_anything_else():
    """换个版本不带尖括号了也不能崩，更不能把值变成空。"""
    assert asr_module._strip_tag("HAPPY") == "HAPPY"
    assert asr_module._strip_tag("  <|SAD|>  ") == "SAD"
    assert asr_module._strip_tag("") == ""
    assert asr_module._strip_tag(None) == ""


def test_informative_tags_become_fields():
    fields = asr_module._audio_tag_fields(
        lang="<|zh|>", emotion="<|HAPPY|>", event="<|Laughter|>")
    assert fields == {"lang": "zh", "emotion": "HAPPY",
                      "audio_event": "Laughter"}


@pytest.mark.parametrize("emotion", ["<|NEUTRAL|>", "<|EMO_UNKNOWN|>",
                                     "<|UNKNOWN|>", ""])
def test_uninformative_emotion_is_dropped(emotion):
    fields = asr_module._audio_tag_fields(lang="<|zh|>", emotion=emotion,
                                          event="<|Speech|>")
    assert "emotion" not in fields, "没判断出情绪 ≠ 语气平淡"


def test_speech_event_is_dropped_but_lang_is_kept():
    """说话这件事由 text 本身证明；语种每次都有值，而值本身是信息。"""
    fields = asr_module._audio_tag_fields(
        lang="<|en|>", emotion="<|NEUTRAL|>", event="<|Speech|>")
    assert fields == {"lang": "en"}


def test_itn_echo_is_not_a_field():
    """`withitn` / `woitn` 是 ITN 开关的回显，连信息都不是。"""
    assert asr_module._audio_tag_fields(event="<|woitn|>") == {}
    assert asr_module._audio_tag_fields(event="<|withitn|>") == {}


def test_event_key_is_namespaced():
    """agent-core 那边 `event` 是事件总线的词；payload 里不能再出现一个同名 key。"""
    fields = asr_module._audio_tag_fields(event="<|Cry|>")
    assert fields == {"audio_event": "Cry"}
    assert "event" not in fields


# ── 适配器 ───────────────────────────────────────────────────────────────────

class _PlainAdapter(asr_module.ASRAdapter):
    """没有 tag 的模型（parakeet / x-asr 走的就是这条默认实现）。"""

    def transcribe(self, wav_bytes, language):
        return "  你好  "


def test_default_adapter_reports_no_tags():
    text, tags = _PlainAdapter().transcribe_rich(b"", "zh-CN")
    assert text == "  你好  ", "默认实现不该替 transcribe 做清理"
    assert tags == {}, "没有这些字段的模型下，payload 里这些 key 不该出现"


class _FakeResult:
    def __init__(self, text, **fields):
        self.text = text
        for key, value in fields.items():
            setattr(self, key, value)


class _FakeStream:
    def __init__(self, result):
        self.result = result
        self.samples = []

    def accept_waveform(self, sample_rate, samples):
        self.samples = list(samples)


class _FakeRecognizer:
    def __init__(self, result):
        self._result = result
        self.decoded = 0

    def create_stream(self):
        return _FakeStream(self._result)

    def decode_streams(self, streams):
        self.decoded += 1


def _sensevoice_adapter(result) -> asr_module.SherpaOnnxSenseVoiceAdapter:
    """绕开 __init__（它会 import sherpa_onnx 并加载 228 MB 权重）。"""
    adapter = object.__new__(asr_module.SherpaOnnxSenseVoiceAdapter)
    adapter._recognizer = _FakeRecognizer(result)
    return adapter


def _wav(n_samples: int = 1600) -> bytes:
    return asr_module._pcm16_to_wav(
        struct.pack(f"<{n_samples}h", *([1000] * n_samples)))


def test_sensevoice_returns_text_and_tags():
    adapter = _sensevoice_adapter(_FakeResult(
        " 哈哈哈 ", lang="<|zh|>", emotion="<|HAPPY|>", event="<|Laughter|>"))
    text, tags = adapter.transcribe_rich(_wav(), "zh-CN")
    assert text == "哈哈哈"
    assert tags == {"lang": "zh", "emotion": "HAPPY",
                    "audio_event": "Laughter"}


def test_sensevoice_transcribe_still_returns_only_text():
    """老接口不能变形：别的调用方（warmup、x_asr 的委托）只要字符串。"""
    adapter = _sensevoice_adapter(_FakeResult(
        "你好", lang="<|zh|>", emotion="<|SAD|>", event="<|Speech|>"))
    assert adapter.transcribe(_wav(), "zh-CN") == "你好"


def test_sensevoice_survives_a_binding_without_those_fields():
    """换个 sherpa 版本少一个字段，代价只能是少一个 tag，不能是听不见。"""
    adapter = _sensevoice_adapter(_FakeResult("你好"))   # 三个字段全没有
    text, tags = adapter.transcribe_rich(_wav(), "zh-CN")
    assert text == "你好"
    assert tags == {}


def test_sensevoice_pads_tail_silence():
    """tag 这条改动不能动到原本的尾部补零（否则末字被截）。"""
    result = _FakeResult("你好", lang="<|zh|>")
    adapter = _sensevoice_adapter(result)
    stream_box = {}
    original = adapter._recognizer.create_stream

    def spy():
        stream = original()
        stream_box["stream"] = stream
        return stream

    adapter._recognizer.create_stream = spy
    adapter.transcribe_rich(_wav(1600), "zh-CN")     # 0.1 s 音频
    samples = stream_box["stream"].samples
    assert len(samples) == 1600 + int(asr_module.SAMPLE_RATE * 0.5)
    assert samples[-1] == 0.0


def test_tags_are_not_kept_on_the_adapter():
    """适配器是所有 node 共享的一个实例 —— 它身上不能留下上一句的情绪。"""
    adapter = _sensevoice_adapter(_FakeResult(
        "哈哈", lang="<|zh|>", emotion="<|HAPPY|>"))
    adapter.transcribe_rich(_wav(), "zh-CN")
    leaked = [name for name in vars(adapter)
              if name not in ("_recognizer",)]
    assert leaked == [], f"适配器实例上留下了状态：{leaked}"


# ── 发布路径 ─────────────────────────────────────────────────────────────────

class _TagAdapter(asr_module.ASRAdapter):
    def __init__(self, text="小范小范 你好", tags=None):
        self._text = text
        self._tags = tags or {"lang": "zh", "emotion": "HAPPY",
                              "audio_event": "Laughter"}

    def transcribe(self, wav_bytes, language):
        return self._text

    def transcribe_rich(self, wav_bytes, language):
        return self._text, dict(self._tags)


def _make_node(adapter, emit_audio_tags=True, kws_cfg=None):
    return asr_module._ASRNode(
        "/mic/audio", adapter=adapter, language="zh-CN",
        kws_cfg=kws_cfg if kws_cfg is not None else {"trigger_mode": "vad"},
        node_suffix="t", emit_audio_tags=emit_audio_tags,
    )


def _run_one_utterance(node, timeout=5.0) -> dict:
    """喂一段语音给 worker，返回它发出的两条 topic 上的 payload。"""
    node._utterance_queue = queue.Queue()
    node._stop_event = threading.Event()
    node._worker_ready = threading.Event()
    worker = threading.Thread(target=node._worker_inner, daemon=True)
    worker.start()
    assert node._worker_ready.wait(timeout=timeout), "worker 没起来"
    utterance = struct.pack("<16000h", *([1000] * 16000))
    node._utterance_queue.put((utterance, 1.0, 2.0, 0))

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if node._pub.messages or node._background_pub.messages:
            break
        time.sleep(0.02)
    node._stop_event.set()
    worker.join(timeout=timeout)
    return {
        "asr": [json.loads(m) for m in node._pub.messages],
        "background": [json.loads(m) for m in node._background_pub.messages],
    }


def test_foreground_payload_carries_the_tags():
    published = _run_one_utterance(_make_node(_TagAdapter()))
    assert len(published["asr"]) == 1, published
    payload = published["asr"][0]
    assert payload["text"] == "小范小范 你好"
    assert payload["emotion"] == "HAPPY"
    assert payload["audio_event"] == "Laughter"
    assert payload["lang"] == "zh"


def test_emit_audio_tags_off_returns_to_text_only():
    node = _make_node(_TagAdapter(), emit_audio_tags=False)
    payload = _run_one_utterance(node)["asr"][0]
    assert payload["text"] == "小范小范 你好"
    for key in ("emotion", "audio_event", "lang"):
        assert key not in payload


def test_a_plain_adapter_adds_no_keys():
    node = _make_node(_PlainAdapter())
    payload = _run_one_utterance(node)["asr"][0]
    assert payload["text"].strip() == "你好"
    for key in ("emotion", "audio_event", "lang"):
        assert key not in payload


def test_background_payload_carries_the_tags():
    """旁边那个人的笑声/哭声，对后台 subagent 一样是信息。"""
    node = _make_node(
        _TagAdapter(text="今天天气不错"),
        kws_cfg={"trigger_mode": "asr_kws", "asr_kws_keyword": "小范小范"})
    published = _run_one_utterance(node)
    assert published["asr"] == [], "没说唤醒词的话不该进主 agent"
    assert len(published["background"]) == 1, published
    payload = published["background"][0]
    assert payload["priority"] == 0
    assert payload["emotion"] == "HAPPY"
    assert payload["audio_event"] == "Laughter"


def test_background_publish_without_tags_is_unchanged():
    """老调用方式（不传 audio_tags）必须还能用。"""
    node = _make_node(_TagAdapter())
    node._publish_background("你好", 1.0, 2.0, b"\x00" * 3200, {})
    payload = json.loads(node._background_pub.messages[0])
    assert payload["text"] == "你好"
    assert payload["background"] is True
    for key in ("emotion", "audio_event", "lang"):
        assert key not in payload


# ── 配置 ─────────────────────────────────────────────────────────────────────

def test_schema_declares_the_switch_only_for_sensevoice():
    schema = None
    for tool in asr_module.TOOLS:
        if "configSchema" in tool:
            schema = tool["configSchema"]["properties"]
            break
    assert schema is not None, "没找到带 configSchema 的工具申报"
    field = schema["emit_audio_tags"]
    assert field["default"] is True
    assert field["scope"] == "shared"
    assert field["x-show-when"] == {"asr_model": ["sensevoice-small"]}
