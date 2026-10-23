"""Language policy through the unified TTS facade and real async worker.

Only ROS, model download and the acoustic engine are replaced. These tests
cover message routing/lifecycle, not GPU performance or audible quality.
"""
import json
import threading
from types import SimpleNamespace

import pytest

from vision_stubs import _FakeExecutor, _wait_until
import plugins.tts as facade
import plugins.vits2_tts_trt.plugin as base
from plugins.vits2_tts_trt.adapter import Vits2TensorRTAdapter
from plugins.vits2_tts_language.policy import LanguagePolicyError
from plugins.vits2_tts_language.plugin import _language_input, PreparedSpeech, _PreparedAdapterView


class _Adapter(Vits2TensorRTAdapter):
    def __init__(self):
        self.spoken = []
        self.prepared = []
        self.speed = 1.0

    def set_speed(self, value):
        self.speed = value

    def warmup(self):
        return 16

    def iter_text_chunks(self, text):
        self.prepared.append(text)
        yield text, ((1,), (2,), (3,))

    def synthesize_stream(self, text):
        # The adapter view must reuse each request's prepared chunks.
        self.spoken.extend(normalized for normalized, _ in self.iter_text_chunks(text))
        yield b"\x01\x00" * 16


@pytest.fixture
def make_plugin(monkeypatch):
    import utils.model_downloader as downloader
    monkeypatch.setattr(base, "SUBSCRIBER_SETTLE_MS", 0)
    monkeypatch.setattr(base, "SUBSCRIBER_WAIT_MS", 100)
    # Test-only: no wall-clock pacing is needed for fake PCM.
    monkeypatch.setattr(base, "FRAME_INTERVAL_MS", 0)
    states = []
    original_threads = set(threading.enumerate())
    completions = []
    monkeypatch.setattr(base, "_complete_action", lambda action_id, text, frames, interrupted:
                        completions.append((action_id, text, frames, interrupted)))

    def make(gated=False, fail=False, **config):
        state = SimpleNamespace(gate=threading.Event(), entered=threading.Event(), loads=0,
                                adapter=None, completions=completions)
        if not gated:
            state.gate.set()

        def ensure(*args, **kwargs):
            state.loads += 1
            state.entered.set()
            assert state.gate.wait(5), "test did not release load"
            if fail:
                raise RuntimeError("controlled download failure")
            return "/models/vits2/engines/jp61"

        def build(cfg):
            adapter = _Adapter()
            adapter.set_speed(cfg.get("speed", 1.0))
            state.adapter = adapter
            return adapter

        monkeypatch.setattr(downloader, "ensure_vits2_model", ensure)
        monkeypatch.setattr(base, "build_adapter", build)
        state.plugin = facade.TTSPlugin({"engine": "vits2-zh-en", **config}, _FakeExecutor())
        states.append(state)
        return state

    yield make
    for state in states:
        state.gate.set()
        state.plugin.dispatch("tts", {"action": "stop"})
    # Keep background loaders inside their mocks' lifetime, including cancelled
    # loads. Otherwise a released loader can call real build_adapter in the
    # next test after monkeypatch teardown.
    for thread in set(threading.enumerate()) - original_threads:
        if thread.name == "vits2-adapter-loader":
            thread.join(timeout=3)
            assert not thread.is_alive()


def call(state, action, **args):
    return state.plugin.dispatch("tts", {"action": action, **args})


def test_opt_in_prepare_runs_through_empty_language_config(make_plugin):
    state = make_plugin(prepare_on_config=True)
    result = call(state, "config", instance_id="evaluation")
    assert result["status"] == "configured"
    assert result["model_state"] == "ready"
    assert state.loads == 1
    assert state.adapter is not None
    assert call(state, "config", instance_id="evaluation")["model_state"] == "ready"
    assert state.loads == 1
    assert call(state, "start", instance_id="evaluation", input_topic="/say/evaluation")["state"] == "running"


def test_opt_in_prepare_keeps_language_config_and_loading_bounded(make_plugin, monkeypatch):
    monkeypatch.setattr(base, "_CONFIG_PREPARE_WAIT_S", 0.02)
    state = make_plugin(gated=True, prepare_on_config=True)
    first = call(state, "config", instance_id="evaluation", reply_language="en")
    assert first["state"] == "loading"
    assert first["reply_language"] == "en"
    second = call(state, "config", instance_id="evaluation", speed=0.8)
    assert second["state"] == "loading"
    assert state.loads == 1
    state.gate.set()
    assert _wait_until(lambda: state.adapter is not None
                       and state.adapter.speed == pytest.approx(0.8))
    assert call(state, "config", instance_id="evaluation")["model_state"] == "ready"
    assert state.loads == 1


def start(state, key="a"):
    call(state, "start", instance_id=key, input_topic=f"/say/{key}")
    assert _wait_until(lambda: call(state, "info", instance_id=key)["state"] == "running")


def finished(state, result):
    assert _wait_until(lambda: any(row[0] == result["action_id"] for row in state.completions))
    rows = [row for row in state.completions if row[0] == result["action_id"]]
    assert len(rows) == 1
    return rows[0]


def test_unified_schema_exposes_vits_only_preference_and_keeps_acp():
    tool = facade.TOOLS[0]
    pref = tool["configSchema"]["properties"]["reply_language"]
    assert pref["enum"] == ["auto", "zh", "en"]
    # Existing sidebar.js renders oneOf's const/title; it does not understand
    # enumNames. Do not require an agent-core change to show readable labels.
    assert pref["oneOf"] == [{"const": "auto", "title": "自动"},
                             {"const": "zh", "title": "中文"},
                             {"const": "en", "title": "English"}]
    assert pref["x-show-when"] == {"tts_engine": "vits2-zh-en"}
    assert tool["inputSchema"]["x-completion"]["actions"] == ["speak"]
    assert tool["inputSchema"]["x-resource"] == "mouth"
    assert "language" in tool["inputSchema"]["properties"]["action"]["enum"]


@pytest.mark.parametrize("mode,text,expected", [
    ("zh", "2026 10/ 20", "二零二六"),
    ("en", "2026 10/ 20", "October"),
    ("zh", "Qwen3", "千问"),
    ("en", "Qwen3", "Qwen three"),
    ("zh", "Qwen 3", "千问三"),
    ("en", "Qwen 3", "Qwen three"),
    ("auto", "Qwen 3", "千问三"),
    ("zh", "Qwen 2.5。", "千问二点五。"),
    ("auto", "Hello.", "Hello."),
    ("auto", "你好，Qwen3 is ready，然后继续。", "你好"),
    ("auto", "123", "一百二十三"),
])
def test_direct_speech_uses_panel_readings_and_completes_acp(make_plugin, mode, text, expected):
    state = make_plugin(reply_language=mode)
    start(state)
    result = call(state, "speak", instance_id="a", text=text)
    assert finished(state, result)[3] is False
    assert expected.lower() in " ".join(state.adapter.spoken).lower()
    assert len(state.adapter.prepared) == 1, "frontend must not run twice"
    assert finished(state, result)[1] == text, "ACP must retain original request text"
    assert state.plugin._impl._nodes["a"]._pub.messages[-1] == list(base.AUDIO_EOF_MAGIC)


@pytest.mark.parametrize("mode", ["zh", "en", "auto"])
@pytest.mark.parametrize("text,zh,en", [
    ("y = 2x-3", "y 等于二乘以 x 减三", "y equals two times x minus three"),
    ("y = 2x‑3", "y 等于二乘以 x 减三", "y equals two times x minus three"),
    ("y = 2x−3", "y 等于二乘以 x 减三", "y equals two times x minus three"),
    ("RTX‑4090", "R T X 四零九零", "R T X four zero nine zero"),
    ("1.2A", "一点二安培", "one point two amperes"),
    ("8.5GB", "八点五 G B", "eight point five gigabytes"),
    ("SNR≥28dB", "S N R 大于等于二十八分贝",
     "S N R greater than or equal to twenty-eight decibels"),
    ("lr=1e‑4", "L R 等于一乘以十的负四次方",
     "L R equals one times ten to the power of minus four"),
    ("NO.0917", "N O 零九一七", "N O zero nine one seven"),
    ("2/3", "三分之二", "two thirds"),
    ("x^2", "x 的平方", "x squared"),
    ("https://example.com", "H T T P S 冒号斜杠斜杠 E X A M P L E 点 C O M",
     "H T T P S colon slash slash E X A M P L E dot C O M"),
    ("user@example.com", "U S E R 艾特 E X A M P L E 点 C O M",
     "U S E R at E X A M P L E dot C O M"),
    ("https://example.com/2026/10/20?q=2/3",
     "H T T P S 冒号斜杠斜杠 E X A M P L E 点 C O M 斜杠二零二六斜杠一零斜杠二零问号 Q 等号二斜杠三",
     "H T T P S colon slash slash E X A M P L E dot C O M slash two zero two six slash one zero slash two zero question mark Q equals two slash three"),
    ("user+qwen3@example.com", "U S E R 加号 Q W E N 三艾特 E X A M P L E 点 C O M",
     "U S E R plus Q W E N three at E X A M P L E dot C O M"),
])
def test_symbolic_speech_uses_selected_reading_through_facade(make_plugin, mode, text, zh, en):
    state = make_plugin(reply_language=mode)
    start(state)
    result = call(state, "speak", instance_id="a", text=text)
    assert finished(state, result)[3] is False
    assert result["reply_language"] == ("en" if mode == "en" else "zh")
    assert state.adapter.spoken == [en if mode == "en" else zh]
    assert finished(state, result)[1] == text


def test_formula_loading_freezes_language_and_explicit_asr_still_wins(make_plugin):
    state = make_plugin(gated=True)
    call(state, "config", instance_id="a", reply_language="zh")
    call(state, "start", instance_id="a", input_topic="/say/a")
    first = call(state, "speak", instance_id="a", text="y = 2x‑3")
    call(state, "config", instance_id="a", reply_language="auto")
    turn = call(state, "language", instance_id="a",
                asr_result={"text": "y = 2x‑3", "lang": "en"})
    second = call(state, "speak", instance_id="a", text="RTX‑4090", turn_id=turn["turn_id"])
    state.gate.set()
    assert finished(state, first)[3] is False
    assert finished(state, second)[3] is False
    assert state.adapter.spoken == ["y 等于二乘以 x 减三", "R T X four zero nine zero"]


def test_auto_card_keeps_neutral_reading_within_instance_and_resets_on_stop(make_plugin):
    state = make_plugin(reply_language="auto")
    start(state, "a")
    start(state, "b")
    first = call(state, "speak", instance_id="a", text="Hello")
    assert first["reply_language"] == "en"
    finished(state, first)
    date = call(state, "speak", instance_id="a", text="2026 10/23")
    assert date["reply_language"] == "en"
    finished(state, date)
    assert "October twenty third" in state.adapter.spoken[-1]

    # A second card/instance has not received English evidence.
    other = call(state, "speak", instance_id="b", text="2026 10/23")
    assert other["reply_language"] == "zh"
    finished(state, other)
    assert "二零二六年十月二十三日" in state.adapter.spoken[-1]

    # Current Chinese evidence wins over the previous English context.
    greeting = call(state, "speak", instance_id="a", text="你好")
    assert greeting["reply_language"] == "zh"
    finished(state, greeting)
    following = call(state, "speak", instance_id="a", text="2/3")
    assert following["reply_language"] == "zh"
    finished(state, following)
    assert state.adapter.spoken[-1] == "三分之二"

    again = call(state, "speak", instance_id="a", text="Hello")
    finished(state, again)
    call(state, "stop", instance_id="a")
    start(state, "a")
    restarted = call(state, "speak", instance_id="a", text="2026 10/23")
    assert restarted["reply_language"] == "zh"
    finished(state, restarted)


@pytest.mark.parametrize("mode,user_text,reply,expected", [
    ("zh", "你好", "2026 10/23 message", "二零二六年十月二十三日 message"),
    ("en", "Hello", "2026 10/23 星期五", "October twenty third, two thousand twenty-six 星期五"),
    ("zh", "你好", "Current is 1.2A", "一点二安培"),
])
def test_explicit_turn_checks_primary_script_after_supported_fields(make_plugin, mode, user_text, reply, expected):
    state = make_plugin(reply_language=mode)
    start(state)
    turn = call(state, "language", instance_id="a", text=user_text)
    result = call(state, "speak", instance_id="a", text=reply, turn_id=turn["turn_id"])
    finished(state, result)
    assert expected in state.adapter.spoken[-1]
    assert result["reply_language"] == mode


def test_explicit_turn_still_rejects_foreign_prose_without_selected_reading(make_plugin):
    state = make_plugin(reply_language="zh")
    start(state)
    turn = call(state, "language", instance_id="a", text="你好")
    with pytest.raises(LanguagePolicyError, match="selected primary language"):
        call(state, "speak", instance_id="a", text="Hello, the robot is ready.", turn_id=turn["turn_id"])
    assert state.adapter.spoken == []


@pytest.mark.parametrize("mode,text,expected", [
    ("zh", "2026 10/23 message", ("二零二六年十月二十三日", "message")),
    ("en", "2026 10/23 星期五", ("October twenty third", "two thousand twenty-six", "星期五")),
    ("zh", "message 2026-10-23 received", ("message", "二零二六年十月二十三日", "received")),
    ("en", "开始2026-10-23结束", ("开始", "October twenty third", "two thousand twenty-six", "结束")),
    ("zh", "ratio 2/3 confirmed", ("ratio", "三分之二", "confirmed")),
    ("en", "电流1.2A已确认", ("电流", "one point two amperes", "已确认")),
])
def test_fixed_mode_reads_fields_in_selected_language_without_translating_prose(
        make_plugin, mode, text, expected):
    state = make_plugin(reply_language=mode)
    start(state)
    result = call(state, "speak", instance_id="a", text=text)
    assert result["reply_language"] == mode
    assert finished(state, result)[3] is False
    assert finished(state, result)[1] == text
    assert len(state.adapter.prepared) == 1, "accepted text must be prepared once"
    spoken = " ".join(state.adapter.spoken)
    compact = lambda value: "".join(value.split())
    for fragment in expected:
        assert compact(fragment) in compact(spoken)
    assert not any(char.isdecimal() for char in spoken)


@pytest.mark.parametrize("text", [
    "The robot has 12 sensors, 它有3个摄像头。",
    "日期是2026-10-23，the current is 1.2A.",
])
def test_auto_prose_keeps_original_fields_for_local_frontend_context(make_plugin, text):
    state = make_plugin(reply_language="auto")
    start(state)
    result = call(state, "speak", instance_id="a", text=text)
    assert finished(state, result)[3] is False
    # The fake adapter records the request handed to the actual local-language
    # frontend. Auto must not pre-read every field in one global language.
    assert state.adapter.prepared == [text]
    assert state.adapter.spoken == [text]


def test_loading_preserves_each_fixed_mode_for_fields_inside_prose(make_plugin):
    state = make_plugin(gated=True, reply_language="zh")
    call(state, "start", instance_id="a", input_topic="/say/a")
    first_text, second_text = "2026 10/23 message", "2026 10/23 星期五"
    first = call(state, "speak", instance_id="a", text=first_text)
    call(state, "config", instance_id="a", reply_language="en")
    second = call(state, "speak", instance_id="a", text=second_text)
    state.gate.set()
    assert finished(state, first)[3] is False
    assert finished(state, second)[3] is False
    assert first["reply_language"] == "zh"
    assert second["reply_language"] == "en"
    assert len(state.adapter.prepared) == 2
    assert "二零二六年十月二十三日" in state.adapter.spoken[0]
    assert "message" in state.adapter.spoken[0]
    assert "October twenty third" in state.adapter.spoken[1]
    assert "星期五" in state.adapter.spoken[1]
    assert finished(state, first)[1] == first_text
    assert finished(state, second)[1] == second_text


@pytest.mark.parametrize("language,text,expected", [
    ("zh", "请查看2026 10/23 message", "二零二六年十月二十三日"),
    ("en", "Please read 2026 10/23 星期五", "October twenty third"),
])
def test_explicit_current_turn_reads_embedded_date_in_resolved_language(
        make_plugin, language, text, expected):
    state = make_plugin(reply_language="auto")
    start(state)
    turn = call(state, "language", instance_id="a",
                asr_result={"text": "2026 10/23", "lang": language})
    result = call(state, "speak", instance_id="a", text=text, turn_id=turn["turn_id"])
    assert result["reply_language"] == language
    assert finished(state, result)[3] is False
    assert expected in " ".join(state.adapter.spoken)
    assert ("message" if language == "zh" else "星期五") in " ".join(state.adapter.spoken)
    assert finished(state, result)[1] == text


@pytest.mark.parametrize("mode,text", [
    ("zh", "WAIC NVIDIA OpenMLDB AI API CAR Agentic"),
    ("zh", "Hello, the robot is ready."),
    ("en", "你好，小方。"),
    ("zh", "你好，Qwen 3 is ready，然后继续。"),
])
def test_direct_speech_treats_panel_language_as_reading_preference(make_plugin, mode, text):
    state = make_plugin(reply_language=mode)
    start(state)
    result = call(state, "speak", instance_id="a", text=text)
    assert result["reply_language"] == mode
    assert finished(state, result)[3] is False
    assert finished(state, result)[1] == text
    assert state.adapter.spoken
    assert len(state.adapter.prepared) == 1


def test_literal_speech_exception_does_not_relax_explicit_dialogue(make_plugin):
    state = make_plugin(reply_language="zh")
    start(state)
    turn = call(state, "language", instance_id="a", text="你好")
    with pytest.raises(LanguagePolicyError, match="selected primary language"):
        call(state, "speak", instance_id="a", text="The robot is ready.", turn_id=turn["turn_id"])
    with pytest.raises(LanguagePolicyError) as missing:
        call(state, "speak", instance_id="a", text="你好。")
    assert missing.value.code == "stale_turn"
    assert state.adapter.spoken == []


@pytest.mark.parametrize("text", ["🙂", "?!", "สวัสดี"])
def test_literal_speech_still_rejects_unspoken_or_unsupported_input(make_plugin, text):
    state = make_plugin(reply_language="zh")
    start(state)
    with pytest.raises(LanguagePolicyError):
        call(state, "speak", instance_id="a", text=text)
    assert state.adapter.spoken == []


def test_loading_keeps_each_instances_language_context_and_own_topic(make_plugin):
    state = make_plugin(gated=True)
    for key, mode in (("a", "zh"), ("b", "en")):
        call(state, "config", instance_id=key, reply_language=mode)
        assert call(state, "start", instance_id=key, input_topic=f"/say/{key}")["state"] == "loading"
    a = call(state, "speak", instance_id="a", text="Qwen3")
    b = call(state, "speak", instance_id="b", text="Qwen3")
    # Updating a setting cannot relabel speech already accepted into a queue.
    call(state, "config", instance_id="a", reply_language="en")
    state.gate.set()
    assert finished(state, a)[3] is False
    assert finished(state, b)[3] is False
    assert any("千问" in text for text in state.adapter.spoken)
    assert any("Qwen three" in text for text in state.adapter.spoken)
    assert state.loads == 1
    for key in ("a", "b"):
        node = state.plugin._impl._nodes[key]
        assert node._output_topic == f"/say/{key}/tts"


def test_explicit_instance_does_not_use_another_running_node(make_plugin):
    state = make_plugin()
    start(state, "a")
    start(state, "b")
    a = state.plugin._impl._nodes["a"]
    b = state.plugin._impl._nodes["b"]
    before_a = len(a._pub.messages)
    result = call(state, "speak", instance_id="b", text="Hello.")
    assert finished(state, result)[3] is False
    assert len(a._pub.messages) == before_a
    assert len(b._pub.messages) > 0


def test_empty_darvin_config_does_not_restart_loading_or_invalidate_turn(make_plugin):
    state = make_plugin(gated=True)
    call(state, "start", instance_id="a", input_topic="/say/a")
    assert state.entered.wait(2)
    turn = call(state, "language", instance_id="a", text="hello")
    generation = state.plugin._impl._load_generation
    cfg = dict(instance_id="a", api_key="", model="", voice="", url="")
    assert call(state, "config", **cfg)["status"] == "configured"
    assert state.plugin._impl._load_generation == generation
    assert call(state, "info", instance_id="a")["turn_id"] == turn["turn_id"]
    result = call(state, "speak", instance_id="a", text="Hello.", turn_id=turn["turn_id"])
    state.gate.set()
    assert finished(state, result)[3] is False
    assert state.loads == 1


def test_dialogue_uses_current_asr_label_and_rejects_stale_missing_tokens(make_plugin):
    state = make_plugin()
    start(state)
    old = call(state, "language", instance_id="a", asr_result={"text": "你好 hello", "lang": "en-US"})
    assert old["reply_language"] == "en" and old["language_source"] == "asr"
    new = call(state, "language", instance_id="a", asr_result={"text": "123", "lang": "zh"})
    for token in ({}, {"turn_id": old["turn_id"]}, {"turn_id": None}):
        with pytest.raises(LanguagePolicyError):
            call(state, "speak", instance_id="a", text="你好。", **token)
    result = call(state, "speak", instance_id="a", text="你好，Hello。", turn_id=new["turn_id"])
    assert finished(state, result)[3] is False


@pytest.mark.parametrize("asr,expected", [
    ({"text": "hello", "language": "zh"}, None),
    ({"text": "hello", "lang": "unknown"}, None),
    ({"text": "hello", "lang": "en_GB"}, "en"),
])
def test_asr_reads_detected_lang_only(asr, expected):
    assert _language_input({"asr_result": asr}) == ("hello", expected)


@pytest.mark.parametrize("payload", [
    {"asr_result": {"text": "hello", "lang": 3}},
    {"asr_result": {"text": "hello", "lang": "en"}, "input_language": "zh"},
    {"asr_result": {"text": "hello"}, "text": "different"},
])
def test_conflicting_asr_input_is_rejected(payload):
    with pytest.raises(LanguagePolicyError):
        _language_input(payload)


@pytest.mark.parametrize("action", ["stop", "interrupt"])
def test_cancel_while_loading_completes_action_once(make_plugin, action):
    state = make_plugin(gated=True)
    result = call(state, "speak", instance_id="a", text="Hello.")
    call(state, action, instance_id="a")
    assert finished(state, result)[3] is True
    state.gate.set()
    assert state.adapter is None or not state.adapter.spoken


def test_load_failure_completes_deferred_language_speech(make_plugin):
    state = make_plugin(gated=True, fail=True)
    result = call(state, "speak", instance_id="a", text="你好。")
    state.gate.set()
    assert finished(state, result)[3] is True
    assert call(state, "info", instance_id="a")["state"] == "error"


def test_interrupt_idle_does_not_drop_next_language_reply(make_plugin):
    state = make_plugin()
    start(state)
    call(state, "interrupt", instance_id="a")
    result = call(state, "speak", instance_id="a", text="Hello.")
    assert finished(state, result)[3] is False


def test_ros_topic_has_direct_and_explicit_dialogue_contract(make_plugin):
    state = make_plugin()
    start(state)
    node = state.plugin._impl._nodes["a"]
    node._text_callback(SimpleNamespace(data="Hello."))
    assert _wait_until(lambda: state.adapter.spoken == ["Hello."])
    turn = call(state, "language", instance_id="a", text="你好")
    node._text_callback(SimpleNamespace(data=json.dumps({"text": "你好。"})))
    assert node.policy.snapshot()["last_language_error"]["code"] == "stale_turn"
    node._text_callback(SimpleNamespace(data=json.dumps({"text": "你好。", "turn_id": turn["turn_id"]})))
    assert _wait_until(lambda: state.adapter.spoken[-1] == "你好。")


def test_instance_preference_does_not_change_facade_shared_default(make_plugin):
    state = make_plugin(reply_language="zh")
    call(state, "config", instance_id="a", reply_language="English")
    assert state.plugin._cfg["reply_language"] == "zh"
    start(state, "a")
    start(state, "b")
    assert call(state, "info", instance_id="a")["reply_language"] == "en"
    assert call(state, "info", instance_id="b")["reply_language"] == "zh"


@pytest.mark.parametrize("config", [{"speed": float("nan")}, {"speed": 0}, {"speaker_id": 1}, {"reply_language": "fr"}])
def test_invalid_config_does_not_change_live_language(make_plugin, config):
    state = make_plugin(reply_language="zh")
    start(state)
    before = call(state, "info", instance_id="a")
    before_config = dict(state.plugin._cfg)
    with pytest.raises((LanguagePolicyError, ValueError)):
        call(state, "config", **config)
    assert call(state, "info", instance_id="a")["reply_language"] == before["reply_language"]
    assert state.plugin._cfg == before_config


def test_prepared_request_preserves_real_adapter_pcm_rng_lock_and_speed():
    class Engine:
        max_text_tokens = 256

        def __init__(self):
            self.reset_count = 0
            self.calls = []
            self.adapter = None

        def reset_random_state(self):
            self.reset_count += 1

        def synthesize(self, text, *, text_ids, length_scale):
            self.calls.append((text, text_ids, length_scale, self.adapter._lock.locked()))
            return bytes([sum(text_ids[0]), 0]) * 2401

    chunks = (("Hello.", ((1, 2), (0, 0), (1, 1))), ("你好。", ((3,), (1,), (0,))))
    baseline_engine, prepared_engine = Engine(), Engine()
    baseline = Vits2TensorRTAdapter(speed=0.95, engine=baseline_engine)
    prepared = Vits2TensorRTAdapter(speed=0.95, engine=prepared_engine)
    baseline_engine.adapter, prepared_engine.adapter = baseline, prepared
    baseline.iter_text_chunks = lambda text: iter(chunks)
    prepared.iter_text_chunks = lambda text: pytest.fail("prepared frontend must not rerun")
    text = "Hello. 你好。"
    expected = list(baseline.synthesize_stream(text))
    view = _PreparedAdapterView(prepared, PreparedSpeech(text, "en", "test-token", chunks))
    actual = list(Vits2TensorRTAdapter.synthesize_stream(view, text))
    assert actual == expected
    assert prepared_engine.calls == baseline_engine.calls
    assert prepared_engine.reset_count == baseline_engine.reset_count == 1
    assert all(call[3] is True for call in prepared_engine.calls)


def test_language_preference_survives_engine_switch_without_affecting_other_engine(make_plugin, monkeypatch):
    class OtherEngine:
        def __init__(self, cfg, executor):
            self.cfg = dict(cfg)

        def dispatch(self, name, args):
            return {"state": "idle", "status": "configured"}

    monkeypatch.setattr(facade, "SherpaOnnxTTSPlugin", OtherEngine)
    state = make_plugin()
    start(state)
    call(state, "config", reply_language="en")
    old_turn = call(state, "language", instance_id="a", text="Hello.")
    call(state, "config", tts_engine="mms-th")
    assert isinstance(state.plugin._impl, OtherEngine)
    with pytest.raises(ValueError, match="vits2-zh-en only"):
        call(state, "language", instance_id="a", text="Hello.")
    call(state, "config", tts_engine="vits2-zh-en")
    start(state)
    assert call(state, "info", instance_id="a")["reply_language"] == "en"
    with pytest.raises(LanguagePolicyError):
        call(state, "speak", instance_id="a", text="Hello.", turn_id=old_turn["turn_id"])
    result = call(state, "speak", instance_id="a", text="Hello.")
    assert finished(state, result)[3] is False


@pytest.mark.parametrize("mode,expected", [("zh", "二千零二十六"), ("en", "two thousand")])
def test_ros_bare_number_is_text_not_json_envelope(make_plugin, mode, expected):
    state = make_plugin(reply_language=mode)
    start(state)
    node = state.plugin._impl._nodes["a"]
    node._text_callback(SimpleNamespace(data="2026"))
    assert _wait_until(lambda: bool(state.adapter.spoken))
    assert expected in state.adapter.spoken[0]
    assert node.policy.snapshot()["last_language_error"] is None


def test_speed_update_does_not_block_info_or_interrupt(make_plugin, monkeypatch):
    state = make_plugin()
    start(state)
    entered, release, responded = threading.Event(), threading.Event(), threading.Event()
    failures, responses = [], []

    def slow_speed(value):
        entered.set()
        assert release.wait(3)

    def update_speed():
        try:
            call(state, "config", speed=0.9)
        except Exception as error:
            failures.append(error)

    def observe_and_interrupt():
        try:
            responses.append(call(state, "info", instance_id="a"))
            responses.append(call(state, "interrupt", instance_id="a"))
        except Exception as error:
            failures.append(error)
        finally:
            responded.set()

    monkeypatch.setattr(state.adapter, "set_speed", slow_speed)
    updater = threading.Thread(target=update_speed)
    reader = threading.Thread(target=observe_and_interrupt)
    updater.start()
    try:
        assert entered.wait(1)
        reader.start()
        assert responded.wait(0.5), "language lock must not cover adapter.set_speed"
    finally:
        release.set()
        updater.join(2)
        if reader.ident is not None:
            reader.join(2)
    assert not failures
    assert responses[0]["state"] == "running"
    assert responses[1]["status"] == "interrupted"


@pytest.mark.parametrize("source", ["speak", "ros", "loading"])
def test_interrupt_during_frontend_discards_old_request(make_plugin, monkeypatch, source):
    state = make_plugin(gated=source == "loading")
    entered, release = threading.Event(), threading.Event()
    original = _Adapter.iter_text_chunks
    results, failures = [], []

    def slow_frontend(adapter, text):
        entered.set()
        assert release.wait(3)
        yield from original(adapter, text)

    monkeypatch.setattr(_Adapter, "iter_text_chunks", slow_frontend)

    def submit():
        try:
            if source == "ros":
                state.plugin._impl._nodes["a"]._text_callback(SimpleNamespace(data="Hello."))
            else:
                results.append(call(state, "speak", instance_id="a", text="Hello."))
        except Exception as error:
            failures.append(error)

    thread = None
    if source == "loading":
        submit()
        state.gate.set()
    else:
        start(state)
        thread = threading.Thread(target=submit)
        thread.start()
    try:
        assert entered.wait(2)
        assert call(state, "interrupt", instance_id="a")["status"] == "interrupted"
    finally:
        release.set()
        if thread is not None:
            thread.join(2)
    assert not failures
    if results:
        assert finished(state, results[0])[3] is True
    else:
        assert _wait_until(lambda: bool(state.completions))
        assert state.completions[-1][3] is True
    assert state.adapter.spoken == []


@pytest.mark.parametrize("mode,text,expected", [
    ("zh", "2026 10/ 20", "二零二六"),
    ("en", "2026 10/ 20", "October"),
    ("zh", "Qwen3", "千问"),
    ("en", "Qwen3", "Qwen three"),
    ("zh", "Qwen 3", "千问三"),
    ("en", "Qwen 3", "Qwen three"),
    ("auto", "Qwen 3", "千问三"),
])
def test_http_audition_applies_shared_preference_without_changing_turn(make_plugin, mode, text, expected):
    state = make_plugin(reply_language=mode)
    start(state)
    call(state, "language", instance_id="a", text="Hello.")
    before = call(state, "info", instance_id="a")
    # The HTTP endpoint has no instance_id: a per-instance override must not
    # replace the configured shared preference used for its audition.
    call(state, "config", instance_id="b", reply_language="en" if mode == "zh" else "zh")
    pcm = state.plugin.synthesize_raw(text)
    assert pcm == b"\x01\x00" * 16
    assert expected in state.adapter.spoken[-1]
    assert len(state.adapter.prepared) == 1
    assert call(state, "info", instance_id="a")["turn_id"] == before["turn_id"]


def test_ros_frontend_exception_is_contained_and_reported(make_plugin, monkeypatch):
    state = make_plugin()
    start(state)

    def broken(text):
        raise ValueError("protected span cannot fit the engine token limit")

    monkeypatch.setattr(state.adapter, "iter_text_chunks", broken)
    node = state.plugin._impl._nodes["a"]
    node._text_callback(SimpleNamespace(data="Hello."))
    assert node.policy.snapshot()["last_language_error"]["code"] == "frontend_error"
    assert state.adapter.spoken == []


def test_speak_response_keeps_request_turn_during_concurrent_language_update(make_plugin, monkeypatch):
    state = make_plugin()
    start(state)
    old = call(state, "language", instance_id="a", text="Hello.")
    original = state.adapter.iter_text_chunks

    def change_turn_during_frontend(text):
        call(state, "language", instance_id="a", text="你好")
        yield from original(text)

    monkeypatch.setattr(state.adapter, "iter_text_chunks", change_turn_during_frontend)
    result = call(state, "speak", instance_id="a", text="Hello.", turn_id=old["turn_id"])
    assert result["turn_id"] == old["turn_id"]
    assert result["reply_language"] == "en"
    assert finished(state, result)[3] is False
