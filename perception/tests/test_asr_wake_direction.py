"""Wake direction is attached only to a leading wake word's capture window."""

from vision_stubs import PERCEPTION_ROOT, _FakeExecutor  # noqa: F401

from plugins import asr
import sys
import types
from types import SimpleNamespace


def window(angle, start, end):
    return (angle, start, end)


def test_leading_wake_matches_voice_onset_after_preroll():
    fields = asr._wake_direction_fields(
        "小范小范，你好", list(range(1, 7)), 4, 4,
        1700000000.0, 16000,
        [window(90, 1700000000500000, 1700000000600000)], True)
    assert fields == {
        "wake_direction_state": "valid", "wake_direction_deg": 90,
        "wake_direction_window_start_us": 1700000000500000,
        "wake_direction_window_end_us": 1700000000600000,
    }


def test_filler_prefix_allowed_but_mid_sentence_not_matched():
    candidate = [window(-45, 1700000000000000, 1700000000020000)]
    assert asr._wake_direction_fields("嗯，小范小范", list(range(1, 8)), 5, 4,
                                      1700000000.0, 0, candidate, True)[
        "wake_direction_state"] == "valid"
    assert asr._wake_direction_fields("先说别的，小范小范", list(range(1, 12)), 8, 4,
                                      1700000000.0, 0, candidate, True) == {
        "wake_direction_state": "unknown"}


def test_circular_angles_and_conflicts():
    near_back = [window(179, 1700000000000000, 1700000000020000),
                 window(-179, 1700000000020000, 1700000000040000)]
    fields = asr._wake_direction_fields("小范", [1, 2], 2, 2,
                                        1700000000.0, 0, near_back, True)
    assert fields["wake_direction_deg"] == 180
    conflict = [window(0, 1700000000000000, 1700000000020000),
                window(180, 1700000000020000, 1700000000040000)]
    assert asr._wake_direction_fields("小范", [1, 2], 2, 2,
                                      1700000000.0, 0, conflict, True) == {
        "wake_direction_state": "unknown"}


def test_missing_input_or_untrusted_windows_never_use_last_angle():
    stale = [window(90, 0, 0),
             window(180, 1700000001000000, 1700000001020000)]
    assert asr._wake_direction_fields("小范", [1, 2], 2, 2,
                                      1700000000.0, 0, stale, True) == {
        "wake_direction_state": "unknown"}
    assert asr._wake_direction_fields("小范", [1, 2], 2, 2,
                                      1700000000.0, 0, stale, False) == {
        "wake_direction_state": "no_input"}
    close = [window(90, 1700000000000000, 1700000000020000)]
    assert asr._wake_direction_fields("小范", [1, 2], 2, 2,
                                      1700000000.0, 0, close, True,
                                      trusted_audio=False) == {
        "wake_direction_state": "unknown"}


def test_input_bindings_identify_audio_and_direction_in_any_order():
    args = {"input_topic": "/direction", "input_topics": ["/direction", "/mic"],
            "input_bindings": [
                {"topic": "/direction", "format": "sensor/sound-direction-v1", "port_index": 1},
                {"topic": "/mic", "format": "audio/pcm-16k", "port_index": 0}]}
    assert asr._resolve_asr_inputs(args) == ("/mic", "/direction")
    assert asr._resolve_asr_inputs({"input_topic": "/mic"}) == ("/mic", "")
    assert asr._resolve_asr_inputs({"input_topics": ["/mic", "/direction"]}) == (
        "/mic", "/direction")


def test_direction_buffer_ignores_invalid_time_and_no_event_keeps_history():
    node = asr._ASRNode.__new__(asr._ASRNode)
    node._direction_windows = asr.deque(maxlen=50)
    node._direction_lock = asr.threading.Lock()
    node._direction_cb(SimpleNamespace(state=1, angle_deg=90,
        window_start_us=1700000000000000, window_end_us=1700000000020000))
    node._direction_cb(SimpleNamespace(state=0, angle_deg=0,
        window_start_us=0, window_end_us=0))
    node._direction_cb(SimpleNamespace(state=1, angle_deg=180,
        window_start_us=0, window_end_us=0))
    assert list(node._direction_windows) == [(90, 1700000000000000, 1700000000020000)]


def test_timestamp_trust_is_scoped_to_speech_chunks(monkeypatch):
    class Stop:
        done = False

        def is_set(self):
            return self.done

    stop = Stop()

    class Input:
        chunks = iter([(b'\0\0' * 1000, 1.0, False),  # idle, untrusted
                       (b'\0\0' * 1000, 2.0, True),   # first utterance
                       (b'\0\0' * 1000, 3.0, True),
                       (b'\0\0' * 1000, 4.0, True),   # idle again
                       (b'\0\0' * 1000, 5.0, False),  # second utterance
                       (b'\0\0' * 1000, 6.0, True)])

        def get(self, timeout):
            try:
                return next(self.chunks)
            except StopIteration:
                stop.done = True
                raise

    class Vad:
        frame = 0
        popped = False
        front = SimpleNamespace(start=0, samples=[0] * 10000)

        def accept_waveform(self, samples):
            self.frame += 1
            self.popped = False

        def is_speech_detected(self):
            return self.frame in (2, 5)

        def empty(self):
            return self.frame not in (3, 6) or self.popped

        def pop(self):
            self.popped = True

    fake_sherpa = types.ModuleType('sherpa_onnx')
    fake_sherpa.SileroVadModelConfig = lambda **kwargs: None
    fake_sherpa.VadModelConfig = lambda **kwargs: None
    fake_sherpa.VoiceActivityDetector = lambda *args, **kwargs: Vad()
    monkeypatch.setitem(sys.modules, 'sherpa_onnx', fake_sherpa)
    monkeypatch.setattr('utils.model_downloader.ensure_model', lambda *args: None)
    results = []
    asr._vad_worker(Input(), SimpleNamespace(put=results.append), stop,
                    'sherpa_onnx', 0.5, 0, pre_roll_ms=0)
    assert [item[4] for item in results if isinstance(item[0], bytes)] == [True, False]


def test_rewiring_a_running_asr_resubscribes_to_the_new_direction(monkeypatch):
    monkeypatch.setattr(asr, '_build_asr_adapter', lambda cfg: object())

    def start_pipeline(node):
        node._sub = node.create_subscription(object, node._input_topic, None, None)
        if node._direction_topic:
            node._direction_sub = node.create_subscription(
                object, node._direction_topic, None, None)
        node.state = 'running'
        return {'state': 'running'}

    monkeypatch.setattr(asr._ASRNode, '_start_inner', start_pipeline)
    plugin = asr.ASRPlugin({}, _FakeExecutor())

    def start(direction):
        return plugin.dispatch('asr', {'action': 'start', 'instance_id': 'card-1',
            'input_bindings': [
                {'topic': direction, 'format': 'sensor/sound-direction-v1', 'port_index': 1},
                {'topic': '/mic', 'format': 'audio/pcm-16k', 'port_index': 0}]})

    assert start('/direction/old')['state'] == 'running'
    node = plugin._nodes['card-1']
    assert start('/direction/new')['state'] == 'running'
    assert node.state == 'running'
    assert [sub.topic for sub in node.subscriptions] == ['/mic', '/direction/new']
