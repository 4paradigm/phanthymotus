"""Wake direction is attached only to a leading wake word's capture window."""

from vision_stubs import PERCEPTION_ROOT, _FakeExecutor  # noqa: F401

from plugins import asr
import sys
import types
from types import SimpleNamespace

import pytest


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


@pytest.mark.parametrize('prefix', ['嗯', '那个', '就是', '先说别的'])
@pytest.mark.parametrize('candidates', [
    [window(-45, 1700000000000000, 1700000000020000)],
    [window(90, 1700000000800000, 1700000000820000)],
    [window(-45, 1700000000000000, 1700000000020000),
     window(90, 1700000000800000, 1700000000820000)],
])
def test_prefix_without_audio_alignment_never_supplies_wake_direction(prefix, candidates):
    text = prefix + '，小范小范'
    char_ends = list(range(1, len(prefix) + 1)) + list(range(len(prefix) + 2, len(text) + 1))
    assert asr._wake_direction_fields(text, char_ends, len(char_ends), 4,
                                      1700000000.0, 0, candidates, True) == {
        'wake_direction_state': 'unknown'}


@pytest.mark.parametrize('prefix', ['嗯', '那个', '先说别的'])
def test_aligned_wake_uses_keyword_time_instead_of_prefix_time(prefix):
    text = prefix + '，小范小范'
    char_ends = list(range(1, len(prefix) + 1)) + list(range(len(prefix) + 2, len(text) + 1))
    alignment = {'tokens': list(prefix + '小范小范'),
                 'timestamps': [0.2] * len(prefix) + [1.0, 1.1, 1.2, 1.3],
                 'duration_s': 2.0}
    fields = asr._wake_direction_fields(text, char_ends, len(char_ends), 4,
        1700000000.0, 16000,
        [window(-45, 1700000000500000, 1700000000600000),
         window(90, 1700000001000000, 1700000001100000)], True,
        alignment=alignment)
    assert fields['wake_direction_state'] == 'valid'
    assert fields['wake_direction_deg'] == 90
    assert fields['wake_direction_window_start_us'] == 1700000001000000


@pytest.mark.parametrize('alignment', [
    {'tokens': ['嗯', '小范'], 'timestamps': [0.2], 'duration_s': 2},
    {'tokens': ['啊', '小范'], 'timestamps': [0.2, 1.0], 'duration_s': 2},
    {'tokens': ['嗯小范'], 'timestamps': [0.2], 'duration_s': 2},
    {'tokens': ['嗯', '小范'], 'timestamps': [0.2, float('nan')], 'duration_s': 2},
    {'tokens': ['嗯', '小范'], 'timestamps': [0.2, 3.0], 'duration_s': 2},
    {'tokens': ['嗯', '小范'], 'timestamps': [1.0, 0.2], 'duration_s': 2},
])
def test_unusable_alignment_does_not_guess_prefix_duration(alignment):
    assert asr._wake_direction_fields('嗯，小范', [1, 3, 4], 3, 2,
        1700000000.0, 0, [window(90, 1700000000000000, 1700000000020000)],
        True, alignment=alignment) == {'wake_direction_state': 'unknown'}


def test_english_token_boundaries_map_to_custom_wake_time():
    alignment = {'tokens': ['▁well', '▁little', '▁fan'],
                 'timestamps': [0.2, 0.8, 1.0], 'duration_s': 2.0}
    assert asr._aligned_wake_offset('well, Little Fan', 4, alignment) == 0.8


@pytest.mark.parametrize('prefix', ['嗯', '先说别的'])
@pytest.mark.parametrize('emit_tags', [True, False])
@pytest.mark.parametrize('evict_during_decode', [True, False])
def test_worker_publishes_aligned_keyword_direction_after_prefix(monkeypatch, prefix, emit_tags, evict_during_decode):
    import json
    import queue

    text = prefix + '，小范小范你好'
    result = SimpleNamespace(text=text, lang='<|zh|>',
        tokens=list(prefix + '小范小范你好'),
        timestamps=[0.5] * len(prefix) + [1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    decoded = []
    adapter = asr.SherpaOnnxSenseVoiceAdapter.__new__(asr.SherpaOnnxSenseVoiceAdapter)
    adapter._recognizer = SimpleNamespace(
        create_stream=lambda: SimpleNamespace(result=result, accept_waveform=lambda *args: None),
        decode_streams=lambda streams: decoded.append(streams))
    node = asr._ASRNode('/mic', adapter, 'zh-CN', direction_topic='/direction',
        emit_audio_tags=emit_tags,
        kws_cfg={'trigger_mode': 'asr_kws', 'asr_kws_keyword': '小范小范'})
    node._direction_windows.extend([
        window(-45, 1700000000500000, 1700000000600000),
        window(90, 1700000001000000, 1700000001100000)])

    def decode(streams):
        decoded.append(streams)
        if evict_during_decode:
            node._direction_cb(SimpleNamespace(state=1, angle_deg=-90,
                window_start_us=1700000120000000, window_end_us=1700000120050000))

    adapter._recognizer.decode_streams = decode
    node._stop_event = asr.threading.Event()
    node._worker_ready = asr.threading.Event()
    items = iter([(b'\0\0' * 32000, 1700000000.0, 1700000002.0, 16000, True)])

    def get(timeout):
        try:
            return next(items)
        except StopIteration:
            node._stop_event.set()
            raise queue.Empty

    def phones(value, with_positions=False):
        chars = [(char, i + 1) for i, char in enumerate(value) if char.isalnum()]
        return ([char for char, _ in chars], [end for _, end in chars]) if with_positions else [char for char, _ in chars]

    monkeypatch.setattr(asr, '_text_to_ipa', phones)
    monkeypatch.setattr(asr.time, 'time', lambda: 1900000000.0)
    node._utterance_queue = SimpleNamespace(get=get)
    node._worker_inner()
    payload = json.loads(node._pub.messages[0])
    assert payload['text'] == '你好'
    assert payload['wake_direction_state'] == 'valid'
    assert payload['wake_direction_deg'] == 90
    assert payload['wake_direction_window_start_us'] == 1700000001000000
    assert payload['audio_start_ts'] == 1700000000.0
    assert payload['asr_complete_ts'] == 1900000000.0
    assert ('lang' in payload) is emit_tags
    assert 'tokens' not in payload and 'timestamps' not in payload
    assert len(decoded) == 1


def test_long_utterance_retains_leading_wake_window_despite_later_direction_samples():
    node = asr._ASRNode('/mic', None, 'zh-CN', direction_topic='/direction')
    base_us = 1700000000000000
    # 32 seconds at Bumi's maximum 20 Hz, beyond one maximum-length VAD segment.
    for i in range(650):
        begin = base_us + i * 50000
        node._direction_cb(SimpleNamespace(state=1, angle_deg=90 if i < 20 else -45,
            window_start_us=begin, window_end_us=begin + 50000))
    fields = asr._wake_direction_fields('小范你好', [1, 2, 3, 4], 2, 2,
        base_us / 1000000, 0, list(node._direction_windows), True,
        alignment={'tokens': list('小范你好'), 'timestamps': [0.2, 0.4, 10, 25], 'duration_s': 30})
    assert fields['wake_direction_state'] == 'valid'
    assert fields['wake_direction_deg'] == 90


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


def test_unstamped_audio_chunk_uses_arrival_time_without_trusting_capture_clock(monkeypatch):
    node = asr._ASRNode.__new__(asr._ASRNode)
    node._stop_event = asr.threading.Event()
    node._first_chunk_event = asr.threading.Event()
    node._vad_proc = None
    queued = []
    node._pcm_queue = SimpleNamespace(put_nowait=queued.append)
    node._check_audio_contract = lambda fmt, data: data
    monkeypatch.setattr(asr.time, 'time', lambda: 1700000000.25)
    msg = SimpleNamespace(format='audio/pcm-16k', data=b'\0\0',
        header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0)))

    node._audio_cb(msg)

    assert queued == [(b'\0\0', 1700000000.25, False)]


def test_timestamp_trust_covers_each_vad_segment_including_early_speech(monkeypatch):
    class Stop:
        done = False

        def is_set(self):
            return self.done

    stop = Stop()

    class Input:
        pcm = b'\0\0' * 10000
        chunks = iter([(pcm, 1.0, False),  # idle, outside either utterance
                       (pcm, 2.0, False),  # early speech before VAD reports it
                       (pcm, 3.0, True),
                       (pcm, 4.0, True),   # first segment completes
                       (pcm, 5.0, False),  # idle again
                       (pcm, 6.0, True),   # second utterance
                       (pcm, 7.0, True),
                       (pcm, 8.0, True),   # third utterance
                       (pcm, 9.0, False)])  # untrusted completion timestamp

        def get(self, timeout):
            try:
                return next(self.chunks)
            except StopIteration:
                stop.done = True
                raise

    class Vad:
        frame = 0
        popped = False

        @property
        def front(self):
            return (SimpleNamespace(start=10000, samples=[0] * 20000)
                    if self.frame == 4 else
                    SimpleNamespace(start=50000 if self.frame == 7 else 70000,
                                    samples=[0] * 10000))

        def accept_waveform(self, samples):
            self.frame += 1
            self.popped = False

        def is_speech_detected(self):
            return self.frame in (3, 6, 8)

        def empty(self):
            return self.frame not in (4, 7, 9) or self.popped

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
    assert [item[4] for item in results if isinstance(item[0], bytes)] == [False, True, False]


@pytest.mark.parametrize('capture_trusted', [True, False, None])
def test_vad_first_sample_stamps_align_utterance_and_wake_direction(monkeypatch, capture_trusted):
    base = 1700000000.0
    stop = SimpleNamespace(done=False)
    stop.is_set = lambda: stop.done
    frames = iter(range(24))

    def get(timeout):
        try:
            i = next(frames)
        except StopIteration:
            stop.done = True
            raise
        item = (b'\0\0' * 512, base + i * 512 / asr.SAMPLE_RATE)
        return item if capture_trusted is None else (*item, capture_trusted)

    class Vad:
        frame = 0
        popped = False
        front = SimpleNamespace(start=0, samples=[0] * (20 * 512))
        def accept_waveform(self, samples): self.frame += 1
        def is_speech_detected(self): return self.frame <= 20
        def empty(self): return self.frame != 24 or self.popped
        def pop(self): self.popped = True

    sherpa = types.ModuleType('sherpa_onnx')
    sherpa.SileroVadModelConfig = lambda **kwargs: None
    sherpa.VadModelConfig = lambda **kwargs: None
    sherpa.VoiceActivityDetector = lambda *args, **kwargs: Vad()
    monkeypatch.setitem(sys.modules, 'sherpa_onnx', sherpa)
    monkeypatch.setattr('utils.model_downloader.ensure_model', lambda *args: None)
    results = []
    asr._vad_worker(SimpleNamespace(get=get), SimpleNamespace(put=results.append),
                    stop, 'sherpa_onnx', 0.5, 128, pre_roll_ms=0)
    utterance = next(item for item in results if isinstance(item[0], bytes))
    expected_start = base if capture_trusted is True else base - 0.032
    assert utterance[1] == pytest.approx(expected_start, abs=1e-6, rel=0)
    assert utterance[2] == pytest.approx(expected_start + 0.640, abs=1e-6, rel=0)
    if capture_trusted is True:
        fields = asr._wake_direction_fields('嗯，小范', [1, 3, 4], 3, 2,
            utterance[1], utterance[3],
            [window(-45, 1700000000160000, 1700000000190000),
             window(90, 1700000000200000, 1700000000240000)], True,
            trusted_audio=utterance[4],
            alignment={'tokens': ['嗯', '小', '范'],
                       'timestamps': [0.05, 0.2, 0.25], 'duration_s': 0.64})
        assert fields['wake_direction_deg'] == 90
        assert fields['wake_direction_window_start_us'] == 1700000000200000


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


def test_missing_optional_direction_interface_does_not_stop_asr(monkeypatch):
    message_module = types.ModuleType('audio_msgs.msg')
    message_module.AudioChunk = object
    monkeypatch.setitem(sys.modules, 'audio_msgs.msg', message_module)
    node = asr._ASRNode('/mic', object(), 'zh-CN', direction_topic='/direction')

    class Event:
        def set(self): pass
        def is_set(self): return False
        def wait(self, timeout=None): return False

    class Worker:
        pid = 123
        def __init__(self, *args, **kwargs): pass
        def start(self): pass

    monkeypatch.setattr(asr.threading, 'Event', Event)
    monkeypatch.setattr(asr.threading, 'Thread', Worker)
    monkeypatch.setattr(asr.multiprocessing, 'Event', Event)
    monkeypatch.setattr(asr.multiprocessing, 'Queue', lambda **kwargs: object())
    monkeypatch.setattr(asr.multiprocessing, 'Process', Worker)

    result = node._start_inner()
    assert result['state'] == 'running'
    assert node._direction_topic == ''
    assert [sub.topic for sub in node.subscriptions] == ['/mic']
