"""Wake direction is attached only to a leading wake word's capture window."""

from vision_stubs import PERCEPTION_ROOT  # noqa: F401

from plugins import asr
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
