"""The VAD circular buffer must be big enough for sherpa's force-cut to fire.

sherpa-onnx force-cuts an over-long utterance from the top of
``VoiceActivityDetector::AcceptWaveform``::

    if (buffer_.Size() > max_utterance_length_) {   // max_speech_duration
      model_->SetMinSilenceDuration(0.1);
      model_->SetThreshold(0.90);
    }

but ``CircularBuffer::Push`` resizes — and logs ``Overflow! ... Increase
capacity to:`` — as soon as ``n + size > capacity``
(``buffer_size_in_seconds``). ``Size()`` therefore cannot exceed
``max_utterance_length_`` unless ``Push`` has *already* overflowed, so setting
both to the same number makes the escape hatch structurally unreachable.

That is what shipped: both were 30. Counting out loud for 30 s produced the
overflow warning, a doubling of the buffer to 60 s that sherpa never gives
back, a 33.4 s segment rather than a 30 s one, and a spurious 1.1 s fragment
right after it (the force-cut leaves threshold 0.90 / min_silence 0.1 s in
place until the buffer drains).

These tests reproduce sherpa's own arithmetic rather than just asserting
``VAD_BUFFER_S > VAD_MAX_SPEECH_S``, so they stay honest if either constant
moves.
"""

from plugins.asr import SAMPLE_RATE, VAD_BUFFER_S, VAD_MAX_SPEECH_S

# Silero window/shift used at plugins/asr.py's VAD config (window_size=512).
WINDOW_SHIFT = 512


def _simulate_continuous_speech(max_speech_s, buffer_s):
    """Push windows of unbroken speech; report which guard trips first.

    Mirrors CircularBuffer::Push and the AcceptWaveform force-cut check. During
    speech nothing is popped, so Size() just grows one window_shift at a time.

    Returns ``('force_cut', n)`` or ``('overflow', n)`` for the sample count at
    which that happened.
    """
    capacity = int(SAMPLE_RATE * buffer_s)
    max_utterance_length = int(SAMPLE_RATE * max_speech_s)
    size = 0
    # Bound the loop well past either guard so a bug cannot hang the suite.
    for _ in range(int(SAMPLE_RATE * (buffer_s + max_speech_s) / WINDOW_SHIFT)):
        # AcceptWaveform checks before pushing this batch of windows.
        if size > max_utterance_length:
            return "force_cut", size
        if WINDOW_SHIFT + size > capacity:
            return "overflow", size
        size += WINDOW_SHIFT
    raise AssertionError("neither guard tripped")


def test_force_cut_fires_before_the_buffer_overflows():
    outcome, _ = _simulate_continuous_speech(VAD_MAX_SPEECH_S, VAD_BUFFER_S)
    assert outcome == "force_cut", (
        "the circular buffer overflows before sherpa can force-cut a long "
        "utterance; VAD_BUFFER_S must exceed VAD_MAX_SPEECH_S by more than "
        "one window"
    )


def test_equal_sizes_are_the_regression_this_guards_against():
    # The shipped-and-broken configuration, kept as the negative control: it is
    # what makes the test above more than a restatement of `>`.
    outcome, _ = _simulate_continuous_speech(30, 30)
    assert outcome == "overflow"


def test_force_cut_has_room_to_actually_cut():
    # The force-cut does not truncate on the spot — it tightens the VAD and
    # waits for 0.1 s of silence at threshold 0.90. The buffer has to hold the
    # speech that keeps arriving in the meantime, so leave at least a second of
    # headroom past the force-cut point rather than a single window.
    assert (VAD_BUFFER_S - VAD_MAX_SPEECH_S) * SAMPLE_RATE >= SAMPLE_RATE


def test_pcm_history_outlives_the_longest_possible_segment():
    # _vad_worker sizes PcmHistory at VAD_BUFFER_S + 1 so pre_roll() can still
    # address the samples immediately before a maximum-length segment.
    from plugins.asr import PcmHistory

    history = PcmHistory(SAMPLE_RATE * (VAD_BUFFER_S + 1))
    longest_segment = int(SAMPLE_RATE * VAD_BUFFER_S)
    pre_roll = SAMPLE_RATE // 2

    history.append(b"\x00\x00" * (pre_roll + longest_segment))
    got = history.pre_roll(
        segment_start=pre_roll,
        segment_samples=longest_segment,
        pre_roll_samples=pre_roll,
        silence_samples=0,
    )
    assert len(got) == pre_roll * 2
