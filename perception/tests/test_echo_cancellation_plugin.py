"""Host-side tests for the echo cancellation card — no ROS, no audio hardware.

Two halves. The first drives `_DelayTracker` with synthetic signals, which is
the only way the thresholds in it can be defended: every one of them is a
number chosen to separate "there is an echo and it arrives 420 ms late" from "a
flat correlation profile still has a maximum". The second drives the node and
the plugin facade through the lifecycle the canvas actually produces
(config→start→stop→config→start within seconds, from several threads).

What these tests deliberately cannot cover is the acoustics: whether a given
robot's speaker is linear enough for cancellation, and what its real delay is.
That needs a machine with a speaker and a microphone in the same room.
"""

from __future__ import annotations

import threading
import types

import numpy as np
import pytest

from vision_stubs import _FakeAudioChunk, _FakeExecutor, _FakeNode

import plugins.echo_cancellation as ec


BASE_T = 1_700_000_000.0
MIC_CHUNK_SAMPLES = 512        # 32 ms, the documented minimum
REF_CHUNK_SAMPLES = 1600       # 100 ms, what plugins/tts.py publishes


# ── helpers ──────────────────────────────────────────────────────────────────

def _chunk(pcm: bytes, timestamp: float, fmt: str = ec.AUDIO_FORMAT) -> _FakeAudioChunk:
    message = _FakeAudioChunk()
    seconds = int(timestamp)
    message.header.stamp = types.SimpleNamespace(
        sec=seconds, nanosec=round((timestamp - seconds) * 1_000_000_000))
    message.format = fmt
    message.data = pcm
    return message


def _bursts(seconds: float, seed: int = 0, amplitude: int = 6000) -> np.ndarray:
    """Speech-like audio: loud stretches separated by pauses.

    The envelope has to actually vary — correlating two constant-level noises
    measures nothing, and a test built on them would pass with the lag search
    broken.
    """
    rng = np.random.default_rng(seed)
    total = int(seconds * ec.SAMPLE_RATE)
    wave = rng.normal(0, amplitude, total)
    gate = np.zeros(total)
    position = 0
    while position < total:
        on = int(rng.uniform(0.15, 0.35) * ec.SAMPLE_RATE)
        off = int(rng.uniform(0.10, 0.25) * ec.SAMPLE_RATE)
        gate[position:position + on] = rng.uniform(0.4, 1.0)
        position += on + off
    return (wave * gate).astype(np.int16)


def _delayed(wave: np.ndarray, delay_s: float, gain: float = 0.5,
             noise: int = 40, seed: int = 7) -> np.ndarray:
    """`wave` as it would arrive at a microphone `delay_s` later, attenuated."""
    rng = np.random.default_rng(seed)
    shift = int(round(delay_s * ec.SAMPLE_RATE))
    echo = np.zeros_like(wave, dtype=np.float64)
    if shift < wave.size:
        echo[shift:] = wave[:wave.size - shift] * gain
    return (echo + rng.normal(0, noise, wave.size)).astype(np.int16)


def _feed(tracker: ec._DelayTracker, mic: np.ndarray, ref: np.ndarray,
          now: float) -> None:
    for start in range(0, ref.size - REF_CHUNK_SAMPLES + 1, REF_CHUNK_SAMPLES):
        timestamp = BASE_T + start / ec.SAMPLE_RATE
        tracker.add_ref(ref[start:start + REF_CHUNK_SAMPLES].tobytes(), timestamp, now)
    for start in range(0, mic.size - MIC_CHUNK_SAMPLES + 1, MIC_CHUNK_SAMPLES):
        timestamp = BASE_T + start / ec.SAMPLE_RATE
        tracker.add_mic(mic[start:start + MIC_CHUNK_SAMPLES].tobytes(), timestamp, now)


def _published(node) -> list[bytes]:
    return [bytes(bytearray(message.data)) for message in node._pub.raw_messages]


def _silent(pcm: bytes) -> bool:
    return not any(pcm)


# ── _DelayTracker ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("delay_s", [0.12, 0.42, 0.95])
def test_tracker_measures_a_known_delay(delay_s):
    ref = _bursts(6.0, seed=1)
    mic = _delayed(ref, delay_s)
    tracker = ec._DelayTracker()
    now = BASE_T + 6.0
    _feed(tracker, mic, ref, now)

    measured = tracker.estimate(now)
    assert measured is not None, "a 6 s window with a clean echo must produce an estimate"
    # One envelope hop is 10 ms, so two hops of error is the resolution floor.
    assert abs(measured - delay_s) <= 0.03


def test_tracker_says_nothing_when_the_robot_is_silent():
    ref = np.zeros(int(6.0 * ec.SAMPLE_RATE), dtype=np.int16)
    mic = _bursts(6.0, seed=2)        # a person talking, robot quiet
    tracker = ec._DelayTracker()
    now = BASE_T + 6.0
    _feed(tracker, mic, ref, now)

    assert tracker.estimate(now) is None
    assert tracker.measured is None


def test_tracker_refuses_an_uncorrelated_microphone():
    """No acoustic path at all must report nothing — not a confident 0 ms.

    This is the failure that matters: a flat correlation profile still has an
    argmax, and reporting its position is how a card ends up stating a delay
    nobody can reproduce.
    """
    ref = _bursts(6.0, seed=3)
    mic = _bursts(6.0, seed=99)       # independent audio, same statistics
    tracker = ec._DelayTracker()
    now = BASE_T + 6.0
    _feed(tracker, mic, ref, now)

    assert tracker.estimate(now) is None
    assert tracker.report()["echo_delay_rejected"] >= 1


def test_tracker_reports_a_median_and_a_spread():
    tracker = ec._DelayTracker()
    for value in (0.40, 0.42, 0.44, 0.60):
        tracker._accept(value)
    assert tracker.measured == pytest.approx(0.43)
    assert tracker.report()["measured_echo_delay_ms"] == 430
    assert tracker.report()["echo_delay_spread_ms"] is not None


def test_a_quiet_echo_is_still_measurable():
    """A speaker at low volume still has to be found: 20x attenuation, measured."""
    ref = _bursts(6.0, seed=5)
    mic = _delayed(ref, 0.42, gain=0.05)
    tracker = ec._DelayTracker()
    now = BASE_T + 6.0
    _feed(tracker, mic, ref, now)
    assert tracker.estimate(now) == pytest.approx(0.42, abs=0.03)


def test_a_delay_beyond_the_search_range_is_mostly_refused():
    """An echo later than the search range must not be reported as a short one.

    Measured with a 2 s echo against the 1.5 s range: most windows return
    nothing, and the ones that do not used to land on the outermost lag — which
    is now refused outright, because a peak pinned to the end of the range is
    the signature of a delay beyond it rather than one exactly at it. A few
    windows still find a spurious correlation between two burst patterns; that
    is what the median over many windows is for, and it is the reason this
    asserts a rate rather than a flat "never".
    """
    accepted = []
    for seed in range(10):
        ref = _bursts(6.0, seed=seed)
        mic = _delayed(ref, 2.0, seed=seed + 7)
        tracker = ec._DelayTracker()
        _feed(tracker, mic, ref, BASE_T + 6.0)
        estimate = tracker.estimate(BASE_T + 6.0)
        if estimate is not None:
            accepted.append(estimate)

    assert len(accepted) <= 3, f"too many spurious measurements: {accepted}"
    limit = ec.DELAY_SEARCH_MAX_S - ec.ENVELOPE_HOP_S
    assert all(value < limit for value in accepted), \
        "a peak at the end of the search range must be refused"


def test_double_talk_fools_a_single_window_but_not_the_median():
    """The measured weakness of this estimator, and the defence that covers it.

    A person talking over the robot at the robot's own level can push a single
    window's peak anywhere — 1.43 s and 0 ms both turned up in a sweep. That is
    why `measured` is a median over many windows rather than the latest value,
    and why `echo_delay_spread_ms` is reported: the median held at 420 ms with
    80% of windows contaminated, while the spread grew to ~1 s and says so.
    """
    truth = 0.42
    tracker = ec._DelayTracker()
    rng = np.random.default_rng(3)
    for window in range(20):
        ref = _bursts(6.0, seed=window)
        mic = _delayed(ref, truth, seed=window + 7)
        if rng.random() < 0.5:
            talker = _bursts(6.0, seed=window + 500, amplitude=8000)
            mic = np.clip(mic.astype(np.int32) + talker.astype(np.int32),
                          -32768, 32767).astype(np.int16)
        window_tracker = ec._DelayTracker()
        _feed(window_tracker, mic, ref, BASE_T + 6.0)
        estimate = window_tracker.estimate(BASE_T + 6.0)
        if estimate is not None:
            tracker._accept(estimate)

    assert len(tracker._estimates) >= 10
    assert tracker.measured == pytest.approx(truth, abs=0.08)
    assert tracker.spread is not None


def test_tracker_needs_both_streams_covering_the_window():
    """A reference that only started a moment ago cannot be correlated yet."""
    ref = _bursts(1.0, seed=4)
    mic = _delayed(_bursts(6.0, seed=4), 0.4)
    tracker = ec._DelayTracker()
    now = BASE_T + 6.0
    _feed(tracker, mic, ref, now)
    assert tracker.estimate(now) is None


def test_a_prebuffer_burst_is_laid_out_as_contiguous_audio():
    """plugins/tts.py publishes 500 ms of audio with near-identical stamps.

    `emit()` backdates the pacing schedule so the first PREBUF_FRAMES=5 frames
    are all already due and go out together, and `publish()` stamps each with
    now(). Positioning those by stamp piles five chunks on one another at the
    start of every utterance — exactly where the echo starts — and leaves the
    series non-monotonic, which makes `np.interp` return nonsense silently.
    """
    stream = ec._EnvelopeStream()
    wave = _bursts(0.5, seed=31)
    burst_stamp = BASE_T
    for index in range(5):                     # five 100 ms frames, one burst
        start = index * REF_CHUNK_SAMPLES
        stream.add(wave[start:start + REF_CHUNK_SAMPLES].tobytes(),
                   burst_stamp + index * 0.0004,     # ~0.4 ms apart, as measured
                   BASE_T + 0.5)
    times, _ = stream.arrays()

    assert len(times) == 50, "500 ms of audio must become 50 hops, not 10"
    assert np.all(np.diff(times) > 0), "an interpolated series must be monotonic"
    assert times[-1] == pytest.approx(BASE_T + 0.49, abs=1e-6)


def test_a_real_gap_resynchronises_instead_of_continuing():
    """After a silence the next utterance must land where its stamp says."""
    stream = ec._EnvelopeStream()
    chunk = _bursts(0.1, seed=32)[:REF_CHUNK_SAMPLES].tobytes()
    stream.add(chunk, BASE_T, BASE_T)
    stream.add(chunk, BASE_T + 5.0, BASE_T + 5.1)       # 5 s later
    times, _ = stream.arrays()
    assert times[10] == pytest.approx(BASE_T + 5.0, abs=1e-6)
    assert np.all(np.diff(times) > 0)


def test_a_backwards_clock_cannot_break_monotonicity():
    stream = ec._EnvelopeStream()
    chunk = _bursts(0.1, seed=33)[:REF_CHUNK_SAMPLES].tobytes()
    stream.add(chunk, BASE_T + 10.0, BASE_T + 10.0)
    stream.add(chunk, BASE_T, BASE_T + 10.1)            # clock jumped back 10 s
    times, _ = stream.arrays()
    assert np.all(np.diff(times) > 0)


def test_the_tracker_measures_through_prebuffer_bursts():
    """The same measurement, with the reference published the way TTS does it."""
    delay = 0.42
    ref = _bursts(6.0, seed=34)
    mic = _delayed(ref, delay)
    tracker = ec._DelayTracker()
    now = BASE_T + 6.0

    # Reference in bursts of five frames sharing a stamp, as tts.py emits them.
    frame = 0
    while (frame + 5) * REF_CHUNK_SAMPLES <= ref.size:
        stamp = BASE_T + frame * 0.1
        for index in range(5):
            start = (frame + index) * REF_CHUNK_SAMPLES
            tracker.add_ref(ref[start:start + REF_CHUNK_SAMPLES].tobytes(),
                            stamp + index * 0.0004, now)
        frame += 5
    for start in range(0, mic.size - MIC_CHUNK_SAMPLES + 1, MIC_CHUNK_SAMPLES):
        tracker.add_mic(mic[start:start + MIC_CHUNK_SAMPLES].tobytes(),
                        BASE_T + start / ec.SAMPLE_RATE, now)

    assert tracker.estimate(now) == pytest.approx(delay, abs=0.03)


# ── the gate ─────────────────────────────────────────────────────────────────

@pytest.fixture
def node():
    instance = ec._EchoCancellationNode('/mic/audio', '/tts/audio',
                                     node_suffix='t', mute_tail='300ms')
    instance.start()
    yield instance
    instance.retire()


def _say(node, seconds: float, start_t: float) -> None:
    """Publish `seconds` of robot speech on the reference topic."""
    wave = _bursts(seconds, seed=11)
    for start in range(0, wave.size - REF_CHUNK_SAMPLES + 1, REF_CHUNK_SAMPLES):
        node._ref_cb(_chunk(wave[start:start + REF_CHUNK_SAMPLES].tobytes(),
                            start_t + start / ec.SAMPLE_RATE))


def _hear(node, pcm: bytes, timestamp: float) -> None:
    node._mic_cb(_chunk(pcm, timestamp))


def test_the_microphone_is_silenced_while_the_robot_talks(node):
    """The warmup: before a delay is known there is nothing to align to.

    Not a mode — this ends as soon as the tracker has a measurement, which is a
    few seconds into the robot's first utterance. Passing the audio through
    instead would hand ASR the robot's own first sentence.
    """
    speech = _bursts(0.032, seed=12).tobytes()
    _hear(node, speech, BASE_T)                       # before: untouched
    _say(node, 1.0, BASE_T + 0.1)
    _hear(node, speech, BASE_T + 0.5)                 # during: silenced

    published = _published(node)
    assert len(published) == 2
    assert not _silent(published[0])
    assert _silent(published[1])
    assert len(published[1]) == len(speech), "a muted chunk keeps its length"


def test_the_mute_outlasts_the_robots_last_frame_by_the_tail(node):
    """Still true during the warmup: publishing the last frame is not the end
    of the sound, because the speaker buffers."""
    speech = _bursts(0.032, seed=13).tobytes()
    _say(node, 0.5, BASE_T)
    # The reference ends at BASE_T + 0.5; with a 300 ms tail the gate reopens at
    # 0.8 (plus the final chunk's own 100 ms).
    _hear(node, speech, BASE_T + 0.70)
    _hear(node, speech, BASE_T + 1.20)

    published = _published(node)
    assert _silent(published[0]), "still muted inside the tail"
    assert not _silent(published[1]), "mic reopens once the tail has passed"


def test_the_published_chunk_keeps_the_input_stamp(node):
    """Re-stamping would charge this card's latency to the speaker's timeline.

    ASR derives `audio_start_ts` from this header, so the stamp has to describe
    when the sound was captured, not when it was forwarded.
    """
    speech = _bursts(0.032, seed=15).tobytes()
    incoming = _chunk(speech, BASE_T + 3.25)
    node._mic_cb(incoming)
    published = node._pub.raw_messages[0]
    assert published.header.stamp.sec == incoming.header.stamp.sec
    assert published.header.stamp.nanosec == incoming.header.stamp.nanosec
    assert published.format == ec.AUDIO_FORMAT


def test_the_end_of_utterance_marker_is_not_reference_audio(node):
    """TTS's 8-byte EOF magic means 'finished publishing', not 'finished playing'.

    Treating it as audio would extend the mute by its own duration and, worse,
    feed four samples of square wave to the delay tracker.
    """
    node._ref_cb(_chunk(ec.AUDIO_EOF_MAGIC, BASE_T))
    diagnostics = node.diagnostics()
    assert diagnostics["ref_utterances_ended"] == 1
    assert diagnostics["ref_chunks"] == 0
    assert diagnostics["muting_now"] is False

    speech = _bursts(0.032, seed=16).tobytes()
    _hear(node, speech, BASE_T + 0.01)
    assert not _silent(_published(node)[0])


def test_a_silent_reference_chunk_does_not_arm_the_gate(node):
    """TTS pads and the speaker driver's own silence must not mute the mic."""
    node._ref_cb(_chunk(b'\x00\x00' * REF_CHUNK_SAMPLES, BASE_T))
    speech = _bursts(0.032, seed=17).tobytes()
    _hear(node, speech, BASE_T + 0.05)
    assert not _silent(_published(node)[0])
    assert node.diagnostics()["robot_voice_seen"] is True


def test_the_microphone_topic_is_refused_as_a_reference():
    """Correlating the mic against itself would mute permanently at delay 0."""
    instance = ec._EchoCancellationNode('/mic/audio', '/mic/audio', node_suffix='self')
    assert instance._ref_topic == ''
    instance.start()
    assert [sub.topic for sub in instance.subscriptions] == ['/mic/audio']
    instance.retire()


def test_an_odd_length_chunk_is_truncated_not_dropped(node):
    _hear(node, b'\x01\x02\x03', BASE_T)
    published = _published(node)
    assert len(published) == 1 and len(published[0]) == 2
    assert node.diagnostics()["misaligned_chunks"] == 1


def test_a_gap_in_the_microphone_stream_is_counted(node):
    speech = _bursts(0.032, seed=18).tobytes()
    _hear(node, speech, BASE_T)
    _hear(node, speech, BASE_T + 0.032)      # on time
    _hear(node, speech, BASE_T + 0.500)      # late — bus was busy
    assert node.diagnostics()["mic_gaps"] == 1


def test_a_foreign_format_is_counted_but_still_processed(node):
    speech = _bursts(0.032, seed=19).tobytes()
    node._mic_cb(_chunk(speech, BASE_T, fmt="audio/pcm-48k"))
    assert node.diagnostics()["format_violations"] == 1
    assert _published(node) == [speech]


def test_auto_tail_follows_the_measured_delay(node):
    node.apply_config(mute_tail=ec.MUTE_TAIL_AUTO)
    assert node._tail_s() == pytest.approx(ec.AUTO_TAIL_FALLBACK_S)
    with node._tracker_lock:
        node._tracker._accept(1.10)
    assert node._tail_s() == pytest.approx(1.10 + ec.AUTO_TAIL_MARGIN_S)


def test_the_output_topic_is_the_input_plus_a_suffix(node):
    topic_in, topic_out = node.topics()
    assert topic_in[0]["topic"] == '/mic/audio'
    assert topic_out[0]["topic"] == '/mic/audio' + ec.OUTPUT_SUFFIX
    assert topic_out[0]["format"] == ec.AUDIO_FORMAT


def test_starting_without_a_microphone_is_idle_and_never_an_error():
    """`error` here stops every other card on the canvas.

    agent-core's start-project rolls the whole project back when any card
    returns `state: error` (`api/config.py`, "有 card 失败 → 全部回滚"). Measured
    on Orin 5: a second echo_cancellation card that nobody had wired up made
    OCR, face recognition, TTS, ASR and the *working* copy of this card all
    start correctly and then stop one second later. A card waiting to be wired
    is an ordinary state — tts, ocr and barcode all support a topic-less card.
    """
    instance = ec._EchoCancellationNode('', '/tts/audio', node_suffix='bare')
    result = instance.start()
    assert result["state"] == "idle"
    assert "microphone" in result["message"]


# ── reference buffer ─────────────────────────────────────────────────────────

def test_the_reference_buffer_answers_by_time():
    buffer = ec._RefBuffer(2.0)
    wave = _bursts(0.3, seed=40)
    for index in range(3):
        start = index * REF_CHUNK_SAMPLES
        buffer.add(wave[start:start + REF_CHUNK_SAMPLES].tobytes(),
                   BASE_T + index * 0.1)
    fetched = buffer.fetch(BASE_T + 0.1, REF_CHUNK_SAMPLES)
    expected = wave[REF_CHUNK_SAMPLES:2 * REF_CHUNK_SAMPLES]
    assert np.array_equal(fetched.astype(np.int16), expected)


def test_a_hole_in_the_reference_is_silence_not_a_seam():
    """Nothing published means the speaker played nothing.

    Closing the gap by sliding the next utterance earlier would misalign every
    block after it — and the misalignment is indistinguishable from the delay
    having changed, which is the one thing the card is trying to track.
    """
    buffer = ec._RefBuffer(10.0)
    chunk = _bursts(0.1, seed=41)[:REF_CHUNK_SAMPLES].tobytes()
    buffer.add(chunk, BASE_T)
    buffer.add(chunk, BASE_T + 3.0)
    assert np.all(buffer.fetch(BASE_T + 1.5, REF_CHUNK_SAMPLES) == 0)
    assert np.any(buffer.fetch(BASE_T + 3.0, REF_CHUNK_SAMPLES) != 0)


def test_the_reference_buffer_keeps_a_prebuffer_burst_contiguous():
    buffer = ec._RefBuffer(2.0)
    wave = _bursts(0.5, seed=42)
    for index in range(5):
        start = index * REF_CHUNK_SAMPLES
        buffer.add(wave[start:start + REF_CHUNK_SAMPLES].tobytes(),
                   BASE_T + index * 0.0004)        # one burst, as tts.py emits
    assert buffer.covers(BASE_T, wave.size)
    assert np.array_equal(buffer.fetch(BASE_T, wave.size).astype(np.int16), wave)


def test_the_reference_buffer_drops_the_oldest_audio_and_keeps_its_axis():
    buffer = ec._RefBuffer(0.3)
    wave = _bursts(1.0, seed=43)
    for index in range(10):
        start = index * REF_CHUNK_SAMPLES
        buffer.add(wave[start:start + REF_CHUNK_SAMPLES].tobytes(),
                   BASE_T + index * 0.1)
    assert np.all(buffer.fetch(BASE_T, REF_CHUNK_SAMPLES) == 0)   # long gone
    last = wave[9 * REF_CHUNK_SAMPLES:10 * REF_CHUNK_SAMPLES]
    assert np.array_equal(buffer.fetch(BASE_T + 0.9, REF_CHUNK_SAMPLES).astype(np.int16),
                          last)


# ── the canceller ────────────────────────────────────────────────────────────

def _room(ms: float = 60, seed: int = 0, decay_ms: float = 40.0) -> np.ndarray:
    """A synthetic echo path: decaying random taps, unit energy."""
    rng = np.random.default_rng(seed)
    taps = int(ms * ec.SAMPLE_RATE / 1000)
    impulse = rng.normal(0, 1, taps) * np.exp(
        -np.arange(taps) / (decay_ms * ec.SAMPLE_RATE / 1000))
    return impulse / np.sqrt(np.square(impulse).sum())


def _cancel_session(strength="gentle", seconds=18.0, seed=1, gain=0.5, room_ms=60,
                    noise=30, talk=None, talker=4000):
    """Run the canceller over a synthetic session. Returns (mic, out, person, aec)."""
    reference = _bursts(seconds, seed=seed).astype(np.float32)
    echo = np.convolve(reference, _room(room_ms, seed) * gain)[:reference.size]
    person = np.zeros_like(reference)
    if talk is not None:
        voice = _bursts(seconds, seed=seed + 77, amplitude=talker).astype(np.float32)
        lo, hi = int(talk[0] * ec.SAMPLE_RATE), int(talk[1] * ec.SAMPLE_RATE)
        person[lo:hi] = voice[lo:hi]
    rng = np.random.default_rng(seed + 9)
    mic = (echo.astype(np.float32) + person
           + rng.normal(0, noise, reference.size).astype(np.float32))

    canceller = ec._EchoCanceller(strength=strength)
    out = np.zeros_like(mic)
    for start in range(0, reference.size - ec.AEC_BLOCK + 1, ec.AEC_BLOCK):
        stop = start + ec.AEC_BLOCK
        out[start:stop] = canceller.process(mic[start:stop], reference[start:stop])
    return mic, out, person, canceller


def _db(signal, residual, window=None):
    if window is not None:
        signal, residual = signal[window], residual[window]
    return 10 * np.log10((np.mean(signal ** 2) + 1e-9)
                         / (np.mean(residual ** 2) + 1e-9))


def test_the_canceller_converges_on_a_synthetic_room():
    mic, out, _, canceller = _cancel_session()
    settled = slice(int(12 * ec.SAMPLE_RATE), mic.size)
    assert _db(mic, out, settled) > 15.0
    # Reported over the last few seconds of robot speech, not the whole run,
    # so it tracks the settled figure rather than being dragged down by
    # convergence. Measured 9.9 dB here against 18 dB for the settled window —
    # the gap is the pauses in the robot's own speech, which are in the average
    # and have no echo to remove.
    assert canceller.erle_db is not None and canceller.erle_db > 8.0
    assert canceller._resets == 0, "a converging filter must not be judged useless"


@pytest.mark.parametrize("strength,min_erle,min_person", [
    # The measured trade, asserted a couple of dB inside each figure so the
    # test fails on a regression rather than on noise. Measured: 18.1/18.1,
    # 22.9/12.0, 27.9/6.5.
    ("gentle", 15.0, 15.0),
    ("normal", 20.0, 9.0),
    ("aggressive", 25.0, 4.0),
])
def test_removal_strength_is_the_trade_it_claims(strength, min_erle, min_person):
    mic, out, person, _ = _cancel_session(strength=strength, talk=(8.0, 12.0))
    echo_only = slice(int(5 * ec.SAMPLE_RATE), int(8 * ec.SAMPLE_RATE))
    double_talk = slice(int(8 * ec.SAMPLE_RATE), int(12 * ec.SAMPLE_RATE))
    assert _db(mic, out, echo_only) > min_erle
    assert _db(person, out - person, double_talk) > min_person


def test_double_talk_does_not_destroy_a_converged_filter():
    """A bad second must not undo a good minute.

    Measured: 19.8 dB before, 1.7 dB during (the person is in the microphone and
    is supposed to survive, so low ERLE there is correct), 22.1 dB after — the
    filter kept improving rather than being dragged apart.
    """
    mic, out, _, _ = _cancel_session(talk=(8.0, 12.0))
    before = _db(mic, out, slice(int(5 * ec.SAMPLE_RATE), int(8 * ec.SAMPLE_RATE)))
    after = _db(mic, out, slice(int(13 * ec.SAMPLE_RATE), int(18 * ec.SAMPLE_RATE)))
    assert before > 15.0
    assert after >= before - 1.0


def test_a_double_talk_window_does_not_trip_the_divergence_guard():
    """The rig found this one, and it cost a converged filter three times.

    A filter converged to 20 dB that is then talked over looks identical, in
    powers alone, to a filter that has learned nothing: the residual is
    dominated by the person either way. The guard's threshold was +1 dB, so on
    Orin 5 it zeroed the filter during a 4 s double-talk stretch and reported
    "this microphone barely hears the speaker". Powers cannot separate those
    cases; the guard now only catches the unambiguous half.
    """
    mic, out, _, canceller = _cancel_session(talk=(8.0, 12.0), talker=8000)
    assert canceller._resets == 0
    after = _db(mic, out, slice(int(13 * ec.SAMPLE_RATE), int(18 * ec.SAMPLE_RATE)))
    assert after > 15.0, "the filter must still be converged afterwards"


def test_a_diverging_filter_is_zeroed():
    """The half that powers *can* judge: an output worse than its input."""
    canceller = ec._EchoCanceller(strength="gentle")
    rng = np.random.default_rng(5)
    reference = rng.normal(0, 6000, ec.AEC_BLOCK * 80).astype(np.float32)
    for index in range(80):
        block = slice(index * ec.AEC_BLOCK, (index + 1) * ec.AEC_BLOCK)
        # Force the state a diverged filter would be in, then feed it audio.
        canceller._weights += 50.0
        canceller.process(rng.normal(0, 300, ec.AEC_BLOCK).astype(np.float32),
                          reference[block])
    assert canceller._resets >= 1
    assert canceller._benefit_db is not None and canceller._benefit_db < 0


def test_a_measurement_goes_stale_after_enough_unexplained_speech():
    """Counted in robot speech, not wall clock.

    A delay measured an hour ago is still the right answer if the robot has been
    silent since. Fifteen seconds of it *talking* with no measurable echo is a
    different thing: the microphone has stopped hearing the speaker.
    """
    tracker = ec._DelayTracker()
    tracker._accept(0.45)
    assert tracker.fresh() is True
    tracker.note_far_active(ec.DELAY_STALE_AFTER_FAR_ACTIVE_S + 1.0)
    assert tracker.fresh() is False
    assert tracker.report()["echo_delay_fresh"] is False


def test_the_reported_erle_ignores_double_talk_blocks():
    """Otherwise a card reports 4 dB while removing 17 of them.

    Found on Orin 5: an utterance whose last 0.4 s overlapped the next person's
    turn was enough to drag the reported figure from 14 dB down to 4.3, because
    during double-talk the residual is dominated by the person — so ERLE there
    is near 0 dB and describes nothing about echo removal. The person's voice is
    also *louder* than the echo, so those blocks dominate a ratio of averages.
    """
    mic, out, _, canceller = _cancel_session(seconds=14.0, talk=(11.0, 14.0),
                                             talker=12000)
    echo_only = slice(int(8 * ec.SAMPLE_RATE), int(10.5 * ec.SAMPLE_RATE))
    settled = _db(mic, out, echo_only)
    assert settled > 15.0
    assert canceller.erle_db is not None
    assert canceller.erle_db > settled - 8.0, (
        f"reported {canceller.erle_db} dB against {settled:.1f} dB measured on "
        f"the echo-only span — double-talk blocks are leaking into the average")


def _changing_room(reference, seed, gain, change_at, seconds):
    """An echo path that changes halfway: the robot turned its head."""
    first = np.convolve(reference, _room(60, seed) * gain)[:reference.size]
    second = np.convolve(reference, _room(60, seed + 31) * gain)[:reference.size]
    at = int(change_at * ec.SAMPLE_RATE)
    return np.concatenate([first[:at], second[at:]]).astype(np.float32)


def _session(mic, reference, strength="normal"):
    canceller = ec._EchoCanceller(strength=strength)
    out = np.zeros_like(mic)
    for start in range(0, reference.size - ec.AEC_BLOCK + 1, ec.AEC_BLOCK):
        stop = start + ec.AEC_BLOCK
        out[start:stop] = canceller.process(mic[start:stop], reference[start:stop])
    return out, canceller


def test_a_changed_echo_path_is_relearned():
    """The robot turns its head. Without the shadow filter this never recovers.

    The skip guard refuses an update whenever the error exceeds the microphone,
    which is what double-talk looks like — and what a changed path looks like
    too. Measured before the shadow filter existed: -2.9 dB and still -2.9 dB
    six seconds later, with 355 consecutive updates refused. Worse than passing
    the microphone through, because a stale filter adds a wrong estimate.
    """
    seconds, change_at = 20.0, 9.0
    reference = _bursts(seconds, seed=1).astype(np.float32)
    echo = _changing_room(reference, 1, 0.5, change_at, seconds)
    mic = (echo + np.random.default_rng(10).normal(0, 30, echo.size)).astype(np.float32)
    out, canceller = _session(mic, reference)

    def erle(lo, hi):
        return _db(mic, out, slice(int(lo * ec.SAMPLE_RATE), int(hi * ec.SAMPLE_RATE)))

    assert erle(6, 9) > 20.0, "converged before the change"
    assert erle(15, 20) > 15.0, "and again after it"
    assert canceller._promotions >= 1
    assert canceller.report()["aec_shadow_promotions"] >= 1


def test_double_talk_does_not_promote_the_shadow():
    """The shadow adapts to a person and gets worse, so it is not promoted.

    This is the whole reason the comparison is on outcome rather than on
    powers: double-talk and a changed path are indistinguishable from the
    error's size alone, which is the third time that is true in this file.
    """
    mic, out, person, canceller = _cancel_session(talk=(8.0, 12.0), talker=8000)
    assert canceller._promotions == 0
    after = _db(mic, out, slice(int(13 * ec.SAMPLE_RATE), int(18 * ec.SAMPLE_RATE)))
    before = _db(mic, out, slice(int(5 * ec.SAMPLE_RATE), int(8 * ec.SAMPLE_RATE)))
    assert after >= before - 1.0
    assert _db(person, out - person,
               slice(int(8 * ec.SAMPLE_RATE), int(12 * ec.SAMPLE_RATE))) > 12.0


def test_the_shadow_filter_costs_nothing_when_nothing_changes():
    """A steady path must not be disturbed by a second filter looking over it."""
    mic, out, _, canceller = _cancel_session()
    assert canceller._promotions == 0
    assert _db(mic, out, slice(int(12 * ec.SAMPLE_RATE), mic.size)) > 20.0


def test_a_lost_shadow_filter_is_restarted():
    """Otherwise it can never win again, and the foreground is stuck for good."""
    canceller = ec._EchoCanceller(strength="gentle")
    canceller._shadow += 50.0                       # as if it had diverged
    rng = np.random.default_rng(4)
    reference = rng.normal(0, 6000, ec.AEC_BLOCK * 200).astype(np.float32)
    for index in range(200):
        block = slice(index * ec.AEC_BLOCK, (index + 1) * ec.AEC_BLOCK)
        canceller.process(rng.normal(0, 300, ec.AEC_BLOCK).astype(np.float32),
                          reference[block])
    assert np.abs(canceller._shadow).max() < 50.0


def test_the_step_size_stays_under_the_measured_stability_edge():
    """The constant encodes a measurement, so a plausible edit must fail here.

    Synthetic 60 ms room, 12 s, three seeds: 0.2 reaches 20-23 dB and 0.3
    reaches 22-25, but 0.4 collapses to about 0 dB and 0.5 diverges to -4. A
    diverged filter is worse than no filter at all, so this sits a factor of two
    below the edge rather than taking the last 2 dB.
    """
    assert ec.AEC_MU <= 0.3


def test_the_suppressor_ramps_instead_of_stepping():
    """A gain that changes every 16 ms would be heard as a click."""
    canceller = ec._EchoCanceller(strength="aggressive")
    error = np.full(ec.AEC_BLOCK, 1000.0, dtype=np.float32)
    estimate = np.full(ec.AEC_BLOCK, 4000.0, dtype=np.float32)
    out = canceller._suppress(error, estimate, far_active=True)
    assert out[0] == pytest.approx(error[0], rel=0.01), "starts at the old gain"
    assert out[-1] < error[-1] * 0.5, "ends at the new one"
    assert np.all(np.diff(out) <= 0)


# ── cancellation through the node ────────────────────────────────────────────

def _cancelling_node(delay=0.45, strength="gentle", measured=True):
    node = ec._EchoCancellationNode('/mic/audio', '/tts/audio', node_suffix='aec',
                                    removal_strength=strength)
    node.start()
    if measured:
        # The tracker needs seconds of audio to measure; the cancellation path
        # is what is under test here, so the measurement is primed directly.
        with node._tracker_lock:
            node._tracker._accept(delay)
    return node


def _play_and_listen(node, delay=0.45, seconds=10.0, seed=2, gain=0.5):
    """Publish a robot utterance and the microphone that hears it `delay` later.

    Interleaved by stamp, the way the executor delivers them. Publishing the
    whole reference first instead does not work and should not: the reference
    ring only holds a couple of seconds, because TTS paces its output in real
    time and audio that far ahead of the microphone cannot be an echo of
    anything the microphone has not heard yet.
    """
    reference = _bursts(seconds, seed=seed)
    echo = np.convolve(reference.astype(np.float32),
                       _room(60, seed) * gain)[:reference.size]
    shift = int(delay * ec.SAMPLE_RATE)
    mic = np.zeros(reference.size, dtype=np.float32)
    mic[shift:] = echo[:reference.size - shift]
    mic += np.random.default_rng(seed + 3).normal(0, 30, mic.size).astype(np.float32)
    mic16 = np.clip(mic, -32768, 32767).astype(np.int16)

    events = []
    for start in range(0, reference.size - REF_CHUNK_SAMPLES + 1, REF_CHUNK_SAMPLES):
        events.append((start / ec.SAMPLE_RATE, 'ref', start, REF_CHUNK_SAMPLES))
    for start in range(0, mic16.size - MIC_CHUNK_SAMPLES + 1, MIC_CHUNK_SAMPLES):
        events.append((start / ec.SAMPLE_RATE, 'mic', start, MIC_CHUNK_SAMPLES))
    events.sort(key=lambda item: (item[0], item[1]))     # ref before mic at a tie

    for offset, kind, start, count in events:
        source = reference if kind == 'ref' else mic16
        message = _chunk(source[start:start + count].tobytes(), BASE_T + offset)
        (node._ref_cb if kind == 'ref' else node._mic_cb)(message)
    return mic16


def test_the_node_cancels_a_delayed_echo_end_to_end():
    """Through the real callbacks: chunked, reframed, aligned by measurement."""
    node = _cancelling_node()
    mic = _play_and_listen(node)
    out = np.frombuffer(b''.join(_published(node)), dtype='<i2').astype(np.float32)

    # Compare the settled half of the output against the same span of the mic.
    offset = mic.size - out.size
    settled = slice(int(out.size * 0.5), out.size)
    reference_span = mic[offset:].astype(np.float32)
    assert _db(reference_span, out, settled) > 12.0
    diagnostics = node.diagnostics()
    assert diagnostics["echo_removed_db"] > 8.0
    assert diagnostics["filter_benefit_db"] > 0.0
    assert diagnostics["filter_tail_ms"] == 128
    node.retire()


def test_the_node_mutes_until_it_has_measured_the_delay():
    """Subtracting an unaligned reference would add noise, not remove echo."""
    node = _cancelling_node(measured=False)
    speech = _bursts(0.032, seed=50).tobytes()
    _say(node, 1.0, BASE_T)
    _hear(node, speech, BASE_T + 0.5)

    published = _published(node)
    assert _silent(published[0]), "muted, not cancelled, while the delay is unknown"
    diagnostics = node.diagnostics()
    assert diagnostics["waiting_for_delay"] is True
    assert diagnostics["cancel_chunks_without_a_delay"] == 1
    node.retire()


def test_cancelled_chunks_carry_their_own_stamp_and_keep_the_cadence():
    """Reframing into 16 ms blocks moves the audio; the stamp has to follow it.

    ASR reads this header as the capture time, so a chunk of reframed audio
    published under the stamp of the chunk that triggered it would be a small
    lie told on every chunk — and one that accumulates into ASR's
    `audio_start_ts`.
    """
    node = _cancelling_node()
    speech = _bursts(0.032, seed=51).tobytes()
    for index in range(20):
        _hear(node, speech, BASE_T + index * 0.032)

    stamps = [message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
              for message in node._pub.raw_messages]
    sizes = {len(bytes(bytearray(m.data))) for m in node._pub.raw_messages}
    assert sizes == {len(speech)}, "every published chunk keeps the input's size"
    assert len(stamps) >= 19, "at most one chunk of startup latency"
    gaps = np.diff(stamps)
    assert np.allclose(gaps, 0.032, atol=1e-6), "no gaps and no overlaps"
    assert stamps[0] >= BASE_T
    assert stamps[0] <= BASE_T + ec.AEC_BLOCK / ec.SAMPLE_RATE + 1e-9
    node.retire()


def test_a_lost_coupling_passes_audio_through_rather_than_muting_forever():
    """No measurable echo means there is nothing to remove.

    Muting here would be a permanently half-duplex microphone, which is the one
    behaviour this card must not have — it is why muting is not offered as a
    mode at all. The warmup mute is bounded and ends; this would not.

    It is also where the no-coupling protection lives: with the reference
    playing and no echo in the mic the tracker accepts nothing, so the filter is
    never given the chance to adapt to noise and inject it. Measured before that
    was true: 18 s of adapting against no coupling left near-end speech 4.5 dB
    *below* passthrough.
    """
    node = _cancelling_node()
    with node._tracker_lock:
        node._tracker.note_far_active(ec.DELAY_STALE_AFTER_FAR_ACTIVE_S + 1.0)
    speech = _bursts(0.032, seed=53).tobytes()
    _say(node, 1.0, BASE_T)
    _hear(node, speech, BASE_T + 0.5)

    assert _published(node) == [speech], "passed through, not silenced"
    diagnostics = node.diagnostics()
    assert diagnostics["coupling_lost"] is True
    assert diagnostics["waiting_for_delay"] is False
    assert diagnostics["cancel_chunks_after_coupling_lost"] == 1
    node.retire()


def test_changing_the_strength_keeps_the_converged_filter():
    node = _cancelling_node(strength="gentle")
    _play_and_listen(node, seconds=6.0)
    weights = node._aec._weights.copy()
    node.apply_config(removal_strength="aggressive")
    assert node._aec._weights is not weights
    assert np.array_equal(node._aec._weights, weights), \
        "the suppressor sits after the filter and shares no state with it"
    assert node._aec._beta == ec.SUPPRESSION_STRENGTHS["aggressive"]
    node.retire()


def test_a_gap_in_the_microphone_resynchronises_the_canceller():
    node = _cancelling_node()
    speech = _bursts(0.032, seed=52).tobytes()
    _hear(node, speech, BASE_T)
    _hear(node, speech, BASE_T + 0.032)
    _hear(node, speech, BASE_T + 5.0)          # a long gap
    assert node.diagnostics()["cancel_resyncs"] >= 0   # counted, never raises
    stamps = [m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
              for m in node._pub.raw_messages]
    assert all(b > a for a, b in zip(stamps, stamps[1:]))
    node.retire()


# ── plugin facade ────────────────────────────────────────────────────────────

class _FakeTTS:
    """Enough of TTSPlugin for the reference lookup: the `info` action."""

    def __init__(self, topics=('/tts/audio',)):
        self.topics = list(topics)
        self.calls = 0

    def dispatch(self, name, args):
        self.calls += 1
        return {"topic_out": [{"topic": topic, "format": ec.AUDIO_FORMAT,
                               "desc": ""} for topic in self.topics]}


class _BrokenTTS:
    def dispatch(self, name, args):
        raise RuntimeError("engine is still loading")


@pytest.fixture
def plugin():
    executor = _FakeExecutor()
    instance = ec.EchoCancellationPlugin({}, executor, tts=_FakeTTS())
    yield instance
    instance.dispatch("echo_cancellation", {"action": "stop"})


def test_auto_resolves_the_reference_from_the_tts_plugin(plugin):
    plugin.dispatch("echo_cancellation", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/mic/audio"})
    node = plugin._nodes["c1"]
    assert node._ref_topic == '/tts/audio'
    assert sorted(sub.topic for sub in node.subscriptions) == ['/mic/audio', '/tts/audio']


def test_auto_with_no_tts_in_this_process_says_so_on_the_card():
    plugin = ec.EchoCancellationPlugin({}, _FakeExecutor(), tts=None)
    plugin.dispatch("echo_cancellation", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/mic/audio"})
    info = plugin.dispatch("echo_cancellation", {"action": "info", "instance_id": "c1"})
    assert info["diagnostics"]["robot_voice_topic"] is None
    assert "does not pass through Agent Core" in info["desc"]
    # And it is a passthrough rather than a gate that mutes at random.
    node = plugin._nodes["c1"]
    _hear(node, _bursts(0.032, seed=20).tobytes(), BASE_T)
    assert not _silent(_published(node)[0])


def test_a_broken_tts_does_not_stop_this_card_answering(plugin):
    plugin._tts = _BrokenTTS()
    tools = plugin.get_tools()
    options = tools[0]["configSchema"]["properties"]["robot_voice_topic"]["oneOf"]
    assert [option["const"] for option in options] == [ec.REF_AUTO]
    assert plugin.dispatch("echo_cancellation", {"action": "info"})["state"] == "idle"


def test_the_dropdown_lists_the_topics_tts_is_publishing_now(plugin):
    plugin._tts = _FakeTTS(('/g1/tts/audio', '/ext/tts/audio'))
    options = plugin.get_tools()[0]["configSchema"]["properties"][
        "robot_voice_topic"]["oneOf"]
    assert [option["const"] for option in options] == [
        ec.REF_AUTO, '/g1/tts/audio', '/ext/tts/audio']


def test_a_configured_topic_stays_selectable_when_its_publisher_is_down():
    plugin = ec.EchoCancellationPlugin({"robot_voice_topic": "/other/box/tts/audio"},
                                    _FakeExecutor(), tts=_FakeTTS(()))
    options = plugin.get_tools()[0]["configSchema"]["properties"][
        "robot_voice_topic"]["oneOf"]
    assert [option["const"] for option in options] == [
        ec.REF_AUTO, "/other/box/tts/audio"]


def test_the_module_level_tools_are_not_mutated_by_a_lookup(plugin):
    before = len(ec.TOOLS[0]["configSchema"]["properties"]["robot_voice_topic"]["oneOf"])
    plugin.get_tools()
    plugin.get_tools()
    after = len(ec.TOOLS[0]["configSchema"]["properties"]["robot_voice_topic"]["oneOf"])
    assert before == after == 1


def test_changing_the_reference_topic_rebuilds_the_node(plugin):
    plugin.dispatch("echo_cancellation", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/mic/audio"})
    first = plugin._nodes["c1"]
    plugin.dispatch("echo_cancellation", {"action": "config", "instance_id": "c1",
                                       "robot_voice_topic": "/other/tts/audio"})
    second = plugin._nodes["c1"]
    # A new node, because a subscription cannot be recreated on a spinning one:
    # it wedges, with the graph and the executor both looking healthy.
    assert second is not first
    assert first.destroyed is True
    assert second._ref_topic == "/other/tts/audio"
    assert second.state == "running"


def test_start_is_idempotent(plugin):
    args = {"action": "start", "instance_id": "c1", "input_topic": "/mic/audio"}
    plugin.dispatch("echo_cancellation", args)
    node = plugin._nodes["c1"]
    plugin.dispatch("echo_cancellation", args)
    assert plugin._nodes["c1"] is node
    assert [sub.topic for sub in node.subscriptions].count('/mic/audio') == 1


def test_rewiring_the_microphone_replaces_the_node(plugin):
    plugin.dispatch("echo_cancellation", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/mic/audio"})
    first = plugin._nodes["c1"]
    plugin.dispatch("echo_cancellation", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/ext_mic/1/audio"})
    second = plugin._nodes["c1"]
    assert second is not first and first.destroyed is True
    assert second._output_topic == '/ext_mic/1/audio' + ec.OUTPUT_SUFFIX


def test_stop_disposes_the_node_and_leaves_the_executor_clean(plugin):
    plugin.dispatch("echo_cancellation", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/mic/audio"})
    node = plugin._nodes["c1"]
    plugin.dispatch("echo_cancellation", {"action": "stop", "instance_id": "c1"})
    assert plugin._nodes == {}
    assert node.destroyed is True
    assert plugin._executor.nodes == []


def test_a_concurrent_stop_cannot_orphan_a_live_node(plugin):
    """The canvas really does this: config→start→stop→start within a second.

    An orphaned node is the expensive failure here — nothing can reach it to
    stop it, so its subscription and publisher outlive every card on the page
    (perception/README.md § Plugin Concurrency).
    """
    args = {"action": "start", "instance_id": "c1", "input_topic": "/mic/audio"}
    errors = []

    def hammer(action):
        for _ in range(40):
            try:
                plugin.dispatch("echo_cancellation",
                                args if action == "start"
                                else {"action": "stop", "instance_id": "c1"})
            except Exception as error:  # noqa: BLE001
                errors.append(error)

    threads = [threading.Thread(target=hammer, args=(name,))
               for name in ("start", "stop", "start", "stop")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors

    plugin.dispatch("echo_cancellation", {"action": "stop"})
    assert plugin._nodes == {}
    assert plugin._executor.nodes == []
    live = [instance for instance in _FakeNode.instances
            if isinstance(instance, ec._EchoCancellationNode) and not instance.destroyed]
    assert live == [], "every node built must end up destroyed"


def test_an_unwired_card_says_what_it_is_waiting_for(plugin):
    plugin.dispatch("echo_cancellation", {"action": "start", "instance_id": "c1"})
    info = plugin.dispatch("echo_cancellation", {"action": "info", "instance_id": "c1"})
    assert info["state"] == "idle"
    assert "Not connected to a microphone" in info["desc"]


def test_info_says_nothing_has_been_heard_yet(plugin):
    plugin.dispatch("echo_cancellation", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/mic/audio"})
    info = plugin.dispatch("echo_cancellation", {"action": "info", "instance_id": "c1"})
    assert info["diagnostics"]["robot_voice_seen"] is False
    assert "nothing heard yet" in info["desc"]


def test_info_flags_a_tail_shorter_than_the_measured_delay(plugin):
    plugin.dispatch("echo_cancellation", {"action": "config", "instance_id": "c1",
                                       "mute_tail": "300ms"})
    plugin.dispatch("echo_cancellation", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/mic/audio"})
    node = plugin._nodes["c1"]
    _say(node, 0.3, BASE_T)
    with node._tracker_lock:
        node._tracker._accept(1.05)

    info = plugin.dispatch("echo_cancellation", {"action": "info", "instance_id": "c1"})
    # Still reported, because `mute_tail` can still be pinned from config.yaml
    # for an experiment. It no longer has a message on the card's face: with no
    # knob for it there, nobody could act on one.
    assert info["diagnostics"]["mute_tail_shorter_than_measured_delay"] is True


def test_info_without_an_instance_reports_the_card(plugin):
    plugin.dispatch("echo_cancellation", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/mic/audio"})
    info = plugin.dispatch("echo_cancellation", {"action": "info"})
    assert info["state"] == "running"
    assert info["topic_out"][0]["topic"] == '/mic/audio' + ec.OUTPUT_SUFFIX
    assert info["instances"] == {"c1": {"state": "running"}}


def test_the_summary_reports_the_card_that_is_working(plugin):
    """A canvas routinely holds a second card nobody has wired up yet.

    On Orin 5 the summary came from that one: it truthfully said "no
    microphone, no reference heard" while the wired card beside it was running,
    which reads as the working card being broken — and sent a debugging session
    after a reference topic that was fine.
    """
    plugin.dispatch("echo_cancellation", {"action": "start", "instance_id": "unwired"})
    plugin.dispatch("echo_cancellation", {"action": "start", "instance_id": "wired",
                                          "input_topic": "/mic/audio"})
    info = plugin.dispatch("echo_cancellation", {"action": "info"})
    assert info["diagnostics"]["instance_id"] == "wired"
    assert info["diagnostics"]["robot_voice_topic"] == "/tts/audio"
    assert "Not connected to a microphone" not in info["desc"]


def test_an_unknown_action_is_declined_not_answered(plugin):
    assert plugin.dispatch("echo_cancellation", {"action": "speak"}) is None


def test_the_callback_cost_is_reported(plugin):
    """Reported as wall time in the callback, which is what the name says.

    It is not CPU time: the other plugins' threads contend for the GIL in this
    process, and on Orin 5 the live figure was 29.7 ms per audio second against
    3.6 ms for the same callback profiled on its own. Naming it `cpu_*` invited
    exactly the wrong conclusion from the difference.
    """
    plugin.dispatch("echo_cancellation", {"action": "start", "instance_id": "c1",
                                       "input_topic": "/mic/audio"})
    node = plugin._nodes["c1"]
    speech = _bursts(0.032, seed=21).tobytes()
    for index in range(30):
        _hear(node, speech, BASE_T + index * 0.032)
    cost = node.diagnostics()["callback_wall_ms_per_audio_s"]
    assert cost is not None and cost >= 0.0
    assert "cpu_ms_per_audio_s" not in node.diagnostics()
