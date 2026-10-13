"""Speech-over-background PCM mixing, independent of ROS, SDKs and the catalogue.

Everything here used to live inside `MusicPlayer`, which made the music card
secretly the mixer: it took TTS in on its own `topic_in`, ducked the song under
it, and published the *combined* stream. So "the music card" owned speech
routing, and the wiring read `TTS → phanthy_music → Speaker`.

Splitting it buys two things. The music card becomes a plain actuator that emits
music and nothing else, and the mixing becomes available to any future audio
source (a chime, an alarm, a second robot's voice) rather than being reachable
only by going through the music player.

Both cards share `paced_loop` so the drift correction exists once: a frame that
is late must not be burst into the speaker, it must be dropped flat.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

# The 8-byte end-of-stream sentinel. A normal frame is always FRAME_BYTES, so a
# short frame cannot be mistaken for audio. Shared with tts.py and every
# speaker driver, each of which carries its own copy of these bytes.
AUDIO_EOF = b'\x01\x00\xff\xff\x01\x00\xff\xff'
FRAME_BYTES = 3200                  # 100 ms @ 16 kHz, 16-bit, mono
FRAME_SECONDS = 0.1
BYTES_PER_SECOND = 32000
# 20 s per source. A bound rather than a drop-oldest ring: speech that has fallen
# 20 s behind is not worth saying, and an unbounded buffer turns a stuck reader
# into an OOM.
MAX_VOICE_BYTES = 20 * BYTES_PER_SECOND
NO_READER_GRACE_S = 5.0
# 语音的抖动缓冲：开始播一段话之前，先攒够这么多。
#
# 没有这个的话，只要背景在响，混音器就每帧必发（`if speech or background`），于是
# 语音每一次晚到都被固化成输出流里一个 100ms 的静音洞 —— 听感就是说话断断续续。
# 而**没有**背景时同样的晚到不会有洞：那一拍两个源都空，什么都不发，节奏自然跟着
# 语音走。所以这个 bug 只在"一边放音乐一边说话"时出现。
#
# 200ms 而不是更多：tts.py 本来就先突发 5 帧再按节奏发，所以这点余量几乎立刻就
# 攒满，不会给首字增加可感知的延迟 —— 而延迟直接影响打断响应。
SPEECH_PREFILL_MS = 200
# A lost end-of-speech must not leave the background permanently inaudible.
DUCK_FAILSAFE_S = 60.0


def bounded_number(value, low, high, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f'{name} must be between {low} and {high}')
    return float(value)


def paced_loop(stop_event: threading.Event, tick, *, clock=time.monotonic,
               period: float = FRAME_SECONDS) -> None:
    """Call `tick` every `period` seconds until `stop_event` is set.

    The deadline is advanced by `period` rather than recomputed from now, so
    scheduling jitter does not accumulate. But a deadline already in the past is
    reset to now instead of being chased: catching up would publish several
    frames back-to-back, and a speaker fed a burst plays it as a chirp.
    """
    deadline = clock()
    while not stop_event.is_set():
        tick()
        deadline += period
        now = clock()
        if deadline < now:
            deadline = now
        stop_event.wait(max(0, deadline - now))


def _ramp_coefficient(duck_ramp_ms: float) -> float:
    """Per-frame approach fraction for an exponential gain ramp.

    0.5 at the 200 ms default, which is the value this mixer shipped with and
    the one its click-free assertions were written against. Scaled inversely so
    a longer configured ramp really is gentler, and clamped so a very short one
    degrades to an immediate jump rather than overshooting past the target.
    """
    return min(1.0, max(0.01, 0.5 * (200.0 / max(1.0, float(duck_ramp_ms)))))


class Mixer:
    """Sums speech and background PCM into one 16 kHz mono stream.

    Speech is never attenuated and never interrupted — it is the robot talking,
    and the whole point of ducking is that it stays intelligible. Background is
    attenuated to `duck_gain` while speech is present and restored after, over a
    ramp, because a step change in gain is audible as a click.
    """

    def __init__(self, publish, *, duck_gain=0.15, duck_ramp_ms=200,
                 duck_hold_ms=300, background_gain=100,
                 speech_prefill_ms=SPEECH_PREFILL_MS,
                 has_reader=lambda: True, clock=time.monotonic):
        self._publish, self._has_reader, self._clock = publish, has_reader, clock
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None

        self.duck_gain = bounded_number(duck_gain, 0, 1, 'duck_gain')
        self.duck_ramp_ms = bounded_number(duck_ramp_ms, 20, 1000, 'duck_ramp_ms')
        self.duck_hold_ms = bounded_number(duck_hold_ms, 0, 5000, 'duck_hold_ms')
        self.background_gain = bounded_number(background_gain, 0, 100, 'background_gain')
        self.speech_prefill_ms = bounded_number(speech_prefill_ms, 0, 2000,
                                                'speech_prefill_ms')

        self._speech = bytearray()
        self._speech_open = False
        # 这一段话是否已经开始输出。一旦开始就一直放到说完，中途的欠载由上面攒的
        # 余量吸收，而不是变成一个洞。
        self._speech_playing = False
        self._speech_last = 0.0
        self._discard_speech = False
        self._background = bytearray()
        self._background_open = False
        self._background_last = 0.0

        self._duck_until = 0.0
        self._gain = 0.0
        self._output_active = False
        self._wait_started = self._clock()
        self.error = ''

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self):
        self._thread = threading.Thread(target=self._run, name='audio-mix', daemon=True)
        self._thread.start()

    def request_stop(self):
        self._stop.set()
        with self._lock:
            self._speech.clear()
            self._background.clear()
            self._speech_open = self._background_open = False
            self._speech_playing = False

    def close(self):
        self.request_stop()
        if self._thread:
            self._thread.join(timeout=2)

    # ── input ────────────────────────────────────────────────────────────────
    def feed(self, pcm: bytes, *, speech: bool) -> None:
        """Accept one chunk from a source, routed by the role its topic resolved to."""
        with self._lock:
            if pcm == AUDIO_EOF:
                if speech:
                    # The end of one utterance also lifts the post-interrupt
                    # discard: frames after this point belong to the next thing
                    # the robot says, which must not be swallowed.
                    self._speech_open = self._discard_speech = False
                else:
                    self._background_open = False
                return
            if self._stop.is_set():
                return
            if speech and self._discard_speech:
                return
            if len(pcm) % 2:
                self.error = ('Speech PCM must contain whole S16LE samples' if speech
                              else 'Background PCM must contain whole S16LE samples')
                return
            buffer = self._speech if speech else self._background
            if len(buffer) + len(pcm) > MAX_VOICE_BYTES:
                self.error = (f'{"Speech" if speech else "Background"} input exceeded '
                              'the bounded playback buffer')
                buffer.clear()
                if speech:
                    self._discard_speech = True
                return
            buffer.extend(pcm)
            if speech:
                self._speech_open = True
                self._speech_last = self._clock()
            else:
                self._background_open = True
                self._background_last = self._clock()

    def flush_speech(self):
        """Drop buffered speech and ignore the rest of this utterance.

        Barge-in: synthesis has been told to stop, but up to half a second of it
        is already sitting here (tts.py bursts a 5-frame prebuffer). Without
        this the robot keeps talking after being interrupted. Discarding runs
        until the next AUDIO_EOF so the *following* utterance is unaffected.
        """
        with self._lock:
            self._discard_speech = self._speech_open
            self._speech.clear()
            self._speech_open = self._speech_playing = False
            return self.status()

    def duck(self, active: bool):
        """Duck on demand, for a hook or an operator rather than for PCM."""
        with self._lock:
            self._duck_until = self._clock() + (
                DUCK_FAILSAFE_S if active else self.duck_hold_ms / 1000)
            return self.status()

    def configure(self, **kw):
        with self._lock:
            for name, (low, high) in (('duck_gain', (0, 1)), ('duck_ramp_ms', (20, 1000)),
                                      ('duck_hold_ms', (0, 5000)),
                                      ('speech_prefill_ms', (0, 2000)),
                                      ('background_gain', (0, 100))):
                if kw.get(name) is not None:
                    setattr(self, name, bounded_number(kw[name], low, high, name))
            return self.status()

    def _speech_ready(self) -> bool:
        """这一段话可以开始往外送了吗。调用方持有 `_lock`。"""
        if self._speech_playing:
            return True
        if not self._speech:
            return False
        # EOF 已经到了：这段话就这么多，再等也不会更多。短句不能被 prefill 卡住。
        if not self._speech_open:
            return True
        return len(self._speech) >= (self.speech_prefill_ms / 1000 * BYTES_PER_SECOND)

    # ── output ───────────────────────────────────────────────────────────────
    def ducked(self) -> bool:
        with self._lock:
            return bool(self._speech or self._speech_open
                        or self._clock() < self._duck_until)

    def status(self):
        with self._lock:
            return {'speech_buffered_ms': len(self._speech) / BYTES_PER_SECOND * 1000,
                    'background_buffered_ms': len(self._background) / BYTES_PER_SECOND * 1000,
                    'speaking': bool(self._speech or self._speech_open),
                    'speech_prefill_ms': self.speech_prefill_ms,
                    'ducked': self.ducked(),
                    'background_gain_now': round(self._gain, 4),
                    'duck_gain': self.duck_gain, 'duck_ramp_ms': self.duck_ramp_ms,
                    'duck_hold_ms': self.duck_hold_ms,
                    'background_gain': self.background_gain,
                    'error': self.error}

    def tick(self):
        """Emit at most one 100 ms frame. Only the output worker calls this."""
        with self._lock:
            if self._stop.is_set():
                return
            now = self._clock()
            # A source that said something and then went quiet without an EOF is
            # not still speaking. Without this the duck would never lift.
            if self._speech_open and not self._speech and now - self._speech_last > 2:
                self._speech_open = False
            if self._background_open and not self._background and now - self._background_last > 2:
                self._background_open = False

            if not self._has_reader():
                if ((self._speech or self._background)
                        and now - self._wait_started > NO_READER_GRACE_S):
                    self.error = ('No PCM subscriber; connect audio_mixer output '
                                  'to Speaker')
                return
            self._wait_started = now

            if self._speech_ready():
                speech = bytes(self._speech[:FRAME_BYTES])
                del self._speech[:len(speech)]
                self._speech_playing = True
            else:
                # 还在攒。背景照常走，语音这一帧先不动 —— 攒的过程中已经在 duck 了
                # （`_speech_open` 就是 duck 的触发条件之一），所以音乐早就让开了。
                speech = b''
            if self._speech_playing and not self._speech and not self._speech_open:
                self._speech_playing = False      # 这段话说完了
            background = bytes(self._background[:FRAME_BYTES])
            del self._background[:len(background)]

            if speech or background:
                count = max(len(speech), len(background)) // 2
                output = np.zeros(count, dtype=np.float32)
                duck = bool(speech or self._speech_open or now < self._duck_until)
                target = (self.background_gain / 100
                          * (self.duck_gain if duck else 1))
                new_gain = self._gain + (target - self._gain) * _ramp_coefficient(
                    self.duck_ramp_ms)
                if background:
                    samples = np.frombuffer(background, dtype='<i2').astype(np.float32)
                    # Per-sample ramp, not a per-frame step: a 100 ms step in gain
                    # is audible as a click at the start of every utterance.
                    output[:len(samples)] += samples * np.linspace(
                        self._gain, new_gain, len(samples))
                self._gain = new_gain
                if speech:
                    # Unity gain, always. duck_gain and background_gain never
                    # touch the robot's own voice.
                    output[:len(speech) // 2] += np.frombuffer(speech, dtype='<i2')
                self._publish(np.clip(output, -32768, 32767).astype('<i2').tobytes())
                self._output_active = True
            elif self._output_active and not self._speech_open and not self._background_open:
                # A speech EOF is internal while background is still live — only
                # the *combined* stream's boundary is a boundary for the speaker.
                self._publish(AUDIO_EOF)
                self._output_active = False

    def _run(self):
        try:
            paced_loop(self._stop, self.tick, clock=self._clock)
        except Exception:
            with self._lock:
                self.error = 'Audio mixing failed'
        finally:
            if self._output_active:
                try:
                    self._publish(AUDIO_EOF)
                except Exception:
                    pass        # Output failure is already reflected in status.
