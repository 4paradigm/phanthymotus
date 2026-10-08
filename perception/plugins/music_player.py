"""One music session and a speech-priority PCM mixer, independent of ROS/SDKs."""
from collections import deque
from datetime import datetime, timezone
import math
import queue
import threading
import time
import uuid

import numpy as np

from plugins.music_decoder import Decoder, FRAME_BYTES, validate_url

AUDIO_EOF = b'\x01\x00\xff\xff\x01\x00\xff\xff'
FRAME_SECONDS = 0.1
MAX_VOICE_BYTES = 20 * 32000


def bounded_number(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return float(value)


class MusicPlayer:
    def __init__(self, publish, *, decoder_factory=Decoder, volume=50, duck_gain=0.15,
                 has_reader=lambda: True, clock=time.monotonic):
        self._publish, self._factory, self._has_reader, self._clock = publish, decoder_factory, has_reader, clock
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._decoder = None
        self._retired = deque()
        self._voice = bytearray()
        self._voice_open = False
        self._voice_last = 0.0
        self._discard_voice = False
        self._duck_until = 0.0
        self._output_active = False
        self._music_gain = 0.0
        self.volume = bounded_number(volume, 0, 100, 'volume')
        self.duck_gain = bounded_number(duck_gain, 0, 1, 'duck_gain')
        self.state = 'idle'
        self.playback_id = ''
        self.track_id = ''
        self.attribution = ''
        self.error = ''
        self.position_s = 0.0
        self._last_audio_at = self._clock()
        self._wait_started = self._clock()

    def start(self):
        self._thread = threading.Thread(target=self._run, name='music-output', daemon=True)
        self._thread.start()

    def request_stop(self):
        """Signal and discard queued output without waiting for retiring workers."""
        self._stop.set()
        with self._lock:
            self._cancel_music()
            self._voice.clear()
            self._voice_open = False

    def close(self):
        self.request_stop()
        if self._thread:
            self._thread.join(timeout=2)
        for decoder in list(self._retired):
            decoder.close()
        self._retired.clear()

    def _cancel_music(self):
        if self._decoder:
            self._decoder.cancel()
            self._retired.append(self._decoder)
            self._decoder = None
        self.state = 'cancelled' if self.playback_id else 'idle'

    def play(self, url, audio_format='mp3', track_id='', expires_at=None, usage=None):
        validate_url(url)
        if audio_format not in ('mp3', 'wav'):
            raise ValueError('Only mp3 and wav audio are supported')
        if expires_at:
            try:
                expiry = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc):
                    raise ValueError
            except (ValueError, TypeError, AttributeError):
                raise ValueError('Audio URL expiry is invalid or passed; refresh the track by ID') from None
        if usage is not None and (not isinstance(usage, dict) or usage.get('scope') != 'personal_playback'):
            raise ValueError('Unsupported usage scope; only current-session personal_playback is supported')
        with self._lock:
            if self._stop.is_set():
                raise ValueError('Music card is stopped')
            # At most two cancelling downloads may still be waiting for the HTTPS
            # read timeout. Rapid skip must not create unbounded worker threads.
            self._retired = deque(d for d in self._retired if not d.is_closed())
            if len(self._retired) >= 2:
                raise ValueError('Previous audio requests are still closing; retry shortly')
            decoder = self._factory(url, audio_format)
            self._cancel_music()
            self._decoder = decoder
            self.playback_id = uuid.uuid4().hex
            self.track_id = str(track_id)[:200]
            self.attribution = str((usage or {}).get('attribution', ''))[:500]
            self.position_s, self.error, self.state = 0.0, '', 'buffering'
            self._wait_started = self._last_audio_at = self._clock()
            decoder.start()
            return self.status()

    def interrupt(self):
        with self._lock:
            self._cancel_music()
            return self.status()

    def pause(self):
        with self._lock:
            if self.state not in ('playing', 'buffering', 'paused'):
                raise ValueError('There is no active song to pause')
            self.state = 'paused'
            return self.status()

    def resume(self):
        with self._lock:
            if self.state != 'paused':
                raise ValueError('There is no paused song')
            self.state = 'buffering'
            self._last_audio_at = self._wait_started = self._clock()
            return self.status()

    def set_volume(self, volume):
        with self._lock:
            self.volume = bounded_number(volume, 0, 100, 'volume')
            return self.status()

    def hearing(self, active):
        with self._lock:
            # A lost end event cannot leave music permanently inaudible.
            self._duck_until = self._clock() + (60 if active else 0.3)
            return self.status()

    def interrupt_voice(self):
        with self._lock:
            self._discard_voice = self._voice_open
            self._voice.clear()
            self._voice_open = False
            return self.status()

    def feed_voice(self, pcm):
        with self._lock:
            if pcm == AUDIO_EOF:
                self._voice_open = self._discard_voice = False
                return
            if self._discard_voice or self._stop.is_set():
                return
            if len(pcm) % 2:
                self.error = 'Speech PCM must contain whole S16LE samples'
                return
            if len(self._voice) + len(pcm) > MAX_VOICE_BYTES:
                self.error = 'Speech input exceeded the bounded playback buffer'
                self._voice.clear()
                self._discard_voice = True
                return
            self._voice.extend(pcm)
            self._voice_open = True
            self._voice_last = self._clock()

    def status(self):
        with self._lock:
            return {'playback_state': self.state, 'playback_id': self.playback_id,
                    'track_id': self.track_id, 'position_s': round(self.position_s, 2),
                    'volume': self.volume, 'ducked': bool(self._voice or self._voice_open or self._clock() < self._duck_until),
                    'attribution': self.attribution, 'error': self.error,
                    'completion_scope': 'source_drained; speaker buffer may still contain audio'}

    def tick(self):
        """Emit at most 100 ms. Called only by the output worker (or a fake clock)."""
        with self._lock:
            if self._stop.is_set():
                return
            now = self._clock()
            if self._voice_open and now - self._voice_last > 2 and not self._voice:
                self._voice_open = False
            if not self._has_reader():
                if self._decoder and self.state != 'paused' and now - self._wait_started > 5:
                    self._cancel_music()
                    self.state, self.error = 'error', 'No PCM subscriber; connect phanthy-music output to Speaker'
                return
            voice = bytes(self._voice[:FRAME_BYTES])
            del self._voice[:len(voice)]
            music = b''
            decoder = self._decoder
            if decoder and self.state != 'paused':
                try:
                    music = decoder.frames.get_nowait()
                    self.position_s += len(music) / 32000
                    self.state, self._last_audio_at = 'playing', now
                except queue.Empty:
                    if decoder.done.is_set():
                        self.error = decoder.error
                        self.state = 'error' if self.error else 'completed'
                        # A failed decoder can finish before its HTTPS reader.
                        # Keep it counted against the bound on retiring workers.
                        self._retired.append(decoder)
                        self._decoder = None
                    elif now - self._last_audio_at > 15:
                        self._cancel_music()
                        self.state, self.error = 'error', 'Audio stream stalled'
            if voice or music:
                count = max(len(voice), len(music)) // 2
                output = np.zeros(count, dtype=np.float32)
                duck = bool(voice or self._voice_open or now < self._duck_until)
                target = self.volume / 100 * (self.duck_gain if duck else 1)
                # 200 ms gain ramp avoids clicks on speech onset and restoration.
                new_gain = self._music_gain + (target - self._music_gain) * 0.5
                if music:
                    samples = np.frombuffer(music, dtype='<i2').astype(np.float32)
                    output[:len(samples)] += samples * np.linspace(self._music_gain, new_gain, len(samples))
                self._music_gain = new_gain
                if voice:
                    samples = np.frombuffer(voice, dtype='<i2')
                    output[:len(samples)] += samples
                self._publish(np.clip(output, -32768, 32767).astype('<i2').tobytes())
                self._output_active = True
            elif self._output_active and not self._voice and not self._voice_open:
                # A speech EOF is internal while music is still active. Only the
                # combined output's boundary reaches Speaker / browser playback.
                if self._decoder is None or self.state == 'paused':
                    self._publish(AUDIO_EOF)
                    self._output_active = False

    def _run(self):
        deadline = self._clock()
        try:
            while not self._stop.is_set():
                self.tick()
                deadline += FRAME_SECONDS
                now = self._clock()
                if deadline < now:
                    deadline = now  # never burst delayed audio into the speaker
                self._stop.wait(max(0, deadline - now))
        except Exception:
            with self._lock:
                self._cancel_music()
                self.state, self.error = 'error', 'Audio output failed'
        finally:
            if self._output_active:
                try:
                    self._publish(AUDIO_EOF)
                except Exception:
                    pass  # Output failure is already reflected in status.
