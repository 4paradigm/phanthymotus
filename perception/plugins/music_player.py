"""One music session: decode a signed URL and emit paced music PCM.

Mixing used to live here too, which is what made the music card secretly the
mixer — it took TTS in and published the combined stream. Speech handling now
lives in `audio_mix.Mixer` behind its own card, and this file is left with the
one job its name claims: a single song, its decoder, and its transport states.

`AUDIO_EOF` and `bounded_number` are re-exported because they were imported from
here before the split, and moving a name is not worth breaking every caller over.
"""
from collections import deque
from datetime import datetime, timezone
import queue
import threading
import time
import uuid

import numpy as np

from plugins.music_decoder import Decoder, validate_url
from plugins.audio_mix import (AUDIO_EOF, BYTES_PER_SECOND, FRAME_BYTES,
                               FRAME_SECONDS, MAX_VOICE_BYTES, bounded_number,
                               paced_loop)

__all__ = ['AUDIO_EOF', 'FRAME_BYTES', 'FRAME_SECONDS', 'MAX_VOICE_BYTES',
           'bounded_number', 'MusicPlayer']


class MusicPlayer:
    def __init__(self, publish, *, decoder_factory=Decoder, volume=50,
                 has_reader=lambda: True, clock=time.monotonic, on_drained=None):
        self._publish, self._factory, self._has_reader, self._clock = publish, decoder_factory, has_reader, clock
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._decoder = None
        self._retired = deque()
        self._output_active = False
        self._gain = 0.0
        # Called once per session when the track reaches a terminal state, off
        # the output thread's critical section. The card uses it to post ACP
        # completion and, in station mode, to queue the next track — neither of
        # which may happen inline: completion posts over HTTP and the next track
        # costs a catalogue round trip, and this thread owes a frame every 100 ms.
        self._on_drained = on_drained
        self.volume = bounded_number(volume, 0, 100, 'volume')
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
            self._gain = 0.0        # ramp in rather than slam to full volume
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

    def status(self):
        with self._lock:
            return {'playback_state': self.state, 'playback_id': self.playback_id,
                    'track_id': self.track_id, 'position_s': round(self.position_s, 2),
                    'volume': self.volume,
                    'attribution': self.attribution, 'error': self.error,
                    'completion_scope': 'source_drained; speaker buffer may still contain audio'}

    def tick(self):
        """Emit at most 100 ms of music. Only the output worker calls this."""
        drained = None
        with self._lock:
            if self._stop.is_set():
                return
            now = self._clock()
            if not self._has_reader():
                if self._decoder and self.state != 'paused' and now - self._wait_started > 5:
                    self._cancel_music()
                    self.state, self.error = 'error', ('No PCM subscriber; connect '
                                                       'phanthy_music output to audio_mixer')
                return
            music = b''
            decoder = self._decoder
            if decoder and self.state != 'paused':
                try:
                    music = decoder.frames.get_nowait()
                    self.position_s += len(music) / BYTES_PER_SECOND
                    self.state, self._last_audio_at = 'playing', now
                except queue.Empty:
                    if decoder.done.is_set():
                        self.error = decoder.error
                        self.state = 'error' if self.error else 'completed'
                        # A failed decoder can finish before its HTTPS reader.
                        # Keep it counted against the bound on retiring workers.
                        self._retired.append(decoder)
                        self._decoder = None
                        drained = (self.playback_id, self.state, self.error)
                    elif now - self._last_audio_at > 15:
                        self._cancel_music()
                        self.state, self.error = 'error', 'Audio stream stalled'
                        drained = (self.playback_id, self.state, self.error)
            if music:
                samples = np.frombuffer(music, dtype='<i2').astype(np.float32)
                target = self.volume / 100
                # Ramp rather than step, for the same reason the mixer does: the
                # first frame of a track would otherwise start at full level
                # against whatever gain the previous one ended on.
                new_gain = self._gain + (target - self._gain) * 0.5
                output = samples * np.linspace(self._gain, new_gain, len(samples))
                self._gain = new_gain
                self._publish(np.clip(output, -32768, 32767).astype('<i2').tobytes())
                self._output_active = True
            elif self._output_active and self._decoder is None:
                self._publish(AUDIO_EOF)
                self._output_active = False
        # Outside the lock: the callback posts HTTP and may fetch the next track,
        # and this thread owes a frame every 100 ms.
        if drained and self._on_drained:
            try:
                self._on_drained(*drained)
            except Exception:
                pass    # A card-level failure must not kill the output loop.

    def _run(self):
        try:
            paced_loop(self._stop, self.tick, clock=self._clock)
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
