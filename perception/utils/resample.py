"""Sample-rate conversion for TTS engines whose model rate is not the pipeline's.

The whole audio path is 16 kHz: `AudioChunk.msg` has no `sample_rate` field, the
rate lives inside the `"audio/pcm-16k"` format string, and the publisher's pacing
constants are derived from it (plugins/tts.py:28-30). Every engine before Kokoro
either was natively 16 kHz or was *refused* — `MmsThaiTTSAdapter` raises rather
than resample a 22.05 kHz voice, and `tools/export_mms_thai_onnx.py` refuses to
export one.

Kokoro is 24 kHz and worth having anyway, so the conversion happens here, inside
the adapter's own output path. Nothing downstream learns about it: the topic is
still `audio/pcm-16k` and still carries 16 kHz.

Deliberately numpy-only. scipy, soxr, librosa and torchaudio are all absent from
the perception image, and scipy in particular would pull its own numpy pin —
which is the exact failure both plugin requirements files carry long comments
about preventing (numpy is 1.24.4 on jp5.11/cp38 and 1.26.4 on jp6.1/cp310, and
torch, cv2 and rapidocr are all built against those ABIs).
"""

import math

import numpy as np

# 24000 -> 16000 is exactly 2:3, which is why Kokoro is cheap to support and a
# 22.05 kHz model would not be: that one is 160:441 and needs a 441-phase filter.
UP = 2
DOWN = 3

# 6 * DOWN * 8 + 1. Odd, so the group delay is an exact integer number of samples
# and `mode="same"` leaves no fractional offset.
TAPS = 97

# The target's Nyquist (8 kHz) expressed against the *upsampled* rate of
# 24000 * UP = 48 kHz: 8000/48000 = 1/6.
CUTOFF = 1.0 / 6


def _design_lowpass(taps: int = TAPS, up: int = UP, cutoff: float = CUTOFF) -> np.ndarray:
    """A windowed-sinc lowpass, normalised to unity DC gain after zero-stuffing.

    Blackman rather than Kaiser because numpy has `np.blackman` and not
    `scipy.signal.kaiser`; its ~-58 dB sidelobes are already well below a 16-bit
    LSB (-90 dB is the floor, but the model's own noise sits far above -58 dB),
    so the extra stopband depth a Kaiser would buy is inaudible here.

    The `up` factor compensates the zero-stuffing in `resample_poly`: inserting
    UP-1 zeros between samples divides the signal's average power by UP, and
    normalising the taps to sum to `up` puts it back. Normalising by the actual
    tap sum rather than the analytic 2*cutoff matters — the window truncates the
    sinc, so the two differ by a fraction of a percent, and that difference would
    show up as a small but real DC gain error.
    """
    n = np.arange(taps) - (taps - 1) / 2.0
    h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.blackman(taps)
    return (h * (up / h.sum())).astype(np.float32)


_H = _design_lowpass()


def resample_poly(samples: np.ndarray, up: int = UP, down: int = DOWN,
                  taps: np.ndarray = None) -> np.ndarray:
    """Rational resampling by `up/down`: zero-stuff, lowpass, decimate.

    `samples` must be the **whole** waveform for one synthesis call, not a frame
    of one. The filter keeps no state between calls, so resampling 100 ms frames
    independently would inject a transient at every frame boundary — ten clicks a
    second. The adapters therefore resample what `generate()` returned and only
    then slice it into CHUNK_BYTES frames.

    The centred slice out of the `"full"` convolution drops the filter's
    (taps-1)/2 samples of group delay, so the output is time-aligned with the
    input and no leading silence is introduced.

    Not `mode="same"`: numpy defines that as `max(len(signal), len(taps))`, so an
    utterance shorter than the 97-tap filter comes back padded out to 97 samples
    — the tail of the filter's own impulse response, resampled. A punctuation-only
    chunk hits exactly that. Slicing the `"full"` result is what `"same"` does for
    long inputs and stays correct for short ones.
    """
    if taps is None:
        taps = _H
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        return samples
    upsampled = np.zeros(samples.size * up, dtype=np.float32)
    upsampled[::up] = samples
    full = np.convolve(upsampled, taps, mode="full")
    start = (len(taps) - 1) // 2
    return full[start:start + upsampled.size][::down]


def resampled_length(n: int, up: int = UP, down: int = DOWN) -> int:
    """How many samples `resample_poly` returns for `n` inputs.

    Exposed so a caller (or a test) can assert the arithmetic without running the
    filter: `[::down]` over `n * up` points yields ceil(n * up / down).
    """
    return 0 if n <= 0 else math.ceil(n * up / down)


def float_to_pcm16(samples: np.ndarray) -> bytes:
    """Float samples in [-1, 1] -> little-endian signed 16-bit PCM bytes.

    Clipping happens in float, before the cast: `astype(np.int16)` on a value
    outside the int16 range is undefined in numpy and in practice wraps, so a
    sample that overshot 1.0 would come out as full-scale *opposite* polarity —
    an audible tick exactly on the loudest part of the utterance.

    Scaled by 32767 rather than 32768 so that +1.0 maps to +32767 and cannot
    reach the asymmetric negative rail.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        return b""
    scaled = np.clip(samples * 32767.0, -32768.0, 32767.0)
    return scaled.astype("<i2").tobytes()


def downsample_24k_to_16k(samples: np.ndarray) -> bytes:
    """24 kHz float samples -> 16 kHz little-endian int16 PCM bytes.

    The one function TTS adapters call. Kept as a named 24->16 wrapper rather than
    leaving callers to pass `up`/`down` themselves, so the rate pair is stated in
    one place and a future engine at another rate has to add its own entry point
    (and think about its own filter) instead of quietly reusing this filter at a
    ratio it was not designed for.
    """
    return float_to_pcm16(resample_poly(samples))
