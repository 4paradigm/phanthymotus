"""
Host-side unit tests for utils/resample (no sherpa-onnx and no model required).

Run from the repo root:
    python -m pytest perception/tests -q

This is the only part of the Kokoro engine that can be proven without a Jetson,
so it carries the weight: if the filter is wrong, the symptom on device is
"the English voice sounds slightly metallic", which nobody would trace back to a
resampler. The stopband test is the one that actually earns its keep — a plain
`samples[::3]` decimation with no lowpass passes every other test in this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from utils import resample  # noqa: E402

SRC_RATE = 24000
DST_RATE = 16000


def _tone(freq: float, seconds: float = 0.5, rate: int = SRC_RATE,
          amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * seconds), dtype=np.float32) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _dominant_freq(samples: np.ndarray, rate: int) -> float:
    """Frequency of the largest spectral peak, ignoring DC."""
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    spectrum[0] = 0.0
    return float(np.fft.rfftfreq(len(samples), 1.0 / rate)[int(np.argmax(spectrum))])


def _power_at(samples: np.ndarray, rate: int, freq: float, width: float = 120.0) -> float:
    """Summed spectral power in a narrow band around `freq`."""
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples)))) ** 2
    freqs = np.fft.rfftfreq(len(samples), 1.0 / rate)
    band = (freqs >= freq - width) & (freqs <= freq + width)
    return float(spectrum[band].sum())


# ── length arithmetic ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 100, 24000, 24001, 35999])
def test_output_length_is_ceil_two_thirds(n):
    """The pacing clock divides byte counts by 16000*2, so length must be exact."""
    out = resample.resample_poly(np.zeros(n, dtype=np.float32))
    assert len(out) == resample.resampled_length(n)
    assert len(out) == -(-n * 2 // 3)  # ceil(n*2/3) without float rounding


def test_one_second_in_is_one_second_out():
    """Duration must survive, or audio drifts against the 100 ms frame clock."""
    out = resample.resample_poly(_tone(1000.0, seconds=1.0))
    assert abs(len(out) / DST_RATE - 1.0) < 1e-3


def test_empty_input_is_not_an_error():
    """A frontend can normalise a whole utterance away; that must not raise."""
    assert resample.resample_poly(np.zeros(0, dtype=np.float32)).size == 0
    assert resample.downsample_24k_to_16k(np.zeros(0, dtype=np.float32)) == b""


# ── the filter actually filters ───────────────────────────────────────────────

@pytest.mark.parametrize("freq", [200.0, 1000.0, 3000.0, 6000.0])
def test_passband_tones_keep_their_pitch(freq):
    """Wrong pitch is the failure a rate mismatch produces, so test for it."""
    out = resample.resample_poly(_tone(freq, seconds=1.0))
    assert abs(_dominant_freq(out, DST_RATE) - freq) < 15.0


def test_a_tone_above_the_new_nyquist_is_attenuated_not_aliased():
    """The test a bare `samples[::3]` fails.

    11 kHz is above the 16 kHz output's 8 kHz Nyquist. Without a lowpass it folds
    down to |16000 - 11000| = 5 kHz and lands in the middle of the speech band as
    a phantom tone. With the lowpass it is simply gone.
    """
    out = resample.resample_poly(_tone(11000.0, seconds=1.0))

    alias_power = _power_at(out, DST_RATE, 5000.0)
    reference = _power_at(resample.resample_poly(_tone(5000.0, seconds=1.0)),
                          DST_RATE, 5000.0)
    # The alias must be far below what a genuine 5 kHz tone of the same amplitude
    # looks like. 40 dB is a loose bound the Blackman design clears easily; it is
    # written loose on purpose so a legitimate change of window or tap count does
    # not fail it, while `[::3]` (which lands at roughly 0 dB) still does.
    assert alias_power < reference / 10_000, (
        f"alias power {alias_power:.3e} vs reference {reference:.3e} — "
        "the lowpass is not attenuating above the output Nyquist"
    )


def test_dc_gain_is_unity():
    """Zero-stuffing divides average power by UP; the taps must put it back."""
    out = resample.resample_poly(np.full(6000, 0.5, dtype=np.float32))
    # Ignore the filter's edge transients at both ends.
    assert np.allclose(out[200:-200], 0.5, atol=1e-3)


def test_taps_sum_to_the_upsampling_factor():
    """Guards the normalisation directly, independent of any signal."""
    assert resample._H.sum() == pytest.approx(resample.UP, abs=1e-5)
    assert len(resample._H) == resample.TAPS
    assert resample.TAPS % 2 == 1, "an even tap count gives a half-sample delay"


# ── int16 conversion ──────────────────────────────────────────────────────────

def test_pcm16_is_little_endian_signed_16_bit():
    pcm = resample.float_to_pcm16(np.array([0.0, 1.0, -1.0], dtype=np.float32))
    assert len(pcm) == 6
    assert np.frombuffer(pcm, dtype="<i2").tolist() == [0, 32767, -32767]


def test_overshoot_clips_instead_of_wrapping():
    """astype(int16) on an out-of-range value wraps, flipping the sign.

    A sample that overshot 1.0 would come out as full-scale *negative* — a tick
    on the loudest part of the utterance. Clipping has to happen in float first.
    """
    pcm = resample.float_to_pcm16(np.array([1.5, -1.5, 12.0], dtype=np.float32))
    assert np.frombuffer(pcm, dtype="<i2").tolist() == [32767, -32768, 32767]


def test_downsample_returns_bytes_of_the_expected_frame_count():
    """The adapter slices this into CHUNK_BYTES frames, so byte count matters."""
    pcm = resample.downsample_24k_to_16k(_tone(440.0, seconds=1.0))
    assert isinstance(pcm, bytes)
    assert len(pcm) == resample.resampled_length(SRC_RATE) * 2
    assert len(pcm) == DST_RATE * 2  # exactly one second of 16-bit mono


def test_a_loud_utterance_survives_without_clipping_artefacts():
    """Near-full-scale input is the realistic case; check it stays a clean tone."""
    pcm = resample.downsample_24k_to_16k(_tone(700.0, seconds=0.5, amplitude=0.95))
    out = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32767.0
    assert abs(_dominant_freq(out, DST_RATE) - 700.0) < 15.0
    assert np.abs(out).max() < 1.0
