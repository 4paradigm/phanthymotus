from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock

import numpy as np


PERCEPTION_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

import plugins.duplexaudio.aec as aec_module  # noqa: E402
from plugins.duplexaudio.aec import (  # noqa: E402
    AECBackendError,
    AECProcessor,
    FRAME_SAMPLES,
    build_backend,
)


class _FakeBackend:
    name = "fake-aec"

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.references = []
        self.closed = False

    def process(self, mic: np.ndarray, reference: np.ndarray) -> np.ndarray:
        self.references.append(reference.copy())
        if self.fail:
            raise RuntimeError("backend failure")
        return np.clip(
            mic.astype(np.int32) - reference.astype(np.int32), -32768, 32767
        ).astype(np.int16)

    def close(self) -> None:
        self.closed = True


def _processor(delay_ms: int = 100, fail: bool = False):
    backend = _FakeBackend(fail=fail)
    processor = AECProcessor(
        delay_ms=delay_ms,
        backend_factory=lambda: backend,
    )
    return processor, backend


class AECProcessorTests(unittest.TestCase):
    def test_timestamped_reference_aligns_and_is_removed(self):
        processor, backend = _processor(delay_ms=100)
        start = 1000.0
        reference = np.full(FRAME_SAMPLES * 3, 1000, dtype=np.int16)
        microphone = np.full(FRAME_SAMPLES * 3, 1500, dtype=np.int16)
        processor.push_reference(reference.tobytes(), start)

        capture_end = start + 0.100 + 0.030
        clean = np.frombuffer(
            processor.process_pcm(microphone.tobytes(), capture_end),
            dtype=np.int16,
        )

        np.testing.assert_array_equal(clean, np.full_like(clean, 500))
        self.assertEqual(len(backend.references), 3)
        stats = processor.stats()
        self.assertEqual(stats["aligned_frames"], 3)
        self.assertEqual(stats["frames_with_reference"], 3)
        self.assertEqual(stats["silence_frames"], 0)
        self.assertEqual(stats["align_rate"], 1.0)

    def test_no_reference_is_a_signal_passthrough_for_fake_backend(self):
        processor, _backend = _processor()
        microphone = np.arange(FRAME_SAMPLES * 2, dtype=np.int16)
        clean = np.frombuffer(
            processor.process_pcm(microphone.tobytes(), 2000.02),
            dtype=np.int16,
        )
        np.testing.assert_array_equal(clean, microphone)
        stats = processor.stats()
        self.assertEqual(stats["aligned_frames"], 0)
        self.assertEqual(stats["silence_frames"], 2)

    def test_stream_lock_survives_bursty_capture_timestamps(self):
        processor, _backend = _processor(delay_ms=100)
        start = 3000.0
        reference = np.full(FRAME_SAMPLES * 6, 700, dtype=np.int16)
        microphone = np.full(FRAME_SAMPLES * 3, 900, dtype=np.int16)
        processor.push_reference(reference.tobytes(), start)

        first = processor.process_pcm(microphone.tobytes(), start + 0.13)
        second = processor.process_pcm(microphone.tobytes(), start + 0.16 + 0.08)

        self.assertEqual(len(first), microphone.nbytes)
        self.assertEqual(len(second), microphone.nbytes)
        stats = processor.stats()
        self.assertEqual(stats["aligned_frames"], 6)
        self.assertEqual(stats["relocks"], 0)
        self.assertEqual(stats["mic_relocks"], 0)

    def test_partial_microphone_frame_is_carried_without_padding(self):
        processor, _backend = _processor()
        first = np.full(FRAME_SAMPLES - 10, 123, dtype=np.int16)
        second = np.full(20, 123, dtype=np.int16)

        self.assertEqual(processor.process_pcm(first.tobytes(), 4000.01), b"")
        output = processor.process_pcm(second.tobytes(), 4000.02)

        self.assertEqual(len(output), FRAME_SAMPLES * 2)
        self.assertEqual(processor.stats()["frames_processed"], 1)

    def test_backend_failure_is_visible(self):
        processor, _backend = _processor(fail=True)
        microphone = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        with self.assertRaisesRegex(RuntimeError, "backend failure"):
            processor.process_pcm(microphone.tobytes(), 5000.01)
        self.assertEqual(processor.stats()["process_errors"], 1)

    def test_envelope_calibration_recovers_known_delay(self):
        processor, _backend = _processor(delay_ms=120)
        rng = np.random.default_rng(7)
        amplitudes = rng.integers(100, 5000, size=300, dtype=np.int16)
        chunks = [
            np.full(FRAME_SAMPLES, int(amplitude), dtype=np.int16)
            for amplitude in amplitudes
        ]
        pcm = np.concatenate(chunks)
        start = 6000.0
        processor.push_reference(pcm.tobytes(), start)
        processor.process_pcm(pcm.tobytes(), start + 0.120 + 3.0)

        result = processor.calibrate()

        self.assertTrue(result["ok"], result)
        self.assertGreater(result["peak_corr"], 0.9)
        self.assertLessEqual(abs(result["d_real_ms"] - 120), 10)

    def test_invalid_backend_is_rejected(self):
        with self.assertRaisesRegex(AECBackendError, "unsupported AEC backend"):
            build_backend("invalid", 16000, FRAME_SAMPLES, 200)

    def test_auto_backend_falls_back_to_speexdsp(self):
        speex = _FakeBackend()
        speex.name = "speexdsp"
        with mock.patch.object(
            aec_module,
            "_LiveKitBackend",
            side_effect=AECBackendError("LiveKit unavailable"),
        ), mock.patch.object(aec_module, "_SpeexDSPBackend", return_value=speex):
            backend = build_backend("auto", 16000, FRAME_SAMPLES, 200)

        self.assertIs(backend, speex)

    def test_close_releases_backend(self):
        processor, backend = _processor()
        processor.close()
        self.assertTrue(backend.closed)


if __name__ == "__main__":
    unittest.main()
