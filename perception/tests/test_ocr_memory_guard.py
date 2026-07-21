import sys
import unittest
from pathlib import Path
from unittest import mock


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins.ocr_memory_guard import MemoryGuardConfig, OCRMemoryGuard


class OCRMemoryGuardTest(unittest.TestCase):
    def _guard(self, **overrides):
        values = {
            "enabled": True,
            "expected_workers": 10,
            "min_decode_mb": 8,
            "headroom_ratio": 0.2,
        }
        values.update(overrides)
        return OCRMemoryGuard(MemoryGuardConfig.from_mapping(values))

    def test_roomy_eight_gib_host_keeps_hard_limit(self):
        guard = self._guard()
        with mock.patch.object(
            guard, "_cgroup_headroom_bytes", return_value=None
        ), mock.patch.object(
            guard, "_host_available_bytes", return_value=8 * 1024**3
        ):
            limit = guard.decode_limit_bytes(64 * 1024**2)

        self.assertEqual(limit, 64 * 1024**2)

    def test_host_pressure_reduces_per_worker_decode_limit(self):
        guard = self._guard()
        with mock.patch.object(
            guard, "_cgroup_headroom_bytes", return_value=None
        ), mock.patch.object(
            guard, "_host_available_bytes", return_value=1024**3
        ):
            limit = guard.decode_limit_bytes(64 * 1024**2)

        self.assertEqual(limit, int(1024**3 / 10 * 0.2))

    def test_low_cgroup_headroom_enters_emergency_reject_mode(self):
        guard = self._guard()
        with mock.patch.object(
            guard, "_cgroup_headroom_bytes", return_value=20 * 1024**2
        ), mock.patch.object(
            guard, "_host_available_bytes", return_value=8 * 1024**3
        ):
            limit = guard.decode_limit_bytes(64 * 1024**2)

        self.assertEqual(limit, 0)

    def test_missing_memory_metrics_falls_back_to_hard_limit(self):
        guard = self._guard()
        with mock.patch.object(
            guard, "_cgroup_headroom_bytes", return_value=None
        ), mock.patch.object(
            guard, "_host_available_bytes", return_value=None
        ):
            limit = guard.decode_limit_bytes(64 * 1024**2)

        self.assertEqual(limit, 64 * 1024**2)


if __name__ == "__main__":
    unittest.main()
