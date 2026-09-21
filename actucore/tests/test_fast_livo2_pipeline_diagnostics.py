from __future__ import annotations

import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "plugins"
    / "navigation"
    / "runtime"
    / "fast_livo2"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from fast_livo2.fault_diagnostics import (  # noqa: E402
    BRIDGE_STATUS_TOPIC,
    MAPPER_RUNTIME_TOPIC,
    SENSOR_REJECTION_TOPIC,
    build_fault_summary,
)
from fast_livo2.pipeline_diagnostics import PipelineDiagnostics  # noqa: E402


def _bridge(cloud: int, imu: int, *, dropped: int = 0) -> dict:
    return {
        "counters": {
            "cloud_received": cloud,
            "cloud_published": cloud,
            "cloud_dropped": dropped,
            "imu_received": imu,
            "imu_published": imu,
            "imu_dropped": 0,
        }
    }


def _mapper(cloud: int, imu: int, processed: int, **extra) -> dict:
    return {
        "lidar_callbacks": cloud,
        "imu_callbacks": imu,
        "processed_lidar": processed,
        **extra,
    }


def _adapter(cloud: int, **extra) -> dict:
    return {
        "pipeline_counters": {
            "raw_odom_received": cloud,
            "raw_cloud_received": cloud,
            "canonical_odom_published": cloud,
            "canonical_cloud_published": cloud,
            "mapping_work_processed": cloud,
            "map_view_encoded": cloud,
            "map_view_published": cloud,
        },
        "mapping_work_dropped": 0,
        **extra,
    }


class PipelineDiagnosticsTest(unittest.TestCase):
    def _snapshot(self, second, *, bridge, mapper, adapter):
        diagnostic = PipelineDiagnostics()
        for now, scale in ((0.0, 0), (10.0, 1)):
            diagnostic.observe("bridge", bridge(scale), now)
            diagnostic.observe("mapper", mapper(scale), now)
            diagnostic.observe("adapter", adapter(scale), now)
        return diagnostic.snapshot(algorithm_running=True, now_monotonic=second)

    def test_healthy_pipeline_uses_cross_layer_rates(self) -> None:
        result = self._snapshot(
            10.0,
            bridge=lambda scale: _bridge(100 * scale, 2000 * scale),
            mapper=lambda scale: _mapper(100 * scale, 2000 * scale, 100 * scale),
            adapter=lambda scale: _adapter(100 * scale),
        )

        self.assertEqual(result["classification"], "healthy")
        self.assertEqual(result["ratios"]["processed_over_mapper_lidar"], 1.0)
        self.assertNotIn("adapter_lidar_over_driver_cloud", result["ratios"])

    def test_pipeline_distinguishes_source_drop_dds_and_mapper_backlog(self) -> None:
        source = self._snapshot(
            10.0,
            bridge=lambda scale: _bridge(30 * scale, 2000 * scale),
            mapper=lambda scale: _mapper(30 * scale, 2000 * scale, 30 * scale),
            adapter=lambda scale: _adapter(30 * scale),
        )
        dds = self._snapshot(
            10.0,
            bridge=lambda scale: _bridge(100 * scale, 2000 * scale),
            mapper=lambda scale: _mapper(80 * scale, 1800 * scale, 80 * scale),
            adapter=lambda scale: _adapter(80 * scale),
        )
        backlog = self._snapshot(
            10.0,
            bridge=lambda scale: _bridge(100 * scale, 2000 * scale),
            mapper=lambda scale: _mapper(100 * scale, 2000 * scale, 50 * scale),
            adapter=lambda scale: _adapter(100 * scale),
        )

        self.assertEqual(source["classification"], "driver_source_drop")
        self.assertEqual(dds["classification"], "dds_or_subscriber_drop")
        self.assertEqual(backlog["classification"], "fast_livo_processing_backlog")

    def test_counter_restart_uses_only_the_new_segment(self) -> None:
        diagnostic = PipelineDiagnostics()
        for now, count in ((0.0, 100), (5.0, 150), (6.0, 0), (16.0, 100)):
            diagnostic.observe("bridge", _bridge(count, count * 20), now)
            diagnostic.observe("mapper", _mapper(count, count * 20, count), now)
            diagnostic.observe("adapter", _adapter(count), now)

        result = diagnostic.snapshot(algorithm_running=True, now_monotonic=16.0)

        cloud_rate = result["layers"]["bridge"]["rates"]["cloud_published"]
        self.assertEqual(cloud_rate["hz"], 10.0)
        self.assertTrue(cloud_rate["counter_reset"])
        self.assertEqual(result["layers"]["mapper"]["latest"]["processed_lidar"], 100)

    def test_stale_samples_are_not_reported_as_healthy(self) -> None:
        diagnostic = PipelineDiagnostics()
        for now, count in ((0.0, 0), (10.0, 100)):
            diagnostic.observe("bridge", _bridge(count, count * 20), now)
            diagnostic.observe("mapper", _mapper(count, count * 20, count), now)
            diagnostic.observe("adapter", _adapter(count), now)

        result = diagnostic.snapshot(algorithm_running=True, now_monotonic=80.1)

        self.assertEqual(result["classification"], "insufficient_evidence")
        self.assertEqual(result["layers"]["bridge"]["sample_count"], 0)

    def test_fault_summary_does_not_claim_source_drop_without_bridge_evidence(self) -> None:
        records = [
            {
                "topic": MAPPER_RUNTIME_TOPIC,
                "receive_ns": 1_000_000_000,
                "json": _mapper(0, 0, 0),
            },
            {
                "topic": MAPPER_RUNTIME_TOPIC,
                "receive_ns": 11_000_000_000,
                "json": _mapper(20, 2000, 20),
            },
            {
                "topic": SENSOR_REJECTION_TOPIC,
                "receive_ns": 11_100_000_000,
                "json": {"error_code": "raw_odom_discontinuity"},
            },
        ]

        result = build_fault_summary(records, {"reason": "test"})

        self.assertFalse(result["source_drop_proven"])
        self.assertEqual(result["pipeline"]["classification"], "insufficient_evidence")
        self.assertEqual(result["rejections"]["raw_odom_discontinuity"], 1)


if __name__ == "__main__":
    unittest.main()
