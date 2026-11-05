"""Correlate existing Driver, FAST-LIVO2 and adapter counters."""

from __future__ import annotations

from collections import deque
import threading
import time


_WINDOW_SEC = 60.0
_LAYERS = ("bridge", "mapper", "adapter")


def _number(value, default=0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _counter(payload: dict, name: str) -> float:
    counters = payload.get("counters")
    if isinstance(counters, dict) and name in counters:
        return _number(counters.get(name))
    counters = payload.get("pipeline_counters")
    if isinstance(counters, dict) and name in counters:
        return _number(counters.get(name))
    return _number(payload.get(name))


def _rate(samples: list[tuple[float, dict]], name: str) -> dict:
    if len(samples) < 2:
        return {"delta": None, "hz": None, "counter_reset": False}
    segment_start = 0
    for index in range(1, len(samples)):
        if _counter(samples[index][1], name) < _counter(samples[index - 1][1], name):
            segment_start = index
    active = samples[segment_start:]
    if len(active) < 2:
        return {
            "delta": None,
            "hz": None,
            "counter_reset": segment_start > 0,
        }
    first_time, first = active[0]
    last_time, last = active[-1]
    first_value = _counter(first, name)
    last_value = _counter(last, name)
    elapsed = max(0.0, last_time - first_time)
    delta = last_value - first_value
    return {
        "delta": int(delta),
        "hz": None if elapsed <= 0.0 else round(delta / elapsed, 3),
        "counter_reset": segment_start > 0,
    }


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return round(numerator / denominator, 3)


class PipelineDiagnostics:
    """Keep one minute of small JSON snapshots and identify the failing layer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples = {layer: deque(maxlen=128) for layer in _LAYERS}

    def observe(self, layer: str, payload: dict, now_monotonic=None) -> None:
        if layer not in self._samples or not isinstance(payload, dict):
            return
        observed = time.monotonic() if now_monotonic is None else float(now_monotonic)
        with self._lock:
            samples = self._samples[layer]
            samples.append((observed, dict(payload)))
            cutoff = observed - _WINDOW_SEC
            while samples and samples[0][0] < cutoff:
                samples.popleft()

    def snapshot(self, *, algorithm_running: bool, now_monotonic=None) -> dict:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        cutoff = now - _WINDOW_SEC
        with self._lock:
            samples = {
                name: [item for item in items if item[0] >= cutoff]
                for name, items in self._samples.items()
            }

        rates = {
            "bridge": {
                name: _rate(samples["bridge"], name)
                for name in (
                    "cloud_received",
                    "cloud_published",
                    "cloud_dropped",
                    "imu_received",
                    "imu_published",
                    "imu_dropped",
                )
            },
            "mapper": {
                name: _rate(samples["mapper"], name)
                for name in ("lidar_callbacks", "imu_callbacks", "processed_lidar")
            },
            "adapter": {
                name: _rate(samples["adapter"], name)
                for name in (
                    "raw_odom_received",
                    "raw_cloud_received",
                    "canonical_odom_published",
                    "canonical_cloud_published",
                    "mapping_work_processed",
                    "map_view_encoded",
                    "map_view_published",
                    "mapping_work_dropped",
                )
            },
        }
        bridge_cloud = rates["bridge"]["cloud_published"]["hz"]
        bridge_imu = rates["bridge"]["imu_published"]["hz"]
        mapper_cloud = rates["mapper"]["lidar_callbacks"]["hz"]
        mapper_imu = rates["mapper"]["imu_callbacks"]["hz"]
        processed = rates["mapper"]["processed_lidar"]["hz"]
        latest_mapper = samples["mapper"][-1][1] if samples["mapper"] else {}
        latest_adapter = samples["adapter"][-1][1] if samples["adapter"] else {}
        ratios = {
            "mapper_lidar_over_driver_cloud": _ratio(mapper_cloud, bridge_cloud),
            "mapper_imu_over_driver_imu": _ratio(mapper_imu, bridge_imu),
            "processed_over_mapper_lidar": _ratio(processed, mapper_cloud),
        }
        missing = [name for name, items in samples.items() if len(items) < 2]
        classification = "healthy"
        evidence = []
        if algorithm_running and missing:
            classification = "insufficient_evidence"
            evidence.append("missing two samples from: " + ",".join(missing))
        elif (
            bridge_cloud is not None
            and bridge_cloud < 9.5
            or (rates["bridge"]["cloud_dropped"]["delta"] or 0) > 0
        ):
            classification = "driver_source_drop"
            evidence.append("Driver LiDAR publish rate/drop counter is abnormal")
        elif any(
            ratio is not None and ratio < 0.99
            for ratio in (
                ratios["mapper_lidar_over_driver_cloud"],
                ratios["mapper_imu_over_driver_imu"],
            )
        ):
            classification = "dds_or_subscriber_drop"
            evidence.append("FAST-LIVO2 callback rate is below Driver publish rate")
        elif (
            ratios["processed_over_mapper_lidar"] is not None
            and ratios["processed_over_mapper_lidar"] < 0.9
            or _number(latest_mapper.get("lidar_buffer_span_sec")) > 0.2
        ):
            classification = "fast_livo_processing_backlog"
            evidence.append("FAST-LIVO2 consumes fewer scans than its callbacks receive")
        elif _number(latest_mapper.get("consecutive_zero_effective_features")) >= 2:
            classification = "scan_match_degraded"
            evidence.append("FAST-LIVO2 has consecutive scans without effective features")
        elif (
            (rates["adapter"]["mapping_work_dropped"]["delta"] or 0) > 0
            or _number(
                (latest_adapter.get("latency_max_ms") or {}).get("cloud_end_to_end")
            )
            > 200.0
        ):
            classification = "adapter_backlog"
            evidence.append("adapter latest-only work or cloud latency is overloaded")

        return {
            "schema": "phanthy.navigation.pipeline_diagnostics.v1",
            "window_sec": _WINDOW_SEC,
            "classification": classification,
            "evidence": evidence,
            "layers": {
                name: {
                    "sample_count": len(items),
                    "receive_age_sec": (
                        None if not items else round(max(0.0, now - items[-1][0]), 3)
                    ),
                    "rates": rates[name],
                    "latest": None if not items else dict(items[-1][1]),
                }
                for name, items in samples.items()
            },
            "ratios": ratios,
            "queue_policy": (
                "DDS KEEP_LAST evicts old samples automatically; no global topic "
                "clear is performed. FAST-LIVO2 internal buffer spans are reported."
            ),
        }
