"""Summarize a short FAST-LIVO2 fault rosbag without retaining point payloads."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from .pipeline_diagnostics import PipelineDiagnostics


BRIDGE_STATUS_TOPIC = "/ubuntu/navigation/_bridge_status"
MAPPER_RUNTIME_TOPIC = "/ubuntu/navigation/fast_livo2/mapper_runtime"
ADAPTER_DIAGNOSTICS_TOPIC = "/ubuntu/navigation/fast_livo2/diagnostics"
SENSOR_REJECTION_TOPIC = "/ubuntu/navigation/fast_livo2/sensor_rejection"


def _percentile(values: list[float], fraction: float):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[round((len(ordered) - 1) * fraction)], 3)


def _timing(stamps_ns: list[int]) -> dict:
    gaps_ms = [
        max(0.0, (right - left) / 1_000_000.0)
        for left, right in zip(stamps_ns, stamps_ns[1:])
    ]
    duration_sec = (
        0.0
        if len(stamps_ns) < 2
        else max(0.0, (stamps_ns[-1] - stamps_ns[0]) / 1_000_000_000.0)
    )
    return {
        "count": len(stamps_ns),
        "duration_sec": round(duration_sec, 3),
        "rate_hz": (
            None
            if duration_sec <= 0.0
            else round((len(stamps_ns) - 1) / duration_sec, 3)
        ),
        "gap_p95_ms": _percentile(gaps_ms, 0.95),
        "gap_max_ms": None if not gaps_ms else round(max(gaps_ms), 3),
    }


def build_fault_summary(records, trigger=None) -> dict:
    receive_stamps = defaultdict(list)
    source_stamps = defaultdict(list)
    source_delays = defaultdict(list)
    pipeline = PipelineDiagnostics()
    rejected = defaultdict(int)
    last_receive_sec = 0.0
    for record in records:
        topic = str(record["topic"])
        receive_ns = int(record["receive_ns"])
        receive_stamps[topic].append(receive_ns)
        last_receive_sec = max(last_receive_sec, receive_ns / 1_000_000_000.0)
        source_ns = record.get("source_stamp_ns")
        if isinstance(source_ns, int) and source_ns > 0:
            source_stamps[topic].append(source_ns)
            source_delays[topic].append((receive_ns - source_ns) / 1_000_000.0)
        payload = record.get("json")
        if not isinstance(payload, dict):
            continue
        if topic == BRIDGE_STATUS_TOPIC:
            pipeline.observe("bridge", payload, receive_ns / 1_000_000_000.0)
        elif topic == MAPPER_RUNTIME_TOPIC:
            pipeline.observe("mapper", payload, receive_ns / 1_000_000_000.0)
        elif topic == ADAPTER_DIAGNOSTICS_TOPIC:
            pipeline.observe("adapter", payload, receive_ns / 1_000_000_000.0)
        elif topic == SENSOR_REJECTION_TOPIC:
            rejected[str(payload.get("error_code") or "unknown")] += 1

    topics = {}
    for topic, stamps in receive_stamps.items():
        source = source_stamps[topic]
        topics[topic] = {
            "receive": _timing(stamps),
            "source": _timing(source),
            "receive_minus_source_ms": {
                "p50": _percentile(source_delays[topic], 0.50),
                "p95": _percentile(source_delays[topic], 0.95),
                "max": (
                    None
                    if not source_delays[topic]
                    else round(max(source_delays[topic]), 3)
                ),
            },
        }
    result = pipeline.snapshot(
        algorithm_running=True,
        now_monotonic=last_receive_sec,
    )
    return {
        "schema": "phanthy.navigation.fast_livo2_fault_summary.v1",
        "trigger": dict(trigger or {}),
        "pipeline": result,
        "topics": topics,
        "rejections": dict(sorted(rejected.items())),
        "source_drop_proven": result["classification"] == "driver_source_drop",
    }


def summarize_fault_capture(directory: Path, trigger=None) -> dict:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError(f"fault_summary_dependency_unavailable:{exc}") from exc

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(directory), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {
        item.name: get_message(item.type)
        for item in reader.get_all_topics_and_types()
    }

    def records():
        while reader.has_next():
            topic, serialized, receive_ns = reader.read_next()
            message = deserialize_message(serialized, types[topic])
            header = getattr(message, "header", None)
            stamp = getattr(header, "stamp", None)
            source_ns = None
            if stamp is not None:
                source_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            payload = None
            if topic in {
                BRIDGE_STATUS_TOPIC,
                MAPPER_RUNTIME_TOPIC,
                ADAPTER_DIAGNOSTICS_TOPIC,
                SENSOR_REJECTION_TOPIC,
            }:
                try:
                    payload = json.loads(message.data)
                except (AttributeError, TypeError, ValueError):
                    pass
            yield {
                "topic": topic,
                "receive_ns": int(receive_ns),
                "source_stamp_ns": source_ns,
                "json": payload,
            }

    return build_fault_summary(records(), trigger)


def write_fault_summary(directory: Path, trigger=None) -> Path:
    target = Path(directory) / "diagnostic_summary.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            summarize_fault_capture(Path(directory), trigger),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target
