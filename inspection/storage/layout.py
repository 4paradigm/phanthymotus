from __future__ import annotations

import hashlib
import re
import socket
from datetime import datetime, timezone
from pathlib import Path


_DEVICE_SERIAL_PATHS = (
    Path("/proc/device-tree/serial-number"),
    Path("/sys/firmware/devicetree/base/serial-number"),
    Path("/sys/class/dmi/id/product_uuid"),
)

_CARD_STORAGE_SLUGS = {
    "audioinspector": "audio-inspector",
    "videoinspector": "video-inspector",
}

_TOPIC_BOILERPLATE = {
    "robot",
    "robots",
    "phanthymotus",
    "phanthy-motus",
    "driver",
    "g1-driver",
    "phanthymotus-g1-driver",
}


def safe_component(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9.-]+", "-", str(value)).strip("-.").lower()
    if not cleaned:
        raise ValueError("path component is empty after sanitization")
    return cleaned


def _read_identity(path: Path) -> str:
    try:
        return path.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="ignore").strip()
    except OSError:
        return ""


def detect_device_id(*, serial_paths: tuple[Path, ...] = _DEVICE_SERIAL_PATHS) -> str:
    """Return a stable device identity without exposing it as card configuration."""
    for path in serial_paths:
        identity = _read_identity(path)
        if not identity:
            continue
        prefix = "jetson" if "device-tree" in str(path) else "device"
        return f"{prefix}-{safe_component(identity)}"

    machine_id = _read_identity(Path("/etc/machine-id"))
    if machine_id:
        return f"device-{safe_component(machine_id)[:16]}"

    hostname = safe_component(socket.gethostname())
    digest = hashlib.sha256(hostname.encode("utf-8")).hexdigest()[:12]
    return f"device-{hostname}-{digest}"


def card_storage_slug(card_id: str) -> str:
    return _CARD_STORAGE_SLUGS.get(card_id, safe_component(card_id))


def _is_dynamic_topic_component(component: str) -> bool:
    return bool(re.fullmatch(r"(?:card|mcp|instance)-[a-z0-9]{6,}", component))


def source_storage_slug(input_topic: str) -> str:
    components: list[str] = []
    for raw in str(input_topic).split("/"):
        if not raw:
            continue
        component = safe_component(raw)
        if component in _TOPIC_BOILERPLATE or component.endswith("-driver"):
            continue
        if _is_dynamic_topic_component(component):
            continue
        components.append(component)
    return "-".join(components[-3:]) if components else "unresolved-source"


def instance_storage_slug(instance_id: str, input_topic: str) -> str:
    digest = hashlib.sha256(str(instance_id).encode("utf-8")).hexdigest()[:8]
    return f"{source_storage_slug(input_topic)}--{digest}"


def utc_hour_partition(wall_clock_ns: int) -> str:
    utc = datetime.fromtimestamp(wall_clock_ns / 1_000_000_000, tz=timezone.utc)
    return f"utc-hour={utc.strftime('%Y-%m-%dT%HZ')}"


def segment_basename(wall_clock_ns: int, sequence: int, extension: str) -> str:
    seconds, nanoseconds = divmod(int(wall_clock_ns), 1_000_000_000)
    utc = datetime.fromtimestamp(seconds, tz=timezone.utc)
    suffix = str(extension).lstrip(".")
    return f"{utc.strftime('%Y%m%dT%H%M%S')}.{nanoseconds:09d}Z--{int(sequence):06d}.{suffix}"


def segment_start_ns_from_name(path: Path) -> int:
    name = path.name
    legacy = re.match(r"^(\d+)_\d+\.", name)
    if legacy:
        return int(legacy.group(1))
    readable = re.match(r"^(\d{8}T\d{6})\.(\d{9})Z--\d+\.", name)
    if not readable:
        raise ValueError(f"unsupported segment filename: {name}")
    seconds = int(datetime.strptime(readable.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).timestamp())
    return seconds * 1_000_000_000 + int(readable.group(2))
