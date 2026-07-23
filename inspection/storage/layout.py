from __future__ import annotations

import hashlib
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


_ROBOT_SERIAL_PATHS = (
    (Path("/run/phanthymotus/robot-sn"), "provisioned-robot-sn", "unitree", True),
    (Path("/etc/phanthy-motus/robot-sn"), "provisioned-robot-sn", "unitree", True),
    (Path("/proc/device-tree/serial-number"), "jetson-module-serial", "jetson", False),
    (Path("/sys/firmware/devicetree/base/serial-number"), "jetson-module-serial", "jetson", False),
    (Path("/sys/class/dmi/id/product_serial"), "system-product-serial", "host", False),
    (Path("/sys/class/dmi/id/product_uuid"), "system-product-uuid", "host", False),
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

_SHANGHAI_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")


@dataclass(frozen=True)
class HardwareIdentity:
    value: str
    source: str
    manufacturer_serial: bool


@dataclass(frozen=True)
class _USBDevice:
    path: Path
    port: str
    vendor_id: str
    product_id: str
    manufacturer: str
    product: str
    serial: str

    @property
    def searchable(self) -> str:
        return " ".join((
            self.manufacturer,
            self.product,
            self.vendor_id,
            self.product_id,
        )).lower()


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


def detect_robot_identity(
    *,
    serial_paths: Iterable[tuple[Path, str, str, bool]] = _ROBOT_SERIAL_PATHS,
    machine_id_path: Path = Path("/etc/machine-id"),
    hostname: str | None = None,
) -> HardwareIdentity:
    """Return a stable robot identity and disclose whether it is a real robot SN."""
    for path, source, prefix, manufacturer_serial in serial_paths:
        identity = _read_identity(path)
        if not identity:
            continue
        return HardwareIdentity(
            value=f"{safe_component(prefix)}-{safe_component(identity)}",
            source=source,
            manufacturer_serial=manufacturer_serial,
        )

    machine_id = _read_identity(machine_id_path)
    if machine_id:
        return HardwareIdentity(
            value=f"host-{safe_component(machine_id)[:16]}",
            source="machine-id-fallback",
            manufacturer_serial=False,
        )

    resolved_hostname = safe_component(hostname or socket.gethostname())
    digest = hashlib.sha256(resolved_hostname.encode("utf-8")).hexdigest()[:12]
    return HardwareIdentity(
        value=f"host-{resolved_hostname}-{digest}",
        source="hostname-hash-fallback",
        manufacturer_serial=False,
    )


def detect_device_id(
    *,
    serial_paths: tuple[Path, ...] | None = None,
) -> str:
    """Backward-compatible alias for the automatically detected robot host ID."""
    if serial_paths is None:
        return detect_robot_identity().value
    annotated = tuple(
        (
            path,
            "jetson-module-serial" if "device-tree" in str(path) else "system-product-serial",
            "jetson" if "device-tree" in str(path) else "host",
            False,
        )
        for path in serial_paths
    )
    return detect_robot_identity(serial_paths=annotated).value


def _read_usb_devices(usb_root: Path) -> list[_USBDevice]:
    devices: list[_USBDevice] = []
    try:
        children = sorted(usb_root.iterdir())
    except OSError:
        return devices
    for child in children:
        vendor_id = _read_identity(child / "idVendor")
        product_id = _read_identity(child / "idProduct")
        if not vendor_id or not product_id:
            continue
        devices.append(_USBDevice(
            path=child,
            port=child.name,
            vendor_id=vendor_id,
            product_id=product_id,
            manufacturer=_read_identity(child / "manufacturer"),
            product=_read_identity(child / "product"),
            serial=_read_identity(child / "serial"),
        ))
    return devices


def _identity_brand(device: _USBDevice) -> str:
    searchable = device.searchable
    for marker, brand in (
        ("insta360", "insta360"),
        ("realsense", "realsense"),
        ("dji", "dji"),
        ("unitree", "unitree"),
    ):
        if marker in searchable:
            return brand
    if device.manufacturer:
        return safe_component(device.manufacturer).split("-")[0]
    return f"usb-{safe_component(device.vendor_id)}{safe_component(device.product_id)}"


def _usb_candidates(input_topic: str, kind: str, devices: list[_USBDevice]) -> list[_USBDevice]:
    topic = str(input_topic).lower()
    if "/ext_mic/" in topic:
        markers = ("mic", "audio", "wireless", "dji")
        return [device for device in devices if any(marker in device.searchable for marker in markers)]
    if "/ext_camera/" in topic:
        markers = ("camera", "insta360", "webcam", "uvc")
        return [
            device for device in devices
            if any(marker in device.searchable for marker in markers)
            and "realsense" not in device.searchable
        ]
    if kind == "video" and any(marker in topic for marker in ("camera", "rgb", "depth")):
        preferred = [device for device in devices if "realsense" in device.searchable]
        if preferred:
            return preferred
        markers = ("camera", "insta360", "webcam", "uvc")
        return [device for device in devices if any(marker in device.searchable for marker in markers)]
    return []


def detect_source_identity(
    input_topic: str,
    kind: str,
    *,
    robot_identity: HardwareIdentity | None = None,
    usb_root: Path = Path("/sys/bus/usb/devices"),
) -> HardwareIdentity:
    """Resolve the physical input device without exposing an editable card field."""
    robot = robot_identity or detect_robot_identity()
    topic = str(input_topic).lower()
    if kind == "audio" and "/mic/audio" in topic and "/ext_mic/" not in topic:
        return HardwareIdentity(
            value=f"{robot.value}-builtin-mic-array",
            source="robot-builtin-composite",
            manufacturer_serial=False,
        )

    candidates = _usb_candidates(input_topic, kind, _read_usb_devices(usb_root))
    if len(candidates) == 1:
        device = candidates[0]
        brand = _identity_brand(device)
        if device.serial:
            return HardwareIdentity(
                value=f"{brand}-{safe_component(device.serial)}",
                source="usb-manufacturer-serial",
                manufacturer_serial=True,
            )
        return HardwareIdentity(
            value=(
                f"{brand}-{safe_component(device.vendor_id)}{safe_component(device.product_id)}"
                f"-port-{safe_component(device.port)}"
            ),
            source="usb-topology-composite",
            manufacturer_serial=False,
        )

    semantic = source_storage_slug(input_topic)
    digest = hashlib.sha256(str(input_topic).encode("utf-8")).hexdigest()[:8]
    return HardwareIdentity(
        value=f"{robot.value}-{semantic}-{digest}",
        source="topic-composite-fallback",
        manufacturer_serial=False,
    )


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


def local_date_partition(wall_clock_ns: int) -> str:
    local = datetime.fromtimestamp(wall_clock_ns / 1_000_000_000, tz=_SHANGHAI_TIMEZONE)
    return local.strftime("%Y-%m-%d")


def utc_hour_partition(wall_clock_ns: int) -> str:
    """Legacy v1 partition retained for recovery and upload compatibility."""
    utc = datetime.fromtimestamp(wall_clock_ns / 1_000_000_000, tz=timezone.utc)
    return f"utc-hour={utc.strftime('%Y-%m-%dT%HZ')}"


def storage_relative_directory(
    robot_identity: HardwareIdentity,
    source_identity: HardwareIdentity,
    kind: str,
    wall_clock_ns: int,
) -> Path:
    modality = safe_component(kind)
    return Path(
        safe_component(robot_identity.value),
        modality,
        safe_component(source_identity.value),
        local_date_partition(wall_clock_ns),
    )


def segment_basename(wall_clock_ns: int, sequence: int, extension: str) -> str:
    seconds, nanoseconds = divmod(int(wall_clock_ns), 1_000_000_000)
    local = datetime.fromtimestamp(seconds, tz=_SHANGHAI_TIMEZONE)
    suffix = str(extension).lstrip(".")
    return f"{local.strftime('%Y%m%dT%H%M%S')}.{nanoseconds:09d}+0800--{int(sequence):06d}.{suffix}"


def segment_start_ns_from_name(path: Path) -> int:
    name = path.name
    legacy = re.match(r"^(\d+)_\d+\.", name)
    if legacy:
        return int(legacy.group(1))
    readable = re.match(r"^(\d{8}T\d{6})\.(\d{9})(Z|[+-]\d{4})--\d+\.", name)
    if not readable:
        raise ValueError(f"unsupported segment filename: {name}")
    tz_text = readable.group(3)
    parsed_timezone = timezone.utc
    if tz_text != "Z":
        sign = 1 if tz_text[0] == "+" else -1
        parsed_timezone = timezone(sign * timedelta(
            hours=int(tz_text[1:3]),
            minutes=int(tz_text[3:5]),
        ))
    seconds = int(
        datetime.strptime(readable.group(1), "%Y%m%dT%H%M%S")
        .replace(tzinfo=parsed_timezone)
        .timestamp()
    )
    return seconds * 1_000_000_000 + int(readable.group(2))
