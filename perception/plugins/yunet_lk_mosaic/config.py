"""Shared validation and MCP configuration metadata."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace


class MosaicError(ValueError):
    """Invalid configuration, input frame or processing result."""


@dataclass(frozen=True)
class MosaicConfig:
    target_fps: int = 30
    publish_delay_ms: int = 0
    miss_compensation: bool = True
    adaptive_detection: bool = False
    detect_every_frames: int = 4
    confidence_threshold: float = 0.28
    stable_expansion: float = 0.35
    unstable_expansion: float = 0.50
    max_buffer_frames: int = 20
    output_width: int = 640
    max_faces: int = 6
    jpeg_quality: int = 72
    pixelate_block: int = 24
    consent_acknowledged: bool = False

    def updated(self, values: dict) -> MosaicConfig:
        if not isinstance(values, dict):
            raise MosaicError("config_must_be_object")
        for name, value in values.items():
            if name not in FIELDS:
                raise MosaicError(f"unknown_config_field:{name}")
            field = FIELDS[name]
            if field["type"] == "boolean":
                valid = type(value) is bool
            elif field["type"] == "integer" and "enum" in field:
                valid = type(value) is int and value in field["enum"]
            else:
                types = (int, float) if field["type"] == "number" else (int,)
                valid = type(value) in types and field["minimum"] <= value <= field["maximum"]
                if valid:
                    valid = math.isfinite(value)
            if not valid:
                raise MosaicError(f"invalid_config:{name}")
        cfg = replace(self, **values)
        if cfg.unstable_expansion < cfg.stable_expansion:
            raise MosaicError("unstable_expansion_must_be_at_least_stable_expansion")
        return cfg


def _range(kind, title, low, high):
    return {"type": kind, "title": title, "minimum": low, "maximum": high}


FIELDS = {
    "consent_acknowledged": {"type": "boolean", "title": "已征得出镜人员同意"},
    "target_fps": {"type": "integer", "title": "目标处理 FPS", "enum": [15, 20, 24, 30]},
    "publish_delay_ms": {"type": "integer", "title": "发布延迟（毫秒）",
                         "enum": [0, 200, 300, 500]},
    "miss_compensation": {"type": "boolean", "title": "漏检补偿与历史帧纠错"},
    "adaptive_detection": {"type": "boolean", "title": "自适应 YuNet 检测"},
    "detect_every_frames": _range("integer", "固定检测间隔（帧）", 1, 5),
    "confidence_threshold": _range("number", "YuNet 检测阈值", 0.1, 0.9),
    "stable_expansion": _range("number", "正常时每边扩框比例", 0.0, 1.0),
    "unstable_expansion": _range("number", "不稳定时每边扩框比例", 0.0, 1.0),
    "max_buffer_frames": _range("integer", "待发布缓存帧数上限", 1, 32),
    "output_width": _range("integer", "输出宽度（像素）", 320, 960),
    "max_faces": _range("integer", "每帧最多处理人脸", 1, 8),
    "jpeg_quality": _range("integer", "JPEG 质量", 35, 95),
    "pixelate_block": _range("integer", "马赛克块大小（像素）", 8, 64),
}

_DEFAULTS = MosaicConfig()
CONFIG_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        name: {**field, "default": getattr(_DEFAULTS, name), "scope": "instance"}
        for name, field in FIELDS.items()
    },
}
