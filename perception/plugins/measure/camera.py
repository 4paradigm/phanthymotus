"""The consumer side of `motus.camera/1` for the measurement card.

The camera owns its geometry. A measurement card must not hard-code a field of
view just because one robot's lens happens to be wide: the same code runs on a
D435i, an Orbbec, and a phone camera. The declaration gives us `K` when it is
available and a half field of view when it is not. Missing geometry is reported
as missing, not replaced by a plausible number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

SCHEMA = "motus.camera/1"


@dataclass
class CameraModel:
    width: int | None = None
    height: int | None = None
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    half_fov_h: float | None = None
    half_fov_v: float | None = None
    source: str = "missing"
    notes: list[str] = field(default_factory=list)

    @property
    def has_width(self) -> bool:
        return bool(self.fx and self.fx > 0 and self.width and self.width > 0)

    @property
    def has_height(self) -> bool:
        return bool(self.fy and self.fy > 0 and self.height and self.height > 0)

    def as_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fx": round(self.fx, 4) if self.fx else None,
            "fy": round(self.fy, 4) if self.fy else None,
            "half_fov_h_rad": round(self.half_fov_h, 5) if self.half_fov_h else None,
            "half_fov_v_rad": round(self.half_fov_v, 5) if self.half_fov_v else None,
            "source": self.source,
        }


def for_topic(declarations, topic: str) -> dict:
    """The declaration covering `topic`, or `{}`.

    agent-core passes declarations keyed by topic or as a list of entries. The
    topic is the stable join key; the upstream card's name is not, because a
    consumer may be wired to two different cameras and the cards can be renamed.
    """
    if isinstance(declarations, dict):
        entry = declarations.get(topic)
        return entry if isinstance(entry, dict) else {}
    for entry in declarations or ():
        if isinstance(entry, dict) and entry.get("topic") == topic:
            return entry
    return {}


def camera_id(declaration: dict) -> str:
    if not isinstance(declaration, dict):
        return ""
    return str(declaration.get("id") or "")


def from_declaration(declaration: dict, *,
                     fallback_width: int | None = None,
                     fallback_height: int | None = None,
                     fallback_half_fov: float | None = None,
                     fallback_half_fov_v: float | None = None) -> CameraModel:
    """Read a camera declaration, falling back only where the fallback is honest.

    `K` wins over an angle. A calibration matrix is solved from observations; an
    angle beside it is often a tape measure and trigonometry. If `K` is absent,
    a declared angle plus the frame size is enough to derive `fx`/`fy`; if both
    are absent, the model says so and the card reports distance only.
    """
    notes: list[str] = []
    if not isinstance(declaration, dict) or not declaration:
        declaration = {}
        notes.append("没有收到 motus.camera/1；宽度/高度无法可靠换算")
    elif declaration.get("schema") != SCHEMA:
        notes.append(f"相机声明 schema={declaration.get('schema')!r}，不是 {SCHEMA!r}")
        declaration = {}
    else:
        width = _positive_int(declaration.get("width"))
        height = _positive_int(declaration.get("height"))
        if width and height:
            fallback_width, fallback_height = width, height
        else:
            notes.append("相机声明没有有效的 width/height")

    width = _positive_int(declaration.get("width")) or _positive_int(fallback_width)
    height = _positive_int(declaration.get("height")) or _positive_int(fallback_height)

    model = CameraModel(width=width, height=height, notes=notes)

    K = declaration.get("K")
    if isinstance(K, (list, tuple)) and len(K) == 9:
        values = _finite(K)
        if values is not None and values[0] > 0 and values[4] > 0:
            model.fx, model.fy = values[0], values[4]
            model.cx, model.cy = values[2], values[5]
            model.source = "derived-from-K"
            if model.width and model.height:
                model.half_fov_h = math.atan((model.width / 2.0) / model.fx)
                model.half_fov_v = math.atan((model.height / 2.0) / model.fy)
            return model
        notes.append("相机声明的 K 不可用")

    angle_h = _angle(declaration.get("half_fov_rad"), "half_fov_rad", notes)
    angle_v = _angle(declaration.get("half_fov_v_rad"), "half_fov_v_rad", notes)
    if angle_h is None and fallback_half_fov is not None:
        angle_h = _angle(fallback_half_fov, "config half_fov_rad", notes,
                         prefix="使用卡片配置兜底：")
        model.source = "config-angle" if angle_h else "missing"
    if angle_v is None and fallback_half_fov_v is not None:
        angle_v = _angle(fallback_half_fov_v, "config half_fov_v_rad", notes,
                         prefix="使用卡片配置兜底：")
        if model.source == "missing":
            model.source = "config-angle"

    model.half_fov_h, model.half_fov_v = angle_h, angle_v
    if angle_h:
        if model.source == "missing":
            model.source = "declared-angle"
        if model.width:
            model.fx = (model.width / 2.0) / math.tan(angle_h)
    if angle_v:
        if model.source == "missing":
            model.source = "declared-angle"
        if model.height:
            model.fy = (model.height / 2.0) / math.tan(angle_v)
    if angle_h is None:
        notes.append("没有水平视场角或 fx；只能输出距离，不能输出宽度")
    if angle_v is None:
        notes.append("没有垂直视场角或 fy；不能输出高度")
    if not width or not height:
        notes.append("没有相机分辨率；无法把归一化 bbox 与深度像素对齐")
    return model


def _positive_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _finite(values) -> list[float] | None:
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(v) for v in out):
        return None
    return out


def _angle(value, name: str, notes: list[str], *, prefix: str = "") -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        notes.append(f"{prefix}{name} 不是数字")
        return None
    try:
        angle = float(value)
    except (TypeError, ValueError):
        notes.append(f"{prefix}{name} 不是数字")
        return None
    if not math.isfinite(angle) or not 0.0 < angle < math.pi / 2:
        notes.append(f"{prefix}{name}={value!r} 不在 (0, π/2) 内")
        return None
    return angle
