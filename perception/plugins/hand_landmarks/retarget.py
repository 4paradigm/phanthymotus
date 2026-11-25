"""Turn 21 hand landmarks into five finger flexions, then into joint values.

Two deliberately separate steps:

1. ``flexions()`` maps geometry to normalized 0..1 finger curl. It never emits
   joint angles, because a curl is a description of a hand and not a command to
   any particular robot.
2. ``build_values()`` maps those curls onto the *downstream descriptor's* joint
   order and limits. That is the only place where units and ranges appear, and
   it reads them from the control card rather than hardcoding I20-lite angles.

This split is what keeps a change of hand model independent from a change of
dexterous hand. It also means a wrong joint map is rejected before any command
is published, instead of moving the wrong finger at 10 Hz.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

# MediaPipe / RTMPose hand keypoint order, shared with perception/hand_runtime.
FINGERS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FINGER_ORDER = tuple(FINGERS)


class JointMappingError(ValueError):
    """The descriptor does not give us a safe way to address five fingers."""


def _bend(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Normalized curl at ``b``: 0 when straight, 1 when fully folded.

    The geometric interior angle between ``a-b`` and ``c-b`` is pi for a
    straight chain and approaches 0 for a folded one, so the curl is
    ``(pi - angle) / pi`` -- the opposite convention from an angle reading.
    """
    u, v = a - b, c - b
    norm = float(np.linalg.norm(u) * np.linalg.norm(v))
    if norm < 1e-8:
        raise ValueError("degenerate hand keypoints")
    cosine = float(np.clip(np.dot(u, v) / norm, -1.0, 1.0))
    return (math.pi - math.acos(cosine)) / math.pi


def _extension_curl(points: np.ndarray, a: int, b: int, d: int, *,
                    open_ratio: float, fist_ratio: float) -> float:
    """Finger extension ratio -> 0 (open) .. 1 (fist).

    Point-to-point mapping in the survey: compare MCP->tip length to the
    MCP->PIP segment of the SAME finger. This is scale-invariant, ignores the
    wrist/palm centre, and is robust when the whole palm faces the camera --
    the case where fingertip-to-palm distance made an open hand look curled.
    """
    proximal = float(np.linalg.norm(points[b, :2] - points[a, :2]))
    full = float(np.linalg.norm(points[d, :2] - points[a, :2]))
    if proximal < 1e-6:
        raise ValueError("degenerate hand keypoints")
    ratio = full / proximal
    curl = (open_ratio - ratio) / max(open_ratio - fist_ratio, 1e-6)
    return float(np.clip(curl, 0.0, 1.0))


def _thumb_opposition_curl(points: np.ndarray, *,
                           open_reach: float = 2.2,
                           fist_reach: float = 1.0) -> float:
    """Thumb opposition (tip moving toward the little-finger MCP).

    The thumb does not just flex -- it sweeps across the palm. Its tip-to-
    pinky-MCP distance collapses during opposition (measured ~2.4 open to
    ~1.0 closed, normalized by palm width), a much stronger and more robust
    signal than the CMC->tip extension ratio when the thumb is partly hidden
    behind the palm in a side view.
    """
    scale = float(np.linalg.norm(points[9, :2] - points[0, :2]))
    if scale < 1e-6:
        raise ValueError("degenerate hand keypoints")
    reach = float(np.linalg.norm(points[4, :2] - points[17, :2])) / scale
    curl = (open_reach - reach) / max(open_reach - fist_reach, 1e-6)
    return float(np.clip(curl, 0.0, 1.0))


def flexions(landmarks, *, min_visibility: float = 0.0,
             open_ratio: float = 2.2, fist_ratio: float = 0.5,
             thumb_open_ratio: float = 1.8,
             thumb_fist_ratio: float = 0.75) -> dict:
    """Return ``{thumb, index, middle, ring, pinky}`` in 0..1 grasp curl.

    Per-finger extension (point-to-point mapping, Li et al. TCSVT 2022) is the
    primary cue; the proximal bend is a secondary cue for partial curls.
    """
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("expected 21 finite (x, y, z) landmarks")
    if min_visibility > 0 and points.shape[1] >= 3:
        if float(np.mean(points[:, 2])) < -min_visibility:
            raise ValueError("hand landmarks are not visible")

    result = {}
    for name, (a, b, c, d) in FINGERS.items():
        proximal_angle = _bend(points[a], points[b], points[c])
        distal_angle = _bend(points[b], points[c], points[d])
        angle_curl = 0.8 * proximal_angle + 0.2 * distal_angle
        # The thumb has only two phalanges and moves largely across the palm;
        # its MCP->tip / MCP->PIP ratio is much smaller than a finger's.
        # Using the finger thresholds made it look permanently half-open and
        # unresponsive. Separate, lower thresholds restore thumb travel.
        if name == "thumb":
            # Opposition dominates; extension ratio is a secondary cue.
            opp = _thumb_opposition_curl(points)
            ext = _extension_curl(points, a, b, d,
                                  open_ratio=thumb_open_ratio,
                                  fist_ratio=thumb_fist_ratio)
            result[name] = float(np.clip(0.7 * opp + 0.3 * ext,
                                         0.0, 1.0))
        else:
            # Static fists fold the fingertips behind the palm, so their
            # position is often mis-detected until the hand is shaken. The
            # PIP joint (b) stays visible outside the palm and is the reliable
            # signal: measured bend 0.845 at fist vs DIP's diluted 0.25.
            # Lead on PIP angle; keep fingertip reach as a secondary cue for
            # the position-oriented target. DIP is dropped entirely -- it is
            # coupled to PIP (theta_DIP ~= 2/3 theta_PIP) and only dilutes.
            ext = _extension_curl(points, a, b, d,
                                  open_ratio=open_ratio, fist_ratio=fist_ratio)
            result[name] = float(np.clip(
                0.6 * proximal_angle + 0.4 * ext, 0.0, 1.0))
    return result


class RateLimiter:
    """Slew-rate limit per finger to avoid a jump on a dropped frame."""

    def __init__(self, rate: float = 2.5):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = float(rate)
        self.previous: Optional[dict] = None

    def reset(self) -> None:
        self.previous = None

    def update(self, target: dict, dt: float) -> dict:
        step = self.rate * max(0.0, float(dt))
        if self.previous is None:
            self.previous = dict(target)
            return dict(self.previous)
        self.previous = {
            name: max(self.previous[name] - step,
                      min(self.previous[name] + step, float(target[name])))
            for name in FINGER_ORDER
        }
        return dict(self.previous)


def _find_index(key, names: list) -> int:
    """Resolve one finger key to a descriptor index, by name or by index."""
    if isinstance(key, bool):
        raise JointMappingError(f"joint_map[{key!r}] must be a name or an index")
    if isinstance(key, int) or (isinstance(key, str) and key.strip().lstrip("-").isdigit()):
        index = int(key)
        if index < 0 or index >= len(names):
            raise JointMappingError(f"joint_map[{key!r}] 超出 dof={len(names)}")
        return index
    text = str(key).strip().lower()
    exact = [i for i, name in enumerate(names) if str(name).strip().lower() == text]
    if len(exact) == 1:
        return exact[0]
    partial = [i for i, name in enumerate(names)
               if text in str(name).strip().lower()
               or str(name).strip().lower() in text]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise JointMappingError(
            f"joint_map[{key!r}] 在下游 joint_names 里找不到：{names}")
    raise JointMappingError(
        f"joint_map[{key!r}] 匹配到多个关节 {partial}，请写完整名字或索引")


def resolve_joint_indices(descriptor: dict, joint_map: dict | None,
                          *, allow_unmapped_neutral: bool = False) -> dict:
    """Map each finger to one index in the downstream descriptor.

    With no ``joint_map`` a 5-DoF hand is taken in FINGER_ORDER. Any other
    width is refused unless an explicit map is supplied: guessing which five
    of a 12-DoF hand are the fingers is how one card silently pinches the
    wrong actuator.
    """
    dof = int(descriptor.get("dof") or 0)
    names = list(descriptor.get("joint_names") or [])
    if dof <= 0:
        raise JointMappingError("下游 descriptor 没有有效的 dof")
    if names and len(names) != dof:
        raise JointMappingError(
            f"下游 joint_names 有 {len(names)} 项，dof 却是 {dof}")

    joint_map = joint_map or {}
    if not joint_map:
        if dof != len(FINGER_ORDER):
            raise JointMappingError(
                f"下游有 {dof} 个关节，无法自动判断哪五个是手指；"
                "请配置 joint_map（例如 thumb: thumb_curl）")
        return {name: i for i, name in enumerate(FINGER_ORDER)}

    unknown = [key for key in joint_map if key not in FINGERS]
    if unknown:
        raise JointMappingError(
            f"joint_map 里有未知手指 {unknown}，可用的是 {list(FINGER_ORDER)}")

    index_names = names or [str(i) for i in range(dof)]
    mapping = {}
    for finger in FINGER_ORDER:
        if finger not in joint_map:
            raise JointMappingError(f"joint_map 缺少 {finger}")
        mapping[finger] = _find_index(joint_map[finger], index_names)
    if len(set(mapping.values())) != len(mapping):
        raise JointMappingError(f"joint_map 把多根手指映射到了同一个关节：{mapping}")

    missing = [i for i in range(dof) if i not in mapping.values()]
    if missing and not allow_unmapped_neutral:
        labels = [index_names[i] for i in missing]
        raise JointMappingError(
            f"下游还有 {len(missing)} 个未映射关节 {labels}；"
            "请补全 joint_map，或显式设置 allow_unmapped_neutral=true"
            "（这些关节会被保持在中立位）")
    return mapping


def _limits(descriptor: dict, dof: int) -> tuple:
    limits = descriptor.get("limits") or {}
    lower = limits.get("lower")
    upper = limits.get("upper")
    if not isinstance(lower, list) or not isinstance(upper, list) \
            or len(lower) != dof or len(upper) != dof:
        raise JointMappingError(
            f"下游 descriptor 需要 {dof} 维 limits.lower / limits.upper，"
            "否则无法把 0..1 的弯曲量安全地换算成关节值")
    try:
        low = [float(v) for v in lower]
        high = [float(v) for v in upper]
    except (TypeError, ValueError) as error:
        raise JointMappingError("limits.lower / limits.upper 必须是数字") from error
    if any(not math.isfinite(v) for v in low + high) or any(h <= l for l, h in zip(low, high)):
        raise JointMappingError("每个关节的 limits.upper 必须大于 limits.lower")
    return low, high


def build_values(descriptor: dict, curl: dict, joint_map: dict | None,
                 *, allow_unmapped_neutral: bool = False,
                 neutral_values=None) -> list:
    """Return one full-width command vector for ``motus.control/1``.

    Unmapped joints are held at their limit midpoint (or an explicit
    ``neutral_values`` vector) only when the operator has opted in. A
    normalized curl is interpolated linearly between the descriptor's own
    lower and upper limits; no radians/degrees assumption is made here.
    """
    dof = int(descriptor.get("dof") or 0)
    low, high = _limits(descriptor, dof)
    mapping = resolve_joint_indices(
        descriptor, joint_map, allow_unmapped_neutral=allow_unmapped_neutral)

    if neutral_values is None:
        values = [(l + h) / 2.0 for l, h in zip(low, high)]
    else:
        raw = list(neutral_values)
        if len(raw) != dof:
            raise JointMappingError(f"neutral_values 需要 {dof} 项，收到 {len(raw)} 项")
        values = [float(v) for v in raw]
        if any(not math.isfinite(v) for v in values):
            raise JointMappingError("neutral_values 必须是有限数字")

    for finger, index in mapping.items():
        amount = float(np.clip(curl.get(finger, 0.0), 0.0, 1.0))
        values[index] = low[index] + amount * (high[index] - low[index])
    return values
