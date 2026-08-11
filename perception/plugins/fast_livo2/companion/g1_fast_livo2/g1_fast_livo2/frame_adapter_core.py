"""Pure coordinate and Canvas-map helpers for FAST-LIVO2 outputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Iterable, Sequence


FLOAT32 = 7
FLOAT64 = 8
CANVAS_MAPPING_MAX_POINTS = 80_000
_FRAME_HEADER = struct.Struct("<fffBI")
_POINT = struct.Struct("<fff")
_FULL_MAP_WITH_Z = 0x01 | 0x02


class InvalidFastLivo2Frame(ValueError):
    pass


@dataclass(frozen=True)
class Quaternion:
    x: float
    y: float
    z: float
    w: float


@dataclass(frozen=True)
class Pose3:
    x: float
    y: float
    z: float
    q: Quaternion


def normalize_quaternion(q: Quaternion) -> Quaternion:
    values = (q.x, q.y, q.z, q.w)
    if not all(math.isfinite(value) for value in values):
        raise InvalidFastLivo2Frame("quaternion must be finite")
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-9:
        raise InvalidFastLivo2Frame("quaternion norm is zero")
    return Quaternion(*(value / norm for value in values))


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    if not all(math.isfinite(value) for value in (roll, pitch, yaw)):
        raise InvalidFastLivo2Frame("RPY must be finite")
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return normalize_quaternion(
        Quaternion(
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    )


def quaternion_multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    left = normalize_quaternion(left)
    right = normalize_quaternion(right)
    return normalize_quaternion(
        Quaternion(
            left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
            left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
            left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
            left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z,
        )
    )


def quaternion_inverse(q: Quaternion) -> Quaternion:
    q = normalize_quaternion(q)
    return Quaternion(-q.x, -q.y, -q.z, q.w)


def rotate_vector(q: Quaternion, vector: Sequence[float]) -> tuple[float, float, float]:
    if len(vector) != 3 or not all(math.isfinite(float(value)) for value in vector):
        raise InvalidFastLivo2Frame("vector must contain three finite values")
    q = normalize_quaternion(q)
    x, y, z = (float(value) for value in vector)
    ix = q.w * x + q.y * z - q.z * y
    iy = q.w * y + q.z * x - q.x * z
    iz = q.w * z + q.x * y - q.y * x
    iw = -q.x * x - q.y * y - q.z * z
    return (
        ix * q.w + iw * -q.x + iy * -q.z - iz * -q.y,
        iy * q.w + iw * -q.y + iz * -q.x - ix * -q.z,
        iz * q.w + iw * -q.z + ix * -q.y - iy * -q.x,
    )


def inverse_pose(pose: Pose3) -> Pose3:
    q_inverse = quaternion_inverse(pose.q)
    tx, ty, tz = rotate_vector(q_inverse, (-pose.x, -pose.y, -pose.z))
    return Pose3(tx, ty, tz, q_inverse)


def compose_pose(left: Pose3, right: Pose3) -> Pose3:
    if not all(math.isfinite(value) for value in (left.x, left.y, left.z, right.x, right.y, right.z)):
        raise InvalidFastLivo2Frame("pose translation must be finite")
    rx, ry, rz = rotate_vector(left.q, (right.x, right.y, right.z))
    return Pose3(
        left.x + rx,
        left.y + ry,
        left.z + rz,
        quaternion_multiply(left.q, right.q),
    )


def canonical_base_pose(map_to_sensor: Pose3, base_to_sensor: Pose3) -> Pose3:
    """Return map->base from FAST-LIVO2 map->sensor and measured base->sensor."""

    return compose_pose(map_to_sensor, inverse_pose(base_to_sensor))


def yaw_from_quaternion(q: Quaternion) -> float:
    q = normalize_quaternion(q)
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _field_value(field, key: str):
    return field[key] if isinstance(field, dict) else getattr(field, key)


def iter_xyz_points(
    *,
    fields: Iterable,
    data: bytes,
    point_step: int,
    width: int,
    height: int,
    is_bigendian: bool,
) -> Iterable[tuple[float, float, float]]:
    if is_bigendian:
        raise InvalidFastLivo2Frame("big-endian PointCloud2 is unsupported")
    if point_step <= 0 or width < 0 or height < 0:
        raise InvalidFastLivo2Frame("invalid PointCloud2 dimensions")
    layout = {}
    for field in fields:
        name = str(_field_value(field, "name"))
        if name in {"x", "y", "z"}:
            datatype = int(_field_value(field, "datatype"))
            if int(_field_value(field, "count")) != 1 or datatype not in {FLOAT32, FLOAT64}:
                raise InvalidFastLivo2Frame(f"{name} must be scalar float32/float64")
            layout[name] = (int(_field_value(field, "offset")), datatype)
    if set(layout) != {"x", "y", "z"}:
        raise InvalidFastLivo2Frame("PointCloud2 requires x/y/z fields")
    point_count = width * height
    if len(data) < point_count * point_step:
        raise InvalidFastLivo2Frame("PointCloud2 data is truncated")
    for index in range(point_count):
        base = index * point_step
        values = []
        for name in ("x", "y", "z"):
            offset, datatype = layout[name]
            fmt = "<f" if datatype == FLOAT32 else "<d"
            size = 4 if datatype == FLOAT32 else 8
            if offset < 0 or offset + size > point_step:
                raise InvalidFastLivo2Frame(f"{name} field exceeds point_step")
            values.append(struct.unpack_from(fmt, data, base + offset)[0])
        if all(math.isfinite(value) for value in values):
            yield values[0], values[1], values[2]


class VoxelMap:
    def __init__(self, voxel_size_m: float, max_points: int = CANVAS_MAPPING_MAX_POINTS):
        if not math.isfinite(voxel_size_m) or voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be finite and positive")
        if not 1 <= max_points <= CANVAS_MAPPING_MAX_POINTS:
            raise ValueError("max_points is out of range")
        self._voxel = voxel_size_m
        self._max_points = max_points
        self._points: dict[tuple[int, int, int], tuple[float, float, float]] = {}

    def clear(self) -> None:
        self._points.clear()

    def add(self, points: Iterable[tuple[float, float, float]]) -> None:
        for point in points:
            if len(self._points) >= self._max_points:
                break
            x, y, z = point
            if not all(math.isfinite(value) for value in point):
                continue
            key = (
                math.floor(x / self._voxel),
                math.floor(y / self._voxel),
                math.floor(z / self._voxel),
            )
            self._points.setdefault(key, (x, y, z))

    @property
    def point_count(self) -> int:
        return len(self._points)

    def project_xy(
        self,
        *,
        min_z: float,
        max_z: float,
        output_z: float = 0.0,
    ) -> tuple[tuple[float, float, float], ...]:
        """Project the non-floor/non-ceiling slice into one point per XY voxel."""

        if not all(math.isfinite(value) for value in (min_z, max_z, output_z)):
            raise ValueError("projection heights must be finite")
        if min_z >= max_z:
            raise ValueError("min_z must be less than max_z")

        projected: dict[tuple[int, int], tuple[float, float, float]] = {}
        for (ix, iy, _iz), point in self._points.items():
            if min_z <= point[2] <= max_z:
                projected.setdefault(
                    (ix, iy),
                    ((ix + 0.5) * self._voxel, (iy + 0.5) * self._voxel, output_z),
                )
        return tuple(projected.values())

    def encode(self, robot_pose: Pose3) -> bytes:
        yaw = yaw_from_quaternion(robot_pose.q)
        body = bytearray(len(self._points) * _POINT.size)
        for index, point in enumerate(self._points.values()):
            _POINT.pack_into(body, index * _POINT.size, *point)
        return _FRAME_HEADER.pack(
            robot_pose.x,
            robot_pose.y,
            yaw,
            _FULL_MAP_WITH_Z,
            len(self._points),
        ) + body


__all__ = [
    "InvalidFastLivo2Frame",
    "Pose3",
    "Quaternion",
    "VoxelMap",
    "canonical_base_pose",
    "compose_pose",
    "iter_xyz_points",
    "quaternion_from_rpy",
    "yaw_from_quaternion",
]
