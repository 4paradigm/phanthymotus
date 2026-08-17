"""Pure coordinate and Canvas-map helpers for FAST-LIVO2 outputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import struct
from typing import Iterable, Sequence


FLOAT32 = 7
FLOAT64 = 8
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


@dataclass(frozen=True)
class RelocalizationResult:
    map_from_session: Pose3
    map_base_pose: Pose3
    match_ratio: float
    matched_points: int
    evaluated_points: int


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


def transform_points(
    pose: Pose3, points: Iterable[tuple[float, float, float]]
) -> Iterable[tuple[float, float, float]]:
    q = normalize_quaternion(pose.q)
    xx, yy, zz = q.x * q.x, q.y * q.y, q.z * q.z
    xy, xz, yz = q.x * q.y, q.x * q.z, q.y * q.z
    wx, wy, wz = q.w * q.x, q.w * q.y, q.w * q.z
    matrix = (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )
    for x, y, z in points:
        rx = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z
        ry = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z
        rz = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z
        yield pose.x + rx, pose.y + ry, pose.z + rz


def read_pcd_xyz(path: str | Path, *, max_points: int | None = None):
    """Read finite XYZ points from an ASCII or uncompressed binary PCD file."""

    if max_points is not None and max_points < 1:
        raise ValueError("max_points must be positive when provided")
    source = Path(path)
    try:
        stream = source.open("rb")
    except OSError as exc:
        raise InvalidFastLivo2Frame(f"cannot open PCD {source.name}: {exc}") from exc
    with stream:
        header: dict[str, list[str]] = {}
        while True:
            raw = stream.readline()
            if not raw:
                raise InvalidFastLivo2Frame("PCD header is missing DATA")
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise InvalidFastLivo2Frame("PCD header must be ASCII") from exc
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0].upper()
            header[key] = parts[1:]
            if key == "DATA":
                break

        fields = header.get("FIELDS") or header.get("FIELD")
        if not fields or not {"x", "y", "z"}.issubset(fields):
            raise InvalidFastLivo2Frame("PCD requires x/y/z fields")
        try:
            sizes = [int(value) for value in header["SIZE"]]
            types = [value.upper() for value in header["TYPE"]]
            counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
            fallback_points = int(header["WIDTH"][0]) * int(header.get("HEIGHT", ["1"])[0])
            declared_points = int((header.get("POINTS") or [str(fallback_points)])[0])
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise InvalidFastLivo2Frame("PCD field layout is invalid") from exc
        if not (len(fields) == len(sizes) == len(types) == len(counts)):
            raise InvalidFastLivo2Frame("PCD field layout lengths differ")
        if declared_points < 0 or any(size <= 0 for size in sizes) or any(count <= 0 for count in counts):
            raise InvalidFastLivo2Frame("PCD dimensions are invalid")

        token_offsets: dict[str, int] = {}
        byte_offsets: dict[str, int] = {}
        token_offset = 0
        byte_offset = 0
        for name, size, count in zip(fields, sizes, counts):
            token_offsets[name] = token_offset
            byte_offsets[name] = byte_offset
            token_offset += count
            byte_offset += size * count
        for name in ("x", "y", "z"):
            index = fields.index(name)
            if counts[index] != 1 or types[index] != "F" or sizes[index] not in {4, 8}:
                raise InvalidFastLivo2Frame(f"PCD {name} must be scalar float32/float64")

        mode = (header.get("DATA") or [""])[0].lower()
        points: list[tuple[float, float, float]] = []
        sample_stride = (
            1
            if max_points is None
            else max(1, math.ceil(max(1, declared_points) / max_points))
        )
        if mode == "ascii":
            source_index = 0
            for raw in stream:
                if max_points is not None and len(points) >= max_points:
                    break
                values = raw.split()
                if not values:
                    continue
                should_sample = source_index % sample_stride == 0
                source_index += 1
                if not should_sample:
                    continue
                try:
                    point = tuple(float(values[token_offsets[name]]) for name in ("x", "y", "z"))
                except (ValueError, IndexError) as exc:
                    raise InvalidFastLivo2Frame("PCD ASCII point is malformed") from exc
                if all(math.isfinite(value) for value in point):
                    points.append(point)
        elif mode == "binary":
            payload_offset = stream.tell()
            required = declared_points * byte_offset
            try:
                available = source.stat().st_size - payload_offset
            except OSError as exc:
                raise InvalidFastLivo2Frame(
                    f"cannot inspect PCD {source.name}: {exc}"
                ) from exc
            if available < required:
                raise InvalidFastLivo2Frame("PCD binary payload is truncated")
            for point_index in range(0, declared_points, sample_stride):
                if max_points is not None and len(points) >= max_points:
                    break
                stream.seek(payload_offset + point_index * byte_offset)
                record = stream.read(byte_offset)
                if len(record) != byte_offset:
                    raise InvalidFastLivo2Frame("PCD binary payload is truncated")
                values = []
                for name in ("x", "y", "z"):
                    field_index = fields.index(name)
                    fmt = "<f" if sizes[field_index] == 4 else "<d"
                    values.append(struct.unpack_from(fmt, record, byte_offsets[name])[0])
                point = tuple(values)
                if all(math.isfinite(value) for value in point):
                    points.append(point)
        else:
            raise InvalidFastLivo2Frame(
                f"PCD DATA {mode or '<missing>'} is unsupported; use ascii or binary"
            )
    if not points:
        raise InvalidFastLivo2Frame("PCD contains no finite XYZ points")
    return tuple(points)


def _search_values(radius: float, step: float) -> tuple[float, ...]:
    count = int(math.floor(radius / step))
    values = {index * step for index in range(-count, count + 1)}
    values.update({-radius, 0.0, radius})
    return tuple(sorted(values))


def estimate_planar_relocalization(
    *,
    reference_points: Iterable[tuple[float, float, float]],
    session_points: Iterable[tuple[float, float, float]],
    session_base_pose: Pose3,
    initial_map_base_pose: Pose3,
    search_xy_m: float,
    search_yaw_rad: float,
    min_z: float,
    max_z: float,
    match_voxel_m: float = 0.20,
    min_match_ratio: float = 0.20,
    min_points: int = 40,
    max_scan_points: int = 1200,
) -> RelocalizationResult:
    """Bounded 2D correlative scan match around an operator pose guess."""

    numeric = (
        search_xy_m,
        search_yaw_rad,
        min_z,
        max_z,
        match_voxel_m,
        min_match_ratio,
    )
    if not all(math.isfinite(value) for value in numeric):
        raise InvalidFastLivo2Frame("relocalization parameters must be finite")
    if search_xy_m <= 0 or search_yaw_rad <= 0 or match_voxel_m <= 0:
        raise InvalidFastLivo2Frame("relocalization search bounds must be positive")
    if min_z >= max_z or not 0.0 < min_match_ratio <= 1.0:
        raise InvalidFastLivo2Frame("relocalization thresholds are invalid")

    reference_cells = {
        (math.floor(x / match_voxel_m), math.floor(y / match_voxel_m))
        for x, y, z in reference_points
        if all(math.isfinite(value) for value in (x, y, z)) and min_z <= z <= max_z
    }
    if len(reference_cells) < min_points:
        raise InvalidFastLivo2Frame("saved map has too few obstacle points")
    expanded_reference_cells = {
        (ix + dx, iy + dy)
        for ix, iy in reference_cells
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
    }

    scan_voxels: dict[tuple[int, int], tuple[float, float]] = {}
    relative_z_offset = initial_map_base_pose.z - session_base_pose.z
    for x, y, z in session_points:
        if not all(math.isfinite(value) for value in (x, y, z)):
            continue
        if not min_z <= z + relative_z_offset <= max_z:
            continue
        key = (math.floor(x / match_voxel_m), math.floor(y / match_voxel_m))
        scan_voxels.setdefault(key, (x, y))
    samples = list(scan_voxels.values())
    if len(samples) > max_scan_points:
        stride = math.ceil(len(samples) / max_scan_points)
        samples = samples[::stride]
    if len(samples) < min_points:
        raise InvalidFastLivo2Frame("current scan has too few obstacle points")

    session_yaw = yaw_from_quaternion(session_base_pose.q)
    initial_yaw = yaw_from_quaternion(initial_map_base_pose.q)

    def evaluate(base_x: float, base_y: float, base_yaw: float):
        theta = base_yaw - session_yaw
        cosine = math.cos(theta)
        sine = math.sin(theta)
        tx = base_x - (cosine * session_base_pose.x - sine * session_base_pose.y)
        ty = base_y - (sine * session_base_pose.x + cosine * session_base_pose.y)
        hits = 0
        quality = 0.0
        for x, y in samples:
            mx = cosine * x - sine * y + tx
            my = sine * x + cosine * y + ty
            ix = math.floor(mx / match_voxel_m)
            iy = math.floor(my / match_voxel_m)
            if (ix, iy) in reference_cells:
                hits += 1
                quality += 1.0
            elif (ix, iy) in expanded_reference_cells:
                hits += 1
                quality += 0.25
        return quality / len(samples), hits

    def correction_distance(base_x: float, base_y: float, base_yaw: float) -> float:
        return (
            ((base_x - initial_map_base_pose.x) / search_xy_m) ** 2
            + ((base_y - initial_map_base_pose.y) / search_xy_m) ** 2
            + ((base_yaw - initial_yaw) / search_yaw_rad) ** 2
        )

    def in_bounds(base_x: float, base_y: float, base_yaw: float) -> bool:
        epsilon = 1e-9
        return (
            abs(base_x - initial_map_base_pose.x) <= search_xy_m + epsilon
            and abs(base_y - initial_map_base_pose.y) <= search_xy_m + epsilon
            and abs(base_yaw - initial_yaw) <= search_yaw_rad + epsilon
        )

    def consider(base_x: float, base_y: float, base_yaw: float, current):
        if not in_bounds(base_x, base_y, base_yaw):
            return current
        score, hits = evaluate(base_x, base_y, base_yaw)
        distance = correction_distance(base_x, base_y, base_yaw)
        if current is None or score > current[0] + 1e-12 or (
            abs(score - current[0]) <= 1e-12
            and (hits > current[1] or (hits == current[1] and distance < current[5]))
        ):
            return (score, hits, base_x, base_y, base_yaw, distance)
        return current

    coarse_xy = min(0.25, search_xy_m)
    coarse_yaw = min(0.10, search_yaw_rad)
    best = None
    for dx in _search_values(search_xy_m, coarse_xy):
        for dy in _search_values(search_xy_m, coarse_xy):
            for dyaw in _search_values(search_yaw_rad, coarse_yaw):
                best = consider(
                    initial_map_base_pose.x + dx,
                    initial_map_base_pose.y + dy,
                    initial_yaw + dyaw,
                    best,
                )
    if best is None:
        raise InvalidFastLivo2Frame("relocalization search produced no candidates")

    fine_xy = coarse_xy / 5.0
    fine_yaw = coarse_yaw / 5.0
    coarse_best = best
    for dx in _search_values(coarse_xy, fine_xy):
        for dy in _search_values(coarse_xy, fine_xy):
            for dyaw in _search_values(coarse_yaw, fine_yaw):
                best = consider(
                    coarse_best[2] + dx,
                    coarse_best[3] + dy,
                    coarse_best[4] + dyaw,
                    best,
                )

    if best[0] < min_match_ratio:
        raise InvalidFastLivo2Frame(
            f"scan-to-map match ratio {best[0]:.3f} is below {min_match_ratio:.3f}"
        )
    map_base = Pose3(
        best[2],
        best[3],
        initial_map_base_pose.z,
        quaternion_from_rpy(0.0, 0.0, best[4]),
    )
    return RelocalizationResult(
        map_from_session=compose_pose(map_base, inverse_pose(session_base_pose)),
        map_base_pose=map_base,
        match_ratio=best[0],
        matched_points=best[1],
        evaluated_points=len(samples),
    )


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
    def __init__(self, voxel_size_m: float):
        if not math.isfinite(voxel_size_m) or voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be finite and positive")
        self._voxel = voxel_size_m
        self._points: dict[tuple[int, int, int], tuple[float, float, float]] = {}

    def clear(self) -> None:
        self._points.clear()

    def add(self, points: Iterable[tuple[float, float, float]]) -> None:
        for point in points:
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

    @property
    def points(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(self._points.values())

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


def source_age_is_valid(
    source_age_sec: float,
    *,
    max_age_sec: float,
    tolerance_sec: float,
) -> bool:
    if not all(
        math.isfinite(value)
        for value in (source_age_sec, max_age_sec, tolerance_sec)
    ):
        return False
    if max_age_sec <= 0 or not 0 <= tolerance_sec <= 0.1:
        return False
    return -0.1 <= source_age_sec <= max_age_sec + tolerance_sec


__all__ = [
    "InvalidFastLivo2Frame",
    "Pose3",
    "Quaternion",
    "RelocalizationResult",
    "VoxelMap",
    "canonical_base_pose",
    "compose_pose",
    "estimate_planar_relocalization",
    "iter_xyz_points",
    "quaternion_from_rpy",
    "read_pcd_xyz",
    "source_age_is_valid",
    "transform_points",
    "yaw_from_quaternion",
]
