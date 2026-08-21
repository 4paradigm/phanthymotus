"""Pure coordinate and Canvas-map helpers for FAST-LIVO2 outputs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
import math
import os
from pathlib import Path
import struct
import time
from typing import Iterable, Sequence


FLOAT32 = 7
FLOAT64 = 8
_FRAME_HEADER = struct.Struct("<fffBI")
_POINT = struct.Struct("<fff")
_MAX_PCD_HEADER_BYTES = 65_536
_MAX_PCD_ASCII_RECORD_BYTES = 65_536
_FILTER_METADATA = struct.Struct("<8sff")
_FILTER_METADATA_MAGIC = b"MVFILT2\0"
_FULL_MAP_WITH_Z = 0x01 | 0x02


class InvalidFastLivo2Frame(ValueError):
    pass


class FastLivo2PersistenceError(OSError):
    """A valid map could not be persisted and may succeed on retry."""


def _require_deadline(deadline_monotonic: float | None, *, stage: str) -> None:
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise TimeoutError(f"PCD operation deadline exceeded during {stage}")


def _float32_xyz(
    point: Sequence[float],
    *,
    context: str,
) -> tuple[float, float, float]:
    """Normalize finite coordinates and reject values unsafe for float32 output."""

    if len(point) != 3:
        raise InvalidFastLivo2Frame(f"{context} must contain three coordinates")
    try:
        values = tuple(float(value) for value in point)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidFastLivo2Frame(f"{context} coordinates are invalid") from exc
    if not all(math.isfinite(value) for value in values):
        raise InvalidFastLivo2Frame(f"{context} coordinates must be finite")
    try:
        _POINT.pack(*values)
    except (OverflowError, struct.error) as exc:
        raise InvalidFastLivo2Frame(
            f"{context} coordinates exceed float32 range"
        ) from exc
    return values


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


@dataclass(frozen=True)
class OccupancyGridSnapshot:
    """ROS-independent full occupancy snapshot for Nav2 StaticLayer."""

    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int
    data: tuple[int, ...]
    occupied_cells: int
    free_cells: int


@dataclass(frozen=True)
class _PreparedConfirmedState:
    """Fully validated saved-map state awaiting one atomic pointer swap."""

    evidence: dict[tuple[int, int, int], tuple[float, float, float]]


def nearest_stamped_pose(
    history: Sequence[tuple[int, Pose3]],
    stamp_ns: int,
    *,
    tolerance_ns: int,
) -> Pose3 | None:
    """Return the closest pose only when its source timestamp is compatible."""

    if tolerance_ns < 0:
        raise ValueError("tolerance_ns must be non-negative")
    if not history:
        return None
    nearest_stamp, nearest_pose = min(
        history,
        key=lambda item: abs(int(item[0]) - int(stamp_ns)),
    )
    if abs(int(nearest_stamp) - int(stamp_ns)) > tolerance_ns:
        return None
    return nearest_pose


def bracketed_stamped_pose(
    history: Sequence[tuple[int, Pose3]],
    stamp_ns: int,
    *,
    tolerance_ns: int,
) -> Pose3 | None:
    """Return a nearby pose only after TF history brackets the source stamp."""

    if tolerance_ns < 0:
        raise ValueError("tolerance_ns must be non-negative")
    before: tuple[int, Pose3] | None = None
    after: tuple[int, Pose3] | None = None
    for candidate_stamp, candidate_pose in history:
        candidate_stamp = int(candidate_stamp)
        if candidate_stamp <= stamp_ns and (
            before is None or candidate_stamp > before[0]
        ):
            before = (candidate_stamp, candidate_pose)
        if candidate_stamp >= stamp_ns and (
            after is None or candidate_stamp < after[0]
        ):
            after = (candidate_stamp, candidate_pose)
    if before is None or after is None:
        return None
    before_delta = stamp_ns - before[0]
    after_delta = after[0] - stamp_ns
    if min(before_delta, after_delta) > tolerance_ns:
        return None
    if before_delta <= after_delta:
        return before[1]
    return after[1]


def normalize_obstacle_height_range(
    value,
    *,
    field_name: str = "obstacle_height_range_m",
) -> tuple[float, float]:
    """Validate a persisted navigation-height interval without coercing text."""

    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value
        )
    ):
        raise ValueError(f"{field_name} must contain two numbers")
    minimum, maximum = (float(item) for item in value)
    if not all(math.isfinite(item) for item in (minimum, maximum)):
        raise ValueError(f"{field_name} must contain finite numbers")
    if minimum >= maximum:
        raise ValueError(f"{field_name} minimum must be below maximum")
    return minimum, maximum


def obstacle_height_ranges_match(
    saved: Sequence[float],
    active: Sequence[float],
    *,
    tolerance: float = 1e-6,
) -> bool:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("height-range tolerance must be finite and non-negative")
    saved_range = normalize_obstacle_height_range(saved, field_name="saved range")
    active_range = normalize_obstacle_height_range(active, field_name="active range")
    return all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)
        for left, right in zip(saved_range, active_range)
    )


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
    translation = _float32_xyz(
        (left.x + rx, left.y + ry, left.z + rz),
        context="composed pose translation",
    )
    return Pose3(
        translation[0],
        translation[1],
        translation[2],
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
    translation = _float32_xyz(
        (pose.x, pose.y, pose.z),
        context="transform translation",
    )
    q = normalize_quaternion(pose.q)
    xx, yy, zz = q.x * q.x, q.y * q.y, q.z * q.z
    xy, xz, yz = q.x * q.y, q.x * q.z, q.y * q.z
    wx, wy, wz = q.w * q.x, q.w * q.y, q.w * q.z
    matrix = (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )
    for point in points:
        x, y, z = _float32_xyz(point, context="transform input point")
        rx = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z
        ry = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z
        rz = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z
        yield _float32_xyz(
            (
                translation[0] + rx,
                translation[1] + ry,
                translation[2] + rz,
            ),
            context="transformed point",
        )


def read_pcd_xyz(
    path: str | Path,
    *,
    max_points: int | None = None,
    max_declared_points: int | None = 5_000_000,
    max_file_bytes: int = 1_073_741_824,
    deadline_monotonic: float | None = None,
):
    """Read finite XYZ points from an ASCII or uncompressed binary PCD file."""

    if max_points is not None and max_points < 1:
        raise ValueError("max_points must be positive when provided")
    if max_declared_points is not None and max_declared_points < 1:
        raise ValueError("max_declared_points must be positive when provided")
    if max_file_bytes < 1:
        raise ValueError("max_file_bytes must be positive")
    if deadline_monotonic is not None:
        if (
            isinstance(deadline_monotonic, bool)
            or not isinstance(deadline_monotonic, (int, float))
            or not math.isfinite(float(deadline_monotonic))
        ):
            raise ValueError("deadline_monotonic must be finite when provided")
        deadline_monotonic = float(deadline_monotonic)
    _require_deadline(deadline_monotonic, stage="PCD inspection")
    source = Path(path)
    try:
        file_size = source.stat().st_size
    except OSError as exc:
        raise InvalidFastLivo2Frame(f"cannot inspect PCD {source.name}: {exc}") from exc
    if file_size > max_file_bytes:
        raise InvalidFastLivo2Frame(
            f"PCD file exceeds {max_file_bytes} byte safety limit"
        )
    try:
        stream = source.open("rb")
    except OSError as exc:
        raise InvalidFastLivo2Frame(f"cannot open PCD {source.name}: {exc}") from exc
    with stream:
        header: dict[str, list[str]] = {}
        header_bytes = 0
        while True:
            _require_deadline(deadline_monotonic, stage="PCD header parsing")
            remaining_header_bytes = _MAX_PCD_HEADER_BYTES - header_bytes
            raw = stream.readline(remaining_header_bytes + 1)
            if len(raw) > remaining_header_bytes:
                raise InvalidFastLivo2Frame(
                    f"PCD header exceeds {_MAX_PCD_HEADER_BYTES} byte limit"
                )
            if not raw:
                raise InvalidFastLivo2Frame("PCD header is missing DATA")
            header_bytes += len(raw)
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
        if (
            max_declared_points is not None
            and declared_points > max_declared_points
        ):
            raise InvalidFastLivo2Frame(
                "PCD declared point count exceeds "
                f"{max_declared_points} point safety limit"
            )

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
            while True:
                _require_deadline(deadline_monotonic, stage="PCD ASCII parsing")
                raw = stream.readline(_MAX_PCD_ASCII_RECORD_BYTES + 1)
                if len(raw) > _MAX_PCD_ASCII_RECORD_BYTES:
                    raise InvalidFastLivo2Frame(
                        "PCD ASCII record exceeds "
                        f"{_MAX_PCD_ASCII_RECORD_BYTES} byte limit"
                    )
                if not raw:
                    break
                values = raw.split()
                if not values:
                    continue
                if len(values) != token_offset:
                    raise InvalidFastLivo2Frame(
                        "PCD ASCII point token count does not match field layout"
                    )
                if source_index >= declared_points:
                    raise InvalidFastLivo2Frame(
                        "PCD ASCII payload has more rows than declared"
                    )
                should_sample = source_index % sample_stride == 0
                source_index += 1
                try:
                    point = tuple(
                        float(values[token_offsets[name]])
                        for name in ("x", "y", "z")
                    )
                except (ValueError, IndexError) as exc:
                    raise InvalidFastLivo2Frame("PCD ASCII point is malformed") from exc
                if not all(math.isfinite(value) for value in point):
                    continue
                point = _float32_xyz(point, context="PCD ASCII point")
                if should_sample and (
                    max_points is None or len(points) < max_points
                ):
                    points.append(point)
            if source_index != declared_points:
                raise InvalidFastLivo2Frame(
                    "PCD ASCII payload row count does not match declared points"
                )
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
            if available > required:
                raise InvalidFastLivo2Frame(
                    "PCD binary payload has more records than declared"
                )
            for point_index in range(0, declared_points, sample_stride):
                _require_deadline(deadline_monotonic, stage="PCD binary parsing")
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
                    points.append(_float32_xyz(point, context="PCD binary point"))
        else:
            raise InvalidFastLivo2Frame(
                f"PCD DATA {mode or '<missing>'} is unsupported; use ascii or binary"
            )
    _require_deadline(deadline_monotonic, stage="PCD completion")
    if not points:
        raise InvalidFastLivo2Frame("PCD contains no finite XYZ points")
    return tuple(points)


def write_pcd_xyz_atomic(
    path: str | Path,
    points: Iterable[tuple[float, float, float]],
) -> int:
    """Atomically persist finite XYZ points as an uncompressed binary PCD."""

    destination = Path(path)
    finite = []
    for point in points:
        if len(point) != 3:
            continue
        try:
            normalized = tuple(float(value) for value in point)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidFastLivo2Frame(
                "static map point coordinates are invalid"
            ) from exc
        if not all(math.isfinite(value) for value in normalized):
            continue
        finite.append(_float32_xyz(normalized, context="static map point"))
    if not finite:
        raise InvalidFastLivo2Frame("static map contains no finite XYZ points")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    header = (
        "# .PCD v0.7\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {len(finite)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(finite)}\n"
        "DATA binary\n"
    ).encode("ascii")
    try:
        with temporary.open("wb") as stream:
            stream.write(header)
            for point in finite:
                stream.write(_POINT.pack(*point))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise FastLivo2PersistenceError(
            f"cannot persist static PCD {destination.name}: {exc}"
        ) from exc
    return len(finite)


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
    min_match_ratio: float = 0.35,
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
    boundary_margin_xy = fine_xy * 0.5
    boundary_margin_yaw = fine_yaw * 0.5
    correction_x = abs(best[2] - initial_map_base_pose.x)
    correction_y = abs(best[3] - initial_map_base_pose.y)
    correction_yaw = abs(best[4] - initial_yaw)
    if (
        correction_x >= search_xy_m - boundary_margin_xy
        or correction_y >= search_xy_m - boundary_margin_xy
        or correction_yaw >= search_yaw_rad - boundary_margin_yaw
    ):
        raise InvalidFastLivo2Frame(
            "best relocalization candidate lies on the search boundary; "
            "increase the search radius or improve the initial pose"
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
    max_points: int | None = None,
    max_data_bytes: int | None = None,
) -> Iterable[tuple[float, float, float]]:
    if is_bigendian:
        raise InvalidFastLivo2Frame("big-endian PointCloud2 is unsupported")
    if point_step <= 0 or width < 0 or height < 0:
        raise InvalidFastLivo2Frame("invalid PointCloud2 dimensions")
    if max_points is not None and max_points < 1:
        raise ValueError("max_points must be positive when provided")
    if max_data_bytes is not None and max_data_bytes < 1:
        raise ValueError("max_data_bytes must be positive when provided")
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
    if max_points is not None and point_count > max_points:
        raise InvalidFastLivo2Frame(
            f"PointCloud2 exceeds {max_points} point safety limit"
        )
    if max_data_bytes is not None and len(data) > max_data_bytes:
        raise InvalidFastLivo2Frame(
            f"PointCloud2 exceeds {max_data_bytes} byte safety limit"
        )
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
            yield _float32_xyz(values, context="PointCloud2 point")


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
            if len(point) != 3:
                raise InvalidFastLivo2Frame(
                    "voxel point must contain three coordinates"
                )
            try:
                normalized = tuple(float(value) for value in point)
            except (TypeError, ValueError, OverflowError) as exc:
                raise InvalidFastLivo2Frame(
                    "voxel point coordinates are invalid"
                ) from exc
            if not all(math.isfinite(value) for value in normalized):
                continue
            x, y, z = _float32_xyz(normalized, context="voxel point")
            try:
                key = (
                    math.floor(x / self._voxel),
                    math.floor(y / self._voxel),
                    math.floor(z / self._voxel),
                )
            except OverflowError as exc:
                raise InvalidFastLivo2Frame(
                    "voxel point exceeds supported coordinate range"
                ) from exc
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

    def encode(
        self,
        robot_pose: Pose3,
        *,
        obstacle_min_height_m: float | None = None,
        obstacle_max_height_m: float | None = None,
        max_points: int | None = None,
    ) -> bytes:
        if (obstacle_min_height_m is None) != (obstacle_max_height_m is None):
            raise ValueError("both obstacle height limits are required")
        if max_points is not None and (
            isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 1
        ):
            raise ValueError("max_points must be a positive integer")
        metadata = b""
        if obstacle_min_height_m is not None and obstacle_max_height_m is not None:
            if not all(
                math.isfinite(value)
                for value in (obstacle_min_height_m, obstacle_max_height_m)
            ):
                raise ValueError("obstacle height limits must be finite")
            if obstacle_min_height_m >= obstacle_max_height_m:
                raise ValueError("obstacle minimum height must be below maximum")
            metadata = _FILTER_METADATA.pack(
                _FILTER_METADATA_MAGIC,
                obstacle_min_height_m,
                obstacle_max_height_m,
            )
        points = tuple(self._points.values())
        if max_points is not None and len(points) > max_points:
            if obstacle_min_height_m is None:
                points = self._even_sample(points, max_points)
            else:
                below = tuple(
                    point for point in points if point[2] < obstacle_min_height_m
                )
                obstacle = tuple(
                    point
                    for point in points
                    if obstacle_min_height_m <= point[2] <= obstacle_max_height_m
                )
                above = tuple(
                    point for point in points if point[2] > obstacle_max_height_m
                )
                groups = (below, obstacle, above)
                allocations = [
                    min(len(below), max_points * 35 // 100),
                    min(len(obstacle), max_points * 55 // 100),
                    min(len(above), max_points * 10 // 100),
                ]
                remaining = max_points - sum(allocations)
                for index in sorted(
                    range(len(groups)),
                    key=lambda item: len(groups[item]) - allocations[item],
                    reverse=True,
                ):
                    extra = min(remaining, len(groups[index]) - allocations[index])
                    allocations[index] += extra
                    remaining -= extra
                    if remaining == 0:
                        break
                points = tuple(
                    point
                    for group, allocation in zip(groups, allocations)
                    for point in self._even_sample(group, allocation)
                )
        yaw = yaw_from_quaternion(robot_pose.q)
        body = bytearray(len(points) * _POINT.size)
        for index, point in enumerate(points):
            normalized = _float32_xyz(point, context="encoded map point")
            _POINT.pack_into(body, index * _POINT.size, *normalized)
        header_pose = _float32_xyz(
            (robot_pose.x, robot_pose.y, yaw),
            context="encoded robot pose",
        )
        try:
            header = _FRAME_HEADER.pack(
                header_pose[0],
                header_pose[1],
                header_pose[2],
                _FULL_MAP_WITH_Z,
                len(points),
            )
        except (OverflowError, struct.error) as exc:
            raise InvalidFastLivo2Frame("encoded map header is invalid") from exc
        return header + body + metadata

    @staticmethod
    def _even_sample(
        points: tuple[tuple[float, float, float], ...],
        count: int,
    ) -> tuple[tuple[float, float, float], ...]:
        if count <= 0:
            return ()
        if len(points) <= count:
            return points
        return tuple(points[index * len(points) // count] for index in range(count))


def encode_map_view_points(
    points: Iterable[tuple[float, float, float]],
    robot_pose: Pose3,
    *,
    obstacle_min_height_m: float,
    obstacle_max_height_m: float,
    max_points: int,
) -> bytes:
    """Encode already bounded display sources without rebuilding a voxel map.

    The static map, out-of-band context, and current scan are already bounded
    independently. Rehashing all of them into a temporary ``VoxelMap`` once
    per second made Canvas rendering monopolize the single-threaded adapter and
    delayed the canonical odom/cloud outputs. Callers pass points already
    validated by ``VoxelMap``, ``TemporalOccupancyMap``, or the live cloud
    decoder. This display-only encoder keeps the existing height-balanced point
    budget without repeating the same Python validation and voxel hashing.
    """

    if not all(
        math.isfinite(value)
        for value in (obstacle_min_height_m, obstacle_max_height_m)
    ):
        raise ValueError("obstacle height limits must be finite")
    if obstacle_min_height_m >= obstacle_max_height_m:
        raise ValueError("obstacle minimum height must be below maximum")
    if (
        isinstance(max_points, bool)
        or not isinstance(max_points, int)
        or max_points < 1
    ):
        raise ValueError("max_points must be a positive integer")

    selected = tuple(points)
    if len(selected) > max_points:
        below = tuple(
            point for point in selected if point[2] < obstacle_min_height_m
        )
        obstacle = tuple(
            point
            for point in selected
            if obstacle_min_height_m <= point[2] <= obstacle_max_height_m
        )
        above = tuple(
            point for point in selected if point[2] > obstacle_max_height_m
        )
        groups = (below, obstacle, above)
        allocations = [
            min(len(below), max_points * 35 // 100),
            min(len(obstacle), max_points * 55 // 100),
            min(len(above), max_points * 10 // 100),
        ]
        remaining = max_points - sum(allocations)
        for index in sorted(
            range(len(groups)),
            key=lambda item: len(groups[item]) - allocations[item],
            reverse=True,
        ):
            extra = min(remaining, len(groups[index]) - allocations[index])
            allocations[index] += extra
            remaining -= extra
            if remaining == 0:
                break
        selected = tuple(
            point
            for group, allocation in zip(groups, allocations)
            for point in VoxelMap._even_sample(group, allocation)
        )

    try:
        body = struct.pack(
            f"<{len(selected) * 3}f",
            *chain.from_iterable(selected),
        )
    except (OverflowError, struct.error, TypeError, ValueError) as exc:
        raise InvalidFastLivo2Frame(
            "validated map-view points cannot be encoded"
        ) from exc
    header_pose = _float32_xyz(
        (robot_pose.x, robot_pose.y, yaw_from_quaternion(robot_pose.q)),
        context="encoded robot pose",
    )
    try:
        header = _FRAME_HEADER.pack(
            header_pose[0],
            header_pose[1],
            header_pose[2],
            _FULL_MAP_WITH_Z,
            len(selected),
        )
    except (OverflowError, struct.error) as exc:
        raise InvalidFastLivo2Frame("encoded map header is invalid") from exc
    metadata = _FILTER_METADATA.pack(
        _FILTER_METADATA_MAGIC,
        obstacle_min_height_m,
        obstacle_max_height_m,
    )
    return header + body + metadata


class TemporalOccupancyMap:
    """Accumulate every observed navigation-height voxel until reset."""

    def __init__(
        self,
        voxel_size_m: float,
        *,
        raytrace_min_range_m: float = 0.20,
        raytrace_max_range_m: float = 8.5,
        angular_bin_deg: float = 1.0,
        grid_margin_m: float = 6.0,
        max_grid_dimension_cells: int = 2048,
        max_grid_cells: int = 2_000_000,
        max_evidence_points: int = 200_000,
    ):
        values = (
            voxel_size_m,
            raytrace_min_range_m,
            raytrace_max_range_m,
            angular_bin_deg,
            grid_margin_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("temporal occupancy parameters must be finite")
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        if not 0 <= raytrace_min_range_m < raytrace_max_range_m:
            raise ValueError("raytrace range is invalid")
        if not 0 < angular_bin_deg <= 10:
            raise ValueError("angular_bin_deg must be within (0, 10]")
        if grid_margin_m < voxel_size_m:
            raise ValueError("grid_margin_m must cover at least one voxel")
        if max_grid_dimension_cells < 1 or max_grid_cells < 1:
            raise ValueError("occupancy grid limits must be positive")
        if max_evidence_points < 40:
            raise ValueError("max_evidence_points must be at least 40")

        self._voxel = voxel_size_m
        self._raytrace_min_range = raytrace_min_range_m
        self._raytrace_max_range = raytrace_max_range_m
        self._angular_bin = math.radians(angular_bin_deg)
        self._grid_margin_cells = math.ceil(grid_margin_m / voxel_size_m)
        self._max_grid_dimension_cells = int(max_grid_dimension_cells)
        self._max_grid_cells = int(max_grid_cells)
        self._max_evidence_points = int(max_evidence_points)
        self._evidence: dict[
            tuple[int, int, int], tuple[float, float, float]
        ] = {}
        self._free_cells: set[tuple[int, int]] = set()
        self._bounds: tuple[int, int, int, int] | None = None
        self._read_only_baseline = False
        self._last_observation_monotonic: float | None = None

    def clear(self) -> None:
        self.retire_and_clear()

    def retire_and_clear(self) -> tuple[object, ...]:
        """Detach the active state in O(1) so callers can release it later."""

        retired = (
            self._evidence,
            self._free_cells,
        )
        self._evidence = {}
        self._free_cells = set()
        self._bounds = None
        self._read_only_baseline = False
        self._last_observation_monotonic = None
        return retired

    @property
    def point_count(self) -> int:
        return len(self._evidence)

    @property
    def free_cell_count(self) -> int:
        return len(self._free_cells)

    @property
    def confirmed_points(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(
            point for key, point in sorted(self._evidence.items())
        )

    @property
    def map_view_points(self) -> tuple[tuple[float, float, float], ...]:
        """Return an unordered immutable snapshot for display-only encoding."""

        return tuple(self._evidence.values())

    def load_confirmed(
        self, points: Iterable[tuple[float, float, float]]
    ) -> None:
        """Replace temporal evidence with an immutable saved-map baseline."""

        self.apply_prepared_confirmed(self.prepare_confirmed(points))

    def prepare_confirmed(
        self, points: Iterable[tuple[float, float, float]]
    ) -> _PreparedConfirmedState:
        """Build all saved-map state without changing the active map."""

        return _PreparedConfirmedState(
            evidence=self._validated_confirmed_state(points),
        )

    def apply_prepared_confirmed(
        self, prepared: _PreparedConfirmedState
    ) -> tuple[object, ...]:
        """Atomically activate prepared state and return detached old storage."""

        if not isinstance(prepared, _PreparedConfirmedState):
            raise TypeError("prepared confirmed state is invalid")
        retired = (
            self._evidence,
            self._free_cells,
        )
        self._evidence = prepared.evidence
        self._free_cells = set()
        self._bounds = None
        self._read_only_baseline = True
        self._last_observation_monotonic = None
        return retired

    def validate_confirmed(
        self, points: Iterable[tuple[float, float, float]]
    ) -> int:
        """Validate a saved-map baseline without changing the active map."""

        return len(self.prepare_confirmed(points).evidence)

    def _validated_confirmed_state(
        self, points: Iterable[tuple[float, float, float]]
    ) -> dict[tuple[int, int, int], tuple[float, float, float]]:
        """Build a complete immutable-map replacement before mutating state."""

        evidence_by_key: dict[
            tuple[int, int, int], tuple[float, float, float]
        ] = {}
        for point in points:
            key = self._key(point)
            if key is None or key in evidence_by_key:
                continue
            normalized = tuple(float(value) for value in point)
            evidence_by_key[key] = normalized
            if len(evidence_by_key) > self._max_evidence_points:
                raise ValueError(
                    "confirmed static map exceeds "
                    f"{self._max_evidence_points} point safety limit"
                )
        return evidence_by_key

    def observe_scan(
        self,
        *,
        sensor_origin: tuple[float, float, float],
        points: Iterable[tuple[float, float, float]],
        now_monotonic: float,
        obstacle_min_height_m: float,
        obstacle_max_height_m: float,
    ) -> None:
        if self._read_only_baseline:
            raise ValueError("loaded static map is immutable")
        if not math.isfinite(now_monotonic):
            raise ValueError("now_monotonic must be finite")
        if (
            self._last_observation_monotonic is not None
            and now_monotonic < self._last_observation_monotonic
        ):
            raise ValueError("now_monotonic must not move backwards")
        if not all(math.isfinite(value) for value in sensor_origin):
            raise ValueError("sensor_origin must contain finite values")
        if not all(
            math.isfinite(value)
            for value in (obstacle_min_height_m, obstacle_max_height_m)
        ) or obstacle_min_height_m >= obstacle_max_height_m:
            raise ValueError("obstacle height range is invalid")
        observed: dict[
            tuple[int, int, int], tuple[float, float, float]
        ] = {}
        obstacle_endpoints: dict[tuple[int, int], tuple[float, float, float]] = {}
        for point in points:
            key = self._key(point)
            if key is None:
                continue
            normalized = tuple(float(value) for value in point)
            if not obstacle_min_height_m <= normalized[2] <= obstacle_max_height_m:
                continue
            planar_range = math.hypot(
                normalized[0] - sensor_origin[0],
                normalized[1] - sensor_origin[1],
            )
            if not self._raytrace_min_range <= planar_range <= self._raytrace_max_range:
                continue
            observed.setdefault(key, normalized)
            obstacle_endpoints.setdefault(key[:2], normalized)

        observed_xy = set(obstacle_endpoints)
        ray_free = self._ray_free_cells(
            sensor_origin=sensor_origin,
            endpoints=obstacle_endpoints.values(),
        )
        ray_free.difference_update(observed_xy)
        projected_evidence = len(self._evidence) + sum(
            1 for key in observed if key not in self._evidence
        )
        if projected_evidence > self._max_evidence_points:
            raise ValueError(
                "static map evidence exceeds "
                f"{self._max_evidence_points} point safety limit"
            )
        self._last_observation_monotonic = now_monotonic
        for key, point in observed.items():
            self._evidence[key] = point

        self._free_cells.update(ray_free)
        self._free_cells.difference_update(observed_xy)

    def project_xy(
        self,
        *,
        min_z: float,
        max_z: float,
        output_z: float = 0.0,
    ) -> tuple[tuple[float, float, float], ...]:
        if not all(math.isfinite(value) for value in (min_z, max_z, output_z)):
            raise ValueError("projection heights must be finite")
        if min_z >= max_z:
            raise ValueError("min_z must be less than max_z")
        projected: dict[tuple[int, int], tuple[float, float, float]] = {}
        for key, point in self._evidence.items():
            if min_z <= point[2] <= max_z:
                ix, iy, _ = key
                projected.setdefault(
                    (ix, iy),
                    ((ix + 0.5) * self._voxel, (iy + 0.5) * self._voxel, output_z),
                )
        return tuple(projected[key] for key in sorted(projected))

    def occupancy_snapshot(
        self,
        *,
        center_x: float,
        center_y: float,
        min_z: float,
        max_z: float,
    ) -> OccupancyGridSnapshot:
        if not all(math.isfinite(value) for value in (center_x, center_y, min_z, max_z)):
            raise ValueError("occupancy snapshot values must be finite")
        if min_z >= max_z:
            raise ValueError("min_z must be less than max_z")
        snapshot, bounds = self._build_occupancy_snapshot(
            evidence=self._evidence,
            free_cells=self._free_cells,
            center_x=center_x,
            center_y=center_y,
            min_z=min_z,
            max_z=max_z,
        )
        self._bounds = bounds
        return snapshot

    def prepared_occupancy_snapshot(
        self,
        prepared: _PreparedConfirmedState,
        *,
        center_x: float,
        center_y: float,
        min_z: float,
        max_z: float,
    ) -> OccupancyGridSnapshot:
        """Build the latched grid before activating a prepared saved map."""

        if not isinstance(prepared, _PreparedConfirmedState):
            raise TypeError("prepared confirmed state is invalid")
        snapshot, _bounds = self._build_occupancy_snapshot(
            evidence=prepared.evidence,
            free_cells=(),
            center_x=center_x,
            center_y=center_y,
            min_z=min_z,
            max_z=max_z,
        )
        return snapshot

    def _build_occupancy_snapshot(
        self,
        *,
        evidence: dict[tuple[int, int, int], tuple[float, float, float]],
        free_cells: Iterable[tuple[int, int]],
        center_x: float,
        center_y: float,
        min_z: float,
        max_z: float,
    ) -> tuple[OccupancyGridSnapshot, tuple[int, int, int, int]]:
        if not all(math.isfinite(value) for value in (center_x, center_y, min_z, max_z)):
            raise ValueError("occupancy snapshot values must be finite")
        if min_z >= max_z:
            raise ValueError("min_z must be less than max_z")
        center_ix, center_iy = self._xy_key(center_x, center_y)
        min_ix = center_ix - self._grid_margin_cells
        max_ix = center_ix + self._grid_margin_cells
        min_iy = center_iy - self._grid_margin_cells
        max_iy = center_iy + self._grid_margin_cells
        expanded = (min_ix, max_ix, min_iy, max_iy)
        width = max_ix - min_ix + 1
        height = max_iy - min_iy + 1
        if (
            width > self._max_grid_dimension_cells
            or height > self._max_grid_dimension_cells
            or width * height > self._max_grid_cells
        ):
            raise ValueError(
                "configured occupancy window exceeds dimension or cell-count limit"
            )
        data = [-1] * (width * height)

        def offset(cell: tuple[int, int]) -> int | None:
            ix, iy = cell
            if not min_ix <= ix <= max_ix or not min_iy <= iy <= max_iy:
                return None
            return (iy - min_iy) * width + (ix - min_ix)

        free_count = 0
        for cell in free_cells:
            index = offset(cell)
            if index is not None and data[index] != 0:
                data[index] = 0
                free_count += 1

        occupied = {
            key[:2]
            for key, item in evidence.items()
            if min_z <= item[2] <= max_z
        }
        occupied_count = 0
        for cell in occupied:
            index = offset(cell)
            if index is not None:
                if data[index] == 0:
                    free_count -= 1
                data[index] = 100
                occupied_count += 1
        return (
            OccupancyGridSnapshot(
                resolution=self._voxel,
                origin_x=min_ix * self._voxel,
                origin_y=min_iy * self._voxel,
                width=width,
                height=height,
                data=tuple(data),
                occupied_cells=occupied_count,
                free_cells=free_count,
            ),
            expanded,
        )

    def cleared_snapshot(self) -> OccupancyGridSnapshot:
        """Return a free grid over the current bounds to clear latched consumers."""

        if self._bounds is None:
            return OccupancyGridSnapshot(
                resolution=self._voxel,
                origin_x=0.0,
                origin_y=0.0,
                width=1,
                height=1,
                data=(0,),
                occupied_cells=0,
                free_cells=1,
            )
        min_ix, max_ix, min_iy, max_iy = self._bounds
        width = max_ix - min_ix + 1
        height = max_iy - min_iy + 1
        return OccupancyGridSnapshot(
            resolution=self._voxel,
            origin_x=min_ix * self._voxel,
            origin_y=min_iy * self._voxel,
            width=width,
            height=height,
            data=(0,) * (width * height),
            occupied_cells=0,
            free_cells=width * height,
        )

    def _key(
        self, point: Sequence[float]
    ) -> tuple[int, int, int] | None:
        if len(point) != 3:
            return None
        try:
            normalized = tuple(float(value) for value in point)
        except (TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(value) for value in normalized):
            return None
        x, y, z = _float32_xyz(normalized, context="temporal map point")
        try:
            return (
                math.floor(x / self._voxel),
                math.floor(y / self._voxel),
                math.floor(z / self._voxel),
            )
        except OverflowError as exc:
            raise InvalidFastLivo2Frame(
                "temporal map point exceeds supported coordinate range"
            ) from exc

    def _xy_key(self, x: float, y: float) -> tuple[int, int]:
        try:
            normalized_x = float(x)
            normalized_y = float(y)
            if not all(math.isfinite(value) for value in (normalized_x, normalized_y)):
                raise InvalidFastLivo2Frame(
                    "temporal map XY coordinates must be finite"
                )
            return (
                math.floor(normalized_x / self._voxel),
                math.floor(normalized_y / self._voxel),
            )
        except OverflowError as exc:
            raise InvalidFastLivo2Frame(
                "temporal map XY coordinates exceed supported range"
            ) from exc

    def _ray_free_cells(
        self,
        *,
        sensor_origin: tuple[float, float, float],
        endpoints: Iterable[tuple[float, float, float]],
    ) -> set[tuple[int, int]]:
        nearest_by_angle: dict[
            int, tuple[float, tuple[int, int]]
        ] = {}
        origin_x, origin_y, _ = sensor_origin
        for point in endpoints:
            delta_x = point[0] - origin_x
            delta_y = point[1] - origin_y
            distance = math.hypot(delta_x, delta_y)
            if not self._raytrace_min_range <= distance <= self._raytrace_max_range:
                continue
            angle_bin = math.floor(math.atan2(delta_y, delta_x) / self._angular_bin)
            endpoint = self._xy_key(point[0], point[1])
            previous = nearest_by_angle.get(angle_bin)
            if previous is None or distance < previous[0]:
                nearest_by_angle[angle_bin] = (distance, endpoint)

        origin = self._xy_key(origin_x, origin_y)
        free: set[tuple[int, int]] = set()
        for _distance, endpoint in nearest_by_angle.values():
            free.update(self._bresenham(origin, endpoint))
        free.discard(origin)
        return free

    @staticmethod
    def _bresenham(
        start: tuple[int, int], end: tuple[int, int]
    ) -> tuple[tuple[int, int], ...]:
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        step_x = 1 if x0 < x1 else -1
        step_y = 1 if y0 < y1 else -1
        error = dx + dy
        cells: list[tuple[int, int]] = []
        while (x0, y0) != (x1, y1):
            cells.append((x0, y0))
            twice_error = 2 * error
            if twice_error >= dy:
                error += dy
                x0 += step_x
            if twice_error <= dx:
                error += dx
                y0 += step_y
        return tuple(cells)


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
    "bracketed_stamped_pose",
    "FastLivo2PersistenceError",
    "InvalidFastLivo2Frame",
    "nearest_stamped_pose",
    "normalize_obstacle_height_range",
    "obstacle_height_ranges_match",
    "OccupancyGridSnapshot",
    "Pose3",
    "Quaternion",
    "RelocalizationResult",
    "TemporalOccupancyMap",
    "VoxelMap",
    "canonical_base_pose",
    "compose_pose",
    "estimate_planar_relocalization",
    "iter_xyz_points",
    "quaternion_from_rpy",
    "read_pcd_xyz",
    "source_age_is_valid",
    "transform_points",
    "write_pcd_xyz_atomic",
    "yaw_from_quaternion",
]
