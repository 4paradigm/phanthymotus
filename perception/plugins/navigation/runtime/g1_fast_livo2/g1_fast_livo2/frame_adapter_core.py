"""Pure coordinate and Canvas-map helpers for FAST-LIVO2 outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
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
_COMPONENT_SAMPLE_RATE_HZ = 20.0
_MAX_COMPONENT_OBSERVATION_WINDOW_SEC = 30.0
_MAX_COMPONENT_HISTORY_UNITS = 1_000_000
_COMPONENT_TRACK_HISTORY_UNITS = 64
_COMPONENT_CELL = struct.Struct("<iiffB")
_COMPONENT_POINT_2D = struct.Struct("<ff")


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

    evidence: dict[tuple[int, int, int], _TemporalVoxelEvidence]
    xy_index: dict[tuple[int, int], set[tuple[int, int, int]]]
    bounds: tuple[int, int, int, int] | None


@dataclass
class _TemporalVoxelEvidence:
    point: tuple[float, float, float]
    hit_frames: int
    first_hit_monotonic: float
    first_hit_point: tuple[float, float, float]
    last_hit_monotonic: float
    last_hit_scan: int
    confirmed: bool = False
    free_miss_frames: int = 0


@dataclass(frozen=True)
class _ComponentObservation:
    keys: frozenset[tuple[int, int, int]]
    cells: frozenset[tuple[int, int]]
    point_source: dict[
        tuple[int, int], list[tuple[float, float, float]]
    ] = field(compare=False, repr=False)
    centroid_x: float
    centroid_y: float
    span_x: float
    span_y: float


@dataclass(frozen=True)
class _ComponentMotionSample:
    stamp: float
    centroid_x: float
    centroid_y: float
    cell_count: int
    payload: bytes


@dataclass
class _TemporalComponentTrack:
    track_id: int
    centroid_x: float
    centroid_y: float
    first_seen_monotonic: float
    last_seen_monotonic: float
    cells: frozenset[tuple[int, int]]
    samples: list[_ComponentMotionSample]
    motion_votes: int = 0
    dynamic: bool = False
    last_motion_monotonic: float | None = None
    recent_dynamic_cells: dict[tuple[int, int], float] = field(
        default_factory=dict
    )
    recent_keys: dict[tuple[int, int, int], float] = field(
        default_factory=dict
    )
    history_saturated: bool = False


def _component_cell_geometry(
    component: _ComponentObservation,
) -> tuple[
    tuple[tuple[int, int, float, float], ...],
    tuple[tuple[int, int, tuple[tuple[float, float], ...]], ...],
]:
    """Build detailed geometry only when motion comparison needs it."""

    cell_centroids = []
    sampled_cell_points = []
    for cell in sorted(component.cells):
        cell_points = component.point_source[cell]
        cell_centroids.append(
            (
                cell[0],
                cell[1],
                sum(point[0] for point in cell_points) / len(cell_points),
                sum(point[1] for point in cell_points) / len(cell_points),
            )
        )
        ordered_points = sorted(
            (float(point[0]), float(point[1])) for point in cell_points
        )
        if len(ordered_points) > 8:
            stride = max(1, len(ordered_points) // 8)
            ordered_points = ordered_points[::stride][:8]
        sampled_cell_points.append(
            (cell[0], cell[1], tuple(ordered_points))
        )
    return tuple(cell_centroids), tuple(sampled_cell_points)


def _pack_component_motion_sample(
    component: _ComponentObservation,
    stamp: float,
) -> _ComponentMotionSample:
    """Pack bounded per-cell motion evidence without retaining frame objects."""

    cell_centroids, cell_points = _component_cell_geometry(component)
    centroids_by_cell = {
        (cell_x, cell_y): (point_x, point_y)
        for cell_x, cell_y, point_x, point_y in cell_centroids
    }
    payload = bytearray()
    for cell_x, cell_y, points in cell_points:
        centroid_x, centroid_y = centroids_by_cell[(cell_x, cell_y)]
        payload.extend(
            _COMPONENT_CELL.pack(
                cell_x,
                cell_y,
                centroid_x,
                centroid_y,
                len(points),
            )
        )
        for point_x, point_y in points:
            payload.extend(_COMPONENT_POINT_2D.pack(point_x, point_y))
    return _ComponentMotionSample(
        stamp=float(stamp),
        centroid_x=component.centroid_x,
        centroid_y=component.centroid_y,
        cell_count=len(cell_points),
        payload=bytes(payload),
    )


def _unpack_component_motion_sample(
    sample: _ComponentMotionSample,
) -> tuple[
    frozenset[tuple[int, int]],
    dict[tuple[int, int], tuple[float, float]],
    dict[tuple[int, int], tuple[tuple[float, float], ...]],
]:
    cells: set[tuple[int, int]] = set()
    centroids: dict[tuple[int, int], tuple[float, float]] = {}
    point_sets: dict[
        tuple[int, int], tuple[tuple[float, float], ...]
    ] = {}
    offset = 0
    for _index in range(sample.cell_count):
        cell_x, cell_y, centroid_x, centroid_y, point_count = (
            _COMPONENT_CELL.unpack_from(sample.payload, offset)
        )
        offset += _COMPONENT_CELL.size
        points = []
        for _point_index in range(point_count):
            points.append(
                _COMPONENT_POINT_2D.unpack_from(sample.payload, offset)
            )
            offset += _COMPONENT_POINT_2D.size
        cell = (cell_x, cell_y)
        cells.add(cell)
        centroids[cell] = (centroid_x, centroid_y)
        point_sets[cell] = tuple(points)
    if offset != len(sample.payload):
        raise RuntimeError("component motion sample payload is inconsistent")
    return frozenset(cells), centroids, point_sets


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
    ) -> bytes:
        if (obstacle_min_height_m is None) != (obstacle_max_height_m is None):
            raise ValueError("both obstacle height limits are required")
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
        yaw = yaw_from_quaternion(robot_pose.q)
        body = bytearray(len(self._points) * _POINT.size)
        for index, point in enumerate(self._points.values()):
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
                len(self._points),
            )
        except (OverflowError, struct.error) as exc:
            raise InvalidFastLivo2Frame("encoded map header is invalid") from exc
        return header + body + metadata


class TemporalOccupancyMap:
    """Separate stable static geometry from one-frame live obstacles.

    Hits and misses are counted once per scan, regardless of point density.
    Bounded spatial shards must first remain geometrically stationary;
    moving shards stay available to the live obstacle layer but cannot
    enter this static map. Confirmed cells are removed only when later rays
    repeatedly observe them as free; merely leaving the field of view never
    erases a wall.
    """

    def __init__(
        self,
        voxel_size_m: float,
        *,
        confirmation_frames: int = 8,
        candidate_ttl_sec: float = 1.0,
        clear_miss_frames: int = 3,
        raytrace_min_range_m: float = 0.20,
        raytrace_max_range_m: float = 8.5,
        angular_bin_deg: float = 1.0,
        grid_margin_m: float = 6.0,
        max_grid_dimension_cells: int = 2048,
        max_grid_cells: int = 2_000_000,
        component_motion_window_sec: float = 0.40,
        component_history_sec: float = 0.80,
        component_motion_distance_m: float = 0.03,
        component_motion_speed_mps: float = 0.03,
        component_stationary_sec: float = 1.50,
        component_max_span_m: float = 1.00,
        component_match_distance_m: float = 0.60,
        component_min_cells: int = 1,
        max_evidence_points: int = 200_000,
        max_component_history_units: int = 1_000_000,
    ):
        values = (
            voxel_size_m,
            candidate_ttl_sec,
            raytrace_min_range_m,
            raytrace_max_range_m,
            angular_bin_deg,
            grid_margin_m,
            component_motion_window_sec,
            component_history_sec,
            component_motion_distance_m,
            component_motion_speed_mps,
            component_stationary_sec,
            component_max_span_m,
            component_match_distance_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("temporal occupancy parameters must be finite")
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        if confirmation_frames < 2:
            raise ValueError("confirmation_frames must be at least 2")
        if candidate_ttl_sec <= 0:
            raise ValueError("candidate_ttl_sec must be positive")
        if clear_miss_frames < 1:
            raise ValueError("clear_miss_frames must be positive")
        if not 0 <= raytrace_min_range_m < raytrace_max_range_m:
            raise ValueError("raytrace range is invalid")
        if not 0 < angular_bin_deg <= 10:
            raise ValueError("angular_bin_deg must be within (0, 10]")
        if grid_margin_m < voxel_size_m:
            raise ValueError("grid_margin_m must cover at least one voxel")
        if max_grid_dimension_cells < 1 or max_grid_cells < 1:
            raise ValueError("occupancy grid limits must be positive")
        if not 0 < component_motion_window_sec <= component_history_sec:
            raise ValueError("component motion/history window is invalid")
        if component_motion_distance_m <= 0 or component_motion_speed_mps <= 0:
            raise ValueError("component motion thresholds must be positive")
        if component_stationary_sec <= component_history_sec:
            raise ValueError("component stationary time must exceed history time")
        if component_max_span_m <= voxel_size_m:
            raise ValueError("component maximum span must exceed one voxel")
        if component_match_distance_m <= voxel_size_m:
            raise ValueError("component match distance must exceed one voxel")
        if component_min_cells < 1:
            raise ValueError("component_min_cells must be positive")
        if max_evidence_points < 40:
            raise ValueError("max_evidence_points must be at least 40")
        if not 40 <= max_component_history_units <= _MAX_COMPONENT_HISTORY_UNITS:
            raise ValueError(
                "max_component_history_units must be within "
                f"[40, {_MAX_COMPONENT_HISTORY_UNITS}]"
            )

        self._voxel = voxel_size_m
        self._confirmation_frames = confirmation_frames
        self._candidate_ttl_sec = candidate_ttl_sec
        self._clear_miss_frames = clear_miss_frames
        self._raytrace_min_range = raytrace_min_range_m
        self._raytrace_max_range = raytrace_max_range_m
        self._angular_bin = math.radians(angular_bin_deg)
        self._grid_margin_cells = math.ceil(grid_margin_m / voxel_size_m)
        self._max_grid_dimension_cells = int(max_grid_dimension_cells)
        self._max_grid_cells = int(max_grid_cells)
        self._component_motion_window = component_motion_window_sec
        self._component_history = component_history_sec
        self._component_motion_distance = component_motion_distance_m
        self._component_motion_speed = component_motion_speed_mps
        # A moving return can stay inside one XY voxel without changing its
        # occupied cell.  Do not admit that cell to the static map until an
        # object moving at the configured threshold would necessarily have
        # crossed a full voxel diagonal (plus one motion-comparison window).
        # The diagonal covers every entry phase and direction through a cell.
        # This is
        # deliberately based on occupancy residence rather than the raw point
        # centroid inside a cell: LiDAR sampling can move several centimetres
        # along a perfectly static wall while the occupied voxel stays fixed.
        self._component_observation_window = max(
            component_history_sec,
            component_motion_distance_m / component_motion_speed_mps,
            math.sqrt(2.0) * voxel_size_m / component_motion_speed_mps
            + component_motion_window_sec,
        )
        if (
            self._component_observation_window
            > _MAX_COMPONENT_OBSERVATION_WINDOW_SEC
        ):
            raise ValueError(
                "component observation window exceeds "
                f"{_MAX_COMPONENT_OBSERVATION_WINDOW_SEC:.0f}s safety limit"
            )
        self._component_sample_interval = 1.0 / _COMPONENT_SAMPLE_RATE_HZ
        self._component_track_sample_limit = math.ceil(
            self._component_observation_window
            / self._component_sample_interval
        ) + 2
        self._component_stationary = component_stationary_sec
        self._component_max_span = component_max_span_m
        self._component_match_distance = component_match_distance_m
        self._component_min_cells = int(component_min_cells)
        self._max_evidence_points = int(max_evidence_points)
        self._component_history_unit_limit = int(
            max_component_history_units
        )
        self._evidence: dict[
            tuple[int, int, int], _TemporalVoxelEvidence
        ] = {}
        self._candidates: set[tuple[int, int, int]] = set()
        self._xy_index: dict[tuple[int, int], set[tuple[int, int, int]]] = {}
        self._free_cells: set[tuple[int, int]] = set()
        self._bounds: tuple[int, int, int, int] | None = None
        self._scan_index = 0
        self._read_only_baseline = False
        self._component_tracks: dict[int, _TemporalComponentTrack] = {}
        self._next_component_track_id = 1
        self._dynamic_track_count = 0
        self._quarantined_point_count = 0
        self._last_observation_monotonic: float | None = None

    def clear(self) -> None:
        self.retire_and_clear()

    def retire_and_clear(self) -> tuple[object, ...]:
        """Detach the active state in O(1) so callers can release it later."""

        retired = (
            self._evidence,
            self._candidates,
            self._xy_index,
            self._free_cells,
            self._component_tracks,
        )
        self._evidence = {}
        self._candidates = set()
        self._xy_index = {}
        self._free_cells = set()
        self._bounds = None
        self._scan_index = 0
        self._read_only_baseline = False
        self._component_tracks = {}
        self._next_component_track_id = 1
        self._dynamic_track_count = 0
        self._quarantined_point_count = 0
        self._last_observation_monotonic = None
        return retired

    @property
    def point_count(self) -> int:
        return sum(1 for evidence in self._evidence.values() if evidence.confirmed)

    @property
    def candidate_count(self) -> int:
        return len(self._candidates)

    @property
    def free_cell_count(self) -> int:
        return len(self._free_cells)

    @property
    def dynamic_track_count(self) -> int:
        return self._dynamic_track_count

    @property
    def quarantined_point_count(self) -> int:
        return self._quarantined_point_count

    @property
    def component_history_unit_count(self) -> int:
        """Return bounded cell summaries retained across component tracks."""

        return sum(
            _COMPONENT_TRACK_HISTORY_UNITS
            + len(track.cells)
            + sum(sample.cell_count for sample in track.samples)
            + len(track.recent_dynamic_cells)
            + len(track.recent_keys)
            for track in self._component_tracks.values()
        )

    @property
    def component_history_unit_limit(self) -> int:
        return self._component_history_unit_limit

    @property
    def confirmed_points(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(
            evidence.point
            for key, evidence in sorted(self._evidence.items())
            if evidence.confirmed
        )

    def load_confirmed(
        self, points: Iterable[tuple[float, float, float]]
    ) -> None:
        """Replace temporal evidence with an immutable saved-map baseline."""

        self.apply_prepared_confirmed(self.prepare_confirmed(points))

    def prepare_confirmed(
        self, points: Iterable[tuple[float, float, float]]
    ) -> _PreparedConfirmedState:
        """Build all saved-map state without changing the active map."""

        evidence_by_key, xy_index, replacement_bounds = (
            self._validated_confirmed_state(points)
        )
        return _PreparedConfirmedState(
            evidence=evidence_by_key,
            xy_index=xy_index,
            bounds=replacement_bounds,
        )

    def apply_prepared_confirmed(
        self, prepared: _PreparedConfirmedState
    ) -> tuple[object, ...]:
        """Atomically activate prepared state and return detached old storage."""

        if not isinstance(prepared, _PreparedConfirmedState):
            raise TypeError("prepared confirmed state is invalid")
        retired = (
            self._evidence,
            self._candidates,
            self._xy_index,
            self._free_cells,
            self._component_tracks,
        )
        self._evidence = prepared.evidence
        self._candidates = set()
        self._xy_index = prepared.xy_index
        self._free_cells = set()
        self._bounds = prepared.bounds
        self._scan_index = 0
        self._read_only_baseline = True
        self._component_tracks = {}
        self._next_component_track_id = 1
        self._dynamic_track_count = 0
        self._quarantined_point_count = 0
        self._last_observation_monotonic = None
        return retired

    def validate_confirmed(
        self, points: Iterable[tuple[float, float, float]]
    ) -> int:
        """Validate a saved-map baseline without changing the active map."""

        return len(self.prepare_confirmed(points).evidence)

    def _validated_confirmed_state(
        self, points: Iterable[tuple[float, float, float]]
    ) -> tuple[
        dict[tuple[int, int, int], _TemporalVoxelEvidence],
        dict[tuple[int, int], set[tuple[int, int, int]]],
        tuple[int, int, int, int] | None,
    ]:
        """Build a complete immutable-map replacement before mutating state."""

        evidence_by_key: dict[
            tuple[int, int, int], _TemporalVoxelEvidence
        ] = {}
        xy_index: dict[
            tuple[int, int], set[tuple[int, int, int]]
        ] = {}
        for point in points:
            key = self._key(point)
            if key is None or key in evidence_by_key:
                continue
            normalized = tuple(float(value) for value in point)
            evidence_by_key[key] = _TemporalVoxelEvidence(
                point=normalized,
                hit_frames=self._confirmation_frames,
                first_hit_monotonic=0.0,
                first_hit_point=normalized,
                last_hit_monotonic=0.0,
                last_hit_scan=0,
                confirmed=True,
            )
            xy = key[:2]
            xy_index.setdefault(xy, set()).add(key)
            if len(evidence_by_key) > self._max_evidence_points:
                raise ValueError(
                    "confirmed static map exceeds "
                    f"{self._max_evidence_points} point safety limit"
                )
        # The confirmed map remains sparse and may span a large site.  Dense
        # OccupancyGrid publication is a rolling window around the active
        # pose, so the full point extent must not become an allocation bound.
        replacement_bounds = None
        return evidence_by_key, xy_index, replacement_bounds

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
        component_points: dict[
            tuple[int, int], list[tuple[float, float, float]]
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
            component_points.setdefault(key[:2], []).append(normalized)
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

        self._scan_index += 1
        scan_index = self._scan_index
        eligible_keys, quarantined_keys, purge_keys = self._component_gate(
            observed,
            component_points=component_points,
            now_monotonic=now_monotonic,
        )
        for key in purge_keys:
            evidence = self._evidence.get(key)
            if evidence is None:
                continue
            current_point = observed.get(key)
            if (
                current_point is not None
                and math.dist(current_point, evidence.first_hit_point)
                <= min(
                    self._voxel * 0.15,
                    self._component_motion_distance * 0.50,
                )
            ):
                continue
            self._remove(key)
        for key, point in observed.items():
            if key in quarantined_keys:
                prior = self._evidence.get(key)
                if (
                    prior is None
                    or math.dist(point, prior.first_hit_point)
                    > min(
                        self._voxel * 0.15,
                        self._component_motion_distance * 0.50,
                    )
                ):
                    continue
            evidence = self._evidence.get(key)
            if evidence is None:
                evidence = _TemporalVoxelEvidence(
                    point=point,
                    hit_frames=1,
                    first_hit_monotonic=now_monotonic,
                    first_hit_point=point,
                    last_hit_monotonic=now_monotonic,
                    last_hit_scan=scan_index,
                )
                self._evidence[key] = evidence
                self._candidates.add(key)
                self._xy_index.setdefault(key[:2], set()).add(key)
            else:
                evidence.point = point
                if evidence.confirmed:
                    evidence.hit_frames = self._confirmation_frames
                elif (
                    evidence.last_hit_scan == scan_index - 1
                    and now_monotonic - evidence.last_hit_monotonic
                    <= self._candidate_ttl_sec
                ):
                    evidence.hit_frames += 1
                else:
                    evidence.hit_frames = 1
                    evidence.first_hit_monotonic = now_monotonic
                    evidence.first_hit_point = point
                evidence.last_hit_monotonic = now_monotonic
                evidence.last_hit_scan = scan_index
                evidence.free_miss_frames = 0
            if (
                evidence.hit_frames >= self._confirmation_frames
                and key in eligible_keys
                and now_monotonic - evidence.first_hit_monotonic
                >= self._component_observation_window
            ):
                evidence.confirmed = True
                self._candidates.discard(key)

        for xy in ray_free:
            for key in tuple(self._xy_index.get(xy, ())):
                evidence = self._evidence.get(key)
                if evidence is None:
                    continue
                if not obstacle_min_height_m <= evidence.point[2] <= obstacle_max_height_m:
                    continue
                evidence.hit_frames = 0
                evidence.free_miss_frames += 1
                if evidence.free_miss_frames >= self._clear_miss_frames:
                    self._remove(key)

        for key in tuple(self._candidates):
            evidence = self._evidence.get(key)
            if evidence is None:
                self._candidates.discard(key)
                continue
            if now_monotonic - evidence.last_hit_monotonic > self._candidate_ttl_sec:
                self._remove(key)

        self._free_cells.update(ray_free)
        confirmed_xy = {
            key[:2]
            for key, evidence in self._evidence.items()
            if evidence.confirmed
            and obstacle_min_height_m <= evidence.point[2] <= obstacle_max_height_m
        }
        self._free_cells.difference_update(confirmed_xy)

    def expire(self, *, now_monotonic: float) -> None:
        if not math.isfinite(now_monotonic):
            raise ValueError("now_monotonic must be finite")
        for key in tuple(self._candidates):
            evidence = self._evidence.get(key)
            if evidence is None:
                self._candidates.discard(key)
                continue
            if (
                now_monotonic - evidence.last_hit_monotonic
                > self._candidate_ttl_sec
            ):
                self._remove(key)

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
        for key, evidence in self._evidence.items():
            if evidence.confirmed and min_z <= evidence.point[2] <= max_z:
                ix, iy, _ = key
                projected.setdefault(
                    (ix, iy),
                    ((ix + 0.5) * self._voxel, (iy + 0.5) * self._voxel, output_z),
                )
        return tuple(projected[key] for key in sorted(projected))

    def as_voxel_map(self) -> VoxelMap:
        output = VoxelMap(self._voxel)
        output.add(self.confirmed_points)
        return output

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
            bounds=self._bounds,
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
            bounds=prepared.bounds,
            center_x=center_x,
            center_y=center_y,
            min_z=min_z,
            max_z=max_z,
        )
        return snapshot

    def _build_occupancy_snapshot(
        self,
        *,
        evidence: dict[tuple[int, int, int], _TemporalVoxelEvidence],
        free_cells: Iterable[tuple[int, int]],
        bounds: tuple[int, int, int, int] | None,
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
            if item.confirmed and min_z <= item.point[2] <= max_z
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

    def _remove(self, key: tuple[int, int, int]) -> None:
        self._evidence.pop(key, None)
        self._candidates.discard(key)
        xy = key[:2]
        keys = self._xy_index.get(xy)
        if keys is None:
            return
        keys.discard(key)
        if not keys:
            self._xy_index.pop(xy, None)

    def _component_gate(
        self,
        observed: dict[
            tuple[int, int, int], tuple[float, float, float]
        ],
        *,
        component_points: dict[
            tuple[int, int], list[tuple[float, float, float]]
        ],
        now_monotonic: float,
    ) -> tuple[
        set[tuple[int, int, int]],
        set[tuple[int, int, int]],
        set[tuple[int, int, int]],
    ]:
        """Return static-eligible, quarantined, and candidate-purge keys.

        Large connected structures are split into bounded spatial shards so a
        person touching a wall cannot bypass motion qualification. Motion uses
        common-cell point displacement before whole-shard centroid movement,
        which keeps a changing visible subset of one wall stationary.
        """

        eligible = set(observed)
        quarantined: set[tuple[int, int, int]] = set()
        purge: set[tuple[int, int, int]] = set()
        observations = [
            component
            for component in self._connected_components(
                observed,
                component_points=component_points,
            )
            if self._track_component(component)
        ]
        for component in observations:
            eligible.difference_update(component.keys)

        stale_before = now_monotonic - max(
            self._candidate_ttl_sec,
            self._component_stationary,
            self._component_observation_window,
        )
        for track_id in tuple(self._component_tracks):
            if self._component_tracks[track_id].last_seen_monotonic < stale_before:
                self._component_tracks.pop(track_id, None)

        bucket_size = self._component_match_distance
        buckets: dict[tuple[int, int], list[int]] = {}
        for track_id, track in self._component_tracks.items():
            bucket = (
                math.floor(track.centroid_x / bucket_size),
                math.floor(track.centroid_y / bucket_size),
            )
            buckets.setdefault(bucket, []).append(track_id)

        history_units = self.component_history_unit_count
        used_tracks: set[int] = set()
        for component in sorted(observations, key=lambda item: -len(item.cells)):
            bucket_x = math.floor(component.centroid_x / bucket_size)
            bucket_y = math.floor(component.centroid_y / bucket_size)
            matches: list[tuple[float, float, int]] = []
            for delta_x in (-1, 0, 1):
                for delta_y in (-1, 0, 1):
                    for track_id in buckets.get(
                        (bucket_x + delta_x, bucket_y + delta_y), ()
                    ):
                        if track_id in used_tracks:
                            continue
                        track = self._component_tracks[track_id]
                        distance = math.hypot(
                            component.centroid_x - track.centroid_x,
                            component.centroid_y - track.centroid_y,
                        )
                        if distance > self._component_match_distance:
                            continue
                        overlap = len(component.cells.intersection(track.cells))
                        overlap_ratio = overlap / max(
                            1,
                            min(len(component.cells), len(track.cells)),
                        )
                        if overlap == 0 and distance > bucket_size * 0.5:
                            continue
                        matches.append((-overlap_ratio, distance, track_id))

            if matches:
                _overlap, _distance, track_id = min(matches)
                track = self._component_tracks[track_id]
                used_tracks.add(track_id)
            else:
                track_units = (
                    _COMPONENT_TRACK_HISTORY_UNITS + len(component.cells)
                )
                if (
                    history_units + track_units
                    > self._component_history_unit_limit
                ):
                    # Do not allocate an unbounded population of one-cell
                    # tracks.  The current component remains withheld and its
                    # existing evidence is removed conservatively.
                    quarantined.update(component.keys)
                    purge.update(component.keys)
                    continue
                track_id = self._next_component_track_id
                self._next_component_track_id += 1
                track = _TemporalComponentTrack(
                    track_id=track_id,
                    centroid_x=component.centroid_x,
                    centroid_y=component.centroid_y,
                    first_seen_monotonic=now_monotonic,
                    last_seen_monotonic=now_monotonic,
                    cells=component.cells,
                    samples=[],
                )
                self._component_tracks[track_id] = track
                used_tracks.add(track_id)
                history_units += track_units

            track.centroid_x = component.centroid_x
            track.centroid_y = component.centroid_y
            track.last_seen_monotonic = now_monotonic
            cell_unit_delta = len(component.cells) - len(track.cells)
            if (
                history_units + cell_unit_delta
                <= self._component_history_unit_limit
            ):
                track.cells = component.cells
                history_units += cell_unit_delta
            else:
                track.history_saturated = True
            sample_interval = self._component_sample_interval
            motion_stale_before = (
                now_monotonic - self._component_observation_window
            )
            while (
                len(track.samples) > 1
                and track.samples[1].stamp <= motion_stale_before
            ):
                history_units -= track.samples.pop(0).cell_count
            expired_dynamic_cells = [
                cell
                for cell, stamp in track.recent_dynamic_cells.items()
                if stamp < motion_stale_before
            ]
            for cell in expired_dynamic_cells:
                track.recent_dynamic_cells.pop(cell, None)
                history_units -= 1
            expired_keys = [
                key
                for key, stamp in track.recent_keys.items()
                if stamp < motion_stale_before
            ]
            for key in expired_keys:
                track.recent_keys.pop(key, None)
                history_units -= 1

            sample_due = (
                not track.samples
                or now_monotonic - track.samples[-1].stamp
                >= sample_interval - 1e-9
            )
            if sample_due:
                current_sample = _pack_component_motion_sample(
                    component,
                    now_monotonic,
                )
                if (
                    history_units + current_sample.cell_count
                    > self._component_history_unit_limit
                ):
                    # Reset only this track.  Until a complete observation
                    # window is rebuilt it remains ineligible for the static
                    # map, so pressure cannot turn missing motion evidence
                    # into a persistent obstacle.
                    history_units -= sum(
                        sample.cell_count for sample in track.samples
                    )
                    track.samples.clear()
                    track.history_saturated = True
                if (
                    history_units + current_sample.cell_count
                    <= self._component_history_unit_limit
                ):
                    track.samples.append(current_sample)
                    history_units += current_sample.cell_count
                else:
                    track.history_saturated = True
            elif track.samples:
                replacement = _pack_component_motion_sample(
                    component,
                    track.samples[-1].stamp,
                )
                previous_units = track.samples[-1].cell_count
                projected_units = (
                    history_units
                    - previous_units
                    + replacement.cell_count
                )
                if projected_units <= self._component_history_unit_limit:
                    track.samples[-1] = replacement
                    history_units = projected_units
                else:
                    history_units -= sum(
                        sample.cell_count for sample in track.samples
                    )
                    track.samples.clear()
                    track.history_saturated = True

            new_component_keys = component.keys.difference(track.recent_keys)
            if (
                history_units + len(new_component_keys)
                > self._component_history_unit_limit
            ):
                history_units -= sum(
                    sample.cell_count for sample in track.samples
                )
                track.samples.clear()
                track.history_saturated = True
            if (
                history_units + len(new_component_keys)
                <= self._component_history_unit_limit
            ):
                for key in component.keys:
                    if key not in track.recent_keys:
                        history_units += 1
                    track.recent_keys[key] = now_monotonic
            else:
                track.history_saturated = True

            if len(track.samples) > self._component_track_sample_limit:
                removed = track.samples[
                    : -self._component_track_sample_limit
                ]
                history_units -= sum(
                    sample.cell_count for sample in removed
                )
                track.samples = track.samples[
                    -self._component_track_sample_limit :
                ]

            reference = next(
                (
                    sample
                    for sample in track.samples
                    if now_monotonic - sample.stamp
                    >= self._component_motion_window
                ),
                None,
            )
            motion_cells: frozenset[tuple[int, int]] = frozenset()
            if reference is not None:
                elapsed = now_monotonic - reference.stamp
                motion_cells = self._component_motion_cells(
                    component,
                    reference=reference,
                    elapsed=elapsed,
                )
            motion_now = bool(motion_cells)

            if motion_now:
                track.motion_votes = min(2, track.motion_votes + 1)
                new_motion_cells = motion_cells.difference(
                    track.recent_dynamic_cells
                )
                if (
                    history_units + len(new_motion_cells)
                    > self._component_history_unit_limit
                ):
                    history_units -= sum(
                        sample.cell_count for sample in track.samples
                    )
                    track.samples.clear()
                    track.history_saturated = True
                if (
                    history_units + len(new_motion_cells)
                    <= self._component_history_unit_limit
                ):
                    for cell in motion_cells:
                        if cell not in track.recent_dynamic_cells:
                            history_units += 1
                        track.recent_dynamic_cells[cell] = now_monotonic
                else:
                    # Current motion is still quarantined below even when the
                    # bounded history cannot retain its complete footprint.
                    track.history_saturated = True
                    track.dynamic = True
                    track.last_motion_monotonic = now_monotonic
                if track.motion_votes >= 2:
                    track.dynamic = True
                    track.last_motion_monotonic = now_monotonic
            else:
                track.motion_votes = 0

            history_ready = bool(track.samples) and (
                now_monotonic - track.samples[0].stamp
                >= self._component_observation_window - 1e-9
            )
            if history_ready:
                track.history_saturated = False
            if (
                track.dynamic
                and not motion_now
                and not track.history_saturated
                and track.last_motion_monotonic is not None
                and now_monotonic - track.last_motion_monotonic
                >= self._component_stationary
            ):
                reset_sample = _pack_component_motion_sample(
                    component,
                    now_monotonic,
                )
                replacement_units = reset_sample.cell_count
                retained_units = history_units - sum(
                    sample.cell_count for sample in track.samples
                ) - len(track.recent_dynamic_cells) - len(track.recent_keys)
                replacement_units += len(component.keys)
                if (
                    retained_units + replacement_units
                    <= self._component_history_unit_limit
                ):
                    history_units = retained_units + replacement_units
                    track.dynamic = False
                    track.motion_votes = 0
                    track.first_seen_monotonic = now_monotonic
                    track.samples = [reset_sample]
                    track.recent_dynamic_cells.clear()
                    track.recent_keys = {
                        key: now_monotonic for key in component.keys
                    }
                    track.history_saturated = False
                    history_ready = False

            dynamic_cells = set(track.recent_dynamic_cells)
            if track.history_saturated:
                # Fail closed: without a complete bounded history, withhold
                # the whole current shard instead of admitting an unobserved
                # moving object to the persistent map.
                dynamic_cells.update(component.cells)
            if track.dynamic or track.history_saturated:
                # Only remove the local cells that supplied motion evidence.
                # A person can be connected to a wall in the 2D grid; purging
                # the whole connected shard would punch holes in that wall.
                quarantined.update(
                    key for key in component.keys if key[:2] in dynamic_cells
                )
                purge.update(
                    key
                    for key in (
                        set(track.recent_keys)
                        | (set(component.keys) if track.history_saturated else set())
                    )
                    if key[:2] in dynamic_cells
                )
            # Admission belongs to each occupied voxel, not to the lifetime of
            # the connected component track.  A moving person can repeatedly
            # quantize between motion/stationary while touching a wall; using
            # track age would reset the wall's static observation every time.
            # Stable wall voxels retain their own first-hit clock, while cells
            # implicated by recent motion remain quarantined.
            eligible.update(
                key
                for key in component.keys
                if key[:2] not in dynamic_cells
                and not track.history_saturated
                and (evidence := self._evidence.get(key)) is not None
                and now_monotonic - evidence.first_hit_monotonic
                >= self._component_observation_window
            )

        self._dynamic_track_count = sum(
            1 for track in self._component_tracks.values() if track.dynamic
        )
        self._quarantined_point_count = len(quarantined)
        return eligible, quarantined, purge

    def _component_motion_cells(
        self,
        component: _ComponentObservation,
        *,
        reference: _ComponentMotionSample,
        elapsed: float,
    ) -> frozenset[tuple[int, int]]:
        """Return a local motion footprint without erasing connected walls."""

        if elapsed <= 0:
            return frozenset()
        required_displacement = self._component_motion_distance
        reference_cells, previous_cells, previous_point_sets = (
            _unpack_component_motion_sample(reference)
        )

        # The packed cell set is sufficient for the overwhelmingly common
        # static case.  Avoid rebuilding detailed geometry for every cell in
        # every frame when occupancy has not changed.
        if component.cells == reference_cells:
            return frozenset()

        cell_centroids, cell_points = _component_cell_geometry(component)
        current_cells = {
            (cell_x, cell_y): (point_x, point_y)
            for cell_x, cell_y, point_x, point_y in cell_centroids
        }
        current_point_sets = {
            (cell_x, cell_y): points
            for cell_x, cell_y, points in cell_points
        }
        common_cells = current_cells.keys() & previous_cells.keys()

        anchor_tolerance = min(
            self._voxel * 0.15,
            self._component_motion_distance * 0.50,
        )
        anchored_common: set[tuple[int, int]] = set()
        moving_common: set[tuple[int, int]] = set()
        for cell in common_cells:
            current = current_cells[cell]
            previous = previous_cells[cell]
            displacement = math.hypot(
                current[0] - previous[0],
                current[1] - previous[1],
            )
            current_points = current_point_sets.get(cell, ())
            previous_points = previous_point_sets.get(cell, ())
            anchored = any(
                math.hypot(
                    current_point[0] - previous_point[0],
                    current_point[1] - previous_point[1],
                )
                <= anchor_tolerance
                for current_point in current_points
                for previous_point in previous_points
            )
            if anchored:
                anchored_common.add(cell)
            elif displacement + 1e-9 >= required_displacement:
                moving_common.add(cell)
        if moving_common:
            footprint = set(moving_common)
            changed = component.cells.symmetric_difference(reference_cells)
            # Include turnover cells immediately next to the measured motion,
            # but not the rest of a connected wall.
            for cell in component.cells.union(reference_cells):
                if cell not in anchored_common and any(
                    max(abs(cell[0] - moving[0]), abs(cell[1] - moving[1]))
                    <= 1
                    for moving in moving_common
                ):
                    footprint.add(cell)
            for cell in changed:
                if cell not in anchored_common and any(
                    max(abs(cell[0] - moving[0]), abs(cell[1] - moving[1]))
                    <= 2
                    for moving in moving_common
                ):
                    footprint.add(cell)
            return frozenset(footprint)

        # With no common-cell evidence, accept only a clear component
        # translation.  A changing visible subset of one wall retains high
        # overlap and is deliberately not treated as a moving object.
        displacement = math.hypot(
            component.centroid_x - reference.centroid_x,
            component.centroid_y - reference.centroid_y,
        )
        if (
            displacement + 1e-9 < required_displacement
            or displacement / elapsed + 1e-9 < self._component_motion_speed
        ):
            return frozenset()
        overlap = len(component.cells.intersection(reference_cells))
        overlap_ratio = overlap / max(
            1,
            min(len(component.cells), len(reference_cells)),
        )
        all_cells = component.cells.union(reference_cells)
        minimum_x = min(cell[0] for cell in all_cells)
        maximum_x = max(cell[0] for cell in all_cells)
        minimum_y = min(cell[1] for cell in all_cells)
        maximum_y = max(cell[1] for cell in all_cells)
        width = maximum_x - minimum_x + 1
        height = maximum_y - minimum_y + 1
        compact_shape = (
            min(width, height) >= 3
            and max(width, height) / min(width, height) <= 4.0
        )
        if common_cells and overlap_ratio >= 0.50 and not compact_shape:
            return frozenset()
        return frozenset(all_cells.difference(anchored_common))

    def _connected_components(
        self,
        observed: dict[
            tuple[int, int, int], tuple[float, float, float]
        ],
        *,
        component_points: dict[
            tuple[int, int], list[tuple[float, float, float]]
        ],
    ) -> tuple[_ComponentObservation, ...]:
        keys_by_cell: dict[
            tuple[int, int], list[tuple[int, int, int]]
        ] = {}
        for key in observed:
            keys_by_cell.setdefault(key[:2], []).append(key)
        remaining = set(keys_by_cell)
        components: list[_ComponentObservation] = []
        neighbours = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        )
        while remaining:
            first = remaining.pop()
            cells = {first}
            stack = [first]
            while stack:
                cell_x, cell_y = stack.pop()
                for delta_x, delta_y in neighbours:
                    neighbour = (cell_x + delta_x, cell_y + delta_y)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        cells.add(neighbour)
                        stack.append(neighbour)
            minimum_x = min(cell[0] for cell in cells)
            maximum_x = max(cell[0] for cell in cells)
            minimum_y = min(cell[1] for cell in cells)
            maximum_y = max(cell[1] for cell in cells)
            tile_cells = max(1, math.floor(self._component_max_span / self._voxel))
            if (
                maximum_x - minimum_x + 1 <= tile_cells
                and maximum_y - minimum_y + 1 <= tile_cells
            ):
                shards = (cells,)
            else:
                by_tile: dict[tuple[int, int], set[tuple[int, int]]] = {}
                for cell in cells:
                    tile = (cell[0] // tile_cells, cell[1] // tile_cells)
                    by_tile.setdefault(tile, set()).add(cell)
                shards = tuple(by_tile.values())
            for shard in shards:
                keys = frozenset(
                    key for cell in shard for key in keys_by_cell[cell]
                )
                point_count = sum(
                    len(component_points[cell]) for cell in shard
                )
                centroid_x = sum(
                    point[0]
                    for cell in shard
                    for point in component_points[cell]
                ) / point_count
                centroid_y = sum(
                    point[1]
                    for cell in shard
                    for point in component_points[cell]
                ) / point_count
                shard_minimum_x = min(cell[0] for cell in shard)
                shard_maximum_x = max(cell[0] for cell in shard)
                shard_minimum_y = min(cell[1] for cell in shard)
                shard_maximum_y = max(cell[1] for cell in shard)
                components.append(
                    _ComponentObservation(
                        keys=keys,
                        cells=frozenset(shard),
                        point_source=component_points,
                        centroid_x=centroid_x,
                        centroid_y=centroid_y,
                        span_x=(shard_maximum_x - shard_minimum_x + 1)
                        * self._voxel,
                        span_y=(shard_maximum_y - shard_minimum_y + 1)
                        * self._voxel,
                    )
                )
        return tuple(components)

    def _track_component(self, component: _ComponentObservation) -> bool:
        if len(component.cells) < self._component_min_cells:
            return False
        longest = max(component.span_x, component.span_y)
        return longest <= self._component_max_span

    def _expand_bounds(self, cells: Iterable[tuple[int, int]]) -> None:
        expanded = self._expanded_bounds(cells)
        if expanded is not None:
            self._bounds = expanded

    def _expanded_bounds(
        self, cells: Iterable[tuple[int, int]]
    ) -> tuple[int, int, int, int] | None:
        return self._expanded_bounds_from(self._bounds, cells)

    def _expanded_bounds_from(
        self,
        bounds: tuple[int, int, int, int] | None,
        cells: Iterable[tuple[int, int]],
    ) -> tuple[int, int, int, int] | None:
        cells = tuple(cells)
        if not cells:
            return bounds
        minimum_x = min(cell[0] for cell in cells) - self._grid_margin_cells
        maximum_x = max(cell[0] for cell in cells) + self._grid_margin_cells
        minimum_y = min(cell[1] for cell in cells) - self._grid_margin_cells
        maximum_y = max(cell[1] for cell in cells) + self._grid_margin_cells
        if bounds is None:
            expanded = (minimum_x, maximum_x, minimum_y, maximum_y)
        else:
            old_min_x, old_max_x, old_min_y, old_max_y = bounds
            expanded = (
                min(old_min_x, minimum_x),
                max(old_max_x, maximum_x),
                min(old_min_y, minimum_y),
                max(old_max_y, maximum_y),
            )
        width = expanded[1] - expanded[0] + 1
        height = expanded[3] - expanded[2] + 1
        if (
            width > self._max_grid_dimension_cells
            or height > self._max_grid_dimension_cells
            or width * height > self._max_grid_cells
        ):
            raise ValueError(
                "occupancy grid exceeds configured dimension or cell-count limit"
            )
        return expanded

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
