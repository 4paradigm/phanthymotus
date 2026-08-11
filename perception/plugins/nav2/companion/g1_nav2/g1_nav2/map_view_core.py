"""ROS-independent conversion from OccupancyGrid data to Canvas mapping frames."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Sequence


CANVAS_MAPPING_MAX_POINTS = 80_000
_FRAME_HEADER = struct.Struct("<fffBI")
_POINT = struct.Struct("<fff")
_FULL_MAP_WITH_Z = 0x01 | 0x02


class InvalidMapView(ValueError):
    """Raised when a map cannot be represented without misleading Canvas."""


@dataclass(frozen=True)
class CanvasMapSnapshot:
    """Cached map point body; the robot pose is added to each live frame."""

    point_body: bytes
    point_count: int
    occupied_cell_count: int


def build_occupancy_snapshot(
    *,
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    origin_yaw: float,
    data: Sequence[int],
    occupancy_threshold: int = 65,
    max_points: int = CANVAS_MAPPING_MAX_POINTS,
) -> CanvasMapSnapshot:
    """Convert occupied cell centers into an evenly sampled XYZ point body."""

    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise InvalidMapView("map width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise InvalidMapView("map height must be a positive integer")
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise InvalidMapView("map resolution must be finite and positive")
    if not all(math.isfinite(value) for value in (origin_x, origin_y, origin_yaw)):
        raise InvalidMapView("map origin must be finite")
    if not isinstance(occupancy_threshold, int) or not 0 <= occupancy_threshold <= 100:
        raise InvalidMapView("occupancy threshold must be an integer in [0, 100]")
    if not isinstance(max_points, int) or not 1 <= max_points <= CANVAS_MAPPING_MAX_POINTS:
        raise InvalidMapView(
            f"max_points must be in [1, {CANVAS_MAPPING_MAX_POINTS}]"
        )
    if len(data) != width * height:
        raise InvalidMapView(
            f"map data size mismatch: {len(data)} != {width} * {height}"
        )

    occupied_indices: list[int] = []
    for index, value in enumerate(data):
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidMapView("map occupancy values must be integers")
        if value < -1 or value > 100:
            raise InvalidMapView("map occupancy values must be in [-1, 100]")
        if value >= occupancy_threshold:
            occupied_indices.append(index)

    occupied_cell_count = len(occupied_indices)
    sample_stride = max(1, math.ceil(occupied_cell_count / max_points))
    selected_indices = occupied_indices[::sample_stride][:max_points]
    point_body = bytearray(len(selected_indices) * _POINT.size)
    cos_yaw = math.cos(origin_yaw)
    sin_yaw = math.sin(origin_yaw)

    for point_index, cell_index in enumerate(selected_indices):
        row, column = divmod(cell_index, width)
        grid_x = (column + 0.5) * resolution
        grid_y = (row + 0.5) * resolution
        world_x = origin_x + cos_yaw * grid_x - sin_yaw * grid_y
        world_y = origin_y + sin_yaw * grid_x + cos_yaw * grid_y
        _POINT.pack_into(
            point_body,
            point_index * _POINT.size,
            world_x,
            world_y,
            0.0,
        )

    return CanvasMapSnapshot(
        point_body=bytes(point_body),
        point_count=len(selected_indices),
        occupied_cell_count=occupied_cell_count,
    )


def encode_canvas_mapping_frame(
    snapshot: CanvasMapSnapshot,
    *,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
) -> bytes:
    """Prefix the cached map with the robot pose expected by MappingRenderer."""

    if not isinstance(snapshot, CanvasMapSnapshot):
        raise InvalidMapView("snapshot must be a CanvasMapSnapshot")
    if snapshot.point_count < 0 or snapshot.point_count > CANVAS_MAPPING_MAX_POINTS:
        raise InvalidMapView("snapshot point count is out of bounds")
    if len(snapshot.point_body) != snapshot.point_count * _POINT.size:
        raise InvalidMapView("snapshot point body size is inconsistent")
    if not all(math.isfinite(value) for value in (robot_x, robot_y, robot_yaw)):
        raise InvalidMapView("robot pose must be finite")

    return _FRAME_HEADER.pack(
        robot_x,
        robot_y,
        robot_yaw,
        _FULL_MAP_WITH_Z,
        snapshot.point_count,
    ) + snapshot.point_body


__all__ = [
    "CANVAS_MAPPING_MAX_POINTS",
    "CanvasMapSnapshot",
    "InvalidMapView",
    "build_occupancy_snapshot",
    "encode_canvas_mapping_frame",
]
