"""Persistent collection post-processing and Canvas progress publication."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
from typing import Callable, Iterable

import numpy as np


ANNOTATION_SCHEMA = "phanthy.navigation.obstacle_frame.v1"
POSTPROCESS_SCHEMA = "phanthy.navigation.collection_postprocess.v1"
_SYNC_TOLERANCE_NS = {
    "lidar": 60_000_000,
    "imu": 20_000_000,
    "odom": 120_000_000,
}
_MAX_PENDING_IMAGES = 32
_MAX_PENDING_IMAGE_BYTES = 64 * 1024 * 1024
_MIN_CLUSTER_POINTS = 4
_GROUND_DISTANCE_M = 0.08
_GROUND_NORMAL_ANGLE_RAD = math.radians(15.0)
_DOWNSAMPLE_M = 0.05
_CLUSTER_CELL_M = 0.15


class PostprocessError(RuntimeError):
    pass


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _camera_parameters(metadata: dict) -> dict:
    try:
        intrinsics = metadata["intrinsics"]
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        width = int(metadata["width"])
        height = int(metadata["height"])
        coefficients = [float(value) for value in intrinsics.get("coefficients", [])]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise PostprocessError("camera_intrinsics_invalid") from exc
    values = [fx, fy, cx, cy, *coefficients]
    if (
        fx <= 0.0
        or fy <= 0.0
        or width <= 0
        or height <= 0
        or not all(math.isfinite(value) for value in values)
    ):
        raise PostprocessError("camera_intrinsics_invalid")
    return {
        "calibration_id": metadata.get("calibration_id"),
        "frame_id": metadata.get("frame_id"),
        "width_px": width,
        "height_px": height,
        "fx_px": fx,
        "fy_px": fy,
        "cx_px": cx,
        "cy_px": cy,
        "equivalent_focal_length_px": round(math.sqrt(fx * fy), 6),
        "distortion_model": str(intrinsics.get("distortion_model", "none")),
        "distortion_coefficients": coefficients,
    }


def _pcd_bytes(points: np.ndarray) -> bytes:
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise PostprocessError("lidar_points_must_be_n_by_3")
    cloud = cloud[np.isfinite(cloud).all(axis=1)].astype("<f4", copy=False)
    point_count = int(len(cloud))
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {point_count}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {point_count}\n"
        "DATA binary\n"
    ).encode("ascii")
    return header + cloud.tobytes(order="C")


def _matrix(value, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (16,) or not np.isfinite(array).all():
        raise PostprocessError(f"{field} must contain 16 finite numbers")
    result = array.reshape(4, 4)
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise PostprocessError(f"{field} is not a homogeneous transform")
    return result


def _quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-9:
        raise PostprocessError("odometry quaternion is invalid")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64)
    return source @ transform[:3, :3].T + transform[:3, 3]


def _voxel_downsample(points: np.ndarray, size: float = _DOWNSAMPLE_M) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    cells = np.floor(np.asarray(points, dtype=np.float64) / float(size)).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    return np.asarray(points, dtype=np.float64)[np.sort(indices)]


def remove_ground(points: np.ndarray, gravity: np.ndarray) -> np.ndarray:
    """Remove the dominant gravity-aligned plane using bounded deterministic RANSAC."""

    cloud = np.asarray(points, dtype=np.float64)
    g = np.asarray(gravity, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < 12:
        raise PostprocessError("ground_plane_unavailable")
    g_norm = float(np.linalg.norm(g))
    if not math.isfinite(g_norm) or g_norm < 1e-6:
        raise PostprocessError("gravity_unavailable")
    g /= g_norm
    sampled = cloud
    if len(sampled) > 4000:
        sampled = sampled[np.linspace(0, len(sampled) - 1, 4000, dtype=np.int64)]
    rng = np.random.default_rng(0)
    best_normal = None
    best_offset = None
    best_count = 0
    cosine_limit = math.cos(_GROUND_NORMAL_ANGLE_RAD)
    for _ in range(96):
        chosen = sampled[rng.choice(len(sampled), size=3, replace=False)]
        normal = np.cross(chosen[1] - chosen[0], chosen[2] - chosen[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            continue
        normal /= norm
        if abs(float(np.dot(normal, g))) < cosine_limit:
            continue
        offset = -float(np.dot(normal, chosen[0]))
        count = int(np.count_nonzero(np.abs(sampled @ normal + offset) <= _GROUND_DISTANCE_M))
        if count > best_count:
            best_normal, best_offset, best_count = normal, offset, count
    if best_normal is None or best_count < max(10, int(len(sampled) * 0.03)):
        raise PostprocessError("ground_plane_unavailable")
    distance = np.abs(cloud @ best_normal + float(best_offset))
    return cloud[distance > _GROUND_DISTANCE_M]


def cluster_points(points: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Cluster a downsampled cloud using a 26-neighbour spatial hash."""

    cloud = np.asarray(points, dtype=np.float64)
    if len(cloud) == 0:
        return np.empty((0,), dtype=np.int32), []
    cells = np.floor(cloud / _CLUSTER_CELL_M).astype(np.int64)
    cell_members: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(cells):
        cell_members.setdefault(tuple(int(v) for v in cell), []).append(index)
    labels = np.full(len(cloud), -1, dtype=np.int32)
    clusters = []
    visited_cells = set()
    neighbours = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
    for start in cell_members:
        if start in visited_cells:
            continue
        pending = [start]
        visited_cells.add(start)
        indices = []
        while pending:
            cell = pending.pop()
            indices.extend(cell_members[cell])
            for delta in neighbours:
                candidate = tuple(cell[i] + delta[i] for i in range(3))
                if candidate in cell_members and candidate not in visited_cells:
                    visited_cells.add(candidate)
                    pending.append(candidate)
        if len(indices) < _MIN_CLUSTER_POINTS:
            continue
        cluster_index = len(clusters)
        array = np.asarray(indices, dtype=np.int64)
        labels[array] = cluster_index
        clusters.append(array)
    return labels, clusters


def visible_cluster_points(
    points_camera: np.ndarray,
    labels: np.ndarray,
    metadata: dict,
) -> dict[int, np.ndarray]:
    intrinsics = metadata["intrinsics"]
    width, height = int(metadata["width"]), int(metadata["height"])
    points = np.asarray(points_camera, dtype=np.float64)
    z = points[:, 2]
    positive = z > 0.05
    safe_z = np.where(positive, z, 1.0)
    normalized_x = points[:, 0] / safe_z
    normalized_y = points[:, 1] / safe_z
    coefficients = list(intrinsics.get("coefficients", []))
    model = str(intrinsics.get("distortion_model", "none"))
    if model in {"plumb_bob", "brown_conrady", "rational_polynomial"}:
        k1, k2, p1, p2, k3 = coefficients[:5]
        radius2 = normalized_x * normalized_x + normalized_y * normalized_y
        radius4 = radius2 * radius2
        radius6 = radius4 * radius2
        radial = 1.0 + k1 * radius2 + k2 * radius4 + k3 * radius6
        if model == "rational_polynomial":
            k4, k5, k6 = coefficients[5:8]
            denominator = 1.0 + k4 * radius2 + k5 * radius4 + k6 * radius6
            radial = np.divide(
                radial,
                denominator,
                out=np.full_like(radial, np.nan),
                where=np.abs(denominator) > 1e-12,
            )
        distorted_x = (
            normalized_x * radial
            + 2.0 * p1 * normalized_x * normalized_y
            + p2 * (radius2 + 2.0 * normalized_x * normalized_x)
        )
        distorted_y = (
            normalized_y * radial
            + p1 * (radius2 + 2.0 * normalized_y * normalized_y)
            + 2.0 * p2 * normalized_x * normalized_y
        )
    elif model in {
        "inverse_brown_conrady",
        "realsense_inverse_brown_conrady",
    }:
        k1, k2, p1, p2, k3 = coefficients[:5]
        distorted_x = normalized_x.copy()
        distorted_y = normalized_y.copy()
        for _ in range(10):
            radius2 = distorted_x * distorted_x + distorted_y * distorted_y
            radius4 = radius2 * radius2
            radius6 = radius4 * radius2
            radial = 1.0 + k1 * radius2 + k2 * radius4 + k3 * radius6
            delta_x = (
                2.0 * p1 * distorted_x * distorted_y
                + p2 * (radius2 + 2.0 * distorted_x * distorted_x)
            )
            delta_y = (
                p1 * (radius2 + 2.0 * distorted_y * distorted_y)
                + 2.0 * p2 * distorted_x * distorted_y
            )
            valid = np.abs(radial) > 1e-12
            distorted_x = np.divide(
                normalized_x - delta_x,
                radial,
                out=np.full_like(radial, np.nan),
                where=valid,
            )
            distorted_y = np.divide(
                normalized_y - delta_y,
                radial,
                out=np.full_like(radial, np.nan),
                where=valid,
            )
    else:
        distorted_x, distorted_y = normalized_x, normalized_y
    finite_projection = np.isfinite(distorted_x) & np.isfinite(distorted_y)
    pixel_x = np.where(finite_projection, distorted_x, 0.0)
    pixel_y = np.where(finite_projection, distorted_y, 0.0)
    u = np.rint(
        float(intrinsics["fx"]) * pixel_x
        + float(intrinsics["cx"])
    ).astype(np.int64)
    v = np.rint(
        float(intrinsics["fy"]) * pixel_y
        + float(intrinsics["cy"])
    ).astype(np.int64)
    inside = (
        positive
        & finite_projection
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
        & (labels >= 0)
    )
    candidates = np.flatnonzero(inside)
    if not len(candidates):
        return {}
    pixels = v[candidates] * width + u[candidates]
    order = np.argsort(z[candidates], kind="stable")
    sorted_indices = candidates[order]
    sorted_pixels = pixels[order]
    _, first = np.unique(sorted_pixels, return_index=True)
    visible = sorted_indices[np.sort(first)]
    result: dict[int, list[int]] = {}
    for index in visible:
        result.setdefault(int(labels[index]), []).append(int(index))
    return {
        label: np.asarray(indices, dtype=np.int64)
        for label, indices in result.items()
        if len(indices) >= _MIN_CLUSTER_POINTS
    }


@dataclass
class Track:
    obstacle_id: str
    centroid_map: np.ndarray
    extent: np.ndarray
    last_frame: int


class SessionTracker:
    def __init__(self, *, max_distance_m: float = 0.75, max_missed_frames: int = 5):
        self._max_distance = float(max_distance_m)
        self._max_missed = int(max_missed_frames)
        self._tracks: list[Track] = []
        self._next_id = 1

    def assign(self, centroids: list[np.ndarray], extents: list[np.ndarray], frame: int) -> list[str]:
        result = [""] * len(centroids)
        candidates = []
        for object_index, centroid in enumerate(centroids):
            for track_index, track in enumerate(self._tracks):
                if frame - track.last_frame > self._max_missed:
                    continue
                distance = float(np.linalg.norm(centroid - track.centroid_map))
                size_delta = float(np.linalg.norm(extents[object_index] - track.extent))
                if distance <= self._max_distance and size_delta <= 1.5:
                    candidates.append((distance + 0.1 * size_delta, object_index, track_index))
        used_objects, used_tracks = set(), set()
        for _, object_index, track_index in sorted(candidates):
            if object_index in used_objects or track_index in used_tracks:
                continue
            track = self._tracks[track_index]
            track.centroid_map = np.asarray(centroids[object_index], dtype=np.float64)
            track.extent = np.asarray(extents[object_index], dtype=np.float64)
            track.last_frame = frame
            result[object_index] = track.obstacle_id
            used_objects.add(object_index)
            used_tracks.add(track_index)
        for index, centroid in enumerate(centroids):
            if result[index]:
                continue
            obstacle_id = f"obs-{self._next_id:06d}"
            self._next_id += 1
            self._tracks.append(
                Track(
                    obstacle_id,
                    np.asarray(centroid, dtype=np.float64),
                    np.asarray(extents[index], dtype=np.float64),
                    frame,
                )
            )
            result[index] = obstacle_id
        return result

    def manifest(self) -> list[dict]:
        return [
            {
                "obstacle_id": track.obstacle_id,
                "last_centroid_map_m": [round(float(v), 6) for v in track.centroid_map],
                "last_extent_m": [round(float(v), 6) for v in track.extent],
                "last_frame_index": track.last_frame,
            }
            for track in self._tracks
        ]


def annotate_frame(
    image: dict,
    lidar: dict | None,
    imu: dict | None,
    odom: dict | None,
    tracker: SessionTracker,
    frame_index: int,
) -> dict:
    image_stamp_ns = int(image["stamp_ns"])
    lidar_stamp_ns = None if lidar is None else int(lidar["stamp_ns"])
    imu_stamp_ns = None if imu is None else int(imu["stamp_ns"])
    odom_stamp_ns = None if odom is None else int(odom["stamp_ns"])
    base = {
        "schema": ANNOTATION_SCHEMA,
        "image_id": image["image_id"],
        "image_stamp_ns": image_stamp_ns,
        "image_path": image["image_path"],
        "lidar_id": None if lidar is None else lidar.get("lidar_id"),
        "lidar_path": None if lidar is None else lidar.get("lidar_path"),
        "lidar_frame_id": None if lidar is None else lidar.get("frame_id"),
        "calibration_id": image["metadata"].get("calibration_id"),
        "frame_id": image["metadata"].get("frame_id"),
        "coordinate_convention": "camera_optical_x_right_y_down_z_forward",
        "timestamps_ns": {
            "image_source": image_stamp_ns,
            "image_driver_receive": image["metadata"].get("receive_stamp_ns"),
            "lidar_source": lidar_stamp_ns,
            "imu_source": imu_stamp_ns,
            "odom_source": odom_stamp_ns,
        },
        "matched_lidar_stamp_ns": lidar_stamp_ns,
        "matched_imu_stamp_ns": imu_stamp_ns,
        "matched_odom_stamp_ns": odom_stamp_ns,
        "distance_ground_truth": {
            "source": "matched_lidar_nearest_visible_point",
            "unit": "meter",
            "nearest_obstacle_distance_m": None,
        },
        "obstacles": [],
    }
    try:
        base["camera_parameters"] = _camera_parameters(image["metadata"])
    except (KeyError, TypeError, ValueError, PostprocessError) as exc:
        return {**base, "status": "invalid", "failure_reason": str(exc)}
    missing = [name for name, value in (("lidar", lidar), ("imu", imu), ("odom", odom)) if value is None]
    if missing:
        return {**base, "status": "invalid", "failure_reason": "missing_time_match:" + ",".join(missing)}
    skews = {
        "image_lidar": abs(lidar_stamp_ns - image_stamp_ns) / 1_000_000.0,
        "lidar_imu": abs(imu_stamp_ns - lidar_stamp_ns) / 1_000_000.0,
        "image_odom": abs(odom_stamp_ns - image_stamp_ns) / 1_000_000.0,
    }
    base["time_skew_ms"] = {key: round(value, 3) for key, value in skews.items()}
    if (
        skews["image_lidar"] > _SYNC_TOLERANCE_NS["lidar"] / 1_000_000.0
        or skews["lidar_imu"] > _SYNC_TOLERANCE_NS["imu"] / 1_000_000.0
        or skews["image_odom"] > _SYNC_TOLERANCE_NS["odom"] / 1_000_000.0
    ):
        return {**base, "status": "invalid", "failure_reason": "time_skew_exceeded"}
    try:
        metadata = image["metadata"]
        lidar_points = _voxel_downsample(np.asarray(lidar["points"], dtype=np.float64))
        obstacles_lidar = remove_ground(lidar_points, np.asarray(imu["gravity"], dtype=np.float64))
        points_camera = _transform(obstacles_lidar, _matrix(metadata["t_camera_lidar"], "t_camera_lidar"))
        finite = np.isfinite(points_camera).all(axis=1) & (np.linalg.norm(points_camera, axis=1) <= 30.0)
        points_camera = points_camera[finite]
        labels, clusters = cluster_points(points_camera)
        visible = visible_cluster_points(points_camera, labels, metadata)
        t_map_base = np.asarray(odom["t_map_base"], dtype=np.float64)
        t_base_camera = _matrix(metadata["t_base_camera"], "t_base_camera")
        centroids, extents, retained = [], [], []
        for cluster_index, visible_indices in visible.items():
            cluster = points_camera[clusters[cluster_index]]
            visible_points = points_camera[visible_indices]
            map_points = _transform(_transform(cluster, t_base_camera), t_map_base)
            centroids.append(np.mean(map_points, axis=0))
            extents.append(np.ptp(map_points, axis=0))
            retained.append((visible_indices, visible_points))
        ids = tracker.assign(centroids, extents, frame_index)
        output = []
        for obstacle_id, (visible_indices, visible_points) in zip(ids, retained):
            distances = np.linalg.norm(visible_points, axis=1)
            nearest = visible_points[int(np.argmin(distances))]
            distance = float(np.linalg.norm(nearest))
            output.append(
                {
                    "obstacle_id": obstacle_id,
                    "nearest_point_camera_m": {
                        "x": round(float(nearest[0]), 6),
                        "y": round(float(nearest[1]), 6),
                        "z": round(float(nearest[2]), 6),
                    },
                    "distance_m": round(distance, 6),
                    "distance_ground_truth_m": round(distance, 6),
                    "visible_point_count": int(len(visible_indices)),
                    "point_source": "lidar",
                }
            )
        ground_truth = {
            **base["distance_ground_truth"],
            "nearest_obstacle_distance_m": (
                None
                if not output
                else min(item["distance_ground_truth_m"] for item in output)
            ),
        }
        return {
            **base,
            "status": "valid",
            "failure_reason": None,
            "distance_ground_truth": ground_truth,
            "obstacles": output,
        }
    except (KeyError, TypeError, ValueError, PostprocessError) as exc:
        return {**base, "status": "invalid", "failure_reason": str(exc)}


class RosbagRecordReader:
    """Stream normalized records from one finalized rosbag2 MCAP directory."""

    def __init__(self, session_directory: Path):
        self._directory = str(session_directory)

    def _reader(self):
        try:
            import rosbag2_py
        except ImportError as exc:
            raise PostprocessError("rosbag2_py_unavailable") from exc
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=self._directory, storage_id="mcap"),
            rosbag2_py.ConverterOptions("", ""),
        )
        return reader

    def count_images(self) -> int:
        return sum(1 for record in self.iter_records() if record["kind"] == "rgb_v2")

    def iter_records(self) -> Iterable[dict]:
        try:
            from g1_fast_livo2.camera_rgb_v2 import decode as decode_rgb_v2
            from g1_fast_livo2.vectorized_cloud import decode_xyz_array
            from nav_msgs.msg import Odometry
            from rclpy.serialization import deserialize_message
            from sensor_msgs.msg import Imu, PointCloud2
            from std_msgs.msg import UInt8MultiArray
        except ImportError as exc:
            raise PostprocessError(f"offline_reader_dependency_unavailable:{exc}") from exc
        topics = {
            "/ubuntu/navigation/camera/rgb": ("rgb_v2", UInt8MultiArray),
            "/ubuntu/navigation/collection/camera/rgb": (
                "rgb_v2",
                UInt8MultiArray,
            ),
            "/ubuntu/navigation/lidar": ("lidar", PointCloud2),
            "/ubuntu/navigation/collection/lidar": ("lidar", PointCloud2),
            "/ubuntu/navigation/imu": ("imu", Imu),
            "/ubuntu/navigation/collection/imu": ("imu", Imu),
            "/ubuntu/navigation/odom": ("odom", Odometry),
            "/ubuntu/navigation/collection/odom": ("odom", Odometry),
        }
        reader = self._reader()
        while reader.has_next():
            topic, payload, _ = reader.read_next()
            spec = topics.get(topic)
            if spec is None:
                continue
            kind, message_type = spec
            message = deserialize_message(payload, message_type)
            if kind == "rgb_v2":
                metadata, jpeg = decode_rgb_v2(bytes(message.data))
                yield {
                    "kind": kind,
                    "stamp_ns": int(metadata["source_stamp_ns"]),
                    "metadata": metadata,
                    "jpeg": jpeg,
                }
                continue
            header = message.header.stamp
            stamp_ns = int(header.sec) * 1_000_000_000 + int(header.nanosec)
            if stamp_ns <= 0:
                continue
            if kind == "lidar":
                yield {
                    "kind": kind,
                    "stamp_ns": stamp_ns,
                    "frame_id": str(message.header.frame_id),
                    "points": decode_xyz_array(
                        fields=message.fields,
                        data=bytes(message.data),
                        point_step=int(message.point_step),
                        row_step=int(message.row_step),
                        width=int(message.width),
                        height=int(message.height),
                        is_bigendian=bool(message.is_bigendian),
                        max_points=200_000,
                        max_data_bytes=64 * 1024 * 1024,
                    ),
                }
            elif kind == "imu":
                yield {
                    "kind": kind,
                    "stamp_ns": stamp_ns,
                    "gravity": np.asarray(
                        (
                            message.linear_acceleration.x,
                            message.linear_acceleration.y,
                            message.linear_acceleration.z,
                        ),
                        dtype=np.float64,
                    ),
                }
            else:
                pose = message.pose.pose
                t_map_base = np.eye(4, dtype=np.float64)
                t_map_base[:3, :3] = _quaternion_matrix(
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                )
                t_map_base[:3, 3] = (
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                )
                yield {"kind": kind, "stamp_ns": stamp_ns, "t_map_base": t_map_base}


class OfflineAnnotationProcessor:
    def __init__(self, reader_factory: Callable[[Path], object] = RosbagRecordReader):
        self._reader_factory = reader_factory

    @staticmethod
    def _nearest(records: deque, stamp_ns: int, kind: str) -> dict | None:
        if not records:
            return None
        candidate = min(records, key=lambda value: abs(int(value["stamp_ns"]) - stamp_ns))
        if (
            abs(int(candidate["stamp_ns"]) - stamp_ns)
            > _SYNC_TOLERANCE_NS[kind]
        ):
            return None
        return candidate

    def process_session(
        self,
        session_directory: Path,
        progress: Callable[[str, int, int, dict | None], None],
        wait_if_paused: Callable[[], None],
    ) -> dict:
        session = Path(session_directory)
        reader = self._reader_factory(session)
        progress("scanning", 0, 0, None)
        total = int(reader.count_images())
        if total <= 0:
            raise PostprocessError("no_rgb_v2_images")
        partial = session / "derived.partial"
        final = session / "derived"
        if final.is_dir() and (final / "manifest.json").is_file():
            return json.loads((final / "manifest.json").read_text(encoding="utf-8"))
        partial.mkdir(parents=True, exist_ok=True)
        (partial / "rgb").mkdir(exist_ok=True)
        (partial / "lidar").mkdir(exist_ok=True)
        (partial / "frames").mkdir(exist_ok=True)
        buffers = {
            "lidar": deque(maxlen=8),
            "imu": deque(maxlen=64),
            "odom": deque(maxlen=16),
        }
        pending = deque()
        pending_bytes = 0
        tracker = SessionTracker()
        processed = valid = invalid = 0
        lidar_ids: set[str] = set()

        def consume(image: dict) -> None:
            nonlocal processed, valid, invalid
            wait_if_paused()
            image_id = f"frame-{processed + 1:08d}"
            image_path = partial / "rgb" / f"{image_id}.jpg"
            _atomic_bytes(image_path, bytes(image["jpeg"]))
            normalized = {
                **image,
                "image_id": image_id,
                "image_path": f"rgb/{image_id}.jpg",
            }
            lidar = self._nearest(
                buffers["lidar"], int(image["stamp_ns"]), "lidar"
            )
            if lidar is not None:
                lidar_id = f"lidar-{int(lidar['stamp_ns']):019d}"
                lidar_relative_path = f"lidar/{lidar_id}.pcd"
                if lidar_id not in lidar_ids:
                    _atomic_bytes(
                        partial / lidar_relative_path,
                        _pcd_bytes(lidar["points"]),
                    )
                    lidar_ids.add(lidar_id)
                lidar = {
                    **lidar,
                    "lidar_id": lidar_id,
                    "lidar_path": lidar_relative_path,
                }
            imu_reference_stamp_ns = (
                int(image["stamp_ns"])
                if lidar is None
                else int(lidar["stamp_ns"])
            )
            annotation = annotate_frame(
                normalized,
                lidar,
                self._nearest(buffers["imu"], imu_reference_stamp_ns, "imu"),
                self._nearest(buffers["odom"], int(image["stamp_ns"]), "odom"),
                tracker,
                processed,
            )
            _atomic_json(partial / "frames" / f"{image_id}.json", annotation)
            processed += 1
            if annotation["status"] == "valid":
                valid += 1
            else:
                invalid += 1
            progress(
                "processing",
                processed,
                total,
                {
                    "current_image_stamp_ns": int(image["stamp_ns"]),
                    "current_lidar_id": (
                        None if lidar is None else lidar["lidar_id"]
                    ),
                    "generated_lidar_frames": len(lidar_ids),
                },
            )

        for record in reader.iter_records():
            wait_if_paused()
            kind = record["kind"]
            if kind == "rgb_v2":
                pending.append(record)
                pending_bytes += len(record["jpeg"])
            elif kind in buffers:
                buffers[kind].append(record)
            while pending and all(
                buffers[name]
                and int(buffers[name][-1]["stamp_ns"])
                >= int(pending[0]["stamp_ns"]) + _SYNC_TOLERANCE_NS[name]
                for name in buffers
            ):
                image = pending.popleft()
                pending_bytes -= len(image["jpeg"])
                consume(image)
            while (
                len(pending) > _MAX_PENDING_IMAGES
                or pending_bytes > _MAX_PENDING_IMAGE_BYTES
            ):
                image = pending.popleft()
                pending_bytes -= len(image["jpeg"])
                consume(image)
        while pending:
            image = pending.popleft()
            pending_bytes -= len(image["jpeg"])
            consume(image)
        progress("finalizing", processed, total, None)
        _atomic_json(partial / "tracks.json", {"tracks": tracker.manifest()})
        manifest = {
            "schema": "phanthy.navigation.obstacle_dataset.v1",
            "state": "complete" if invalid == 0 else "degraded",
            "session_directory": str(session),
            "total_images": total,
            "processed_images": processed,
            "lidar_frames": len(lidar_ids),
            "valid_images": valid,
            "invalid_images": invalid,
            "artifacts": {
                "rgb": {"directory": "rgb", "format": "jpeg"},
                "lidar": {
                    "directory": "lidar",
                    "format": "pcd_binary_xyz_float32_m",
                },
                "frames": {"directory": "frames", "format": "json"},
            },
            "completed_unix_ms": int(time.time() * 1000),
        }
        _atomic_json(partial / "manifest.json", manifest)
        if final.exists():
            raise PostprocessError("derived_directory_already_exists")
        partial.replace(final)
        return manifest


class CollectionPostprocessManager:
    """One persistent, resumable worker owned by the ActuCore card process."""

    def __init__(
        self,
        root_directory: str,
        *,
        processor_factory: Callable[[], OfflineAnnotationProcessor] = OfflineAnnotationProcessor,
    ):
        self._root = Path(root_directory)
        self._processor_factory = processor_factory
        self._lock = threading.Condition()
        self._jobs: queue.Queue[Path] = queue.Queue()
        self._known: set[Path] = set()
        self._runtime_active = False
        self._paused_stage: str | None = None
        self._current_session: Path | None = None
        self._status = self._idle_status()
        self._raw_collection_status: dict = {
            "schema": "phanthy.navigation.fast_livo2_collection_status.v1",
            "enabled": False,
            "state": "disabled",
            "healthy": True,
        }
        self._worker = threading.Thread(
            target=self._run,
            daemon=True,
            name="navigation-collection-postprocess",
        )
        self._worker.start()
        self._discover()

    @staticmethod
    def _idle_status() -> dict:
        return {
            "schema": POSTPROCESS_SCHEMA,
            "state": "idle",
            "stage": None,
            "session_id": None,
            "processed_images": 0,
            "total_images": 0,
            "generated_lidar_frames": 0,
            "percent": 0.0,
            "output_directory": None,
            "paused_reason": None,
            "failure_reason": None,
        }

    def update_root(self, root_directory: str) -> None:
        with self._lock:
            self._root = Path(root_directory)
        self._discover()

    def update_raw_status(self, value: dict) -> None:
        if not isinstance(value, dict):
            return
        with self._lock:
            self._raw_collection_status = dict(value)

    def set_runtime_active(self, active: bool) -> None:
        with self._lock:
            self._runtime_active = bool(active)
            if active and self._status["state"] in {"scanning", "processing", "finalizing"}:
                self._paused_stage = self._status.get("stage") or "processing"
                self._status.update(
                    state="paused",
                    paused_reason="navigation_runtime_active",
                )
            elif not active and self._status["state"] == "paused":
                self._status.update(
                    state=self._paused_stage or "processing",
                    paused_reason=None,
                )
                self._paused_stage = None
            self._lock.notify_all()

    def enqueue_receipt(self, receipt: dict | None) -> bool:
        if not isinstance(receipt, dict):
            return False
        with self._lock:
            self._raw_collection_status = {
                **self._raw_collection_status,
                "enabled": False,
                "state": "disabled",
                "healthy": receipt.get("state") == "complete",
                "failure_reason": receipt.get("failure_reason"),
                "last_receipt": dict(receipt),
            }
        if receipt.get("storage_complete") is not True:
            return False
        directory = receipt.get("directory")
        if not isinstance(directory, str) or not directory:
            return False
        try:
            session = Path(directory).resolve()
            root = self._root.resolve()
            session.relative_to(root)
        except (OSError, ValueError):
            return False
        return self.enqueue(session)

    def enqueue(self, session_directory: Path) -> bool:
        session = Path(session_directory)
        with self._lock:
            if session in self._known:
                return False
            self._known.add(session)
            self._jobs.put(session)
            if self._status["state"] in {"idle", "complete", "degraded", "error"}:
                self._status.update(
                    state="queued",
                    stage="queued",
                    session_id=session.name,
                    output_directory=str(session / "derived"),
                    failure_reason=None,
                )
            self._lock.notify_all()
        return True

    def _discover(self) -> None:
        root = self._root
        if not root.is_dir():
            return
        for receipt_path in sorted(root.rglob("collection.json")):
            session = receipt_path.parent
            if (session / "derived" / "manifest.json").is_file():
                continue
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if receipt.get("storage_complete") is True:
                self.enqueue(session)

    def _wait_if_paused(self) -> None:
        with self._lock:
            while self._runtime_active:
                if self._paused_stage is None:
                    self._paused_stage = self._status.get("stage") or "processing"
                self._status.update(
                    state="paused",
                    paused_reason="navigation_runtime_active",
                )
                self._lock.wait(timeout=1.0)

    def _persist_status(self, session: Path | None, status: dict) -> None:
        if session is None:
            return
        _atomic_json(
            session / "postprocess.json",
            {**status, "updated_unix_ms": int(time.time() * 1000)},
        )

    def _progress(self, stage: str, processed: int, total: int, details: dict | None) -> None:
        with self._lock:
            percent = 0.0 if total <= 0 else min(100.0, 100.0 * processed / total)
            self._status.update(
                state=stage,
                stage=stage,
                processed_images=int(processed),
                total_images=int(total),
                percent=round(percent, 2),
                paused_reason=None,
                failure_reason=None,
            )
            if details:
                self._status.update(details)
            status = dict(self._status)
            session = self._current_session
        self._persist_status(session, status)

    def _run(self) -> None:
        while True:
            session = self._jobs.get()
            self._wait_if_paused()
            with self._lock:
                self._current_session = session
                self._status = {
                    **self._idle_status(),
                    "state": "scanning",
                    "stage": "scanning",
                    "session_id": session.name,
                    "output_directory": str(session / "derived"),
                }
                status = dict(self._status)
            try:
                self._persist_status(session, status)
                manifest = self._processor_factory().process_session(
                    session,
                    self._progress,
                    self._wait_if_paused,
                )
                final_state = str(manifest.get("state", "complete"))
                with self._lock:
                    self._status.update(
                        state=final_state,
                        stage="complete",
                        percent=100.0,
                        processed_images=int(manifest.get("processed_images", 0)),
                        total_images=int(manifest.get("total_images", 0)),
                        generated_lidar_frames=int(
                            manifest.get("lidar_frames", 0)
                        ),
                        failure_reason=(
                            "invalid_frames_present" if final_state == "degraded" else None
                        ),
                    )
                    status = dict(self._status)
                self._persist_status(session, status)
            except Exception as exc:
                with self._lock:
                    self._status.update(
                        state="error",
                        stage="error",
                        failure_reason=f"{type(exc).__name__}:{exc}",
                    )
                    status = dict(self._status)
                try:
                    self._persist_status(session, status)
                except OSError:
                    pass
            finally:
                with self._lock:
                    self._current_session = None
                self._jobs.task_done()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                **dict(self._raw_collection_status),
                "postprocess": dict(self._status),
            }


class DisabledCollectionController:
    def set_runtime_active(self, active: bool) -> None:
        pass

    def enqueue_receipt(self, receipt: dict | None) -> bool:
        return False

    def update_root(self, root_directory: str) -> None:
        pass

    def snapshot(self) -> dict:
        return {"postprocess": CollectionPostprocessManager._idle_status()}


class RosCollectionController:
    """Bridge child-runtime raw status to one persistent public Canvas topic."""

    def __init__(self, root_directory: str, namespace: str, executor):
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        root = f"/{namespace.strip('/')}"
        self._String = String
        self._node = Node("navigation_collection_status")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._manager = CollectionPostprocessManager(root_directory)
        self._publisher = self._node.create_publisher(
            String, f"{root}/navigation/fast_livo2/collection_status", qos
        )
        self._subscription = self._node.create_subscription(
            String,
            f"{root}/navigation/fast_livo2/collection_status_raw",
            self._on_raw,
            qos,
        )
        self._timer = self._node.create_timer(1.0, self._publish)
        executor.add_node(self._node)

    def _on_raw(self, message) -> None:
        try:
            value = json.loads(message.data)
        except (TypeError, ValueError):
            return
        self._manager.update_raw_status(value)

    def _publish(self) -> None:
        message = self._String()
        message.data = json.dumps(
            self._manager.snapshot(), ensure_ascii=False, separators=(",", ":")
        )
        self._publisher.publish(message)

    def set_runtime_active(self, active: bool) -> None:
        self._manager.set_runtime_active(active)

    def enqueue_receipt(self, receipt: dict | None) -> bool:
        return self._manager.enqueue_receipt(receipt)

    def update_root(self, root_directory: str) -> None:
        self._manager.update_root(root_directory)

    def snapshot(self) -> dict:
        return self._manager.snapshot()


def build_collection_controller(root_directory: str, namespace: str, executor):
    if executor is None:
        return DisabledCollectionController()
    return RosCollectionController(root_directory, namespace, executor)


__all__ = [
    "ANNOTATION_SCHEMA",
    "CollectionPostprocessManager",
    "DisabledCollectionController",
    "OfflineAnnotationProcessor",
    "PostprocessError",
    "RosCollectionController",
    "SessionTracker",
    "annotate_frame",
    "build_collection_controller",
    "cluster_points",
    "remove_ground",
    "visible_cluster_points",
]
