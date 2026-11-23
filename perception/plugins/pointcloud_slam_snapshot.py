"""Rebuild a complete saved-map snapshot from a finalized native RTABMap DB.

The live ``sensor/mapping`` preview is assembled from RTAB-Map's *local
optimized graph* (rtabmap_ros ``getLocalOptimizedPoses()``).  Memory management
(``Rtabmap/TimeThr`` / ``Rtabmap/MemoryThr`` - this card's ``max_update_ms`` /
``max_working_nodes``) can transfer older graph nodes to long-term memory, after
which the local graph - and therefore the last preview snapshot - no longer
contains them.  The native database keeps raw odometry in ``Node.pose``, the
saved optimized graph in ``Admin.opt_ids/opt_poses``, and the cells ``cloud_map`` is
assembled from (``Data.obstacle_cells`` / ``Data.ground_cells``), so a complete
snapshot can be rebuilt at save time without re-running RTAB-Map.

Only verified database formats are handled; any unexpected content makes
``rebuild`` return the provided fallback packet, so saving never regresses.
"""
from __future__ import annotations

import logging
import math
import sqlite3
import struct
import time
import zlib
from pathlib import Path

import numpy as np

from .pointcloud_slam_preview import encode_mapping, update_place_position

log = logging.getLogger(__name__)

_CV_32FC3 = 21              # OpenCV type of RTAB-Map point blobs (Nx1 CV_32FC3)
_MAX_BLOB_POINTS = 2_000_000


def _count(blob):
    """Point count from an RTAB-Map compressed cv::Mat without decompressing."""
    if not blob or len(blob) < 12:
        return 0
    rows, cols, kind = struct.unpack('<iii', blob[-12:])
    if kind != _CV_32FC3 or rows < 1 or cols < 1 or rows * cols > _MAX_BLOB_POINTS:
        return 0
    return rows * cols


def _unpack(blob):
    """RTAB-Map compressData2: zlib payload + int32 rows/cols/type trailer."""
    if not blob or len(blob) < 12:
        return None
    rows, cols, kind = struct.unpack('<iii', blob[-12:])
    if kind != _CV_32FC3 or rows < 1 or cols < 1 or rows * cols > _MAX_BLOB_POINTS:
        return None
    raw = zlib.decompress(blob[:-12])
    if len(raw) != rows * cols * 12:
        return None
    return np.frombuffer(raw, dtype='<f4').reshape(-1, 3).astype(np.float64)


def _matrix(blob):
    """Node.pose / scan_info local transform: 12 float32 (3x4, row major)."""
    if not blob or len(blob) < 48:
        return None
    matrix = np.eye(4)
    matrix[:3, :] = np.frombuffer(blob, dtype='<f4')[:12].reshape(3, 4)
    return matrix if np.isfinite(matrix).all() else None


def _transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _marker(packet):
    """Keep the marker (last mapped pose) of the preview snapshot replaced."""
    x, y, display_yaw, _, _ = struct.unpack('<fffBI', packet[:17])
    if not np.isfinite([x, y, display_yaw]).all():
        raise ValueError('invalid snapshot marker')
    # A preview packet stores -ROS yaw. Undo that only while constructing
    # the pose matrix; encode_mapping will apply the display convention again.
    yaw = -display_yaw
    c, s = math.cos(yaw), math.sin(yaw)
    matrix = np.eye(4)
    matrix[:3, :3] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
    matrix[:3, 3] = [x, y, 0.]
    return matrix


def _graph_nodes(connection):
    """Graph-attached nodes. Ignored (small-movement) frames are saved to the DB
    without links and were never part of the map, so they are excluded."""
    nodes = {int(node_id) for (node_id,) in
             connection.execute('SELECT id FROM Node WHERE id > 0')}
    graph = set()
    for from_id, to_id in connection.execute('SELECT from_id, to_id FROM Link'):
        for node_id in (from_id, to_id):
            if node_id and node_id > 0 and node_id in nodes:
                graph.add(int(node_id))
    return graph


def _optimized_poses(connection):
    """Pinned DBDriverSqlite3::saveOptimizedPosesQuery, not raw Node.pose."""
    columns = {row[1] for row in connection.execute('PRAGMA table_info(Admin)')}
    if not {'opt_ids', 'opt_poses'} <= columns:
        return None  # Historical database formats without an optimized cache.
    row = connection.execute('SELECT opt_ids, opt_poses FROM Admin').fetchone()
    if row is None or not all(row):
        raise ValueError('optimized graph unavailable')
    arrays = []
    for blob, kind in zip(row, (4, 5)):  # CV_32SC1 ids, CV_32FC1 poses
        rows, cols, actual_kind = struct.unpack('<iii', blob[-12:])
        if rows != 1 or not 0 < cols <= _MAX_BLOB_POINTS*12 or actual_kind != kind:
            raise ValueError('invalid optimized graph dimensions')
        decoder = zlib.decompressobj()
        raw = decoder.decompress(blob[:-12], cols*4+1)
        if len(raw) != cols*4 or not decoder.eof or decoder.unused_data:
            raise ValueError('invalid optimized graph payload')
        arrays.append(np.frombuffer(raw, dtype='<i4' if kind == 4 else '<f4'))
    ids, values = arrays
    if len(values) != len(ids)*12 or len(set(ids)) != len(ids):
        raise ValueError('invalid optimized graph cardinality')
    poses = {}
    for node_id, values in zip(ids, values.reshape(-1, 12)):
        matrix = _matrix(values.tobytes())
        if matrix is None:
            raise ValueError('invalid optimized pose')
        poses[int(node_id)] = matrix
    return poses


def _plan(connection, graph, max_points):
    """Two passes over Data: count points first (bounded memory), then decode."""
    rows = list(connection.execute(
        'SELECT Node.id, Node.pose, Data.obstacle_cells, Data.ground_cells,'
        '       Data.scan, Data.scan_info'
        '  FROM Data JOIN Node ON Node.id = Data.id'))
    optimized = _optimized_poses(connection)
    sources, total = [], 0
    for node_id, pose, cells, ground, scan, scan_info in rows:
        node_id = int(node_id)
        if node_id not in graph or _matrix(pose) is None:
            continue
        if optimized is not None:
            if node_id not in optimized:
                continue  # Never mix an evicted node's raw pose into optimized space.
            pose = np.asarray(optimized[node_id][:3], dtype='<f4').tobytes()
        count = _count(cells) + _count(ground)
        if count == 0:
            count = _count(scan) if scan_info else 0
        if count:
            sources.append((node_id, pose, cells, ground, scan, scan_info))
            total += count
    budget = max(4 * int(max_points), 200_000)
    stride = int(math.ceil(total / float(budget))) if total > budget else 1
    return sources, stride, optimized


def _load(sources, stride):
    clouds = []
    for node_id, pose, cells, ground, scan, scan_info in sources:
        local = None
        for blob in (cells, ground):
            points = _unpack(blob) if blob else None
            if points is not None and len(points):
                local = points if local is None else np.concatenate((local, points))
        if local is None and scan:
            points = _unpack(scan)
            # Pinned scan_info: seven float32 metadata values, then 3x4 transform.
            transform = _matrix(scan_info[28:])
            if points is not None and transform is not None:
                local = _transform(points, transform)   # sensor frame -> node frame
        if local is None or not len(local):
            continue
        matrix = _matrix(pose)
        if matrix is None:
            continue
        clouds.append(_transform(local[::stride], matrix))  # node frame -> map frame
    return clouds


def rebuild(database_path, *, max_points, fallback=None, places=()):
    """Return a complete ``sensor/mapping`` packet, or ``fallback`` on any problem."""
    started = time.monotonic()
    try:
        if fallback is None or len(fallback) < 17:
            return fallback
        marker = _marker(fallback)
        database = Path(database_path).resolve()
        connection = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
        try:
            if connection.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                raise ValueError('database integrity check failed')
            graph = _graph_nodes(connection)
            sources, stride, optimized = _plan(connection, graph, max_points)
            if optimized is not None:
                located = [node_id for node_id in graph if node_id in optimized]
                if located:
                    marker = optimized[max(located)]
            clouds = _load(sources, stride)
        finally:
            connection.close()
        if not clouds:
            raise ValueError('no graph node data in database')
        points = np.concatenate(clouds)
        if not len(points) or not np.isfinite(points).all():
            raise ValueError('no finite graph points')
        flags = fallback[12]
        reference = 'camera_optical' if flags & 8 else 'body' if flags & 16 else None
        packet = encode_mapping(points, marker, max_points=int(max_points), pose_reference=reference)
        # Historical caches without optimized poses retain their original marker.
        if optimized is None:
            packet = fallback[:13] + packet[13:]
        log.info('rebuilt saved map snapshot from database: graph_nodes=%d '
                 'points=%d bytes=%d time=%.2fs',
                 len(sources), len(points), len(packet), time.monotonic() - started)
        if optimized is not None:
            for place in places:
                anchor = optimized.get(place.get('node_id'))
                if anchor is not None:
                    update_place_position(place, anchor)
        return packet
    except Exception as error:  # never break saving because diagnostics failed
        log.warning('keeping preview snapshot, database rebuild failed: %s', error)
        return fallback
