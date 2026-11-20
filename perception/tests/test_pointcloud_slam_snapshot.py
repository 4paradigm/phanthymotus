"""Saved-map snapshot rebuild from the finalized native RTAB-Map database."""
import sqlite3
import struct
import zlib

import numpy as np

from plugins import pointcloud_slam_snapshot as module


def _fallback(x=5.0, y=6.0, yaw=0.25, count=0):
    return struct.pack('<fffBI', x, y, yaw, 3, count)


def _cloud_blob(points):
    array = np.asarray(points, dtype='<f4').reshape(-1, 3)
    return zlib.compress(array.tobytes()) + struct.pack('<iii', 1, len(array), 21)


def _pose_blob(tx, ty, tz=0.0):
    matrix = np.eye(4)
    matrix[:3, 3] = [tx, ty, tz]
    return np.asarray(matrix[:3, :], dtype='<f4').tobytes()


def _database(path, nodes, links=()):
    connection = sqlite3.connect(path)
    connection.executescript('''
        CREATE TABLE Node(id INTEGER PRIMARY KEY, pose BLOB);
        CREATE TABLE Data(id INTEGER PRIMARY KEY, obstacle_cells BLOB,
                          ground_cells BLOB, scan BLOB, scan_info BLOB);
        CREATE TABLE Link(from_id INTEGER, to_id INTEGER);
    ''')
    for node_id, pose, obstacle, ground, scan, scan_info in nodes:
        connection.execute('INSERT INTO Node VALUES(?,?)', (node_id, pose))
        connection.execute('INSERT INTO Data VALUES(?,?,?,?,?)',
                           (node_id, obstacle, ground, scan, scan_info))
    connection.executemany('INSERT INTO Link VALUES(?,?)', links)
    connection.commit()
    connection.close()
    return str(path)


def _decode(packet):
    x, y, yaw, flags, count = struct.unpack('<fffBI', packet[:17])
    points = np.frombuffer(packet[17:], dtype='<f4').reshape(-1, 3)
    return x, y, yaw, flags, points


def test_rebuild_uses_linked_graph_nodes_and_keeps_marker(tmp_path):
    database = _database(tmp_path / 'native.db', [
        (1, _pose_blob(1, 0), _cloud_blob([[0, 0, 0], [0, 1, 0]]), None, None, None),
        (2, _pose_blob(2, 0), None, _cloud_blob([[0, 0, 0]]), None, None),
        # No link: this ignored frame was never part of the map and must not appear.
        (3, _pose_blob(3, 0), _cloud_blob([[0, 0, 5]]), None, None, None),
    ], links=[(1, 2)])
    packet = module.rebuild(database, max_points=100, fallback=_fallback())
    x, y, yaw, flags, points = _decode(packet)
    assert (x, y, yaw, flags) == (5.0, 6.0, 0.25, 3)
    assert len(points) == 3
    expected = np.array([[1, 0, 0], [1, 1, 0], [2, 0, 0]], dtype=float)
    for point in expected:
        assert np.min(np.linalg.norm(points - point, axis=1)) < 1e-5
    assert np.max(np.linalg.norm(points - np.array([3, 0, 5]), axis=1)) > 1


def test_saved_snapshot_does_not_flip_display_yaw_again(tmp_path):
    database = _database(tmp_path / 'native.db', [
        (1, _pose_blob(1, 2), _cloud_blob([[0, 0, 0]]), None, None, None),
    ], links=[(1, 1)])
    for display_yaw in (0.25, -0.25, 1.1, -1.1):
        fallback = _fallback(yaw=display_yaw)
        packet = module.rebuild(database, max_points=20, fallback=fallback)
        assert np.isclose(_decode(packet)[2], display_yaw, atol=1e-6)


def test_rebuild_falls_back_for_corrupt_or_empty_database(tmp_path):
    fallback = _fallback()
    missing = module.rebuild(str(tmp_path / 'missing.db'), max_points=100, fallback=fallback)
    assert missing == fallback
    empty = sqlite3.connect(tmp_path / 'empty.db')
    empty.close()
    assert module.rebuild(str(tmp_path / 'empty.db'), max_points=100, fallback=fallback) == fallback


def test_rebuild_respects_point_limit(tmp_path):
    points = [[i * 0.1, 0, 0] for i in range(100)]
    database = _database(tmp_path / 'native.db', [
        (1, _pose_blob(0, 0), _cloud_blob(points), None, None, None)], links=[])
    # A database with no graph link falls back; add a self link so the node is in-graph.
    connection = sqlite3.connect(database)
    connection.execute('INSERT INTO Link VALUES(1,1)')
    connection.commit()
    connection.close()
    packet = module.rebuild(database, max_points=40, fallback=_fallback())
    _, _, _, _, rebuilt = _decode(packet)
    assert len(rebuilt) == 40


def test_rebuild_uses_scan_and_local_transform_when_cells_are_absent(tmp_path):
    local = np.eye(4)
    scan_info = np.asarray([0.0] * 7 + list(local[:3, :].ravel()), dtype='<f4').tobytes()
    database = _database(tmp_path / 'native.db', [
        (1, _pose_blob(2, 0), None, None, _cloud_blob([[0, 1, 0]]), scan_info)],
        links=[(1, 1)])
    packet = module.rebuild(database, max_points=100, fallback=_fallback())
    _, _, _, _, points = _decode(packet)
    assert len(points) == 1
    assert np.allclose(points[0], [2, 1, 0], atol=1e-5)


def test_rebuilt_snapshot_preserves_typed_historical_heading(tmp_path):
    database = _database(tmp_path / 'native.db', [
        (1, _pose_blob(1, 2), _cloud_blob([[0, 0, 0]]), None, None, None),
    ], links=[(1, 1)])
    for flags in (15, 23, 39, 47):
        fallback = struct.pack('<fffBI', 5., 6., -.75, flags, 0)
        packet = module.rebuild(database, max_points=20, fallback=fallback)
        assert packet[:13] == fallback[:13]
        assert len(packet) == 29
