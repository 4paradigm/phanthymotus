"""Offline packaging checks; native engine acceptance runs separately in Docker."""
import importlib.util
import json
from pathlib import Path
import re
import sqlite3
import struct
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest


ROOT = Path(__file__).resolve().parents[2]


def smoke():
    path = ROOT / 'perception/tools/rtabmap_pointcloud_smoke.py'
    spec = importlib.util.spec_from_file_location('rtabmap_pointcloud_smoke', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_params_from_command(command):
    """Read the same ROS params file that the native process receives."""
    assert command.count('--params-file') == 1
    assert '-p' not in command  # RTAB-Map names with '/' cannot use CLI overrides.
    path = Path(command[command.index('--params-file') + 1])
    assert path.is_file()
    return json.loads(path.read_text())['/**']['ros__parameters']


def test_native_command_selects_external_odom_cloud_and_pcl_toro(tmp_path):
    command = smoke().native_command('/native/rtabmap', tmp_path / 'map.db', 'build_smoke',
        {'max_pending': 7, 'voxel_size_m': .12, 'max_working_nodes': 100, 'max_update_ms': 250})
    assert command[0] == '/native/rtabmap'
    params = native_params_from_command(command)
    remaps = {value for flag, value in zip(command, command[1:]) if flag == '-r'}
    assert {'__ns:=/build_smoke', 'odom:=odom', 'scan_cloud:=cloud'} <= remaps
    assert params['database_path'] == str(tmp_path / 'map.db')
    assert params['subscribe_scan_cloud'] is True
    assert params['subscribe_rgb'] is params['subscribe_depth'] is False
    assert params['approx_sync'] is False
    assert params['Reg/Strategy'] == '1'
    assert params['Icp/Strategy'] == params['Optimizer/Strategy'] == '0'
    assert params['frame_id'] == 'build_smoke/base'
    assert params['Icp/VoxelSize'] == params['Grid/CellSize'] == '0.12'
    assert params['topic_queue_size'] == params['sync_queue_size'] == 7
    assert params['Grid/3D'] == 'true'


def test_smoke_forwards_shared_production_parameters_without_overrides(tmp_path, monkeypatch):
    module = smoke()
    expected = {'database_path': str(tmp_path / 'native.db'), 'Grid/3D': 'false',
                'Icp/Iterations': '3', 'frame_id': 'production-frame'}
    monkeypatch.setattr(module, 'mapping_parameters', lambda *args: expected, raising=False)
    command = module.native_command('/native/rtabmap', tmp_path / 'native.db', '/smoke', {})
    params = native_params_from_command(command)
    assert params == expected


def map_cloud(z_values, *, bigendian=False):
    # Native PointXYZRGB layout includes padding after XYZ; row_step includes it.
    data = b''.join(struct.pack(('>' if bigendian else '<') + 'ffffffff', 1., 2., z, 1., 0., 0., 0., 0.)
                    for z in z_values)
    return NS(width=len(z_values), height=1, point_step=32, row_step=len(data), data=data,
              is_bigendian=bigendian, fields=[NS(name=name, offset=offset, datatype=7, count=1)
                  for name, offset in [('x', 0), ('y', 4), ('z', 8), ('rgb', 16)]])


@pytest.mark.parametrize('bigendian', [False, True])
def test_native_map_gate_requires_actual_vertical_extent(bigendian):
    assert smoke().require_3d_map(map_cloud([.2, 1.8], bigendian=bigendian)) == pytest.approx([.2, 1.8])


@pytest.mark.parametrize('heights', [[0.], [1., 1.], [float('nan'), float('inf')]])
def test_native_map_gate_rejects_single_flat_or_nonfinite_clouds(heights):
    with pytest.raises(RuntimeError, match='three-dimensional'):
        smoke().require_3d_map(map_cloud(heights))


def test_gateway_fixture_is_a_valid_measurement_with_exact_millisecond_alignment():
    from plugins.pointcloud_slam_odom import decode_slam_odom
    sample = smoke().tracking_sample(1_700_000_000_123_000_000, .45)
    pose = decode_slam_odom(sample)
    assert pose.stamp_ms == 1_700_000_000_123
    assert pose.position == (.45, 0., 0.)
    assert (pose.frame_id, pose.child_frame_id, pose.segment_id, pose.clock_id) == (
        'smoke_odom', 'smoke_body', 'gateway_smoke', 'ros')
    assert sample['vendor']['motus_slam']['stamp_source'] == 'measurement'


@pytest.mark.parametrize('stamp', [0, 1_700_000_000_123_000_001])
def test_gateway_fixture_rejects_nonpositive_or_non_aligned_capture_time(stamp):
    with pytest.raises(ValueError, match='aligned'):
        smoke().tracking_sample(stamp, 0.)


@pytest.mark.parametrize('flags', [3, 15, 23, 39, 47])
def test_mapping_summary_reads_real_wire_pose_and_3d_geometry(flags):
    packet = struct.pack('<fffBIffffff', 1.25, -2.5, .5, flags, 2, 1., 2., .25, 3., 4., 1.75)
    assert smoke().mapping_packet_summary(packet) == {
        'pose': (1.25, -2.5, .5), 'point_count': 2, 'z_range_m': (.25, 1.75)}


@pytest.mark.parametrize('packet', [
    b'', struct.pack('<fffBI', 0, 0, 0, 3, 0),
    struct.pack('<fffBIfff', 0, 0, 0, 0, 1, 1, 2, 3),
    struct.pack('<fffBI', 0, 0, 0, 3, 1),
    struct.pack('<fffBIfff', float('nan'), 0, 0, 3, 1, 1, 2, 3),
    struct.pack('<fffBIfff', 0, 0, 0, 3, 1, 1, 2, float('inf')),
])
def test_mapping_summary_rejects_empty_corrupt_or_nonfinite_packets(packet):
    with pytest.raises(RuntimeError, match='mapping'):
        smoke().mapping_packet_summary(packet)


def test_gateway_snapshot_waits_for_matching_native_stamp_and_frame():
    """Exercise the production snapshot gate rather than a deleted smoke helper."""
    from threading import RLock

    import numpy as np
    from plugins.pointcloud_slam_ros import RtabmapNode

    pose = np.eye(4)
    pose[0, 3] = 2.
    points = np.array([[1., 2., 0.], [3., 4., 1.5]])
    segment = {'graph': (2_000_000_000, 'native/map', pose),
               'cloud': (1_000_000_000, 'native/map', points),
               'snapshot': None, 'frame_id': '', 'last_preview': 0.}
    packets = []
    gateway = NS(_result_lock=RLock(),
                 _settings={'preview_max_points': 10, 'preview_interval_s': .1},
                 _active=segment, _closed=False,
                 _map_pub=NS(publish=lambda msg: packets.append(bytes(msg.data))))

    # A stale cloud cannot be paired with the newer native optimized pose.
    RtabmapNode._publish_snapshot(gateway, segment)
    assert segment['snapshot'] is None and not packets

    # Equal capture stamps are insufficient if the native map frames differ.
    segment['cloud'] = (2_000_000_000, 'other/map', points)
    RtabmapNode._publish_snapshot(gateway, segment)
    assert segment['snapshot'] is None and not packets

    # The matching pair emits the actual optimized graph pose into the packet.
    segment['cloud'] = (2_000_000_000, 'native/map', points)
    RtabmapNode._publish_snapshot(gateway, segment)
    assert len(packets) == 1
    assert segment['frame_id'] == 'native/map'
    assert segment['snapshot'] == packets[0]
    assert smoke().mapping_packet_summary(packets[0]) == {
        'pose': (2., 0., 0.), 'point_count': 2, 'z_range_m': (0., 1.5)}

def test_synthetic_cloud_is_three_dimensional_and_tracks_known_external_motion():
    module = smoke()
    stationary = module.synthetic_points(0.)
    moving = module.synthetic_points(.25)
    assert len(stationary) >= 300
    assert stationary[0] == (3., -2., 0.)
    assert moving[0] == (2.75, -2., 0.)
    assert {point[2] for point in stationary} >= {0., 1., 2.}
    assert all(abs(a[0] - b[0] - .25) < 1e-7 and a[1:] == b[1:]
               for a, b in zip(stationary, moving))
    packed = module.pack_cloud([(1., -2., .5), (0., 3., 4.)])
    assert len(packed) == 24
    assert struct.unpack('<ffffff', packed) == (1., -2., .5, 0., 3., 4.)


def test_database_validation_requires_persisted_native_nodes(tmp_path):
    path = tmp_path / 'map.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE Node (id INTEGER PRIMARY KEY)')
        db.executemany('INSERT INTO Node VALUES (?)', [(1,), (2,)])
    assert smoke().validate_database(path) == 2


@pytest.mark.parametrize('kind', ['missing', 'corrupt', 'empty_graph'])
def test_database_validation_rejects_unfinished_or_invalid_save(tmp_path, kind):
    path = tmp_path / 'map.db'
    if kind == 'corrupt':
        path.write_bytes(b'not SQLite')
    elif kind == 'empty_graph':
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE Node (id INTEGER PRIMARY KEY)')
    with pytest.raises(RuntimeError, match='database|nodes|SQLite'):
        smoke().validate_database(path)
    if kind == 'missing':
        assert not path.exists()


def test_native_shutdown_sends_sigint_and_waits_for_clean_exit():
    process = subprocess.Popen([sys.executable, '-u', '-c',
        'import signal,time,sys; signal.signal(signal.SIGINT, lambda *_:sys.exit(0)); '
        'print("ready", flush=True); time.sleep(30)'], stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == 'ready'
        smoke().stop_native(process, timeout_s=2.)
        assert process.poll() == 0
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()


def test_native_crash_is_not_accepted_as_graceful_shutdown():
    process = subprocess.Popen([sys.executable, '-c', 'raise SystemExit(3)'])
    process.wait()
    with pytest.raises(RuntimeError, match='exit.*3'):
        smoke().stop_native(process, timeout_s=2.)


def test_native_shutdown_timeout_kills_and_reaps_failed_process():
    process = subprocess.Popen([sys.executable, '-u', '-c',
        'import signal,time; signal.signal(signal.SIGINT, signal.SIG_IGN); '
        'print("ready", flush=True); time.sleep(30)'], stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == 'ready'
        with pytest.raises(RuntimeError, match='shutdown.*timed out'):
            smoke().stop_native(process, timeout_s=.05)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()


def test_jetson_recipe_uses_immutable_native_source_commits():
    # This is a recipe lint, not evidence that a native image built or ran.
    recipe = (ROOT / 'perception/Dockerfile.jetson').read_text()
    arguments = dict(re.findall(r'^ARG (RTABMAP_\w+)=(\S+)$', recipe, flags=re.MULTILINE))
    for name in ('RTABMAP_CORE', 'RTABMAP_ROS'):
        assert re.fullmatch('[0-9a-f]{40}', arguments.get(name + '_COMMIT', ''))
    calls = re.findall(r'add_missing_rtabmap_dependencies\s+(\S+)\s+(\S+)\s+(\S+)',
                       recipe.replace('\\\n', ' '))
    assert calls
    for _, _, commit in calls:
        assert re.fullmatch('[0-9a-f]{40}', commit)
    assert 'SLAM_GTSAM_COMMIT' not in recipe
    assert "'open3d==" not in recipe
