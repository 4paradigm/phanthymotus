"""Native process supervision uses actual subprocesses and real SQLite files."""
import json
import sqlite3
import time

import pytest


def backend():
    from plugins import pointcloud_slam_rtabmap
    return pointcloud_slam_rtabmap


def native_script(tmp_path, *, exit_code=0, invalid_db=False, ignore_interrupt=False, ignore_terminate=False, terminate_success=False):
    path = tmp_path / 'native_rtabmap'
    path.write_text('''#!/usr/bin/env python3
import json, signal, sqlite3, sys, time
from pathlib import Path
params = json.loads(Path(sys.argv[sys.argv.index('--params-file') + 1]).read_text())['/**']['ros__parameters']
p = Path(params['database_path'])
''' + ("p.write_text('not sqlite')\n" if invalid_db else
        "db = sqlite3.connect(p)\ndb.execute('CREATE TABLE Node(id INTEGER PRIMARY KEY)')\ndb.execute('INSERT INTO Node VALUES(1)')\ndb.commit()\n") +
        ("signal.signal(signal.SIGINT, signal.SIG_IGN)\n" if ignore_interrupt else
         f"signal.signal(signal.SIGINT, lambda *args: sys.exit({exit_code}))\n") +
        ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_terminate else
         "signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))\n" if terminate_success else "") +
        "p.with_suffix('.ready').write_text('ready')\nwhile True: time.sleep(.02)\n")
    path.chmod(0o755)
    return path


def running(tmp_path, **kw):
    module = backend()
    script = native_script(tmp_path, **kw)
    process = module.RtabmapProcess(tmp_path / 'segment' / 'rtabmap.db', '/private/s1',
        {'voxel_size_m': .08, 'max_working_nodes': 300, 'max_update_ms': 100,
         'max_pending': 2}, executable=script)
    process.start()
    deadline = time.monotonic() + 10
    while not process.database_path.with_suffix('.ready').exists():
        if time.monotonic() >= deadline:
            process._process.kill(); process._process.wait(timeout=3)
            raise AssertionError('native process did not create its ready marker')
        time.sleep(.01)
    return process


def test_native_process_finalizes_database_on_one_graceful_signal(tmp_path):
    process = running(tmp_path)
    path = process.close(timeout_s=3)
    assert path == process.database_path and process.poll() == 0
    assert process.close(timeout_s=3) == path
    with sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True) as db:
        assert db.execute('SELECT COUNT(*) FROM Node').fetchone() == (1,)


def test_runtime_parameters_use_external_odom_exact_sync_and_scan_only(tmp_path):
    process = running(tmp_path)
    try:
        params = json.loads(process.parameter_path.read_text())['/**']['ros__parameters']
        assert params['subscribe_scan_cloud'] is True and params['subscribe_odom'] is True
        assert params['odom_frame_id'] == '' and params['approx_sync'] is False
        assert all(params[key] is False for key in ['subscribe_depth', 'subscribe_rgb',
            'subscribe_rgbd', 'subscribe_stereo', 'publish_tf'])
        assert params['Reg/Strategy'] == '1' and params['Optimizer/Strategy'] == '0'
        assert params['Rtabmap/MemoryThr'] == '300' and params['Rtabmap/TimeThr'] == '100'
        assert params['Grid/Sensor'] == '0' and params['Icp/VoxelSize'] == '0.08'
        assert params['Grid/3D'] == 'true'
    finally:
        process.close(timeout_s=3)


def test_optional_rgb_uses_one_sensor_data_process_for_image_and_scan_fallback(tmp_path):
    params = backend().mapping_parameters(tmp_path / 'map.db', '/private/rgb', {
        'max_pending': 10, 'voxel_size_m': .08, 'max_working_nodes': 300,
        'max_update_ms': 100, 'rgb_input': {'topic': '/rgb', 'camera_info': {}}})
    assert params['subscribe_sensor_data'] is True
    assert params['subscribe_scan_cloud'] is False
    assert params['subscribe_odom'] is True and params['approx_sync'] is False
    assert all(params[k] is False for k in ('subscribe_rgb', 'subscribe_rgbd', 'subscribe_depth'))
    assert params['Reg/Strategy'] == '1'
    assert params['RGBD/ProximityOdomGuess'] == 'false'
    assert params['RGBD/ProximityPathMaxNeighbors'] == '1'
    assert params['RGBD/NeighborLinkRefining'] == 'false'
    assert params['RGBD/LoopCovLimited'] == 'true'
    assert params['Grid/RangeMax'] == '0'
    assert params['Mem/BadSignaturesIgnored'] == 'false'
    assert params['Rtabmap/DetectionRate'] == '2'


@pytest.mark.parametrize('options,match', [({'exit_code': 7}, 'exit'),
    ({'invalid_db': True}, 'database'), ({'ignore_interrupt': True}, 'timed out')])
def test_failed_or_incomplete_stop_never_reports_saved_and_retains_data(tmp_path, options, match):
    process = running(tmp_path, **options)
    try:
        with pytest.raises(RuntimeError, match=match):
            process.close(timeout_s=.1 if options.get('ignore_interrupt') else 3)
        assert process.database_path.exists()
        assert process.parameter_path.exists()
    finally:
        if process.poll() is None:
            process._process.kill()
            process._process.wait(timeout=3)


def test_unavailable_runtime_fails_without_starting_a_process(monkeypatch):
    module = backend()
    def missing():
        raise RuntimeError('rtabmap_slam executable unavailable')
    monkeypatch.setattr(module, 'resolve_executable', missing)
    with pytest.raises(RuntimeError, match='rtabmap_slam'):
        module.require_runtime()


def test_gracefully_closed_engine_with_no_nodes_is_valid_but_reports_empty_capture(tmp_path):
    script = native_script(tmp_path)
    script.write_text(script.read_text().replace("db.execute('INSERT INTO Node VALUES(1)')", 'pass'))
    module = backend()
    process = module.RtabmapProcess(tmp_path/'empty'/'rtabmap.db', '/private/empty',
        {'voxel_size_m': .08, 'max_working_nodes': 300, 'max_update_ms': 100,
         'max_pending': 2}, executable=script)
    process.start()
    deadline = time.monotonic()+10
    while not process.database_path.with_suffix('.ready').exists():
        assert time.monotonic() < deadline
        time.sleep(.01)
    process.close(timeout_s=3)
    assert process.mapped_nodes == 0


@pytest.mark.parametrize('options', [{'ignore_terminate': True}, {'terminate_success': True}])
def test_timeout_reaps_native_child_and_never_converts_forced_exit_into_saved(tmp_path, options):
    process = running(tmp_path, ignore_interrupt=True, **options)
    try:
        with pytest.raises(RuntimeError, match='timed out|still stopping'):
            process.close(timeout_s=.05)
        assert process.poll() is not None, 'timed-out native child must not outlive its owner'
        assert process.database_path.is_file()
        with pytest.raises(RuntimeError, match='forced|timed out|finalize'):
            process.close(timeout_s=.05)
    finally:
        if process.poll() is None:
            process._process.kill(); process._process.wait(timeout=3)


def test_mapping_parameters_are_available_without_creating_runtime_or_files(tmp_path):
    value = backend().mapping_parameters(tmp_path/'unused'/'map.db', '/isolated/s0',
        {'voxel_size_m': .12, 'max_working_nodes': 50, 'max_update_ms': 80, 'max_pending': 3})
    assert value['database_path'] == str(tmp_path/'unused'/'map.db')
    assert value['frame_id'] == 'isolated/s0/base' and value['map_frame_id'] == 'isolated/s0/map'
    assert value['Grid/3D'] == 'true' and value['Rtabmap/MemoryThr'] == '50'
    assert not (tmp_path/'unused').exists()


def test_native_defaults_preserved_and_whitelist_rejects_unknown(tmp_path):
    from plugins.pointcloud_slam_rtabmap import mapping_parameters
    settings=dict(max_pending=2,voxel_size_m=.08,max_working_nodes=300,max_update_ms=100)
    before=mapping_parameters(tmp_path/'map.db','/scope',settings)
    assert 'RGBD/LinearUpdate' not in before and 'RGBD/AngularUpdate' not in before
    changed=mapping_parameters(tmp_path/'map.db','/scope',dict(settings,native_parameters={'RGBD/LinearUpdate':.2}))
    assert changed['RGBD/LinearUpdate']=='0.2'
    assert {k:v for k,v in changed.items() if k!='RGBD/LinearUpdate'}==before
    scheduled=mapping_parameters(tmp_path/'map.db','/scope',dict(settings,native_parameters={'Rtabmap/DetectionRate':0.}))
    assert scheduled['Rtabmap/DetectionRate']=='0.0'
    assert {k:v for k,v in scheduled.items() if k!='Rtabmap/DetectionRate'}==before
    with pytest.raises(ValueError):mapping_parameters(tmp_path/'map.db','/scope',dict(settings,native_parameters={'Optimizer/Strategy':2}))


def test_mapping_child_receives_own_buffered_dds_profile(monkeypatch, tmp_path):
    import os
    from pathlib import Path
    import xml.etree.ElementTree as ET
    module = backend()
    profile = tmp_path/'source.xml'
    profile.write_text('<profiles><transport_descriptors><transport_descriptor>'
        '<transport_id>local</transport_id><type>UDPv4</type>'
        '<interfaceWhiteList><address>127.0.0.1</address></interfaceWhiteList>'
        '</transport_descriptor></transport_descriptors></profiles>')
    monkeypatch.setenv('FASTRTPS_DEFAULT_PROFILES_FILE', str(profile))
    popen = module.subprocess.Popen
    environments = []
    def launch(*args, **kwargs):
        env = kwargs.get('env')
        assert env is not None
        private = Path(env['FASTRTPS_DEFAULT_PROFILES_FILE'])
        assert private != profile and private.is_file()
        tree = ET.parse(private)
        assert tree.find('.//receiveBufferSize').text == '1048576'
        assert tree.find('.//interfaceWhiteList/address').text == '127.0.0.1'
        assert os.environ['FASTRTPS_DEFAULT_PROFILES_FILE'] == str(profile)
        environments.append(private)
        return popen(*args, **kwargs)
    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    process = running(tmp_path)
    process.close(timeout_s=3)
    assert len(environments) == 1


def test_scan_only_proximity_has_an_odometry_guess(tmp_path):
    value = backend().mapping_parameters(tmp_path/'map.db', '/isolated/s0',
        dict(max_pending=2, voxel_size_m=.08, max_working_nodes=100, max_update_ms=100))
    assert value['RGBD/ProximityOdomGuess'] == 'true'
    assert value['RGBD/ProximityPathMaxNeighbors'] == '1'
    assert value['Reg/Strategy'] == '1'
    assert value['approx_sync'] is False


def test_rgb_recovery_fails_before_launch_with_an_unpatched_native_runtime(tmp_path, monkeypatch):
    module = backend()
    executable = tmp_path/'runtime/lib/rtabmap_slam/rtabmap'
    executable.parent.mkdir(parents=True)
    executable.write_text('#!/bin/sh\nexit 0\n'); executable.chmod(0o755)
    monkeypatch.setattr(module, 'resolve_executable', lambda: executable)
    process = module.RtabmapProcess(tmp_path/'session/map.db', '/scope', dict(
        rgb_input={}, max_pending=2, voxel_size_m=.08, max_working_nodes=100, max_update_ms=100))
    with pytest.raises(RuntimeError, match='visual recovery'):
        process.start()
    assert process._process is None


def test_restored_process_preserves_existing_database_nodes(tmp_path):
    script = native_script(tmp_path)
    script.write_text(script.read_text().replace('CREATE TABLE Node', 'CREATE TABLE IF NOT EXISTS Node'))
    path = tmp_path/'restored.db'
    with sqlite3.connect(path) as db:
        db.executescript('CREATE TABLE Node(id INTEGER PRIMARY KEY); INSERT INTO Node VALUES(42);')
    process = backend().RtabmapProcess(path, '/private/resumed', dict(max_pending=2,
        voxel_size_m=.08,max_working_nodes=300,max_update_ms=100), executable=script, restored=True)
    process.start()
    try:
        deadline = time.monotonic()+5
        while not path.with_suffix('.ready').exists():
            assert time.monotonic()<deadline
            time.sleep(.01)
    finally:
        process.close(timeout_s=3)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT id FROM Node ORDER BY id').fetchall() == [(1,), (42,)]
