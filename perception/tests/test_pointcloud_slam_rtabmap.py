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
