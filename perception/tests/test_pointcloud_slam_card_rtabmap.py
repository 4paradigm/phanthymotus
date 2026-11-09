"""Card lifecycle with real catalog persistence and a stand-in native process."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import struct
import threading
import time

import pytest

from plugins import pointcloud_slam_plugin as module
from vision_stubs import _FakeExecutor, _FakeNode, _wait_until


def settings():
    return {'cloud_frame': 'sensor', 'child_frame': '{segment_id}_base', 'clock_id': 'ros',
            'child_from_cloud': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            'reserve_bytes': 0}


def arguments():
    return {'action': 'start', 'input_topics': ['/cloud', '/pose'],
            'input_bindings': {'/cloud': 'sensor/pointcloud2', '/pose': 'state/odom'},
            'odom_interfaces': {'/pose': {'schema': 'motus.odom/1', 'slam_pose': {
                'available': True, 'profile': 'motus_slam/1'}}}}


class Session(_FakeNode):
    def __init__(self, odom_topic, cloud_topic, config, *, session_dir,
                 input_contract=None, capture_stop=None):
        super().__init__('fake_native_session')
        self.inputs = (odom_topic, cloud_topic)
        self.directory = Path(session_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / 'native.db'
        with sqlite3.connect(self.database) as db:
            db.executescript('CREATE TABLE Node(id INTEGER); INSERT INTO Node VALUES(1);'
                             'CREATE TABLE Data(id INTEGER); INSERT INTO Data VALUES(1);')
        self.capture_stop = capture_stop
        self.frozen = False
        self.closed = False
        self.before_close = None
        self.failure = None

    def start(self):
        pass

    def freeze_capture(self):
        self.frozen = True

    def close(self):
        self.freeze_capture()
        if self.before_close:
            self.before_close()
        if self.failure:
            raise OSError(self.failure)
        self.closed = True

    def outputs(self):
        return [{'topic': '/cloud/slam_odom_preview', 'format': 'sensor/mapping'},
                {'topic': '/cloud/slam_odom_preview_status', 'format': 'data/json'}]

    def status(self):
        return {'backend': 'rtabmap', 'odometry': 'tracking', 'accepted_clouds': 1}

    def artifacts(self):
        assert self.closed
        return {'segments': [{'segment_id': 'run', 'frame_id': 'run_map',
                 'database': str(self.database),
                 'snapshot': struct.pack('<fffBIfff', 1, 2, .5, 3, 1, -1, 2, 3)}],
                'statistics': {'accepted_clouds': 1}}


@pytest.fixture
def card(tmp_path, monkeypatch):
    monkeypatch.setattr(module, '_ARCHIVE_DIR', str(tmp_path))
    monkeypatch.setattr(module, 'RtabmapNode', Session, raising=False)
    monkeypatch.setattr(module, 'require_runtime', lambda: None, raising=False)
    plugin = module.PointCloudSlamPlugin(settings(), _FakeExecutor())
    yield plugin
    if plugin._pending is not None:
        plugin._pending.failure = None
    plugin.shutdown()


def call(card, action, **args):
    return card.dispatch('pointcloud_slam', {'action': action, **args})


def ready(card):
    assert card.dispatch('pointcloud_slam', arguments())['state'] == 'ready'


def capture(card):
    ready(card)
    assert call(card, 'start_mapping', map_name='Hall')['state'] == 'running'
    return card._node


def test_card_starts_mapping_with_cloud_and_existing_odom_without_rgbd(card):
    ready(card)
    info = call(card, 'info')
    assert {item['format'] for item in info['topic_in']} == {'sensor/pointcloud2', 'state/odom'}
    assert info['backend'] == 'rtabmap'
    assert call(card, 'start_mapping', map_name='Hall')['state'] == 'running'
    assert card._node.inputs == ('/pose', '/cloud')
    assert call(card, 'info')['topic_out'][0]['format'] == 'sensor/mapping'


def test_legacy_core_roles_use_native_publisher_discovery(card, monkeypatch):
    monkeypatch.setattr(module, 'discover_input_bindings', lambda topics:
        {'/cloud': 'sensor/pointcloud2', '/pose': 'state/odom'})
    assert call(card, 'start', input_topics=['/cloud', '/pose'], input_topic='/cloud')['state'] == 'ready'
    assert len(call(card, 'info')['topic_in']) == 2


def test_missing_native_runtime_prevents_false_ready(card, monkeypatch):
    def missing():
        raise RuntimeError('rtabmap_slam executable unavailable')
    monkeypatch.setattr(module, 'require_runtime', missing)
    result = card.dispatch('pointcloud_slam', arguments())
    assert result['state'] == 'error'
    assert 'rtabmap' in result['message']
    assert not card._executor.nodes


def test_budget_defaults_are_generic_and_not_robot_preset_values(card):
    props = card.get_tools()[0]['configSchema']['properties']
    assert props['max_working_nodes']['default'] == 3000
    assert props['max_update_ms']['default'] == 3000
    # The Bumi preset keeps only the installation contract; budgets stay generic.
    result = call(card, 'config', robot='Bumi')
    assert result['state'] == 'configured'
    assert result['config']['max_working_nodes'] == 3000
    assert result['config']['max_update_ms'] == 3000
    assert result['config']['cloud_frame'] == 'ubuntu_rgb_optical'
    # An explicit card/deployment value is no longer overwritten by the preset.
    result = call(card, 'config', robot='Bumi', max_working_nodes=1500, max_update_ms=2000)
    assert result['state'] == 'configured'
    assert result['config']['max_working_nodes'] == 1500
    assert result['config']['max_update_ms'] == 2000
    assert result['config']['cloud_frame'] == 'ubuntu_rgb_optical'


def test_native_database_save_is_idempotent_and_old_geometry_config_is_not_exposed(card):
    node = capture(card)
    saved = call(card, 'stop_mapping', request_id='save1')
    assert saved['state'] == 'saved'
    assert node.closed and node.destroyed
    replay = call(card, 'stop_mapping', request_id='save1')
    for key in ('state', 'map_id', 'map_name', 'revision', 'request_id'):
        assert replay[key] == saved[key]
    maps = call(card, 'list_maps')['maps']
    assert len(maps) == 1 and maps[0]['name'] == 'Hall'
    assert list(Path(module._ARCHIVE_DIR).glob('maps/*/revisions/*/*.db'))
    props = card.get_tools()[0]['configSchema']['properties']
    assert set(props) == {'robot', 'child_from_cloud'}


def test_card_exposes_map_management_without_annotation_controls(card):
    tool = card.get_tools()[0]
    schema = tool['inputSchema']
    removed = {'tag_place', 'list_tags', 'untag_place'}
    assert not removed.intersection(schema['properties']['action']['enum'])
    assert not removed.intersection(schema['x-action-params'])
    assert not removed.intersection(schema['x-action-lifecycle']['actions'])
    assert not {'tag_id', 'tag_name', 'segment_id', 'x', 'y', 'yaw'}.intersection(schema['properties'])
    assert 'map_plane' not in tool['configSchema']['properties']
    capture(card)
    saved = call(card, 'stop_mapping')
    assert call(card, 'list_maps')['maps'] == [
        {'id': saved['map_id'], 'name': 'Hall', 'revision': saved['revision']}]


@pytest.mark.parametrize('action', ['tag_place', 'list_tags', 'untag_place'])
def test_removed_annotation_actions_cannot_modify_saved_maps(card, action):
    capture(card)
    saved = call(card, 'stop_mapping')
    manifest = Path(module._ARCHIVE_DIR) / 'maps' / saved['map_id'] / 'manifest.json'
    before = manifest.read_bytes()
    assert call(card, action, map_id=saved['map_id'], revision=saved['revision'],
                segment_id='run', tag_name='Entrance', x=1, y=2, tag_id='old') is None
    assert manifest.read_bytes() == before


def test_old_annotation_setting_does_not_prevent_project_start(card):
    plugin = module.PointCloudSlamPlugin({**settings(), 'map_plane': 'xy'}, _FakeExecutor())
    try:
        ready(plugin)
        assert 'map_plane' not in plugin._settings()
        assert call(plugin, 'config', map_plane='xy')['state'] == 'configured'
    finally:
        plugin.shutdown()


def test_project_stop_acknowledges_before_native_close_and_preserves_saving_state(card):
    node = capture(card)
    entered, release = threading.Event(), threading.Event()
    node.before_close = lambda: (entered.set(), release.wait(5))
    try:
        before = time.monotonic()
        result = call(card, 'stop')
        assert time.monotonic() - before < .5
        assert result['state'] == 'idle' and result['capture_stopped'] is True
        assert node.frozen and entered.wait(1)
        assert call(card, 'info')['save_state'] == 'saving'
        assert card.dispatch('pointcloud_slam', arguments())['state'] == 'error'
        assert call(card, 'stop')['save_state'] == 'saving'
    finally:
        release.set()
    card.stop()
    assert call(card, 'info')['save_state'] == 'saved'


def test_failed_save_retains_staging_database_for_retry(card):
    node = capture(card)
    node.failure = 'database finalization failed'
    result = call(card, 'stop_mapping', request_id='retry1')
    assert result['state'] == 'error'
    assert card._pending is node and node.database.exists()
    assert call(card, 'start_mapping', map_name='Other')['state'] == 'error'
    with pytest.raises(RuntimeError, match='finalization'):
        card.shutdown()
    node.failure = None
    saved = call(card, 'stop_mapping', request_id='retry1')
    assert saved['state'] == 'saved'
    assert len(call(card, 'list_maps')['maps']) == 1


def test_rejected_configuration_does_not_poison_the_previous_valid_extrinsic(card):
    ready(card)
    assert call(card, 'config', child_from_cloud='invalid')['state'] == 'error'
    # The rejected value never entered the config, so the previous valid
    # matrix is still what mapping uses.
    assert call(card, 'start_mapping', map_name='Hall')['state'] == 'running'
    assert card._node is not None


@pytest.mark.parametrize('interfaces', [['invalid'], {'/pose': 'invalid'}, {'/pose': {'slam_pose': 'invalid'}}])
def test_malformed_explicit_capability_returns_error_without_starting(card, interfaces):
    result = card.dispatch('pointcloud_slam', {**arguments(), 'odom_interfaces': interfaces})
    assert result['state'] == 'error'
    assert not card._executor.nodes


def test_running_configuration_replay_cannot_change_mapping_or_poison_save(card):
    node = capture(card)
    assert call(card, 'config', child_from_cloud='invalid')['state'] == 'error'
    assert card._node is node
    assert call(card, 'stop_mapping', request_id='save')['state'] == 'saved'


@pytest.mark.parametrize('phase', ['mapping', 'pending', 'saving'])
def test_core_bumi_configuration_replay_is_noop_through_mapping_and_save(card, phase):
    assert call(card, 'config', robot='Bumi')['state'] == 'configured'
    node = capture(card)
    original = card._config
    entered, release = threading.Event(), threading.Event()
    if phase == 'pending':
        node.failure = 'retryable save failure'
        assert call(card, 'stop_mapping', request_id='replay')['state'] == 'error'
        node.failure = None
    elif phase == 'saving':
        node.before_close = lambda: (entered.set(), release.wait(5))
        assert call(card, 'stop_mapping', request_id='replay', background=True)['state'] == 'working'
        assert entered.wait(1)
    try:
        result = call(card, 'config', robot='Bumi', instance_id='existing-slam-card')
        assert result['state'] == 'configured'
        assert result['config'] == original
        assert card._config is original
        assert (card._node or card._pending) is node
    finally:
        release.set()
        if phase == 'pending' and card._pending is not None:
            call(card, 'stop_mapping', request_id='replay')
    if phase == 'saving':
        assert _wait_until(lambda: call(card, 'operation_status', request_id='replay')['state'] == 'saved')
        saved = call(card, 'operation_status', request_id='replay')
    else:
        saved = call(card, 'stop_mapping', request_id='replay')
    assert saved['state'] == 'saved'
    assert call(card, 'list_maps')['maps'][0]['id'] == saved['map_id']
    assert call(card, 'start_mapping', map_name='After replay')['state'] == 'running'


@pytest.mark.parametrize('pending', [False, True])
@pytest.mark.parametrize('updates', [{'robot': 'G1'}, {'robot': 'unknown'}, {'preview_enabled': True}])
def test_active_configuration_changes_are_rejected_without_poisoning_valid_settings(card, pending, updates):
    assert call(card, 'config', robot='Bumi')['state'] == 'configured'
    node = capture(card)
    original = dict(card._config)
    if pending:
        node.failure = 'retryable save failure'
        assert call(card, 'stop_mapping', request_id='unchanged')['state'] == 'error'
        node.failure = None
    assert call(card, 'config', **updates)['state'] == 'error'
    assert card._config == original
    assert call(card, 'stop_mapping', request_id='unchanged')['state'] == 'saved'
    assert call(card, 'start_mapping', map_name='Still valid')['state'] == 'running'


def test_native_revision_records_input_provenance_without_policy_flags(card):
    capture(card)
    saved = call(card, 'stop_mapping', request_id='evidence')
    path = Path(module._ARCHIVE_DIR) / 'maps' / saved['map_id'] / '1.json'
    resolved = json.loads(path.read_text())['resolved']
    assert resolved['resolved_inputs'] == {}
    assert 'input_validation' not in resolved
    assert 'extrinsic_calibrated' not in resolved and 'clock_relation_verified' not in resolved


def test_saving_displayed_defaults_preserves_loaded_resource_budgets():
    plugin = module.PointCloudSlamPlugin({**settings(), 'max_cloud_points': 20000,
                                          'preview_interval_s': 2}, _FakeExecutor())
    try:
        props = plugin.get_tools()[0]['configSchema']['properties']
        displayed = {key: value['default'] for key, value in props.items() if 'default' in value}
        result = call(plugin, 'config', **displayed)
        assert result['state'] == 'configured'
        assert result['config']['max_cloud_points'] == 20000
        assert result['config']['preview_interval_s'] == 2
        # Frames come from deployment config/preset, never from the canvas fields.
        assert result['config']['cloud_frame'] == 'sensor'
        assert set(props) == {'robot', 'child_from_cloud'}
    finally:
        plugin.shutdown()


def test_bumi_profile_wins_over_stale_manual_form_fields(card):
    stale = {key: prop['default'] for key, prop in
             card.get_tools()[0]['configSchema']['properties'].items() if 'default' in prop}
    stale.update(robot='Bumi', cloud_frame='old_sensor', child_from_cloud='invalid')
    result = call(card, 'config', **stale)
    assert result['state'] == 'configured'
    assert result['config']['cloud_frame'] == 'ubuntu_rgb_optical'
    assert result['config']['child_frame'] == '{segment_id}_base'
    assert result['config']['clock_id'] == 'ros'
    assert result['config']['child_from_cloud'] == [[0, 0, 1, 0], [-1, 0, 0, 0],
                                                  [0, -1, 0, 0], [0, 0, 0, 1]]
    assert result['config']['max_cloud_points'] == 20000
    for key in ('input_policy', 'clock_relation_verified', 'extrinsic_calibrated'):
        assert key not in result['config']
    capture(card)
    saved = call(card, 'stop_mapping', request_id='bumi-profile')
    manifest = Path(module._ARCHIVE_DIR) / 'maps' / saved['map_id'] / '1.json'
    resolved = json.loads(manifest.read_text())['resolved']
    assert resolved['resolved_inputs'] == {} and 'input_validation' not in resolved


@pytest.mark.parametrize('selector', [{'robot': '其他'}, {}])
def test_manual_and_legacy_config_after_bumi_do_not_retain_profile_selection(card, selector):
    assert call(card, 'config', robot='Bumi')['state'] == 'configured'
    result = call(card, 'config', **settings(), **selector)
    assert result['state'] == 'configured'
    assert result['config']['robot'] == '其他'
    assert result['config']['cloud_frame'] == 'sensor'
    assert result['config']['child_from_cloud'] == [[1, 0, 0, 0], [0, 1, 0, 0],
                                                  [0, 0, 1, 0], [0, 0, 0, 1]]
    assert 'input_policy' not in result['config']
    assert 'input_validation' not in call(card, 'info')
    ready(card)


@pytest.mark.parametrize('robot', ['G1', 'Tianyi'])
def test_pending_robot_selection_clears_bumi_facts_and_explains_missing_odometry(card, robot):
    assert call(card, 'config', robot='Bumi')['state'] == 'configured'
    result = call(card, 'config', robot=robot, child_from_cloud='stale invalid matrix')
    assert result['state'] == 'configured'
    assert result['config']['robot'] == robot
    assert not result['config'].get('cloud_frame')
    assert not result['config'].get('child_from_cloud')
    assert 'input_validation' not in call(card, 'info')
    result = card.dispatch('pointcloud_slam', arguments())
    assert result['state'] == 'error'
    assert robot in result['message'] and 'odom' in result['message']
    assert card._node is None


def test_unknown_robot_selection_is_rejected(card):
    result = call(card, 'config', robot='unknown')
    assert result['state'] == 'error'
    assert 'robot' in result['message']


def test_switching_ready_bumi_to_pending_robot_cannot_start_a_native_session(card):
    assert call(card, 'config', robot='Bumi')['state'] == 'configured'
    ready(card)
    assert call(card, 'config', robot='G1')['state'] == 'configured'
    result = call(card, 'start_mapping', map_name='Hall')
    assert result['state'] == 'error'
    assert 'G1' in result['message'] and 'odom' in result['message']
    assert card._node is None


def test_manual_config_after_bumi_uses_manual_sensor_facts_without_flags(card):
    assert call(card, 'config', robot='Bumi')['state'] == 'configured'
    manual = {key: value for key, value in settings().items() if key in
              ('cloud_frame', 'child_frame', 'clock_id', 'child_from_cloud')}
    result = call(card, 'config', **manual)
    assert result['state'] == 'configured'
    assert result['config']['robot'] == '其他'
    assert result['config']['cloud_frame'] == 'sensor'
    assert result['config']['child_from_cloud'] == [[1, 0, 0, 0], [0, 1, 0, 0],
                                                  [0, 0, 1, 0], [0, 0, 0, 1]]
    assert card.dispatch('pointcloud_slam', arguments())['state'] == 'ready'


@pytest.mark.parametrize('action', ['start', 'stop', 'info', 'config', 'operation_status'])
def test_core_json_schema_validation_accepts_lifecycle_actions(card, action):
    jsonschema = pytest.importorskip('jsonschema')
    jsonschema.validate({'action': action}, card.get_tools()[0]['inputSchema'])


@pytest.mark.parametrize('manual_pending', [False, True])
def test_project_stop_preserves_acknowledged_save_request_id(card, monkeypatch, manual_pending):
    capture(card)
    entered, release = threading.Event(), threading.Event()
    if manual_pending:
        original = card._dispatch
        def paused(action, args):
            if action == 'stop_mapping':
                entered.set()
                assert release.wait(5)
            return original(action, args)
        monkeypatch.setattr(card, '_dispatch', paused)
        assert call(card, 'stop_mapping', request_id='manual', background=True)['state'] == 'working'
        assert entered.wait(1)
    try:
        ack = call(card, 'stop')
        request = ack['save_result']['request_id']
    finally:
        release.set()
    card.stop()
    assert call(card, 'info')['save_result']['request_id'] == request
    assert call(card, 'operation_status', request_id=request)['state'] == 'saved'


def test_later_mapping_does_not_reuse_completed_background_save_id(card):
    capture(card)
    assert call(card, 'stop_mapping', request_id='first', background=True)['state'] == 'working'
    assert _wait_until(lambda: call(card, 'operation_status', request_id='first')['state'] == 'saved')
    assert call(card, 'start_mapping', map_name='Other')['state'] == 'running'
    ack = call(card, 'stop')
    card.stop()
    assert ack['save_result']['request_id'] != 'first'
    assert call(card, 'info')['save_state'] == 'saved'
    assert len(call(card, 'list_maps')['maps']) == 2


def test_stop_during_later_constructor_does_not_inherit_previous_map_save_id(card, monkeypatch):
    capture(card)
    assert call(card, 'stop_mapping', request_id='first')['state'] == 'saved'
    entered, release = threading.Event(), threading.Event()
    class SlowSession(Session):
        def __init__(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            super().__init__(*args, **kwargs)
    monkeypatch.setattr(module, 'RtabmapNode', SlowSession)
    starter = threading.Thread(target=lambda: call(card, 'start_mapping', map_name='Second'))
    starter.start()
    try:
        assert entered.wait(1)
        ack = call(card, 'stop')
    finally:
        release.set()
        starter.join(2)
        card.stop()
    assert ack['save_result']['request_id'] != 'first'
    assert call(card, 'info')['save_state'] == 'saved'
    assert len(call(card, 'list_maps')['maps']) == 2


def test_stop_before_later_mapping_admission_cancels_constructor(card, monkeypatch):
    from plugins import pointcloud_slam_library as library
    capture(card)
    assert call(card, 'stop_mapping', request_id='first')['state'] == 'saved'
    entered, release = threading.Event(), threading.Event()
    original = library.map_entries
    pause = [True]
    def slow(root):
        if pause[0]:
            pause[0] = False
            entered.set()
            assert release.wait(5)
        return original(root)
    monkeypatch.setattr(library, 'map_entries', slow)
    results = []
    starter = threading.Thread(target=lambda: results.append(call(card, 'start_mapping', map_name='Canceled')))
    starter.start()
    try:
        assert entered.wait(1)
        call(card, 'stop')
    finally:
        release.set()
        starter.join(2)
        card.stop()
    assert results[0]['state'] == 'idle'
    assert call(card, 'info')['save_state'] == 'saved'
    assert len(call(card, 'list_maps')['maps']) == 1


def test_saved_view_opened_after_project_stop_is_visible_and_closes_on_repeated_stop(card):
    capture(card)
    call(card, 'stop')
    card.stop()
    saved = call(card, 'list_maps')['maps'][0]
    selected = call(card, 'select_map', map_id=saved['id'])
    assert selected['state'] == 'map_selected'
    assert selected['topic_out'] and selected['topic_out'][0]['format'] == 'sensor/mapping'
    assert call(card, 'info')['topic_out'] == selected['topic_out']
    view = card._view
    call(card, 'stop')
    card.stop()
    assert card._view is None and view.destroyed


def test_project_stop_cancels_mapping_constructor_that_has_not_returned(card, monkeypatch):
    ready(card)
    entered, release = threading.Event(), threading.Event()
    class SlowSession(Session):
        def __init__(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            super().__init__(*args, **kwargs)
    monkeypatch.setattr(module, 'RtabmapNode', SlowSession)
    results = []
    starter = threading.Thread(target=lambda: results.append(call(card, 'start_mapping', map_name='Late')))
    starter.start()
    try:
        assert entered.wait(1)
        before = time.monotonic()
        result = call(card, 'stop')
        assert time.monotonic() - before < .5
        assert result['capture_stopped'] is True
    finally:
        release.set()
        starter.join(2)
        card.stop()
    assert results[0]['state'] == 'idle'
    assert not card._executor.nodes and card._node is None


def test_shutdown_waits_for_manual_database_save_without_holding_plugin_lock(card):
    node = capture(card)
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    node.before_close = lambda: (entered.set(), release.wait(5))
    saving = threading.Thread(target=lambda: call(card, 'stop_mapping', request_id='manual'))
    closing = threading.Thread(target=lambda: (card.shutdown(), closed.set()))
    saving.start()
    try:
        assert entered.wait(1)
        closing.start()
        assert not closed.wait(.05)
    finally:
        release.set()
        saving.join(2)
        closing.join(2)
    assert closed.is_set() and node.destroyed
    assert len(call(card, 'list_maps')['maps']) == 1


def test_fresh_card_can_view_native_map_without_requiring_backend_or_live_inputs(card, monkeypatch):
    capture(card)
    saved = call(card, 'stop_mapping', request_id='save')
    def missing():
        raise RuntimeError('backend unavailable')
    monkeypatch.setattr(module, 'require_runtime', missing)
    fresh = module.PointCloudSlamPlugin({}, _FakeExecutor())
    try:
        selected = call(fresh, 'select_map', map_id=saved['map_id'])
        assert selected['state'] == 'map_selected' and selected['tracking'] is False
        assert selected['snapshot_available'] is True
        assert call(fresh, 'info')['topic_out'] == selected['topic_out']
    finally:
        fresh.shutdown()


@pytest.mark.parametrize('stop_method', ['dispatch', 'stop', 'shutdown'])
def test_stop_waits_for_view_already_admitted_before_selection_lock(card, monkeypatch, stop_method):
    capture(card)
    call(card, 'stop')
    card.stop()
    saved = call(card, 'list_maps')['maps'][0]
    entered, release = threading.Event(), threading.Event()
    original = card._dispatch
    def paused(action, args):
        if action == 'select_map':
            entered.set()
            assert release.wait(5)
        return original(action, args)
    monkeypatch.setattr(card, '_dispatch', paused)
    select = threading.Thread(target=lambda: call(card, 'select_map', map_id=saved['id']))
    select.start()
    assert entered.wait(1)
    finished = threading.Event()
    stop = threading.Thread(target=lambda: (call(card, 'stop') if stop_method == 'dispatch'
        else getattr(card, stop_method)(), finished.set()))
    stop.start()
    try:
        if stop_method == 'dispatch':
            assert finished.wait(.5)
        else:
            assert not finished.wait(.05)
        assert call(card, 'select_map', map_id=saved['id'])['state'] == 'error'
    finally:
        release.set()
        select.join(2)
        stop.join(2)
        card.stop()
    assert not card._executor.nodes and card._view is None


def test_canvas_config_surface_is_robot_and_extrinsic_only(card):
    props = card.get_tools()[0]['configSchema']['properties']
    assert set(props) == {'robot', 'child_from_cloud'}
    assert props['robot']['enum'] == ['Bumi', 'G1', 'Tianyi', '其他']
    assert props['robot']['scope'] == 'instance'
    assert props['child_from_cloud']['scope'] == 'instance'
    assert props['child_from_cloud']['title'] and props['child_from_cloud']['description']
    assert 'advanced' not in props['robot'] and 'advanced' not in props['child_from_cloud']
    for removed in ('cloud_frame', 'child_frame', 'clock_id', 'voxel_size_m', 'max_working_nodes',
                    'max_update_ms', 'preview_interval_s', 'preview_max_points', 'input_policy',
                    'clock_relation_verified', 'extrinsic_calibrated'):
        assert removed not in props


def test_retired_canvas_config_keys_are_ignored_not_fatal(card):
    original = deepcopy(card._config)
    result = call(card, 'config', robot='其他', input_policy='experimental',
                  clock_relation_verified=True, extrinsic_calibrated=False,
                  preview_enabled=True, cloud_frame='old')
    assert result['state'] == 'configured'
    assert result['config'] == original
    for key in ('input_policy', 'clock_relation_verified', 'extrinsic_calibrated', 'preview_enabled'):
        assert key not in card._config


def test_deployment_resource_overrides_survive_robot_switch(card):
    plugin = module.PointCloudSlamPlugin({**settings(), 'max_gap_ms': 123,
                                          'voxel_size_m': .05,
                                          'preview_max_points': 20000}, _FakeExecutor())
    props = plugin.get_tools()[0]['configSchema']['properties']
    displayed = {key: value['default'] for key, value in props.items() if 'default' in value}
    try:
        assert plugin._settings()['max_gap_ms'] == 123
        assert call(plugin, 'config', **displayed)['state'] == 'configured'
        assert plugin._settings()['max_gap_ms'] == 123
        assert call(plugin, 'config', robot='Bumi')['state'] == 'configured'
        # A preset owns the sensor facts and its own resource values while selected.
        assert plugin._settings()['max_gap_ms'] == 200
        assert call(plugin, 'config', robot='其他')['state'] == 'configured'
        # Switching back re-bases on the deployment config, not the preset.
        assert plugin._settings()['max_gap_ms'] == 123
        assert plugin._settings()['voxel_size_m'] == .05
        assert plugin._settings()['preview_max_points'] == 20000
    finally:
        plugin.shutdown()


def test_missing_frames_and_extrinsic_fail_at_mapping_start_only(tmp_path, monkeypatch):
    monkeypatch.setattr(module, '_ARCHIVE_DIR', str(tmp_path))
    monkeypatch.setattr(module, 'RtabmapNode', Session, raising=False)
    monkeypatch.setattr(module, 'require_runtime', lambda: None, raising=False)
    card = module.PointCloudSlamPlugin({}, _FakeExecutor())
    try:
        assert call(card, 'config', child_from_cloud=json.dumps([[1, 0, 0, 0], [0, 1, 0, 0],
                                                                [0, 0, 1, 0], [0, 0, 0, 1]]))['state'] == 'configured'
        assert card.dispatch('pointcloud_slam', arguments())['state'] == 'ready'
        result = call(card, 'start_mapping', map_name='Hall')
        assert result['state'] == 'error' and 'cloud_frame' in result['message']
        assert card._node is None
    finally:
        card.shutdown()
    framed = module.PointCloudSlamPlugin({'cloud_frame': 'sensor', 'child_frame': '{segment_id}_base',
                                          'clock_id': 'ros'}, _FakeExecutor())
    try:
        assert framed.dispatch('pointcloud_slam', arguments())['state'] == 'ready'
        result = call(framed, 'start_mapping', map_name='Hall')
        assert result['state'] == 'error' and 'child_from_cloud' in result['message']
    finally:
        framed.shutdown()


def test_input_metadata_route_still_prepares_and_starts(card):
    from test_pointcloud_slam_input import declarations
    declared = declarations()
    metadata = {'/pose': declared['/o'], '/cloud': declared['/c']}
    metadata['/pose'] = {**metadata['/pose'], 'topic': '/pose'}
    metadata['/cloud'] = {**metadata['/cloud'], 'topic': '/cloud'}
    for port in metadata.values():
        port['clock'] = {'mapping': 'shared_camera_mapper', 'verified': True, 'uncertainty_ns': 10}
    metadata['/cloud']['extrinsic']['calibrated'] = True
    result = card.dispatch('pointcloud_slam', {**arguments(), 'input_metadata': metadata})
    assert result['state'] == 'ready'
    settings = card._settings()
    assert settings['cloud_frame'] == 'camera'
    assert settings['child_frame'] == 'body'
    assert settings['clock_id'] == 'clock'
    assert call(card, 'start_mapping', map_name='Declared')['state'] == 'running'
    assert 'input_validation' not in call(card, 'info')


def test_bumi_preset_starts_in_legacy_core_mode_without_any_flags(card, monkeypatch):
    assert call(card, 'config', robot='Bumi')['state'] == 'configured'
    monkeypatch.setattr(module, 'discover_input_bindings', lambda topics:
                        {'/cloud': 'sensor/pointcloud2', '/pose': 'state/odom'})
    result = call(card, 'start', input_topics=['/cloud', '/pose'], input_topic='/cloud')
    assert result['state'] == 'ready'
    assert card._legacy_core is True
    assert call(card, 'info')['input_compatibility']['mode'] == 'legacy_core'
    assert call(card, 'start_mapping', map_name='Legacy Bumi')['state'] == 'running'
    settings = card._session_settings
    assert settings['cloud_frame'] == 'ubuntu_rgb_optical'
    assert settings['child_frame'] == '{segment_id}_base'
    assert settings['clock_id'] == 'ros'
    assert settings['child_from_cloud'] == [[0, 0, 1, 0], [-1, 0, 0, 0],
                                            [0, -1, 0, 0], [0, 0, 0, 1]]
    for key in ('input_policy', 'clock_relation_verified', 'extrinsic_calibrated', 'input_validation'):
        assert key not in settings


def test_deployment_robot_default_survives_config_replay_without_selector(card):
    plugin = module.PointCloudSlamPlugin({'robot': 'Bumi'}, _FakeExecutor())
    try:
        props = plugin.get_tools()[0]['configSchema']['properties']
        assert props['robot']['default'] == 'Bumi'
        # An old saved row full of retired fields replays harmlessly.
        result = call(plugin, 'config', input_policy='experimental', max_frames=5,
                      clock_relation_verified=False)
        assert result['state'] == 'configured'
        assert result['config']['robot'] == 'Bumi'
        assert result['config']['cloud_frame'] == 'ubuntu_rgb_optical'
        assert 'input_policy' not in result['config']
    finally:
        plugin.shutdown()
