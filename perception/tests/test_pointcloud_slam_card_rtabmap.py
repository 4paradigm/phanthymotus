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
        return [{'topic': '/pointcloud_slam/map', 'format': 'sensor/mapping'}]

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
    assert {item['format'] for item in info['topic_in'] if not item.get('optional')} == {'sensor/pointcloud2', 'state/odom'}
    assert info['backend'] == 'rtabmap'
    assert call(card, 'start_mapping', map_name='Hall')['state'] == 'running'
    assert card._node.inputs == ('/pose', '/cloud')
    assert call(card, 'info')['topic_out'] == card.get_tools()[0]['topic_out']
    assert len(card.get_tools()[0]['topic_out']) == 1


def test_action_identifiers_match_canvas_display_names(card):
    tool = card.get_tools()[0]
    action = tool['inputSchema']['properties']['action']
    options = action['oneOf']

    assert [item['const'] for item in options] == action['enum']
    assert all(item['title'] == item['const'] for item in options)

    params = tool['inputSchema']['x-action-params']
    assert set(params) == set(module._ACTIONS)
    assert {name: item['params'] for name, item in params.items()} == module._ACTIONS


def test_legacy_core_roles_use_native_publisher_discovery(card, monkeypatch):
    monkeypatch.setattr(module, 'discover_input_bindings', lambda topics:
        {'/cloud': 'sensor/pointcloud2', '/pose': 'state/odom'})
    assert call(card, 'start', input_topics=['/cloud', '/pose'], input_topic='/cloud')['state'] == 'ready'
    assert len(call(card, 'info')['topic_in']) == 3


def test_missing_native_runtime_prevents_false_ready(card, monkeypatch):
    def missing():
        raise RuntimeError('rtabmap_slam executable unavailable')
    monkeypatch.setattr(module, 'require_runtime', missing)
    result = card.dispatch('pointcloud_slam', arguments())
    assert result['state'] == 'error'
    assert 'rtabmap' in result['message']
    assert not card._executor.nodes


def test_budget_defaults_are_generic_and_not_robot_preset_values(tmp_path):
    # Legacy deployment keys must never replace the connected sensor frame.
    plugin = module.PointCloudSlamPlugin({'robot': 'Bumi'}, _FakeExecutor())
    try:
        assert 'robot' not in plugin._settings()
        assert plugin._settings()['max_working_nodes'] == 3000
        assert plugin._settings()['max_update_ms'] == 3000
        assert 'cloud_frame' not in plugin._settings()
        assert list(plugin.get_tools()[0]['configSchema']['properties']) == ['child_from_cloud']
    finally:
        plugin.shutdown()
    tuned = module.PointCloudSlamPlugin(
        {'robot': 'Bumi', 'max_working_nodes': 1500, 'max_update_ms': 2000},
        _FakeExecutor())
    try:
        assert tuned._settings()['max_working_nodes'] == 1500
        assert tuned._settings()['max_update_ms'] == 2000
        assert 'cloud_frame' not in tuned._settings()
    finally:
        tuned.shutdown()

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
    assert set(props) == {'child_from_cloud'}


def test_card_exposes_map_management_without_annotation_controls(card):
    tool = card.get_tools()[0]
    schema = tool['inputSchema']
    removed = {'tag_place', 'list_tags', 'untag_place'}
    assert not removed.intersection(schema['properties']['action']['enum'])
    assert not removed.intersection(schema['x-action-params'])
    assert not removed.intersection(schema['x-action-lifecycle']['actions'])
    assert 'close_map_view' not in schema['properties']['action']['enum']
    assert 'close_map_view' not in schema['x-map-manager']['actions']
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


def test_new_project_start_saves_stale_mapping_before_repreparing(card):
    # Intelligent control may stop without calling this card's stop action.
    old_session = capture(card)
    result = card.dispatch('pointcloud_slam', arguments())
    assert result['state'] == 'ready'
    assert old_session.closed and old_session.destroyed
    assert call(card, 'list_maps')['maps'][0]['name'] == 'Hall'
    assert call(card, 'info')['state'] == 'ready'
    assert call(card, 'start_mapping', map_name='Second')['state'] == 'running'


def test_new_project_start_retries_prior_failed_save(card):
    session = capture(card)
    session.failure = 'transient close error'
    assert call(card, 'stop_mapping', request_id='previous')['state'] == 'error'
    assert session.database.exists()
    session.failure = None
    result = card.dispatch('pointcloud_slam', arguments())
    assert result['state'] == 'ready'
    assert session.closed
    assert call(card, 'list_maps')['maps'][0]['name'] == 'Hall'


def test_abandon_failed_pending_retains_recovery_files_and_allows_restart(card):
    session = capture(card)
    assert call(card, 'abandon_pending')['state'] == 'error'  # cannot discard a live map
    session.failure = 'native database cannot finalize'
    assert call(card, 'stop_mapping', request_id='broken')['state'] == 'error'
    directory = session.directory
    assert directory.exists() and session.database.exists()
    assert card.dispatch('pointcloud_slam', arguments())['state'] == 'error'
    result = call(card, 'abandon_pending')
    assert result['state'] == 'recovery_required'
    assert result['recovery']['path'] == str(directory)
    assert result['recovery']['map_name'] == 'Hall'
    assert session.destroyed and card._pending is None
    assert directory.exists() and session.database.exists()
    info = call(card, 'info')
    assert info['last_recovery']['path'] == str(directory)
    assert call(card, 'operation_status', request_id='broken')['state'] == 'error'
    assert card.dispatch('pointcloud_slam', arguments())['state'] == 'ready'


def test_rejected_configuration_does_not_poison_the_previous_valid_extrinsic(card):
    ready(card)
    assert call(card, 'config', child_from_cloud='invalid')['state'] == 'error'
    # The rejected value never entered the config, so the previous valid
    # matrix is still what mapping uses.
    assert call(card, 'start_mapping', map_name='Hall')['state'] == 'running'
    assert card._node is not None


@pytest.mark.parametrize('interfaces', [
    ['invalid'],
    {'/pose': 'invalid'},
    {'/pose': {'slam_pose': 'invalid'}},
    {'/pose': {'schema': 'motus.odom/1',
               'slam_pose': {'available': False, 'profile': 'motus_slam/1'}}},
    {'/pose': {'schema': 'motus.odom/1',
               'slam_pose': {'available': True, 'profile': 'pose_drift/1'}}},
])
def test_malformed_explicit_capability_returns_error_without_starting(card, interfaces):
    result = card.dispatch('pointcloud_slam', {**arguments(), 'odom_interfaces': interfaces})
    if not isinstance(interfaces, dict):
        assert result['state'] == 'error'
    else:
        assert result['state'] == 'ready'
        assert call(card, 'start_mapping', map_name='NumericPose')['state'] == 'running'


@pytest.mark.parametrize('interfaces_present', [False, True])
def test_missing_odom_capability_is_pending_not_malformed(card, interfaces_present):
    args = arguments()
    if interfaces_present:
        args['odom_interfaces'] = {}
    else:
        args.pop('odom_interfaces')
    result = card.dispatch('pointcloud_slam', args)
    assert result['state'] == 'ready'
    diagnostic = call(card, 'info')['input_diagnostics']
    assert any(item['field'] == 'odom_interface' for item in diagnostic)
    assert not card._executor.nodes


def test_running_configuration_replay_cannot_change_mapping_or_poison_save(card):
    node = capture(card)
    assert call(card, 'config', child_from_cloud='invalid')['state'] == 'error'
    assert card._node is node
    assert call(card, 'stop_mapping', request_id='save')['state'] == 'saved'


@pytest.mark.parametrize('phase', ['mapping', 'pending', 'saving'])
def test_stale_canvas_configuration_replay_is_noop_through_mapping_and_save(card, phase):
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
        # A stale saved-canvas robot (and removed keys) is ignored and cannot
        # alter active deployment settings while a map session is live.
        result = call(card, 'config', robot='Bumi', input_policy='experimental',
                      preview_enabled=True, instance_id='existing-slam-card')
        assert result['state'] == 'configured'
        assert result['config'] == original
        assert card._config is original
        assert 'robot' not in card._config
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
def test_active_extrinsic_change_is_rejected_without_poisoning_valid_settings(card, pending):
    node = capture(card)
    original = dict(card._config)
    if pending:
        node.failure = 'retryable save failure'
        assert call(card, 'stop_mapping', request_id='unchanged')['state'] == 'error'
        node.failure = None
    matrix = json.dumps([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    assert call(card, 'config', child_from_cloud=matrix)['state'] == 'error'
    assert card._config == original
    assert call(card, 'stop_mapping', request_id='unchanged')['state'] == 'saved'
    assert call(card, 'start_mapping', map_name='Still valid')['state'] == 'running'


@pytest.mark.parametrize('pending', [False, True])
def test_stale_canvas_robot_is_ignored_even_while_mapping(card, pending):
    node = capture(card)
    original = dict(card._config)
    if pending:
        node.failure = 'retryable save failure'
        assert call(card, 'stop_mapping', request_id='stale')['state'] == 'error'
        node.failure = None
    assert call(card, 'config', robot='Bumi', input_policy='experimental',
                clock_relation_verified=True)['state'] == 'configured'
    assert card._config == original and 'robot' not in card._config
    assert call(card, 'stop_mapping', request_id='stale')['state'] == 'saved'
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
        assert set(props) == {'child_from_cloud'}
        displayed = {key: value['default'] for key, value in props.items() if 'default' in value}
        result = call(plugin, 'config', **displayed)
        assert result['state'] == 'configured'
        assert result['config']['max_cloud_points'] == 20000
        assert result['config']['preview_interval_s'] == 2
        # Frame values come from deployment config or live inputs, not stale canvas fields.
        assert result['config']['cloud_frame'] == 'sensor'
    finally:
        plugin.shutdown()


def test_legacy_robot_name_cannot_override_explicit_installation_geometry(card):
    plugin = module.PointCloudSlamPlugin({**settings(), 'robot': 'Bumi'}, _FakeExecutor())
    try:
        assert 'robot' not in plugin._config
        assert plugin._settings()['cloud_frame'] == 'sensor'
        assert plugin._settings()['child_from_cloud'] == settings()['child_from_cloud']
        assert call(plugin, 'config', child_from_cloud='invalid', robot='Bumi')['state'] == 'error'
        assert call(plugin, 'config', robot='Tianyi')['state'] == 'configured'
        assert plugin._settings()['child_from_cloud'] == settings()['child_from_cloud']
        ready(plugin)
    finally:
        plugin.shutdown()

def test_stale_robot_canvas_value_has_no_effect_on_geometry(card):
    plugin = module.PointCloudSlamPlugin({**settings(), 'robot': 'Bumi'}, _FakeExecutor())
    try:
        result = call(plugin, 'config', robot='其他')
        assert result['state'] == 'configured'
        assert 'robot' not in plugin._config
        assert plugin._settings()['cloud_frame'] == 'sensor'
        assert plugin._settings()['child_frame'] == '{segment_id}_base'
        assert plugin._settings()['clock_id'] == 'ros'
        assert plugin._settings()['child_from_cloud'] == settings()['child_from_cloud']
        ready(plugin)
    finally:
        plugin.shutdown()

@pytest.mark.parametrize('robot', ['G1', 'Tianyi'])
def test_retired_robot_selector_does_not_supply_missing_runtime_inputs(card, robot, monkeypatch):
    plugin = module.PointCloudSlamPlugin({'robot': robot}, _FakeExecutor())
    def no_live_inputs(*args, **kwargs):
        raise ValueError('tracking odometry and capture-timed PointCloud2 unavailable')
    monkeypatch.setattr(module, 'discover_runtime_geometry', no_live_inputs)
    try:
        assert plugin.dispatch('pointcloud_slam', arguments())['state'] == 'ready'
        assert 'robot' not in plugin._config
        result = call(plugin, 'start_mapping', map_name='Hall')
        assert result['state'] == 'error'
        assert 'tracking odometry' in result['message']
        assert plugin._node is None
        assert call(plugin, 'config', robot='Bumi')['state'] == 'configured'
        assert 'robot' not in plugin._config
    finally:
        plugin.shutdown()

def test_unknown_deployment_robot_is_ignored():
    # Deprecated robot model selectors are ignored rather than required.
    plugin = module.PointCloudSlamPlugin({'robot': 'unknown'}, _FakeExecutor())
    try:
        assert 'robot' not in plugin._settings()
        assert plugin.get_tools()[0]['topic_out'][0]['format'] == 'sensor/mapping'
    finally:
        plugin.shutdown()

def test_stale_canvas_robot_cannot_move_a_ready_deployment_preset(card):
    plugin = module.PointCloudSlamPlugin({**settings(), 'robot': 'Bumi'}, _FakeExecutor())
    try:
        ready(plugin)
        assert call(plugin, 'config', robot='G1')['state'] == 'configured'
        assert 'robot' not in plugin._config
        assert plugin._settings()['cloud_frame'] == 'sensor'
        assert call(plugin, 'start_mapping', map_name='Hall')['state'] == 'running'
    finally:
        plugin.shutdown()

def test_stale_canvas_robot_cannot_change_the_其他_deployment(card):
    result = call(card, 'config', robot='Bumi')
    assert result['state'] == 'configured'
    assert 'robot' not in result['config']
    assert result['config']['cloud_frame'] == 'sensor'
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


def test_canvas_config_surface_is_child_from_cloud_only(card):
    props = card.get_tools()[0]['configSchema']['properties']
    assert list(props) == ['child_from_cloud']
    assert props['child_from_cloud']['scope'] == 'instance'
    assert props['child_from_cloud']['title'] and props['child_from_cloud']['description']
    assert 'advanced' not in props['child_from_cloud']
    for removed in ('robot', 'cloud_frame', 'child_frame', 'clock_id', 'voxel_size_m',
                    'max_working_nodes', 'max_update_ms', 'preview_interval_s',
                    'preview_max_points', 'input_policy', 'clock_relation_verified',
                    'extrinsic_calibrated'):
        assert removed not in props


def test_retired_and_stale_canvas_config_keys_are_ignored_not_fatal(card):
    original = deepcopy(card._config)
    result = call(card, 'config', robot='Bumi', input_policy='experimental',
                  clock_relation_verified=True, extrinsic_calibrated=False,
                  preview_enabled=True, cloud_frame='old')
    assert result['state'] == 'configured'
    assert result['config'] == original
    assert 'robot' not in result['config']
    for key in ('input_policy', 'clock_relation_verified', 'extrinsic_calibrated', 'preview_enabled'):
        assert key not in card._config


def test_deployment_resource_overrides_survive_stale_canvas_robot(card):
    plugin = module.PointCloudSlamPlugin({**settings(), 'max_gap_ms': 123,
                                          'voxel_size_m': .05,
                                          'preview_max_points': 20000}, _FakeExecutor())
    props = plugin.get_tools()[0]['configSchema']['properties']
    assert set(props) == {'child_from_cloud'}
    displayed = {key: value['default'] for key, value in props.items() if 'default' in value}
    try:
        assert plugin._settings()['max_gap_ms'] == 123
        assert call(plugin, 'config', **displayed)['state'] == 'configured'
        assert plugin._settings()['max_gap_ms'] == 123
        # A stale robot-model field is ignored; deployment overrides remain authoritative.
        assert call(plugin, 'config', robot='Bumi')['state'] == 'configured'
        assert 'robot' not in plugin._config
        assert plugin._settings()['max_gap_ms'] == 123
        assert plugin._settings()['voxel_size_m'] == .05
        assert plugin._settings()['preview_max_points'] == 20000
    finally:
        plugin.shutdown()


def test_missing_frames_and_extrinsic_fail_at_mapping_start_only(tmp_path, monkeypatch):
    monkeypatch.setattr(module, '_ARCHIVE_DIR', str(tmp_path))
    monkeypatch.setattr(module, 'RtabmapNode', Session, raising=False)
    monkeypatch.setattr(module, 'require_runtime', lambda: None, raising=False)
    def no_samples(*args, **kwargs):
        raise ValueError('cloud_frame cannot be inferred without valid live data')
    monkeypatch.setattr(module, 'discover_runtime_geometry', no_samples)
    card = module.PointCloudSlamPlugin({}, _FakeExecutor())
    try:
        assert call(card, 'config', child_from_cloud=json.dumps(settings()['child_from_cloud']))['state'] == 'configured'
        assert card.dispatch('pointcloud_slam', arguments())['state'] == 'ready'
        result = call(card, 'start_mapping', map_name='Hall')
        assert result['state'] == 'error' and 'cloud_frame' in result['message']
        assert card._node is None
    finally:
        card.shutdown()
    def no_mount(*args, **kwargs):
        raise ValueError('child_from_cloud requires measured sensor mounting')
    monkeypatch.setattr(module, 'discover_runtime_geometry', no_mount)
    framed = module.PointCloudSlamPlugin(
        {'cloud_frame': 'sensor', 'child_frame': '{segment_id}_base', 'clock_id': 'ros'},
        _FakeExecutor())
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


def test_manual_matrix_priority_survives_live_producer_sidecars(card):
    from test_pointcloud_slam_input import declarations
    from test_pointcloud_slam_stream import stream,odom
    from test_pointcloud_slam_dynamic import tf_packet
    manual=settings()['child_from_cloud'];manual[0][3]=.6
    assert call(card,'config',child_from_cloud=manual)['state']=='configured'
    values=declarations()
    metadata={'/pose':values['/o'],'/cloud':values['/c']}
    assert card.dispatch('pointcloud_slam',{**arguments(),'input_metadata':metadata})['state']=='ready'
    selected=card._settings()
    s=stream(input_contract=card._resolved_inputs,child_from_cloud=selected['child_from_cloud'],
             extrinsic_mode=selected['extrinsic_mode'])
    s.odom(odom(1000),now=1.)
    s.cloud_metadata(tf_packet(1000,x=2),now=1.)
    s.cloud(1_000_000_000,'camera',[[1,0,0]],now=1.)
    assert len(s.accepted)==1 and s.accepted[0].child_from_cloud[0,3]==.6



def test_initializing_source_declarations_do_not_block_project_start(card):
    args = arguments()
    args['input_metadata'] = {}
    args['odom_interfaces'] = {}
    result = card.dispatch('pointcloud_slam', args)
    assert result['state'] == 'ready'
    assert card._resolved_inputs is None
    diagnostics = call(card, 'info')['input_diagnostics']
    assert {d['field'] for d in diagnostics} >= {'odom_interface', 'input_metadata'}
    # Session geometry was provided through the unchanged deployment config.
    assert call(card, 'start_mapping', map_name='Initializing')['state'] == 'running'


def test_different_clock_labels_do_not_gate_project_start(card):
    from test_pointcloud_slam_input import declarations
    declared = declarations()
    pose = {**declared['/o'], 'topic': '/pose'}
    cloud = {**declared['/c'], 'topic': '/cloud', 'clock_id': 'different_clock'}
    result = card.dispatch('pointcloud_slam', {
        **arguments(), 'input_metadata': {'/pose': pose, '/cloud': cloud}})
    assert result['state'] == 'ready'
    assert call(card, 'start_mapping', map_name='ProducerAligned')['state'] == 'running'


def test_retired_robot_selector_does_not_affect_legacy_core(card, monkeypatch):
    # Old deployment robot names are ignored; explicit geometry stays valid.
    plugin = module.PointCloudSlamPlugin({**settings(), 'robot': 'Bumi'}, _FakeExecutor())
    try:
        monkeypatch.setattr(module, 'discover_input_bindings', lambda topics:
                            {'/cloud': 'sensor/pointcloud2', '/pose': 'state/odom'})
        assert call(plugin, 'start', input_topics=['/cloud', '/pose'])['state'] == 'ready'
        assert plugin._legacy_core is True
        assert call(plugin, 'info')['input_compatibility']['mode'] == 'legacy_core'
        assert call(plugin, 'start_mapping', map_name='Legacy')['state'] == 'running'
        session = plugin._session_settings
        assert session['cloud_frame'] == 'sensor'
        assert session['child_frame'] == '{segment_id}_base'
        assert session['clock_id'] == 'ros'
        assert session['child_from_cloud'] == settings()['child_from_cloud']
        assert 'robot' not in session
    finally:
        plugin.shutdown()

def test_retired_robot_selector_is_ignored_on_replay(card):
    plugin = module.PointCloudSlamPlugin({'robot': 'Bumi'}, _FakeExecutor())
    try:
        props = plugin.get_tools()[0]['configSchema']['properties']
        assert set(props) == {'child_from_cloud'}
        assert 'default' not in props['child_from_cloud']
        result = call(plugin, 'config', robot='G1', input_policy='experimental',
                      max_frames=5, clock_relation_verified=False)
        assert result['state'] == 'configured'
        assert 'robot' not in result['config']
        assert 'cloud_frame' not in result['config']
        assert 'input_policy' not in result['config']
    finally:
        plugin.shutdown()

def test_stable_mapping_output_for_live_and_saved_views(card):
    from plugins.pointcloud_slam_preview import PREVIEW_TOPIC
    expected = card.get_tools()[0]['topic_out']
    assert expected == [{'topic': PREVIEW_TOPIC, 'format': 'sensor/mapping',
                         'desc': 'Live or saved RTAB-Map preview'}]
    assert call(card, 'info')['topic_out'] == expected
    assert card.dispatch('pointcloud_slam', arguments())['topic_out'] == expected
    running = call(card, 'start_mapping', map_name='Hall')
    assert running['topic_out'] == expected
    assert call(card, 'select_map', map_id='0' * 32)['state'] == 'error'
    saved = call(card, 'stop_mapping')
    assert saved['state'] == 'saved'
    selected = call(card, 'select_map', map_id=saved['map_id'])
    assert selected['topic_out'] == expected
    assert selected['preview_segment_count'] == 1
    assert card._view.publishers[0].topic == PREVIEW_TOPIC
    assert call(card, 'info')['topic_out'] == expected
    previous_view = card._view
    # Starting another session switches back to the live stream.
    resumed = call(card, 'start_mapping', map_name='Next')
    assert resumed['state'] == 'running' and resumed['topic_out'] == expected
    assert previous_view.destroyed and card._view is None
    assert call(card, 'info')['topic_out'] == expected
    assert 'closed_topics' not in resumed


def test_reselect_saved_map_is_idempotent_and_renames_without_ports(card):
    capture(card)
    saved = call(card, 'stop_mapping')
    selected = call(card, 'select_map', map_id=saved['map_id'])
    view = card._view
    assert call(card, 'select_map', map_id=saved['map_id'])['topic_out'] == selected['topic_out']
    assert card._view is view
    renamed = call(card, 'rename_map', map_id=saved['map_id'], map_name='Lobby')
    assert renamed['topic_out'] == selected['topic_out']
    assert view.outputs()[0]['metadata']['map_view']['name'] == 'Lobby'
    assert 'closed_topics' not in renamed


def test_core_main_topics_can_resolve_generic_geometry_without_robot_name(card, monkeypatch):
    from copy import deepcopy
    inferred = {key: deepcopy(settings()[key]) for key in
                ('cloud_frame', 'child_frame', 'clock_id', 'child_from_cloud')}
    def discover(executor, odom_topic, cloud_topic, **kwargs):
        assert (odom_topic, cloud_topic) == ('/pose', '/cloud')
        assert kwargs['manual'] is None
        return inferred, {'resolution': 'observed_ros_messages',
                          'extrinsic_source': 'producer_calibrated'}
    monkeypatch.setattr(module, 'discover_runtime_geometry', discover)
    plugin = module.PointCloudSlamPlugin({'reserve_bytes': 0}, _FakeExecutor())
    try:
        assert plugin.dispatch('pointcloud_slam', arguments())['state'] == 'ready'
        result = call(plugin, 'start_mapping', map_name='Generic')
        assert result['state'] == 'running'
        assert plugin._session_settings['cloud_frame'] == 'sensor'
        assert plugin._session_settings['child_frame'] == '{segment_id}_base'
        assert call(plugin, 'info')['input_provenance']['resolution'] == 'observed_ros_messages'
        assert len(plugin.get_tools()[0]['topic_out']) == 1
    finally:
        plugin.shutdown()


def test_card_passes_only_connected_camera_evidence_to_runtime_discovery(card, monkeypatch):
    from test_pointcloud_slam_capture_mount import camera, pose
    from plugins.pointcloud_slam_capture_mount import resolve_capture_mount
    c = camera()
    cfg, evidence = resolve_capture_mount(pose(), c, 'camera_optical')
    def discover(executor, odom_topic, cloud_topic, **kwargs):
        assert kwargs['camera_info'] == c
        return cfg, evidence
    monkeypatch.setattr(module, 'discover_runtime_geometry', discover)
    plugin = module.PointCloudSlamPlugin({'reserve_bytes': 0}, _FakeExecutor())
    try:
        args = arguments()
        args['camera_infos'] = {'/cloud': c, '/unconnected': {'invalid': True}}
        assert plugin.dispatch('pointcloud_slam', args)['state'] == 'ready'
        assert call(plugin, 'start_mapping', map_name='Dynamic')['state'] == 'running'
        assert plugin._session_settings['child_from_cloud'] is None
        assert plugin._session_settings['capture_mount']['declaration'] == c['vendor']['body_mount']
        assert plugin._session_settings['input_provenance']['clock_accuracy_validated'] is False
    finally:
        plugin.shutdown()


def rgb_camera():
    return {'schema': 'motus.camera/1', 'topic': '/rgb', 'format': 'image/jpeg',
            'width': 640, 'height': 480, 'distortion_model': 'pinhole',
            'K': [500, 0, 320, 0, 500, 240, 0, 0, 1], 'D': [0, 0, 0, 0, 0],
            'vendor': {'frame_id': 'camera', 'clock_id': 'ros',
                       'registration': 'registered_rectified',
                       'timestamp_semantics': 'source_capture_mapped'}}


def rgb_arguments(camera=None):
    args = arguments()
    args['input_topics'].append('/rgb')
    args['input_bindings']['/rgb'] = 'image/jpeg'
    if camera is not None:
        args['camera_infos'] = {'/rgb': camera}
    return args


def assert_rgb_port(card, topic=None):
    ports = call(card, 'info')['topic_in']
    assert [port['format'] for port in ports] == [
        'state/odom', 'sensor/pointcloud2', 'image/jpeg']
    assert ports[2] == {'format': 'image/jpeg', 'optional': True,
                        **({'topic': topic} if topic else {})}


def test_optional_rgb_port_remains_visible_through_lifecycle_and_config_refresh(card, monkeypatch):
    assert card.get_tools()[0]['topic_in'][2] == {'format': 'image/jpeg', 'optional': True}
    assert_rgb_port(card)
    ready(card)
    assert_rgb_port(card)
    assert call(card, 'config')['state'] == 'configured'
    assert_rgb_port(card)
    assert call(card, 'start_mapping', map_name='Visible')['state'] == 'running'
    assert_rgb_port(card)
    node = card._node
    entered, release = threading.Event(), threading.Event()
    node.before_close = lambda: (entered.set(), release.wait(5))
    try:
        assert call(card, 'stop')['capture_stopped'] is True
        assert entered.wait(1)
        assert_rgb_port(card)
    finally:
        release.set()
    card.stop()
    assert_rgb_port(card)
    saved = call(card, 'list_maps')['maps'][0]
    assert call(card, 'select_map', map_id=saved['id'])['state'] == 'map_selected'
    assert_rgb_port(card)


def test_connected_rgb_reaches_session_settings_without_changing_spatial_contract(card):
    from test_pointcloud_slam_input import declarations
    spatial = declarations()
    metadata = {'/pose': dict(spatial['/o'], topic='/pose'),
                '/cloud': dict(spatial['/c'], topic='/cloud'),
                '/rgb': {'topic': '/rgb', 'format': 'image/jpeg'}}
    for topic in ('/pose', '/cloud'):
        metadata[topic]['clock'] = {'mapping': 'shared_camera_mapper', 'verified': True,
                                   'uncertainty_ns': 10}
    metadata['/cloud']['extrinsic']['calibrated'] = True
    camera = rgb_camera()
    args = {**rgb_arguments(camera), 'input_metadata': metadata}
    assert card.dispatch('pointcloud_slam', args)['state'] == 'ready'
    assert_rgb_port(card, '/rgb')
    assert call(card, 'start_mapping', map_name='RGB')['state'] == 'running'
    assert card._node.inputs == ('/pose', '/cloud')
    assert card._session_settings['rgb_input'] == {'topic': '/rgb', 'camera_info': camera}
    assert set(card._resolved_inputs['inputs']) == {'/pose', '/cloud'}
    camera['vendor']['clock_id'] = 'mutated'
    assert card._session_settings['rgb_input']['camera_info']['vendor']['clock_id'] == 'ros'
    assert call(card, 'config')['state'] == 'configured'
    assert_rgb_port(card, '/rgb')


@pytest.mark.parametrize('rgb_defines_matrix', [False, True])
def test_depth_cloud_numeric_registration_reaches_optional_rgb(card, rgb_defines_matrix):
    camera = rgb_camera()
    supplied = deepcopy(settings()['child_from_cloud'])
    supplied[0][3] = .2
    rgb_value = deepcopy(supplied)
    rgb_value[0][3] = .1
    if rgb_defines_matrix:
        camera['vendor']['color_from_depth'] = rgb_value
    args = rgb_arguments(camera)
    args['camera_infos']['/cloud'] = {'topic': '/cloud', 'format': 'sensor/pointcloud2',
                                     'vendor': {'color_from_depth': supplied}}
    assert card.dispatch('pointcloud_slam', args)['state'] == 'ready'
    assert call(card, 'start_mapping', map_name='RGB')['state'] == 'running'
    selected = card._session_settings['rgb_input']['camera_info']
    actual = selected.get('color_from_depth', selected['vendor'].get('color_from_depth'))
    assert actual == (rgb_value if rgb_defines_matrix else supplied)


@pytest.mark.parametrize('broken', ['missing', 'intrinsics'])
def test_invalid_connected_rgb_falls_back_without_blocking_mapping(card, broken):
    camera = rgb_camera()
    if broken == 'missing':
        camera = None
    elif broken == 'intrinsics':
        camera['K'] = None
    else:
        camera['vendor'].pop('clock_id')
    assert card.dispatch('pointcloud_slam', rgb_arguments(camera))['state'] == 'ready'
    info = call(card, 'info')
    assert any(item['field'] == 'rgb_input' for item in info['input_diagnostics'])
    assert info['rgb_input_status'] == 'fallback'
    assert_rgb_port(card, '/rgb')
    assert call(card, 'start_mapping', map_name='Fallback')['state'] == 'running'
    assert card._node.inputs == ('/pose', '/cloud')


def test_unconnected_rgb_declaration_does_not_enable_image_mode_and_restart_clears_rgb(card):
    camera = rgb_camera()
    assert card.dispatch('pointcloud_slam', rgb_arguments(camera))['state'] == 'ready'
    args = arguments()
    args['camera_infos'] = {'/rgb': camera}
    assert card.dispatch('pointcloud_slam', args)['state'] == 'ready'
    assert call(card, 'start_mapping', map_name='Cloud')['state'] == 'running'
    assert 'rgb_input' not in card._session_settings
    assert call(card, 'info')['rgb_input_status'] == 'disconnected'
    assert_rgb_port(card)


@pytest.mark.parametrize('declaration_key', ['camera_info', 'camera_infos'])
def test_legacy_discovery_can_prepare_the_connected_optional_rgb(card, monkeypatch, declaration_key):
    monkeypatch.setattr(module, 'discover_input_bindings', lambda topics: {
        '/pose': 'state/odom', '/cloud': 'sensor/pointcloud2', '/rgb': 'image/jpeg'})
    camera = rgb_camera()
    args = rgb_arguments()
    args[declaration_key] = {'/rgb': camera}
    del args['input_bindings']
    assert card.dispatch('pointcloud_slam', args)['state'] == 'ready'
    info = call(card, 'info')
    assert info['input_compatibility']['mode'] == 'legacy_core'
    assert info['rgb_input_status'] == 'ready'
    assert_rgb_port(card, '/rgb')
    assert call(card, 'start_mapping', map_name='Legacy RGB')['state'] == 'running'
    assert card._session_settings['rgb_input'] == {'topic': '/rgb', 'camera_info': camera}


@pytest.mark.parametrize('extra', [
    {'/other': 'image/jpeg'}, {'/other': 'sensor/pointcloud2'},
    {'/other': 'image/depth-zlib'}])
def test_optional_rgb_does_not_weaken_required_input_topology(card, extra):
    args = rgb_arguments(rgb_camera())
    args['input_bindings'].update(extra)
    args['input_topics'].extend(extra)
    assert card.dispatch('pointcloud_slam', args)['state'] == 'error'
    assert card._node is None


def test_unresolved_optional_rgb_diagnostic_remains_visible_through_scan_mapping_lifecycle(card):
    diagnostic = {'field': 'rgb_input', 'port_index': 2, 'source_card_id': 'camera',
                  'topic': '', 'reason': 'connected RGB source has no live topic'}
    args = {**arguments(), 'optional_input_diagnostics': [diagnostic]}
    assert card.dispatch('pointcloud_slam', args)['state'] == 'ready'
    info = call(card, 'info')
    assert info['rgb_input_status'] == 'fallback'
    assert diagnostic in info['input_diagnostics']
    assert_rgb_port(card)
    assert call(card, 'start_mapping', map_name='Missing RGB')['state'] == 'running'
    assert 'rgb_input' not in card._session_settings
    assert_rgb_port(card)
    entered, release = threading.Event(), threading.Event()
    card._node.before_close = lambda: (entered.set(), release.wait(5))
    try:
        assert call(card, 'stop')['capture_stopped'] is True
        assert entered.wait(1)
        stopping = call(card, 'info')
        assert stopping['rgb_input_status'] == 'fallback'
        assert diagnostic in stopping['input_diagnostics']
        assert_rgb_port(card)
    finally:
        release.set()
    card.stop()
    assert diagnostic in call(card, 'info')['input_diagnostics']


def test_canvas_declares_separate_create_and_resume_actions(card):
    schema = card.get_tools()[0]['inputSchema']
    for action, field in [('start_mapping', 'map_name'), ('resume_mapping', 'map_id')]:
        assert action in schema['properties']['action']['enum']
        assert schema['x-action-params'][action]['params'] == [field]
        assert schema['x-action-lifecycle']['actions'][action]['requires_project']
    assert schema['properties']['map_id']['x-resource-options']['action'] == 'list_maps'


@pytest.mark.parametrize('delete_old', [False, True])
def test_start_mapping_creates_a_new_map_with_a_stale_form_id(card, delete_old):
    capture(card)
    first = call(card, 'stop_mapping')
    if delete_old:
        assert call(card, 'delete_map', map_id=first['map_id'])['state'] == 'deleted'
    assert call(card, 'start_mapping', map_name='Other room', map_id=first['map_id'])['state'] == 'running'
    saved = call(card, 'stop_mapping')
    assert saved['map_id'] != first['map_id']
    assert (saved['map_name'], saved['revision']) == ('Other room', 1)


def test_resume_mapping_requires_an_existing_id_even_with_a_stale_map_name(card):
    capture(card)
    first = call(card, 'stop_mapping')
    assert call(card, 'delete_map', map_id=first['map_id'])['state'] == 'deleted'
    assert call(card, 'resume_mapping', map_name='Other room') == {
        'state': 'error', 'message': 'map_id is required for resume_mapping'}
    assert call(card, 'resume_mapping', map_id=first['map_id'], map_name='Other room') == {
        'state': 'error', 'message': 'unknown map'}
    assert card._node is None


def test_saved_native_map_can_resume_and_save_as_the_next_revision(card, monkeypatch):
    node = capture(card)
    first = call(card, 'stop_mapping')
    from test_pointcloud_slam_rgb_gateway import rgb_camera
    card._rgb_input = {'topic': '/rgb', 'camera_info': rgb_camera()}
    original = module.RtabmapNode
    restored = []
    def resume(*args, restored_map=None, **kwargs):
        restored.append(restored_map)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'RtabmapNode', resume)
    result = call(card, 'resume_mapping', map_name='stale form name', map_id=first['map_id'])
    assert result['state'] == 'running'
    assert result['map_name'] == first['map_name']
    assert restored[0]['segment_id'] == 'run'
    assert card._session_dir in Path(restored[0]['database']).parents
    assert call(card, 'rename_map', map_id=first['map_id'], map_name='Renamed Hall')['state']=='renamed'
    second = call(card, 'stop_mapping')
    assert second['map_id'] == first['map_id']
    assert second['revision'] == 2
    assert second['map_name'] == 'Renamed Hall'


def test_saved_fused_dynamic_mount_route_survives_resume(card, monkeypatch):
    from test_pointcloud_slam_rgb_gateway import rgb_camera

    ready(card)
    card._runtime_inputs = dict(extrinsic_mode='dynamic', child_from_cloud=None,
                                pose_capture_mount=True)
    assert call(card, 'start_mapping', map_name='Fused')['state'] == 'running'
    first = call(card, 'stop_mapping')
    card._config.pop('child_from_cloud')
    card._runtime_inputs = None
    card._rgb_input = {'topic': '/rgb', 'camera_info': rgb_camera()}
    original = module.RtabmapNode
    monkeypatch.setattr(module, 'RtabmapNode',
                        lambda *args, restored_map=None, **kwargs: original(*args, **kwargs))
    assert call(card, 'resume_mapping', map_id=first['map_id'])['state'] == 'running'
    assert card._session_settings['extrinsic_mode'] == 'dynamic'
    assert card._session_settings['pose_capture_mount'] is True
