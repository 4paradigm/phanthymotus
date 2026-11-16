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
    assert {item['format'] for item in info['topic_in']} == {'sensor/pointcloud2', 'state/odom'}
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
    assert len(call(card, 'info')['topic_in']) == 2


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
    assert result['state'] == 'error'
    assert result['message']
    assert not card._executor.nodes


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


def test_contradictory_connected_clock_declarations_fail_project_start(card):
    from test_pointcloud_slam_input import declarations
    declared = declarations()
    pose = {**declared['/o'], 'topic': '/pose'}
    cloud = {**declared['/c'], 'topic': '/cloud', 'clock_id': 'different_clock'}
    result = card.dispatch('pointcloud_slam', {
        **arguments(), 'input_metadata': {'/pose': pose, '/cloud': cloud}})
    assert result['state'] == 'error'
    assert 'clock_id' in result['message']
    assert card._node is None


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
