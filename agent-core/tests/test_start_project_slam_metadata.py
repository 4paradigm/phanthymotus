"""Canvas SLAM metadata routing contract (ROS-free).

Run: cd agent-core && python3 -m pytest -q tests/test_start_project_slam_metadata.py
"""
import asyncio
import os
import pathlib
import sys
import tempfile
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from api import config as config_api  # noqa: E402
from api.slam_metadata import collect_slam_inputs, SlamRoutingError  # noqa: E402
import config  # noqa: E402


def _card(cid, name):
    return {'id': cid, 'toolName': name, 'mcpId': 'test-mcp'}


def _wire(src, port, dst, topic, fmt):
    return {'fromCardId': src, 'fromPortIdx': str(port), 'toCardId': dst,
            'toPortIdx': '0', 'fromTopic': topic, 'format': fmt}


@pytest.fixture
def rig(monkeypatch):
    starts, infos = {}, []
    ports = {
        'odom': [{'topic': '/loco/state', 'format': 'data/json'},
                 {'topic': '/slam/odom', 'format': 'state/odom'}],
        'cloud': [{'topic': '/lidar/display', 'format': 'sensor/pointcloud'},
                  {'topic': '/lidar/native', 'format': 'sensor/pointcloud2'}],
        'other': [{'topic': '/other/cloud', 'format': 'sensor/pointcloud2'}],
        'depth': [{'topic': '/bumi/depth', 'format': 'image/depth-zlib'}],
    }
    extra = {
        'odom': {'odom_interface': {'schema': 'motus.odom/1',
                    'slam_pose': {'available': False, 'profile': 'motus_slam/1'}},
                 'input_metadata': [{'schema': 'motus.slam_input/1', 'topic': '/slam/odom',
                                     'format': 'state/odom', 'source_id': 'g1/odom'}]},
        'cloud': {'input_metadata': [
            {'schema': 'motus.slam_input/1', 'topic': '/lidar/native',
             'format': 'sensor/pointcloud2', 'source_id': 'g1/lidar'},
            {'schema': 'motus.slam_input/1', 'topic': '/lidar/display',
             'format': 'sensor/pointcloud', 'source_id': 'display-only'}]},
        'other': {'input_metadata': [{'schema': 'motus.slam_input/1',
                   'topic': '/other/cloud', 'format': 'sensor/pointcloud2'}]},
        'depth': {'input_metadata': [{'topic': '/bumi/depth',
                                     'format': 'image/depth-zlib',
                                     'source_id': 'measured-depth',
                                     'clock': {'verified': False}}]},
    }

    async def call(mcp_id, req, timeout_s=None):
        action, cid = req.arguments.get('action'), req.arguments.get('instance_id')
        if action == 'start':
            starts[cid] = dict(req.arguments)
            return {'code': 200, 'data': {'state': 'running'}}
        if action == 'info':
            infos.append(cid)
            return {'code': 200, 'data': {'state': 'running',
                    'topic_out': ports.get(cid, []), **extra.get(cid, {})}}
        return {'code': 200, 'data': {'state': 'idle'}}

    class Req:
        def __init__(self, tool, arguments):
            self.tool, self.arguments = tool, arguments

    mcp = types.ModuleType('api.mcp_manage')
    mcp.mcp_call_tool, mcp.MCPCallRequest = call, Req
    stream = types.ModuleType('api.motus_stream')
    events = []
    async def push(event):
        events.append(event)
    stream.push_event = push
    inspection = types.ModuleType('api.inspection')
    async def register(*_args):
        return None
    inspection.register_topic_internal = register
    channel = types.ModuleType('channel.manager')
    channel.manager = types.SimpleNamespace(sync_from_canvas=lambda: None, _adapters={})
    channel._get_channel_configs = lambda: []
    for name, module in (('api.mcp_manage', mcp), ('api.motus_stream', stream),
                         ('api.inspection', inspection), ('channel.manager', channel)):
        monkeypatch.setitem(sys.modules, name, module)
    return types.SimpleNamespace(starts=starts, infos=infos, events=events,
                                 ports=ports, extra=extra)


def run(layout):
    config.main['canvas_layout'] = layout
    return asyncio.run(config_api._do_start_project_impl())


def test_g1_second_ports_flow_to_slam_without_other_ports_or_cards(rig):
    layout = {'cards': [_card('slam', 'pointcloud_slam'),
                        _card('other', 'other'), _card('cloud', 'lidar_cloud'),
                        _card('odom', 'loco_state')],
              'connections': [
                  _wire('odom', 1, 'slam', '/slam/odom', 'state/odom'),
                  _wire('cloud', 1, 'slam', '/lidar/native', 'sensor/pointcloud2')]}
    assert run(layout) is not False
    args = rig.starts['slam']
    assert args['input_topics'] == ['/slam/odom', '/lidar/native']
    assert args['input_bindings'] == {'/slam/odom': 'state/odom',
                                      '/lidar/native': 'sensor/pointcloud2'}
    assert set(args['odom_interfaces']) == {'/slam/odom'}
    assert args['odom_interfaces']['/slam/odom']['slam_pose']['available'] is False
    assert set(args['input_metadata']) == {'/slam/odom', '/lidar/native'}
    assert args['input_metadata']['/lidar/native']['source_id'] == 'g1/lidar'
    assert rig.infos.count('odom') == rig.infos.count('cloud') == 1


def test_unverified_declarations_are_passed_without_upgrading_trust(rig):
    rig.extra['cloud']['input_metadata'][0]['clock'] = {
        'mapping': 'unknown', 'verified': False}
    layout = {'cards': [_card('odom', 'loco_state'), _card('cloud', 'lidar_cloud'),
                        _card('slam', 'pointcloud_slam')],
              'connections': [_wire('odom', 1, 'slam', '/slam/odom', 'state/odom'),
                              _wire('cloud', 1, 'slam', '/lidar/native', 'sensor/pointcloud2')]}
    run(layout)
    assert rig.starts['slam']['input_metadata']['/lidar/native']['clock']['verified'] is False
    assert rig.starts['slam']['odom_interfaces']['/slam/odom']['slam_pose']['available'] is False


def test_unconnected_legacy_card_receives_no_new_slam_arguments(rig):
    layout = {'cards': [_card('odom', 'loco_state'), _card('other', 'other')],
              'connections': [_wire('odom', 0, 'other', '/loco/state', 'data/json')]}
    run(layout)
    assert 'input_bindings' not in rig.starts['other']
    assert 'input_metadata' not in rig.starts['other']


def test_missing_live_source_info_never_creates_calibration_or_clock_proofs():
    conns = [_wire('odom', 1, 'slam', '/slam/odom', 'state/odom')]
    out = collect_slam_inputs(conns, {}, lambda c: c['fromTopic'])
    assert out == {'input_bindings': {'/slam/odom': 'state/odom'},
                   'input_metadata': {}, 'odom_interfaces': {}}


def test_duplicate_topic_and_wrong_port_format_fail_explicitly():
    c = _wire('src', 0, 'slam', '/same', 'state/odom')
    info = {'src': {'topic_out': [{'topic': '/same', 'format': 'state/odom'}]}}
    with pytest.raises(SlamRoutingError, match='duplicate inbound'):
        collect_slam_inputs([c, c], info, lambda x: x['fromTopic'])
    with pytest.raises(SlamRoutingError, match='saved wire format conflicts'):
        collect_slam_inputs([dict(c, format='sensor/pointcloud2')],
                            info, lambda x: x['fromTopic'])


def test_multiple_odometry_outputs_cannot_reuse_one_unkeyed_interface():
    conns = [_wire('src', 1, 'slam', '/b', 'state/odom')]
    info = {'src': {'topic_out': [{'topic': '/a', 'format': 'state/odom'},
                                  {'topic': '/b', 'format': 'state/odom'}],
                    'odom_interface': {'schema': 'motus.odom/1'}}}
    out = collect_slam_inputs(conns, info, lambda c: c['fromTopic'])
    assert out['odom_interfaces'] == {}


def test_malformed_explicit_odom_interface_values_are_not_dropped():
    conns = [_wire('src', 0, 'slam', '/odom', 'state/odom')]
    ports = [{'topic': '/odom', 'format': 'state/odom'}]

    keyed = collect_slam_inputs(conns, {
        'src': {'topic_out': ports, 'odom_interfaces': {'/odom': 'invalid'}}
    }, lambda c: c['fromTopic'])
    assert keyed['odom_interfaces']['/odom'] == 'invalid'

    legacy = collect_slam_inputs(conns, {
        'src': {'topic_out': ports, 'odom_interface': 'invalid'}
    }, lambda c: c['fromTopic'])
    assert legacy['odom_interfaces']['/odom'] == 'invalid'


def test_malformed_odom_interfaces_container_is_rejected():
    conns = [_wire('src', 0, 'slam', '/odom', 'state/odom')]
    info = {'src': {'topic_out': [{'topic': '/odom', 'format': 'state/odom'}],
                    'odom_interfaces': ['invalid']}}
    with pytest.raises(SlamRoutingError, match='odom_interfaces'):
        collect_slam_inputs(conns, info, lambda c: c['fromTopic'])


def test_missing_topic_keyed_odom_interface_remains_pending():
    conns = [_wire('src', 0, 'slam', '/odom', 'state/odom')]
    info = {'src': {'topic_out': [{'topic': '/odom', 'format': 'state/odom'}],
                    'odom_interfaces': {}}}
    result = collect_slam_inputs(conns, info, lambda c: c['fromTopic'])
    assert result['odom_interfaces'] == {}


def test_pointcloud_receives_only_connected_depth_provenance(rig):
    layout = {'cards': [_card('other', 'other'), _card('depth', 'depth'),
                        _card('cloud', 'pointcloud')],
              'connections': [_wire('depth', 0, 'cloud', '/bumi/depth',
                                    'image/depth-zlib')]}
    assert run(layout) is not False
    assert rig.starts['cloud']['input_metadata'] == {
        '/bumi/depth': {'topic': '/bumi/depth', 'format': 'image/depth-zlib',
                        'source_id': 'measured-depth',
                        'clock': {'verified': False}}}
    assert 'input_bindings' not in rig.starts['cloud']
    assert rig.infos.count('depth') == 1


def test_unconnected_or_stale_source_declarations_cannot_be_forwarded(rig):
    from api.slam_metadata import collect_connected_metadata
    conns = [_wire('depth', 0, 'cloud', '/bumi/depth', 'image/depth-zlib')]
    # A second declaration for another robot lives in the same info() but was
    # never connected to this consumer.
    info = {'depth': {'topic_out': rig.ports['depth'],
                      'input_metadata': [
                          rig.extra['depth']['input_metadata'][0],
                          {'topic': '/elsewhere/depth', 'format': 'image/depth-zlib'}]}}
    metadata = collect_connected_metadata(conns, info, lambda c: c['fromTopic'])
    assert list(metadata) == ['/bumi/depth']
    # A saved canvas topic is not evidence that the live producer still emits
    # it after a restart.
    info['depth']['topic_out'][0] = {'topic': '/new/topic', 'format': 'image/depth-zlib'}
    assert collect_connected_metadata(conns, info, lambda c: c['fromTopic']) == {}


@pytest.mark.parametrize('source,tool,port,topic,fmt', [
    ('cloud', 'pointcloud_slam', 1, '/lidar/native', 'sensor/pointcloud2'),
    ('depth', 'pointcloud', 0, '/bumi/depth', 'image/depth-zlib'),
])
def test_metadata_start_failure_identifies_the_card(rig, monkeypatch, source, tool, port, topic, fmt):
    rig.extra[source]['input_metadata'].append(dict(rig.extra[source]['input_metadata'][0]))
    rolled_back = []
    async def stop():
        rolled_back.append(True)
    monkeypatch.setattr(config_api, '_do_stop_project', stop)
    layout = {'cards': [_card(source, source), _card('failed-instance', tool)],
              'connections': [_wire(source, port, 'failed-instance', topic, fmt)]}
    assert run(layout) is False
    errors = [e['payload'] for e in rig.events
              if e['type'] == 'project_start_item' and e['payload']['status'] == 'error']
    assert len(errors) == 1
    assert errors[0]['card_id'] == 'failed-instance'
    done = next(e['payload'] for e in rig.events if e['type'] == 'project_start_done')
    assert done['errors'] == [f'{tool} (failed-instance)']
    assert 'failed-instance' not in rig.starts and rolled_back == [True]


def test_connected_cloud_capture_mount_is_forwarded_without_other_ports():
    cloud = {'schema': 'motus.camera/1', 'topic': '/metric', 'format': 'sensor/pointcloud2',
             'vendor': {'body_mount': {'topic': '/actual/mount'}}}
    other = {'schema': 'motus.camera/1', 'topic': '/unconnected', 'format': 'sensor/pointcloud2'}
    infos = {'c': {'topic_out': [{'topic': '/metric', 'format': 'sensor/pointcloud2'}],
                   'camera_info': [other, cloud]}}
    wire = _wire('c', 0, 'slam', '/metric', 'sensor/pointcloud2')
    got = collect_slam_inputs([wire], infos, lambda c: c['fromTopic'])
    assert got['camera_infos'] == {'/metric': cloud}
    got['camera_infos']['/metric']['vendor']['body_mount']['topic'] = '/changed'
    assert cloud['vendor']['body_mount']['topic'] == '/actual/mount'
    infos['c']['camera_info'].append(cloud)
    with pytest.raises(SlamRoutingError, match='duplicate'):
        collect_slam_inputs([wire], infos, lambda c: c['fromTopic'])


def test_slam_routing_error_log_escapes_and_caps_untrusted_topic(rig, monkeypatch, capsys):
    topic = '/lidar/\nFORGED\r\x1b[31m\x00\u2028' + 'x' * 5000
    rig.ports['cloud'][1]['topic'] = topic
    rig.extra['cloud']['input_metadata'] = [
        {'topic': topic, 'format': 'sensor/pointcloud2'},
        {'topic': topic, 'format': 'sensor/pointcloud2'},
    ]
    async def stop():
        pass
    monkeypatch.setattr(config_api, '_do_stop_project', stop)
    layout = {'cards': [_card('cloud', 'lidar_cloud'), _card('slam', 'pointcloud_slam')],
              'connections': [_wire('cloud', 1, 'slam', topic, 'sensor/pointcloud2')]}
    assert run(layout) is False
    output = capsys.readouterr().out
    line = next(line for line in output.split('\n') if 'SLAM' in line)
    assert '\\nFORGED\\r\\x1b' in line
    assert not any(c in output for c in ('\r', '\x1b', '\x00', '\u2028'))
    assert len(line) <= 1100
    event = next(e['payload'] for e in rig.events if e['type'] == 'project_start_item'
                 and e['payload'].get('card_id') == 'slam')
    assert topic in event['message']  # UI diagnostics remain complete.


def test_connected_rgb_routes_only_its_live_camera_without_spatial_metadata(rig):
    camera = {'schema': 'motus.camera/1', 'topic': '/rgb/live', 'format': 'image/jpeg',
              'K': [500, 0, 320, 0, 500, 240, 0, 0, 1],
              'vendor': {'clock_id': 'capture-clock'}}
    rig.ports['rgb'] = [{'topic': '/rgb/other', 'format': 'image/jpeg'},
                        {'topic': '/rgb/live', 'format': 'image/jpeg'}]
    rig.extra['rgb'] = {'camera_info': [dict(camera, topic='/rgb/other'), camera],
                        'input_metadata': [{'topic': '/rgb/live', 'format': 'image/jpeg'}]}
    layout = {'cards': [_card('slam', 'pointcloud_slam'), _card('rgb', 'camera'),
                        _card('odom', 'loco_state'), _card('cloud', 'lidar_cloud')],
              'connections': [_wire('odom', 1, 'slam', '/slam/odom', 'state/odom'),
                              _wire('cloud', 1, 'slam', '/lidar/native', 'sensor/pointcloud2'),
                              _wire('rgb', 1, 'slam', '/rgb/live', 'image/jpeg')]}
    assert run(layout) is not False
    args = rig.starts['slam']
    assert args['input_bindings']['/rgb/live'] == 'image/jpeg'
    assert args['camera_infos'] == {'/rgb/live': camera}
    assert set(args['input_metadata']) == {'/slam/odom', '/lidar/native'}
    args['camera_infos']['/rgb/live']['vendor']['clock_id'] = 'changed'
    assert camera['vendor']['clock_id'] == 'capture-clock'


def test_rgb_camera_claim_without_live_output_is_not_forwarded():
    wire = _wire('rgb', 0, 'slam', '/rgb', 'image/jpeg')
    info = {'rgb': {'camera_info': [{'schema': 'motus.camera/1',
                                     'topic': '/rgb', 'format': 'image/jpeg'}]}}
    got = collect_slam_inputs([wire], info, lambda c: c['fromTopic'])
    assert got['input_bindings'] == {'/rgb': 'image/jpeg'}
    assert 'camera_infos' not in got


def test_invalid_rgb_calibration_is_forwarded_for_optional_fallback():
    wire = _wire('rgb', 0, 'slam', '/rgb', 'image/jpeg')
    malformed = {'schema': 'motus.camera/1', 'topic': '/rgb', 'format': 'wrong'}
    info = {'rgb': {'topic_out': [{'topic': '/rgb', 'format': 'image/jpeg'}],
                    'camera_info': [malformed]}}
    got = collect_slam_inputs([wire], info, lambda c: c['fromTopic'])
    assert got['camera_infos']['/rgb'] == malformed
    info['rgb']['camera_info'].append(dict(malformed))
    got = collect_slam_inputs([wire], info, lambda c: c['fromTopic'])
    assert got['camera_infos']['/rgb'] == [malformed, malformed]


def optional_rgb_layout(rig, monkeypatch, *, live_format='image/jpeg', live_topic=''):
    monkeypatch.setitem(config.main, 'services', {'mcp': [{
        'id': 'test-mcp', 'tools': [{'name': 'pointcloud_slam',
        'inputSchema': {'x-slam-input-contract': 'motus.slam_input/1'},
        'topic_in': [{'format': 'state/odom'}, {'format': 'sensor/pointcloud2'},
                     {'format': 'image/jpeg', 'optional': True}]}]}]})
    rig.ports['rgb'] = [{'topic': live_topic, 'format': live_format}]
    return {'cards': [_card('odom', 'loco_state'), _card('cloud', 'lidar_cloud'),
                      _card('rgb', 'camera'), _card('slam', 'pointcloud_slam')],
            'connections': [
                _wire('odom', 1, 'slam', '/slam/odom', 'state/odom'),
                dict(_wire('cloud', 1, 'slam', '/lidar/native', 'sensor/pointcloud2'), toPortIdx='1'),
                dict(_wire('rgb', 0, 'slam', '', 'image/jpeg'), toPortIdx='2')]}


@pytest.mark.parametrize('topic,fmt,saved_topic', [
    ('', 'image/jpeg', ''), ('', 'image/jpeg', '/cached/rgb'),
    ('/live/rgb', 'image/png', ''), ('/live/rgb', 'state/odom', '')])
def test_unavailable_declared_optional_rgb_starts_with_required_pair_and_diagnostic(
        rig, monkeypatch, topic, fmt, saved_topic):
    layout = optional_rgb_layout(rig, monkeypatch, live_topic=topic, live_format=fmt)
    layout['connections'][2]['fromTopic'] = saved_topic
    assert run(layout) is not False
    args = rig.starts['slam']
    assert args['input_topics'] == ['/slam/odom', '/lidar/native']
    assert args['input_bindings'] == {'/slam/odom': 'state/odom',
                                      '/lidar/native': 'sensor/pointcloud2'}
    diagnostic, = args['optional_input_diagnostics']
    assert diagnostic['field'] == 'rgb_input'
    assert diagnostic['port_index'] == 2
    assert diagnostic['source_card_id'] == 'rgb'
    assert diagnostic['reason']
    assert layout['connections'][2]['toPortIdx'] == '2'


@pytest.mark.parametrize('slot', ['0', '1', 'bad', '-1'])
def test_unresolved_required_or_invalid_slot_is_never_optional_rgb(rig, monkeypatch, slot):
    layout = optional_rgb_layout(rig, monkeypatch)
    layout['connections'][2]['toPortIdx'] = slot
    async def stop():
        pass
    monkeypatch.setattr(config_api, '_do_stop_project', stop)
    assert run(layout) is False
    assert 'slam' not in rig.starts


def test_saved_optional_claim_cannot_exempt_a_required_consumer_port(rig, monkeypatch):
    layout = optional_rgb_layout(rig, monkeypatch)
    services = config.main['services']
    services['mcp'][0]['tools'][0]['topic_in'][2]['optional'] = False
    monkeypatch.setitem(config.main, 'services', services)
    layout['cards'][-1]['topicIn'] = [
        {'format': 'state/odom'}, {'format': 'sensor/pointcloud2'},
        {'format': 'image/jpeg', 'optional': True}]
    async def stop():
        pass
    monkeypatch.setattr(config_api, '_do_stop_project', stop)
    assert run(layout) is False
    assert 'slam' not in rig.starts


def test_reconnected_optional_rgb_uses_the_live_single_output(rig, monkeypatch):
    layout = optional_rgb_layout(rig, monkeypatch, live_topic='/live/rgb')
    layout['connections'][2].update(fromPortIdx='1', fromTopic='/old/rgb')
    camera = {'schema': 'motus.camera/1', 'topic': '/live/rgb', 'format': 'image/jpeg'}
    rig.extra['rgb'] = {'camera_info': [camera]}
    assert run(layout) is not False
    args = rig.starts['slam']
    assert args['input_bindings'].get('/live/rgb') == 'image/jpeg'
    assert args['camera_infos'] == {'/live/rgb': camera}
    assert 'optional_input_diagnostics' not in args


def test_reconnected_source_cannot_reuse_a_saved_format_for_a_different_live_type(rig, monkeypatch):
    rig.ports['cloud'] = [{'topic': '/live/state', 'format': 'state/odom'}]
    layout = {'cards': [_card('odom', 'loco_state'), _card('cloud', 'lidar_cloud'),
                        _card('slam', 'pointcloud_slam')],
              'connections': [_wire('odom', 1, 'slam', '/slam/odom', 'state/odom'),
                              _wire('cloud', 1, 'slam', '/old/cloud', 'sensor/pointcloud2')]}
    async def stop():
        pass
    monkeypatch.setattr(config_api, '_do_stop_project', stop)
    assert run(layout) is False
    assert 'slam' not in rig.starts
    assert any('saved wire format conflicts' in e['payload'].get('message', '')
               for e in rig.events if e['type'] == 'project_start_item')


def test_out_of_range_consumer_slot_does_not_inherit_a_sole_optional_rgb_port():
    wire = dict(_wire('rgb', 0, 'slam', '', 'image/jpeg'), toPortIdx='1')
    with pytest.raises(SlamRoutingError, match='unresolved source topic'):
        collect_slam_inputs([wire], {}, lambda c: '',
                            input_ports=[{'format': 'image/jpeg', 'optional': True}])


@pytest.mark.parametrize('declarations', [None, 1, True, 'not-a-list',
    {'topic': '/lidar/native', 'format': 'sensor/pointcloud2', 'K': [1]}])
def test_nonlist_cloud_camera_info_remains_optional_during_start(rig, declarations):
    rig.extra['cloud']['camera_info'] = declarations
    layout = {'cards': [_card('odom', 'loco_state'), _card('cloud', 'lidar_cloud'),
                        _card('slam', 'pointcloud_slam')],
              'connections': [_wire('odom', 1, 'slam', '/slam/odom', 'state/odom'),
                              _wire('cloud', 1, 'slam', '/lidar/native', 'sensor/pointcloud2')]}
    assert run(layout) is not False
    args = rig.starts['slam']
    assert args['input_bindings']['/lidar/native'] == 'sensor/pointcloud2'
    assert 'camera_infos' not in args
    assert args['input_metadata']['/lidar/native'] == rig.extra['cloud']['input_metadata'][0]


@pytest.mark.parametrize('declarations,error', [
    ([{'topic': '/cloud', 'format': 'sensor/pointcloud2'}] * 2, 'duplicate camera_info'),
    ([{'topic': '/cloud', 'format': 'image/jpeg'}], 'camera_info format mismatch'),
])
def test_cloud_camera_info_lists_still_reject_duplicate_or_conflicting_formats(declarations, error):
    wire = _wire('cloud', 0, 'slam', '/cloud', 'sensor/pointcloud2')
    info = {'cloud': {'topic_out': [{'topic': '/cloud', 'format': 'sensor/pointcloud2'}],
                      'camera_info': declarations}}
    with pytest.raises(SlamRoutingError, match=error):
        collect_slam_inputs([wire], info, lambda c: c['fromTopic'])
