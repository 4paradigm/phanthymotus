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
