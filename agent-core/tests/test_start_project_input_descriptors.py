"""Canvas start passes selected live descriptors only to opt-in consumers."""
import asyncio
import copy
import os
import pathlib
import sys
import tempfile
import threading
import types
from unittest import mock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'actucore'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))
from api import config as config_api  # noqa: E402
import config  # noqa: E402
from plugins.controlled_semantic_spatial.contract import controlled_semantic_spatial_tool_definition  # noqa: E402
from tests.test_controlled_semantic_spatial_plugin import FakeComponent, FakeRuntime  # noqa: E402
from plugins.controlled_semantic_spatial.plugin import ControlledSemanticSpatialPlugin  # noqa: E402


@pytest.fixture
def simulated_canvas(monkeypatch):
    tool = controlled_semantic_spatial_tool_definition('ubuntu')
    runtime = FakeRuntime()
    nav = ControlledSemanticSpatialPlugin({'namespace': 'ubuntu'}, 'ubuntu', executor=None,
        runtime=runtime, mapping_plugin=FakeComponent('mapping'),
        planning_plugin=FakeComponent('planning'), semantic_plugin=FakeComponent('semantic'))
    cloud = {'port': 'navigation_lidar', 'topic': '/custom/cloud', 'format': 'sensor/pointcloud',
             'ros_type': 'sensor_msgs/msg/PointCloud2',
             'associated_imu': {'topic': '/custom/imu', 'format': 'sensor/imu',
                                'ros_type': 'sensor_msgs/msg/Imu', 'qos': 'RELIABLE'}}
    live = {'lidar': [{'topic': '/legacy/cloud', 'format': 'sensor/legacy'}, cloud],
            'camera': [{'port': 'rgb', 'topic': '/custom/rgb', 'format': 'image/jpeg',
                        'ros_type': 'sensor_msgs/msg/CompressedImage'}], 'nav': []}
    cards = [{'id': name, 'mcpId': 'sim', 'toolName': name,
              'topicIn': copy.deepcopy(tool['topic_in']) if name == 'nav' else [],
              'topicOut': copy.deepcopy(live[name])} for name in ('lidar', 'camera', 'nav')]
    connections = [
        {'fromCardId': 'lidar', 'fromPortIdx': '1', 'fromPort': 'navigation_lidar',
         'toCardId': 'nav', 'toPortIdx': '0', 'toPort': 'lidar',
         'format': 'sensor/pointcloud', 'fromTopic': '/custom/cloud'},
        {'fromCardId': 'camera', 'fromPortIdx': '0', 'fromPort': 'rgb',
         'toCardId': 'nav', 'toPortIdx': '1', 'toPort': 'rgb',
         'format': 'image/jpeg', 'fromTopic': '/custom/rgb'},
    ]
    monkeypatch.setattr(config, 'main', {
        'canvas_layout': {'cards': cards, 'connections': connections},
        'services': {'mcp': [{'id': 'sim', 'tools': [
            {'name': 'nav', **{k: v for k, v in tool.items() if k != 'name'}},
            {'name': 'lidar'}, {'name': 'camera'}]}]},
    })
    starts, events = {}, []

    class Req:
        def __init__(self, tool, arguments):
            self.tool, self.arguments = tool, arguments

    async def call(mcp_id, req, timeout_s=None):
        args, name = req.arguments, req.tool
        if args.get('action') == 'start':
            starts[name] = copy.deepcopy(args)
            if name == 'nav':
                return {'code': 200, 'data': nav.dispatch(nav.PREFIX, args)}
            return {'code': 200, 'data': {'state': 'ready'}}
        if args.get('action') == 'info':
            assert timeout_s
            if name == 'nav':
                return {'code': 200, 'data': nav.dispatch(nav.PREFIX, args)}
            return {'code': 200, 'data': {'state': 'ready', 'topic_out': live[name]}}
        if name == 'nav':
            return {'code': 200, 'data': nav.dispatch(nav.PREFIX, args)}
        return {'code': 200, 'data': {'state': 'idle'}}

    async def push(event): events.append(event)
    async def register(*args): pass
    for name, module in (
        ('api.mcp_manage', types.SimpleNamespace(mcp_call_tool=call, MCPCallRequest=Req)),
        ('api.motus_stream', types.SimpleNamespace(push_event=push)),
        ('api.inspection', types.SimpleNamespace(register_topic_internal=register)),
        ('channel.manager', types.SimpleNamespace(manager=types.SimpleNamespace(
            sync_from_canvas=lambda: None, _adapters={}), _get_channel_configs=lambda: [])),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return types.SimpleNamespace(tool=tool, runtime=runtime, live=live, cards=cards,
                                 connections=connections, starts=starts, events=events)


def _start():
    return asyncio.run(config_api._do_start_project())


def test_driver_live_descriptors_through_core_to_real_navigation(simulated_canvas, monkeypatch):
    """Offline cross-worktree contract, actual producer methods; no ROS or device."""
    g1 = pathlib.Path(__file__).resolve().parents[3] / 'g1-pr140-pr141-combined' / 'unitree' / 'g1'
    if not (g1 / 'device.py').is_file():
        pytest.skip('adjacent G1 driver worktree unavailable')
    monkeypatch.syspath_prepend(str(g1))
    monkeypatch.syspath_prepend(str(g1 / 'tests'))
    from test_navigation_sensor_card import NavigationSensorCardContractTest
    from test_sensor_output_lifecycle import device_class, CameraLifecycleTests
    from sensor_output import OutputGate

    bridge = NavigationSensorCardContractTest.load_bridge_module()
    nav = bridge.NavigationSensorPlugin.__new__(bridge.NavigationSensorPlugin)
    nav._lifecycle_lock = threading.RLock()
    nav._requested_outputs = {'cloud': True, 'imu': True}
    nav._gates = {name: OutputGate() for name in nav._requested_outputs}
    nav._proc = mock.Mock(pid=123)
    nav._proc.is_alive.return_value = True
    nav._status_node = types.SimpleNamespace(
        cloud_topic='/custom/cloud', imu_topic='/custom/imu',
        lidar_frame='livox_frame', imu_frame='livox_frame',
        status=lambda running: {'ready': True, 'blockers': []})
    lidar_class = device_class('LidarPlugin')
    lidar = lidar_class.__new__(lidar_class)
    lidar._cloud_topic = '/ubuntu/lidar/cloud'
    lidar._node = types.SimpleNamespace(output_gate=OutputGate())
    lidar._navigation = nav
    lidar._lifecycle_lock = threading.RLock()
    camera = CameraLifecycleTests().plugin()

    lidar_tools = lidar.get_tools()[0]['topic_out']
    camera_tools = camera.get_tools()[0]['topic_out']
    lidar_info = lidar.dispatch('info', {})['topic_out']
    camera_info = camera.dispatch('info', {'_tool_name': 'camera_rgb'})['topic_out']
    assert lidar_info == lidar_tools
    assert camera_info == camera_tools
    assert camera_info[0]['port'] == 'rgb'
    s = simulated_canvas
    s.live['lidar'] = lidar_info
    s.live['camera'] = camera_info
    s.connections[1]['fromTopic'] = camera_info[0]['topic']
    assert _start() is True
    assert s.starts['nav']['input_descriptors'][0]['descriptor'] == lidar_info[1]
    assert s.starts['nav']['input_descriptors'][1]['descriptor'] == camera_info[0]
    assert s.runtime.start_kwargs['input_topics'] == {
        'lidar': '/custom/cloud', 'imu': '/custom/imu', 'rgb': camera_info[0]['topic']}


def test_simulated_canvas_live_descriptor_reaches_navigation(simulated_canvas):
    s = simulated_canvas
    assert _start() is True
    assert s.starts['nav']['input_topics'] == ['/custom/cloud', '/custom/rgb']
    assert s.starts['nav']['input_descriptors'][0]['descriptor'] == s.live['lidar'][1]
    assert s.runtime.start_kwargs['input_topics'] == {'lidar': '/custom/cloud',
                                                       'imu': '/custom/imu', 'rgb': '/custom/rgb'}
    assert 'input_descriptors' not in s.starts['lidar']
    assert 'input_descriptors' not in s.starts['camera']


@pytest.mark.parametrize('case', ['missing_identity', 'refreshed_old_imu', 'wrong_format',
    'missing_source', 'duplicate_source_topic', 'duplicate_target', 'bad_index',
    'fractional_index', 'boolean_index', 'wrong_identity', 'missing_from_port',
    'missing_live_port', 'camera_without_port', 'source_reordered', 'source_replaced',
    'duplicate_source_port'])
def test_ambiguous_old_or_invalid_connections_fail_closed(simulated_canvas, case):
    s = simulated_canvas
    c = s.connections[0]
    if case == 'missing_identity': c.pop('toPort')
    elif case == 'refreshed_old_imu':
        # card.topicIn is already current: only connection identity exposes stale wire.
        c.update(toPortIdx='1', toPort='imu', format='sensor/imu')
    elif case == 'wrong_format': c['format'] = 'sensor/imu'
    elif case == 'missing_source': s.live['lidar'] = []
    elif case == 'duplicate_source_topic': s.live['lidar'].append(copy.deepcopy(s.live['lidar'][1]))
    elif case == 'duplicate_target': s.connections.append(copy.deepcopy(c))
    elif case == 'bad_index': c['fromPortIdx'] = '-1'
    elif case == 'fractional_index': c['fromPortIdx'] = 1.1
    elif case == 'boolean_index': c['toPortIdx'] = False
    elif case == 'wrong_identity': c['toPort'] = 'rgb'
    elif case == 'missing_from_port': c.pop('fromPort')
    elif case == 'missing_live_port': s.live['lidar'][1].pop('port')
    elif case == 'camera_without_port': s.live['camera'][0].pop('port')
    elif case == 'source_reordered':
        other = copy.deepcopy(s.live['lidar'][1])
        other.update(port='other_lidar', topic='/custom/other')
        s.live['lidar'].insert(1, other)
    elif case == 'source_replaced':
        s.live['lidar'][1]['port'] = 'replacement_lidar'
    elif case == 'duplicate_source_port':
        other = copy.deepcopy(s.live['lidar'][1])
        other['topic'] = '/custom/other'
        s.live['lidar'].append(other)
    assert _start() is False
    assert 'nav' not in s.starts
    assert not s.runtime.started
