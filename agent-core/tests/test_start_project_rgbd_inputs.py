"""Canvas → start-project → generic RGB-D start validation, end to end offline.

The field symptom was the RGB-D card failing with "connect one calibrated color
and one measured depth input" while camera_head, camera_depth and imu all showed
ready. That message only says the RGB-D card did not receive one image/jpeg and
one image/depth-zlib declaration keyed by its input topics; it does not say which
link, topic, declaration or format was missing.

These tests drive the real `_do_start_project_impl` with Tianyi-shaped source
cards and hand the resulting `start` arguments to the real Perception contract
(`perception/plugins/rgbd_odometry_contract.py`), so the whole chain — canvas
connection, port/topic resolution, `camera_info` collection, start arguments,
validation — runs as in production with only MCP transport faked.

Run: cd agent-core && python3 -m pytest tests/test_start_project_rgbd_inputs.py
"""
import asyncio
from copy import deepcopy
import importlib.util
import os
import pathlib
import sys
import tempfile
import types

import pytest

pytest.importorskip('numpy')

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from api import config as config_api  # noqa: E402
import config  # noqa: E402


def _load_contract():
    """Import Perception's contract under a private package name.

    Perception's own `plugins`/`utils` packages must not shadow anything in the
    agent-core test process, so the directory is mounted as a unique package.
    """
    root = pathlib.Path(__file__).resolve().parents[2] / 'perception' / 'plugins'
    name = '_motus_perception_plugins'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, root / '__init__.py', submodule_search_locations=[str(root)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
    return importlib.import_module(name + '.rgbd_odometry_contract')


contract = _load_contract()

NS = '/tianyi'
COLOR, DEPTH = NS + '/camera/head', NS + '/camera/rgbd/depth'
IMU, OLD_COLOR = NS + '/camera/rgbd/imu', NS + '/camera/rgbd/color'
HEAD, DEPTH_CARD, IMU_CARD, VO = 'card-head', 'card-depth', 'card-imu', 'card-vo'


def _declaration(topic, fmt):
    entry = dict(schema='motus.camera/1', id=NS + '/camera/rgbd/rectified',
                 width=640, height=480, K=[524.59, 0, 319.5, 0, 524.59, 239.5, 0, 0, 1],
                 D=[0.] * 5, distortion_model='pinhole', source='derived-from-K',
                 topic=topic, format=fmt,
                 vendor=dict(frame_id='ob_camera_head_color_optical_frame',
                             clock_id='orbbec-global', registration='registered_rectified',
                             timestamp_semantics='source_capture_mapped'))
    if fmt == 'image/depth-zlib':
        entry['vendor'].update(depth_unit='mm', depth_source='measured')
    return entry


def _card(cid, tool):
    return {'id': cid, 'toolName': tool, 'mcpId': f'mcp-{tool}'}


def _conn(src, dst, topic, port=0, fmt='image/jpeg'):
    return {'fromCardId': src, 'toCardId': dst, 'fromPortIdx': str(port),
            'toPortIdx': '0', 'format': fmt, 'fromTopic': topic}


def _rgbd_start(args, imu_assisted):
    """What RgbdOdometryPlugin.dispatch('start') validates before any runtime."""
    try:
        topics = contract.start_topics(args)
        calibration = contract.validate_start_inputs(args.get('camera_info'), topics, imu_assisted)
    except ValueError as error:
        return {'state': 'error', 'message': str(error)}
    return {'state': 'running', 'calibration': calibration}


@pytest.fixture
def rig(monkeypatch):
    """Tianyi-shaped sources: each card declares only its own port's optics."""
    state = types.SimpleNamespace(
        starts={}, events=[], results={}, imu_assisted=False,
        source_state={}, info_failures=set(),
        topic_out={HEAD: [{'topic': COLOR, 'format': 'image/jpeg'}],
                   DEPTH_CARD: [{'topic': DEPTH, 'format': 'image/depth-zlib'}],
                   IMU_CARD: [{'topic': IMU, 'format': 'sensor/imu'}]},
        camera_info={HEAD: [_declaration(COLOR, 'image/jpeg')],
                     DEPTH_CARD: [_declaration(DEPTH, 'image/depth-zlib')],
                     IMU_CARD: []})

    async def _call(mcp_id, req, timeout_s=None):
        args = dict(req.arguments)
        action, card_id = args.get('action'), args.get('instance_id')
        if card_id == VO:
            if action == 'start':
                state.starts[VO] = deepcopy(args)
                state.results[VO] = _rgbd_start(args, state.imu_assisted)
                return {'code': 200, 'data': {k: v for k, v in state.results[VO].items()
                                              if k != 'calibration'}}
            return {'code': 200, 'data': {'state': 'idle', 'topic_out': []}}
        if action == 'start':
            state.starts[card_id] = args
            return {'code': 200, 'data': {'state': state.source_state.get(card_id, 'running')}}
        if action == 'info':
            if card_id in state.info_failures:
                raise TimeoutError('source info() timed out')
            return {'code': 200, 'data': {
                'state': state.source_state.get(card_id, 'running'),
                'topic_out': state.topic_out.get(card_id, []),
                'camera_info': state.camera_info.get(card_id, [])}}
        return {'code': 200, 'data': {'state': 'idle'}}

    class _Req:
        def __init__(self, tool, arguments):
            self.tool, self.arguments = tool, arguments

    mcp = types.ModuleType('api.mcp_manage')
    mcp.mcp_call_tool, mcp.MCPCallRequest = _call, _Req

    async def _push(event):
        state.events.append(event)

    stream = types.ModuleType('api.motus_stream')
    stream.push_event = _push
    inspection = types.ModuleType('api.inspection')

    async def _register(topic, fmt, mcp_id):
        return None

    inspection.register_topic_internal = _register
    chan = types.ModuleType('channel.manager')
    chan.manager = types.SimpleNamespace(sync_from_canvas=lambda: None, _adapters={})
    chan._get_channel_configs = lambda: []
    for name, module in (('api.mcp_manage', mcp), ('api.motus_stream', stream),
                         ('api.inspection', inspection), ('channel.manager', chan)):
        monkeypatch.setitem(sys.modules, name, module)

    async def _stop():
        return None

    monkeypatch.setattr(config_api, '_do_stop_project', _stop)
    return state


def _layout(connections, imu=False):
    cards = [_card(HEAD, 'camera_head'), _card(DEPTH_CARD, 'camera_depth'),
             _card(VO, 'rgbd_odometry')]
    if imu:
        cards.insert(2, _card(IMU_CARD, 'imu'))
    return {'cards': cards, 'connections': connections}


def _run(rig, layout, imu_assisted=False):
    rig.imu_assisted = imu_assisted
    config.main['canvas_layout'] = layout
    return asyncio.run(config_api._do_start_project_impl())


def _vo_error(rig):
    errors = [e['payload'] for e in rig.events if e.get('type') == 'project_start_item'
              and e['payload'].get('status') == 'error' and e['payload'].get('tool') == 'rgbd_odometry']
    return errors[-1]['message'] if errors else None


RGBD = [_conn(HEAD, VO, COLOR), _conn(DEPTH_CARD, VO, DEPTH, fmt='image/depth-zlib')]
IMU_LINK = _conn(IMU_CARD, VO, IMU, fmt='sensor/imu')


@pytest.mark.parametrize('order', [(0, 1), (1, 0)])
def test_pure_rgbd_declarations_reach_the_card_in_any_connection_order(rig, order):
    assert _run(rig, _layout([RGBD[i] for i in order])) is True
    result = rig.results[VO]
    assert result['state'] == 'running', result
    assert (result['calibration']['color_topic'], result['calibration']['depth_topic']) == (COLOR, DEPTH)
    assert set(rig.starts[VO]['camera_info']) == {COLOR, DEPTH}


@pytest.mark.parametrize('first', ['imu', 'depth', 'color'])
def test_rgbd_with_imu_selects_the_undeclared_third_topic_as_imu(rig, first):
    links = {'imu': IMU_LINK, 'color': RGBD[0], 'depth': RGBD[1]}
    order = [first] + [k for k in ('color', 'depth', 'imu') if k != first]
    assert _run(rig, _layout([links[k] for k in order], imu=True), imu_assisted=True) is True
    calibration = rig.results[VO]['calibration']
    assert calibration['imu_topic'] == IMU
    assert (calibration['color_topic'], calibration['depth_topic']) == (COLOR, DEPTH)


def test_source_without_calibration_yet_is_named_in_the_rgbd_error(rig):
    """Reproduces the field message by one verified offline path.

    The Tianyi camera card answers `unavailable` with no `camera_info` until its
    first calibrated capture. Core has no special case for that state, so the
    source is reported started and the RGB-D card receives no declaration for
    the color topic. This does not prove it was the field cause.
    """
    rig.source_state[HEAD] = 'unavailable'
    rig.camera_info[HEAD] = []
    assert _run(rig, _layout(RGBD)) is False
    message = _vo_error(rig)
    assert message.startswith('connect one calibrated color and one measured depth input')
    assert f'{COLOR}: no camera_info declaration' in message
    assert 'missing image/jpeg' in message
    assert f'{DEPTH}: declared image/depth-zlib' in message
    head = [e['payload'] for e in rig.events if e['payload'].get('tool') == 'camera_head']
    assert head[-1]['status'] == 'ready'  # current Core semantics, recorded as a field check item


def test_source_info_failure_falls_back_to_snapshot_topic_without_declaration(rig):
    """A source whose info() fails still starts; its link resolves from the
    canvas snapshot (here a removed pre-83dbaea0 port), with no optics."""
    rig.info_failures.add(HEAD)
    stale = [_conn(HEAD, VO, OLD_COLOR, port=1), RGBD[1]]
    assert _run(rig, _layout(stale)) is False
    message = _vo_error(rig)
    assert f'{OLD_COLOR}: no camera_info declaration' in message


def test_reconnected_old_port_index_uses_live_single_output(rig):
    """The removed second RGB port: a saved link on port 1 resolves to the live
    sole output and the declaration that names it, not to the stale snapshot."""
    stale = [_conn(HEAD, VO, OLD_COLOR, port=1), RGBD[1]]
    assert _run(rig, _layout(stale)) is True
    assert rig.results[VO]['calibration']['color_topic'] == COLOR


def test_duplicate_links_to_one_topic_report_the_missing_role(rig):
    """Two links carrying one topic collapse to one input. Core then sends the
    singular `input_topic`; the card must still say what is missing."""
    duplicate = [RGBD[1], _conn(DEPTH_CARD, VO, DEPTH, fmt='image/depth-zlib')]
    assert _run(rig, _layout(duplicate)) is False
    assert 'input_topics' not in rig.starts[VO] and rig.starts[VO]['input_topic'] == DEPTH
    message = _vo_error(rig)
    assert '1 input(s) connected' in message and DEPTH in message


def test_wrong_declared_format_is_named(rig):
    rig.camera_info[DEPTH_CARD] = [_declaration(DEPTH, 'image/jpeg')]
    assert _run(rig, _layout(RGBD)) is False
    message = _vo_error(rig)
    assert 'missing image/depth-zlib' in message
    assert f'{DEPTH}: declared image/jpeg' in message


def test_imu_link_without_imu_assistance_explains_the_count(rig):
    assert _run(rig, _layout(RGBD + [IMU_LINK], imu=True), imu_assisted=False) is False
    message = _vo_error(rig)
    assert 'imu_assisted=false' in message and '3 input(s) connected' in message
    assert f'{IMU}: no camera_info declaration' in message


def test_imu_assistance_without_imu_link_explains_the_count(rig):
    assert _run(rig, _layout(RGBD, imu=True), imu_assisted=True) is False
    message = _vo_error(rig)
    assert 'imu_assisted=true' in message and '2 input(s) connected' in message


def test_missing_role_survives_core_message_truncation_for_long_topics(rig):
    """Core keeps only 200 characters of a tool message; the missing role comes
    first and per-topic text is shortened rather than growing without bound."""
    long = NS + '/x' * 200
    rig.topic_out[HEAD] = [{'topic': long, 'format': 'image/jpeg'}]
    rig.camera_info[HEAD] = [_declaration(long + '/other', 'image/jpeg')]
    assert _run(rig, _layout([_conn(HEAD, VO, long), RGBD[1]])) is False
    assert 'missing image/jpeg' in _vo_error(rig)
    assert len(rig.results[VO]['message']) < 600
