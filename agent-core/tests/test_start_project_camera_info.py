"""What start-project does for `camera_info`, which runs *with* its grain.

The mirror image of the control plane. There, a command producer needs its
consumer's action space and so the lookup runs against the start order and has to
call `info()` itself. Here a consumer needs what its **sources** say about the
optics, and dependency order has already started them and already called
`info()` — so this costs a dictionary lookup and no round trip.

Why it exists at all: navi's avoidance corridor is metric, so every frame it
converts its half-width back into a column range using the camera's field of
view. On r1_sz that number was typed into navi's own config as 0.55 rad against a
lens that measures 0.888, which made the corridor 1.86 m wide — wider than any
door. Every doorframe read as dead ahead and the robot turned away from gaps it
fitted through, while the depth map reported clear. The camera knew; it had no way
to say so. Format: `phanthymotus-driver/README_dev.md` § Camera Parameters.

Run: cd agent-core && python3 -m pytest tests/test_start_project_camera_info.py
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
import config  # noqa: E402


CAM, CAM2 = 'card-cam', 'card-cam2'
DEPTH, VOP, NAVI, MIC = 'card-depth', 'card-vop', 'card-navi', 'card-mic'

CAM_TOPIC, CAM2_TOPIC = '/ubuntu/camera/main', '/ubuntu/camera/side'
DEPTH_TOPIC = '/ubuntu/camera/main/visual_depth'
OBJECTS_TOPIC = '/ubuntu/camera/main/objects'


def _decl(topic, **over):
    out = {
        'schema': 'motus.camera/1',
        'topic': topic,
        'id': 'unitree/r1/camera_main',
        'width': 1280, 'height': 720,
        'distortion_model': 'unknown',
        'D': None, 'K': None,
        'half_fov_rad': 0.888,
        'half_fov_v_rad': None,
        'source': 'measured',
        'pipeline': ['unitree/r1/camera_main'],
    }
    out.update(over)
    return out


def _card(cid, tool):
    return {'id': cid, 'toolName': tool, 'mcpId': f'mcp-{tool}'}


def _conn(src, dst, topic):
    return {'fromCardId': src, 'toCardId': dst, 'fromPortIdx': '0',
            'toPortIdx': '0', 'format': 'data/json', 'fromTopic': topic}


@pytest.fixture
def rig(monkeypatch):
    """Cards that answer `info` the way a camera and a depth processor do."""
    starts = {}
    infos = []

    topic_outs = {
        CAM:   [{'topic': CAM_TOPIC, 'format': 'image/jpeg'}],
        CAM2:  [{'topic': CAM2_TOPIC, 'format': 'image/jpeg'}],
        DEPTH: [{'topic': DEPTH_TOPIC, 'format': 'image/depth-zlib'}],
        VOP:   [{'topic': OBJECTS_TOPIC, 'format': 'data/json'}],
        MIC:   [{'topic': '/ubuntu/mic', 'format': 'audio/pcm-16k'}],
    }
    # Set by individual tests. A card absent from here declares nothing, which
    # is the normal case for almost every card in the repository.
    declarations: dict = {}

    async def _call(mcp_id, req, timeout_s=None):
        args = dict(req.arguments)
        action, card_id = args.get('action'), args.get('instance_id')
        if action == 'start':
            starts[card_id] = args
            return {'code': 200, 'data': {'state': 'running'}}
        if action == 'info':
            infos.append(card_id)
            data = {'state': 'idle', 'topic_out': topic_outs.get(card_id, [])}
            if card_id in declarations:
                data['camera_info'] = declarations[card_id]
            return {'code': 200, 'data': data}
        return {'code': 200, 'data': {'state': 'idle'}}

    class _Req:
        def __init__(self, tool, arguments):
            self.tool = tool
            self.arguments = arguments

    mcp = types.ModuleType('api.mcp_manage')
    mcp.mcp_call_tool = _call
    mcp.MCPCallRequest = _Req

    events = []

    async def _push(event):
        events.append(event)

    stream = types.ModuleType('api.motus_stream')
    stream.push_event = _push

    inspection = types.ModuleType('api.inspection')

    async def _register(topic, fmt, mcp_id):
        return None

    inspection.register_topic_internal = _register

    chan = types.ModuleType('channel.manager')
    chan.manager = types.SimpleNamespace(sync_from_canvas=lambda: None, _adapters={})
    chan._get_channel_configs = lambda: []

    for name, mod in (('api.mcp_manage', mcp), ('api.motus_stream', stream),
                      ('api.inspection', inspection), ('channel.manager', chan)):
        monkeypatch.setitem(sys.modules, name, mod)

    return types.SimpleNamespace(starts=starts, events=events, infos=infos,
                                 declarations=declarations)


def _run(layout) -> bool:
    config.main['canvas_layout'] = layout
    return asyncio.run(config_api._do_start_project_impl())


def _errors(events):
    return [e['payload'] for e in events
            if e.get('type') == 'project_start_item'
            and e['payload'].get('status') == 'error']


# ── it arrives, keyed by the topic the consumer bound ────────────────────────

def test_the_cameras_optics_reach_the_card_that_subscribes_to_it(rig):
    rig.declarations[CAM] = [_decl(CAM_TOPIC)]
    layout = {
        'cards': [_card(CAM, 'camera_main'), _card(DEPTH, 'visual_depth')],
        'connections': [_conn(CAM, DEPTH, CAM_TOPIC)],
    }
    _run(layout)

    assert rig.starts[DEPTH]['camera_info'] == {CAM_TOPIC: _decl(CAM_TOPIC)}


def test_it_is_keyed_by_topic_so_a_consumer_can_find_its_own_input(rig):
    """A consumer knows which topic it bound as which role. It does not know the
    upstream card's name, and must not need to: inputs are dispatched by what
    they carry, which is what lets a different depth source work."""
    rig.declarations[VOP] = [_decl(OBJECTS_TOPIC, source='inherited')]
    rig.declarations[DEPTH] = [_decl(DEPTH_TOPIC, width=640, height=480,
                                     source='inherited',
                                     pipeline=['unitree/r1/camera_main',
                                               'perception/visual_depth'])]
    layout = {
        'cards': [_card(VOP, 'vop'), _card(DEPTH, 'visual_depth'),
                  _card(NAVI, 'navi')],
        'connections': [_conn(VOP, NAVI, OBJECTS_TOPIC),
                        _conn(DEPTH, NAVI, DEPTH_TOPIC)],
    }
    _run(layout)

    got = rig.starts[NAVI]['camera_info']
    assert set(got) == {OBJECTS_TOPIC, DEPTH_TOPIC}
    # The depth entry describes the image *that port* publishes, resampled —
    # while the id stays the same, which is what downstream lookups join on.
    assert (got[DEPTH_TOPIC]['width'], got[DEPTH_TOPIC]['height']) == (640, 480)
    assert got[DEPTH_TOPIC]['id'] == got[OBJECTS_TOPIC]['id']


def test_two_cameras_do_not_bleed_into_each_other(rig):
    """The failure this keying prevents: one declaration standing in for the
    other means a consumer silently applies one lens's geometry to the other's
    picture, which is the original bug with extra steps."""
    rig.declarations[CAM] = [_decl(CAM_TOPIC, half_fov_rad=0.888)]
    rig.declarations[CAM2] = [_decl(CAM2_TOPIC, id='unitree/r1/camera_left',
                                    half_fov_rad=None, source='unknown')]
    layout = {
        'cards': [_card(CAM, 'camera_main'), _card(CAM2, 'camera_left'),
                  _card(NAVI, 'navi')],
        'connections': [_conn(CAM, NAVI, CAM_TOPIC),
                        _conn(CAM2, NAVI, CAM2_TOPIC)],
    }
    _run(layout)

    got = rig.starts[NAVI]['camera_info']
    assert got[CAM_TOPIC]['half_fov_rad'] == 0.888
    assert got[CAM2_TOPIC]['half_fov_rad'] is None


# ── an upstream that declares nothing is not a fault ─────────────────────────

def test_a_source_that_declares_nothing_still_starts_the_project(rig):
    """Almost no card in this repository has heard of the format. If a missing
    declaration failed a start, adding it would have broken every canvas."""
    layout = {
        'cards': [_card(CAM, 'camera_main'), _card(DEPTH, 'visual_depth')],
        'connections': [_conn(CAM, DEPTH, CAM_TOPIC)],
    }
    assert _run(layout) is not False
    assert not _errors(rig.events)
    # And the argument is absent rather than present-and-empty, so a consumer's
    # "did anybody tell me anything" test is a plain membership check.
    assert 'camera_info' not in rig.starts[DEPTH]


def test_a_non_camera_source_contributes_nothing(rig):
    rig.declarations[CAM] = [_decl(CAM_TOPIC)]
    layout = {
        'cards': [_card(CAM, 'camera_main'), _card(MIC, 'mic'),
                  _card(DEPTH, 'visual_depth')],
        'connections': [_conn(CAM, DEPTH, CAM_TOPIC),
                        _conn(MIC, DEPTH, '/ubuntu/mic')],
    }
    _run(layout)

    assert list(rig.starts[DEPTH]['camera_info']) == [CAM_TOPIC]


def test_a_malformed_declaration_does_not_block_the_start(rig):
    """A card can ship a broken declaration. That is a bug in that card, not a
    reason for the project to refuse to come up — the consumer validates and
    degrades, and says which input it could not read."""
    rig.declarations[CAM] = ['not a dict', {'no': 'topic'}, _decl(CAM_TOPIC)]
    layout = {
        'cards': [_card(CAM, 'camera_main'), _card(DEPTH, 'visual_depth')],
        'connections': [_conn(CAM, DEPTH, CAM_TOPIC)],
    }
    assert _run(layout) is not False
    assert list(rig.starts[DEPTH]['camera_info']) == [CAM_TOPIC]


def test_a_card_with_no_inbound_connection_gets_no_argument(rig):
    rig.declarations[CAM] = [_decl(CAM_TOPIC)]
    layout = {'cards': [_card(CAM, 'camera_main')], 'connections': []}
    _run(layout)

    assert 'camera_info' not in rig.starts[CAM]


# ── it rides the info() call that was already being made ─────────────────────

def test_no_extra_round_trip_is_taken_for_it(rig):
    """The whole reason this direction is cheap. `_resolve_and_register` already
    calls `info()` on every card right after it starts; this reuses that answer.

    A second call would also be *wrong* here — it would run after the source is
    live, so a card reporting different optics once streaming could contradict
    what the consumer already adopted.

    Counted in the fixture's own `_call` rather than by wrapping `mcp_call_tool`
    a second time. That wrapper depended on which module object
    `api.mcp_manage` resolved to, and the other start-project test files replace
    it too — so it passed when run alone and silently counted **nothing** in a
    full run, which is the failure mode a test must not have.
    """
    rig.declarations[CAM] = [_decl(CAM_TOPIC)]
    layout = {
        'cards': [_card(CAM, 'camera_main'), _card(DEPTH, 'visual_depth')],
        'connections': [_conn(CAM, DEPTH, CAM_TOPIC)],
    }
    _run(layout)

    assert rig.infos, 'info() 一次都没被调用 —— 这条测试在空转'
    assert rig.infos.count(CAM) == 1, f'camera 被 info() 了 {rig.infos.count(CAM)} 次'
