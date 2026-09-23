"""Saved Canvas graph -> arm -> headset session -> confirmed project shutdown.

Only in-process MCP doubles are used. No robot, SSH, DDS or live service is read.
"""
import asyncio
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import types

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
os.environ.setdefault('DB_PATH', str(Path(tempfile.mkdtemp()) / 'core.db'))
from teleop_project import (TeleopProjectError, resolve_bindings, validate_profile,
                            validate_target, require_editable, shutdown_complete)
from api import config as api
import config


COMMAND = '/testbot/motion/teleop/command'
FEEDBACK = '/testbot/motion/teleop/feedback'
META = {'protocol_version': 1, 'robot_profile': 'tianyi2', 'namespace': 'testbot',
        'command_topic': COMMAND, 'feedback_topic': FEEDBACK}
SOURCE = {'name': 'teleop', 'type': 'processor',
          'inputSchema': {'properties': {'action': {'enum': ['project_start', 'project_stop']}}},
          'topic_out': [{'topic': '', 'format': 'control/teleop'}]}
TARGET = {'name': 'teleop_executor', 'type': 'actuator', 'x-teleop-target': META,
          'inputSchema': {'type':'object','properties':{'action':{'type':'string','enum':['info','start','stop']}}, 'required':['action'],'additionalProperties':False},
          'topic_in': [{'topic': COMMAND, 'format': 'control/teleop'}],
          'topic_out': [{'topic': FEEDBACK, 'format': 'data/json'}]}
REGISTRY = [{'id': 'ac', 'url': 'http://127.0.0.1:15730/mcp', 'tools': [SOURCE]},
            {'id': 'robot-dynamic-id', 'url': 'http://127.0.0.1:15888/mcp', 'tools': [TARGET]}]
TELEOP = {'id': 'pendant', 'toolName': 'teleop', 'mcpId': 'ac'}
DRIVER = {'id': 'arm', 'toolName': 'teleop_executor', 'mcpId': 'robot-dynamic-id'}
LAYOUT = {'cards': [DRIVER, TELEOP], 'connections': [{
    'fromCardId': 'pendant', 'toCardId': 'arm', 'fromPortIdx': '0', 'toPortIdx': '0',
    'format': 'control/teleop', 'fromTopic': '/untrusted/cached/topic'}]}
COMPLETE = {'state': 'idle', 'armed': False, 'return_completed': True, 'authority_released': True}


def test_binding_uses_registry_and_metadata_not_cached_url_or_topic():
    layout = copy.deepcopy(LAYOUT)
    layout['cards'][0]['url'] = 'http://remote/mcp'
    binding = resolve_bindings(layout, REGISTRY)['pendant']
    assert binding == {**META, 'mcp_id': 'robot-dynamic-id', 'tool': 'teleop_executor',
                       'url': 'http://127.0.0.1:15888/mcp'}


@pytest.mark.parametrize('mutation', [
    lambda l, r: l.update(connections=[]),
    lambda l, r: r[0]['tools'][0]['inputSchema']['properties']['action'].update(enum=['info', 'start', 'stop']),
    lambda l, r: l['connections'].append(copy.deepcopy(l['connections'][0])),
    lambda l, r: l['cards'][0].update(toolName='arm'),
    lambda l, r: l['connections'][0].update(format='data/json'),
    lambda l, r: l['connections'][0].update(fromPortIdx='-1'),
    lambda l, r: l['connections'][0].update(toPortIdx='1'),
    lambda l, r: r[1].update(url='http://robot.example/mcp'),
    lambda l, r: r[1].update(url='http://127.0.0.1:15888/mcp?redirect=1'),
    lambda l, r: r[1].update(url='http://secret@127.0.0.1/mcp'),
    lambda l, r: r[1]['tools'][0].pop('x-teleop-target'),
    lambda l, r: r[1]['tools'][0]['x-teleop-target'].update(protocol_version=True),
    lambda l, r: r[1]['tools'][0]['x-teleop-target'].update(command_topic='/other'),
    lambda l, r: r[1]['tools'][0]['x-teleop-target'].update(namespace='../bad'),
    lambda l, r: r[1]['tools'][0]['topic_in'][0].update(format='data/json'),
    lambda l, r: r[1]['tools'][0].update(topic_out=[]),
    lambda l, r: l['cards'].append({**TELEOP, 'id': 'second'}),
])
def test_bad_binding_is_rejected_before_any_motion(mutation):
    layout, registry = copy.deepcopy(LAYOUT), copy.deepcopy(REGISTRY)
    mutation(layout, registry)
    with pytest.raises(TeleopProjectError):
        resolve_bindings(layout, registry)


def test_profile_mismatch_rejected():
    binding = resolve_bindings(LAYOUT, REGISTRY)['pendant']
    with pytest.raises(TeleopProjectError, match='机型'):
        validate_profile(binding, {'configuration': {'robot_profile': 'g1_23'}})
    validate_profile(binding, {'configuration': {'robot_profile': 'tianyi2'}})


@pytest.mark.parametrize('mutation', [
    lambda d: d.pop('x-teleop-target'),
    lambda d: d['x-teleop-target'].update(namespace='changed'),
    lambda d: d['x-teleop-target'].update(robot_profile='g1_23'),
    lambda d: d['x-teleop-target'].update(protocol_version=True),
    lambda d: d['x-teleop-target'].update(command_topic='/different'),
    lambda d: d.update(topic_in=[]),
    lambda d: d.update(topic_out=[]),
])
def test_current_driver_must_match_registered_target(mutation):
    binding = resolve_bindings(LAYOUT, REGISTRY)['pendant']
    live = copy.deepcopy(TARGET)
    validate_target(binding, live)
    mutation(live)
    with pytest.raises(TeleopProjectError, match='Driver 当前'):
        validate_target(binding, live)


@pytest.mark.parametrize('payload', [
    {}, {'state': 'accepted'}, {**COMPLETE, 'authority_released': False},
    {**COMPLETE, 'armed': True}, {**COMPLETE, 'return_completed': False},
    {**COMPLETE, 'error': 'collision'}, {**COMPLETE, 'state': 'returning'},
])
def test_shutdown_requires_completion_not_acceptance(payload):
    assert not shutdown_complete(payload)


@pytest.mark.parametrize('phase', ['stopping', 'stop_failed'])
def test_layout_stays_locked_even_if_legacy_running_bit_is_false(phase):
    with pytest.raises(TeleopProjectError):
        require_editable({'project_running': False, 'project_phase': phase})


@pytest.fixture
def robot(monkeypatch):
    """The real Core orchestration talks to a bounded deterministic local sink."""
    state = types.SimpleNamespace(calls=[], events=[], armed=False, moving=False,
                                  finish=copy.deepcopy(COMPLETE), finish_gate=None,
                                  profile='tianyi2', fail_driver_start=False,
                                  target=copy.deepcopy(TARGET), driver_ready=False)
    async def call(mid, req, timeout_s=None):
        args = dict(req.arguments)
        state.calls.append((mid, req.tool, args))
        action = args['action']
        if req.tool == 'teleop_executor':
            import jsonschema
            jsonschema.validate(args, TARGET['inputSchema'])
        if action == 'info':
            assert timeout_s
            data = {'state': 'idle', **state.target}
            if req.tool == 'teleop':
                data = {'state': 'idle', 'armed': state.armed,
                        'configuration': {'robot_profile': state.profile},
                        'topic_out': [{'topic': '/previous/motion/teleop/command', 'format': 'control/teleop'}]}
            return {'code': 200, 'data': data}
        if action == 'project_start':
            assert req.tool == 'teleop' and args['driver_binding']['command_topic'] == COMMAND
            assert state.driver_ready, 'PICO enabled before downstream was ready'
            state.armed = True
            assert state.moving is False
            return {'code': 200, 'data': {'state': 'armed', 'armed': True}}
        if action == 'project_stop':
            assert timeout_s == 55.
            if state.finish_gate:
                await state.finish_gate.wait()
            if shutdown_complete(state.finish):
                state.moving = state.armed = False
            return {'code': 200, 'data': copy.deepcopy(state.finish)}
        if action == 'start' and state.fail_driver_start:
            return {'code': 400, 'message': 'driver_not_ready'}
        if action == 'start' and req.tool == 'teleop_executor':
            assert not state.armed
            state.driver_ready = True
        if action == 'stop' and req.tool == 'teleop_executor':
            assert not state.moving and not state.armed, 'Driver stopped before return/release'
        return {'code': 200, 'data': {'state': 'idle'}}
    async def push(event): state.events.append(event)
    async def register(*args): pass
    modules = {
        'api.mcp_manage': dict(mcp_call_tool=call, MCPCallRequest=types.SimpleNamespace),
        'api.motus_stream': dict(push_event=push),
        'api.inspection': dict(register_topic_internal=register),
        'channel.manager': dict(manager=types.SimpleNamespace(sync_from_canvas=lambda: None, _adapters={}),
                                _get_channel_configs=lambda: []),
    }
    for name, values in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(api, '_start_project_task', None)
    monkeypatch.setattr(api, '_stop_project_task', None)
    config.main['core'] = {'project_running': False}
    config.main['canvas_layout'] = copy.deepcopy(LAYOUT)
    config.main['services'] = {'mcp': copy.deepcopy(REGISTRY)}
    return state


def actions(robot):
    return [(tool, args['action']) for _, tool, args in robot.calls if args['action'] != 'info']


def test_real_project_start_arms_without_session_and_stop_returns_before_driver(robot):
    async def run():
        assert await api._do_start_project_impl() is True
        assert actions(robot) == [('teleop_executor', 'start'), ('teleop', 'project_start')]
        assert robot.armed and not robot.moving
        # Only the paired operator starts moving after arming, outside Core's lifecycle.
        robot.moving = True
        assert await api._do_stop_project() is True
        assert actions(robot)[-2:] == [('teleop', 'project_stop'), ('teleop_executor', 'stop')]
        assert not config.main['core']['project_running']
        assert config.main['core']['project_phase'] == 'idle'
    asyncio.run(run())


@pytest.mark.parametrize('bad', ['missing_edge', 'wrong_profile', 'live_target_changed'])
def test_project_preflight_failure_does_not_start_driver_or_arm(robot, bad):
    if bad == 'missing_edge': config.main['canvas_layout'] = {**LAYOUT, 'connections': []}
    elif bad == 'wrong_profile': robot.profile = 'g1_23'
    else: robot.target['x-teleop-target']['namespace'] = 'changed'
    assert asyncio.run(api._do_start_project_impl()) is False
    assert actions(robot) == []
    assert config.main['core']['project_start_error']


def test_failed_shutdown_preserves_graph_driver_and_retry(robot):
    async def run():
        await api._do_start_project_impl()
        robot.moving = True
        robot.finish = {'state': 'error', 'error': 'return_collision'}
        response = await api.api_stop_project()
        assert response.status_code == 409
        assert json.loads(response.body)['detail'].endswith('return_collision')
        assert ('teleop_executor', 'stop') not in actions(robot)
        assert config.main['canvas_layout'] == LAYOUT
        assert config.main['core']['project_running'] is True
        assert config.main['core']['project_phase'] == 'stop_failed'
        blocked = await api.api_start_project()
        assert blocked.status_code == 409
        robot.finish = copy.deepcopy(COMPLETE)
        response = await api.api_stop_project()
        assert response['ok'] is True
        assert ('teleop_executor', 'stop') == actions(robot)[-1]
    asyncio.run(run())


def test_two_stop_requests_join_and_browser_cancel_does_not_cancel_return(robot):
    async def run():
        robot.armed = robot.moving = True
        robot.finish_gate = asyncio.Event()
        first = asyncio.create_task(api._do_stop_project())
        await asyncio.sleep(0)
        second = asyncio.create_task(api._do_stop_project())
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError): await first
        assert actions(robot) == [('teleop', 'project_stop')]
        robot.finish_gate.set()
        assert await second is True
        assert actions(robot) == [('teleop', 'project_stop'), ('teleop_executor', 'stop')]
    asyncio.run(run())


def test_legacy_running_put_uses_real_lifecycle(robot):
    async def run():
        assert (await api.set_project_running(api.ProjectRunningRequest(running=True)))['ok']
        assert robot.armed and not robot.moving
        robot.moving = True
        assert (await api.set_project_running(api.ProjectRunningRequest(running=False)))['ok']
        assert actions(robot)[-2:] == [('teleop', 'project_stop'), ('teleop_executor', 'stop')]
    asyncio.run(run())


def test_start_rollback_disarms_before_driver_shutdown(robot):
    robot.fail_driver_start = True
    assert asyncio.run(api._do_start_project_impl()) is False
    assert actions(robot)[-2:] == [('teleop', 'project_stop'), ('teleop_executor', 'stop')]
    assert robot.armed is False
    assert ('teleop', 'project_start') not in actions(robot)


@pytest.mark.parametrize('path', ['layout', 'solution'])
@pytest.mark.parametrize('phase', ['running', 'stopping', 'stop_failed'])
def test_live_graph_cannot_be_replaced_through_any_save_entry(robot, monkeypatch, path, phase):
    from api import canvas, solutions
    config.main['core'] = {'project_running': phase == 'running', 'project_phase': phase}
    monkeypatch.setattr(canvas, '_editor_session', 'editor')
    monkeypatch.setattr(canvas, '_editor_last_seen', canvas.time.monotonic())
    with pytest.raises(api.fastapi.HTTPException) as exc:
        if path == 'layout':
            asyncio.run(canvas.save_layout(canvas.CanvasLayout(cards=[], session_id='editor')))
        else:
            asyncio.run(solutions._apply_canvas({'cards': []}, {}))
    assert exc.value.status_code == 409
    assert config.main['canvas_layout'] == LAYOUT
    assert actions(robot) == []


@pytest.mark.parametrize('path', ['layout', 'solution'])
def test_removed_teleop_keeps_old_graph_until_return_is_confirmed(robot, monkeypatch, path):
    from api import canvas, solutions
    monkeypatch.setattr(canvas, '_editor_session', 'editor')
    monkeypatch.setattr(canvas, '_editor_last_seen', canvas.time.monotonic())
    robot.finish = {'state': 'error', 'error': 'return_unconfirmed'}
    async def write():
        if path == 'layout':
            return await canvas.save_layout(canvas.CanvasLayout(cards=[], session_id='editor'))
        return await solutions._apply_canvas({'cards': []}, {})
    with pytest.raises(api.fastapi.HTTPException) as exc:
        asyncio.run(write())
    assert exc.value.status_code == 409
    assert config.main['canvas_layout'] == LAYOUT
    assert actions(robot) == [('teleop', 'project_stop')]
    robot.finish = copy.deepcopy(COMPLETE)
    asyncio.run(write())
    assert config.main['canvas_layout']['cards'] == []
    assert actions(robot)[-2:] == [('teleop', 'project_stop'), ('teleop_executor', 'stop')]


def test_g1_standalone_shadow_can_stop_without_promising_return(robot):
    registry = copy.deepcopy(REGISTRY)
    registry[0]['tools'][0]['inputSchema']['properties']['action']['enum'] = ['info', 'start', 'stop']
    config.main['services'] = {'mcp': registry}
    assert asyncio.run(api.stop_removed_cards([TELEOP], [])) == 1
    assert actions(robot) == [('teleop', 'stop')]


@pytest.mark.parametrize('phase,expected', [('running', True), ('stopping', False), ('stop_failed', False)])
def test_collector_rejects_new_work_while_returning_or_stop_failed(robot, phase, expected):
    import collector
    config.main['core'] = {'project_running': True, 'project_phase': phase}
    assert collector.project_running() is expected


def test_nonteleop_layout_retains_existing_edit_contract():
    require_editable({'project_running': True}, {'cards': [{'toolName': 'tts'}]}, {'cards': []})
    with pytest.raises(TeleopProjectError):
        require_editable({'project_running': True}, LAYOUT, {'cards': []})
    with pytest.raises(TeleopProjectError):
        require_editable({'project_running': True}, {'cards': []}, LAYOUT)
