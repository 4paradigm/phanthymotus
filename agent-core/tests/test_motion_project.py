"""Three-stage graphs, including real Core start/stop and legacy compatibility."""
import asyncio
import copy
import json
import sys
import types

import pytest

from test_teleop_project_lifecycle import api, config, robot  # existing bounded Core fixture
from motion_project import (build_motion_template, motion_targets, validate_execution_target,
                            with_feedback_edges)
from teleop_project import TeleopProjectError, resolve_bindings, validate_target

META = {'protocol_version': 2, 'robot_profile': 'tianyi2', 'namespace': 'robot',
        'command_topic': '/robot/motion/control/command', 'feedback_topic': '/robot/motion/teleop/feedback'}
ARM_META = {**META, 'command_topic': '/robot/motion/arm/command', 'resources': ['arm_l', 'arm_r']}
SOURCE = {'name': 'teleop', 'type': 'processor',
          'inputSchema': {'properties': {'action': {'enum': ['info', 'project_start', 'project_stop']}}},
          'topic_out': [{'id': 'targets', 'format': 'control/eef'}],
          'topic_in': [{'id': 'feedback', 'role': 'feedback', 'format': 'data/json'}]}
SCHEMA = {'type': 'object', 'properties': {
    'action': {'enum': ['info', 'start', 'stop']}, 'instance_id': {'type': 'string'},
    'input_topic': {'type': 'string'}, 'control_interface': {'type': 'object'},
    'control_interfaces': {'type': 'object'}, 'execution_binding': {'type': 'object'}}, 'required': ['action'], 'additionalProperties': False}
JOINTS = {'control_interface': 'motus.control/2', 'mode': 'joint_position', 'dof': 14,
          'joint_names': ['joint' + str(i) for i in range(14)], 'units': {'angle': 'rad'}}
MOTION = {'name': 'motion_control', 'type': 'processor', 'inputSchema': SCHEMA,
          'x-teleop-target': META, 'x-motion-control': {**META, 'execution_tool': 'arm',
                                                     'execution_command_topic': ARM_META['command_topic']},
          'control_interface': {'control_interface': 'motus.control/2', 'mode': 'eef_pose', 'dof': 14},
          'topic_in': [{'port_id': 'targets', 'format': 'control/eef', 'topic': META['command_topic']}],
          'topic_out': [{'port_id': 'joints', 'format': 'control/joint', 'topic': ARM_META['command_topic']},
                        {'port_id': 'feedback', 'format': 'data/json', 'topic': META['feedback_topic']}]}
ARM = {'name': 'arm', 'type': 'actuator', 'inputSchema': SCHEMA, 'x-control-target': ARM_META,
       'control_interface': JOINTS, 'topic_in': [{'port_id': 'targets', 'format': 'control/joint',
                                              'topic': ARM_META['command_topic']}], 'topic_out': []}
REGISTRY = [{'id': 'ac', 'url': 'http://127.0.0.1:15730/mcp', 'tools': [SOURCE]},
            {'id': 'driver', 'url': 'http://127.0.0.1:15707/mcp', 'tools': [MOTION, ARM]}]
BARE = {'cards': [{'id': 'vr', 'mcpId': 'ac', 'toolName': 'teleop', 'x': 0, 'y': 0}], 'connections': []}


def template():
    return build_motion_template(BARE, REGISTRY, 'vr', 'driver')


def test_template_is_idempotent_registered_and_has_feedback_without_dag_cycle():
    layout = template()
    assert build_motion_template(layout, REGISTRY, 'vr', 'driver') == layout
    assert BARE['connections'] == [] and len(BARE['cards']) == 1
    binding = resolve_bindings(layout, REGISTRY)['vr']
    assert binding['tool'] == 'motion_control' and binding['execution_binding']['tool'] == 'arm'
    assert binding['execution_binding']['resources'] == ['arm_l', 'arm_r']
    ordered, cyclic = api.order_cards_by_dependency(layout['cards'], layout['connections'])
    assert not cyclic
    assert [c['toolName'] for c in ordered] == ['teleop', 'motion_control', 'arm']
    feedback = [e for e in layout['connections'] if e.get('role') == 'feedback']
    assert len(feedback) == 1 and feedback[0]['toCardId'] == 'vr'
    assert feedback[0]['fromTopic'] == META['feedback_topic']


@pytest.mark.parametrize('mutate', [
    lambda l, r: l['connections'].pop(1),
    lambda l, r: l['connections'].append(copy.deepcopy(l['connections'][1])),
    lambda l, r: r[1]['tools'][1]['x-control-target'].update(namespace='other'),
    lambda l, r: r[1]['tools'][1]['x-control-target'].update(resources=['arm_l']),
    lambda l, r: r[1]['tools'][0]['x-motion-control'].update(execution_tool='leg'),
    lambda l, r: r[1]['tools'][0]['topic_out'][0].update(topic='/unrelated'),
    lambda l, r: r[1].update(url='http://192.0.2.1/mcp'),
    lambda l, r: r[1]['tools'][0]['x-teleop-target'].update(protocol_version=True),
])
def test_wrong_chain_refused(mutate):
    layout, registry = template(), copy.deepcopy(REGISTRY)
    mutate(layout, registry)
    with pytest.raises(TeleopProjectError):
        resolve_bindings(layout, registry)


def test_feedback_flags_cannot_exempt_arbitrary_dependencies():
    layout = template()
    layout['connections'].append({'role': 'feedback', 'fromCardId': 'fake', 'toCardId': 'vr'})
    result = with_feedback_edges(layout, REGISTRY)
    assert not any(e.get('fromCardId') == 'fake' for e in result['connections'])
    assert len(result['connections']) == 3


def test_current_driver_and_execution_metadata_are_both_verified():
    binding = resolve_bindings(template(), REGISTRY)['vr']
    validate_target(binding, MOTION)
    validate_execution_target(binding['execution_binding'], ARM)
    changed = copy.deepcopy(ARM)
    changed['x-control-target']['namespace'] = 'changed'
    with pytest.raises(TeleopProjectError):
        validate_execution_target(binding['execution_binding'], changed)


@pytest.fixture
def chain(robot, monkeypatch):
    mcp = sys.modules['api.mcp_manage']
    robot.calls.clear()
    robot.started = set()
    robot.current_arm = copy.deepcopy(ARM)
    config.main['canvas_layout'] = template()
    config.main['services'] = {'mcp': copy.deepcopy(REGISTRY)}

    async def call(mid, req, timeout_s=None):
        import jsonschema
        args = req.arguments
        robot.calls.append((mid, req.tool, copy.deepcopy(args)))
        if req.tool in ('motion_control', 'arm'):
            jsonschema.validate(args, SCHEMA)
        action = args['action']
        if action == 'info':
            assert timeout_s
            tool = {'teleop': SOURCE, 'motion_control': MOTION, 'arm': robot.current_arm}[req.tool]
            return {'code': 200, 'data': {'state': 'idle', **copy.deepcopy(tool)}}
        if action == 'start':
            if req.tool == 'motion_control':
                assert args['input_topic'] == META['command_topic']
                assert args['control_interface'] == JOINTS
                assert args['control_interfaces'] == {'joints': JOINTS}
                assert args['execution_binding']['resources'] == ['arm_l', 'arm_r']
            if req.tool == 'arm':
                assert args['input_topic'] == ARM_META['command_topic']
            robot.started.add(req.tool)
            assert not robot.armed
        if action == 'project_start':
            assert robot.started == {'motion_control', 'arm'}
            assert args['driver_binding']['execution_binding']['tool'] == 'arm'
            robot.armed = True
            return {'code': 200, 'data': {'state': 'armed', 'armed': True}}
        if action == 'project_stop':
            if robot.finish.get('return_completed'):
                robot.armed = False
            return {'code': 200, 'data': copy.deepcopy(robot.finish)}
        if action == 'stop':
            assert not robot.armed, 'execution stopped before return confirmation'
        return {'code': 200, 'data': {'state': 'idle'}}
    monkeypatch.setattr(mcp, 'mcp_call_tool', call)
    return robot


def test_real_core_three_stage_start_and_confirmed_shutdown(chain):
    async def run():
        assert await api._do_start_project_impl() is True, config.main.get('core', {})
        assert chain.armed
        assert await api._do_stop_project() is True
        actions = [(t, a['action']) for _, t, a in chain.calls if a['action'] != 'info']
        assert actions[:3] == [('motion_control', 'start'), ('arm', 'start'), ('teleop', 'project_start')]
        assert actions[3] == ('teleop', 'project_stop')
        assert set(actions[4:]) == {('motion_control', 'stop'), ('arm', 'stop')}
    asyncio.run(run())


def test_execution_metadata_drift_fails_before_any_start(chain):
    chain.current_arm['x-control-target']['namespace'] = 'changed'
    assert asyncio.run(api._do_start_project_impl()) is False
    assert not [args for _, _, args in chain.calls if args['action'] != 'info']


def test_failed_return_keeps_both_driver_cards_available_for_retry(chain):
    async def run():
        assert await api._do_start_project_impl(), config.main.get('core', {})
        chain.finish = {'state': 'fault', 'error': 'return_failed'}
        assert await api._do_stop_project() is False
        assert not [args for _, tool, args in chain.calls if tool != 'teleop' and args['action'] == 'stop']
        assert config.main['core']['project_phase'] == 'stop_failed'
    asyncio.run(run())
