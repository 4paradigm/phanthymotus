"""PICO operator lifecycle through v2 EEF, real IK and the actual arm gate.

Only host allocation, transport, sensor sampling and the finite plant are
substitutes. Operator start, calibration, fixed-body preparation, mapping, IK,
command validation, hold and return remain production code. No sockets, ROS
entities or robot hardware are allocated. This proves an offline control chain,
not physical calibration or PICO rendering.
"""
import ast
import asyncio
import copy
import json
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from test_motion_control_chain import chain, motion_frame, target_frame, wait_for  # noqa: F401
from test_teleop import frame
from test_tianyi_execution_chain import tianyi_driver_source
from teleop.motion_control import EefIntentAdapter
from teleop.operator_session import OperatorCommands
from teleop.plugin import TeleopPlugin
from teleop.protocol import bind_rtc_frame_v1
from teleop.runtime import TeleopRuntime


@pytest.fixture
def operator_chain(chain, monkeypatch):
    h = chain
    # Both admission paths consume an explicitly synthetic evidence document;
    # do not replace the actual acceptance predicate with an always-true stub.
    from teleop_executor import load_profile
    profile = json.loads(h.path.read_text())
    profile['first_acceptance'] = copy.deepcopy(profile['acceptance'])
    h.path.write_text(json.dumps(profile))
    h.e.profile, _, h.e.profile_sha256 = load_profile(h.path)
    h.c.calibrate()
    # Import only the literal motor maps: importing device.py would allocate its
    # ROS-dependent module graph. Baseline validation uses the real map values.
    wanted = {'_HEAD_JOINTS', '_WAIST_JOINTS', '_LEG_JOINTS'}
    tree = ast.parse((tianyi_driver_source()/'device.py').read_text())
    maps = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id in wanted}
    assert set(maps) == wanted
    monkeypatch.setitem(sys.modules, 'device', SimpleNamespace(**maps))
    fixed = {'positions': {str(mid): .01 for values in maps.values() for mid in values},
             'timestamp': time.monotonic_ns(), 'incomplete': False, 'power_stale': False}
    # Sampling is independent of the time the baseline is requested; queries
    # cannot turn a stale sample into a fresh one.
    sample = h.p.tick

    def tick():
        sample()
        with h.e._lock:
            fixed['timestamp'] = time.monotonic_ns()
            h.e._streams.update({part: (fixed['timestamp'],
                {mid: (.01, 0., 0) for mid in maps[name]})
                for part, name in (('head', '_HEAD_JOINTS'),
                                   ('waist', '_WAIST_JOINTS'), ('leg', '_LEG_JOINTS'))})

    snapshot = h.p.snapshot

    def feedback():
        value = snapshot()
        with h.e._lock:
            positions = dict(fixed['positions'])
            if fixed['incomplete']:
                positions.pop(next(iter(positions)))
            return {**value, 'fixed_ns': fixed['timestamp'],
                    'power_ns': 0 if fixed['power_stale'] else value['power_ns'],
                    'fixed_motor_positions_rad': positions}

    monkeypatch.setattr(h.p, 'tick', tick)
    monkeypatch.setattr(h.e.gate, 'snapshot', feedback)
    # The lower-level fixture intentionally elides fixed-body preparation.
    # Restore it here so an operator start cannot pass on an incomplete sensor.
    monkeypatch.setattr(h.e, '_capture_fixed_baseline',
                        type(h.e)._capture_fixed_baseline.__get__(h.e))
    tick()
    h.wire.feedback()
    adapter = EefIntentAdapter(h.link, 'live')
    card = TeleopPlugin({'robot_profile': 'tianyi2', 'mode': 'live',
                        'control_backend': 'motion_control', 'namespace': 'offline',
                        'driver_mcp_url': h.link.url, 'operator_session_enabled': True}, None)
    connection = SimpleNamespace(connection_id='offline-pico', events=asyncio.Queue())
    capture = SimpleNamespace(_connection=connection, presence_expired=lambda c: False)
    assignments, revocations = [], []

    async def assign():
        assignments.append(card.runtime.rtc_authority_snapshot()[0])

    async def revoke(reason):
        revocations.append(reason)

    capture.issue_assignment_if_connected, capture.revoke_assignment = assign, revoke
    broker = OperatorCommands(capture, card._operator_execute)

    def open_host():
        card.link, card.adapter, card.capture = h.link, adapter, capture
        card.operator_commands = broker
        card.runtime = TeleopRuntime(mode='live', adapter=adapter, auto_watchdog=False,
                                     pose_timeout_ms=300, dispatch_io_timeout_ms=150)

    monkeypatch.setattr(card, '_open_host', open_host)
    monkeypatch.setattr(card, '_run', lambda coroutine, timeout=2: asyncio.run(coroutine))
    binding = {'mcp_id': 'offline-driver', 'tool': 'motion_control', 'url': h.link.url,
               'namespace': 'offline', 'robot_profile': 'tianyi2', 'protocol_version': 2,
               'command_topic': '/offline/motion/control/command',
               'feedback_topic': '/offline/motion/teleop/feedback'}
    binding['execution_binding'] = {**binding, 'tool': 'arm',
                                   'command_topic': '/offline/motion/arm/command',
                                   'resources': ['arm_l', 'arm_r']}
    result = card.dispatch('teleop', {'action': 'project_start', 'driver_binding': binding})
    assert result.get('armed') is True, result
    value = SimpleNamespace(**vars(h), card=card, adapter=adapter, broker=broker,
                            connection=connection, fixed=fixed, assignments=assignments,
                            revocations=revocations, sequence=0)
    try:
        yield value
    finally:
        card._operator_cancel.set()
        card.runtime.close()
        adapter.close()


def operator(h, action):
    async def run():
        request = {'action': action, 'connection_id': h.connection.connection_id,
                   'request_id': f'{action}-{time.monotonic_ns()}'}
        accepted = await h.broker.submit(h.connection, request)
        assert accepted['state'] == 'accepted'
        await h.broker.task
        return await h.broker.submit(h.connection, request)
    return asyncio.run(run())


def submit(h, held, clutch=1, poses=None):
    value = frame(h.sequence, held, clutch)
    value['mode'] = 'live'
    if poses:
        for key in ('head', 'left_controller', 'right_controller'):
            value[key] = copy.deepcopy(poses[key])
    authority, _ = h.card.runtime.rtc_authority_snapshot()
    h.card.runtime.submit_frame(bind_rtc_frame_v1(value, authority=authority,
                                                expected_mode='live'), source='offline-pico')
    if held:
        assert h.card.runtime._dispatcher.wait_dispatched(h.sequence, 1.), h.card.info()
    h.sequence += 1


def start(h):
    receipt = operator(h, 'start')
    assert receipt['state'] == 'completed', receipt
    assert receipt['result']['state'] == 'ready'
    assert h.e._operator_prepared and h.card.runtime.status()['authority_valid']
    assert h.e._fixed_baseline == h.fixed['positions']
    assert h.assignments and not h.link.lease and not h.e.gate.session_id
    assert not h.commands and not h.vendor_calls and not h.p.writes
    return receipt


def follow(h):
    start(h)
    submit(h, False, 0)
    submit(h, True)
    assert h.link.lease and not h.wire.eef_packets and not h.p.writes
    goal = h.p.q.copy()
    goal[3], goal[10] = .12, -.10
    reachable = target_frame(h.adapter, h.c.solver.palms(goal))
    for _ in range(8):
        submit(h, True, poses=reachable)
        assert h.c.process_latest()
        time.sleep(.025)
    assert h.commands[-1]['mode'] == 'joint_position'
    assert h.wire.eef_packets[-1]['mode'] == 'eef_pose'
    wait_for(lambda: bool(h.vendor_calls))
    wait_for(lambda: np.max(np.abs(h.p.q)) > .03)
    assert np.isfinite(h.p.q).all() and np.isfinite(h.p.dq).all()
    assert np.max(np.abs(h.p.dq)) <= 1.


def test_start_rejects_incomplete_fixed_feedback_then_retries_without_restart(operator_chain):
    h = operator_chain
    h.fixed['incomplete'] = True
    failed = operator(h, 'start')
    assert failed['state'] == 'failed' and failed['error'] == 'fixed_feedback_incomplete', failed
    assert not h.e._operator_prepared and not h.card.runtime.status()['authority_valid']
    assert not h.assignments and not h.link.lease and not h.p.writes
    h.fixed['incomplete'] = False
    start(h)
    assert operator(h, 'stop')['state'] == 'completed'
    assert not h.e._operator_prepared and not h.link.lease and not h.p.writes


def test_ready_start_is_zero_output_until_grips_and_stop_is_repeatable(operator_chain):
    h = operator_chain
    start(h)
    submit(h, False, 0)
    assert not h.link.lease and not h.wire.eef_packets and not h.p.writes
    for _ in range(2):
        receipt = operator(h, 'stop')
        assert receipt['state'] == 'completed' and receipt['result']['authority_released'], receipt
    assert not h.card.runtime.status()['authority_valid'] and not h.e._operator_prepared
    assert not h.p.writes


@pytest.mark.parametrize('fresh_sample_arrives', [True, False])
def test_power_stale_prepare_uses_real_gate_and_retries_only_fresh_samples(
        operator_chain, monkeypatch, fresh_sample_arrives):
    h = operator_chain
    h.fixed['power_stale'] = True
    rejected = threading.Event()
    prepare = h.e._capture_fixed_baseline

    def observed_prepare():
        try:
            return prepare()
        except ValueError as exc:
            if str(exc) == 'power_ns_stale':
                rejected.set()
            raise

    monkeypatch.setattr(h.e, '_capture_fixed_baseline', observed_prepare)

    def restore_sensor():
        assert rejected.wait(2)
        h.fixed['power_stale'] = False

    updater = threading.Thread(target=restore_sensor) if fresh_sample_arrives else None
    if updater:
        updater.start()
    try:
        receipt = operator(h, 'start')
    finally:
        if updater:
            updater.join(2)
            assert not updater.is_alive()
    assert rejected.is_set()
    assert len([r for r in h.wire.management if r['action'] == 'prepare_operator_session']) >= 2
    assert not h.link.lease and not h.p.writes
    if fresh_sample_arrives:
        assert receipt['state'] == 'completed' and h.e._operator_prepared, receipt
    else:
        assert receipt['state'] == 'failed' and receipt['error'] == 'power_ns_stale', receipt
        assert not h.e._operator_prepared and not h.card.runtime.status()['authority_valid']
        h.fixed['power_stale'] = False
        start(h)
    assert operator(h, 'stop')['state'] == 'completed'


def test_real_operator_follow_regrip_then_finish_returns_measured_neutral(operator_chain):
    h = operator_chain
    follow(h)
    first_session = h.link.lease['session_id']
    submit(h, False)
    wait_for(lambda: h.e.gate.status()['hold_confirmed'])
    wait_for(lambda: h.card.runtime.status()['dispatch']['stop_acknowledged'])
    submit(h, True, clutch=2)
    assert h.link.lease['session_id'] != first_session
    previous_count = len(h.commands)
    submit(h, True, clutch=2, poses=motion_frame(h, h.adapter))
    assert h.c.process_latest()
    assert len(h.commands) == previous_count+1
    assert h.commands[-1]['mapping_epoch'] == 2
    wait_for(lambda: h.e.gate.applied_seq >= 0)
    receipt = operator(h, 'finish')
    assert receipt['state'] == 'completed', receipt
    assert receipt['result']['return_completed'] and receipt['result']['authority_released']
    assert not h.link.lease and not h.e.gate.session_id and not h.e._operator_prepared
    state = h.e.gate.status()
    assert state['stop_confirmed'] and state['feedback']['arm_ns'] > h.e.gate.stop_sent_ns
    assert np.max(np.abs(h.p.q)) <= .02 and np.max(np.abs(h.p.dq)) <= .02
    assert h.card.info()['project']['armed']  # Another explicit start stays possible.


def test_real_operator_stop_releases_and_later_explicit_finish_can_return(operator_chain):
    h = operator_chain
    follow(h)
    stopped = operator(h, 'stop')
    assert stopped['state'] == 'completed', stopped
    assert stopped['result']['authority_released'] and not h.link.lease
    assert not h.e._operator_prepared and h.e.gate.status()['stop_confirmed']
    finished = operator(h, 'finish')
    assert finished['state'] == 'completed', finished
    assert finished['result']['return_completed'] and finished['result']['authority_released']
    assert not h.link.lease and not h.e.gate.session_id
    assert np.max(np.abs(h.p.q)) <= .02
