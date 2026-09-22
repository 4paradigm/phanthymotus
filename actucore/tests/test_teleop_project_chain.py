"""Canvas/PICO lifecycle against the real execution gate; no ROS or network.

The existing managed fixture supplies DriverLink's authenticated wire and the
actual Driver MCP dispatcher. Only site acceptance, geometry and the physical
plant are substitutes. Command publication never directly changes feedback.
"""
import asyncio
import os
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from test_teleop import frame
from test_tianyi_execution_chain import chain  # noqa: F401 -- pytest dependency
from test_tianyi_management_recovery import managed  # noqa: F401
from teleop.operator_session import OperatorCommands
from teleop.plugin import TeleopPlugin
from teleop.protocol import ProtocolError, bind_rtc_frame_v1
from teleop.runtime import TeleopRuntime


def await_condition(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError('condition did not become true within deadline')


class LagPlant:
    """Independently sampled, velocity/acceleration limited position servo."""

    def __init__(self, gate, writes):
        self.gate, self.writes = gate, writes
        self.lock = threading.RLock()
        self.q = np.full(14, .08)
        self.dq = np.zeros(14)
        self.goal = self.q.copy()
        self.stamp = time.monotonic_ns()
        self.samples = []
        self.commands = []
        self.failure = None
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def snapshot(self):
        with self.lock:
            return {**{key: self.stamp for key in ('arm_ns', 'power_ns', 'hand_ns', 'fixed_ns')},
                    'q': self.q.tolist(), 'dq': self.dq.tolist(), 'power_on': True,
                    'estop': False, 'fault': False, 'fixed_body': True}

    def emit(self, target, hands):
        with self.lock:
            self.goal = np.asarray(target, dtype=float).copy()
            self.writes.append((list(target), hands))
            self.commands.append((time.monotonic_ns(), self.q.copy(), self.goal.copy()))

    def _run(self):
        previous = time.monotonic()
        next_gate = previous
        while not self.closed.wait(.005):
            now = time.monotonic()
            dt = min(now - previous, .025)
            previous = now
            with self.lock:
                wanted = np.clip((self.goal - self.q) / .08, -.6, .6)
                self.dq += np.clip(wanted - self.dq, -8 * dt, 8 * dt)
                if self.failure == 'jammed':
                    self.dq[:] = 0
                self.q += self.dq * dt
                if self.failure != 'feedback_frozen':
                    self.stamp = time.monotonic_ns()
                self.samples.append((now, self.q.copy(), self.dq.copy(), self.stamp))
            # Never hold the plant lock while acquiring the Driver gate lock.
            if now >= next_gate:
                self.gate.tick()
                next_gate = now + .02

    def assert_settled_receipt(self):
        samples = [s for s in self.samples
                   if np.max(np.abs(s[1])) <= .02 and np.max(np.abs(s[2])) <= .02]
        assert samples, 'no measured neutral-and-stationary samples'
        end = self.samples[-1][0]
        continuous = [s for s in self.samples if end - .095 <= s[0] <= end]
        assert len(continuous) >= 5
        assert all(np.max(np.abs(s[1])) <= .02 and np.max(np.abs(s[2])) <= .02
                   for s in continuous), 'completion preceded actual settling'
        assert len({s[3] for s in continuous}) > 1, 'reused one stale feedback sample'


@pytest.fixture
def project_chain(request, monkeypatch, chain):
    # Capture the actual executor constructed by the shared managed fixture so
    # its preparation flag, including clearing it on release, stays real.
    monkeypatch.syspath_prepend(os.environ['TIANYI_DRIVER_SOURCE'])
    import teleop_executor
    executors = []
    initialize = teleop_executor.TeleopExecutor.__init__

    def tracked_init(instance, *args, **kwargs):
        initialize(instance, *args, **kwargs)
        executors.append(instance)

    monkeypatch.setattr(teleop_executor.TeleopExecutor, '__init__', tracked_init)
    adapter, link, gate, writes, requests, lost_replies = request.getfixturevalue('managed')
    driver, = executors
    driver.cfg['operator_session_enabled'] = True
    driver.profile_error = None
    driver.profile_sha256 = 'fixture'
    monkeypatch.setattr(teleop_executor, 'accepted',
                        lambda profile, first_acceptance=False: first_acceptance)
    monkeypatch.setattr(driver, '_capture_fixed_baseline', lambda: None)
    gate.acceptance_check = driver._execution_accepted
    gate.velocity = 1.
    plant = LagPlant(gate, writes)
    gate.snapshot, gate.emit = plant.snapshot, plant.emit
    solver = adapter.solver
    solver.hands_enabled = False
    solver.velocity = 1.
    solver.lock = threading.RLock()
    solver.indices = np.arange(14)
    solver.model = SimpleNamespace(lowerPositionLimit=np.full(14, -2.),
                                   upperPositionLimit=np.full(14, 2.))
    geometry_failure = [None]

    def transition(previous, target, budget):
        budget()
        if geometry_failure[0]:
            raise ValueError(geometry_failure[0])
        assert np.isfinite(target).all()
        assert np.max(np.abs(np.asarray(target) - previous)) <= .21

    solver._safe_transition = transition
    calibrations = []

    def calibrate(path):
        assert link.lease is None
        calibrations.append(link.feedback()['feedback']['q'])
        adapter.clutch = None
        return {'calibrated': True}

    monkeypatch.setattr(adapter, 'calibrate', calibrate)
    cfg = {'robot_profile': 'tianyi2', 'mode': 'live', 'namespace': 'isolated',
           'driver_mcp_url': link.url, 'calibration_path': 'synthetic',
           'operator_session_enabled': True}
    card = TeleopPlugin(cfg, None)
    connection = SimpleNamespace(connection_id='synthetic-pico', events=asyncio.Queue())
    capture = SimpleNamespace(_connection=connection, presence_expired=lambda c: False)
    revocations, assignments = [], []

    async def revoke(reason):
        revocations.append(reason)

    async def assign():
        assignments.append(card.runtime.rtc_authority_snapshot()[0])

    capture.revoke_assignment, capture.issue_assignment_if_connected = revoke, assign
    broker = OperatorCommands(capture, card._operator_execute)

    def open_host():
        card.link, card.adapter, card.capture = link, adapter, capture
        card.operator_commands = broker
        card.runtime = TeleopRuntime(mode='live', adapter=adapter, auto_watchdog=False,
                                    pose_timeout_ms=300, dispatch_io_timeout_ms=150)

    monkeypatch.setattr(card, '_open_host', open_host)
    monkeypatch.setattr(card, '_run', lambda coroutine, timeout=2: asyncio.run(coroutine))
    binding = {'mcp_id': 'synthetic-driver', 'tool': 'teleop_executor', 'url': link.url,
               'namespace': 'isolated', 'robot_profile': 'tianyi2', 'protocol_version': 1,
               'command_topic': '/isolated/motion/teleop/command',
               'feedback_topic': '/isolated/motion/teleop/feedback'}
    plant.thread.start()
    harness = SimpleNamespace(card=card, adapter=adapter, link=link, gate=gate, driver=driver,
                              writes=writes, plant=plant, requests=requests,
                              lost_replies=lost_replies, capture=capture, broker=broker,
                              connection=connection, binding=binding, sequence=0,
                              geometry_failure=geometry_failure, calibrations=calibrations,
                              revocations=revocations, assignments=assignments)
    yield harness
    card._operator_cancel.set()
    plant.failure = None
    geometry_failure[0] = None
    if card.runtime:
        card.runtime.close()
    adapter.close()
    plant.closed.set()
    plant.thread.join(1)
    assert not plant.thread.is_alive()


def project_start(h):
    result = h.card.dispatch('teleop', {'action': 'project_start', 'driver_binding': h.binding})
    assert result.get('armed') is True, result
    return result


def pico(h, action, request_id=None):
    async def run():
        message = {'action': action, 'connection_id': h.connection.connection_id,
                   'request_id': request_id or f'{action}-{time.monotonic_ns()}'}
        accepted = await h.broker.submit(h.connection, message)
        if accepted['state'] == 'accepted':
            await h.broker.task
        return await h.broker.submit(h.connection, message)
    return asyncio.run(run())


def send(h, held, clutch=1, forward=0.):
    value = frame(h.sequence, held, clutch)
    value['mode'] = 'live'
    for side in ('left', 'right'):
        value[side + '_controller']['position'][2] -= forward
    authority, _ = h.card.runtime.rtc_authority_snapshot()
    result = h.card.runtime.submit_frame(
        bind_rtc_frame_v1(value, authority=authority, expected_mode='live'), source='synthetic')
    if held:
        assert h.card.runtime._dispatcher.wait_dispatched(h.sequence, 1.), h.card.info()
    h.sequence += 1
    return result


def start_follow(h):
    project_start(h)
    receipt = pico(h, 'start')
    assert receipt['state'] == 'completed', receipt
    assert not h.link.lease and not h.writes
    send(h, False, 0)
    send(h, True)
    assert h.link.lease and not h.writes  # Acquisition is distinct from a target.
    for step in range(1, 7):
        send(h, True, forward=.012 * step)
        time.sleep(.025)
    assert h.gate.applied_seq > 0
    assert any(np.max(np.abs(actual - target)) > .001
               for _, actual, target in h.plant.commands)
    assert np.max(np.abs(h.plant.snapshot()['dq'])) > 0


def assert_released(h, result):
    assert result.get('return_completed') is True, result
    assert result.get('authority_released') is True, result
    assert h.link.lease is None and h.link.management_request is None
    status = h.gate.status()
    assert status['ownership_held'] is False and status['stop_confirmed'] is True
    assert status['feedback']['arm_ns'] > h.gate.stop_sent_ns
    h.plant.assert_settled_receipt()


def test_project_arm_and_unused_stop_do_not_prepare_claim_or_home(project_chain):
    h = project_chain
    project_start(h)
    project_start(h)  # An identical binding is idempotent.
    assert not h.card.runtime.status()['authority_valid']
    assert not h.requests and not h.writes and not h.link.lease
    before = h.plant.snapshot()['q']
    for _ in range(2):
        result = h.card.dispatch('teleop', {'action': 'project_stop'})
        assert result.get('armed') is False and result.get('authority_released') is True, result
        assert result.get('return_required') is False
    assert not h.writes and not h.link.lease
    assert h.plant.snapshot()['q'] == before
    assert not any(r['action'] in ('claim', 'resume', 'prepare_operator_session') for r in h.requests)
    assert pico(h, 'start')['error'] == 'project_not_armed'


@pytest.mark.parametrize('origin', ['pico', 'canvas', 'canvas_disconnected'])
def test_follow_release_regrip_and_end_waits_for_actual_neutral(project_chain, origin):
    h = project_chain
    start_follow(h)
    old_session = h.link.lease['session_id']
    send(h, False)
    await_condition(lambda: h.gate.status()['hold_confirmed'])
    await_condition(lambda: h.card.runtime.status()['dispatch']['stop_acknowledged'])
    assert h.gate.status()['hold_confirmed'] and h.link.lease
    deadline = time.monotonic() + 1.
    while h.link.lease['session_id'] == old_session and time.monotonic() < deadline:
        send(h, True, clutch=2, forward=.08)
        time.sleep(.025)
    assert h.link.lease['session_id'] != old_session
    send(h, True, clutch=2, forward=.10)
    assert h.gate.status()['state'] in ('ready', 'active')
    delayed = frame(h.sequence, True, 2)
    delayed['mode'] = 'live'
    delayed = bind_rtc_frame_v1(delayed, authority=h.assignments[-1], expected_mode='live')
    if origin == 'canvas_disconnected':
        h.capture._connection = None
    if origin == 'pico':
        receipt = pico(h, 'finish', 'finish-once')
        assert receipt['state'] == 'completed', receipt
        result = receipt['result']
        assert h.card.info()['project']['armed'] is True
    else:
        result = h.card.dispatch('teleop', {'action': 'project_stop'})
        assert result.get('armed') is False, result
    assert_released(h, result)
    writes = len(h.writes)
    claims = sum(r['action'] in ('claim', 'resume') for r in h.requests)
    with pytest.raises(ProtocolError):
        h.card.runtime.submit_frame(delayed, source='synthetic')
    if origin == 'pico':
        assert pico(h, 'finish', 'finish-once') == receipt
    else:
        assert h.card.dispatch('teleop', {'action': 'project_stop'}).get('authority_released') is True
    time.sleep(.04)
    assert len(h.writes) == writes
    assert sum(r['action'] in ('claim', 'resume') for r in h.requests) == claims


def test_failed_return_is_not_success_and_same_project_stop_can_retry(project_chain):
    h = project_chain
    start_follow(h)
    h.geometry_failure[0] = 'synthetic_return_collision'
    failed = h.card.dispatch('teleop', {'action': 'project_stop'})
    assert failed.get('error') == 'synthetic_return_collision', failed
    assert failed.get('authority_released') is not True and failed.get('return_completed') is not True
    assert not h.card.info()['project']['armed']
    assert h.link.lease and h.gate.status()['hold_confirmed']
    assert h.card.dispatch('teleop', {'action': 'project_start', 'driver_binding': h.binding}).get('error')
    h.geometry_failure[0] = None
    result = h.card.dispatch('teleop', {'action': 'project_stop'})
    assert_released(h, result)
    assert result['armed'] is False and h.card.info()['project']['error'] is None


def test_immediate_stop_then_explicit_finish_prepares_a_new_return_session(project_chain):
    h = project_chain
    start_follow(h)
    stopped = pico(h, 'stop')
    assert stopped['state'] == 'completed', stopped
    assert h.link.lease is None and h.driver._operator_prepared is False
    assert np.max(np.abs(h.plant.snapshot()['q'])) > .02
    finished = pico(h, 'finish')
    assert finished['state'] == 'completed', finished
    assert_released(h, finished['result'])
    assert h.driver._operator_prepared is False


def test_jammed_plant_times_out_without_success_and_explicit_retry_can_finish(project_chain, monkeypatch):
    from teleop import operator_session
    h = project_chain
    start_follow(h)
    h.plant.failure = 'jammed'
    real_return = operator_session.return_arms
    short_trial = [True]

    def bounded_return(*args, **kwargs):
        if short_trial[0]:
            kwargs['timeout'] = .5
        return real_return(*args, **kwargs)

    monkeypatch.setattr(operator_session, 'return_arms', bounded_return)
    failed = h.card.dispatch('teleop', {'action': 'project_stop'})
    assert failed.get('error') == 'return_timeout', failed
    assert failed.get('return_completed') is not True and failed.get('authority_released') is not True
    assert h.gate.status()['hold_confirmed'] and h.link.lease
    assert np.max(np.abs(h.plant.snapshot()['q'])) > .02
    h.plant.failure = None
    short_trial[0] = False
    result = h.card.dispatch('teleop', {'action': 'project_stop'})
    assert_released(h, result)


def test_frozen_feedback_never_completes_return_or_releases_client_authority(project_chain):
    h = project_chain
    start_follow(h)
    h.plant.failure = 'feedback_frozen'
    time.sleep(.12)
    failed = h.card.dispatch('teleop', {'action': 'project_stop'})
    assert failed.get('error'), failed
    assert failed.get('return_completed') is not True and failed.get('authority_released') is not True
    assert not h.card.info()['project']['armed']
    assert h.link.lease is not None


def test_release_reply_loss_uses_fresh_feedback_instead_of_duplicate_return(project_chain):
    h = project_chain
    start_follow(h)
    h.lost_replies.append('release')
    result = h.card.dispatch('teleop', {'action': 'project_stop'})
    assert_released(h, result)
    assert not h.lost_replies
    assert sum(r['action'] == 'release' for r in h.requests) >= 1


def test_old_release_receipt_does_not_claim_success_and_fresh_receipt_allows_retry(project_chain, monkeypatch):
    h = project_chain
    start_follow(h)
    feedback = h.link.feedback
    stale = [True]

    def delayed_release_receipt():
        result = feedback()
        if stale[0] and h.link.release_requested_ns and result['ownership_held'] is False:
            # Hardware has stopped, but the client has only a pre-request
            # receipt. A current arm sample cannot validate that old receipt.
            result['monotonic_ns'] = h.link.release_requested_ns - 1
        return result

    monkeypatch.setattr(h.link, 'feedback', delayed_release_receipt)
    failed = h.card.dispatch('teleop', {'action': 'project_stop'})
    assert failed.get('error'), failed
    assert failed.get('authority_released') is not True
    assert failed.get('return_completed') is not True
    assert h.link.lease is not None
    assert h.gate.status()['ownership_held'] is False
    stale[0] = False
    result = h.card.dispatch('teleop', {'action': 'project_stop'})
    assert_released(h, result)


@pytest.mark.parametrize('fail_return', [False, True])
def test_real_core_project_orchestration_calls_real_plugin_and_driver(project_chain, monkeypatch, tmp_path, fail_return):
    """Optional Core dependencies; the MCP transport only routes real calls."""
    pytest.importorskip('fastapi')
    jsonschema = pytest.importorskip('jsonschema')
    h = project_chain
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / 'agent-core' / 'src'))
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'core.db'))
    monkeypatch.chdir(tmp_path)  # Never load a checkout's private .env.
    import config as core_config
    from api import config as core_api
    # Keep the real project orchestration while replacing only persistent
    # storage. DBConfig does not implement __delitem__ for monkeypatch undo.
    monkeypatch.setattr(core_config, 'main', {})
    calls, events = [], []
    driver_tool = h.driver.get_tool()
    source_tool = h.card.get_tools()[0]

    async def call(mcp_id, request, timeout_s=None):
        args = dict(request.arguments)
        calls.append((request.tool, args['action']))
        if request.tool == 'teleop':
            assert mcp_id == 'actucore'
            jsonschema.validate(args, source_tool['inputSchema'])
            value = await asyncio.to_thread(h.card.dispatch, 'teleop', args)
        else:
            assert request.tool == 'teleop_executor' and mcp_id == h.binding['mcp_id']
            jsonschema.validate(args, driver_tool['inputSchema'])
            if args['action'] == 'stop':
                assert h.link.lease is None and not h.card.info()['project']['armed']
                h.plant.assert_settled_receipt()
            value = await asyncio.to_thread(h.driver.dispatch, args['action'], args)
        return {'code': 400 if value.get('error') else 200, 'data': value}

    async def push(event):
        events.append(event)

    async def register(*args):
        pass

    for name, attrs in {
        'api.mcp_manage': {'mcp_call_tool': call, 'MCPCallRequest': SimpleNamespace},
        'api.motus_stream': {'push_event': push},
        'api.inspection': {'register_topic_internal': register},
        'channel.manager': {'manager': SimpleNamespace(sync_from_canvas=lambda: None, _adapters={}),
                            '_get_channel_configs': lambda: []},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(core_api, '_start_project_task', None)
    monkeypatch.setattr(core_api, '_stop_project_task', None)
    monkeypatch.setitem(core_config.main, 'core', {'project_running': False})
    monkeypatch.setitem(core_config.main, 'services', {'mcp': [
        {'id': 'actucore', 'url': 'http://127.0.0.1:15730/mcp', 'tools': [source_tool]},
        {'id': h.binding['mcp_id'], 'url': h.link.url, 'tools': [driver_tool]},
    ]})
    monkeypatch.setitem(core_config.main, 'canvas_layout', {
        'cards': [{'id': 'driver', 'mcpId': h.binding['mcp_id'], 'toolName': 'teleop_executor'},
                  {'id': 'pendant', 'mcpId': 'actucore', 'toolName': 'teleop'}],
        'connections': [{'fromCardId': 'pendant', 'toCardId': 'driver',
                         'fromPortIdx': '0', 'toPortIdx': '0', 'format': 'control/teleop'}],
    })

    async def run():
        assert await core_api._do_start_project_impl() is True
        assert h.card.info()['project']['armed'] and not h.link.lease and not h.writes
        await asyncio.to_thread(start_follow, h)
        h.capture._connection = None
        if fail_return:
            h.geometry_failure[0] = 'synthetic_return_collision'
            assert await core_api._do_stop_project() is False
            assert core_config.main['core']['project_phase'] == 'stop_failed'
            assert ('teleop_executor', 'stop') not in calls
            assert h.link.lease is not None
            h.geometry_failure[0] = None
        assert await core_api._do_stop_project() is True
        assert core_config.main['core']['project_running'] is False
        assert core_config.main['core']['project_phase'] == 'idle'
        assert h.link.lease is None and h.gate.status()['ownership_held'] is False
        assert calls.index(('teleop', 'project_stop')) < calls.index(('teleop_executor', 'stop'))
        h.plant.assert_settled_receipt()
    asyncio.run(run())
