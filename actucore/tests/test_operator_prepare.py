"""Operator start through real RPC encoding and Driver admission, without I/O.

Only ROS/network allocation, calibration and the physical sensor producers are
substituted. Driver feedback, fixed-body capture, freshness and safety checks
remain real; power and joint feedback deliberately have independent clocks.
"""
import asyncio
import io
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'plugins'))
from teleop.adapter import DriverLink
from teleop.plugin import TeleopPlugin
from test_tianyi_execution_chain import tianyi_driver_source


@pytest.fixture
def preparation(monkeypatch):
    monkeypatch.syspath_prepend(str(tianyi_driver_source()))
    from motion_control import MotionControl
    from teleop_executor import MOTOR_IDS, TeleopExecutor

    monkeypatch.setitem(sys.modules, 'device', SimpleNamespace(
        _HEAD_JOINTS={1: 'head'}, _WAIST_JOINTS={31: 'waist'}, _LEG_JOINTS={51: 'leg'}))
    driver = TeleopExecutor({'live_enabled': True, 'operator_session_enabled': True},
                           'offline', None, None, None, [])
    driver.profile = {'hands_enabled': False, 'first_acceptance': {
        **{k: True for k in ('model_verified', 'workspace_verified', 'pico_verified',
                            'external_control_excluded')},
        'operator': 'offline fixture', 'date': 'offline', 'evidence_sha256': 'a'*64}}
    driver.gate.hands_enabled = False
    monkeypatch.setattr(driver, 'start', lambda: None)
    monkeypatch.setattr(driver, 'foreign_publishers', lambda: [])
    controller = MotionControl({}, driver)
    controller.solver = object()  # The previously calibrated solver is not used by prepare.
    monkeypatch.setattr(controller, 'start', lambda: None)
    writes, calls, results, assignments, deadlines = [], [], [], [], []
    driver.gate.emit = lambda *args: writes.append(args)
    conditions = {'recover_after': 2, 'power_on': True, 'estop': False, 'fault': False,
                  'connected': True, 'after_attempt': lambda: None}
    # Power is stale initially while actual joint callbacks continue below.
    driver._power = (time.monotonic_ns()-150_000_000, True, False)

    class Wire:
        def open(self, request, timeout):
            assert 0 < timeout <= .08
            message = json.loads(request.data)
            assert message['params']['name'] == 'motion_control'
            args = message['params']['arguments']
            calls.append(args['action'])
            if args['action'] == 'prepare_operator_session':
                for part, ids in [('arm', MOTOR_IDS), ('head', [1]), ('waist', [31]), ('leg', [51])]:
                    driver._motors(part, SimpleNamespace(status=[SimpleNamespace(
                        name=mid, pos=0., speed=0., error=int(conditions['fault'])) for mid in ids]))
                if calls.count('prepare_operator_session') > conditions['recover_after']:
                    driver._power_cb(SimpleNamespace(is_power_on=SimpleNamespace(data=conditions['power_on']),
                        is_estop=SimpleNamespace(data=conditions['estop']),
                        is_remote_estop=SimpleNamespace(data=False)))
            value = controller.dispatch(args['action'], args)
            results.append(value)
            conditions['after_attempt']()
            return io.BytesIO(json.dumps({'result': {'isError': bool(value.get('error')),
                'content': [{'type': 'text', 'text': json.dumps(value)}]}}).encode())

    link = DriverLink.__new__(DriverLink)
    link.url, link.tool, link.opener = 'http://127.0.0.1:1/mcp', 'motion_control', Wire()
    link.lease, link.management_request = None, None

    def call(action, deadline):
        deadlines.append(deadline)
        return DriverLink.call(link, action, deadline)

    link.call = call
    card = TeleopPlugin({'robot_profile': 'tianyi2', 'mode': 'live',
                        'operator_session_enabled': True}, None)
    card.link, card.adapter = link, SimpleNamespace(hardware_output=True)
    card._project_armed = True
    card.runtime = SimpleNamespace(status=lambda: {'authority_valid': False, 'dispatch': {}},
        prepare_local_session=lambda: assignments.append('runtime'), release_local=lambda: None)

    async def assign():
        assignments.append('capture')

    async def revoke(reason):
        pass

    card.capture = SimpleNamespace(issue_assignment_if_connected=assign, revoke_assignment=revoke)
    monkeypatch.setattr(card, '_run', asyncio.run)
    monkeypatch.setattr(card, '_calibrate_recovered_host', lambda: None)
    monkeypatch.setattr(card, '_headset_connected', lambda: conditions['connected'])
    monkeypatch.setattr(card, 'info', lambda: {'state': 'ready', 'mode': 'live'})
    yield SimpleNamespace(card=card, driver=driver, calls=calls, results=results,
                          conditions=conditions, assignments=assignments, writes=writes,
                          deadlines=deadlines)
    assert not writes and driver.gate.session_id is None
    assert 'claim' not in calls and 'resume' not in calls


def test_operator_prepare_retries_stale_power_then_uses_real_fresh_sample(preparation):
    h = preparation
    result = h.card.dispatch('teleop', {'action': 'start'})
    assert result == {'state': 'ready', 'mode': 'live'}
    assert [r.get('code') for r in h.results] == ['power_ns_stale', 'power_ns_stale', None]
    assert h.assignments == ['runtime', 'capture']
    assert h.card._operator_session_prepared and h.driver._operator_prepared
    assert h.driver._fixed_baseline == {'1': 0., '31': 0., '51': 0.}


def test_operator_prepare_persistent_stale_preserves_reason_and_bounded_deadline(preparation):
    h = preparation
    h.conditions['recover_after'] = 10000
    power_stamp = h.driver._power[0]
    started = time.monotonic()
    result = h.card.dispatch('teleop', {'action': 'start'})
    elapsed = time.monotonic()-started
    assert result['code'] == 'power_ns_stale'
    assert .45 <= elapsed < .65  # Scheduling tolerance; production budget is 500 ms.
    assert len(h.calls) > 1 and not h.assignments
    assert len(set(h.deadlines)) == 1  # Repeated errors cannot extend the budget.
    assert 0 < h.deadlines[0]-started <= .51
    assert h.driver._power[0] == power_stamp
    assert not h.driver._operator_prepared and not h.card._operator_session_prepared
    # A later explicit retry is usable, without restarting either component.
    h.conditions['recover_after'] = 0
    assert h.card.dispatch('teleop', {'action': 'start'})['state'] == 'ready'


@pytest.mark.parametrize('condition', ['power_on', 'estop', 'fault', 'competitor'])
def test_operator_prepare_does_not_retry_real_safety_refusals(preparation, monkeypatch, condition):
    h = preparation
    h.conditions['recover_after'] = 0
    if condition == 'competitor':
        monkeypatch.setattr(h.driver, 'foreign_publishers', lambda: ['other'])
    else:
        h.conditions[condition] = condition != 'power_on'
    result = h.card.dispatch('teleop', {'action': 'start'})
    expected = 'external_motion_publishers_present' if condition == 'competitor' else 'robot_safety_not_ready'
    assert result['code'] == expected
    assert h.calls == ['prepare_operator_session'] and not h.assignments


def test_operator_prepare_disconnect_prevents_the_next_attempt(preparation):
    h = preparation
    h.conditions['after_attempt'] = lambda: h.conditions.update(connected=False)
    assert h.card.dispatch('teleop', {'action': 'start'})['code'] == 'operator_connection_changed'
    assert h.calls == ['prepare_operator_session'] and not h.assignments


@pytest.mark.parametrize('stop', ['stop', 'project_stop'])
def test_operator_prepare_wait_releases_status_lock_and_stop_cancels(preparation, monkeypatch, stop):
    h = preparation
    h.conditions['recover_after'] = 10000
    waiting = threading.Event()

    class Cancel(threading.Event):
        def wait(self, timeout=None):
            waiting.set()
            return super().wait(timeout)

    cancel = Cancel()
    output = {}
    monkeypatch.setattr(h.card, '_finish_session', lambda *a, **k: {
        'state': 'idle', 'return_completed': True, 'authority_released': True})

    def start():
        try:
            output['start'] = h.card._operator_execute('start', cancel, lambda: True)
        except Exception as exc:
            output['start'] = {'code': exc.code}

    worker = threading.Thread(target=start)
    worker.start()
    try:
        assert waiting.wait(1)
        # Status/config access is available during the retry wait.
        assert h.card._lock.acquire(timeout=.05)
        h.card._lock.release()
        stopped = time.monotonic()
        result = h.card.dispatch('teleop', {'action': stop})
        worker.join(.2)
        assert not worker.is_alive() and time.monotonic()-stopped < .2
        assert result.get('authority_released') is True
        assert output['start']['code'] == 'operator_start_cancelled'
        assert not h.assignments and not h.driver._operator_prepared
    finally:
        cancel.set()
        worker.join(1)
