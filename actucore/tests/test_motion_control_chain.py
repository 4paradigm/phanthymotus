"""Cross-repository EEF -> actual Driver IK -> arm admission -> finite plant.

MCP HTTP and both DDS edges are explicit in-memory substitutes. No ROS node,
network, vendor device or hardware action is allocated. Synthetic URDF/plant
exercise real numerical IK, signing, admission and execution code; they are not
physical calibration or robot acceptance evidence. Set TIANYI_DRIVER_SOURCE to
the corresponding Driver checkout, as for the existing execution-chain tests.
"""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import threading
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1]/'plugins'))
from teleop.dispatch import MotionIntent, StopRequest
from teleop.motion_control import EefIntentAdapter, MotionControlLink
from teleop.plugin import TeleopPlugin
from test_tianyi_execution_chain import tianyi_driver_source


def wait_for(predicate, timeout=2.):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    assert predicate()


class InMemoryTransport:
    """Replace allocation/I/O only; keep MotionControlLink.call/send/feedback real."""
    def __init__(self, driver):
        self.driver = driver
        self.eef_packets, self.management = [], []
        self.errors = []
        self.closed = threading.Event()
        link = self.link = MotionControlLink.__new__(MotionControlLink)
        link.url = 'http://127.0.0.1:1/mcp'
        link.opener = self
        link.topic = '/offline/motion/teleop'
        link.command_topic = '/offline/motion/control/command'
        link.node = SimpleNamespace(create_publisher=self.publisher)
        link.qos = None
        link.publisher = None
        link.condition = threading.Condition()
        link.latest = None
        link.lease = link.preview_lease = None
        link.seq = 0
        link.pending = deque()
        link.lease_started_ns = link.release_requested_ns = link.execution_progress_ns = 0
        link.last_send = link.last_feedback_received_ns = link.last_call = None
        link.transport_prepare_ms = None
        link.management_retry = False
        link.management_request = None
        self.feedback()
        self.thread = threading.Thread(target=self._feedback_loop, daemon=True)
        self.thread.start()

    def publisher(self, message_type, topic, qos):
        assert topic == self.driver.c.topic+'/command'
        def publish(message):
            packet = json.loads(message.data)
            self.driver.c.receive_eef(packet)
            self.eef_packets.append(copy.deepcopy(packet))
        return SimpleNamespace(publish=publish)

    def feedback(self):
        self.driver.c._refresh_snapshot()
        self.link._feedback(SimpleNamespace(data=json.dumps(self.driver.c.info())))

    def _feedback_loop(self):
        while not self.closed.wait(.01):
            try:
                self.feedback()
            except Exception as exc:
                self.errors.append(exc)
                return

    def open(self, request, timeout):
        assert timeout > 0 and request.full_url == self.link.url
        message = json.loads(request.data)
        assert message['method'] == 'tools/call'
        assert message['params']['name'] == 'motion_control'
        args = message['params']['arguments']
        self.management.append(copy.deepcopy(args))
        value = self.driver.c.dispatch(args['action'], args)
        self.feedback()
        return io.BytesIO(json.dumps({'result': {'isError':bool(value.get('error')),
            'content':[{'type':'text','text':json.dumps(value)}]}}).encode())

    def close(self):
        self.closed.set()
        self.thread.join(1)
        assert not self.thread.is_alive()
        assert not self.errors


@pytest.fixture
def chain(tmp_path, monkeypatch):
    root = tianyi_driver_source()
    source = root/'tests/test_motion_control.py'
    if not source.is_file():
        pytest.fail('Matching Driver motion-control numerical fixture is required')
    spec = importlib.util.spec_from_file_location('_driver_motion_chain_fixture', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = module.chain.__wrapped__(tmp_path, monkeypatch)
    driver = next(fixture)
    # Both levels use the real std_msgs payload shape but create no ROS entities.
    monkeypatch.setitem(sys.modules, 'std_msgs', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'std_msgs.msg', SimpleNamespace(String=type('String', (), {})))
    vendor_calls = []
    send = driver.e.arm._send_pos
    def measured_separately(poses, speed):
        with driver.p.lock:
            before = driver.p.q.copy()
            result = send(poses, speed)
            np.testing.assert_array_equal(driver.p.q, before)
            vendor_calls.append(copy.deepcopy(poses))
            return result
    monkeypatch.setattr(driver.e.arm, '_send_pos', measured_separately)
    driver.p.run(driver.e.gate)
    transport = InMemoryTransport(driver)
    value = SimpleNamespace(**vars(driver), wire=transport, link=transport.link, vendor_calls=vendor_calls)
    try:
        yield value
    finally:
        transport.close()
        fixture.close()


def frame():
    pose = {'position':[0., 1., 0.], 'orientation':[0., 0., 0., 1.]}
    return {'head':copy.deepcopy(pose), 'left_controller':copy.deepcopy(pose),
        'right_controller':copy.deepcopy(pose),
        'controllers':{s:{'buttons':[0., 1.]} for s in ('left','right')}}


def apply(chain, adapter, sequence, clutch=1, value=None, lifetime=.3, generation=1):
    chain.wire.feedback()
    now = time.monotonic()
    intent = MotionIntent(1, generation, sequence, clutch, now, now+lifetime,
        value or frame(), received_monotonic=now)
    ack = adapter.apply(intent)
    assert ack.ok, adapter.output
    return intent


def prepare(chain, mode):
    adapter = EefIntentAdapter(chain.link, mode)
    assert adapter.calibrate()['calibrated']
    apply(chain, adapter, 1)
    assert not chain.wire.eef_packets
    return adapter


def target_frame(adapter, palms):
    """Synthesize controller input for reachable FK targets, not fake IK output."""
    value = frame()
    for side, (controller, baseline), target in zip(('left','right'), adapter.mapper.reference, palms):
        pose = value[side+'_controller']
        pose['position'] = (controller[:3,3] + adapter.mapper.rotation.T @
            (target[:3,3]-baseline[:3,3])/adapter.mapper.scale).tolist()
        pose['orientation'] = Rotation.from_matrix(adapter.mapper.rotation.T @ target[:3,:3] @
            baseline[:3,:3].T @ adapter.mapper.rotation @ controller[:3,:3]).as_quat().tolist()
    return value


def motion_frame(chain, adapter):
    q = chain.p.q.copy()
    q[3] += .03
    q[10] -= .025
    return target_frame(adapter, chain.c.solver.palms(q))


def test_shadow_real_ik_without_claim_joint_or_vendor_calls(chain):
    adapter = prepare(chain, 'shadow')
    sent = apply(chain, adapter, 2, value=motion_frame(chain, adapter))
    assert chain.c.process_latest()
    chain.wire.feedback()
    packet = chain.wire.eef_packets[-1]
    assert packet['schema'] == 'motus.control/2' and packet['mode'] == 'eef_pose'
    assert packet['source_seq'] == sent.sequence and packet['mapping_epoch'] == 1
    assert chain.c.info()['control_decision']['state'] == 'preview'
    assert chain.c.solver.last_ms > 0
    assert adapter.solver is None  # Numerical IK is exclusively in the Driver.
    assert not chain.link.lease and not chain.e.gate.session_id
    assert 'claim' not in [r['action'] for r in chain.wire.management]
    assert not chain.commands and not chain.vendor_calls and not chain.p.writes


def test_live_eef_joint_vendor_and_measured_regrip_baseline(chain):
    adapter = prepare(chain, 'live')
    original_lease = dict(chain.link.lease)
    measured_before = chain.p.q.copy()
    sent = apply(chain, adapter, 2, value=motion_frame(chain, adapter))
    assert chain.c.process_latest()
    joint = chain.commands[-1]
    assert joint['mode'] == 'joint_position' and len(joint['values']) == 14
    assert joint['source_seq'] == sent.sequence and joint['mapping_epoch'] == 1
    assert joint['valid_until_ns'] <= chain.wire.eef_packets[-1]['valid_until_ns']
    assert 0 < joint['valid_until_ns']-joint['generated_ns'] <= 100_000_000
    wait_for(lambda: bool(chain.vendor_calls))
    first = np.deg2rad(chain.vendor_calls[0]['left']+chain.vendor_calls[0]['right'])
    # The first tick may occur immediately after claim. The real gate slews
    # toward IK at 1 rad/s; it must not jump directly to the final target.
    assert np.max(np.abs(first-measured_before)) <= chain.e.gate.velocity*.02+1e-12
    assert np.all(np.abs(first-joint['values']) <= np.abs(measured_before-joint['values'])+1e-12)
    wait_for(lambda: np.allclose(np.deg2rad(
        chain.vendor_calls[-1]['left']+chain.vendor_calls[-1]['right']), joint['values']))
    wait_for(lambda: np.max(np.abs(chain.p.q)) > 0)
    assert np.max(np.abs(chain.p.dq)) <= 1.
    assert chain.link.pause(time.monotonic()+1.)
    assert chain.e.gate.status()['hold_confirmed']
    # A relocated controller begins a new clutch at measured FK, with no jump.
    neutral = frame()
    for side in ('left', 'right'):
        neutral[side+'_controller']['position'] = [.3, 1.3, -.2]
    apply(chain, adapter, 3, clutch=2, value=neutral)
    baseline = copy.deepcopy(adapter.mapping_snapshot['poses'])
    apply(chain, adapter, 4, clutch=2, value=neutral)
    packet = chain.wire.eef_packets[-1]
    assert packet['mapping_epoch'] == 2
    np.testing.assert_allclose(packet['values'], np.array(baseline).reshape(-1), atol=1e-9)
    assert chain.link.lease['session_id'] != original_lease['session_id']
    assert chain.c.process_latest()
    assert chain.commands[-1]['mapping_epoch'] == 2


def test_short_ik_failure_continues_same_session_only_with_new_frame(chain, monkeypatch):
    adapter = prepare(chain, 'live')
    identity = chain.link.lease['session_id']
    solve = chain.c.solver.solve
    def failed(*args, **kwargs):
        raise ValueError('ik_target_unreachable')
    monkeypatch.setattr(chain.c.solver, 'solve', failed)
    apply(chain, adapter, 2)
    assert not chain.c.process_latest()
    assert not chain.commands
    wait_for(lambda: chain.e.gate.status().get('continuation_ready'))
    monkeypatch.setattr(chain.c.solver, 'solve', solve)
    # No background retry of the failed target when the solver becomes available.
    assert not chain.c.process_latest()
    assert not chain.commands
    apply(chain, adapter, 3, value=motion_frame(chain, adapter))
    assert chain.c.process_latest()
    wait_for(lambda: chain.e.gate.state == 'active')
    assert chain.e.gate.session_id == chain.link.lease['session_id'] == identity
    assert chain.commands[-1]['source_seq'] == 3
    assert not any(r['action']=='resume' for r in chain.wire.management)


def test_solution_completed_after_original_deadline_is_never_sent_to_arm(chain, monkeypatch):
    adapter = prepare(chain, 'live')
    solve = chain.c.solver.solve
    def late(*args, **kwargs):
        solution = solve(*args, **kwargs)
        # Deliberate fault injection AFTER genuine IK; do not relax the input TTL.
        time.sleep(max(0., kwargs['deadline_monotonic']-time.monotonic())+.005)
        return solution
    monkeypatch.setattr(chain.c.solver, 'solve', late)
    sent = apply(chain, adapter, 2)
    assert not chain.c.process_latest()
    assert chain.wire.eef_packets[-1]['valid_until_ns'] <= int(sent.expires_monotonic*1e9)
    assert chain.c.info()['control_decision']['reason'] == 'command_expired'
    assert not chain.commands  # Holds may publish measured positions; no expired IK target.
    assert not chain.c.process_latest()


@pytest.mark.parametrize('mode', ['shadow', 'live'])
def test_feedback_loss_recovers_without_regrip_or_mapping_jump(chain, monkeypatch, mode):
    adapter = prepare(chain, mode)
    apply(chain, adapter, 2, value=motion_frame(chain, adapter))
    assert chain.c.process_latest()
    old_packet = copy.deepcopy(chain.wire.eef_packets[-1])
    old_identity = (chain.link.lease or chain.link.preview_lease)['session_id']
    baseline = copy.deepcopy(adapter.mapping_snapshot)
    reference = copy.deepcopy(adapter.mapper.reference)
    epoch = adapter.mapping_epoch
    def missing():raise ValueError('driver_feedback_missing')
    with monkeypatch.context() as patch:
        patch.setattr(chain.link, 'feedback', missing)
        now = time.monotonic()
        ack = adapter.apply(MotionIntent(1, 1, 3, 1, now, now+.3, frame(), received_monotonic=now))
    assert not ack.ok and ack.code == 'feedback_unavailable'
    now = time.monotonic()
    assert adapter.safe_stop(StopRequest(2, 'feedback_unavailable', now, now+1.)).ok
    assert adapter._automatic_resume
    before = len(chain.wire.eef_packets)
    # Same held grips: management changes identity without resetting mapping.
    apply(chain, adapter, 4, generation=2)
    assert len(chain.wire.eef_packets) == before
    assert not adapter._automatic_resume
    assert (chain.link.lease or chain.link.preview_lease)['session_id'] != old_identity
    assert adapter.mapping_epoch == epoch and adapter.mapping_snapshot == baseline
    for actual, expected in zip(adapter.mapper.reference, reference):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError):chain.c.receive_eef(old_packet)
    apply(chain, adapter, 5, generation=2, value=motion_frame(chain, adapter))
    assert chain.c.process_latest()
    assert chain.c.info()['control_decision']['source_seq'] == 5
    if mode == 'shadow':
        assert not chain.commands and not chain.vendor_calls and not chain.link.lease
    else:
        wait_for(lambda: chain.e.gate.state == 'active')
        assert chain.commands[-1]['source_seq'] == 5


def test_shadow_finish_retires_preview_identity_without_hardware_stop(chain):
    adapter = prepare(chain, 'shadow')
    token = dict(chain.link.preview_lease)
    plugin = TeleopPlugin({'control_backend':'motion_control', 'mode':'shadow'}, None)
    plugin.link, plugin.adapter = chain.link, adapter
    plugin._return_required = True
    result = plugin._finish_session(threading.Event(), lambda: True)
    assert result['authority_released'] and result['return_completed']
    assert chain.link.preview_lease is None and chain.c._preview is None
    release = next(r for r in chain.wire.management if r['action'] == 'release')
    assert release['session_id'] == token['session_id'] and release['secret'] == token['secret']
    assert chain.link.feedback()['preview'] is False
    assert not chain.e.gate.session_id and not chain.commands and not chain.vendor_calls


def test_shadow_explicit_recovery_requires_fresh_measurements_not_power_or_stop_receipt(chain, monkeypatch):
    adapter = prepare(chain, 'shadow')
    plugin = TeleopPlugin({'control_backend':'motion_control', 'mode':'shadow'}, None)
    plugin.link, plugin.adapter = chain.link, adapter
    plugin.runtime = SimpleNamespace(status=lambda: {'dispatch':{}}, release_local=lambda: None)
    monkeypatch.setattr(plugin, 'stop', lambda: None)
    monkeypatch.setattr(plugin, '_ensure_host', lambda: None)
    feedback = chain.link.feedback
    def without_hardware_acceptance():
        state = copy.deepcopy(feedback())
        state['stop_confirmed'] = False
        state['feedback'].update(power_ns=0, fixed_ns=0, power_on=False)
        return state
    monkeypatch.setattr(chain.link, 'feedback', without_hardware_acceptance)
    previous = chain.link.preview_lease['session_id']
    plugin._recover_tianyi_host()
    assert chain.link.preview_lease['session_id'] != previous
    assert not chain.link.lease and not chain.commands and not chain.vendor_calls


def test_true_driver_fault_never_consumes_automatic_resume_or_publishes(chain):
    adapter = prepare(chain, 'live')
    # A physical fault arriving after a recoverable stop must win over retries.
    adapter._automatic_resume = True
    with chain.e.gate.lock:
        chain.e.gate.state = 'fault'
        chain.e.gate.reason = 'estop'
    chain.wire.feedback()
    before = len(chain.wire.management)
    now = time.monotonic()
    ack = adapter.apply(MotionIntent(1, 1, 2, 1, now, now+.3, frame(), received_monotonic=now))
    assert not ack.ok and adapter.output['code'] == 'driver_fault'
    assert len(chain.wire.management) == before
    assert not chain.wire.eef_packets and not chain.commands
