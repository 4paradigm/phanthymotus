"""Actual adapter + authenticated DriverLink serialization + Driver MotionGate.

ROS publisher is replaced by an in-memory wire; plant feedback is synthetic.
Set TIANYI_DRIVER_SOURCE to the matching Driver's tianyi2.0 directory to run the
cross-repository cases. An unset variable skips those cases; invalid paths fail.
This test sends no network traffic and cannot access physical hardware.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'plugins'))
from teleop.adapter import DriverLink
from teleop.tianyi import TianyiIntentAdapter
from teleop.dispatch import MotionIntent, StopRequest


def tianyi_driver_source():
    value = os.environ.get('TIANYI_DRIVER_SOURCE')
    if value is None:
        pytest.skip('cross-repository Tianyi test requires TIANYI_DRIVER_SOURCE '
                    'pointing to the matching Driver x-humanoid/tianyi2.0 directory')
    if not value.strip():
        pytest.fail('TIANYI_DRIVER_SOURCE is set but empty', pytrace=False)
    root = Path(value).expanduser().resolve()
    required = ('motion_stream.py', 'teleop_executor.py')
    if not root.is_dir() or any(not (root / name).is_file() for name in required):
        pytest.fail('TIANYI_DRIVER_SOURCE must be a Driver directory containing '
                    f'{", ".join(required)}: {root}', pytrace=False)
    return root


@pytest.mark.parametrize('case',['temporary','persistent','fault'])
def test_physical_stop_confirmation_retries_only_transient_feedback(monkeypatch,case):
    now=[100.];calls=[]
    monkeypatch.setattr('teleop.tianyi.time.monotonic',lambda:now[0])
    monkeypatch.setattr('teleop.tianyi.time.sleep',lambda t:now.__setitem__(0,now[0]+t))
    def operation(deadline):
        calls.append(deadline)
        if case=='fault':raise ValueError('driver_fault')
        if case=='persistent' or len(calls)<4:raise ValueError('driver_feedback_stale_or_different_clock')
        return True
    if case=='fault':
        with pytest.raises(ValueError,match='driver_fault'):TianyiIntentAdapter._confirm_stop(operation,100.1)
    else:
        assert TianyiIntentAdapter._confirm_stop(operation,100.1)==(case=='temporary')
    assert all(d<=100.1 for d in calls) and now[0]<=100.111


@pytest.fixture
def chain(monkeypatch):
    root = tianyi_driver_source()
    spec = importlib.util.spec_from_file_location('tianyi_gate_under_test', root/'motion_stream.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, 'std_msgs.msg', SimpleNamespace(String=SimpleNamespace))
    q = [0.]*14
    writes = []
    def snapshot():
        result = {k: time.monotonic_ns() for k in ('arm_ns','power_ns','hand_ns','fixed_ns')}
        result.update(q=list(q), dq=[0.]*14, power_on=True, estop=False, fault=False, fixed_body=True)
        return result
    def emit(target, hands):
        writes.append((list(target), hands))
        # Finite lag: only half the commanded displacement is observed.
        q[:] = [a+.5*(b-a) for a,b in zip(q,target)]
    gate = module.MotionGate(snapshot, emit, [(-2,2)]*14,
                             live_enabled=True, acceptance_check=lambda: True)
    class Wire(DriverLink):
        def __init__(self):
            self.lease=None; self.seq=0; self.pending=deque()
            self.lease_started_ns=0; self.release_requested_ns=0; self.execution_progress_ns=0
            self.condition=threading.Condition()
            self.publisher=SimpleNamespace(publish=lambda msg: gate.accept(json.loads(msg.data)))
            self.calls=[]
        def feedback(self):
            return {**gate.status(), 'calibration_sha256':'fixture'}
        def call(self, action, deadline):
            self.calls.append(action)
            if action=='claim':return gate.claim()
            if action=='resume':return gate.resume()
            if action in ('pause','release','recoverable_hold'):
                gate.hold('ik_recoverable' if action=='recoverable_hold' else 'operator_pause',
                          release=action=='release', recoverable=action=='recoverable_hold')
                gate.tick()
                time.sleep(.001)
                gate.tick()
                return gate.status()
            raise AssertionError(action)
        def claim(self, deadline):
            if self.lease:return
            self.lease=self.call('claim',deadline)
            self.lease_started_ns=time.monotonic_ns()
            self.pending.clear();self.seq=0
    class Solver:
        profile={'version':'fixture'};profile_sha256='fixture';last_ms=0.
        def palms(self, q):return [np.eye(4),np.eye(4)]
        def solve(self, targets, q, **kwargs):
            # Replace IK only: the intended displacement still comes from the
            # actual OpenXR relative mapping rather than a prebuilt command.
            target=list(q)
            target[0]+=float(targets[0][0,3])
            target[7]+=float(targets[1][0,3])
            return target
    link=Wire();adapter=TianyiIntentAdapter(link,'live');adapter.solver=Solver()
    return adapter,link,gate,writes


def intent(seq=1, clutch=1, generation=1, forward=0.):
    pose={'position':[0.,1.,0.], 'orientation':[0.,0.,0.,1.]}
    frame={'head':copy.deepcopy(pose), 'left_controller':copy.deepcopy(pose),
           'right_controller':copy.deepcopy(pose),
           'controllers':{s:{'buttons':[0.,1.]} for s in ('left','right')}}
    frame['left_controller']['position'][2]-=forward
    frame['right_controller']['position'][2]-=forward/2
    now=time.monotonic()
    return MotionIntent(1,generation,seq,clutch,now,now+.1,frame,True,now)


def stop(adapter, reason='deadman_released', generation=2):
    now=time.monotonic()
    return adapter.safe_stop(StopRequest(generation,reason,now,now+.1))


def test_clutch_follow_hold_reclutch_and_release(chain):
    adapter,link,gate,writes=chain
    assert adapter.apply(intent()).ok
    assert not writes and gate.state=='ready'
    assert adapter.apply(intent(2,forward=.1)).ok
    gate.tick()
    assert writes[-1][0][0]>0 and writes[-1][0][7]>0
    assert stop(adapter).ok
    assert gate.status()['hold_confirmed'] and link.lease
    old_session=gate.session_id
    before=len(writes)
    assert adapter.apply(intent(3,clutch=2,generation=2,forward=.2)).ok
    assert gate.session_id!=old_session and len(writes)==before
    assert adapter.apply(intent(4,clutch=2,generation=2,forward=.3)).ok
    gate.tick()
    assert gate.state=='active'
    assert adapter.close().ok
    assert not gate.session_id and gate.status()['stop_confirmed']
    assert link.calls==['claim','pause','resume','release']


@pytest.mark.parametrize('reason',['arm_collision'])
def test_collision_recovery_keeps_relative_reference(chain,reason):
    adapter,link,gate,writes=chain
    assert adapter.apply(intent()).ok
    reference=copy.deepcopy(adapter.mapper.reference)
    assert stop(adapter,reason).ok
    held_writes=len(writes)
    # Runtime permits collision recovery with the same held clutch, unlike
    # tracking loss or an explicit release; no remapping to the offending pose.
    assert adapter.apply(intent(2,generation=2,forward=.05)).ok
    assert gate.state=='hold' and adapter.output['state']=='recovery_validated'
    assert adapter.apply(intent(3,generation=2,forward=.05)).ok
    assert gate.state=='ready' and link.calls[-1]=='resume'
    assert len(writes)==held_writes  # pause writes hold; rearm adds no motion
    for (a,b),(c,d) in zip(adapter.mapper.reference, reference):
        np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)


def test_shadow_never_claims_or_sends(chain):
    adapter,link,gate,writes=chain
    adapter.hardware_output=False
    assert adapter.apply(intent()).ok
    assert adapter.apply(intent(2,forward=.1)).ok
    assert adapter.output['state']=='would_apply'
    assert not link.calls and not writes and not gate.session_id


def test_real_tianyi_ik_reaches_the_driver_wire(chain, tmp_path):
    from test_teleop import synthetic_profile
    from teleop.kinematics import TianyiIK
    adapter, link, gate, writes = chain
    path = synthetic_profile(tmp_path)
    adapter.solver = TianyiIK(path)
    adapter.solver.self_test([0.]*14)  # Numerical warmup is not a motion frame.
    original_feedback = link.feedback
    link.feedback = lambda: {**original_feedback(),
                             'calibration_sha256':adapter.solver.profile_sha256}
    assert adapter.apply(intent()).ok
    # This synthetic arm is fully extended along X at zero: further forward
    # translation is unreachable. Upward translation exercises a movable axis.
    moved = intent(2)
    moved.frame['left_controller']['position'][1] += .003
    moved.frame['right_controller']['position'][1] += .003
    assert adapter.apply(moved).ok
    gate.tick()
    assert gate.state == 'active' and gate.applied_seq == 1
    assert len(writes[-1][0]) == 14
    assert max(abs(x) for x in writes[-1][0]) > 1e-6


def test_arm_only_ignores_trigger_input(chain):
    adapter, link, gate, writes = chain
    adapter.solver.hands_enabled = False
    for seq in (1, 2):
        frame = intent(seq, forward=.01 * seq)
        for side in ('left', 'right'):
            frame.frame['controllers'][side]['buttons'][0] = 1.
        assert adapter.apply(frame).ok
    assert adapter.output['hands_enabled'] is False
    assert adapter.output['hands'] == [0., 0.]


def test_arm_only_profile_needs_no_hand_endpoints(tmp_path):
    root = tianyi_driver_source()
    sys.path.insert(0, str(root))
    try:
        from teleop_executor import load_profile
        from tianyi_fixture import synthetic_profile
        path = synthetic_profile(tmp_path)
        profile = json.loads(path.read_text())
        profile['hands_enabled'] = False
        path.write_text(json.dumps(profile))
        assert load_profile(path)[0]['hands_enabled'] is False
        profile['hands_enabled'] = 'false'
        path.write_text(json.dumps(profile))
        with pytest.raises(ValueError, match='hands_enabled_boolean_required'):
            load_profile(path)
    finally:
        sys.path.remove(str(root))


def test_expired_wire_packet_holds_then_reclutches_without_replaying(chain):
    adapter, link, gate, writes = chain
    assert adapter.apply(intent()).ok
    assert adapter.apply(intent(2, forward=.02)).ok
    gate.tick()
    old_session = gate.session_id
    old_packet = dict(gate.latest)
    # Valid authenticated packet, delayed in transit. It must not execute.
    import hmac, hashlib
    body = {**old_packet, 'seq': old_packet['seq'] + 1,
            'generated_ns': time.monotonic_ns() - 200_000_000}
    packet = {**body, 'mac': hmac.new(bytes.fromhex(gate.secret),
              json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode(),
              hashlib.sha256).hexdigest()}
    before = len(writes)
    with pytest.raises(ValueError, match='command_expired'):
        gate.accept(packet)
    assert len(writes) == before and gate.latest is None
    waiting = adapter.apply(intent(3, forward=.03))
    assert waiting.ok and waiting.code == 'waiting_driver_hold'
    assert stop(adapter, 'command_timeout').ok
    assert gate.status()['hold_confirmed']
    before = len(writes)
    assert adapter.apply(intent(4, clutch=2, generation=2)).ok
    assert gate.session_id != old_session and len(writes) == before
    assert gate.latest is None and gate.applied_seq == -1
    for seq in range(5, 25):
        assert adapter.apply(intent(seq, clutch=2, generation=2, forward=.02)).ok
        time.sleep(.002)
        gate.tick()
    assert gate.state == 'active' and gate.applied_seq == 20
    assert writes[-1][0][0] > writes[0][0][0]
    assert stop(adapter, generation=3).ok
    assert adapter.close().ok


@pytest.mark.parametrize('code',['driver_feedback_missing','driver_feedback_stale_or_different_clock','arm_feedback_stale'])
def test_transient_feedback_is_hold_not_generic_fault(chain,code):
    adapter,link,gate,writes=chain
    if code=='arm_feedback_stale':
        original=link.feedback
        def feedback():
            value=original();value['feedback']['arm_ns']-=200_000_000;return value
        link.feedback=feedback
    else:
        link.feedback=lambda:(_ for _ in ()).throw(ValueError(code))
    ack=adapter.apply(intent())
    assert not ack.ok and ack.code=='feedback_unavailable'
    assert not writes and not gate.session_id


def test_recorded_slow_solve_continues_through_authenticated_wire(chain, monkeypatch):
    adapter, link, gate, writes = chain
    clock = [100.]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(time, 'monotonic_ns', lambda: int(clock[0]*1e9))
    gate.clock = time.monotonic_ns
    assert adapter.apply(intent()).ok
    session = gate.session_id
    reference = copy.deepcopy(adapter.mapper.reference)
    solve = adapter.solver.solve
    def first_solve(*args, **kwargs):
        clock[0] += .058
        return solve(*args, **kwargs)
    adapter.solver.solve = first_solve
    assert adapter.apply(intent(2, forward=.02)).ok
    gate.tick()
    assert gate.applied_seq == 1
    def next_solve(*args, **kwargs):
        clock[0] += .043;gate.tick()  # Previous target expired; hold it now.
        assert gate.state == 'hold' and gate.latest is None
        clock[0] += .020;gate.tick()  # Fresh feedback confirms that hold.
        assert gate.status()['continuation_ready']
        return solve(*args, **kwargs)
    adapter.solver.solve = next_solve
    clock[0]+=.060  # Transport/next-frame interval consumes the command TTL.
    assert adapter.apply(intent(3, forward=.03)).ok
    gate.tick()
    assert gate.state == 'active' and gate.applied_seq == 2
    assert gate.session_id == session and link.calls == ['claim']
    assert link.last_send['valid_for_ms']==100
    assert gate.diagnostics['first_hold']['code'] == 'command_timeout'
    assert len(writes) == 3  # target, confirmed hold target, new target
    for (a,b),(c,d) in zip(adapter.mapper.reference, reference):
        np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)


def test_waiting_for_fresh_feedback_has_bounded_budget_and_no_output(chain, monkeypatch):
    adapter, link, gate, writes = chain
    clock = [100.]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(time, 'monotonic_ns', lambda: int(clock[0]*1e9))
    gate.clock = time.monotonic_ns
    assert adapter.apply(intent()).ok
    session = gate.session_id
    link.feedback = lambda: (_ for _ in ()).throw(ValueError('driver_feedback_stale_or_different_clock'))
    assert adapter.apply(intent(2)).code == 'waiting_driver_feedback'
    clock[0] += .2
    assert adapter.apply(intent(3)).code == 'waiting_driver_feedback'
    assert not writes and link.calls == ['claim'] and gate.session_id == session
    clock[0] += .101
    ack = adapter.apply(intent(4))
    assert not ack.ok and ack.code == 'feedback_unavailable'
    assert not writes


def test_grip_release_during_timeout_closes_continuation(chain):
    adapter, link, gate, writes = chain
    assert adapter.apply(intent()).ok
    assert adapter.apply(intent(2, forward=.02)).ok
    gate.tick()
    gate.lease_deadline = time.monotonic_ns()-1
    gate.tick()
    assert gate.status()['continuation_allowed']
    assert stop(adapter).ok
    assert not gate.status()['continuation_allowed']
    before = len(writes)
    ack = adapter.apply(intent(3, forward=.02))  # Old dispatch generation is fenced.
    assert not ack.ok and len(writes) == before


def test_hold_beginning_during_ik_waits_for_receipt_without_reclutch(chain, monkeypatch):
    adapter, link, gate, writes = chain
    clock = [100.]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(time, 'monotonic_ns', lambda: int(clock[0]*1e9))
    gate.clock = time.monotonic_ns
    assert adapter.apply(intent()).ok
    assert adapter.apply(intent(2, forward=.01)).ok
    gate.tick()
    solver = adapter.solver.solve
    def solve(*args, **kwargs):
        gate.lease_deadline = time.monotonic_ns()-1
        gate.tick()
        return solver(*args, **kwargs)
    adapter.solver.solve = solve
    ack = adapter.apply(intent(3, forward=.02))
    assert ack.ok and ack.code == 'waiting_driver_hold'
    assert link.seq == 1 and gate.latest is None
    clock[0] += .02;gate.tick()
    adapter.solver.solve = solver
    assert adapter.apply(intent(4, forward=.03)).ok
    gate.tick()
    assert gate.state == 'active' and gate.applied_seq == 2 and link.calls == ['claim']


def test_original_failure_survives_hold_and_snapshot(chain):
    adapter,link,gate,writes=chain
    original=link.feedback
    link.feedback=lambda:(_ for _ in ()).throw(ValueError('driver_feedback_missing'))
    assert adapter.apply(intent()).code=='feedback_unavailable'
    link.feedback=original
    assert stop(adapter,'feedback_unavailable').ok
    failure=adapter.snapshot()['diagnostics']['last_failure']
    assert failure['phase']=='feedback' and failure['code']=='driver_feedback_missing'
    assert failure['sequence']==1 and failure['budget_at_start_ms']>0
    assert not writes


@pytest.mark.parametrize('mode',['shadow','live'])
def test_feedback_recovery_requires_tianyi_opt_in(chain,mode):
    from teleop.runtime import TeleopRuntime
    adapter,link,gate,writes=chain
    adapter.hardware_output=mode=='live'
    runtime=TeleopRuntime(mode=mode,adapter=adapter,auto_watchdog=False)
    try:
        assert 'feedback_unavailable' in runtime._auto_retry_codes
        assert ('command_timeout' in runtime._auto_retry_codes)==(mode=='live')
        assert 'tracking_lost' not in runtime._auto_retry_codes
        assert 'driver_fault' not in runtime._auto_retry_codes
    finally:runtime.close()


def test_initial_claim_does_not_spend_deadline_on_unused_ik(chain):
    from test_teleop import frame
    from teleop.protocol import bind_rtc_frame_v1
    from teleop.runtime import TeleopRuntime
    adapter,link,gate,writes=chain
    claim=link.claim;solve=adapter.solver.solve;solves=[]
    def slow_claim(deadline):
        time.sleep(.060)
        return claim(deadline)
    def slow_solve(*args,**kwargs):
        solves.append(True);time.sleep(.060)
        return solve(*args,**kwargs)
    link.claim=slow_claim;adapter.solver.solve=slow_solve
    runtime=TeleopRuntime(mode='live',adapter=adapter,auto_watchdog=False,pose_timeout_ms=100)
    try:
        runtime.prepare_local_session();binding,_=runtime.rtc_authority_snapshot()
        def send(f):
            f['mode']='live'
            return runtime.submit_frame(bind_rtc_frame_v1(f,authority=binding,expected_mode='live'),source='test')
        send(frame(0));send(frame(1,True,1))
        assert runtime._dispatcher.wait_dispatched(1,timeout=.5),runtime.status()['dispatch']
        assert not solves and not writes and link.lease
        assert runtime.status()['dispatch']['fault_code'] is None
        moved=frame(2,True,1);moved['left_controller']['position'][2]-=.04
        send(moved)
        assert runtime._dispatcher.wait_dispatched(2,timeout=.5)
        gate.tick()
        assert len(solves)==1 and writes and gate.applied_seq==1
        assert link.last_send['valid_for_ms']==100
    finally:
        runtime.close();adapter.close()


def test_transport_preparation_creates_only_local_publisher_once(chain):
    _,link,gate,writes=chain
    link.publisher=None;created=[]
    local_publisher=SimpleNamespace(publish=lambda _:pytest.fail('preparation published a target'))
    link.node=SimpleNamespace(create_publisher=lambda *args:created.append(args) or local_publisher)
    link.topic='/isolated/motion/teleop';link.qos=object()
    DriverLink.prepare_transport(link)
    DriverLink.prepare_transport(link)
    assert len(created)==1 and created[0][1]=='/isolated/motion/teleop/command'
    assert not link.lease and not gate.session_id and not writes


def test_pending_ack_wait_does_not_publish_or_extend_watchdog(chain):
    _,link,gate,writes=chain
    link.claim(time.monotonic()+.1)
    link.seq=1;deadline=time.monotonic()-.01
    link.pending.append((1,deadline))
    for target in ([.1]*14,[.2]*14):
        assert link.send(target,[0.,0.],time.monotonic()+.1,
                         wait_for_execution=False,allow_continuation=True) is False
    assert link.seq==1 and list(link.pending)==[(1,deadline)] and not writes
    with pytest.raises(ValueError,match='driver_execution_ack_timeout'):
        link.send([.1]*14,[0.,0.],time.monotonic()+.1,wait_for_execution=False)
    link.pending[0]=(1,time.monotonic()-.201)
    with pytest.raises(ValueError,match='driver_execution_ack_timeout'):
        link.send([.1]*14,[0.,0.],time.monotonic()+.1,
                  wait_for_execution=False,allow_continuation=True)
    gate.applied_seq=1
    assert link.send([.1]*14,[0.,0.],time.monotonic()+.1,
                     wait_for_execution=False,allow_continuation=True) is True
    assert link.seq==2 and link.last_send['valid_for_ms']<=100


@pytest.mark.parametrize('reason',['command_expired','command_timeout','arm_ns_stale'])
def test_first_packet_hold_recovers_without_new_grip_or_failed_target(chain,reason):
    from test_teleop import frame
    from teleop.protocol import bind_rtc_frame_v1
    from teleop.runtime import TeleopRuntime
    a,link,gate,writes=chain
    runtime=TeleopRuntime(mode='live',adapter=a,auto_watchdog=False,pose_timeout_ms=100)
    try:
        runtime.prepare_local_session();binding,_=runtime.rtc_authority_snapshot()
        def send(seq,held=True):
            f=frame(seq,held,1 if held else 0);f['mode']='live'
            runtime.submit_frame(bind_rtc_frame_v1(f,authority=binding,expected_mode='live'),source='test')
        send(0,False);send(1);assert runtime._dispatcher.wait_dispatched(1,.5)
        old=gate.session_id;reference=copy.deepcopy(a.mapper.reference)
        assert gate.applied_seq==-1
        gate.hold(reason);gate.tick();time.sleep(.001);gate.tick()
        send(2)
        end=time.monotonic()+.5
        while time.monotonic()<end:
            status=runtime.status()['dispatch']
            if status['state']=='safe_reclutch_required' and status['stop_acknowledged']:break
            time.sleep(.002)
        count=len(writes)
        send(3);assert runtime._dispatcher.wait_dispatched(3,.5)
        assert a.output['state']=='recovery_validated' and len(writes)==count
        send(4);assert runtime._dispatcher.wait_dispatched(4,.5)
        assert gate.session_id!=old and len(writes)==count
        for (x,y),(u,v) in zip(reference,a.mapper.reference):
            np.testing.assert_array_equal(x,u);np.testing.assert_array_equal(y,v)
        send(5);assert runtime._dispatcher.wait_dispatched(5,.5);gate.tick()
        assert gate.applied_seq==1 and runtime.status()['state']=='active_live'
    finally:runtime.close();a.close()



def test_solve_budget_is_separate_from_new_command_transport_ttl(chain,monkeypatch):
    a,link,gate,writes=chain;clock=[100.];packets=[]
    monkeypatch.setattr(time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(time,'monotonic_ns',lambda:int(clock[0]*1e9));gate.clock=time.monotonic_ns
    assert a.apply(intent()).ok
    solve=a.solver.solve
    def slow(*args,**kw):clock[0]+=.060;return solve(*args,**kw)
    a.solver.solve=slow
    link.publisher=SimpleNamespace(publish=lambda msg:packets.append(json.loads(msg.data)))
    assert a.apply(intent(2,forward=.04)).ok
    assert packets[-1]['valid_for_ms']==100
    clock[0]+=.045;gate.accept(packets[-1]);gate.tick()
    assert gate.applied_seq==1 and writes
    count=len(packets);old=intent(3,forward=.05);clock[0]+=.101
    assert not a.apply(old).ok and len(packets)==count
    with pytest.raises(ValueError,match='motion_deadline'):
        link.send([0.]*14,[0.,0.],time.monotonic()-.001,target_ttl_ms=100)
    for invalid in (0,101,True):
        with pytest.raises(ValueError,match='invalid_target_ttl'):
            link.send([0.]*14,[0.,0.],time.monotonic()+.1,target_ttl_ms=invalid)


def test_tianyi_input_age_300ms_does_not_extend_command_ttl(chain,monkeypatch):
    from dataclasses import replace
    a,link,gate,writes=chain;clock=[100.];packets=[]
    monkeypatch.setattr(time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(time,'monotonic_ns',lambda:int(clock[0]*1e9));gate.clock=time.monotonic_ns
    assert a.apply(intent()).ok
    original=a.solver.solve
    def solve(*args,**kwargs):
        clock[0]+=.070
        return original(*args,**kwargs)
    a.solver.solve=solve
    link.publisher=SimpleNamespace(publish=lambda msg:packets.append(json.loads(msg.data)))
    pending=replace(intent(2,forward=.04),received_monotonic=99.920,expires_monotonic=100.220)
    assert a.apply(pending).ok  # 80 ms queue age + 70 ms solve, within input bound.
    assert packets[-1]['valid_for_ms']==100
    clock[0]+=.045;gate.accept(packets[-1]);gate.tick()
    assert gate.applied_seq==1 and writes
    count=len(packets)
    expired=replace(intent(3,forward=.05),received_monotonic=clock[0]-.301,expires_monotonic=clock[0]+1.)
    assert not a.apply(expired).ok and len(packets)==count


def test_tianyi_adapter_and_runtime_have_matching_input_budget(chain):
    from teleop.runtime import TeleopRuntime
    a,link,gate,writes=chain
    runtime=TeleopRuntime(mode='live',adapter=a,auto_watchdog=False,
                         pose_timeout_ms=a.input_timeout_ms,
                         dispatch_io_timeout_ms=a.dispatch_io_timeout_ms)
    try:
        assert runtime._pose_timeout==.3
        assert runtime._dispatcher._io_timeout==.15
    finally:runtime.close()


@pytest.mark.parametrize('persistent',[False,True])
def test_feedback_ages_during_solve_drops_target_with_bounded_wait(chain,monkeypatch,persistent):
    a,link,gate,writes=chain;clock=[100.]
    monkeypatch.setattr(time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(time,'monotonic_ns',lambda:int(clock[0]*1e9));gate.clock=time.monotonic_ns
    assert a.apply(intent()).ok
    original=link.feedback;stale=[False];solve=a.solver.solve
    def feedback():
        if stale[0]:raise ValueError('driver_feedback_stale_or_different_clock')
        return original()
    def solving(*args,**kwargs):
        result=solve(*args,**kwargs);stale[0]=True;return result
    link.feedback=feedback;a.solver.solve=solving
    reference=copy.deepcopy(a.mapper.reference)
    for seq in (2,3):
        stale[0]=False  # Fresh at admission, stale again at send: not a reset.
        clock[0]+=.1
        result=a.apply(intent(seq,forward=.04))
        assert result.ok and result.code=='waiting_driver_feedback'
        assert link.seq==0 and not writes and gate.state=='ready'
    if persistent:
        stale[0]=False;clock[0]+=.201
        result=a.apply(intent(4,forward=.04))
        assert not result.ok and result.code=='feedback_unavailable'
        assert link.seq==0 and not writes
    else:
        stale[0]=False;a.solver.solve=solve;clock[0]+=.02
        assert a.apply(intent(4,forward=.04)).ok
        gate.tick();assert gate.applied_seq==1 and writes
        assert a._feedback_wait_started is None
        for (x,y),(u,v) in zip(reference,a.mapper.reference):
            np.testing.assert_array_equal(x,u);np.testing.assert_array_equal(y,v)


@pytest.mark.parametrize('failure', ['ik_timeout','ik_target_unreachable','ik_not_converged'])
def test_ik_hold_first_valid_frame_sends_in_same_session(chain, failure):
    a,link,gate,writes=chain
    assert a.apply(intent()).ok
    assert a.apply(intent(2,forward=.04)).ok
    gate.tick()
    owner=gate.session_id; previous_seq=gate.seq
    solve=a.solver.solve
    for seq in (3,4,5):
        def failed(*args,**kwargs):raise ValueError(failure)
        a.solver.solve=failed
        assert not a.apply(intent(seq,generation=seq-2,forward=.06)).ok
        assert stop(a,failure,generation=seq-1).ok
        assert gate.latest is None and gate.session_id==owner
    a.solver.solve=solve
    assert a.apply(intent(6,generation=4,forward=.08)).ok
    assert a.output['state']=='submitted'
    assert gate.session_id==owner and gate.seq>previous_seq
    assert 'resume' not in link.calls and 'pause' not in link.calls
    gate.tick();assert gate.state=='active'
    assert stop(a,generation=5).ok
    assert not gate.status()['continuation_allowed']
    assert a.close().ok


def test_recorded_inputs_injected_ik_failures_keep_session(chain):
    path=os.environ.get('TIANYI_RECOVERY_RECORDING')
    if not path:pytest.skip('set TIANYI_RECOVERY_RECORDING to an existing PICO poses.jsonl')
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    a,link,gate,writes=chain
    # Real recorded input and relative mapping; bounded synthetic solve/plant.
    # This verifies recovery, not numerical IK accuracy or physical tracking.
    a.solver.hands_enabled=False
    def valid(targets,q,**kwargs):
        result=list(q)
        for index,target in zip((0,7),targets):result[index]=float(.02*np.tanh(target[0,3]))
        return result
    a.solver.solve=valid
    failing={30,31,32,100,101,210,400,401,402,403,600}
    generation=1;owner=None;recovering=False;recoveries=0
    for index,row in enumerate(rows):
        if not row['frame']['deadman']:continue
        current=intent(index+1,generation=generation)
        current.frame.update(copy.deepcopy(row['frame']))
        if index in failing:
            def failed(*args,**kwargs):raise ValueError('ik_timeout')
            a.solver.solve=failed
        else:a.solver.solve=valid
        ack=a.apply(current)
        if index in failing:
            assert not ack.ok and ack.code=='ik_timeout'
            generation+=1
            assert stop(a,'ik_timeout',generation=generation).ok
            assert gate.latest is None
            recovering=True
        else:
            assert ack.ok
            if recovering:
                assert a.output['state']=='submitted'
                recoveries+=1;recovering=False
            gate.tick()
        if owner is None:owner=gate.session_id
        assert gate.session_id==owner
    assert len(rows)==719 and recoveries==5
    assert 'resume' not in link.calls and 'pause' not in link.calls
    assert a.close().ok and not gate.session_id


def test_legacy_driver_ik_hold_uses_full_recovery(chain):
    a,link,gate,_=chain
    feedback=link.feedback
    def legacy():
        state=feedback();state['timing_policy'].pop('recoverable_hold',None);return state
    link.feedback=legacy
    assert a.apply(intent()).ok
    assert stop(a,'ik_timeout').ok
    assert link.calls[-1]=='pause'
    assert a.apply(intent(2,generation=2)).ok
    assert a.output['state']=='recovery_validated'
    assert a.apply(intent(3,generation=2)).ok
    assert link.calls[-1]=='resume'
    assert a.close().ok


def test_adapter_never_searches_for_alternative_target(chain):
    a,link,gate,_=chain
    def forbidden(*args,**kwargs):raise AssertionError('projection must not run')
    a.solver.solve_projected=forbidden
    assert a.apply(intent(1)).ok
    assert a.apply(intent(2)).ok;gate.tick()
    assert a.close().ok


def test_ik_failure_racing_noncontinuable_hold_confirms_pause(chain):
    a,link,gate,writes=chain
    assert a.apply(intent()).ok
    assert a.apply(intent(2,forward=.04)).ok
    gate.tick()
    owner=gate.session_id
    gate.hold('power_ns_stale')
    assert stop(a,'ik_timeout',generation=2).ok
    assert not a._same_session_hold and a._automatic_resume
    assert a.output['state']=='held'
    assert link.calls[-2:]==['recoverable_hold','pause']
    assert a.apply(intent(3,generation=2,forward=.05)).ok
    assert a.output['state']=='recovery_validated'
    assert a.apply(intent(4,generation=2,forward=.05)).ok
    assert a.apply(intent(5,generation=2,forward=.05)).ok
    assert a.output['state']=='submitted'
    assert a.close().ok
