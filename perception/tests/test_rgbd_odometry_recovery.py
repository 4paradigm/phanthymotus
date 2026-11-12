"""Automatic recovery exercises real plugin/session lifecycle, without ROS processes."""
import sys
import time
from types import SimpleNamespace as NS

import pytest

from plugins import rgbd_odometry as module
from plugins import rgbd_odometry_runtime as runtime
from test_rgbd_odometry_contract import declarations
from test_rgbd_odometry_status import sessions, tracking, status
from vision_stubs import _FakeExecutor, _FakeNode


@pytest.fixture
def recovery(sessions, monkeypatch):
    _, now, _ = sessions
    seconds = [10.]
    monkeypatch.setattr(time, 'monotonic', lambda: seconds[0])
    card = module.RgbdOdometryPlugin({}, _FakeExecutor())
    args = dict(action='start', instance_id='one', input_topics=['/rgb', '/depth'],
                camera_info=declarations())
    assert card.dispatch('rgbd_odometry', args)['state'] == 'running'
    try:
        yield card, args, now, seconds
    finally:
        for session, *_ in card.instances.values(): session.process.returncode = 0
        card.shutdown()


def images(session, now):
    for kind in ('color', 'depth'):
        session._image(kind, NS(header=NS(frame_id='camera_optical',
            stamp=NS(sec=now//10**9, nanosec=now%10**9))))


def lose(card, now):
    old = card.instances['one'][0]
    old.process.returncode = 0
    old.gate.fault('visual covariance exceeds tracking bounds')
    if old.fusion is not None: old.fusion.fault(old.gate.reason)
    images(old, now)
    card._recover()
    return old


def test_lost_recovers_into_new_segment_with_fresh_origin_and_unchanged_ports(recovery):
    card, args, now, seconds = recovery
    old = card.instances['one'][0]
    tracking(old, [4., 6., 3.])
    ports = card.dispatch('rgbd_odometry', dict(action='info', instance_id='one'))['topic_out']
    lose(card, now[0])
    seconds[0] += .79
    card._recover()
    assert card.instances['one'][0] is old
    seconds[0] += .02
    card._recover()
    new = card.instances['one'][0]
    new.process.returncode = 0
    assert new is not old and new.segment != old.segment
    assert old.node.destroyed and old.log.closed
    assert new.gate.state == 'initializing' and new.gate.good == 0
    assert old.status_publisher.messages == [] or status(old)['status'] != 'idle'
    tracking(new, [9., 8., 7.])
    assert status(new)['displacement_m'] == 0.
    info = card.dispatch('rgbd_odometry', dict(action='info', instance_id='one'))
    assert info['topic_out'] == ports
    assert info['recovery']['restarts'] == 1
    assert info['recovery']['previous_segment'] == old.segment
    assert new.calibration == old.calibration


def test_missing_or_wrong_frame_input_cannot_restart_lost_instance(recovery):
    card, _, now, seconds = recovery
    old = lose(card, now[0])
    seconds[0] += 2.
    now[0] += 2_000_000_000
    card._recover()
    assert card.instances['one'][0] is old
    old._image('color', NS(header=NS(frame_id='wrong', stamp=NS(sec=3, nanosec=10_000_000))))
    card._recover()
    assert card.instances['one'][0] is old
    images(old, now[0])
    card._recover()
    assert card.instances['one'][0] is not old


def test_failed_candidate_is_retried_with_backoff_without_disposing_original(recovery, monkeypatch):
    card, _, now, seconds = recovery
    old = lose(card, now[0]); real = module.RgbdSession; attempts = []
    def unavailable(*args, **kwargs):
        attempts.append(seconds[0])
        raise RuntimeError('temporary estimator launch failure')
    monkeypatch.setattr(module, 'RgbdSession', unavailable)
    seconds[0] += 1.
    card._recover()
    card._recover()
    assert len(attempts) == 1 and not old.node.destroyed
    info = card.dispatch('rgbd_odometry', dict(action='info', instance_id='one'))
    assert info['status'] == 'lost'
    assert 'temporary estimator' in info['recovery']['error']
    seconds[0] += 1.49
    card._recover()
    assert len(attempts) == 1
    monkeypatch.setattr(module, 'RgbdSession', real)
    seconds[0] += .02
    card._recover()
    new = card.instances['one'][0]; new.process.returncode = 0
    assert new is not old and old.node.destroyed


def test_stop_and_shutdown_cancel_recovery_and_supervisor(recovery):
    card, _, now, seconds = recovery
    old = lose(card, now[0]); timer = card.recovery_timer
    assert card.dispatch('rgbd_odometry', dict(action='stop', instance_id='one'))['state'] == 'idle'
    seconds[0] += 10.
    timer.callback()
    assert not card.instances and old.node.destroyed
    node = card.recovery_node
    card.shutdown()
    assert node.destroyed
    timer.callback()
    assert not card.instances


def test_cleanup_failure_closes_candidate_and_keeps_original_for_explicit_stop(recovery, monkeypatch):
    card, _, now, seconds = recovery
    old = lose(card, now[0]); close = old.close; created = []; real = module.RgbdSession
    def candidate(*args, **kwargs):
        s = real(*args, **kwargs); s.process.returncode = 0; created.append(s); return s
    monkeypatch.setattr(module, 'RgbdSession', candidate)
    def fail(**kwargs):
        tracking(created[0], [99., 99., 99.])  # candidate's private callback runs during old join
        raise RuntimeError('old worker did not stop')
    monkeypatch.setattr(old, 'close', fail)
    seconds[0] += 1.
    card._recover()
    assert card.instances['one'][0] is old and not old.node.destroyed
    assert len(created) == 1 and created[0].node.destroyed
    assert not created[0].status_publisher.messages  # a rejected candidate must not report idle
    assert not created[0].publisher.messages
    assert 'old worker' in card.recovery['one']['error']
    monkeypatch.setattr(old, 'close', close)


def test_failed_candidate_constructor_cannot_publish_idle_over_live_lost_session(recovery, monkeypatch):
    card, _, now, seconds = recovery
    old = lose(card, now[0]); seconds[0] += 1.
    def unavailable(*args, **kwargs): raise RuntimeError('native spawn failed')
    monkeypatch.setattr(runtime.subprocess, 'Popen', unavailable)
    card._recover()
    candidate_node = _FakeNode.instances[-1]
    assert candidate_node.destroyed and not old.node.destroyed
    assert card.instances['one'][0] is old
    assert all(not p.messages for p in candidate_node.publishers if p.topic.endswith('/status'))


def test_failed_candidate_cleanup_remains_owned_until_explicit_stop(recovery, monkeypatch):
    card, args, now, seconds = recovery
    old = lose(card, now[0]); old_close = old.close; created = []; real = module.RgbdSession
    def fail_old(**kw): raise RuntimeError('old worker stuck')
    def candidate(*args, **kwargs):
        s = real(*args, **kwargs); s.process.returncode = 0; created.append((s, s.close))
        def fail(**kw): raise RuntimeError('candidate worker stuck')
        s.close = fail
        return s
    monkeypatch.setattr(old, 'close', fail_old)
    monkeypatch.setattr(module, 'RgbdSession', candidate)
    seconds[0] += 1.
    card._recover()
    staged, close = created[0]
    assert card.retiring['one'] is staged
    assert 'old worker stuck' in card.recovery['one']['error']
    assert 'candidate worker stuck' in card.recovery['one']['error']
    seconds[0] += 10.
    card._recover()
    assert len(created) == 1  # no further candidates while cleanup is unresolved
    monkeypatch.setattr(old, 'close', old_close)
    assert card.dispatch('rgbd_odometry', dict(action='stop', instance_id='one'))['state'] == 'error'
    assert old.node.destroyed and 'one' not in card.instances
    monkeypatch.setattr(module, 'RgbdSession', real)
    try:
        assert card.dispatch('rgbd_odometry', args)['state'] == 'error'
    finally:
        staged.close = close
    assert card.dispatch('rgbd_odometry', dict(action='stop', instance_id='one'))['state'] == 'idle'
    assert old.node.destroyed and staged.node.destroyed and not card.retiring


def test_recovery_does_not_replace_healthy_other_instance(recovery):
    card, args, now, seconds = recovery
    card.dispatch('rgbd_odometry', dict(args, instance_id='two'))
    other = card.instances['two'][0]; other.process.returncode = 0
    old = lose(card, now[0]); seconds[0] += 1.
    card._recover()
    assert card.instances['one'][0] is not old
    assert card.instances['two'][0] is other and not other.node.destroyed


def test_lost_image_ingress_updates_freshness_without_feeding_retired_estimator(recovery):
    card, _, now, _ = recovery
    old = lose(card, now[0]); pending = old.pending.qsize()
    now[0] += 100_000_000
    images(old, now[0])
    assert old.has_fresh_inputs(now[0])
    assert not old.has_fresh_inputs(now[0]+1_500_000_001)
    assert old.pending.qsize() == pending and old.gate.state == 'lost'


@pytest.mark.parametrize('fusion', [False, True])
def test_assisted_recovery_retains_settings_and_reinitializes_imu_history(recovery, monkeypatch, fusion):
    from test_rgbd_imu import packet
    card, args, now, seconds = recovery
    card.instances['one'][0].process.returncode = 0
    card.dispatch('rgbd_odometry', dict(action='stop', instance_id='one'))
    assisted_runtime(monkeypatch)
    settings = dict(imu_assisted=True, short_term_compensation=fusion, depth_geometry=fusion,
                    visual_frontend='f2f_optical_flow', max_features=500)
    assert card.dispatch('rgbd_odometry', dict(args, input_topics=['/rgb', '/depth', '/imu'],
                                             **settings))['state'] == 'running'
    old = card.instances['one'][0]
    old.imu.observe(packet(camera_frame_id='camera_optical', clock_id='camera-global'), now[0])
    assert old.imu.history
    lose(card, now[0]); seconds[0] += 1.
    card._recover()
    new = card.instances['one'][0]; new.process.returncode = 0
    assert new is not old and new.owns_executor
    assert not new.imu.history and new.imu.state == 'initializing'
    assert bool(new.fusion) == fusion
    if fusion: assert new.fusion.filter is None
    assert new.calibration == old.calibration
    assert card.dispatch('rgbd_odometry', dict(action='info', instance_id='one'))['config'] == settings


def assisted_runtime(monkeypatch):
    monkeypatch.setattr(_FakeNode, 'context', object(), raising=False)
    def executor(**kw):
        value = _FakeExecutor()
        value.shutdown = lambda **kw: None
        return value
    monkeypatch.setattr(sys.modules['rclpy.executors'], 'SingleThreadedExecutor', executor)
    monkeypatch.setitem(sys.modules, 'rclpy.callback_groups', NS(MutuallyExclusiveCallbackGroup=object))
    subscribe = _FakeNode.create_subscription
    monkeypatch.setattr(_FakeNode, 'create_subscription',
        lambda node, kind, topic, cb, qos, **kw: subscribe(node, kind, topic, cb, qos))
    monkeypatch.setattr(sys.modules['sensor_msgs.msg'], 'Imu', object, raising=False)


@pytest.mark.parametrize('failing_start', [1, 2])
def test_worker_start_failure_cleans_native_process_and_node_before_retry(recovery, monkeypatch, failing_start):
    card, args, now, seconds = recovery
    card.instances['one'][0].process.returncode = 0
    card.dispatch('rgbd_odometry', dict(action='stop', instance_id='one'))
    assisted_runtime(monkeypatch)
    assert card.dispatch('rgbd_odometry', dict(args, imu_assisted=True,
        input_topics=['/rgb', '/depth', '/imu']))['state'] == 'running'
    old = lose(card, now[0]); starts = []
    class Thread:
        def __init__(self, **kw): self.started = False
        def start(self):
            starts.append(self)
            if len(starts) == failing_start: raise RuntimeError('cannot start native callback worker')
            self.started = True
        def join(self, **kw):
            if not self.started: raise RuntimeError('cannot join thread before it is started')
        def is_alive(self): return False
    processes = []
    def process(*args, **kw):
        p = NS(pid=12345, running=True)
        p.poll = lambda: None if p.running else 0
        p.wait = lambda **kw: 0
        processes.append(p)
        return p
    monkeypatch.setattr(runtime.threading, 'Thread', Thread)
    monkeypatch.setattr(runtime.subprocess, 'Popen', process)
    monkeypatch.setattr(runtime.os, 'killpg', lambda pid, sig: setattr(processes[-1], 'running', False))
    seconds[0] += 1.
    card._recover()
    assert card.instances['one'][0] is old and not old.node.destroyed
    assert _FakeNode.instances[-1].destroyed and not processes[0].running
    assert 'cannot start native callback worker' in card.recovery['one']['error']
