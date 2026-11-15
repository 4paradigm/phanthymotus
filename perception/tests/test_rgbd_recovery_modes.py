"""Bounded initialization and optional-source recovery at real session boundaries."""
import json
import time
from types import SimpleNamespace as NS

import pytest

from plugins import rgbd_odometry as module
from plugins import rgbd_odometry_runtime as runtime
from test_rgbd_degradation import feed
from test_rgbd_imu import packet
from test_rgbd_odometry_recovery import assisted_runtime, recovery  # noqa: F401
from test_rgbd_odometry_status import sessions, status  # noqa: F401


def images(session, stamp):
    for kind in ('color', 'depth'):
        session._image(kind, NS(data=b'capture', header=NS(frame_id='camera_optical',
            stamp=NS(sec=stamp//10**9, nanosec=stamp%10**9))))


def assisted(card, args, monkeypatch, *, fusion=False):
    old = card.instances['one'][0]
    old.process.returncode = 0
    card.dispatch('rgbd_odometry', dict(action='stop', instance_id='one'))
    assisted_runtime(monkeypatch)
    result = card.dispatch('rgbd_odometry', dict(args,
        input_topics=['/rgb', '/depth', '/imu'], short_term_compensation=fusion))
    assert result['state'] == 'running', result
    return card.instances['one'][0]


def imu_packet(stamp, **changes):
    return packet(stamp, camera_frame_id='camera_optical', clock_id='camera-global',
                  translation_source='sdk_factory_extrinsics',
                  camera_from_gyro_translation=[.03, 0., 0.],
                  camera_from_accel_translation=[.03, 0., 0.], **changes)


def send_imu(session, now, seconds, stamp, **changes):
    now[0] = stamp + 1_000_000
    session._imu(NS(data=json.dumps(imu_packet(stamp, **changes))))


def test_fresh_weak_visual_start_has_total_budget_and_explains_progress(recovery):
    card, _, now, seconds = recovery
    old = card.instances['one'][0]
    start = now[0]
    for k in range(151):
        seconds[0] = 10. + k*.1
        stamp = start+k*100_000_000
        now[0] = stamp+20_000_000
        images(old, stamp)
        feed(old, now, stamp, cov=.25)
        old._watchdog()
    assert old.gate.state == 'lost'
    assert old.gate.loss_kind == 'visual'
    assert 'initialization' in old.gate.reason and '15' in old.gate.reason
    diagnostic = old.info()['initialization']
    assert diagnostic['phase'] == 'timed_out'
    assert diagnostic['consecutive_visual_estimates'] == 0
    assert diagnostic['required_visual_estimates'] == 3
    assert diagnostic['elapsed_s'] >= diagnostic['budget_s'] == 15.
    assert status(old)['initialization']['phase'] == 'timed_out'
    assert all(json.loads(m)['pose'] is None for m in old.publisher.messages)


def test_never_arriving_attitude_switches_to_new_explicit_visual_segment(recovery, monkeypatch):
    card, args, now, seconds = recovery
    old = assisted(card, args, monkeypatch)
    now[0] += 11_000_000_000
    seconds[0] += 11.
    images(old, now[0])
    old._watchdog()
    assert old.gate.loss_kind == 'imu'
    old.process.returncode = 0
    card._recover()
    seconds[0] += .81
    card._recover()
    new = card.instances['one'][0]
    new.process.returncode = 0
    assert new is not old and new.segment != old.segment
    assert new.imu is None and new.fusion is None
    assert not runtime.estimator_parameters(new.calibration, new.segment).get('wait_imu_to_init')
    assert new.requested_calibration['imu_topic'] == '/imu'
    assert any(s.topic == '/imu' for s in new.node.subscriptions)
    assert not any(p.topic.endswith('/imu') for p in new.node.publishers)
    info = card.dispatch('rgbd_odometry', dict(action='info', instance_id='one'))
    assert info['mode']['requested'] == 'imu_attitude'
    assert info['mode']['effective'] == 'visual_only' and info['mode']['degraded']
    assert info['config']['imu_assisted'] is True
    assert info['recovery']['transitions'][-1]['action'] == 'degrade_assistance'
    # Native frames no longer depend on IMU coverage.
    assert new._imu_frame_ready(now[0], now[0])


def test_healthy_attitude_does_not_allow_visual_warmup_to_run_forever(recovery, monkeypatch):
    card, args, now, seconds = recovery
    old = assisted(card, args, monkeypatch)
    start = now[0]
    for k in range(151):
        seconds[0] = 10.+k*.1
        stamp = start+k*100_000_000
        now[0] = stamp+20_000_000
        old.imu.observe(imu_packet(stamp), now[0])
        images(old, stamp)
        feed(old, now, stamp, cov=.25)
        old._watchdog()
    assert old.imu.state == 'active'
    assert old.gate.state == 'lost' and old.gate.loss_kind == 'visual'
    assert 'visual initialization' in old.gate.reason


def test_initialized_filter_without_first_tracking_has_a_finite_fusion_budget(recovery, monkeypatch):
    from test_rgbd_fusion import images as fusion_images, pose
    card, args, now, seconds = recovery
    session = assisted(card, args, monkeypatch, fusion=True)
    color, depth = fusion_images()
    visual = dict(pose(), frame_id=session.segment+'_odom', child_frame_id='camera_optical')
    start = now[0]
    for k in range(1501):
        seconds[0] = 10.+k*.01
        stamp = start+k*10_000_000
        now[0] = stamp+1_000_000
        session._imu(NS(data=json.dumps(imu_packet(stamp))))
        if k % 10:
            continue
        images(session, stamp)
        sample = session.fusion.observe_frame(stamp, color, depth,
            visual if k <= 60 or k % 20 == 0 else None, now[0])
        session._publish(sample)
        if k >= 80:
            assert session.fusion.filter is not None and session.first_tracking_at is None
        session._watchdog()
    assert session.fusion.state == 'lost'
    assert 'initialization' in session.fusion.reason and '15' in session.fusion.reason
    assert session.fallback is not None and session.gate.state != 'lost'
    assert session.info()['initialization']['elapsed_s'] == 0.
    assert session.info()['mode']['effective'] == 'visual_only'
    assert all(json.loads(m)['pose'] is None for m in session.publisher.messages)


def degraded_session(card, args, now, seconds, monkeypatch, *, fusion=False):
    old = assisted(card, args, monkeypatch, fusion=fusion)
    old.gate.fault('IMU source expired', 'imu')
    old.imu.fault('capture stale')
    old.process.returncode = 0
    images(old, now[0]); card._recover()
    seconds[0] += .81
    card._recover()
    new = card.instances['one'][0]
    assert new is not old and new.imu is None
    return new


@pytest.mark.parametrize('fusion', [False, True])
def test_sustained_recovered_source_restores_requested_mode_only_in_new_segment(
        recovery, monkeypatch, fusion):
    card, args, now, seconds = recovery
    visual = degraded_session(card, args, now, seconds, monkeypatch, fusion=fusion)
    segment = visual.segment
    # One good sample is insufficient, even when the cooldown has elapsed.
    for _ in range(201):
        seconds[0] += .1
        now[0] += 100_000_000
        images(visual, now[0])
        feed(visual, now, now[0])
    first = now[0]
    send_imu(visual, now, seconds, first)
    images(visual, now[0]); card._recover()
    assert card.instances['one'][0] is visual
    assert visual.info()['auxiliary_recovery']['status'] == 'warming'
    for k in range(1, 32):
        seconds[0] += .1
        send_imu(visual, now, seconds, first+k*100_000_000)
        images(visual, now[0])
    visual.process.returncode = 0
    card._recover()
    restored = card.instances['one'][0]
    restored.process.returncode = 0
    assert restored is not visual and restored.segment != segment
    assert restored.imu is not None and not restored.imu.history
    assert bool(restored.fusion) is fusion
    assert restored.info()['mode']['degraded'] is False
    transition = card.dispatch('rgbd_odometry', dict(action='info', instance_id='one'))['recovery']['transitions'][-1]
    assert transition['action'] == 'restore_assistance'
    assert transition['previous_segment'] == segment


def test_probe_requires_real_time_continuity_and_complete_compensation_calibration():
    from plugins.rgbd_recovery import AuxiliaryHealthProbe
    probe = AuxiliaryHealthProbe('camera_optical', 'camera-global', require_translation=True)
    start = 1_000_000_000
    # A rapidly drained backlog cannot manufacture three seconds of health.
    for k in range(31):
        stamp = start+k*100_000_000
        probe.observe(imu_packet(stamp), stamp+1_000_000, 10.)
    assert probe.info(start+3_001_000_000, 10.)['status'] != 'ready'
    # A source epoch change restarts the dwell, and a stale packet does not heal it.
    probe.observe(imu_packet(start+3_100_000_000, epoch='boot2'), start+3_101_000_000, 13.1)
    assert probe.info(start+3_101_000_000, 13.1)['healthy_for_s'] == 0.
    probe.observe(imu_packet(start+3_200_000_000), start+4_000_000_000, 14.)
    assert probe.info(start+4_000_000_000, 14.)['status'] != 'ready'
    for k in range(40):
        stamp = start+4_100_000_000+k*100_000_000
        broken = imu_packet(stamp)
        broken.pop('camera_from_accel_translation')
        probe.observe(broken, stamp+1_000_000, 14.1+k*.1)
    info = probe.info(stamp+1_000_000, 18.)
    assert info['status'] != 'ready' and 'translation' in info['reason']
    assert info['buffered_samples'] <= 512


def test_probe_rejects_wrong_frame_or_clock_without_stopping_visual(recovery, monkeypatch):
    card, args, now, seconds = recovery
    visual = degraded_session(card, args, now, seconds, monkeypatch)
    for k in range(40):
        seconds[0] += .1
        now[0] += 100_000_000
        bad = imu_packet(now[0]); bad['clock_id'] = 'wrong-clock'
        visual._imu(NS(data=json.dumps(bad)))
        images(visual, now[0])
        feed(visual, now, now[0])
    assert visual.gate.state == 'tracking'
    assert visual.info()['auxiliary_recovery']['status'] != 'ready'
    assert 'clock' in visual.info()['auxiliary_recovery']['reason']
    assert card.instances['one'][0] is visual


def test_retry_backoff_grows_and_is_reported(recovery, monkeypatch):
    card, _, now, seconds = recovery
    old = card.instances['one'][0]
    old.gate.fault('visual loss', 'visual')
    images(old, now[0]); card._recover()
    calls = []
    def fail(*args, **kwargs):
        calls.append(seconds[0])
        raise RuntimeError('unavailable')
    monkeypatch.setattr(module, 'RgbdSession', fail)
    seconds[0] += .81; card._recover()
    assert len(calls) == 1
    seconds[0] += 1.51; card._recover()
    assert len(calls) == 2
    seconds[0] += 1.51; card._recover()
    assert len(calls) == 2
    info = card.dispatch('rgbd_odometry', dict(action='info', instance_id='one'))
    assert info['recovery']['retry_in_s'] > 1.
    assert info['recovery']['backoff_s'] == 3.
    seconds[0] += 1.5; card._recover()
    assert len(calls) == 3
    assert card.recovery['one']['backoff_s'] == 6.


def test_weak_visual_budget_can_trigger_existing_auto_frontend_recovery(recovery):
    card, args, now, seconds = recovery
    card.instances['one'][0].process.returncode = 0
    card.dispatch('rgbd_odometry', dict(action='stop', instance_id='one'))
    assert card.dispatch('rgbd_odometry', dict(args, visual_frontend='auto'))['state'] == 'running'
    old = card.instances['one'][0]
    start = now[0]
    for k in range(151):
        seconds[0] = 10.+k*.1
        stamp = start+k*100_000_000
        now[0] = stamp+20_000_000
        images(old, stamp); feed(old, now, stamp, cov=.25)
        old._watchdog()
    assert old.gate.loss_kind == 'visual'
    old.process.returncode = 0
    card._recover()
    seconds[0] += .81
    now[0] += 810_000_000
    images(old, now[0]); card._recover()
    new = card.instances['one'][0]
    assert new is not old and new.calibration['visual_frontend'] == 'f2f_optical_flow'


def healthy_visual_tick(session, now, seconds, *, auxiliary=True):
    seconds[0] += .1
    stamp = now[0]+100_000_000
    now[0] = stamp+1_000_000
    if auxiliary:
        session._imu(NS(data=json.dumps(imu_packet(stamp))))
    images(session, stamp)
    feed(session, now, stamp)


def test_failed_restore_keeps_visual_output_and_waits_twenty_seconds(recovery, monkeypatch):
    card, args, now, seconds = recovery
    visual = degraded_session(card, args, now, seconds, monkeypatch)
    for _ in range(240):
        healthy_visual_tick(visual, now, seconds)
    assert visual.info()['auxiliary_recovery']['status'] == 'ready'
    calls = []
    def fail(*args, **kwargs):
        calls.append(seconds[0])
        raise RuntimeError('temporary restore launch failure')
    monkeypatch.setattr(module, 'RgbdSession', fail)
    card._recover()
    assert calls and card.instances['one'][0] is visual and not visual.node.destroyed
    published = len(visual.publisher.messages)
    for _ in range(199):
        healthy_visual_tick(visual, now, seconds)
        card._recover()
    assert len(calls) == 1
    assert len(visual.publisher.messages) > published and visual.gate.state == 'tracking'
    for _ in range(2):
        healthy_visual_tick(visual, now, seconds)
        card._recover()
    assert len(calls) == 2
    assert card.recovery['one']['restoration_attempts'] == 2


def test_epoch_change_needs_new_sustained_evidence_and_old_epoch_cannot_rejoin():
    from plugins.rgbd_recovery import AuxiliaryHealthProbe
    probe = AuxiliaryHealthProbe('camera_optical', 'camera-global')
    for k in range(70):
        stamp = 1_000_000_000+k*50_000_000
        probe.observe(imu_packet(stamp), stamp+1_000_000, 10.+k*.05)
    assert probe.info(stamp+1_000_000, 13.45)['status'] == 'ready'
    stamp += 50_000_000
    probe.observe(imu_packet(stamp, epoch='boot2'), stamp+1_000_000, 13.5)
    assert probe.info(stamp+1_000_000, 13.5)['healthy_for_s'] == 0.
    for k in range(1, 71):
        current = stamp+k*50_000_000
        probe.observe(imu_packet(current, epoch='boot2'), current+1_000_000, 13.5+k*.05)
    info = probe.info(current+1_000_000, 17.)
    assert info['status'] == 'ready' and info['source']['epoch'] == 'boot2'
    probe.observe(imu_packet(current+50_000_000), current+51_000_000, 17.05)
    assert probe.info(current+51_000_000, 17.05)['status'] != 'ready'


def test_visual_restart_does_not_accidentally_reenable_failed_assistance(recovery, monkeypatch):
    card, args, now, seconds = recovery
    visual = degraded_session(card, args, now, seconds, monkeypatch)
    visual.gate.fault('visual degradation', 'visual')
    visual.process.returncode = 0
    images(visual, now[0]); card._recover()
    seconds[0] += 2.
    card._recover()
    new = card.instances['one'][0]
    assert new is not visual and new.imu is None and new.fusion is None
    assert new.requested_calibration['imu_topic'] == '/imu'
    assert new.info()['mode']['degraded']


def test_degraded_probe_overflow_resets_health_without_retiring_visual(recovery, monkeypatch):
    card, args, now, seconds = recovery
    visual = degraded_session(card, args, now, seconds, monkeypatch)
    for _ in range(31):
        healthy_visual_tick(visual, now, seconds)
    assert visual.gate.state == 'tracking'
    message = NS(data=json.dumps(imu_packet(now[0])))
    for _ in range(101):
        visual._enqueue_imu(message)
    with visual.lock:
        visual._drain_imu()
    assert visual.gate.state == 'tracking'
    assert visual.info()['auxiliary_recovery']['status'] != 'ready'
    assert not visual.imu_overflow
    for _ in range(32):
        healthy_visual_tick(visual, now, seconds)
    assert visual.info()['auxiliary_recovery']['status'] == 'ready'
    assert visual.gate.state == 'tracking'
