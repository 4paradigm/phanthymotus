"""Automatic F2M/F2F selection: one frontend at a time, switch only on
persistent visual degradation with healthy input, always into a new segment."""
import time

import pytest

from plugins import rgbd_odometry as module
from plugins.rgbd_odometry_runtime import estimator_parameters
from test_rgbd_odometry_contract import declarations
from test_rgbd_degradation import feed
from test_rgbd_odometry_recovery import images
from test_rgbd_odometry_status import sessions  # noqa: F401
from vision_stubs import _FakeExecutor


@pytest.fixture
def auto(sessions, monkeypatch):
    _, now, _ = sessions
    seconds = [10.]
    monkeypatch.setattr(time, 'monotonic', lambda: seconds[0])
    card = module.RgbdOdometryPlugin({}, _FakeExecutor())
    args = dict(action='start', instance_id='one', input_topics=['/rgb', '/depth'],
                camera_info=declarations())
    try:
        yield card, args, now, seconds
    finally:
        for session, *_ in card.instances.values():
            session.process.returncode = 0
        card.shutdown()


def start(card, args):
    result = card.dispatch('rgbd_odometry', args)
    assert result['state'] == 'running', result
    return result


def lose(card, now, seconds, kind='visual', advance=2.):
    """Fault the current session, keep inputs fresh, and let supervision act."""
    old = card.instances['one'][0]
    old.process.returncode = 0
    old.gate.fault('visual degradation persisted beyond 1.5 s; restart to begin a new segment', kind)
    images(old, now[0]); card._recover()
    seconds[0] += advance
    now[0] += 1_000_000
    images(old, now[0]); card._recover()
    new = card.instances['one'][0]
    assert new is not old and new.segment != old.segment
    new.process.returncode = None
    return new


def frontend(card):
    return card.dispatch('rgbd_odometry', dict(action='info', instance_id='one'))['frontend']


def test_automatic_selection_starts_with_pinned_default(auto):
    card, args, *_ = auto
    assert card.get_tools()[0]['configSchema']['properties'] == {}
    result = start(card, args)
    assert result['config']['visual_frontend'] == 'auto'
    session = card.instances['one'][0]
    assert session.calibration['visual_frontend'] == 'f2m_feature'  # one frontend runs
    assert estimator_parameters(session.calibration, 'run')['Odom/Strategy'] == '0'
    assert frontend(card) == dict(mode='auto', active='f2m_feature', switches=0,
                                  last_switch_reason=None, decision=None,
                                  cooldown_remaining_s=0., switching_suspended=False)


def test_persistent_visual_degradation_switches_into_a_new_segment(auto):
    card, args, now, seconds = auto
    start(card, args)
    new = lose(card, now, seconds)
    assert new.calibration['visual_frontend'] == 'f2f_optical_flow'
    assert estimator_parameters(new.calibration, 'run')['Odom/Strategy'] == '1'
    state = frontend(card)
    assert state['active'] == 'f2f_optical_flow' and state['switches'] == 1
    assert 'visual degradation persisted' in state['last_switch_reason']
    assert state['cooldown_remaining_s'] > 0
    assert len([s for s in card.instances.values()]) == 1


@pytest.mark.parametrize('kind', ['input', 'output', 'imu', 'clock', 'frame', 'process', 'fusion', 'other'])
def test_input_imu_clock_or_process_failures_never_switch(auto, kind):
    card, args, now, seconds = auto
    start(card, args)
    new = lose(card, now, seconds, kind=kind)
    assert new.calibration['visual_frontend'] == 'f2m_feature'
    assert frontend(card)['switches'] == 0
    assert kind in frontend(card)['decision']


@pytest.mark.parametrize('native_lost,switch', [(True, 'f2f_optical_flow'), (False, None)])
def test_landmark_timeout_switches_only_for_repeated_native_loss(auto, native_lost, switch):
    from test_rgbd_landmark_recovery import anchored, visual

    card, args, _, seconds = auto
    start(card, args)
    session = card.instances['one'][0]
    gate = session.gate = anchored()
    assert gate.status(3_000_000_000) == 'recovering'
    for i in range(3):
        assert visual(gate, 5_000_000_000 + i*100_000_000, lost=native_lost,
                      ids=range(24, 48)) is None
    # Native still answers, so the bounded extension runs before restart.
    assert gate.status(6_001_000_000) == 'recovering'
    assert gate.status(6_800_000_000) == 'lost'
    assert gate.loss_kind == 'relocalization'
    _, selected = card._next_frontend('one', session, {'loss_kind': gate.loss_kind}, seconds[0])
    assert selected == switch


def test_cooldown_and_untracked_switch_limit_prevent_thrashing(auto):
    card, args, now, seconds = auto
    start(card, args)
    assert lose(card, now, seconds).calibration['visual_frontend'] == 'f2f_optical_flow'
    # Within cooldown: ordinary recovery keeps the current frontend.
    assert lose(card, now, seconds).calibration['visual_frontend'] == 'f2f_optical_flow'
    assert 'cooldown' in frontend(card)['decision']
    seconds[0] += 30.
    assert lose(card, now, seconds).calibration['visual_frontend'] == 'f2m_feature'
    seconds[0] += 30.
    # Two switches without any tracking: stop switching until tracking returns.
    assert lose(card, now, seconds).calibration['visual_frontend'] == 'f2m_feature'
    assert frontend(card)['switching_suspended'] is True


def test_tracking_resets_the_untracked_switch_limit(auto):
    card, args, now, seconds = auto
    start(card, args)
    lose(card, now, seconds); seconds[0] += 30.
    lose(card, now, seconds); seconds[0] += 30.
    session = card.instances['one'][0]
    session.gate.state = 'tracking'
    card._recover()
    assert frontend(card)['switching_suspended'] is False
    assert lose(card, now, seconds).calibration['visual_frontend'] == 'f2f_optical_flow'


@pytest.mark.parametrize('manual', ['f2m_feature', 'f2f_optical_flow'])
def test_saved_manual_frontend_still_recovers_automatically(auto, manual):
    card, args, now, seconds = auto
    configured = card.dispatch('rgbd_odometry', dict(action='config', instance_id='one',
                                                      visual_frontend=manual))
    assert configured['state'] == 'configured'
    start(card, args)
    assert frontend(card)['active'] == 'f2m_feature'
    assert lose(card, now, seconds).calibration['visual_frontend'] == 'f2f_optical_flow'
    assert frontend(card)['mode'] == 'auto'


def test_frontend_that_only_reports_lost_can_switch_before_first_tracking(auto):
    card, args, now, seconds = auto
    start(card, args)
    old = card.instances['one'][0]
    initial = now[0]
    for i in range(35):
        seconds[0] += .1
        stamp = initial+i*100_000_000
        now[0] = stamp+20_000_000
        images(old, stamp)
        feed(old, now, stamp, lost=True)
        old._watchdog()
        if old.gate.state == 'lost':
            old.process.returncode = 0  # stubbed process exits before replacement cleanup
        card._recover()
        if card.instances['one'][0] is not old:
            break
    new = card.instances['one'][0]
    assert new is not old and new.segment != old.segment
    assert new.calibration['visual_frontend'] == 'f2f_optical_flow'
    assert frontend(card)['switches'] == 1
