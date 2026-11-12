"""Optional auto F2M/F2F selection: one frontend at a time, switch only on
persistent visual degradation with healthy input, always into a new segment."""
import time

import pytest

from plugins import rgbd_odometry as module
from plugins.rgbd_odometry_runtime import estimator_parameters
from test_rgbd_odometry_contract import declarations
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


def start(card, args, frontend):
    card.dispatch('rgbd_odometry', dict(action='config', instance_id='one', visual_frontend=frontend))
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


def test_auto_is_explicit_optional_and_default_is_unchanged(auto):
    card, args, *_ = auto
    schema = card.get_tools()[0]['configSchema']['properties']['visual_frontend']
    assert schema['enum'] == ['f2m_feature', 'f2f_optical_flow', 'auto']
    assert schema['default'] == 'f2m_feature'
    result = start(card, args, 'auto')
    assert result['config']['visual_frontend'] == 'auto'
    session = card.instances['one'][0]
    assert session.calibration['visual_frontend'] == 'f2m_feature'  # one frontend runs
    assert estimator_parameters(session.calibration, 'run')['Odom/Strategy'] == '0'
    assert frontend(card) == dict(mode='auto', active='f2m_feature', switches=0,
                                  last_switch_reason=None, decision=None,
                                  cooldown_remaining_s=0., switching_suspended=False)


def test_persistent_visual_degradation_switches_into_a_new_segment(auto):
    card, args, now, seconds = auto
    start(card, args, 'auto')
    new = lose(card, now, seconds)
    assert new.calibration['visual_frontend'] == 'f2f_optical_flow'
    assert estimator_parameters(new.calibration, 'run')['Odom/Strategy'] == '1'
    state = frontend(card)
    assert state['active'] == 'f2f_optical_flow' and state['switches'] == 1
    assert 'visual degradation persisted' in state['last_switch_reason']
    assert state['cooldown_remaining_s'] > 0
    assert len([s for s in card.instances.values()]) == 1


@pytest.mark.parametrize('kind', ['input', 'imu', 'clock', 'frame', 'process', 'fusion', 'other'])
def test_input_imu_clock_or_process_failures_never_switch(auto, kind):
    card, args, now, seconds = auto
    start(card, args, 'auto')
    new = lose(card, now, seconds, kind=kind)
    assert new.calibration['visual_frontend'] == 'f2m_feature'
    assert frontend(card)['switches'] == 0
    assert kind in frontend(card)['decision']


def test_cooldown_and_untracked_switch_limit_prevent_thrashing(auto):
    card, args, now, seconds = auto
    start(card, args, 'auto')
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
    start(card, args, 'auto')
    lose(card, now, seconds); seconds[0] += 30.
    lose(card, now, seconds); seconds[0] += 30.
    session = card.instances['one'][0]
    session.gate.state = 'tracking'
    card._recover()
    assert frontend(card)['switching_suspended'] is False
    assert lose(card, now, seconds).calibration['visual_frontend'] == 'f2f_optical_flow'


@pytest.mark.parametrize('manual', ['f2m_feature', 'f2f_optical_flow'])
def test_manual_frontends_never_switch(auto, manual):
    card, args, now, seconds = auto
    start(card, args, manual)
    assert lose(card, now, seconds).calibration['visual_frontend'] == manual
    assert frontend(card)['mode'] == 'manual'


def test_changing_to_or_from_auto_still_requires_stop(auto):
    card, args, *_ = auto
    start(card, args, 'auto')
    assert card.dispatch('rgbd_odometry', dict(action='config', instance_id='one',
                                               visual_frontend='f2m_feature'))['state'] == 'error'
