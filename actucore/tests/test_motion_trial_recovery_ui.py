"""Trial recovery through the real sandbox, bridge and Motion plugin."""
import pytest

from plugins.motion.client import MotionClient, ServiceError
from test_motion_unified_ui import unified_browser


def begin_trial(frame, pw):
    frame.locator('#motionPrimary').click()
    pw.expect(frame.locator('#sim-sessionState')).to_have_text('已暂停')
    pw.expect(frame.locator('#sim-renderedView')).to_be_visible()


def choose_old_version(frame, pw):
    frame.locator('#breadcrumbActions').click()
    frame.locator('.version-row').filter(has_text='版本 1 ·').click()
    pw.expect(frame.locator('#motionModel')).to_have_value('walk-old')


def writes(state, action):
    return [args for name, args in state['calls'] if name == action]


def test_failed_trial_can_reset_run_and_switch_version(unified_browser, monkeypatch):
    _, frame, state, _, _, pw = unified_browser
    original = MotionClient.call

    def remote(client, action, **args):
        if action == 'simulation_reset':
            # Match SimulationSession._reset_data: reset clears the failure.
            state['sessions'][args['session_id']]['error'] = None
        return original(client, action, **args)

    monkeypatch.setattr(MotionClient, 'call', remote)
    begin_trial(frame, pw)
    frame.locator('#sim-run').click()
    pw.expect(frame.locator('#sim-sessionState')).to_have_text('运行中')
    state['sessions']['owned-0'].update(state='error', error='policy target exceeds robot joint limits')
    frame.locator('#sim-refreshView').click()
    pw.expect(frame.locator('#sim-run')).to_be_disabled()
    frame.locator('#sim-reset').click()
    pw.expect(frame.locator('#sim-sessionState')).to_have_text('已暂停')
    pw.expect(frame.locator('#sim-run')).to_be_enabled()
    assert state['sessions']['owned-0']['error'] is None
    frame.locator('#sim-run').click()
    pw.expect(frame.locator('#sim-sessionState')).to_have_text('运行中')
    choose_old_version(frame, pw)
    frame.locator('#motionPrimary').click()
    pw.expect(frame.locator('#sim-title')).to_contain_text('版本 1')
    assert writes(state, 'simulation_create')[-1]['model_id'] == 'walk-old'
    assert state['sessions']['owned-0']['state'] == 'paused'
    assert not writes(state, 'simulation_close')


@pytest.mark.parametrize('owner', [None, 'another-window'])
def test_return_after_lease_expiry_or_takeover_can_switch_version(unified_browser, owner):
    _, frame, state, _, _, pw = unified_browser
    begin_trial(frame, pw)
    choose_old_version(frame, pw)
    previous = state['sessions']['owned-0']
    # The server has advanced its lease clock while the action page has stopped
    # preview polling. The iframe still has the former is_controller=True copy.
    previous['control'].update(owner_view_id=owner, epoch=2, is_controller=False, expires_in_s=0)
    previous['state'] = 'running' if owner else 'paused'
    before = len(state['calls'])
    frame.locator('#motionPrimary').click()
    pw.expect(frame.locator('#sim-title')).to_contain_text('版本 1')
    assert writes(state, 'simulation_create')[-1]['model_id'] == 'walk-old'
    assert not any(action in {'simulation_release', 'simulation_pause', 'simulation_close'}
                   for action, _ in state['calls'][before:])
    assert previous['state'] == ('running' if owner else 'paused')
    assert previous['control']['owner_view_id'] == owner


def test_release_race_rechecks_owner_before_detaching(unified_browser, monkeypatch):
    _, frame, state, _, _, pw = unified_browser
    begin_trial(frame, pw)
    choose_old_version(frame, pw)
    original = MotionClient.call

    def remote(client, action, **args):
        if action == 'simulation_release' and args['session_id'] == 'owned-0':
            state['calls'].append((action, args))
            old = state['sessions']['owned-0']
            old['control'].update(owner_view_id='another-window', epoch=2, is_controller=False)
            old['state'] = 'running'
            raise ServiceError('仿真控制权已变化，请刷新状态后接管控制')
        return original(client, action, **args)

    monkeypatch.setattr(MotionClient, 'call', remote)
    before = len(state['calls'])
    frame.locator('#motionPrimary').click()
    pw.expect(frame.locator('#sim-title')).to_contain_text('版本 1')
    actions = [action for action, _ in state['calls'][before:]]
    release = actions.index('simulation_release')
    assert 'simulation_inspect' in actions[release + 1:]
    assert actions.index('simulation_create') > release
    assert state['sessions']['owned-0']['state'] == 'running'
    assert state['sessions']['owned-0']['control']['owner_view_id'] == 'another-window'
    assert not writes(state, 'simulation_close')


@pytest.mark.parametrize('failure', ['服务连接失败', '仿真控制权已变化，请刷新状态后接管控制'])
def test_unconfirmed_release_keeps_binding_and_retry_can_recover(unified_browser, monkeypatch, failure):
    _, frame, state, _, _, pw = unified_browser
    begin_trial(frame, pw)
    choose_old_version(frame, pw)
    original = MotionClient.call
    failing = [True]

    def remote(client, action, **args):
        if failing[0] and action == 'simulation_release' and args['session_id'] == 'owned-0':
            state['calls'].append((action, args))
            raise ServiceError(failure)
        return original(client, action, **args)

    monkeypatch.setattr(MotionClient, 'call', remote)
    frame.locator('#motionPrimary').click()
    pw.expect(frame.locator('#status')).to_have_text(failure)
    assert len(writes(state, 'simulation_create')) == 1
    assert not writes(state, 'simulation_close')
    assert state['sessions']['owned-0']['control']['is_controller']
    failing[0] = False
    choose_old_version(frame, pw)
    frame.locator('#motionPrimary').click()
    pw.expect(frame.locator('#sim-title')).to_contain_text('版本 1')
    assert writes(state, 'simulation_create')[-1]['model_id'] == 'walk-old'
