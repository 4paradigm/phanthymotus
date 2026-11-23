"""DWA keeps measured obstacles registered across optional absolute-pose delivery."""
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from plugins.navi import active_scan, local_path, plugin, policy
from plugins.navi.odom import PlanarPose
from test_navi_dwa_integration import observation_case


WORLD = ("odom", "base_link", "body", 4, "cal", 7, 3)


def measured_case(*, body=False):
    config, state, original, original_odom = observation_case(body=body, attempted=False)
    history = []

    def sample(index, *, absolute=False, frame=WORLD, offset_ms=0):
        stamp = 1000 + 100 * index
        history.append((stamp, stamp, (.1, 0., 0.)))
        depth = dict(original, frame_id=stamp, capture_ms=stamp, control_ms=stamp, source_frame_id=stamp)
        odom = dict(original_odom, received_ms=stamp, source_sample_id=stamp,
                    velocity_history=tuple(history))
        if absolute:
            # The body moves along world +y with its forward axis rotated 90 degrees.
            pose = PlanarPose(stamp + offset_ms, frame, 10., 20. + index * .01, math.pi / 2)
            odom.update(obstacle_pose=pose, motion_pose=pose)
            depth['obstacle_capture_pose'] = pose
        return depth, odom

    def step(index, **kwargs):
        depth, odom = sample(index, **kwargs)
        decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                               state=state, dt=.1, dwa_admission_ms=100)
        return decision, depth, odom

    return config, state, sample, step


@pytest.mark.parametrize("body", [False, True], ids=["camera", "body"])
@pytest.mark.parametrize("first_absolute", [False, True], ids=["appears", "disappears"])
def test_continuous_measured_motion_keeps_dwa_after_absolute_pose_handoff(body, first_absolute):
    _, state, _, step = measured_case(body=body)
    for index in range(12):
        result, _, _ = step(index, absolute=first_absolute if index < 4 else not first_absolute)
        assert result.dwa_acceptance is not None, (index, state.dwa_trace['fallback_reason'])
        assert result.values[0] > 0.
        assert state.active_scan.registered_history or index == 0


@pytest.mark.parametrize("field", [0, 1, 3, 4, 5, 6])
def test_absolute_frame_reset_cannot_relabel_relative_occupied_history(field):
    _, state, _, step = measured_case()
    assert step(0)[0].dwa_acceptance
    assert step(1, absolute=True)[0].dwa_acceptance
    changed = list(WORLD)
    changed[field] = str(changed[field]) + "-reset"
    result, _, _ = step(2, absolute=True, frame=tuple(changed))
    assert result.dwa_acceptance is None
    assert result.values == [0.] * 6
    assert state.dwa_trace['fallback_reason'] == 'geometry_mismatch'


@pytest.mark.parametrize("offset_ms", [-1000, 1000])
def test_uncovered_absolute_time_cannot_register_old_relative_obstacles(offset_ms):
    _, state, _, step = measured_case()
    assert step(0)[0].dwa_acceptance
    result, _, _ = step(1, absolute=True, offset_ms=offset_ms)
    assert result.dwa_acceptance is None
    assert state.dwa_trace['fallback_reason'] == 'geometry_mismatch'


@pytest.mark.parametrize("timer", [False, True], ids=["direct", "timer"])
@pytest.mark.parametrize("blocked", [False, True])
def test_first_send_registers_new_pose_and_rechecks_current_obstacle(monkeypatch, timer, blocked):
    config, state, sample, _ = measured_case()
    initial_depth, initial_odom = sample(0)
    clock = [1.]
    snapshot = [None, initial_depth, initial_odom, 0]
    packets = []
    card = plugin.NaviPlugin({}, executor=None)
    card._config, card._state, card._running = config, state, True
    card._descriptor = {"rate": {"max_obs_age_ms": 800}}
    card._ttl_ms = 400
    card._command_timer = object() if timer else None
    card._publisher = SimpleNamespace(publish=lambda message: packets.append(json.loads(message.data)))
    monkeypatch.setattr(plugin.time, 'time', lambda: clock[0])
    monkeypatch.setattr(plugin.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(card, '_observation_snapshot', lambda **_: tuple(snapshot))
    message = card.next_command()
    assert message and message['values'][0] > 0. and state.dwa_selected
    original_deadline = card._dwa_for_message(message).acceptance.motion_guard.deadline_ms
    depth, odom = sample(1, absolute=True)
    if blocked:
        depth['map'] = np.full_like(depth['map'], .06)
    snapshot[:] = [None, depth, odom, 0]
    clock[0] = 1.1
    if timer:
        assert card._stage_command(message, card._output_epoch)
    else:
        card._publish_command(message)
    assert len(packets) == 1
    if blocked:
        assert packets[0]['values'] == [0.] * 6
    else:
        assert packets[0]['values'] == message['values']
        assert packets[0]['stamp_ms'] + packets[0]['ttl_ms'] <= original_deadline


def test_registered_camera_front_offset_is_applied_once():
    relative = PlanarPose(1000, ('navi_relative', object()), 0., 0., 0.)
    absolute = PlanarPose(1000, WORLD, 10., 20., math.pi / 2)
    geometry = observation_case(body=False)[0].local_geometry
    origin = replace(relative, x=.20)
    current = replace(absolute, stamp_ms=1100, y=20.30)
    result = policy.dwa_registration(origin, current, 1000, 1100, geometry,
                                     ((.1, 0., 0.),), reference=(relative, absolute))
    assert result.points((1., .3)) == pytest.approx((.9, .3))
    assert result.shift_error == 0.
    assert result.turn_error == 0.


@pytest.mark.parametrize("interrupt", ["gap", "backwards", "missing", "history_missing"])
def test_relative_history_after_feedback_break_cannot_reuse_old_registration(interrupt):
    config, state, sample, step = measured_case()
    assert step(0)[0].dwa_acceptance
    assert step(1, absolute=True)[0].dwa_acceptance
    old_origin = state.active_scan.registered_history[0][5][1]
    before = state.active_scan.pose_registration
    depth, odom = sample(5 if interrupt == 'gap' else 2, absolute=True)
    if interrupt == 'backwards':
        odom['source_sample_id'] = 900
        odom['velocity_history'] = ((odom['received_ms'], 900, (.1, 0., 0.)),)
    elif interrupt == 'missing':
        odom['vx'] = None
        odom['pose_feedback'] = None
    elif interrupt == 'history_missing':
        odom['velocity_history'] = (*odom['velocity_history'][:-1],
                                    (1150, 1150, None), odom['velocity_history'][-1])
    result = policy.step(detections=None, depth=depth, odom=odom, config=config,
                         state=state, dt=.1, dwa_admission_ms=100)
    assert result.dwa_acceptance is None
    assert state.active_scan.pose_registration is not before
    sources = policy.dwa_scene_sources(state.active_scan, depth['control_ms'])
    reference = policy.dwa_pose_reference(sources, odom)
    with pytest.raises(policy.DwaFrameMismatch):
        policy.dwa_registration(old_origin, odom['obstacle_pose'], 1000, depth['control_ms'],
                                config.local_geometry, ((.1, 0., 0.),), reference=reference)


@pytest.mark.parametrize("through_policy", [False, True], ids=["scan", "policy"])
def test_active_scan_cannot_keep_relative_registration_after_missing_current_feedback(through_policy):
    config, state, sample, step = measured_case()
    assert step(0)[0].dwa_acceptance
    assert step(1, absolute=True)[0].dwa_acceptance
    scan = state.active_scan
    assert scan.begin((1.,))
    old_origin = scan.registered_history[0][5][1]
    depth, odom = sample(2, absolute=True)
    odom['vx'] = odom['pose_feedback'] = None
    if through_policy:
        odom.pop('velocity_history')
        stopped = policy.step(detections=None, depth=depth, odom=odom, config=config,
                              state=state, dt=.1, dwa_admission_ms=100)
    else:
        stopped = scan.update(depth, odom, config.local_geometry, receipt_only=True)
    assert stopped is not None
    assert scan.pose_registration is None
    assert not scan.poses
    with pytest.raises(policy.DwaFrameMismatch):
        policy.dwa_registration(old_origin, odom['obstacle_pose'], 1000, depth['control_ms'],
                                config.local_geometry, ((.1, 0., 0.),),
                                reference=policy.dwa_pose_reference(scan, odom))


@pytest.mark.parametrize("occupied", [False, True])
def test_registered_free_evidence_clears_only_a_fully_free_original_square(occupied):
    relative = PlanarPose(1000, ('navi_relative', object()), 0., 0., 0.)
    absolute = PlanarPose(1000, WORLD, 10., 20., math.pi / 2)
    before = replace(relative, x=.20)
    current = replace(absolute, stamp_ms=1100, y=20.20)
    squares = local_path.ObstacleSquares(np.array([[.5, 0.]]), 0.)
    clear = np.isclose(local_path.FORWARD, .5) & np.isclose(local_path.RIGHT, 0.)
    hits = clear if occupied else np.zeros_like(clear)
    result = policy._retire_confirmed((1000, before, squares), current, clear, hits, (relative, absolute))
    if occupied:
        assert result is not None and len(result[2].centers) == 1
    else:
        assert result is None
