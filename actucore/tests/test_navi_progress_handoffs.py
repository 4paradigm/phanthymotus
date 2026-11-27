"""Measured progress must survive a short observation hold on an old corridor.

The policy, tracker, depth projection and odometry accounting are real. A
recorded outward traversal establishes the old passage; a later absolute fix
after a feedback gap starts the return interval without inventing travel.
"""

import numpy as np
import pytest

from plugins.navi import local_path, policy
from plugins.navi.camera import HorizontalIntrinsics
from plugins.navi.odom import PlanarPose
from test_navi_policy import Sim, seed


def _return_over_recorded_corridor(observation, *, local_planning=True, motion="toward-target"):
    geometry = local_path.camera_square(HorizontalIntrinsics(160, 85., 80.),
                                        (120, 72., 60.), .20, at_front=True)
    config = policy.Config(local_controller="dwa", local_planning=local_planning,
                           local_geometry=geometry if local_planning else None, half_width_m=.20,
                           vx_max=.15625, use_lateral=False, vy_max=0.,
                           idle_timeout_s=2.)
    scene = Sim(range_m=4., config=config)
    state = seed(policy.State(target="chair"), *scene.observe(), config)
    original = state.tracker.track
    stamp = 1000
    previous = [0.] * 6

    def tick(*, hold=False, elapsed=.1, stationary=False):
        nonlocal stamp
        stamp += round(elapsed * 1000)
        pose = PlanarPose(stamp, ("odom",), *scene.pose)
        feedback = [0.] * 6 if stationary else previous
        odom = dict(motion_pose=pose, obstacle_pose=pose, received_ms=stamp,
                    source_sample_id=stamp, vx=feedback[0], vy=feedback[1],
                    wz=feedback[5], pose_feedback_since_ms=stamp - 100)
        objects, depth = scene.observe()
        depth.update(map=np.full((120, 160), scene.relative()[0], dtype=np.float32),
                     frame_id=stamp, source_frame_id=stamp, capture_ms=stamp,
                     control_ms=stamp, obstacle_capture_pose=pose)
        if hold:
            if observation == "missing-depth":
                depth = None
            elif observation == "invalid-depth":
                depth["map"][:] = 0.
        result = policy.step(detections=objects, depth=depth, odom=odom,
                             config=config, state=state, dt=elapsed,
                             control_period_s=.1)
        assert state.tracker.track is original and original.generation == 1
        if not hold:
            assert policy.current_target_observed(state)
            assert original.measured_range_m == pytest.approx(scene.relative()[0], abs=.001)
        return result

    # Eight native commands move the body 12.5 cm through this corridor.
    # The measured 6.25 cm and 12.5 cm checkpoints are now old passage.
    for index in range(9):
        scene.pose[0] = index / 64.
        result = tick()
        assert result.status == policy.APPROACHING
        assert result.values[0] == pytest.approx(.15625)
        previous = result.values
    assert scene.pose[0] == pytest.approx(.125) and state.idle_for_s == 0.

    # A later absolute fix records the body back at the start, after an
    # uncovered 800 ms interval. The normal policy rebaselines this sample;
    # neither the unmeasured return nor the old cells earn fresh progress.
    scene.pose[0] = 0.
    previous = [0.] * 6
    assert tick(elapsed=.8).status == policy.APPROACHING
    assert state.idle_for_s == pytest.approx(.8)
    records = []
    for index in range(24):
        if motion == "target-closes-alone":
            scene.target[0] -= previous[0] * .1
        else:
            scene.apply(previous, .1)
            if motion == "target-recedes":
                scene.target[0] += previous[0] * .1
        hold = observation != "continuous" and index % 2 == 1
        result = tick(hold=hold, stationary=motion == "target-closes-alone")
        records.append((index, float(scene.pose[0]), result.status, state.idle_for_s))
        if result.status == policy.FAILED:
            break
        if hold:
            assert result.values is None or not any(result.values)
        previous = result.values or [0.] * 6
    return result, state, records


@pytest.mark.parametrize("observation", ["continuous", "missing-depth", "invalid-depth"])
@pytest.mark.parametrize("local_planning", [False, True], ids=["direct", "local"])
def test_observation_hold_retains_measured_return_progress(observation, local_planning):
    result, state, records = _return_over_recorded_corridor(observation, local_planning=local_planning)
    assert result.status != policy.FAILED, records
    assert len(records) == 24
    # The body and each fresh target measurement agree that it is getting
    # closer. No command is issued during a missing or invalid observation.
    assert any(idle == 0. for index, x, status, idle in records if x <= .125)
    assert state.idle_for_s < 1.


@pytest.mark.parametrize("motion", ["target-closes-alone", "target-recedes"])
def test_observation_recovery_needs_body_and_target_closure(motion):
    result, state, records = _return_over_recorded_corridor("missing-depth", motion=motion)
    assert result.status == policy.FAILED
    assert result.values == [0.] * 6
    assert all(idle > 0. for index, x, status, idle in records)
    assert state.idle_for_s >= 2.
