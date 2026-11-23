"""Automatic DWA adoption respects the feedback capability of each action."""

import pytest

from plugins.navi import local_path, policy
from plugins.navi.camera import HorizontalIntrinsics
from plugins.navi.odom import PlanarPose
from test_navi_policy import Sim, seed


def _approach(*, local_planning=False, odom_expected=False, feedback_after=None):
    geometry = local_path.camera_square(HorizontalIntrinsics(160, 85., 80.),
                                        (120, 72., 60.), .20, at_front=True)
    config = policy.Config(vx_max=.15, half_width_m=.20, use_lateral=False,
                           local_planning=local_planning,
                           local_geometry=geometry if local_planning else None,
                           odom_expected=odom_expected, idle_timeout_s=.6)
    scene = Sim(range_m=3., config=config)
    state = seed(policy.State(target="chair"), *scene.observe(), config)
    previous = [0.] * 6
    records = []
    for tick in range(220):
        stamp = 1000 + tick * 100
        objects, depth = scene.observe()
        depth.update(frame_id=stamp, source_frame_id=stamp, capture_ms=stamp,
                     control_ms=stamp)
        odom = None
        if feedback_after is not None and feedback_after <= tick < feedback_after + 6:
            pose = PlanarPose(stamp, ("odom",), *scene.pose)
            odom = dict(motion_pose=pose, obstacle_pose=pose, received_ms=stamp,
                        source_sample_id=stamp, vx=previous[0], vy=previous[1],
                        wz=previous[5], pose_feedback_since_ms=stamp - 100)
        result = policy.step(detections=objects, depth=depth, odom=odom,
                             config=config, state=state, dt=.1,
                             control_period_s=.1)
        records.append((tick, result, state.idle_for_s, scene.relative()[0]))
        scene.apply(result.values, .1)
        previous = result.values or [0.] * 6
        if result.status in (policy.ARRIVED, policy.FAILED):
            break
    return config, state, records


@pytest.mark.parametrize("local_planning", [False, True], ids=["direct", "camera-geometry"])
def test_default_unwired_card_completes_direct_approach(local_planning):
    config, state, records = _approach(local_planning=local_planning)
    assert config.local_controller == "dwa"
    assert records[-1][1].status == policy.ARRIVED, records[-1]
    assert records[-1][0] * .1 > config.idle_timeout_s
    assert not state.dwa_selected
    assert state.motion_source == ("command" if local_planning else "")


def test_expected_feedback_cannot_be_replaced_by_command_progress():
    config, state, records = _approach(odom_expected=True)
    assert records[-1][1].status == policy.FAILED
    assert "no measured navigation progress" in records[-1][1].reason
    assert records[-1][1].values == [0.] * 6
    assert records[-1][0] * .1 < 1.
    assert all(item[1].values[0] > 0. for item in records[:-1])


@pytest.mark.parametrize("feedback_after", [0, 8], ids=["initial-feedback", "late-feedback"])
def test_feedback_loss_keeps_measured_progress_required(feedback_after):
    config, state, records = _approach(feedback_after=feedback_after)
    assert records[-1][1].status == policy.FAILED, records[-1]
    assert "no measured navigation progress" in records[-1][1].reason
    assert records[-1][1].values == [0.] * 6
    # Fresh images and issued commands continue to move this independent
    # plant closer. They cannot replace feedback once the action has used it.
    loss_tick = feedback_after + 6
    after_loss = [row for row in records if loss_tick <= row[0] < records[-1][0]]
    assert len(after_loss) >= 2
    assert all(row[1].values[0] > 0. for row in after_loss)
    assert after_loss[-1][3] < after_loss[0][3]
    assert loss_tick <= records[-1][0] < loss_tick + 7
