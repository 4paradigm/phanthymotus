"""Validate the TRON2 card adapter with in-process SDK doubles and local FK/IK."""
import math
import os
import sys
import threading
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plugins.autonomous_task.kinematics import ARM_LOWER, ARM_UPPER, JOINT_NAMES, Kinematics
from plugins.autonomous_task.tron2 import Tron2


@pytest.fixture
def kinematics():
    # A representative serial chain with the driver's joint identities.
    urdf = ['<robot name="test"><link name="base_Link"/>']
    axes = ("0 1 0", "1 0 0", "0 0 1", "0 1 0", "0 0 1", "0 1 0", "1 0 0")
    for side, names in (("L", JOINT_NAMES[:7]), ("R", JOINT_NAMES[7:])):
        parent = "base_Link"
        for name, axis in zip(names, axes):
            child = name + "_link"
            urdf.append(f'<link name="{child}"/><joint name="{name}" type="revolute">'
                        f'<parent link="{parent}"/><child link="{child}"/><axis xyz="{axis}"/>'
                        '<origin xyz="0 0 -.08"/><limit lower="-3.5" upper="3.5"/></joint>')
            parent = child
        urdf.append(f'<link name="grasper_{side}_Link"/><joint name="tip_{side}" type="fixed">'
                    f'<parent link="{parent}"/><child link="grasper_{side}_Link"/></joint>')
    return Kinematics("".join(urdf) + "</robot>")


def test_kinematics_reads_model_and_solves_from_joint_feedback(kinematics):
    for side in ("left", "right"):
        seed = [-.3, 0, .1, -.6, -.1, .2, -.2]
        target = kinematics.delta(side, seed, [.003, 0, 0], [0, .005, 0], "base")
        solved = kinematics.solve(side, target["position_m"], target["quaternion_xyzw"], seed)
        assert solved["position_error_m"] < .0001
        assert solved["rotation_error_rad"] < .0001


def test_grasper_axes_are_rotated_into_base(kinematics):
    for side in ("left", "right"):
        # First joint rotates about Y. At +pi/2, local -Z points along base -X.
        pose = kinematics.forward(side, [math.pi/2, 0, 0, 0, 0, 0, 0])
        assert pose["finger_direction_base"] == pytest.approx([-1, 0, 0])
        assert pose["jaw_opening_axis_base"] == pytest.approx([0, 1, 0])
        pose = kinematics.forward(side, [0, .2, 0, 0, 0, 0, 0])
        assert pose["jaw_opening_axis_base"] == pytest.approx([0, math.cos(.2), math.sin(.2)])


def test_redundant_arm_can_move_without_pushing_wrist_past_its_stop(kinematics):
    seed = [-.3, .4, .1, -.6, -.1, .2, -.7505]
    pose = kinematics.delta("left", seed, [0, .02, 0], [0, 0, 0], "base")
    solved = kinematics.solve("left", pose["position_m"], pose["quaternion_xyzw"], seed)
    assert solved["joints_rad"][6] >= seed[6]
    assert solved["position_error_m"] <= .0001
    assert solved["rotation_error_rad"] <= .0001
    with pytest.raises(ValueError, match="closest command limit at final IK iterate"):
        kinematics.solve("left", [2, 2, 2], pose["quaternion_xyzw"], seed)


def test_slight_encoder_overtravel_only_adjusts_the_local_ik_seed(kinematics):
    adapter = Tron2(vendor=Mock(), kinematics=kinematics)
    measured = [-.3, .4, .1, -.6, -.1, .2, -.7513] + [0.] * 9
    args = {"translation_m": [0, .01, 0], "rotation_vector_rad": [0, 0, 0], "duration_s": 4}
    joints, target, changes = adapter._move("left", "move_by", args, measured, measured)
    assert changes and measured[6] == -.7513
    assert targets_within_joint_limits(joints)
    assert joints[7:] == measured[7:14]
    assert target["quaternion_xyzw"] == pytest.approx(kinematics.forward("left", measured[:7])["quaternion_xyzw"])
    # One degree is a bound on the starting estimate. It neither expands the
    # command box nor permits clipping joints of the arm that was to be held.
    for index, value in ((6, ARM_LOWER[6] - math.radians(1.1)), (6, -4.), (13, -.7513)):
        changed = measured.copy()
        changed[index] = value
        with pytest.raises(ValueError):
            adapter._move("left", "move_by", args, changed, changed)


def targets_within_joint_limits(joints):
    return len(joints) == 14 and all(low <= q <= high for q, low, high in zip(joints, ARM_LOWER, ARM_UPPER))


def sdk_sample(stamp, q=None, raw_width=70):
    if q is None:
        q = [-.3, 0, .1, -.6, -.1, .2, -.2] * 2 + [0., 0.]
    grippers = {side: {"opening_raw": raw_width, "opening_mm": raw_width * 85/94,
                       "force_level": 0, "speed": 0, "vendor_timestamp_ms": stamp}
                for side in ("left", "right")}
    return {"q": list(q), "dq": [0.] * 16, "robot_timestamp": stamp, "grippers": grippers}


@pytest.fixture
def sdk(kinematics):
    vendor = Mock()
    vendor.snapshot.side_effect = [sdk_sample(10), sdk_sample(20)]
    return Tron2(vendor=vendor, kinematics=kinematics), vendor


@pytest.mark.parametrize("arms", [("left",), ("right",), ("left", "right")])
def test_observe_exposes_only_selected_arms_without_motion(sdk, arms):
    adapter, vendor = sdk
    current = adapter.observe(arms)
    assert set(current["arms"]) == set(arms)
    for side in arms:
        assert current["arms"][side]["gripper"]["opening_mm"] == pytest.approx(70 * 85/94)
        assert current["arms"][side]["eef"]["frame"] == "base_Link"
    vendor.move_j.assert_not_called()
    vendor.set_gripper.assert_not_called()


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("tool", ["move_by", "move_to"])
def test_pose_call_holds_other_arm_and_reports_measured_result(sdk, side, tool):
    adapter, vendor = sdk
    measured = sdk_sample(10)["q"]
    at_dispatch = measured.copy()
    at_dispatch[0 if side == "left" else 7] += .002
    vendor.snapshot.side_effect = [sdk_sample(10, measured), sdk_sample(20, at_dispatch)]
    def settled(joints, duration, selected_side):
        assert duration == 4 and selected_side == side
        assert targets_within_joint_limits(joints)
        untouched = slice(7, 14) if side == "left" else slice(0, 7)
        assert joints[untouched] == at_dispatch[untouched]
        # Simulated feedback stopped .01 rad short on one joint. Return the
        # actual pose for the next observation without resending the target.
        q = list(joints) + at_dispatch[14:]
        q[0 if side == "left" else 7] -= .01
        return {**sdk_sample(30, q), "action_id": "sdk-move-1", "displacement_observed": True}
    vendor.move_j.side_effect = settled
    state = adapter.observe((side,))
    assert sorted(state["arms"]) == [side]
    args = {"translation_m": [.003, 0., 0.], "rotation_vector_rad": [0., .005, 0.], "duration_s": 4}
    if tool == "move_to":
        target = adapter._kinematics.delta(side, state["arms"][side]["positions_rad"],
                                           args["translation_m"], args["rotation_vector_rad"], "base")
        args = {"position_m": target["position_m"], "quaternion_xyzw": target["quaternion_xyzw"],
                "duration_s": 4}
    result = adapter.execute(f"{side}_{tool}", args, threading.Event())
    assert result["status"] == "completed" and result["motion_sent"] is True
    assert result["action_id"] == "sdk-move-1"
    assert result["actual"]["position_m"] != result["target"]["position_m"]
    fk_slice = slice(0, 7) if side == "left" else slice(7, 14)
    start = adapter._kinematics.forward(side, at_dispatch[fk_slice])["position_m"]
    assert result["measured_translation_m"] == pytest.approx([e - b for e, b in zip(result["actual"]["position_m"], start)])
    vendor.move_j.assert_called_once()
    vendor.set_gripper.assert_not_called()


def test_stable_contact_short_of_commanded_opening_reports_actual_for_visual_verification(sdk):
    adapter, vendor = sdk
    adapter.observe(("left",))
    vendor.set_gripper.return_value = {**sdk_sample(30, raw_width=12), "action_id": "sdk-jaw-1", "displacement_observed": True}
    result = adapter.execute("left_set_gripper", {"opening_mm": 0}, threading.Event())
    assert result["status"] == "completed" and result["motion_sent"] is True
    assert result["action_id"] == "sdk-jaw-1" and result["displacement_observed"]
    assert result["completion_basis"] == "vendor_ack_and_settled_feedback"
    assert result["actual"]["opening_mm"] == pytest.approx(12 * 85/94)
    assert not result["actual"].get("grasp_verified")
    vendor.set_gripper.assert_called_once_with("left", 0)
    vendor.move_j.assert_not_called()


def test_rejects_state_change_during_inference_before_sending(sdk):
    adapter, vendor = sdk
    first = sdk_sample(10)
    changed = sdk_sample(20)
    changed["q"][0] += .2
    vendor.snapshot.side_effect = [first, changed]
    adapter.observe(("left",))
    result = adapter.execute("left_set_gripper", {"opening_mm": 60}, threading.Event())
    assert result["status"] == "rejected" and result["motion_sent"] is False
    vendor.set_gripper.assert_not_called()
    vendor.move_j.assert_not_called()


def test_before_send_cancel_does_not_submit_sdk_motion(sdk):
    adapter, vendor = sdk
    adapter.observe(("left",))
    cancel = threading.Event()
    cancel.set()
    with pytest.raises((RuntimeError, InterruptedError)):
        adapter.execute("left_set_gripper", {"opening_mm": 0}, cancel)
    vendor.set_gripper.assert_not_called()
    vendor.move_j.assert_not_called()


def test_unselected_arm_cannot_receive_commands(sdk):
    adapter, vendor = sdk
    adapter.observe(("left",))
    with pytest.raises(RuntimeError):
        adapter.execute("right_set_gripper", {"opening_mm": 0}, threading.Event())
    vendor.move_j.assert_not_called()
    vendor.set_gripper.assert_not_called()


def test_unknown_sdk_outcome_is_not_replayed(sdk):
    adapter, vendor = sdk
    adapter.observe(("left",))
    vendor.set_gripper.side_effect = TimeoutError("SDK response timeout")
    with pytest.raises((RuntimeError, TimeoutError)):
        adapter.execute("left_set_gripper", {"opening_mm": 0}, threading.Event())
    with pytest.raises(RuntimeError):
        adapter.execute("left_set_gripper", {"opening_mm": 0}, threading.Event())
    vendor.set_gripper.assert_called_once_with("left", 0)
