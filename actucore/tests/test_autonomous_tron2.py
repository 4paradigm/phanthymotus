"""Exercise the driver boundary with fake MCP and health responses (no hardware)."""
import math
import os
import sys
import threading
import unittest
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plugins.autonomous_task.kinematics import ARM_LOWER, ARM_UPPER, JOINT_NAMES, Kinematics
from plugins.autonomous_task.tron2 import Tron2, _command_opening_mm


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


def test_only_confirmed_settled_feedback_can_seed_inside_the_command_box(kinematics):
    driver = Tron2()
    driver._kinematics = kinematics
    q = [-.3, .4, .1, -.6, -.1, .2, -.7513] + [0.] * 9
    receipt = {"action_id": "arm-1", "action": "move", "status": "completed",
               "vendor_acknowledged": True, "arrival_verified": True, "motion_may_continue": False,
               "motion_sent": True, "callback_delivered": True, "joint_count": 14,
               "verification": "post_duration_stable_feedback", "verified_samples": 3,
               "tolerance_deg": 1, "max_error_deg": .3}
    args = {"translation_m": [0, .01, 0], "rotation_vector_rad": [0, 0, 0], "duration_s": 4}
    cmd, target, adjusted = driver._move("left", "move_by", args, q, q, receipt)
    assert adjusted["source_action_id"] == "arm-1"
    assert adjusted["corrections"][0]["correction_rad"] == pytest.approx(.0008)
    assert q[6] == -.7513
    assert cmd["joint"][7:] == q[7:14]
    assert all(lo <= v <= hi for v, lo, hi in zip(cmd["joint"], ARM_LOWER, ARM_UPPER))
    assert target["quaternion_xyzw"] == pytest.approx(kinematics.forward("left", q[:7])["quaternion_xyzw"])
    for invalid in ({}, {**receipt, "arrival_verified": False}, {**receipt, "tolerance_deg": 2}):
        with pytest.raises(ValueError, match="verified seed tolerance 0 rad"):
            driver._move("left", "move_by", args, q, q, invalid)
    for side, index, value, message in (("left", 6, -.8, "outside command limit"),
                                        ("left", 6, -4., "URDF mechanical limits"),
                                        ("left", 13, -.7513, "MoveJ target exceeds joint limits")):
        changed = q.copy()
        changed[index] = value
        with pytest.raises(ValueError, match=message):
            driver._move(side, "move_by", args, changed, changed, receipt)


class TestDriverBoundary(unittest.TestCase):
    def setUp(self):
        self.driver = Tron2()
        self.event = threading.Event()
        self.gripper = {"opening_raw": 93, "vendor_timestamp_ms": 20}
        self.before = ([0.] * 16, 10, {"left": self.gripper})
        self.fresh = (self.before[0], 11, self.before[2], {})
        self.driver._baseline = self.before

    def _blocked_owner(self):
        return {"state": "blocked_requires_operator_check", "side": "left", "recovering": False,
                "can_accept_motion": False, "available": True, "reception_age_seconds": .1,
                "data": {"vendor_timestamp_ms": 21, "left": {"speed": 0, "force": 30, "opening": 65}},
                "motion": {"action_id": "open-1", "status": "error", "state": "blocked_requires_operator_check",
                           "side": "left", "action": "open", "error": "60 mm not reached",
                           "error_code": "target_not_reached", "target_opening_mm": 60,
                           "vendor_acknowledged": True, "callback_delivered": True,
                           "motion_sent": True, "no_op": False, "motion_may_continue": True,
                           "target_opening_verified": False, "reported_speed": 0}}

    def test_only_specific_acknowledged_open_is_exposed_as_a_possible_close(self):
        owner = self._blocked_owner()
        with patch.object(self.driver, "_call", return_value=owner) as rpc:
            state = self.driver._gripper("left")
            assert state["close_handover_action_id"] == "open-1"
            assert state["opening_mm"] == pytest.approx(58.7766, abs=.001)
            rpc.assert_called_once_with("gripper", {"action": "info", "side": "left"})
        for fields in ({"vendor_acknowledged": False}, {"action": "close"}, {"callback_delivered": False},
                       {"error_code": "unknown"}, {"reported_speed": 2}):
            changed = {**owner, "motion": {**owner["motion"], **fields}}
            with patch.object(self.driver, "_call", return_value=changed), pytest.raises(RuntimeError):
                self.driver._gripper("left")

    def test_failed_open_remains_a_sent_failure_for_the_next_observation(self):
        owner = self._blocked_owner()
        with patch.object(self.driver, "_snapshot", return_value=self.fresh), \
                patch.object(self.driver, "_call", side_effect=[{"action_id": "open-1"}, owner]) as rpc, \
                patch.object(self.driver, "_request", return_value={"grippers": {"gripper_left": owner}}):
            result = self.driver.execute("left_set_gripper", {"opening_mm": 60}, self.event)
            assert rpc.call_count == 2
            assert rpc.call_args_list[0].args[1]["opening_mm"] * 94 / 85 == pytest.approx(66)
            assert rpc.call_args_list[1].args == ("gripper", {"action": "info", "side": "left"})
        assert result["status"] == "failed" and result["motion_sent"] is True
        assert result["actual"]["close_handover_action_id"] == "open-1"
        assert result["requested_opening_mm"] == 60 and result["target_opening_mm"] == pytest.approx(66*85/94)
        assert result["motion_may_continue"] is True

    def test_blocked_motion_allows_only_explicit_close_with_the_observed_owner(self):
        grippers = {"left": {**self.gripper, "close_handover_action_id": "open-1"}, "right": self.gripper}
        fresh = (self.before[0], 11, grippers, {})
        for name, args in (("left_move_by", {}), ("left_set_gripper", {"opening_mm": .01}),
                           ("right_set_gripper", {"opening_mm": 0})):
            self.driver._baseline = (self.before[0], 10, grippers)
            with patch.object(self.driver, "_snapshot", return_value=fresh), \
                    patch.object(self.driver, "_call", side_effect=AssertionError):
                assert self.driver.execute(name, args, self.event)["motion_sent"] is False
        self.driver._baseline = (self.before[0], 10, grippers)
        replaced = {**grippers, "left": {**grippers["left"], "close_handover_action_id": "another-open"}}
        with patch.object(self.driver, "_snapshot", return_value=(self.before[0], 11, replaced, {})), \
                patch.object(self.driver, "_call", side_effect=AssertionError):
            assert self.driver.execute("left_set_gripper", {"opening_mm": 0}, self.event)["motion_sent"] is False
        self.driver._baseline = (self.before[0], 10, grippers)
        record = {"side": "left", "motion_sent": True, "vendor_acknowledged": True,
                  "previous_action_error": self._blocked_owner()["motion"]}
        after = {"opening_raw": 1, "vendor_timestamp_ms": 22}
        with patch.object(self.driver, "_snapshot", return_value=fresh), \
                patch.object(self.driver, "_call", return_value={"action_id": "close-1"}) as rpc, \
                patch.object(self.driver, "_completion", return_value=record), \
                patch.object(self.driver, "_gripper", return_value=after):
            result = self.driver.execute("left_set_gripper", {"opening_mm": 0}, self.event)
            rpc.assert_called_once_with("gripper", {"side": "left", "opening_mm": 0, "speed": 15, "force": 30})
            assert result["status"] == "completed" and result["previous_open_action_id"] == "open-1"

    def test_nominal_mm_is_quantized_without_turning_nonzero_requests_into_close(self):
        for mm, raw in ((0, 0), (.001, 1), (60, 66), (85, 94)):
            assert _command_opening_mm(mm) == pytest.approx(raw * 85 / 94)

    def test_mode_is_enforced_without_touching_hardware(self):
        with patch.object(self.driver, "_request", side_effect=AssertionError):
            with self.assertRaisesRegex(RuntimeError, "arm mode"):
                self.driver.execute("right_set_gripper", {"opening_mm": 0}, self.event)

    def test_changed_pose_discards_plan_before_move(self):
        new_q = [.1] + [0.] * 15
        with patch.object(self.driver, "_snapshot", return_value=(new_q, 11, self.before[2], {})), \
                patch.object(self.driver, "_call", side_effect=AssertionError):
            result = self.driver.execute("left_move_by", {}, self.event)
        self.assertEqual(result["status"], "rejected")
        self.assertFalse(result["motion_sent"])

    def test_uncertain_write_is_never_repeated(self):
        with patch.object(self.driver, "_snapshot", return_value=self.fresh), \
                patch.object(self.driver, "_call", side_effect=TimeoutError) as rpc:
            with self.assertRaisesRegex(RuntimeError, "no retry"):
                self.driver.execute("left_set_gripper", {"opening_mm": 0}, self.event)
            self.assertEqual(rpc.call_count, 1)
            with self.assertRaisesRegex(RuntimeError, "observation"):
                self.driver.execute("left_set_gripper", {"opening_mm": 0}, self.event)
            self.assertEqual(rpc.call_count, 1)

    def test_cancel_before_submission(self):
        self.event.set()
        with patch.object(self.driver, "_snapshot", return_value=self.fresh), \
                patch.object(self.driver, "_call", side_effect=AssertionError):
            with self.assertRaisesRegex(RuntimeError, "Cancelled"):
                self.driver.execute("left_set_gripper", {"opening_mm": 0}, self.event)

    def test_translation_is_measured_from_immediate_pre_dispatch_feedback(self):
        planned = self.before[0]
        at_dispatch, settled = planned.copy(), planned.copy()
        at_dispatch[0], settled[0] = .005, .028
        # The mock FK uses joint 0 as x so the two possible baselines differ.
        def pose(side, q):
            return {"position_m": [q[0], 0, 0], "quaternion_xyzw": [0, 0, 0, 1]}
        self.driver._kinematics = Mock()
        self.driver._kinematics.forward.side_effect = pose
        verified = {"motion_sent": True, "vendor_acknowledged": True, "arrival_verified": True}
        with patch.object(self.driver, "_snapshot", return_value=(at_dispatch, 11, self.before[2], {})), \
                patch.object(self.driver, "_move", return_value=({"time": 4}, pose("left", [.075]), None)), \
                patch.object(self.driver, "_call", return_value={"action_id": "arm-1"}) as rpc, \
                patch.object(self.driver, "_completion", return_value=verified), \
                patch.object(self.driver, "_joints", return_value=({}, settled, [0.]*16, 15)):
            result = self.driver.execute("left_move_by", {}, self.event)
        assert result["measured_translation_m"] == pytest.approx([.023, 0, 0])
        assert result["position_error_m"] == pytest.approx(.047)
        assert result["status"] == "completed"
        rpc.assert_called_once()

    def test_inflight_cancel_still_checks_closed_jaw_feedback(self):
        def dispatch(name, command):
            self.event.set()
            self.assertEqual(command["side"], "left")
            return {"action_id": "jaw-1"}
        record = {"vendor_acknowledged": True, "motion_sent": True, "side": "left"}
        after = {"opening_raw": 24, "vendor_timestamp_ms": 21, "grasp_verified": False}
        with patch.object(self.driver, "_snapshot", return_value=self.fresh), \
                patch.object(self.driver, "_call", side_effect=dispatch) as rpc, \
                patch.object(self.driver, "_completion", return_value=record) as completion, \
                patch.object(self.driver, "_gripper", return_value=after):
            result = self.driver.execute("left_set_gripper", {"opening_mm": 0}, self.event)
            completion.assert_called_once_with("left", "gripper", "jaw-1", 20)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["actual"]["opening_raw"], 24)
            self.assertFalse(result["actual"]["grasp_verified"])
            rpc.assert_called_once()

    def test_cannot_count_another_command_or_stationary_miss(self):
        for motion in ({"action_id": "someone-else", "status": "completed"},
                       {"action_id": "ours", "state": "blocked_requires_operator_check",
                        "error": "gripper stopped short of commanded opening"}):
            health = {"grippers": {"gripper_left": {"motion": motion}}}
            with patch.object(self.driver, "_request", return_value=health):
                with self.assertRaises(RuntimeError):
                    self.driver._completion("left", "gripper", "ours", 20)
