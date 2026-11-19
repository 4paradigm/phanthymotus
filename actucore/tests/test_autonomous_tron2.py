"""Exercise the driver boundary with fake MCP and health responses (no hardware)."""
import os
import sys
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plugins.autonomous_task.kinematics import JOINT_NAMES, Kinematics
from plugins.autonomous_task.tron2 import Tron2


def test_kinematics_reads_model_and_solves_from_joint_feedback():
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
    kinematics = Kinematics("".join(urdf) + "</robot>")
    for side in ("left", "right"):
        seed = [-.3, 0, .1, -.6, -.1, .2, -.2]
        target = kinematics.delta(side, seed, [.003, 0, 0], [0, .005, 0], "base")
        solved = kinematics.solve(side, target["position_m"], target["quaternion_xyzw"], seed)
        assert solved["position_error_m"] < .0001
        assert solved["rotation_error_rad"] < .0001


class TestDriverBoundary(unittest.TestCase):
    def setUp(self):
        self.driver = Tron2()
        self.event = threading.Event()
        self.gripper = {"opening_raw": 93, "vendor_timestamp_ms": 20}
        self.before = ([0.] * 16, 10, {"left": self.gripper})
        self.fresh = (self.before[0], 11, self.before[2])
        self.driver._baseline = self.before

    def test_mode_is_enforced_without_touching_hardware(self):
        with patch.object(self.driver, "_request", side_effect=AssertionError):
            with self.assertRaisesRegex(RuntimeError, "arm mode"):
                self.driver.execute("right_set_gripper", {"opening_mm": 0}, self.event)

    def test_changed_pose_discards_plan_before_move(self):
        new_q = [.1] + [0.] * 15
        with patch.object(self.driver, "_snapshot", return_value=(new_q, 11, self.before[2])), \
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

    def test_inflight_cancel_still_checks_closed_jaw_feedback(self):
        def dispatch(name, command):
            self.event.set()
            self.assertEqual(command["side"], "left")
            return {"action_id": "jaw-1"}
        record = {"vendor_acknowledged": True, "motion_sent": True,
                  "target_opening_verified": True, "side": "left"}
        after = {"opening_raw": 1, "vendor_timestamp_ms": 21, "grasp_verified": False}
        with patch.object(self.driver, "_snapshot", return_value=self.fresh), \
                patch.object(self.driver, "_call", side_effect=dispatch), \
                patch.object(self.driver, "_completion", return_value=record) as completion, \
                patch.object(self.driver, "_gripper", return_value=after):
            result = self.driver.execute("left_set_gripper", {"opening_mm": 0}, self.event)
            completion.assert_called_once_with("left", "gripper", "jaw-1", 20)
            self.assertEqual(result["status"], "completed")
            self.assertFalse(result["actual"]["grasp_verified"])

    def test_cannot_count_another_command_or_stationary_miss(self):
        for motion in ({"action_id": "someone-else", "status": "completed"},
                       {"action_id": "ours", "state": "blocked_requires_operator_check",
                        "error": "gripper stopped short of commanded opening"}):
            health = {"grippers": {"gripper_left": {"motion": motion}}}
            with patch.object(self.driver, "_request", return_value=health):
                with self.assertRaises(RuntimeError):
                    self.driver._completion("left", "gripper", "ours", 20)
