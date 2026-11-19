"""Translate the agent's Cartesian actions into the stock TRON2 controller protocol."""
import math
from pathlib import Path

from .kinematics import ARM_LOWER, ARM_UPPER, Kinematics
from .vendor import DEFAULT_URL, Vendor


def _number(value):
    return type(value) in (float, int) and math.isfinite(value)


def _difference(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def _angle(a, b):
    dot = abs(sum(x*y for x, y in zip(a, b)))
    return 2 * math.acos(min(1.0, dot))


class Tron2:
    """One task owns one SDK client. Only its immediately preceding observation can authorize a step."""

    def __init__(self, cfg=None, vendor=None, kinematics=None):
        cfg = cfg or {}
        if kinematics is None:
            model_path = cfg.get("urdf_path") or Path(__file__).with_name("resources") / "tron2a.urdf"
            kinematics = Kinematics(Path(model_path).read_text(encoding="utf-8"))
        self._kinematics = kinematics
        self._vendor = vendor if vendor is not None else Vendor(cfg.get("vendor_url") or DEFAULT_URL)
        self._baseline = None

    def close(self):
        self._baseline = None
        self._vendor.close()

    def _snapshot(self):
        snapshot = self._vendor.snapshot()
        if (any(abs(v) > .05 for v in snapshot["dq"])
                or any(g["speed"] != 0 for g in snapshot["grippers"].values())):
            raise RuntimeError("TRON2 is moving")
        return snapshot

    def observe(self, arms):
        self._baseline = None
        arms = tuple(arms)
        if not arms or any(side not in ("left", "right") for side in arms):
            raise ValueError("Select left and/or right arm")
        snapshot = self._snapshot()
        state = {}
        for side in arms:
            offset = 7 if side == "right" else 0
            q = snapshot["q"][offset:offset+7]
            state[side] = {"positions_rad": q, "eef": self._kinematics.forward(side, q),
                           "gripper": snapshot["grippers"][side]}
        self._baseline = (arms, snapshot)
        return {"arms": state, "robot_timestamp": snapshot["robot_timestamp"]}

    def _move(self, side, tool, args, q, reference):
        duration = args["duration_s"]
        if not _number(duration) or not 1 <= duration <= 60:
            raise ValueError("Duration must be 1–60 seconds")
        offset = 7 if side == "right" else 0
        measured = q[offset:offset+7]
        if tool == "move_by":
            dp, dr = args["translation_m"], args["rotation_vector_rad"]
            if math.hypot(*dp) > .1 or math.hypot(*dr) > math.pi / 6:
                raise ValueError("Requested increment exceeds 10 cm or 30 degrees")
            pose = self._kinematics.delta(side, reference[offset:offset+7], dp, dr, "base")
        else:
            pose = args
        quaternion = pose["quaternion_xyzw"]
        norm = math.hypot(*quaternion)
        if not math.isfinite(norm) or abs(norm - 1) > .001:
            raise ValueError("Expected an xyzw unit quaternion")
        target = {"position_m": pose["position_m"], "quaternion_xyzw": [v/norm for v in quaternion]}
        current = self._kinematics.forward(side, measured)
        if (math.dist(current["position_m"], target["position_m"]) > .1001
                or _angle(current["quaternion_xyzw"], target["quaternion_xyzw"]) > math.pi/6 + .001):
            raise ValueError("Use a smaller move of at most 10 cm and 30 degrees")
        # At a controller stop an encoder can be just outside the command range.
        # Project only the numerical seed. The measured pose above and every
        # joint value sent below still use their respective physical/command bounds.
        seed, changes = self._kinematics.command_seed(side, measured, math.radians(1))
        solved = self._kinematics.solve(side, target["position_m"], target["quaternion_xyzw"], seed)
        joints = list(q[:14])
        joints[offset:offset+7] = solved["joints_rad"]
        jump = _difference(joints, q[:14])
        if jump > math.radians(35) or jump/duration > math.radians(20):
            raise ValueError("IK exceeds 35 degrees per move or 20 degrees/s average joint rate")
        if any(not low <= value <= high for value, low, high in zip(joints, ARM_LOWER, ARM_UPPER)):
            raise ValueError("MoveJ target exceeds joint limits")
        return joints, target, changes

    def execute(self, name, args, cancel_event):
        baseline, self._baseline = self._baseline, None
        side, _, tool = name.partition("_")
        if not baseline or side not in baseline[0] or tool not in ("move_to", "move_by", "set_gripper"):
            raise RuntimeError("Action is outside the most recent observation or arm mode")
        reference = baseline[1]
        current = self._snapshot()
        if current["robot_timestamp"] <= reference["robot_timestamp"]:
            raise RuntimeError("No advancing joint feedback since inference")
        if (_difference(reference["q"], current["q"]) > .035 or any(
                abs(current["grippers"][arm]["opening_raw"] - jaw["opening_raw"]) > 2
                for arm, jaw in reference["grippers"].items())):
            return {"status": "rejected", "motion_sent": False, "reason": "Robot changed during inference"}
        if tool == "set_gripper":
            opening = args.get("opening_mm")
            if not _number(opening) or not 0 <= opening <= 85:
                raise ValueError("Expected nominal opening in [0, 85] mm")
        else:
            try:
                joints, target, changes = self._move(side, tool, args, current["q"], reference["q"])
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                return {"status": "rejected", "motion_sent": False, "reason": str(exc)}
        if cancel_event.is_set():
            raise InterruptedError("Cancelled before dispatch")
        self._vendor.cancel_event = cancel_event

        # A write is issued exactly once. If its outcome is unknown, propagate
        # the error and end this task. Cancellation cannot abandon an admitted move.
        if tool == "set_gripper":
            settled = self._vendor.set_gripper(side, opening)
        else:
            settled = self._vendor.move_j(joints, args["duration_s"], side)
        result = {"status": "completed", "motion_sent": True, "action_id": settled["action_id"],
                  "completion_basis": "vendor_ack_and_settled_feedback",
                  "displacement_observed": settled["displacement_observed"]}
        if tool == "set_gripper":
            result.update(requested_opening_mm=opening, actual=settled["grippers"][side])
        else:
            offset, other = (0, 7) if side == "left" else (7, 0)
            if _difference(settled["q"][other:other+7], current["q"][other:other+7]) > math.radians(.25):
                raise RuntimeError("Unexpected movement of the held arm after MoveJ")
            actual = self._kinematics.forward(side, settled["q"][offset:offset+7])
            start = self._kinematics.forward(side, current["q"][offset:offset+7])
            result.update(actual=actual, target=target,
                          position_error_m=math.dist(actual["position_m"], target["position_m"]),
                          angle_error_rad=_angle(actual["quaternion_xyzw"], target["quaternion_xyzw"]),
                          measured_translation_m=[end - begin for end, begin in
                                                  zip(actual["position_m"], start["position_m"])])
            if changes:
                result["ik_seed_adjustment"] = changes
        return result
