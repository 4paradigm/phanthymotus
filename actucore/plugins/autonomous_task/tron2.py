"""TRON2 endpoint adapter using the existing Motus driver as the only writer.

The model and camera loop know nothing about the joint-space driver. MoveJ
requires both arms in a single packet, so every call holds the unselected arm
at fresh feedback and checks it again after completion.
"""
import http.client
import json
import math
import time
from uuid import uuid4

from .kinematics import ARM_LOWER, ARM_UPPER, JOINT_NAMES, Kinematics


def _number(value):
    return type(value) in (float, int) and math.isfinite(value)


def _difference(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def _angle(a, b):
    dot = abs(sum(x*y for x, y in zip(a, b)))
    return 2 * math.acos(min(1.0, dot))


class Tron2:
    """One instance per task worker; observe() arms an execution exactly once."""

    def __init__(self):
        self._kinematics = None
        self._baseline = None

    @staticmethod
    def _request(method, path, body=None):
        # A motion RPC must never be retried after a transport failure.
        connection = http.client.HTTPConnection("127.0.0.1", 15791, timeout=12)
        try:
            payload = json.dumps(body, allow_nan=False) if body is not None else None
            connection.request(method, path, payload, {"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"TRON2 {path} returned HTTP {response.status}")
            return json.loads(response.read())
        finally:
            connection.close()

    def _call(self, name, args):
        request_id = uuid4().hex
        reply = self._request("POST", "/mcp", {
            "jsonrpc": "2.0", "id": request_id, "method": "tools/call",
            "params": {"name": name, "arguments": args},
        })
        if reply.get("id") != request_id or "result" not in reply:
            raise RuntimeError(f"Invalid TRON2 reply for {name}: {reply.get('error')}")
        result = reply["result"]
        data = json.loads(result["content"][0]["text"])
        if result.get("isError") or "error" in data:
            raise RuntimeError(f"TRON2 {name}: {data.get('error', 'driver error')}")
        return data

    def _joints(self):
        health = self._request("GET", "/health")
        if (health.get("ready") is not True or health.get("robot_connected") is not True
                or health.get("configuration", {}).get("mode") != "upper_limb"):
            raise RuntimeError("TRON2 upper-limb configuration or telemetry unavailable")
        samples = health["samples"]
        sample = samples.get("joints") or samples.get("joint_state") or {}
        data = sample.get("data") or {}
        age = sample.get("reception_age_seconds")
        stamp = data.get("robot_timestamp")
        if (sample.get("available") is not True or not _number(age) or not 0 <= age < 2
                or not _number(stamp) or data.get("position_unit") != "rad"):
            raise RuntimeError("Stale or invalid TRON2 joint feedback")
        q, dq = data.get("q"), data.get("dq")
        if any(not isinstance(v, list) or len(v) != 16 or not all(map(_number, v)) for v in (q, dq)):
            raise RuntimeError("Expected positions and speeds for 14 arm and two head joints")
        # DACH firmware may omit names (official SDK order) or report a
        # permuted named vector. MoveJ always wants SDK order.
        names = data.get("names")
        if names:
            order = JOINT_NAMES + ("head_pitch_Joint", "head_yaw_Joint")
            if len(names) != 16 or set(names) != set(order):
                raise RuntimeError("Unexpected TRON2 joint names")
            q = [q[names.index(name)] for name in order]
            dq = [dq[names.index(name)] for name in order]
        return health, q, dq, stamp

    def _gripper(self, side):
        owner = self._call("gripper", {"action": "info", "side": side})
        data = owner.get("data") or {}
        values = data.get(side) or {}
        age = owner.get("reception_age_seconds")
        if (owner.get("side") != side or owner.get("available") is not True
                or not _number(age) or not 0 <= age < 2
                or not all(_number(values.get(key)) for key in ("opening", "speed", "force"))
                or not _number(data.get("vendor_timestamp_ms"))):
            raise RuntimeError(f"No fresh {side} gripper feedback")
        if (owner.get("state") != "ready" or owner.get("can_accept_motion") is not True
                or owner.get("recovering") is not False or values["speed"] != 0):
            raise RuntimeError(f"{side} gripper is not ready: {owner.get('state')}")
        return {"opening_raw": values["opening"], "opening_mm": values["opening"] * 85 / 94,
                "speed": values["speed"], "force_level": values["force"],
                "vendor_timestamp_ms": data["vendor_timestamp_ms"], "grasp_verified": False}

    def _snapshot(self, arms):
        grippers = {side: self._gripper(side) for side in arms}
        health, q, dq, stamp = self._joints()
        if any(abs(v) > .05 for v in dq):
            raise RuntimeError("TRON2 is moving")
        for key in ("arm_motion", "head_motion", "pose_motion"):
            record = health.get(key) or {}
            if record.get("state") and record.get("state") != "ready":
                raise RuntimeError(f"TRON2 {key} is {record['state']}: {record.get('error', '')}")
        servo = health.get("loco_servo_arm") or {}
        if servo.get("output_active") or servo.get("state") == "running":
            raise RuntimeError("A TRON2 servo controller is active")
        return q, stamp, grippers

    def observe(self, arms):
        self._baseline = None
        arms = tuple(arms)
        if not arms or any(side not in ("left", "right") for side in arms):
            raise ValueError("Select left and/or right arm")
        if self._kinematics is None:
            self._kinematics = Kinematics(self._call("model", {})["urdf"])
        q, stamp, grippers = self._snapshot(arms)
        state = {}
        for side in arms:
            offset = 7 if side == "right" else 0
            state[side] = {"positions_rad": q[offset:offset+7],
                           "eef": self._kinematics.forward(side, q[offset:offset+7]),
                           "gripper": grippers[side]}
        self._baseline = (q, stamp, grippers)
        return {"arms": state, "robot_timestamp": stamp}

    def _move(self, side, tool, args, q, reference):
        duration = args["duration_s"]
        if not _number(duration) or not 1 <= duration <= 60:
            raise ValueError("Duration must be 1–60 seconds")
        offset = 7 if side == "right" else 0
        seed = q[offset:offset+7]
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
        current = self._kinematics.forward(side, seed)
        if (math.dist(current["position_m"], target["position_m"]) > .1001
                or _angle(current["quaternion_xyzw"], target["quaternion_xyzw"]) > math.pi/6 + .001):
            raise ValueError("Use a smaller move of at most 10 cm and 30 degrees")
        solved = self._kinematics.solve(side, target["position_m"], target["quaternion_xyzw"], seed)
        joints = list(q[:14])
        joints[offset:offset+7] = solved["joints_rad"]
        jump = _difference(joints, q)
        if jump > math.radians(35) or jump/duration > math.radians(20):
            raise ValueError("IK exceeds 35 degrees per move or 20 degrees/s average joint rate")
        if any(not low <= value <= high for value, low, high in zip(joints, ARM_LOWER, ARM_UPPER)):
            raise ValueError("MoveJ target exceeds joint limits")
        return {"action": "move", "joint": joints, "time": duration}, target

    def _completion(self, side, driver, action_id, duration):
        deadline = time.monotonic() + duration + 22
        while time.monotonic() < deadline:
            health = self._request("GET", "/health")
            if driver == "arm_control":
                record = health.get("arm_motion") or {}
            else:
                owner = health["grippers"]["gripper_" + side]
                record = owner.get("motion") or {}
            if record.get("action_id") != action_id:
                raise RuntimeError(f"TRON2 lost ownership of {action_id}; outcome unknown")
            if record.get("status") == "error" or record.get("state") == "blocked_requires_operator_check":
                raise RuntimeError(f"TRON2 {action_id} {record.get('error') or record.get('state')}")
            if record.get("status") == "completed" and record.get("callback_delivered") is True:
                if record.get("motion_may_continue") is not False:
                    raise RuntimeError(f"TRON2 {action_id} may still be moving")
                return record
            time.sleep(.2)
        raise RuntimeError(f"TRON2 {action_id} completion timed out; outcome unknown")

    def execute(self, name, args, cancel_event):
        baseline, self._baseline = self._baseline, None
        side, _, tool = name.partition("_")
        if not baseline or side not in baseline[2] or tool not in ("move_to", "move_by", "set_gripper"):
            raise RuntimeError("Action is outside the most recent observation or arm mode")
        reference, observed_stamp, observed = baseline
        q, stamp, grippers = self._snapshot(observed)
        if stamp <= observed_stamp:
            raise RuntimeError("TRON2 joint timestamp did not advance since the observation")
        if (_difference(reference, q) > .035 or any(
                abs(observed[s]["opening_raw"] - g["opening_raw"]) > 2 for s, g in grippers.items())):
            return {"status": "rejected", "motion_sent": False, "reason": "Robot changed during inference"}
        if tool == "set_gripper":
            opening = args.get("opening_mm")
            if not _number(opening) or not 0 <= opening <= 85:
                raise ValueError("Expected nominal opening in [0, 85] mm")
            before = grippers[side]
            driver = "gripper"
            command = {"side": side, "opening_mm": opening, "speed": 15, "force": 30}
            duration = 20
        else:
            driver = "arm_control"
            try:
                command, target = self._move(side, tool, args, q, reference)
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                return {"status": "rejected", "motion_sent": False, "reason": str(exc)}
            duration = command["time"]

        if cancel_event.is_set():
            raise RuntimeError("Cancelled before dispatch")
        try:
            receipt = self._call(driver, command)
        except (OSError, http.client.HTTPException, ValueError, KeyError) as exc:
            raise RuntimeError(f"TRON2 {driver} submission could not be confirmed; no retry: {exc}") from exc
        action_id = receipt.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            raise RuntimeError("TRON2 did not return an action ID; outcome unknown")
        # After admission cancellation only blocks subsequent commands. Keep
        # tracking this ID until the driver releases ownership, even if cancelled.
        try:
            record = self._completion(side, driver, action_id, duration)
        except (OSError, http.client.HTTPException, ValueError, KeyError) as exc:
            raise RuntimeError(f"Lost completion feedback for {action_id}; no retry: {exc}") from exc
        sent = record.get("motion_sent") is True
        if not (sent and record.get("vendor_acknowledged") is True
                or driver == "gripper" and record.get("no_op") is True and not sent):
            raise RuntimeError(f"TRON2 {action_id} lacks confirmed dispatch or gripper no-op")
        result = {"status": "completed", "action_id": action_id, "motion_sent": sent}
        if driver == "gripper":
            after = self._gripper(side)
            actual, goal = after["opening_raw"], opening * 94 / 85
            if (record.get("side") != side or record.get("target_opening_verified") is not True
                    or after["vendor_timestamp_ms"] <= before["vendor_timestamp_ms"]
                    or abs(actual - goal) > 1.000001
                    or (sent and (actual-before["opening_raw"])*(goal-before["opening_raw"]) <= 0)):
                raise RuntimeError(f"TRON2 {action_id} gripper progress/position unverified")
            result.update(actual=after, no_op=not sent, target_opening_mm=opening)
            return result

        _, measured, speeds, new_stamp = self._joints()
        offset, other = (0, 7) if side == "left" else (7, 0)
        if (record.get("arrival_verified") is not True or new_stamp <= stamp
                or any(abs(v) > .05 for v in speeds)
                or _difference(measured[other:other+7], q[other:other+7]) > math.radians(.25)):
            raise RuntimeError(f"TRON2 {action_id} settling or held arm unverified")
        actual = self._kinematics.forward(side, measured[offset:offset+7])
        position_error = math.dist(actual["position_m"], target["position_m"])
        angle_error = _angle(actual["quaternion_xyzw"], target["quaternion_xyzw"])
        result.update(actual=actual, target=target, position_error_m=position_error, angle_error_rad=angle_error)
        start = self._kinematics.forward(side, q[offset:offset+7])
        no_progress = (math.dist(start["position_m"], target["position_m"]) > .0002
                       and math.dist(start["position_m"], actual["position_m"]) < .0002)
        if position_error > .010 or angle_error > math.radians(.5) or no_progress:
            result.update(status="completed_with_warning", reason="Settled; endpoint feedback differs from request")
        return result
