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


def _guarded_open(side, owner):
    """A failed open that the driver's explicit close entry can take over."""
    motion = owner.get("motion") or {}
    if (owner.get("side") == side and owner.get("state") == "blocked_requires_operator_check"
            and owner.get("recovering") is False and owner.get("waiting_action") is None
            and owner.get("can_accept_motion") is False
            and motion.get("action_id") and motion.get("side") == side and motion.get("action") == "open"
            and motion.get("state") == "blocked_requires_operator_check" and motion.get("status") == "error"
            and motion.get("error_code") == "target_not_reached" and motion.get("motion_sent") is True
            and motion.get("vendor_acknowledged") is True and motion.get("callback_delivered") is True
            and motion.get("no_op") is False and motion.get("target_opening_verified") is False
            and motion.get("reported_speed") == 0):
        return motion
    return None


def _command_opening_mm(requested):
    # Zero is the driver's distinct explicit-close command. A small positive
    # request must therefore stay on the open pathway, even below half a tick.
    ticks = max(1, math.floor(requested * 94 / 85 + .5)) if requested else 0
    return ticks * 85 / 94


def _settled_tolerance_rad(receipt):
    # TRON2 verifies MoveJ with three stable samples within one degree. A
    # claimed looser threshold must not silently widen this adapter's bound.
    tolerance = receipt.get("tolerance_deg")
    measured = receipt.get("max_error_deg")
    if (receipt.get("action") == "move" and receipt.get("status") == "completed"
            and receipt.get("action_id") and receipt.get("arrival_verified") is True
            and receipt.get("motion_sent") is True and receipt.get("vendor_acknowledged") is True
            and receipt.get("motion_may_continue") is False and receipt.get("callback_delivered") is True
            and receipt.get("verification") == "post_duration_stable_feedback"
            and receipt.get("joint_count") == 14 and _number(receipt.get("verified_samples"))
            and receipt["verified_samples"] >= 3 and _number(tolerance) and _number(measured)
            and 0 <= measured <= tolerance <= 1):
        return math.radians(tolerance)
    return 0.0


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
        motion = _guarded_open(side, owner)
        if (values["speed"] != 0 or owner.get("recovering") is not False or
                (not motion and (owner.get("state") != "ready" or owner.get("can_accept_motion") is not True))):
            raise RuntimeError(f"{side} gripper is not ready: {owner.get('state')}")
        state = {"opening_raw": values["opening"], "opening_mm": values["opening"] * 85 / 94,
                 "speed": values["speed"], "force_level": values["force"],
                 "vendor_timestamp_ms": data["vendor_timestamp_ms"], "grasp_verified": False}
        if motion:
            state.update(close_handover_action_id=motion["action_id"],
                         previous_open_error=motion.get("error") or motion["error_code"],
                         previous_target_opening_mm=motion.get("target_opening_mm"))
        return state

    def _snapshot(self, arms):
        grippers = {side: self._gripper(side) for side in arms}
        health, q, dq, stamp = self._joints()
        for side in ("left", "right"):
            owner = (health.get("grippers") or {}).get("gripper_" + side) or {}
            if side not in grippers and owner.get("state") == "blocked_requires_operator_check":
                raise RuntimeError(f"Unselected {side} gripper is blocked")
        if any(abs(v) > .05 for v in dq):
            raise RuntimeError("TRON2 is moving")
        for key in ("arm_motion", "head_motion", "pose_motion"):
            record = health.get(key) or {}
            if record.get("state") and record.get("state") != "ready":
                raise RuntimeError(f"TRON2 {key} is {record['state']}: {record.get('error', '')}")
        servo = health.get("loco_servo_arm") or {}
        if servo.get("output_active") or servo.get("state") == "running":
            raise RuntimeError("A TRON2 servo controller is active")
        return q, stamp, grippers, health.get("arm_motion") or {}

    def observe(self, arms):
        self._baseline = None
        arms = tuple(arms)
        if not arms or any(side not in ("left", "right") for side in arms):
            raise ValueError("Select left and/or right arm")
        if self._kinematics is None:
            self._kinematics = Kinematics(self._call("model", {})["urdf"])
        q, stamp, grippers, _ = self._snapshot(arms)
        state = {}
        for side in arms:
            offset = 7 if side == "right" else 0
            state[side] = {"positions_rad": q[offset:offset+7],
                           "eef": self._kinematics.forward(side, q[offset:offset+7]),
                           "gripper": grippers[side]}
        self._baseline = (q, stamp, grippers)
        return {"arms": state, "robot_timestamp": stamp}

    def _move(self, side, tool, args, q, reference, receipt):
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
        tolerance = _settled_tolerance_rad(receipt)
        seed, changes = self._kinematics.command_seed(side, measured, tolerance)
        solved = self._kinematics.solve(side, target["position_m"], target["quaternion_xyzw"], seed)
        joints = list(q[:14])
        joints[offset:offset+7] = solved["joints_rad"]
        jump = _difference(joints, q)
        if jump > math.radians(35) or jump/duration > math.radians(20):
            raise ValueError("IK exceeds 35 degrees per move or 20 degrees/s average joint rate")
        if any(not low <= value <= high for value, low, high in zip(joints, ARM_LOWER, ARM_UPPER)):
            raise ValueError("MoveJ target exceeds joint limits")
        seed_info = ({"corrections": changes, "tolerance_rad": tolerance,
                      "source_action_id": receipt["action_id"]} if changes else None)
        return {"action": "move", "joint": joints, "time": duration}, target, seed_info

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
                if driver == "gripper" and _guarded_open(side, owner):
                    return record
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
        q, stamp, grippers, previous_motion = self._snapshot(observed)
        if stamp <= observed_stamp:
            raise RuntimeError("TRON2 joint timestamp did not advance since the observation")
        handovers = {s: g["close_handover_action_id"] for s, g in grippers.items()
                     if g.get("close_handover_action_id")}
        previous = {s: g["close_handover_action_id"] for s, g in observed.items()
                    if g.get("close_handover_action_id")}
        if handovers != previous:
            return {"status": "rejected", "motion_sent": False, "reason": "Gripper ownership changed during inference"}
        if handovers and not (side in handovers and tool == "set_gripper" and args.get("opening_mm") == 0):
            return {"status": "rejected", "motion_sent": False, "reason":
                    "A failed open retains ownership. Only an explicit zero-width close on its side "
                    "can request the driver's guarded handover; inspect the new images first."}
        if (_difference(reference, q) > .035 or any(
                abs(observed[s]["opening_raw"] - g["opening_raw"]) > 2 for s, g in grippers.items())):
            return {"status": "rejected", "motion_sent": False, "reason": "Robot changed during inference"}
        if tool == "set_gripper":
            opening = args.get("opening_mm")
            if not _number(opening) or not 0 <= opening <= 85:
                raise ValueError("Expected nominal opening in [0, 85] mm")
            before = grippers[side]
            driver = "gripper"
            command = {"side": side, "opening_mm": _command_opening_mm(opening), "speed": 15, "force": 30}
            duration = 20
        else:
            driver = "arm_control"
            try:
                command, target, seed_info = self._move(side, tool, args, q, reference, previous_motion)
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                return {"status": "rejected", "motion_sent": False, "reason": str(exc),
                        "planning_feedback": "No movement. Reassess the fresh views and measured "
                                             "finger direction; a smaller step or a different "
                                             "orientation with visible clearance may be reachable."}
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
            if record.get("side") != side:
                raise RuntimeError(f"TRON2 {action_id} reported the wrong gripper")
            after = self._gripper(side)
            if sent and after["vendor_timestamp_ms"] <= before["vendor_timestamp_ms"]:
                raise RuntimeError(f"TRON2 {action_id} has no new gripper feedback")
            if side in handovers:
                prior = record.get("previous_action_error") or {}
                if prior.get("action_id") != handovers[side] or prior.get("error_code") != "target_not_reached":
                    raise RuntimeError(f"TRON2 {action_id} guarded close did not confirm its previous owner")
                result["previous_open_action_id"] = prior["action_id"]
            if record.get("status") == "error":
                if after.get("close_handover_action_id") != action_id:
                    raise RuntimeError(f"TRON2 {action_id} failed opening no longer owns the gripper")
                result.update(status="failed", reason=after["previous_open_error"],
                              error_code=record["error_code"], motion_may_continue=record.get("motion_may_continue"),
                              close_handover_action_id=action_id)
            result.update(actual=after, no_op=not sent, requested_opening_mm=opening,
                          target_opening_mm=command["opening_mm"])
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
        if seed_info:
            result["ik_seed_adjustment"] = seed_info
        start = self._kinematics.forward(side, q[offset:offset+7])
        result["measured_translation_m"] = [end - begin for end, begin in
                                            zip(actual["position_m"], start["position_m"])]
        return result
