"""Translate one model tool call into a wired absolute EEF target and observe its outcome."""
import copy
import json
import math
import threading
import time
from collections import deque


def _number(value):
    return type(value) in (float, int) and math.isfinite(value)


def _vector(values, size):
    if not isinstance(values, (list, tuple)) or len(values) != size or not all(map(_number, values)):
        raise ValueError(f"Expected {size} finite numbers")
    return list(values)


def _unit(q):
    length = math.hypot(*q)
    if abs(length - 1.) > .001:
        raise ValueError("Expected a unit xyzw quaternion")
    return [v / length for v in q]


def _angle(a, b):
    dot = sum(x*y for x, y in zip(a, b)) / (math.hypot(*a) * math.hypot(*b))
    return 2 * math.acos(min(1., abs(dot)))


def _delta_pose(start, translation, rotation):
    # Left-multiply by Exp(rotation): the model specifies base-frame axes about this TCP.
    theta = math.hypot(*rotation)
    scale = math.sin(theta / 2) / theta if theta else .5
    x, y, z = [v * scale for v in rotation]
    w = math.cos(theta / 2)
    a, b, c, d = start[3:]
    quat = [w*a + x*d + y*c - z*b, w*b - x*c + y*d + z*a,
            w*c + x*b - y*a + z*d, w*d - x*a - y*b - z*c]
    return [a+b for a, b in zip(start[:3], translation)] + _unit(quat)


def _near(first, second, linear, angular):
    if len(first) == 1:
        return abs(first[0] - second[0]) <= linear
    return math.dist(first[:3], second[:3]) <= linear and _angle(first[3:], second[3:]) <= angular


class EefInterface:
    """The servo descriptor determines group offsets, TCP names, units and jaw limits."""

    def __init__(self, descriptor):
        if not isinstance(descriptor, dict) or descriptor.get("mode") != "eef_pose":
            raise ValueError("Expected the downstream servo's eef_pose control_interface")
        self.schema = descriptor.get("control_interface")
        self.frame = descriptor.get("frame")
        if not all(isinstance(v, str) and v.strip() for v in (self.schema, self.frame)):
            raise ValueError("EEF control_interface must declare a schema and reference frame")
        tips = descriptor.get("tip_frames")
        if (not isinstance(tips, dict) or not tips or set(tips) - {"left", "right"}
                or not all(isinstance(v, str) and v.strip() for v in tips.values())):
            raise ValueError("Expected named left and/or right TCP tip_frames")
        self.tip_frames = dict(tips)
        self.opening_m = tuple(_vector(descriptor.get("opening_m"), 2))
        if not 0 <= self.opening_m[0] < self.opening_m[1]:
            raise ValueError("Invalid gripper opening_m range")
        self.dof = descriptor.get("dof")
        if type(self.dof) is not int or self.dof != len(tips) * 8:
            raise ValueError("EEF dof must cover seven pose values and one opening per arm")
        groups = descriptor.get("groups")
        if not isinstance(groups, list):
            raise ValueError("Missing EEF control groups")
        expected = {f"{kind}_{side[0]}": (side, kind, size, unit)
                    for side in tips for kind, size, unit in (("eef", 7, "m+quat_xyzw"), ("gripper", 1, "m"))}
        self.offsets = {}
        used = set()
        for group in groups:
            if not isinstance(group, dict) or group.get("name") not in expected:
                raise ValueError("Unknown or duplicate EEF control group")
            side, kind, size, unit = expected.pop(group["name"])
            offset = group.get("offset")
            if (group.get("unit") != unit or type(group.get("count")) is not int
                    or group["count"] != size or type(offset) is not int or not 0 <= offset <= self.dof - size):
                raise ValueError(f"Invalid offset, count or unit for {group['name']}")
            indices = range(offset, offset + size)
            if used.intersection(indices):
                raise ValueError("Overlapping EEF control groups")
            used.update(indices)
            self.offsets[side, kind] = slice(offset, offset + size)
        if expected:
            raise ValueError("Missing EEF control groups")
        self.descriptor = copy.deepcopy(descriptor)

    def __eq__(self, other):
        return isinstance(other, EefInterface) and self.descriptor == other.descriptor

    def decode(self, message):
        if not isinstance(message, str) or len(message) > 8192:
            raise ValueError("Expected a JSON EEF state packet")
        data = json.loads(message)
        if not isinstance(data, dict) or data.get("schema") != self.schema or data.get("frame") != self.frame:
            raise ValueError("EEF state schema/frame differs from the connected servo")
        stamp = data.get("stamp_ms")
        if type(stamp) is not int or not -100 <= _now_ms() - stamp <= 1000:
            raise ValueError("Stale or invalid EEF state timestamp")
        values = _vector(data.get("values"), self.dof)
        for side in self.tip_frames:
            _unit(values[self.offsets[side, "eef"]][3:])
            if values[self.offsets[side, "gripper"]][0] < 0:
                raise ValueError("Invalid measured gripper opening")
        return stamp, values

    def measured(self, side, kind, values):
        if kind == "gripper":
            return {"opening_mm": values[0] * 1000.}
        tip = self.tip_frames[side]
        x, y, z, w = _unit(values[3:])
        # Columns of the TCP-to-reference rotation matrix, in reference-frame xyz.
        axes = {"x": [1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)],
                "y": [2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x)],
                "z": [2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)]}
        return {"frame": self.frame, "eef_frame": tip, "reference_label": f"TCP {tip} in {self.frame}",
                "position_m": values[:3], "quaternion_xyzw": values[3:], "axes_in_reference": axes}


def _now_ms():
    return time.time_ns() // 1_000_000


class EefEndpoint:
    """One serialized agent owns the command stream; the node owns the ROS entities."""
    STATE_TIMEOUT = 2.
    MOTION_TIMEOUT = 32.
    SETTLED_MS = 500
    SHORTFALL_MS = 4000
    PREFLIGHT_MS = 250

    def __init__(self, node, state_topic, control_topic, interface):
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        self.interface = interface
        self._message_type = String
        self._condition = threading.Condition()
        self._samples = deque(maxlen=128)
        self._closed = False
        self._last_error = None
        self._baseline = None
        self._last_command = 0
        self._uncertain = set()
        # A closing target still applying grip must survive an unrelated pose command.
        # Likewise preserve our own in-flight targets if taking another action.
        self._goals = {}
        qos = QoSProfile(depth=5, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self._publisher = node.create_publisher(String, control_topic, qos)
        node.create_subscription(String, state_topic, self._receive, qos)

    def _receive(self, message):
        try:
            sample = self.interface.decode(message.data)
        except (ValueError, TypeError, OverflowError) as exc:
            with self._condition:
                self._last_error = str(exc)
            return
        with self._condition:
            if self._closed or (self._samples and sample[0] <= self._samples[-1][0]):
                return
            self._samples.append(sample)
            self._last_error = None
            self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _latest(self, after, cancel=None, window_ms=0):
        deadline = time.monotonic() + self.STATE_TIMEOUT
        with self._condition:
            while True:
                if self._closed or (cancel is not None and cancel.is_set()):
                    raise InterruptedError("EEF input closed or task cancelled")
                if self._samples:
                    stamp, values = self._samples[-1]
                    if stamp > after and -100 <= _now_ms() - stamp <= 1000:
                        if not window_ms:
                            return stamp, values, [values]
                        if len(self._samples) >= 3:
                            cutoff = min(stamp - window_ms, self._samples[-3][0])
                            anchor = next((s for s, _ in reversed(self._samples) if s <= cutoff), None)
                            if anchor is not None:
                                # A resumed stream must accumulate fresh history after a gap.
                                samples = [(s, v) for s, v in self._samples if s >= anchor]
                                if all(b[0] - a[0] <= 600 for a, b in zip(samples, samples[1:])):
                                    return stamp, values, [v for _, v in samples]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reason = self._last_error or "no fresh advancing measured frames"
                    raise TimeoutError(f"Timed out waiting for EEF state: {reason}")
                self._condition.wait(min(.1, remaining))

    def observe(self, arms, cancel=None):
        self._baseline = None
        arms = tuple(arms)
        if not arms or set(arms) - self.interface.tip_frames.keys():
            raise ValueError("Requested arms are missing from the connected EEF interface")
        # Allow for measurement latency: wait for a packet newer than the cached one,
        # not necessarily for a hardware measurement timestamp newer than this call.
        with self._condition:
            after = self._samples[-1][0] if self._samples else 0
        stamp, values, _ = self._latest(after, cancel)
        self._baseline = (arms, stamp, values)
        state = {side: {kind: self.interface.measured(side, kind, values[self.interface.offsets[side, kind]])
                        for kind in ("eef", "gripper")} for side in arms}
        return {"stamp_ms": stamp, "arms": state, "control_interface": self.interface.descriptor,
                "opening_range_mm": [v * 1000. for v in self.interface.opening_m]}

    def _target(self, tool, args, start):
        if tool == "set_gripper":
            opening = args.get("opening_mm")
            low, high = self.interface.opening_m
            if not _number(opening) or not low * 1000. <= opening <= high * 1000.:
                raise ValueError("Gripper opening_mm outside the downstream servo limits")
            return [opening / 1000.]
        if tool == "move_by":
            dp = _vector(args.get("translation_m"), 3)
            dr = _vector(args.get("rotation_vector_rad"), 3)
            if math.hypot(*dr) > math.pi/6 + 1e-9:
                raise ValueError("Use a rotation of at most 30 degrees")
            goal = _delta_pose(start, dp, dr)
        else:
            goal = _vector(args.get("position_m"), 3) + _unit(_vector(args.get("quaternion_xyzw"), 4))
        if math.dist(start[:3], goal[:3]) > .1 + 1e-9 or _angle(start[3:], goal[3:]) > math.pi/6 + 1e-9:
            raise ValueError("Use a move of at most 10 cm and 30 degrees")
        return goal

    def execute(self, name, args, cancel_event):
        baseline, self._baseline = self._baseline, None
        side, _, tool = name.partition("_")
        if not baseline or side not in baseline[0] or tool not in ("move_to", "move_by", "set_gripper"):
            raise RuntimeError("Action is outside the most recent observation or arm mode")
        key = (side, "gripper" if tool == "set_gripper" else "eef")
        sl = self.interface.offsets[key]
        try:
            goal = self._target(tool, args, baseline[2][sl])
        except (ValueError, TypeError, OverflowError) as exc:
            return {"status": "rejected", "motion_sent": False, "reason": str(exc)}
        window_ms = self.SETTLED_MS if self._uncertain else self.PREFLIGHT_MS
        _, current, window = self._latest(baseline[1], cancel_event, window_ms)
        values = list(current)
        for group, indices in self.interface.offsets.items():
            if group != key and group in self._goals:
                values[indices] = self._goals[group]
                continue
            if (not _near(baseline[2][indices], current[indices], .002, .02)
                    or any(not _near(v[indices], current[indices], .001, .01) for v in window)):
                return {"status": "rejected", "motion_sent": False,
                        "reason": f"{group[0]} {group[1]} changed during inference; observe again"}
            if group[1] == "gripper":
                low, high = self.interface.opening_m
                values[indices] = [min(high, max(low, current[indices][0]))]
        values[sl] = goal
        if not self._publisher.get_subscription_count():
            raise RuntimeError("No matched downstream EEF control subscriber")
        now = _now_ms()
        stamp = max(now, self._last_command + 1)
        if stamp > now + 100:
            raise RuntimeError("Clock moved backwards since the last EEF command")
        command = {"schema": self.interface.schema, "frame": self.interface.frame, "stamp_ms": stamp, "values": values}
        message = self._message_type()
        message.data = json.dumps(command, allow_nan=False)
        repeated_hold = key[1] == "gripper" and self._goals.get(key) == goal
        with self._condition:
            if self._closed or cancel_event.is_set():
                raise InterruptedError("Cancelled before EEF dispatch")
            self._last_command = stamp
            self._publisher.publish(message)  # Publish exactly once; the state stream carries no command ACK.
            self._uncertain.clear()
            self._goals[key] = goal
        try:
            result = self._feedback(key, stamp, current[sl], goal)
        except (TimeoutError, InterruptedError) as exc:
            # Replaying this absolute goal in another arm's frame could restart an
            # unobserved movement. Resume only from newly observed stationary state.
            self._goals.pop(key, None)
            self._uncertain.add(key)
            return {"status": "outcome_unknown", "motion_sent": True, "command_stamp_ms": stamp,
                    "command": command, "reason": str(exc)}
        result["command"] = command
        # A measured short close may be sustained contact. Continue to request the
        # same closing target until another jaw action even while moving this arm.
        closing_contact = (key[1] == "gripper" and result["displacement_observed"]
                           and goal[0] * 1000. < result["actual"]["opening_mm"] < current[sl][0] * 1000.)
        if not (closing_contact or repeated_hold):
            self._goals.pop(key, None)
        return result

    def _feedback(self, key, command_stamp, start, goal):
        side, kind = key
        sl = self.interface.offsets[key]
        after = anchor_stamp = command_stamp
        anchor = start
        count = steady_frames = 0
        moved = False
        deadline = time.monotonic() + self.MOTION_TIMEOUT
        while time.monotonic() < deadline:
            try:
                stamp, values, _ = self._latest(after)
            except TimeoutError as exc:
                raise TimeoutError(f"EEF command {command_stamp} published; outcome unknown: {exc}") from exc
            gap = stamp - after
            after, actual = stamp, values[sl]
            count += 1
            moved |= not _near(actual, start, .0005, .006)
            reached = _near(actual, goal, .001, .02)
            if count == 1 or gap > 600 or not _near(actual, anchor, .00035, .004):
                anchor, anchor_stamp, steady_frames = actual, stamp, 0
            steady_frames += 1
            wait_ms = self.SETTLED_MS if reached else self.SHORTFALL_MS
            if steady_frames >= 3 and stamp - anchor_stamp >= wait_ms:
                status = "completed" if reached or moved else "no_motion_observed"
                result = {"status": status, "motion_sent": True, "target_reached": reached,
                          "displacement_observed": moved, "completion_basis": "settled_measured_eef",
                          "command_stamp_ms": command_stamp, "state_stamp_ms": stamp, "feedback_frames": count,
                          "actual": self.interface.measured(side, kind, actual),
                          "target": self.interface.measured(side, kind, goal)}
                if not reached:
                    result["reason"] = ("Measured motion settled short of the target" if moved else
                                        "No requested motion observed in advancing EEF feedback")
                if kind == "gripper":
                    result.update(requested_opening_mm=goal[0]*1000., opening_error_mm=(actual[0]-goal[0])*1000.)
                else:
                    result.update(position_error_m=math.dist(actual[:3], goal[:3]),
                                  angle_error_rad=_angle(actual[3:], goal[3:]),
                                  measured_translation_m=[end-begin for end, begin in zip(actual[:3], start[:3])])
                return result
        raise TimeoutError(f"EEF command {command_stamp} published; motion did not settle before timeout")
