"""Feedback-driven EEF control shared by the manual card and model harness."""
import copy
import math
import threading
import time

from .. import control_stream


def _now_ms():
    return time.time_ns() // 1_000_000


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def translate(start, delta, lower, upper):
    """Clip the segment at the first Cartesian box face, preserving its direction."""
    goal = [a + d for a, d in zip(start, delta)]
    if all(lo <= v <= hi for v, lo, hi in zip(goal, lower, upper)):
        return goal
    if not all(lo <= v <= hi for v, lo, hi in zip(start, lower, upper)):
        raise ValueError("TCP is outside the workspace; the requested target must be inside")
    fraction = 1.
    for a, d, v, lo, hi in zip(start, delta, goal, lower, upper):
        if v < lo or v > hi:
            fraction = min(fraction, ((lo if v < lo else hi) - a) / d)
    return [min(hi, max(lo, a + d * fraction)) for a, d, lo, hi in zip(start, delta, lower, upper)]


def rotate(quaternion, rotvec):
    """Left-multiply by the rotation about the reference-frame axes."""
    angle = math.hypot(*rotvec)
    scale = math.sin(angle / 2) / angle if angle else .5
    x, y, z = (v * scale for v in rotvec)
    w = math.cos(angle / 2)
    a, b, c, d = quaternion
    q = (w*a + x*d + y*c - z*b, w*b - x*c + y*d + z*a,
         w*c + x*b - y*a + z*d, w*d - x*a - y*b - z*c)
    norm = math.hypot(*q)
    return [v / norm for v in q]


class EefSpec:
    """Negotiate the controlled TCP and opening groups from the downstream descriptor."""

    def __init__(self, descriptor):
        if (not isinstance(descriptor, dict) or descriptor.get("control_interface") != control_stream.SCHEMA
                or descriptor.get("mode") != "eef_pose"):
            raise ValueError("Connect a motus.control/1 eef_pose actuator")
        self.frame = descriptor.get("frame")
        if not isinstance(self.frame, str) or not self.frame.strip():
            raise ValueError("The actuator must declare a reference frame")
        state = descriptor.get("state")
        if not isinstance(state, dict) or state.get("schema") != "motus.eef/1" or state.get("field") != "eef":
            raise ValueError("The actuator must declare motus.eef/1 feedback in eef")
        self.dof = descriptor.get("dof")
        if type(self.dof) is not int or self.dof not in (8, 16):
            raise ValueError("Expected one TCP and one opening per arm")
        groups = descriptor.get("groups")
        if not isinstance(groups, list):
            raise ValueError("Missing EEF control groups")
        expected = {f"{kind}_{side[0]}": (side, kind, count, unit, mode)
                    for side in ("left", "right") for kind, count, unit, mode in
                    (("eef", 7, "m+quat_xyzw", "eef_pose"), ("gripper", 1, "m", "joint_position"))}
        self.offsets = {}
        used = set()
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("name"), str) or group["name"] not in expected:
                raise ValueError("Unknown or duplicate EEF group")
            side, kind, count, unit, mode = expected.pop(group["name"])
            offset = group.get("offset")
            if (type(offset) is not int or not 0 <= offset <= self.dof - count
                    or type(group.get("count")) is not int or group["count"] != count
                    or group.get("unit") != unit or group.get("advisory") or group.get("mode", mode) != mode):
                raise ValueError(f"Invalid actuator group {group['name']}")
            indices = range(offset, offset + count)
            if used.intersection(indices):
                raise ValueError("Overlapping EEF groups")
            used.update(indices)
            self.offsets[side, kind] = slice(offset, offset + count)
        sides = [side for side in ("left", "right") if (side, "eef") in self.offsets]
        if (len(used) != self.dof or len(sides) * 8 != self.dof
                or any((side, "gripper") not in self.offsets for side in sides)):
            raise ValueError("Expected a complete TCP and opening for each arm")
        tips = descriptor.get("tip_frames", {side: f"eef_{side[0]}" for side in sides})
        if (not isinstance(tips, dict) or set(tips) != set(sides)
                or not all(isinstance(t, str) and t.strip() for t in tips.values())):
            raise ValueError("tip_frames must identify each connected TCP")
        self.tip_frames = dict(tips)
        limits = descriptor.get("limits")
        if not isinstance(limits, dict):
            raise ValueError("Missing actuator limits")
        bounds = [limits.get(key) for key in ("lower", "upper")]
        if (not all(isinstance(b, (list, tuple)) and len(b) == self.dof and all(map(finite, b)) for b in bounds)
                or any(lo > hi for lo, hi in zip(*bounds))):
            raise ValueError("Expected finite ordered actuator bounds for every component")
        self.lower, self.upper = map(list, bounds)
        rate = descriptor.get("rate")
        if (not isinstance(rate, dict) or not all(finite(rate.get(k)) and rate[k] > 0 for k in
                                                   ("expected_hz", "watchdog_ms", "max_obs_age_ms"))):
            raise ValueError("Missing control rate, watchdog or observation age")
        self.max_obs_age_ms = rate["max_obs_age_ms"]
        self.ttl_ms = control_stream.ttl_ms(rate["expected_hz"], descriptor)
        self.descriptor = copy.deepcopy(descriptor)

    def accepts(self, data):
        if (not isinstance(data, dict) or data.keys() != {"schema", "frame", "stamp_ms", "eef"}
                or data["schema"] != "motus.eef/1" or data["frame"] != self.frame):
            return False
        values = data["eef"]
        if not isinstance(values, list) or len(values) != self.dof or not all(map(finite, values)):
            return False
        for side in self.tip_frames:
            pose = values[self.offsets[side, "eef"]]
            if abs(math.hypot(*pose[3:]) - 1.) > .001 or values[self.offsets[side, "gripper"].start] < 0:
                return False
        return True


class EefStream:
    """Construct a full absolute command from each fresh measured frame."""

    PEER_POSITION_LIMIT_M = .010
    PEER_ANGLE_LIMIT_RAD = math.radians(5.)

    def __init__(self, descriptor, source, publish, now_ms=None, *, initial_seq=0):
        self.spec = descriptor if isinstance(descriptor, EefSpec) else EefSpec(descriptor)
        self._source = source
        self._publish = publish
        self._now_ms = now_ms or _now_ms
        self._lock = threading.RLock()
        self.seq = initial_seq
        self.sample = self.last = self._target = None
        self._selected_arm = self._held_peer = None
        self._drift_stamp = self.fault = None

    def fresh(self, stamp):
        return type(stamp) is int and -100 <= self._now_ms() - stamp <= self.spec.max_obs_age_ms

    def begin_task(self):
        """Capture a new peer reference on the first command of a new task."""
        with self._lock:
            if self.fault:
                raise RuntimeError(self.fault)
            self._selected_arm = None

    def receive(self, data):
        """A repeated capture time can carry a new pose paired with the same jaw sample."""
        with self._lock:
            try:
                if (not self.spec.accepts(data) or not self.fresh(data["stamp_ms"])
                        or (self.sample and data["stamp_ms"] < self.sample["stamp_ms"])):
                    return False
            except (ValueError, TypeError, OverflowError):
                return False
            self.sample = {**data, "eef": data["eef"].copy()}
            if self._held_peer is not None and not self.fault:
                self._check_held_peer()
            if not self.fault:
                self._send()
            return True

    def _check_held_peer(self):
        side, held, _ = self._held_peer
        measured = self.sample["eef"][self.spec.offsets[side, "eef"]]
        distance = math.dist(held[:3], measured[:3])
        a, b = held[3:], measured[3:]
        dot = abs(sum(x*y for x, y in zip(a, b))) / (math.hypot(*a) * math.hypot(*b))
        angle = 2 * math.acos(min(1., dot))
        stamp = self.sample["stamp_ms"]
        if distance <= self.PEER_POSITION_LIMIT_M and angle <= self.PEER_ANGLE_LIMIT_RAD:
            self._drift_stamp = None
        elif self._drift_stamp is None:
            self._drift_stamp = stamp
        elif stamp > self._drift_stamp:
            self.fault = (f"Unselected {side} TCP moved {distance*1000:.1f} mm, "
                          f"{math.degrees(angle):.1f} degrees from its measured hold; "
                          "EEF output stopped. Stop and restart the card to resume.")
            self._target = None

    def set_target(self, arm, group, goal):
        """Select one arm's TCP, gripper, or both. An empty group restores measured echo."""
        with self._lock:
            if self.fault:
                raise RuntimeError(self.fault)
            if not self.sample or not self.fresh(self.sample["stamp_ms"]):
                raise ValueError("Wait for fresh eef_state feedback")
            if arm not in self.spec.tip_frames:
                raise ValueError("Requested arm is missing from the EEF descriptor")
            groups = (group,) if isinstance(group, slice) else tuple(group)
            pose, jaw = (self.spec.offsets[arm, kind] for kind in ("eef", "gripper"))
            if (groups not in ((), (pose,), (jaw,), (pose, jaw))
                    or len(goal) != sum(s.stop - s.start for s in groups) or not all(map(finite, goal))):
                raise ValueError("Expected one absolute target value for each selected EEF component")
            if pose in groups and abs(math.hypot(*goal[3:7]) - 1.) > .001:
                raise ValueError("Expected a unit target quaternion")
            if jaw in groups and not self.spec.lower[jaw.start] <= goal[-1] <= self.spec.upper[jaw.start]:
                raise ValueError("Gripper target is outside the declared limits")
            if not groups or arm != self._selected_arm:
                self._held_peer = None
                self._drift_stamp = None
                self._selected_arm = arm if groups else None
                if groups and len(self.spec.tip_frames) == 2:
                    peer = "right" if arm == "left" else "left"
                    values = self.sample["eef"]
                    self._held_peer = (peer, values[self.spec.offsets[peer, "eef"]],
                                       values[self.spec.offsets[peer, "gripper"]])
            self._target = (arm, groups, list(goal)) if groups else None
            self.last = None
            return self._send()

    def cancel(self):
        with self._lock:
            if self.fault:
                return {"state": "error", "message": self.fault}
            if self._target is not None:
                arm, groups, _ = self._target
                self._target = (arm, groups, None)
            self.last = None
            if self.sample and self.fresh(self.sample["stamp_ms"]):
                return self._send()
            return {"state": "running", "status": "waiting_for_state"}

    def _send(self):
        observed = self.sample["stamp_ms"]
        values = self.sample["eef"].copy()
        if self._held_peer is not None:
            side, pose, opening = self._held_peer
            values[self.spec.offsets[side, "eef"]] = pose
            values[self.spec.offsets[side, "gripper"]] = opening
        arm, origin = None, observed
        if self._target is not None:
            arm, groups, goal = self._target
            if goal is None:  # Latch once, including the first fresh feedback after cancellation.
                goal = [v for group in groups for v in values[group]]
                self._target = (arm, groups, goal)
            index = 0
            for group in groups:
                size = group.stop - group.start
                values[group] = goal[index:index + size]
                index += size
            if self.last:
                origin = self.last["state_stamp_ms"]
        count = self.last["publish_count"] if self.last and self.last["state_stamp_ms"] == origin else 0
        self.seq += 1
        command = control_stream.build(seq=self.seq, values=values, mode="eef_pose", dof=self.spec.dof,
                                       source=self._source, stamp_ms=self._now_ms(), obs_stamp_ms=observed,
                                       ttl_ms=self.spec.ttl_ms)
        command["frame"] = self.spec.frame
        self.last = {"arm": arm, "state_stamp_ms": origin, "command": command, "publish_count": count}
        try:
            self._publish(command)
        except RuntimeError as error:
            self.last.update(status="publish_error", message=str(error))
        else:
            self.last.update(status="published", publish_count=count + 1)
        return {"state": "running", **self.last}
