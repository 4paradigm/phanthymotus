"""Translate one model tool call into a wired absolute EEF target and observe its outcome."""
import json
import math
import threading
import time
from collections import deque
from uuid import uuid4

from ..eef.stream import EefSpec, EefStream, finite, rotate, translate


def _vector(values, size):
    if not isinstance(values, (list, tuple)) or len(values) != size or not all(map(finite, values)):
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


def _near(first, second, linear, angular):
    if len(first) == 1:
        return abs(first[0] - second[0]) <= linear
    return math.dist(first[:3], second[:3]) <= linear and _angle(first[3:], second[3:]) <= angular


def _mean_measurement(values):
    """Average nearby poses with equivalent quaternion signs aligned."""
    mean = [sum(axis) / len(values) for axis in zip(*values)]
    if len(mean) == 7:
        q0 = values[0][3:]
        aligned = [q if sum(a*b for a, b in zip(q0, q)) >= 0 else [-v for v in q]
                   for q in (row[3:] for row in values)]
        mean[3:] = [sum(axis) for axis in zip(*aligned)]
    return mean


def _settled(window, duration_ms):
    """Bound both the local excursion and the drift between early and late samples."""
    if len(window) < 3 or window[-1][0] - window[0][0] < duration_ms:
        return False
    first_stamp, first = window[0]
    if any(not _near(value, first, .002, .025) for _, value in window):
        return False
    third = (window[-1][0] - first_stamp) / 3.
    early = [v for t, v in window if t <= first_stamp + third]
    late = [v for t, v in window if t >= window[-1][0] - third]
    return _near(_mean_measurement(early), _mean_measurement(late), .0007, .008)


class EefInterface(EefSpec):
    """Add model-facing measurements and completion reporting to the negotiated layout."""

    def __init__(self, descriptor):
        super().__init__(descriptor)
        tolerance = descriptor.get("pose_tolerance", {"position_m": .001, "orientation_rad": .01})
        if (not isinstance(tolerance, dict)
                or not all(finite(tolerance.get(k)) and tolerance[k] >= 0
                           for k in ("position_m", "orientation_rad"))):
            raise ValueError("pose_tolerance requires finite nonnegative position_m and orientation_rad")
        self.pose_tolerance = (tolerance["position_m"], tolerance["orientation_rad"])
        self.opening_range_mm = {}
        for side in self.tip_frames:
            i = self.offsets[side, "gripper"].start
            if self.lower[i] < 0:
                raise ValueError("Expected a nonnegative gripper range")
            self.opening_range_mm[side] = [1000. * self.lower[i], 1000. * self.upper[i]]

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


class EefStreamFault(RuntimeError):
    """The stream has stopped publishing until the card is restarted."""


class EefEndpoint:
    """One serialized agent owns the command stream; the node owns the ROS entities."""
    STATE_TIMEOUT = 2.
    MOTION_TIMEOUT = 12.
    HOLD_TIMEOUT = 3.
    SETTLED_MS = 1000
    START_GRACE_MS = 4000
    PREFLIGHT_MS = 250

    def __init__(self, node, state_topic, control_topic, interface):
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        self.interface = interface
        self._message_type = String
        self._condition = threading.Condition()
        self._samples = deque(maxlen=128)
        self._closed = False
        self._target_active = False
        self._baseline = None
        qos = QoSProfile(depth=5, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self._publisher = node.create_publisher(String, control_topic, qos)
        self._stream = EefStream(interface, f"mcp__actucore__autonomous_task/{uuid4().hex}",
                                 self._publish, now_ms=_now_ms)
        node.create_subscription(String, state_topic, self._receive, qos)

    def _publish(self, command):
        message = self._message_type()
        message.data = json.dumps(command, allow_nan=False)
        self._publisher.publish(message)

    @property
    def fault(self):
        return self._stream.fault

    def check_fault(self):
        if self.fault:
            raise EefStreamFault(self.fault)

    def begin_task(self):
        with self._condition:
            self.check_fault()
            self._stream.begin_task()

    def _receive(self, message):
        try:
            packet = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self._condition:
            if self._closed or not self._stream.receive(packet):
                return
            sample = self._stream.sample
            measured = (sample["stamp_ms"], sample["eef"])
            # The state driver can pair a new TCP pose with the same gripper acquisition.
            # Refresh its values without counting it twice toward a settled motion.
            if self._samples and measured[0] == self._samples[-1][0]:
                self._samples[-1] = measured
            else:
                self._samples.append(measured)
            self._condition.notify_all()

    def cancel(self):
        with self._condition:
            self._baseline = None
            result = None
            if self._target_active and not self._closed:
                if not self.fault:
                    result = self._stream.cancel()
                self._target_active = False
            self._condition.notify_all()
            return result

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _latest(self, after, cancel=None, window_ms=0):
        deadline = time.monotonic() + self.STATE_TIMEOUT
        with self._condition:
            while True:
                self.check_fault()
                if self._closed or (cancel is not None and cancel.is_set()):
                    raise InterruptedError("EEF input closed or task cancelled")
                if self._samples:
                    stamp, values = self._samples[-1]
                    if stamp > after and self._stream.fresh(stamp):
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
                    raise TimeoutError("Timed out waiting for fresh advancing EEF state")
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
                "opening_range_mm_by_arm": {side: self.interface.opening_range_mm[side] for side in arms}}

    def _target(self, side, tool, args, start):
        if tool == "set_gripper":
            opening = args.get("opening_mm")
            if not finite(opening):
                raise ValueError("Expected finite gripper opening_mm")
            low, high = self.interface.opening_range_mm[side]
            return [min(high, max(low, opening)) / 1000.]
        if tool != "move_by":
            raise ValueError(f"Unsupported EEF tool: {tool}")
        delta = _vector(args.get("translation_m"), 3)
        rotation = _vector(args.get("rotation_vector_rad"), 3)
        if math.hypot(*delta) > .1 + 1e-9 or math.hypot(*rotation) > math.pi/6 + 1e-9:
            raise ValueError("Use a move of at most 10 cm and 30 degrees")
        index = self.interface.offsets[side, "eef"].start
        lower, upper = self.interface.lower[index:index + 3], self.interface.upper[index:index + 3]
        return translate(start[:3], delta, lower, upper) + rotate(start[3:], rotation)

    def execute(self, name, args, cancel_event):
        with self._condition:
            self.check_fault()
            baseline, self._baseline = self._baseline, None
            side, _, tool = name.partition("_")
            if not baseline or side not in baseline[0] or tool not in ("move_by", "set_gripper"):
                raise RuntimeError("Action is outside the most recent observation or arm mode")
            key = (side, "gripper" if tool == "set_gripper" else "eef")
            sl = self.interface.offsets[key]
            _, current, window = self._latest(baseline[1], cancel_event, self.PREFLIGHT_MS)
            if self._closed or cancel_event.is_set():
                raise InterruptedError("Cancelled before EEF dispatch")
            if (not _near(baseline[2][sl], current[sl], .002, .02)
                    or any(not _near(v[sl], current[sl], .001, .01) for v in window)):
                return {"status": "rejected", "motion_sent": False,
                        "reason": f"{side} {key[1]} changed during inference; observe again"}
            try:
                goal = self._target(side, tool, args, current[sl])
            except (ValueError, TypeError, OverflowError) as exc:
                return {"status": "rejected", "motion_sent": False, "reason": str(exc)}
            if not self._publisher.get_subscription_count():
                raise RuntimeError("No matched downstream EEF control subscriber")
            # The shared stream holds the other arm at its measured reference.
            sent = self._stream.set_target(side, sl, goal)
            self._target_active = True
            command = sent["command"]
            stamp = command["stamp_ms"]
            if sent["status"] != "published":
                self.cancel()
                return {"status": "outcome_unknown", "motion_sent": True, "command_stamp_ms": stamp,
                        "command": command, "reason": sent.get("message", "EEF publication failed")}
        try:
            result = self._feedback(key, stamp, current[sl], goal, cancel_event)
        except (TimeoutError, InterruptedError, EefStreamFault) as exc:
            self.cancel()
            return {"status": "outcome_unknown", "motion_sent": True, "command_stamp_ms": stamp,
                    "command": command, "reason": str(exc)}
        result["command"] = command
        # Keep a closing gripper engaged through the visual check. A stopped-short
        # pose must not resume unexpectedly while the model examines the new images.
        if not result["target_reached"] and not (key[1] == "gripper" and goal[0] < current[sl][0]):
            self.cancel()
        return result

    def _feedback(self, key, command_stamp, start, goal, cancel_event):
        side, kind = key
        sl = self.interface.offsets[key]
        # The jaw's opening tolerance is independent of Cartesian pose accuracy.
        tolerance = self.interface.pose_tolerance if kind == "eef" else (.001, 0.)
        after = phase_start = command_stamp
        deadline = time.monotonic() + self.MOTION_TIMEOUT
        hold_stamp = None
        window = deque()
        count = 0
        moved = False
        while True:
            try:
                stamp, values, _ = self._latest(after, cancel_event)
            except TimeoutError as exc:
                raise TimeoutError(f"EEF command {command_stamp} published; outcome unknown: {exc}") from exc
            if stamp - after > 600:
                window.clear()
            after, actual = stamp, values[sl]
            count += 1
            # Small encoder oscillations before startup do not establish travel.
            moved |= not _near(actual, start, .0015, .015)
            reached = _near(actual, goal, *tolerance)
            window.append((stamp, actual))
            # Keep one point at/before the start of the recent motion window.
            while len(window) > 1 and window[1][0] <= stamp - self.SETTLED_MS:
                window.popleft()
            # An unchanged arm might still be waiting for the actuator to start.
            if ((moved or reached or hold_stamp is not None or stamp - command_stamp >= self.START_GRACE_MS)
                    and _settled(window, self.SETTLED_MS)):
                break
            limit = self.HOLD_TIMEOUT if hold_stamp is not None else self.MOTION_TIMEOUT
            if stamp - phase_start < 1000 * limit and time.monotonic() < deadline:
                continue
            if hold_stamp is not None:
                raise TimeoutError(f"EEF command {command_stamp}: motion continued in fresh feedback "
                                   f"after measured hold {hold_stamp}; last sample {stamp}")
            hold = self.cancel()
            if not hold or hold.get("status") != "published":
                raise TimeoutError(f"EEF command {command_stamp}: could not publish a measured hold")
            # Ignore all pre-hold observations, including delayed hardware samples.
            hold_stamp = hold["command"]["stamp_ms"]
            after = phase_start = max(hold_stamp, stamp)
            deadline = time.monotonic() + self.HOLD_TIMEOUT
            window.clear()

        status = "completed" if reached or moved else "no_motion_observed"
        result = {"status": status, "motion_sent": True, "target_reached": reached,
                  "displacement_observed": moved, "completion_basis": "settled_measured_eef",
                  "command_stamp_ms": command_stamp, "state_stamp_ms": stamp, "feedback_frames": count,
                  "actual": self.interface.measured(side, kind, actual),
                  "target": self.interface.measured(side, kind, goal)}
        if hold_stamp is not None:
            result.update(completion_basis="settled_after_measured_hold", hold_command_stamp_ms=hold_stamp)
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
