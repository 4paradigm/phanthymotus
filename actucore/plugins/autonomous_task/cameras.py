"""Bind optional Motus RGB inputs and take one fresh multi-view observation."""
from __future__ import annotations

import threading
import time
from functools import partial


_LABELS = {"head": "头部", "left_wrist": "左腕", "right_wrist": "右腕"}

def _mount(identifier):
    name = str(identifier or "").rsplit("/", 1)[-1]
    for role in _LABELS:
        if name in (role, role + "_camera") or name.endswith("_" + role + "_camera"):
            return role
    return None


def camera_ports(binding=None):
    """Keep all three optional ports in their declared order."""
    binding = binding or {}
    return [{"format": "image/jpeg", "desc": f"{label} RGB（可选）",
             **({"topic": binding[role]} if role in binding else {})}
            for role, label in _LABELS.items()]


def resolve_inputs(args, cfg):
    """Agent Core supplies connected topics without their destination port indices."""
    topics = args.get("input_topics") or []
    if not isinstance(topics, (list, tuple)):
        raise ValueError("input_topics must be a list of topic names")
    topics = list(topics)
    if args.get("input_topic"):
        topics.append(args["input_topic"])
    if any(not isinstance(t, str) or not t.startswith("/") for t in topics):
        raise ValueError("Expected absolute ROS input topic names")
    topics = list(dict.fromkeys(topics))
    declarations = args.get("camera_info") or {}
    if not isinstance(declarations, dict):
        raise ValueError("camera_info must be keyed by input topic")
    camera_topics = {cfg.get(role + "_topic") for role in _LABELS}
    candidates = [t for t in topics if t not in declarations and t not in camera_topics]
    if not candidates:
        raise ValueError("Connect one state/eef input from the state card")
    named = [t for t in candidates if t.endswith("/state/eef")]
    if len(named or candidates) != 1:
        raise ValueError("Cannot identify one state/eef input. Connect one state source and assign "
                         "unidentified cameras using head_topic, left_wrist_topic or right_wrist_topic")
    state, = named or candidates
    return state, resolve_cameras([t for t in topics if t != state], declarations, cfg)


def resolve_cameras(topics, declarations, cfg):
    """Use only explicit mappings and declared mount identities to bind views."""
    bound = {}
    for topic in topics:
        if not isinstance(topic, str) or not topic.startswith("/"):
            raise ValueError(f"Invalid camera topic: {topic!r}")
        declaration = declarations.get(topic, {})
        if not isinstance(declaration, dict):
            raise ValueError(f"Invalid camera_info for {topic}")
        if declaration.get("format") not in (None, "", "image/jpeg"):
            raise ValueError(f"{topic} does not publish image/jpeg")
        roles = {role for role in _LABELS if cfg.get(f"{role}_topic") == topic}
        mount = _mount(declaration.get("id"))
        if mount:
            roles.add(mount)
        if not roles:
            raise ValueError(f"Unknown camera role for {topic}; set head_topic, "
                             "left_wrist_topic or right_wrist_topic to the wired topic")
        if len(roles) != 1:
            raise ValueError(f"Conflicting camera roles for {topic}: {sorted(roles)}")
        role = roles.pop()
        if role in bound and bound[role] != topic:
            raise ValueError(f"Two topics mapped to {role}: {bound[role]} and {topic}")
        bound[role] = topic
    return bound


class CameraInputs:
    """The ROS node owns the subscriptions. The agent owns the sampling window."""

    def __init__(self, node, binding):
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage

        self._node = node
        self._binding = dict(binding)
        self._condition = threading.Condition()
        self._closed = False
        self._after_ns = None
        self._frames = {}
        self._errors = {}
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        for role, topic in self._binding.items():
            node.create_subscription(CompressedImage, topic, partial(self._on_image, role), qos)

    def _on_image(self, role, message):
        with self._condition:
            if self._after_ns is None or self._closed:
                return
            stamp = message.header.stamp
            stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            observed_role = _mount(message.header.frame_id)
            if observed_role and observed_role != role:
                self._errors[role] = f"frame_id identifies {observed_role}"
            elif not self._after_ns < stamp_ns <= self._node.get_clock().now().nanoseconds:
                self._errors[role] = "ROS image timestamp is outside the observation window"
            elif (bytes(message.data[:2]) != b"\xff\xd8" or
                  bytes(message.data[-2:]) != b"\xff\xd9"):
                self._errors[role] = "expected a JPEG image"
            else:
                self._frames[role] = (stamp_ns, message.data)
                self._errors.pop(role, None)
                self._condition.notify_all()
                return
            self._frames.pop(role, None)

    def snapshot(self, cancel, timeout=5.0):
        """Wait for post-request images on every wire. Never use a pre-motion frame."""
        if not self._binding:
            raise RuntimeError("No RGB camera connected to autonomous_task")
        with self._condition:
            if self._after_ns is not None:
                raise RuntimeError("A camera observation is already in progress")
            self._after_ns = self._node.get_clock().now().nanoseconds
            deadline = time.monotonic() + timeout
            try:
                while not self._closed and not cancel.is_set():
                    now_ns = self._node.get_clock().now().nanoseconds
                    # A fast camera may have stopped while we waited for a slow
                    # one. Do not pass its old image along with the new views.
                    missing = [role for role in self._binding
                               if role not in self._frames or
                               not 0 <= now_ns - self._frames[role][0] <= 2_000_000_000]
                    if not missing:
                        return {role: bytes(self._frames[role][1]) for role in self._binding}
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        reasons = [f"{role} ({self._binding[role]}): "
                                   f"{self._errors.get(role, 'no fresh image')}" for role in missing]
                        raise TimeoutError(", ".join(reasons))
                    self._condition.wait(min(remaining, 0.1))
                raise InterruptedError("Camera observation cancelled")
            finally:
                self._after_ns = None
                self._frames.clear()
                self._errors.clear()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
