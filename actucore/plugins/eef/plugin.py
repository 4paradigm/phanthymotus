"""Control TCP pose and gripper opening from end-effector feedback."""
import json
import threading
from uuid import uuid4

from .stream import EefSpec, EefStream, _now_ms, finite, rotate, translate


_CONTROL = "/limx_tron2/control/eef"
_TRANSLATION = ("dx_m", "dy_m", "dz_m")
_ROTATION = ("rx_rad", "ry_rad", "rz_rad")
_DELTAS = (*_TRANSLATION, *_ROTATION, "opening_delta_m")
_POSE = ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")


class EefPlugin:
    PREFIX = "eef"

    def __init__(self, plugin_cfg, executor):
        self._executor = executor
        self._control_topic = plugin_cfg.get("control_topic") or _CONTROL
        self._lock = threading.RLock()
        self._source = f"mcp__actucore__eef/{uuid4().hex}"
        self._seq = 0
        self._stream = None
        self._node = self._state_topic = None

    def _ports(self):
        source = {"format": "state/eef"}
        if self._state_topic:
            source["topic"] = self._state_topic
        return {"topic_in": [source], "topic_out": [{"format": "control/eef", "topic": self._control_topic}]}

    def get_tools(self):
        descriptions = ("前为正，后为负（米）", "左为正，右为负（米）", "上为正，下为负（米）",
                        "绕基座 X 轴的旋转向量分量（弧度）", "绕基座 Y 轴的旋转向量分量（弧度）",
                        "绕基座 Z 轴的旋转向量分量（弧度）", "张开为正，闭合为负（米）")
        fields = {name: {"type": "number", "description": desc} for name, desc in zip(_DELTAS, descriptions)}
        for name, axis in zip(_POSE[:3], "XYZ"):
            fields[name] = {"type": "number", "description": f"目标 TCP 在驱动参考系中的 {axis} 坐标（米）"}
        for name, component in zip(_POSE[3:], "xyzw"):
            fields[name] = {"type": "number", "description": f"目标姿态单位四元数的 {component} 分量"}
        fields["opening_m"] = {"type": "number", "minimum": 0, "description": "夹爪的目标开口宽度（米）"}
        return [{"name": "eef", "type": "processor", "multiInstance": False, **self._ports(),
                 "description": "控制所选机械臂的工具中心点（TCP）位姿和夹爪开口，可设置绝对目标或相对增量。",
                 "inputSchema": {"type": "object", "required": ["action"], "properties": {
                     "action": {"type": "string", "enum": ["info", "start", "stop", "move_by", "move_to", "cancel"]},
                     "arm": {"type": "string", "enum": ["left", "right"]},
                     "control_interface": {"type": "object"}, **fields},
                     "x-action-params": {"move_by": {"params": ["arm", *_DELTAS],
                                                     "description": "将输入增量叠加到所选臂的实测位姿或夹爪开口，发布绝对目标"},
                                         "move_to": {"params": ["arm", *_POSE, "opening_m"],
                                                     "description": "设定完整 TCP 位姿、绝对夹爪开口或两者。位置和开口以米计，姿态为 xyzw 单位四元数"},
                                         "cancel": {"params": [], "description": "将受控组固定在最新实测位置，保持未选臂当前目标"}},
                     "x-hooks": {"on_interrupt_motion": {"action": "cancel"}, "on_interrupt_all": {"action": "cancel"}},
                     "x-is-dangerous": True, "x-resource": ["arm_l", "arm_r", "gripper_l", "gripper_r"]}}]

    def dispatch(self, name, args):
        if name != self.PREFIX:
            return None
        with self._lock:
            try:
                action = args.get("action", name)
                if action == "start":
                    self._start(args)
                elif action == "stop":
                    self._stop()
                elif action in ("move_by", "move_to"):
                    return self._move(action, args)
                elif action == "cancel":
                    if self._stream:
                        return self._stream.cancel()
                elif action != "info":
                    raise ValueError("Unknown eef action")
                fault = self._stream.fault if self._stream else None
                return {"state": "error" if fault else "running" if self._node else "idle",
                        **({"message": fault, "fault": fault} if fault else {}), **self._ports(),
                        "sample": self._stream.sample if self._stream else None,
                        "last": self._stream.last if self._stream else None}
            except (ValueError, TypeError, RuntimeError, OverflowError) as error:
                return {"state": "error", "message": str(error)}

    def _start(self, args):
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        topics = args.get("input_topics", [])
        if not isinstance(topics, list):
            raise ValueError("Expected a list of state/eef input topics")
        topics = topics + ([args["input_topic"]] if args.get("input_topic") else [])
        if (not topics or any(not isinstance(t, str) or not t.startswith("/") for t in topics)
                or len(set(topics)) != 1 or topics[0] == self._control_topic):
            raise ValueError("Connect one state/eef input, distinct from the control/eef output")
        descriptor = args.get("control_interface")
        if self._node:
            if self._state_topic != topics[0] or (descriptor is not None and descriptor != self._stream.spec.descriptor):
                raise ValueError("Stop eef before changing its connections")
            return
        spec = EefSpec(descriptor)
        node = Node("eef")
        try:
            qos = QoSProfile(depth=5, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            pub = node.create_publisher(String, self._control_topic, qos)

            def publish(command):
                msg = String()
                msg.data = json.dumps(command, allow_nan=False)
                pub.publish(msg)

            stream = EefStream(spec, self._source, publish, _now_ms, initial_seq=self._seq)
            node.create_subscription(String, topics[0], lambda msg: self._receive(node, msg), qos)
            self._executor.add_node(node)
        except Exception:
            node.destroy_node()
            raise
        self._node, self._state_topic = node, topics[0]
        self._stream = stream

    def _stop(self):
        if self._node:
            self._executor.remove_node(self._node)
            self._node.destroy_node()
        self._node = self._state_topic = None
        if self._stream:
            self._seq = self._stream.seq
        self._stream = None

    def _receive(self, node, msg):
        try:
            data = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        with self._lock:
            if node is self._node and self._stream:
                self._stream.receive(data)

    def _move(self, action, args):
        stream = self._stream
        if not stream or not stream.sample or not stream.fresh(stream.sample["stamp_ms"]):
            raise ValueError("Start eef and wait for fresh eef_state feedback")
        arm = args.get("arm")
        if arm not in stream.spec.tip_frames:
            raise ValueError("Select an arm declared by the connected actuator")
        spec = stream.spec
        pose = spec.offsets[arm, "eef"]
        groups, goal = [], []
        if action == "move_to":
            if any(name in args for name in _POSE):
                goal = [args.get(name) for name in _POSE]
                if not all(map(finite, goal)):
                    raise ValueError("A TCP target requires finite x_m, y_m, z_m, qx, qy, qz and qw")
                if any(not lo <= v <= hi for v, lo, hi in zip(goal, spec.lower[pose], spec.upper[pose])):
                    raise ValueError(f"Target TCP pose is outside the declared limits for the {arm} arm")
                groups.append(pose)
            if "opening_m" in args:
                opening = args["opening_m"]
                if not finite(opening) or opening < 0:
                    raise ValueError("opening_m must be a finite, non-negative width in metres")
                groups.append(spec.offsets[arm, "gripper"])
                goal.append(opening)
            if not groups:
                raise ValueError("move_to requires a complete TCP pose, opening_m, or both")
            return stream.set_target(arm, groups, goal)
        delta = [args.get(k, 0.) for k in _DELTAS]
        if not all(map(finite, delta)) or not any(k in args for k in _DELTAS):
            raise ValueError("Provide at least one finite increment")
        translation, rotation, opening = delta[:3], delta[3:6], delta[6]
        measured = stream.sample["eef"]
        if any(translation) or any(rotation):
            current = measured[pose]
            xyz = current[:3]
            if any(translation):
                xyz = translate(xyz, translation, spec.lower[pose][:3], spec.upper[pose][:3])
            q = rotate(current[3:], rotation) if any(rotation) else current[3:]
            groups.append(pose)
            goal.extend(xyz + q)
        if opening:
            jaw = spec.offsets[arm, "gripper"]
            i = jaw.start
            groups.append(jaw)
            goal.append(min(spec.upper[i], max(spec.lower[i], measured[i] + opening)))
        return stream.set_target(arm, groups, goal)
