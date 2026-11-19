"""TRON2 high-level SDK WebSocket session for a single motion writer."""
import asyncio
from contextlib import asynccontextmanager
import json
import math
import threading
import time
import uuid

from .kinematics import ARM_LOWER, ARM_UPPER, JOINT_NAMES


DEFAULT_URL = "ws://10.192.1.2:5000"
_RESPONSE_TIMEOUT = 5.
_GRIPPER_START_WINDOW = 20.  # v1 allowed this long for delayed start/contact.
_POLL_INTERVAL = .15
_SETTLE_INTERVAL = .5
_JOINTS = JOINT_NAMES + ("head_pitch_Joint", "head_yaw_Joint")


def _numeric(value):
    return type(value) in (float, int) and math.isfinite(value)


class Vendor:
    """Each operation uses one session and pins all replies to its robot/GUID.

    SDK developer mode is not a motion lease. The caller owns the handoff
    from any previous writer before making this client's first motion call.
    """

    def __init__(self, url=DEFAULT_URL):
        self.url = url
        self._lock = threading.Lock()
        self._serial = None
        self._info = {}
        self._closed = False
        self._uncertain = False
        self.cancel_event = None

    async def _receive(self, ws, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("TRON2 SDK reply timed out")
        data = await asyncio.wait_for(ws.receive_str(), remaining)
        if not isinstance(data, str) or not data or len(data) > 256*1024:
            raise RuntimeError("Invalid TRON2 SDK message")
        message = json.loads(data)
        if not isinstance(message, dict) or not isinstance(message.get("data"), dict):
            raise RuntimeError("Malformed TRON2 SDK message")
        return message

    @asynccontextmanager
    async def _session(self):
        if self._closed:
            raise RuntimeError("TRON2 SDK client is closed")
        import aiohttp
        async with aiohttp.ClientSession(trust_env=False) as session:
            ws = await asyncio.wait_for(session.ws_connect(self.url, max_msg_size=256*1024), _RESPONSE_TIMEOUT)
            try:
                deadline = time.monotonic() + _RESPONSE_TIMEOUT
                while True:
                    msg = await self._receive(ws, deadline)
                    if msg.get("title") == "notify_robot_info":
                        self._accept_info(msg)
                        break
                yield ws
            finally:
                await ws.close()

    def _accept_info(self, msg):
        serial = msg.get("accid") or msg["data"].get("accid")
        if (not isinstance(serial, str) or not serial.startswith("DACH_TRON2")
                or msg["data"].get("accid", serial) != serial
                or self._serial not in (None, serial)):
            raise RuntimeError("TRON2 SDK robot identity changed or is not DACH")
        self._serial = serial
        self._info = msg["data"]

    def _ready(self):
        info = self._info
        if (info.get("working_mode_valid") is not True or info.get("working_mode") != "developer_mode"
                or info.get("collision_enable_valid") is not True or info.get("collision_enable") is not True
                or info.get("compliant_enable_valid") is not True or info.get("compliant_enable") is not False):
            raise RuntimeError("TRON2 motion requires confirmed developer mode, collision protection and compliance off")

    async def _request(self, ws, operation, data):
        """Exactly one send. Ignore unsolicited telemetry and unrelated GUIDs."""
        guid = str(uuid.uuid4())
        request = {"accid": self._serial, "title": "request_" + operation,
                   "timestamp": time.time_ns() // 1_000_000, "guid": guid, "data": data}
        await asyncio.wait_for(ws.send_str(json.dumps(request, allow_nan=False)), _RESPONSE_TIMEOUT)
        deadline = time.monotonic() + _RESPONSE_TIMEOUT
        while True:
            msg = await self._receive(ws, deadline)
            if msg.get("title") == "notify_robot_info":
                self._accept_info(msg)
                continue
            if msg.get("accid") != self._serial:
                raise RuntimeError("TRON2 SDK reply came from another robot")
            if msg.get("guid") != guid:
                continue
            if msg.get("title") == "notify_invalid_request":
                raise RuntimeError(f"TRON2 SDK rejected {operation}")
            if msg.get("title") != "response_" + operation:
                continue
            if msg["data"].get("result") != "success":
                raise RuntimeError(f"TRON2 SDK {operation}: {msg['data'].get('result')}")
            return guid, msg["data"]

    async def _read(self, ws):
        _, joints = await self._request(ws, "get_joint_state", {})
        _, claws = await self._request(ws, "get_limx_2fclaw_state", {})
        q, dq = joints.get("q"), joints.get("dq")
        if any(not isinstance(v, list) or len(v) != 16 or not all(map(_numeric, v)) for v in (q, dq)):
            raise RuntimeError("TRON2 SDK returned invalid joint positions or velocities")
        names = joints.get("names")
        if names:
            if len(names) != 16 or set(names) != set(_JOINTS):
                raise RuntimeError("TRON2 SDK returned unexpected joint names")
            q = [q[names.index(n)] for n in _JOINTS]
            dq = [dq[names.index(n)] for n in _JOINTS]
        if not all(_numeric(d.get("timestamp")) and d["timestamp"] >= 0 for d in (joints, claws)):
            raise RuntimeError("TRON2 SDK feedback lacks a source timestamp")
        grippers = {}
        for side in ("left", "right"):
            opening, speed, force = (claws.get(f"{side}_{key}") for key in ("opening", "speed", "force"))
            if not all(_numeric(v) and 0 <= v <= 100 for v in (opening, speed, force)):
                raise RuntimeError(f"Invalid {side} gripper feedback")
            grippers[side] = {"opening_raw": opening, "opening_mm": opening * 85 / 94,
                              "speed": speed, "force_level": force,
                              "vendor_timestamp_ms": claws["timestamp"], "grasp_verified": False}
        return {"q": q, "dq": dq, "robot_timestamp": joints["timestamp"], "grippers": grippers}

    async def _snapshot(self):
        async with self._session() as ws:
            return await self._read(ws)

    def snapshot(self):
        with self._lock:
            return asyncio.run(self._snapshot())

    def move_j(self, joints, duration_s, side):
        if (side not in ("left", "right") or not isinstance(joints, (tuple, list)) or len(joints) != 14 or
                not all(_numeric(v) and lo <= v <= hi for v, lo, hi in zip(joints, ARM_LOWER, ARM_UPPER))
                or not _numeric(duration_s) or not 1 <= duration_s <= 60):
            raise ValueError("Invalid TRON2 MoveJ target or duration")
        data = {"joint": list(joints), "time": duration_s}
        with self._lock:
            return asyncio.run(self._motion("movej", data, duration_s, side))

    def set_gripper(self, side, opening_mm):
        if side not in ("left", "right") or not _numeric(opening_mm) or not 0 <= opening_mm <= 85:
            raise ValueError("Expected a gripper side and nominal opening in [0,85] mm")
        raw = max(1, math.floor(opening_mm * 94/85 + .5)) if opening_mm else 0
        data = {side + "_opening": raw, side + "_speed": 15, side + "_force": 30}
        with self._lock:
            return asyncio.run(self._motion("set_limx_2fclaw_cmd", data, 0, side))

    async def _motion(self, operation, data, duration, side):
        if self._uncertain:
            raise RuntimeError("Previous TRON2 SDK motion outcome is unknown")
        async with self._session() as ws:
            before = await self._read(ws)
            self._ready()
            arm_motion = operation == "movej"
            other = "right" if side == "left" else "left"
            stationary_jaws = (other,) if arm_motion else (side, other)
            if (any(abs(v) > .05 for v in before["dq"]) or
                    any(before["grippers"][s]["speed"] != 0 for s in stationary_jaws)):
                raise RuntimeError("TRON2 was moving before dispatch")
            held = slice(0, 16)
            if arm_motion:
                jump = max(abs(a-b) for a,b in zip(data["joint"], before["q"][:14]))
                if jump > math.radians(35) or jump/duration > math.radians(20):
                    raise ValueError("MoveJ exceeds per-step angle or average rate limit")
                # MoveJ always carries all 14 joints. The other arm must remain
                # at feedback even though both arm groups share one wire packet.
                held = slice(7,14) if side == "left" else slice(0,7)
                if max(abs(a-b) for a,b in zip(data["joint"][held], before["q"][held])) > math.radians(.25):
                    raise ValueError("MoveJ must hold the other arm at measured feedback")
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise InterruptedError("Cancelled before TRON2 SDK dispatch")
            self._uncertain = True
            try:
                guid, _ = await self._request(ws, operation, data)
                result = await self._settled(ws, before, duration, side, held, stationary_jaws)
            except Exception as error:
                raise RuntimeError(f"TRON2 SDK {operation} outcome unconfirmed; no retry: {error}") from error
            result["action_id"] = guid
            self._uncertain = False
            return result

    async def _settled(self, ws, before, duration, side, held, stationary_jaws):
        # The SDK acknowledges admission. Fresh, stationary telemetry after
        # that acknowledgement provides a separate end-of-motion observation.
        started = time.monotonic()
        earliest = started + duration
        arm_motion = duration > 0
        deadline = earliest + (0 if arm_motion else _GRIPPER_START_WINDOW) + 12
        other = "right" if side == "left" else "left"
        last = before
        stationary_since = None
        samples = 0
        displacement_seen = False
        while time.monotonic() < deadline:
            await asyncio.sleep(_POLL_INTERVAL)
            current = await self._read(ws)
            self._ready()
            jaws = current["grippers"]
            if (current["robot_timestamp"] <= last["robot_timestamp"] or
                    jaws["left"]["vendor_timestamp_ms"] <= last["grippers"]["left"]["vendor_timestamp_ms"]):
                stationary_since, samples = None, 0
                continue
            if max(abs(a-b) for a,b in zip(current["q"][held], before["q"][held])) > math.radians(.25):
                raise RuntimeError("Held arm moved during TRON2 SDK command")
            if abs(jaws[other]["opening_raw"] - before["grippers"][other]["opening_raw"]) > 1:
                raise RuntimeError("Uncommanded gripper moved during TRON2 SDK command")
            arm_settled = (all(abs(v) <= .05 for v in current["dq"]) and
                           max(abs(a-b) for a,b in zip(current["q"], last["q"])) <= .003)
            jaws_settled = all(jaws[s]["speed"] == 0 and
                               abs(jaws[s]["opening_raw"] - last["grippers"][s]["opening_raw"]) <= 1
                               for s in stationary_jaws)
            last = current
            if not arm_motion:
                start = before["grippers"][side]["opening_raw"]
                actual = jaws[side]["opening_raw"]
                displacement_seen |= abs(actual - start) >= 1
            else:
                selected = slice(0,7) if held.start == 7 else slice(7,14)
                displacement_seen |= max(abs(a-b) for a,b in zip(current["q"][selected], before["q"][selected])) >= .001
            now = time.monotonic()
            ready_at = started + _GRIPPER_START_WINDOW if not arm_motion and not displacement_seen else earliest
            if now >= ready_at and arm_settled and jaws_settled:
                if stationary_since is None:
                    stationary_since = now
                samples += 1
                if samples >= 3 and now - stationary_since >= _SETTLE_INTERVAL:
                    current["displacement_observed"] = displacement_seen
                    return current
            else:
                stationary_since, samples = None, 0
        raise TimeoutError("TRON2 did not provide fresh stationary completion feedback")

    def close(self):
        with self._lock:
            self._closed = True
