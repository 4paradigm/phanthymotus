#!/usr/bin/env python3
"""
battery_monitor.py — one battery reading format for every robot.

Every legged driver already publishes a `battery` sensor card, but the fields
are the vendor's: Bumi sends `soc`/`soh`/`temperature`/`alarm`, G1 sends the
BMS record with a package-voltage list, per-cell voltage and per-cell
temperature, and Tianyi sends
`master_voltage`/`little_voltage` with no state of charge at all. Nothing on
the dashboard rendered any of them — `data/json` fell through to the raw JSON
text renderer — and a skill that wants to ask "how much battery is left" had
to know which robot it was talking to.

This card subscribes one driver battery topic and republishes a canonical
`sensor/battery` reading the web renderer draws as a gauge, and answers
`status` with the same reading plus its age for skills. It is read-only: it
never commands, alarms or speaks. Thresholds and low-battery announcements are
the skill's decision, not the card's — a card that announces has to own
deduplication, recovery and "who do I speak to", and none of that belongs in a
format converter.

Why normalize at all rather than asking skills to read the raw topic: a skill
cannot subscribe. Its only view of the robot is `tools/call`, so the reading
has to be a tool result, and a tool result has to be honest about which fields
this particular robot does not have (`soc: null` on Tianyi) instead of
inventing a number from voltage.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from utils.ros_lifecycle import dispose_node

log = logging.getLogger(__name__)

INPUT_FORMAT = "data/json"
OUTPUT_FORMAT = "sensor/battery"
# Versioned so the renderer and any future consumer can tell a field's meaning
# from its name. Same shape as the driver-side `motus.camera/1` declarations.
SCHEMA = "motus.battery/1"

DEFAULT_OUTPUT_TOPIC = "/perception/battery"
DEFAULT_STALE_AFTER_S = 10.0
_REPUBLISH_PERIOD_S = 1.0

# Drivers publish BEST_EFFORT at low rate (Bumi 1 Hz); a RELIABLE subscriber
# would never match them.
_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

# Canonical field → source keys whose **unit is known**, read in order, first
# parseable one wins. The unit is in the canonical name because a number
# without one is the failure mode this card exists to remove: G1 publishes
# `current` in mA and `cell_vol` in mV, Tianyi publishes `master_current` in A,
# and two fields named "current" that differ by 1000× are worse than no field.
#
# Bare `voltage` / `current` / `power` (Adam, K1) are deliberately absent:
# their drivers do not state a unit, and a profile has to be declared for them
# (see `_PROFILES`) before they are mapped.
_COMMON_ALIASES: Dict[str, tuple] = {
    "soc": ("soc", "battery_soc", "percentage", "battery_percentage"),
    "soh": ("soh", "battery_soh"),
    "cycle": ("cycle", "cycles"),
    "alarm": ("alarm", "battery_alarm", "alarm_code"),
    "voltage_v": ("voltage_v", "master_voltage"),
    "current_a": ("current_a", "master_current"),
    "current_ma": ("current_ma",),
    "power_w": ("power_w", "master_power"),
}
# Profile → extra aliases for drivers whose field names are ambiguous alone.
# `g1` is the one this card ships for: unitree/g1/device.py documents
# `current(mA)`, publishes `cell_vol` per-cell in mV, and sends BMS
# temperature as a per-cell list in °C.
_PROFILES: Dict[str, Dict[str, tuple]] = {
    "auto": {},
    "g1": {
        "current_ma": ("current",),
        "temperature_c": ("temperature",),
        "cell_voltage_max_mv": ("cell_vol",),
        # G1 sends up to two battery packs in `bmsvoltage`, in mV: the office
        # robot reads 49000/49011, i.e. 49.0 V per pack. This is package
        # voltage, not a cell list — verified on the robot 2026-11-25.
        "pack_voltage_mv": ("voltage",),
    },
}
_BOOL_KEYS: Dict[str, tuple] = {
    "charging": ("charging", "is_charging", "charge_state"),
    "installed": ("battery_installed", "installed"),
    "working": ("battery_working", "working"),
}
# Profile fields that arrive as per-cell lists: the highest cell is the number
# an operator acts on. A *pack* voltage is not derivable from the cell list
# alone (topology is model-specific), so the list never becomes `voltage_v`.
_LIST_MAX_FIELDS = frozenset(("temperature_c", "cell_voltage_max_mv", "pack_voltage_mv"))

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "battery_monitor",
        "type": "processor",
        "multiInstance": False,
        "description": (
            "Battery monitor — subscribes a driver battery topic and republishes "
            "one normalized reading (SOC/SOH/voltage/current/temperature) on a "
            "sensor/battery topic; `status` returns the latest reading with its age"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "status", "config"],
                    "description": "Action to perform",
                },
                "input_topic": {
                    "type": "string",
                    "description": "Driver battery topic (data/json). start requires it",
                },
                "input_topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Canvas wiring supplies the same topic as a list",
                },
            },
            "required": ["action"],
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "input_topic": {"type": "string", "description": "Default input when the canvas wires none"},
                "output_topic": {"type": "string", "default": DEFAULT_OUTPUT_TOPIC},
                "stale_after_s": {
                    "type": "number",
                    "default": DEFAULT_STALE_AFTER_S,
                    "description": "Age after which the reading is marked stale",
                },
                "input_profile": {
                    "type": "string",
                    "enum": sorted(_PROFILES),
                    "default": "auto",
                    "description": (
                        "Field-name conventions of the wired driver. `auto` maps "
                        "only unit-carrying keys; `g1` additionally reads bare "
                        "current (mA), temperature (C) and per-cell voltage (mV)"
                    ),
                },
            },
        },
        "topic_in": [{"format": INPUT_FORMAT, "desc": "driver battery reading"}],
        "topic_out": [{"format": OUTPUT_FORMAT, "desc": "normalized battery reading"}],
    }
]


# ── normalization ────────────────────────────────────────────────────────────

def _as_float(value: Any) -> Optional[float]:
    """A number, or None. Booleans are rejected on purpose: `True` is an int
    in Python and a battery whose SOC arrives as a flag is a bug, not 1%."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "on", "charging", "1"):
            return True
        if low in ("false", "no", "off", "discharging", "idle", "0"):
            return False
    return None


def _max_of_numbers(value: Any) -> Optional[float]:
    """Largest non-zero numeric entry of a list (per-cell telemetry), or None.

    Zero means an unused cell slot in the G1 BMS arrays, not a measurement.
    """
    if not isinstance(value, (list, tuple)):
        return None
    numbers = [n for n in (_as_float(item) for item in value) if n]
    return max(numbers) if numbers else None


def _rounded(value: float) -> float:
    # Two decimals keep mV/0.1°C detail without publishing float noise.
    return round(value, 2)


def resolve_profile(name: Any) -> str:
    """Known profile name, or "auto". Unknown names fall back loudly at start."""
    candidate = str(name or "auto").strip().lower()
    return candidate if candidate in _PROFILES else "auto"


def normalize_reading(payload: Any, profile: str = "auto") -> Dict[str, Any]:
    """Canonical fields from one driver battery message.

    Only fields present in the message are returned. An absent field stays
    absent — `soc` on Tianyi is genuinely unknown, and a null would let a
    renderer draw an empty bar that looks like 0%.
    """
    if not isinstance(payload, Mapping):
        return {}

    aliases: Dict[str, tuple] = dict(_COMMON_ALIASES)
    for field, keys in _PROFILES.get(resolve_profile(profile), {}).items():
        aliases[field] = tuple(aliases.get(field, ())) + tuple(keys)

    reading: Dict[str, Any] = {}
    for field, keys in aliases.items():
        for key in keys:
            raw = payload.get(key)
            if field in _LIST_MAX_FIELDS:
                value = _as_float(raw)
                if value is None:
                    value = _max_of_numbers(raw)
            else:
                value = _as_float(raw)
            if value is not None:
                reading[field] = _rounded(value)
                break

    for field, keys in _BOOL_KEYS.items():
        for key in keys:
            value = _as_bool(payload.get(key))
            if value is not None:
                reading[field] = value
                break

    return reading


# ── ROS node ─────────────────────────────────────────────────────────────────

class _BatteryNode(Node):
    """One input topic in, one normalized topic out, latest reading retained."""

    def __init__(self, input_topic: str, output_topic: str, stale_after_s: float,
                 profile: str = "auto"):
        super().__init__("battery_monitor")
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.stale_after_s = stale_after_s
        self.profile = resolve_profile(profile)
        self.state = "running"

        self._lock = threading.Lock()
        self._reading: Optional[Dict[str, Any]] = None
        self._received_at: Optional[float] = None
        self._received = 0
        self._unparsed = 0
        self._published = 0

        self._publisher = self.create_publisher(String, output_topic, _QOS)
        self._subscription = self.create_subscription(
            String, input_topic, self._on_message, _QOS
        )
        # A robot that powers off stops publishing without saying so. Republish
        # at 1 Hz so the age keeps moving and the gauge greys out instead of
        # freezing on the last value it saw.
        self._timer = self.create_timer(_REPUBLISH_PERIOD_S, self._tick)

    # -- input ---------------------------------------------------------------

    def _on_message(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError):
            with self._lock:
                self._unparsed += 1
            return
        with self._lock:
            self._reading = normalize_reading(payload, self.profile)
            self._received_at = time.monotonic()
            self._received += 1
        self._publish_latest()

    # -- output --------------------------------------------------------------

    def latest(self) -> Optional[Dict[str, Any]]:
        """The current reading with freshness fields, or None before any input."""
        with self._lock:
            reading = dict(self._reading) if self._reading is not None else None
            received_at = self._received_at
        if reading is None or received_at is None:
            return None
        age = max(0.0, time.monotonic() - received_at)
        return {
            "schema": SCHEMA,
            "source_topic": self.input_topic,
            "profile": self.profile,
            **reading,
            "age_s": round(age, 1),
            "stale": age > self.stale_after_s,
        }

    def _tick(self) -> None:
        if self._reading is not None:
            self._publish_latest()

    def _publish_latest(self) -> None:
        reading = self.latest()
        if reading is None:
            return
        msg = String()
        msg.data = json.dumps(reading, ensure_ascii=False)
        self._publisher.publish(msg)
        with self._lock:
            self._published += 1

    # -- reporting -----------------------------------------------------------

    def statistics(self) -> Dict[str, int]:
        with self._lock:
            return {
                "received": self._received,
                "unparsed": self._unparsed,
                "published": self._published,
            }

    def status(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "profile": self.profile,
            "topic_in": [{"topic": self.input_topic, "format": INPUT_FORMAT, "desc": ""}],
            "topic_out": [{"topic": self.output_topic, "format": OUTPUT_FORMAT, "desc": ""}],
            "latest": self.latest(),
            "statistics": self.statistics(),
        }

    def stop(self) -> Dict[str, Any]:
        self.state = "idle"
        try:
            self._timer.cancel()
        except Exception:
            pass
        return self.status()


# ── plugin facade ────────────────────────────────────────────────────────────

class BatteryMonitorPlugin:
    """Single-instance facade: one battery per robot."""

    PREFIX = "battery_monitor"
    _DESC = "One normalized battery reading from any driver's battery topic"

    def __init__(self, plugin_cfg: Mapping[str, Any], executor: Any):
        self._cfg = dict(plugin_cfg or {})
        self._executor = executor
        self._node: Optional[_BatteryNode] = None
        self._lock = threading.RLock()

    def get_tools(self) -> List[Dict[str, Any]]:
        return TOOLS

    # -- helpers -------------------------------------------------------------

    def _resolve_input(self, args: Mapping[str, Any]) -> str:
        """First usable topic from canvas wiring, then the argument, then config."""
        topics = args.get("input_topics")
        if isinstance(topics, (list, tuple)):
            for topic in topics:
                if isinstance(topic, str) and topic.strip():
                    return topic.strip()
        topic = args.get("input_topic")
        if isinstance(topic, str) and topic.strip():
            return topic.strip()
        fallback = self._cfg.get("input_topic")
        return fallback.strip() if isinstance(fallback, str) else ""

    def _stale_after_s(self) -> float:
        value = _as_float(self._cfg.get("stale_after_s"))
        return value if value and value > 0 else DEFAULT_STALE_AFTER_S

    def _effective_config(self) -> Dict[str, Any]:
        return {
            "input_topic": self._cfg.get("input_topic", ""),
            "output_topic": str(self._cfg.get("output_topic") or DEFAULT_OUTPUT_TOPIC),
            "stale_after_s": self._stale_after_s(),
            "input_profile": resolve_profile(self._cfg.get("input_profile")),
            "profiles": sorted(_PROFILES),
        }

    def _apply_config(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        """Apply the config pushed by the canvas dialog.

        The dialog edits exactly what `configSchema` declares; a card that
        answers "configured" without changing anything turns that control into
        a lie. Keys outside the schema are ignored, which is the second line of
        defence after agent-core's own scope partitioning.

        A running node is restarted when something it owns changes (input
        topic, output topic, profile) and patched in place when only the
        staleness window moved. The wired topic is kept unless the new config
        names one, because the canvas connection is what actually started it.
        """
        with self._lock:
            updates: Dict[str, Any] = {}
            for key in ("input_topic", "output_topic", "input_profile"):
                value = args.get(key)
                if isinstance(value, str):
                    updates[key] = value.strip()
            stale = _as_float(args.get("stale_after_s"))
            if stale is not None:
                updates["stale_after_s"] = stale

            profile = updates.get("input_profile")
            if profile and profile.lower() not in _PROFILES:
                return {
                    "status": "error",
                    "error": "unknown input_profile %r; known: %s"
                             % (profile, ", ".join(sorted(_PROFILES))),
                    "config": self._effective_config(),
                }
            if not updates:
                return {"status": "configured", "config": self._effective_config()}

            self._cfg.update(updates)

            node = self._node
            if node is not None:
                restart = bool(
                    (updates.get("output_topic") and updates["output_topic"] != node.output_topic)
                    or (updates.get("input_topic") and updates["input_topic"] != node.input_topic)
                    or (profile and resolve_profile(profile) != node.profile)
                )
                if restart:
                    topic = updates.get("input_topic") or node.input_topic
                    self._stop()
                    self._start({"input_topics": [topic]})
                else:
                    node.stale_after_s = self._stale_after_s()

            return {"status": "configured", "config": self._effective_config()}

    def _info(self) -> Dict[str, Any]:
        with self._lock:
            node = self._node
        if node is None:
            return {
                "name": "BatteryMonitor",
                "manufacture": "Embodied",
                "model": "normalized-battery",
                "state": "idle",
                "desc": self._DESC,
                "topic_in": [],
                "topic_out": [],
            }
        info = {
            "name": "BatteryMonitor",
            "manufacture": "Embodied",
            "model": "normalized-battery",
            **node.status(),
        }
        info["desc"] = self._DESC
        return info

    def _start(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        with self._lock:
            input_topic = self._resolve_input(args)
            if not input_topic:
                return {
                    "state": "error",
                    "error": "no input topic — wire a driver battery topic to this card",
                    "topic_in": [],
                    "topic_out": [],
                }
            profile = str(self._cfg.get("input_profile") or "auto").strip().lower()
            if profile not in _PROFILES:
                return {
                    "state": "error",
                    "error": "unknown input_profile %r; known: %s"
                             % (profile, ", ".join(sorted(_PROFILES))),
                    "topic_in": [],
                    "topic_out": [],
                }
            if self._node is not None and self._node.input_topic == input_topic:
                return self._info()
            self._stop()

            output_topic = str(self._cfg.get("output_topic") or DEFAULT_OUTPUT_TOPIC)
            node = _BatteryNode(input_topic, output_topic, self._stale_after_s(), profile)
            self._executor.add_node(node)
            self._node = node
            log.info("[battery_monitor] started: %s -> %s", input_topic, output_topic)
            return self._info()

    def _stop(self) -> Dict[str, Any]:
        """Caller holds `_lock` (or is dispatch's stop path)."""
        node, self._node = self._node, None
        if node is None:
            return {"state": "idle"}
        status = node.stop()
        dispose_node(self._executor, node, label="battery_monitor")
        log.info("[battery_monitor] stopped: %s", status.get("topic_in"))
        return {"state": "idle"}

    # -- dispatch ------------------------------------------------------------

    def dispatch(self, name: str, args: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        action = str(args.get("action", name))
        if action == "info":
            return self._info()
        if action == "start":
            return self._start(args)
        if action == "stop":
            with self._lock:
                return self._stop()
        if action == "status":
            with self._lock:
                node = self._node
            if node is None:
                return {"state": "idle", "error": "card is not started", "latest": None}
            return node.status()
        if action == "config":
            return self._apply_config(args)
        return None


__all__ = ["BatteryMonitorPlugin", "TOOLS", "normalize_reading", "SCHEMA"]
