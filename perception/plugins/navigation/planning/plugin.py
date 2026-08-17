"""Perception Bundle adapter for the Nav2 card."""

from __future__ import annotations

import logging
import math
import re
import threading
import time

from .backend import RosTopicNavigationBackend
from .contract import NAV2_ACTIONS, NAV2_CONFIG_DEFAULTS, nav2_tool_definition
from .core import Nav2Core, UnavailableNavigationBackend


log = logging.getLogger(__name__)

_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_/-]{0,127}$")
_NUMBER_RANGES = {
    "request_timeout_sec": (1.0, 120.0),
    "discovery_timeout_sec": (0.5, 30.0),
    "min_x_mps": (0.0, 1.0),
    "max_x_mps": (0.0, 1.0),
    "min_y_mps": (0.0, 1.0),
    "max_y_mps": (0.0, 1.0),
    "min_yaw_rps": (0.0, 2.0),
    "max_yaw_rps": (0.0, 2.0),
}
_INTEGER_RANGES = {
    "input_max_age_ms": (100, 2000),
    "proposal_ttl_ms": (50, 250),
}
_LEGACY_NOOP_CONFIG_FIELDS = {"runtime_switch_timeout_sec"}


class ConfigError(ValueError):
    """Configuration is invalid and must not acquire runtime resources."""


def _validated_config(base: dict, updates: dict) -> dict:
    updates = {
        key: value
        for key, value in updates.items()
        if key not in _LEGACY_NOOP_CONFIG_FIELDS
    }
    unknown = sorted(set(updates) - set(NAV2_CONFIG_DEFAULTS))
    if unknown:
        raise ConfigError("unsupported config fields: " + ",".join(unknown))

    result = {**NAV2_CONFIG_DEFAULTS, **base, **updates}
    namespace = result.get("namespace")
    if not isinstance(namespace, str):
        raise ConfigError("namespace must be a string")
    namespace = namespace.strip("/")
    if not namespace or not _NAMESPACE_RE.fullmatch(namespace):
        raise ConfigError("namespace must be a non-empty ROS topic namespace")
    if "//" in namespace:
        raise ConfigError("namespace must not contain empty path segments")
    result["namespace"] = namespace
    if namespace != "ubuntu":
        raise ConfigError("first-release Nav2 runtime requires namespace=ubuntu")

    backend = result.get("backend")
    if backend not in {"ros_topic", "disabled"}:
        raise ConfigError("backend must be ros_topic or disabled")
    if result.get("shadow_only") is not True:
        raise ConfigError("shadow_only is fixed to true")

    for key, (minimum, maximum) in _NUMBER_RANGES.items():
        raw = result.get(key)
        if isinstance(raw, bool):
            raise ConfigError(f"{key} must be a number")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{key} must be a number") from exc
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ConfigError(f"{key} must be within [{minimum}, {maximum}]")
        result[key] = value

    for key, (minimum, maximum) in _INTEGER_RANGES.items():
        raw = result.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ConfigError(f"{key} must be an integer")
        if not minimum <= raw <= maximum:
            raise ConfigError(f"{key} must be within [{minimum}, {maximum}]")

    for axis, unit in (("x", "m/s"), ("y", "m/s"), ("yaw", "rad/s")):
        minimum = result[f"min_{axis}_{'rps' if axis == 'yaw' else 'mps'}"]
        maximum = result[f"max_{axis}_{'rps' if axis == 'yaw' else 'mps'}"]
        if minimum > maximum:
            raise ConfigError(
                f"min_{axis} must not exceed max_{axis} ({unit})"
            )

    fixed = {
        "input_max_age_ms": 500,
        "proposal_ttl_ms": 250,
    }
    for key, expected in fixed.items():
        if result[key] != expected:
            raise ConfigError(f"first-release {key} is fixed to {expected}")

    return result


class Nav2Plugin:
    """Expose lifecycle plus planner/controller-only Nav2 actions."""

    PREFIX = "nav2"

    def __init__(self, plugin_cfg: dict, executor, *, backend=None):
        raw_cfg = dict(plugin_cfg or {})
        raw_cfg.pop("enabled", None)
        self._executor = executor
        self._backend_override = backend
        self._lifecycle_lock = threading.RLock()
        self._core: Nav2Core | None = None
        self._canvas_started = False
        self._canvas_instance_id = ""
        self._wired_topics: dict[str, str] = {}
        self._config_error: str | None = None
        try:
            self._cfg = _validated_config({}, raw_cfg)
        except ConfigError as exc:
            self._cfg = dict(NAV2_CONFIG_DEFAULTS)
            self._config_error = str(exc)

    def get_tools(self) -> list:
        return [nav2_tool_definition(self._cfg["namespace"])]

    def dispatch(self, name: str, args: dict) -> dict | None:
        if name != "nav2":
            return None
        if not isinstance(args, dict):
            return self._error("invalid_argument", "arguments must be an object")
        action = args.get("action")
        if action == "info":
            return self._info()
        if action == "config":
            return self._configure(args)
        if action == "start":
            return self._start_canvas(args)
        if action == "stop":
            return self._stop_canvas()
        if action not in NAV2_ACTIONS:
            return self._error("unsupported_action", "unsupported Nav2 action")
        with self._lifecycle_lock:
            core = self._core
            canvas_started = self._canvas_started
        if not canvas_started or core is None:
            return self._error(
                "canvas_not_started",
                "connect FAST-LIVO2 odom, registered_cloud and obstacle_map, then start Canvas",
            )
        return core.dispatch(args)

    def stop(self) -> None:
        self._stop_canvas()

    @staticmethod
    def _error(code: str, message: str) -> dict:
        return {
            "state": "error",
            "status": "error",
            "error_code": code,
            "error": message,
            "message": message,
            "shadow_only": True,
            "physical_execution": False,
        }

    def _configure(self, args: dict) -> dict:
        with self._lifecycle_lock:
            if self._canvas_started:
                return self._error(
                    "config_while_running",
                    "stop the Nav2 card before changing shared configuration",
                )
            updates = {
                key: value
                for key, value in args.items()
                if key not in {"action", "instance_id"}
            }
            try:
                self._cfg = _validated_config(self._cfg, updates)
            except ConfigError as exc:
                self._config_error = str(exc)
                return self._error("invalid_config", str(exc))
            self._config_error = None
            return {
                "state": "configured",
                "config": dict(self._cfg),
                "takes_effect": "next_start",
                "shadow_only": True,
                "physical_execution": False,
            }

    def _start_canvas(self, args: dict) -> dict:
        with self._lifecycle_lock:
            if self._canvas_started:
                result = self._info()
                result["already_started"] = True
                return result
            if self._config_error:
                return self._error("invalid_config", self._config_error)

        wiring = self._validate_wiring(args)
        if "error_code" in wiring:
            return wiring

        core = self._ensure_core()
        backend_info = self._await_backend_startup(core)
        backend_state = str(backend_info.get("state", "idle"))
        if backend_state in {"unavailable", "error"}:
            reason = str(
                backend_info.get("reason") or backend_info.get("error") or backend_state
            )
            self._release_core()
            return self._error("backend_not_ready", reason)
        if backend_info.get("backend") == "nav2_ros_topic":
            if int(backend_info.get("bridge_subscribers", 0)) < 1:
                self._release_core()
                return self._error(
                    "nav2_runtime_unavailable",
                    "in-container Nav2 runtime is not subscribed to the command topic",
                )

        instance_id = str(args.get("instance_id", "") or "default").strip()
        with self._lifecycle_lock:
            self._canvas_started = True
            self._canvas_instance_id = instance_id
            self._wired_topics = dict(wiring["wired_topics"])
        result = self._info()
        result.update({"state": "ready", "status": "ready"})
        return result

    def _await_backend_startup(self, core: Nav2Core) -> dict:
        """Wait only for DDS discovery; sensor readiness gates navigation actions."""

        deadline = time.monotonic() + self._cfg["discovery_timeout_sec"]
        while True:
            info = core.info()
            if info.get("backend") != "nav2_ros_topic":
                return info
            if str(info.get("state", "idle")) in {"unavailable", "error"}:
                return info
            if int(info.get("bridge_subscribers", 0)) >= 1:
                return info
            if time.monotonic() >= deadline:
                return info
            time.sleep(0.05)

    def _validate_wiring(self, args: dict) -> dict:
        tool = self.get_tools()[0]
        expected = {item["port"]: item["topic"] for item in tool["topic_in"]}
        required_ports = {
            item["port"] for item in tool["topic_in"] if item.get("required", True)
        }
        raw_topics = args.get("input_topics", [])
        if isinstance(raw_topics, str):
            raw_topics = [raw_topics]
        if not isinstance(raw_topics, list) or any(
            not isinstance(topic, str) for topic in raw_topics
        ):
            return self._error(
                "invalid_canvas_wiring", "input_topics must be an array of topic names"
            )
        single_topic = args.get("input_topic")
        if single_topic:
            if not isinstance(single_topic, str):
                return self._error(
                    "invalid_canvas_wiring", "input_topic must be a topic name"
                )
            raw_topics = [*raw_topics, single_topic]
        unique_topics = {topic.strip() for topic in raw_topics if topic.strip()}

        raw_bindings = args.get("input_bindings", [])
        if raw_bindings is None:
            raw_bindings = []
        if not isinstance(raw_bindings, list) or any(
            not isinstance(binding, dict) for binding in raw_bindings
        ):
            return self._error(
                "invalid_canvas_wiring", "input_bindings must be an array"
            )
        ports = [str(binding.get("port", "")) for binding in raw_bindings]
        duplicates = sorted(port for port in set(ports) if ports.count(port) > 1)
        if duplicates:
            return self._error(
                "invalid_canvas_wiring",
                "input_bindings contain duplicate ports: " + ",".join(duplicates),
            )

        if raw_bindings:
            unknown = sorted(set(ports) - set(expected))
            bound_topics = {
                str(binding.get("port", "")): str(binding.get("topic", "")).strip()
                for binding in raw_bindings
                if str(binding.get("port", "")) in expected
                and str(binding.get("topic", "")).strip()
            }
            missing = sorted(required_ports - set(bound_topics))
            wrong = sorted(
                port
                for port in required_ports
                if bound_topics.get(port) != expected[port]
            )
        else:
            unknown = sorted(unique_topics - set(expected.values()))
            bound_topics = {
                port: topic for port, topic in expected.items() if topic in unique_topics
            }
            missing = sorted(required_ports - set(bound_topics))
            wrong = []

        if unknown or missing or wrong:
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if wrong:
                details.append("wrong_topic=" + ",".join(wrong))
            if unknown:
                details.append("unexpected=" + ",".join(unknown))
            return self._error(
                "invalid_canvas_wiring",
                "Nav2 requires exact FAST-LIVO2 bindings (" + "; ".join(details) + ")",
            )
        return {"wired_topics": bound_topics}

    def _stop_canvas(self) -> dict:
        with self._lifecycle_lock:
            core = self._core
            was_started = self._canvas_started
        stop_result = None
        if was_started and core is not None and core.info().get("active_nav_id"):
            stop_result = core.dispatch({"action": "stop_nav"})
        self._release_core()
        return {
            "state": "idle",
            "status": "idle",
            "canvas_wired": False,
            "stop_result": stop_result,
            "shadow_only": True,
            "physical_execution": False,
        }

    def _ensure_core(self) -> Nav2Core:
        with self._lifecycle_lock:
            if self._core is None:
                backend = self._backend_override
                if backend is None:
                    backend = self._build_backend()
                self._core = Nav2Core(backend)
            return self._core

    def _release_core(self) -> None:
        with self._lifecycle_lock:
            core = self._core
            self._core = None
            self._canvas_started = False
            self._canvas_instance_id = ""
            self._wired_topics = {}
        if core is not None:
            try:
                core.stop()
            except Exception:
                log.exception("[nav2] failed to release backend resources")

    def _build_backend(self):
        if self._cfg.get("shadow_only") is not True:
            return UnavailableNavigationBackend(
                "Perception navigation refuses non-shadow configuration"
            )
        backend_name = self._cfg.get("backend")
        if backend_name == "disabled":
            return UnavailableNavigationBackend("navigation backend is disabled")
        try:
            return RosTopicNavigationBackend(
                self._cfg, self._cfg["namespace"], self._executor
            )
        except Exception as exc:
            log.error("[nav2] backend unavailable: %s", exc, exc_info=True)
            return UnavailableNavigationBackend(
                f"Nav2 ROS topic backend unavailable: {type(exc).__name__}: {exc}"
            )

    def _info(self) -> dict:
        tool = self.get_tools()[0]
        with self._lifecycle_lock:
            core = self._core
            canvas_started = self._canvas_started
            instance_id = self._canvas_instance_id
            wired_topics = dict(self._wired_topics)
            config_error = self._config_error
        if core is None:
            result = {
                "state": "error" if config_error else "idle",
                "status": "error" if config_error else "idle",
                "backend": "not_started",
                "active_nav_id": None,
                "actions": list(NAV2_ACTIONS),
                "shadow_only": True,
                "physical_execution": False,
            }
            if config_error:
                result.update(
                    {"error_code": "invalid_config", "error": config_error}
                )
        else:
            result = core.info()
        result.update(
            {
                "name": "Nav2",
                "type": "processor",
                "canvas_wired": canvas_started,
                "instance_id": instance_id or None,
                "config": dict(self._cfg),
                "topic_in": [
                    {
                        **item,
                        "connected": wired_topics.get(item["port"]) == item["topic"],
                    }
                    for item in tool["topic_in"]
                ],
                "topic_out": tool["topic_out"],
                "control_lease": "requires_agent_core_execution_control",
            }
        )
        return result
