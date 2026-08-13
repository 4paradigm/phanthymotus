"""Perception Bundle adapter for the FAST-LIVO2 card."""

from __future__ import annotations

import logging
import math
from pathlib import PurePosixPath
import re
import threading
import time

from .backend import RosTopicFastLivo2Backend
from .contract import (
    FAST_LIVO2_ACTIONS,
    FAST_LIVO2_CONFIG_DEFAULTS,
    fast_livo2_tool_definition,
)
from .core import FastLivo2Core, UnavailableFastLivo2Backend


log = logging.getLogger(__name__)
_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_/-]{0,127}$")
_COLLECTION_ROOT = PurePosixPath(
    "/opt/phanthy-motus/data/fast_livo2/recordings"
)


class ConfigError(ValueError):
    pass


def _validated_config(base: dict, updates: dict) -> dict:
    unknown = sorted(set(updates) - set(FAST_LIVO2_CONFIG_DEFAULTS))
    if unknown:
        raise ConfigError("unsupported config fields: " + ",".join(unknown))
    result = {**FAST_LIVO2_CONFIG_DEFAULTS, **base, **updates}
    namespace = result.get("namespace")
    if not isinstance(namespace, str):
        raise ConfigError("namespace must be a string")
    namespace = namespace.strip("/")
    if not namespace or not _NAMESPACE_RE.fullmatch(namespace) or "//" in namespace:
        raise ConfigError("namespace must be a non-empty ROS namespace")
    if namespace != "ubuntu":
        raise ConfigError("first-release FAST-LIVO2 requires namespace=ubuntu")
    result["namespace"] = namespace
    if result.get("backend") not in {"ros_topic", "disabled"}:
        raise ConfigError("backend must be ros_topic or disabled")
    ranges = {
        "request_timeout_sec": (5.0, 180.0),
        "discovery_timeout_sec": (0.5, 30.0),
        "map_voxel_size_m": (0.05, 0.50),
    }
    for key, (minimum, maximum) in ranges.items():
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
    if result.get("input_max_age_ms") != 500:
        raise ConfigError("input_max_age_ms is fixed to 500")
    if result.get("map_max_points") != 80_000:
        raise ConfigError("map_max_points is fixed to 80000")
    if not isinstance(result.get("collection_enabled"), bool):
        raise ConfigError("collection_enabled must be a boolean")
    raw_directory = result.get("collection_directory")
    if not isinstance(raw_directory, str) or not raw_directory.strip():
        raise ConfigError("collection_directory must be a non-empty absolute path")
    directory = PurePosixPath(raw_directory.strip())
    if not directory.is_absolute() or ".." in directory.parts:
        raise ConfigError("collection_directory must be a safe absolute path")
    try:
        directory.relative_to(_COLLECTION_ROOT)
    except ValueError as exc:
        raise ConfigError(
            f"collection_directory must be within {_COLLECTION_ROOT}"
        ) from exc
    result["collection_directory"] = str(directory)
    return result


class FastLivo2Plugin:
    PREFIX = "fast_livo2"

    def __init__(self, plugin_cfg: dict, executor, *, backend=None):
        raw_cfg = dict(plugin_cfg or {})
        raw_cfg.pop("enabled", None)
        self._executor = executor
        self._backend_override = backend
        self._lock = threading.RLock()
        self._core: FastLivo2Core | None = None
        self._canvas_started = False
        self._instance_id = ""
        self._wired_topics: dict[str, str] = {}
        self._config_error: str | None = None
        try:
            self._cfg = _validated_config({}, raw_cfg)
        except ConfigError as exc:
            self._cfg = dict(FAST_LIVO2_CONFIG_DEFAULTS)
            self._config_error = str(exc)

    def get_tools(self) -> list:
        return [fast_livo2_tool_definition(self._cfg["namespace"])]

    def dispatch(self, name: str, args: dict) -> dict | None:
        if name != "fast_livo2":
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
        if action not in FAST_LIVO2_ACTIONS:
            return self._error("unsupported_action", "unsupported FAST-LIVO2 action")
        with self._lock:
            core = self._core
            started = self._canvas_started
        if not started or core is None:
            return self._error(
                "canvas_not_started", "connect LiDAR and IMU, then start Canvas"
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
            "physical_execution": False,
        }

    def _configure(self, args: dict) -> dict:
        with self._lock:
            if self._canvas_started:
                return self._error("config_while_running", "stop card before config")
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
            return {"state": "configured", "config": dict(self._cfg), "physical_execution": False}

    def _start_canvas(self, args: dict) -> dict:
        with self._lock:
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
        deadline = time.monotonic() + self._cfg["discovery_timeout_sec"]
        while True:
            info = core.info()
            if info.get("backend") != "fast_livo2_ros_topic":
                break
            if int(info.get("bridge_subscribers", 0)) >= 1:
                break
            if time.monotonic() >= deadline:
                self._release_core()
                return self._error(
                    "fast_livo2_companion_unavailable",
                    "FAST-LIVO2 companion is not subscribed to the command topic",
                )
            time.sleep(0.05)
        if info.get("state") in {"unavailable", "error"}:
            self._release_core()
            return self._error("backend_not_ready", str(info.get("reason", info["state"])))
        collection_result = core.configure_collection(
            {
                "enabled": self._cfg["collection_enabled"],
                "directory": self._cfg["collection_directory"],
                "namespace": self._cfg["namespace"],
            }
        )
        if collection_result.get("status") == "error":
            self._release_core()
            return self._error(
                str(collection_result.get("error_code", "collection_start_failed")),
                str(collection_result.get("error", "data collection could not start")),
            )
        with self._lock:
            self._canvas_started = True
            self._instance_id = str(args.get("instance_id") or "default").strip()
            self._wired_topics = dict(wiring["wired_topics"])
        result = self._info()
        result.update({"state": "ready", "status": "ready"})
        return result

    def _validate_wiring(self, args: dict) -> dict:
        tool = self.get_tools()[0]
        expected = {item["port"]: item["topic"] for item in tool["topic_in"]}
        bindings = args.get("input_bindings") or []
        topics = args.get("input_topics") or []
        if isinstance(topics, str):
            topics = [topics]
        if args.get("input_topic"):
            topics = [*topics, args["input_topic"]]
        if bindings:
            if not isinstance(bindings, list) or any(not isinstance(item, dict) for item in bindings):
                return self._error("invalid_canvas_wiring", "input_bindings must be an array")
            ports = [str(item.get("port", "")) for item in bindings]
            if len(ports) != len(set(ports)):
                return self._error(
                    "invalid_canvas_wiring", "input_bindings contains duplicate ports"
                )
            wired = {str(item.get("port", "")): str(item.get("topic", "")) for item in bindings}
        else:
            if not isinstance(topics, list) or any(not isinstance(item, str) for item in topics):
                return self._error("invalid_canvas_wiring", "input_topics must be an array")
            selected = {item.strip() for item in topics}
            wired = {port: topic for port, topic in expected.items() if topic in selected}
        missing = sorted(port for port, topic in expected.items() if wired.get(port) != topic)
        unknown = sorted(set(wired) - set(expected))
        if missing or unknown:
            details = []
            if missing:
                details.append("missing_or_wrong=" + ",".join(missing))
            if unknown:
                details.append("unexpected=" + ",".join(unknown))
            return self._error(
                "invalid_canvas_wiring",
                "FAST-LIVO2 requires exact LiDAR/IMU bindings (" + "; ".join(details) + ")",
            )
        return {"wired_topics": wired}

    def _stop_canvas(self) -> dict:
        with self._lock:
            core = self._core
        stop_result = None
        if core is not None:
            info = core.info()
            if info.get("active_map"):
                stop_result = core.dispatch({"action": "stop_mapping"})
            elif info.get("loaded_map"):
                stop_result = core.dispatch({"action": "unload_map"})
            collection_stop_result = core.configure_collection(
                {
                    "enabled": False,
                    "directory": self._cfg["collection_directory"],
                    "namespace": self._cfg["namespace"],
                }
            )
        else:
            collection_stop_result = None
        self._release_core()
        return {
            "state": "idle",
            "status": "idle",
            "canvas_wired": False,
            "stop_result": stop_result,
            "collection_stop_result": collection_stop_result,
            "physical_execution": False,
        }

    def _ensure_core(self) -> FastLivo2Core:
        with self._lock:
            if self._core is None:
                backend = self._backend_override or self._build_backend()
                self._core = FastLivo2Core(backend)
            return self._core

    def _release_core(self) -> None:
        with self._lock:
            core = self._core
            self._core = None
            self._canvas_started = False
            self._instance_id = ""
            self._wired_topics = {}
        if core is not None:
            try:
                core.stop()
            except Exception:
                log.exception("[fast_livo2] failed to release backend")

    def _build_backend(self):
        if self._cfg["backend"] == "disabled":
            return UnavailableFastLivo2Backend("FAST-LIVO2 backend is disabled")
        try:
            return RosTopicFastLivo2Backend(
                self._cfg, self._cfg["namespace"], self._executor
            )
        except Exception as exc:
            log.error("[fast_livo2] backend unavailable: %s", exc, exc_info=True)
            return UnavailableFastLivo2Backend(
                f"FAST-LIVO2 ROS backend unavailable: {type(exc).__name__}: {exc}"
            )

    def _info(self) -> dict:
        tool = self.get_tools()[0]
        with self._lock:
            core = self._core
            started = self._canvas_started
            instance_id = self._instance_id
            wired = dict(self._wired_topics)
            config_error = self._config_error
        result = core.info() if core else {
            "state": "error" if config_error else "idle",
            "status": "error" if config_error else "idle",
            "backend": "not_started",
            "active_map": None,
            "loaded_map": None,
            "actions": list(FAST_LIVO2_ACTIONS),
            "physical_execution": False,
        }
        if config_error:
            result.update({"error_code": "invalid_config", "error": config_error})
        result.update(
            {
                "name": "FAST-LIVO2",
                "type": "processor",
                "canvas_wired": started,
                "instance_id": instance_id or None,
                "config": dict(self._cfg),
                "topic_in": [
                    {**item, "connected": wired.get(item["port"]) == item["topic"]}
                    for item in tool["topic_in"]
                ],
                "topic_out": tool["topic_out"],
            }
        )
        return result


__all__ = ["ConfigError", "FastLivo2Plugin", "_validated_config"]
