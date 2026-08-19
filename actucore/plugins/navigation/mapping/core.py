"""ROS-independent validation and mapping lifecycle for FAST-LIVO2."""

from __future__ import annotations

import re
import math
import threading
from typing import Protocol

from .contract import FAST_LIVO2_ACTIONS


_MAP_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class FastLivo2BackendError(RuntimeError):
    def __init__(self, code: str, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


class FastLivo2Backend(Protocol):
    def info(self) -> dict: ...

    def execute(self, action: str, args: dict) -> dict: ...

    def stop(self) -> None: ...


class UnavailableFastLivo2Backend:
    def __init__(self, reason: str):
        self._reason = reason

    def info(self) -> dict:
        return {"state": "unavailable", "backend": "disabled", "reason": self._reason}

    def execute(self, action: str, args: dict) -> dict:
        del action, args
        raise FastLivo2BackendError("backend_unavailable", self._reason)

    def stop(self) -> None:
        return None


def normalize_map_name(value) -> str:
    if not isinstance(value, str):
        raise FastLivo2BackendError("invalid_argument", "map_name must be a string")
    normalized = value.strip()
    if not _MAP_NAME_RE.fullmatch(normalized):
        raise FastLivo2BackendError(
            "invalid_argument",
            "map_name must match ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$",
        )
    return normalized


def _finite_number(args: dict, key: str, minimum: float, maximum: float, *, default=None) -> float:
    raw = args.get(key, default)
    if isinstance(raw, bool):
        raise FastLivo2BackendError("invalid_argument", f"{key} must be a number")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise FastLivo2BackendError("invalid_argument", f"{key} must be a number") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise FastLivo2BackendError(
            "invalid_argument", f"{key} must be within [{minimum}, {maximum}]"
        )
    return value


class FastLivo2Core:
    def __init__(self, backend: FastLivo2Backend):
        self._backend = backend
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._collection_lock = threading.Lock()
        self._obstacle_filter_lock = threading.Lock()
        self._active_map: str | None = None
        self._loaded_map: str | None = None

    def info(self) -> dict:
        with self._lock:
            active_map = self._active_map
            loaded_map = self._loaded_map
        result = dict(self._backend.info())
        result.setdefault("state", "idle")
        result["active_map"] = active_map
        result["loaded_map"] = loaded_map
        result["actions"] = list(FAST_LIVO2_ACTIONS)
        return result

    def dispatch(self, args: dict) -> dict:
        with self._lifecycle_lock:
            return self._dispatch(args)

    def _dispatch(self, args: dict) -> dict:
        if not isinstance(args, dict):
            return self._error("", "invalid_argument", "arguments must be an object")
        action = args.get("action", "")
        if action not in FAST_LIVO2_ACTIONS:
            return self._error(str(action), "unsupported_action", "unsupported action")
        try:
            if action == "start_mapping":
                map_name = normalize_map_name(args.get("map_name"))
                with self._lock:
                    if self._active_map:
                        raise FastLivo2BackendError(
                            "mapping_active", f"mapping {self._active_map} is already active"
                        )
                    if self._loaded_map:
                        raise FastLivo2BackendError(
                            "localization_active",
                            f"map {self._loaded_map} is loaded for localization",
                        )
                result = dict(self._backend.execute(action, {"map_name": map_name}))
                if result.get("status") not in {"error", "rejected"}:
                    with self._lock:
                        self._active_map = map_name
                return self._result(action, result)

            if action == "load_map":
                map_name = normalize_map_name(args.get("map_name"))
                with self._lock:
                    previous_map = self._loaded_map
                    if self._active_map:
                        raise FastLivo2BackendError(
                            "mapping_active", f"mapping {self._active_map} is already active"
                        )
                result = dict(self._backend.execute(action, {"map_name": map_name}))
                if result.get("status") not in {"error", "rejected"}:
                    with self._lock:
                        self._loaded_map = str(result.get("loaded_map") or map_name)
                    result.setdefault("replaced_map", previous_map)
                return self._result(action, result)

            if action == "relocalize":
                with self._lock:
                    loaded_map = self._loaded_map
                if not loaded_map:
                    raise FastLivo2BackendError(
                        "map_not_loaded", "load a saved map before relocalizing"
                    )
                request = {
                    "map_name": loaded_map,
                    "initial_x": _finite_number(args, "initial_x", -1000.0, 1000.0),
                    "initial_y": _finite_number(args, "initial_y", -1000.0, 1000.0),
                    "initial_z": _finite_number(args, "initial_z", -10.0, 10.0, default=0.0),
                    "initial_yaw": _finite_number(args, "initial_yaw", -math.pi, math.pi),
                    "search_xy_m": _finite_number(args, "search_xy_m", 0.1, 3.0, default=1.0),
                    "search_yaw_rad": _finite_number(
                        args, "search_yaw_rad", 0.05, math.pi / 2.0, default=0.35
                    ),
                }
                return self._result(action, dict(self._backend.execute(action, request)))

            with self._lock:
                active_map = self._active_map
            if not active_map:
                return self._result(
                    action,
                    {"status": "stopped", "already_idle": True, "map_name": None},
                )
            result = dict(self._backend.execute(action, {"map_name": active_map}))
            if result.get("status") in {"stopped", "saved", "idle"}:
                with self._lock:
                    self._active_map = None
            return self._result(action, result)
        except FastLivo2BackendError as exc:
            if "loaded_map" in exc.details:
                with self._lock:
                    loaded_map = exc.details.get("loaded_map")
                    self._loaded_map = (
                        str(loaded_map) if isinstance(loaded_map, str) else None
                    )
            error = self._error(str(action), exc.code, str(exc))
            for key in (
                "loaded_map",
                "runtime_mode",
                "replaced_map",
                "rollback_status",
                "rollback_error",
                "retryable",
            ):
                if key in exc.details:
                    error[key] = exc.details[key]
            return error
        except Exception as exc:
            return self._error(str(action), "backend_error", f"{type(exc).__name__}: {exc}")

    def stop_localization(self) -> dict:
        """Stop the private localization runtime without exposing unload_map."""
        with self._lifecycle_lock:
            return self._stop_localization()

    def _stop_localization(self) -> dict:
        with self._lock:
            loaded_map = self._loaded_map
        if not loaded_map:
            return {"status": "idle", "already_idle": True, "map_name": None}
        try:
            result = dict(
                self._backend.execute("unload_map", {"map_name": loaded_map})
            )
            if result.get("status") not in {"unloaded", "stopped", "idle"}:
                return self._error(
                    "stop_localization",
                    "localization_stop_unconfirmed",
                    f"failed to stop localization for map {loaded_map}",
                )
        except FastLivo2BackendError as exc:
            if "loaded_map" in exc.details:
                with self._lock:
                    restored = exc.details.get("loaded_map")
                    self._loaded_map = (
                        str(restored) if isinstance(restored, str) else None
                    )
            error = self._error("stop_localization", exc.code, str(exc))
            for key in ("loaded_map", "runtime_mode", "retryable"):
                if key in exc.details:
                    error[key] = exc.details[key]
            return error
        except Exception as exc:
            return self._error(
                "stop_localization",
                "backend_error",
                f"{type(exc).__name__}: {exc}",
            )
        with self._lock:
            self._loaded_map = None
        return result

    def configure_collection(self, config: dict) -> dict:
        """Apply private recorder configuration without exposing a public action."""
        with self._collection_lock:
            try:
                return dict(self._backend.execute("configure_collection", dict(config)))
            except FastLivo2BackendError as exc:
                error = self._error("configure_collection", exc.code, str(exc))
                if "retryable" in exc.details:
                    error["retryable"] = exc.details["retryable"]
                return error
            except Exception as exc:
                return self._error(
                    "configure_collection",
                    "backend_error",
                    f"{type(exc).__name__}: {exc}",
                )

    def configure_obstacle_filter(self, config: dict) -> dict:
        """Apply the private height band before Nav2 acquires its inputs."""
        with self._obstacle_filter_lock:
            try:
                return dict(
                    self._backend.execute("configure_obstacle_filter", dict(config))
                )
            except FastLivo2BackendError as exc:
                error = self._error(
                    "configure_obstacle_filter", exc.code, str(exc)
                )
                if "retryable" in exc.details:
                    error["retryable"] = exc.details["retryable"]
                return error
            except Exception as exc:
                return self._error(
                    "configure_obstacle_filter",
                    "backend_error",
                    f"{type(exc).__name__}: {exc}",
                )

    def stop(self) -> None:
        self._backend.stop()

    @staticmethod
    def _result(action: str, raw: dict) -> dict:
        result = dict(raw)
        result.setdefault("action", action)
        result.setdefault("status", "ok")
        return result

    @staticmethod
    def _error(action: str, code: str, message: str) -> dict:
        return {"action": action, "status": "error", "error_code": code, "error": message}


__all__ = [
    "FastLivo2BackendError",
    "FastLivo2Core",
    "UnavailableFastLivo2Backend",
    "normalize_map_name",
]
