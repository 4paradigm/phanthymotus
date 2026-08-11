"""ROS-independent validation and lifecycle core for Nav2."""

from __future__ import annotations

import math
import threading
import uuid
from typing import Protocol

from .contract import NAV2_ACTIONS


class NavigationBackendError(RuntimeError):
    """A fail-closed backend error suitable for an MCP response."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class NavigationBackend(Protocol):
    def info(self) -> dict:
        """Return current backend state and capabilities."""

    def execute(self, action: str, args: dict, *, nav_id: str | None) -> dict:
        """Execute one already validated action."""

    def stop(self) -> None:
        """Release backend resources without changing robot state."""


class UnavailableNavigationBackend:
    """Backend used when configuration or ROS dependencies are unavailable."""

    def __init__(self, reason: str):
        self._reason = reason

    def info(self) -> dict:
        return {
            "state": "unavailable",
            "backend": "disabled",
            "shadow_only": True,
            "physical_execution": False,
            "reason": self._reason,
        }

    def execute(self, action: str, args: dict, *, nav_id: str | None) -> dict:
        del action, args, nav_id
        raise NavigationBackendError("backend_unavailable", self._reason)

    def stop(self) -> None:
        return None


_NAVIGATE_ACTIONS = {"navigate_to_pose"}
_NAV_CONTROL_ACTIONS = {
    "wait_navigation_done",
    "pause_nav",
    "resume_nav",
    "stop_nav",
}
_TERMINAL_STATUSES = {
    "arrived",
    "succeeded",
    "cancelled",
    "stopped",
    "timeout",
    "error",
    "aborted",
    "rejected",
}


def _number(args: dict, key: str, *, default=None) -> float:
    if key not in args:
        if default is None:
            raise NavigationBackendError("missing_argument", f"{key} is required")
        raw = default
    else:
        raw = args[key]
    if isinstance(raw, bool):
        raise NavigationBackendError("invalid_argument", f"{key} must be a number")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise NavigationBackendError(
            "invalid_argument", f"{key} must be a number"
        ) from exc
    if not math.isfinite(value):
        raise NavigationBackendError("invalid_argument", f"{key} must be finite")
    return value


def _normalize(action: str, args: dict) -> dict:
    normalized: dict = {}
    if action == "navigate_to_pose":
        normalized["x"] = _number(args, "x")
        normalized["y"] = _number(args, "y")
        normalized["yaw"] = _number(args, "yaw")

    if action in _NAVIGATE_ACTIONS:
        speed = _number(args, "speed", default=0.15)
        if not 0.10 <= speed <= 0.15:
            raise NavigationBackendError(
                "invalid_argument", "speed must be within [0.10, 0.15] m/s"
            )
        raw_mode = args.get("mode", 0)
        if isinstance(raw_mode, bool) or not isinstance(raw_mode, int):
            raise NavigationBackendError(
                "invalid_argument", "mode must be integer 0"
            )
        if raw_mode != 0:
            raise NavigationBackendError(
                "invalid_argument", "mode must be 0 (detour)"
            )
        normalized["speed"] = speed
        normalized["mode"] = raw_mode

    if action == "wait_navigation_done":
        timeout = _number(args, "stall_timeout", default=90.0)
        if not 1.0 <= timeout <= 3600.0:
            raise NavigationBackendError(
                "invalid_argument", "stall_timeout must be within [1, 3600] seconds"
            )
        normalized["stall_timeout"] = timeout

    return normalized


def _trusted_nav_id(args: dict) -> str | None:
    """Read the private task lease injected by the trusted Agent Core."""

    if "_control_nav_id" not in args:
        return None
    raw = args.get("_control_nav_id")
    if not isinstance(raw, str):
        raise NavigationBackendError(
            "invalid_control_nav_id", "_control_nav_id must be a string"
        )
    value = raw.strip()
    if not value or len(value) > 128 or any(ord(char) < 33 for char in value):
        raise NavigationBackendError(
            "invalid_control_nav_id", "_control_nav_id is invalid"
        )
    return value


class Nav2Core:
    """Validate the frozen contract and serialize one active navigation task."""

    def __init__(self, backend: NavigationBackend):
        self._backend = backend
        self._lock = threading.Lock()
        self._active_nav_id: str | None = None

    def info(self) -> dict:
        with self._lock:
            active_nav_id = self._active_nav_id
        result = dict(self._backend.info())
        result.setdefault("state", "idle")
        result["active_nav_id"] = active_nav_id
        result["actions"] = list(NAV2_ACTIONS)
        return result

    def dispatch(self, args: dict) -> dict:
        if not isinstance(args, dict):
            return self._error("", "invalid_argument", "arguments must be an object")
        action = args.get("action", "")
        if not isinstance(action, str) or action not in NAV2_ACTIONS:
            return self._error(
                str(action),
                "unsupported_action",
                f"action must be one of the {len(NAV2_ACTIONS)} business actions",
            )
        try:
            trusted_nav_id = _trusted_nav_id(args)
            normalized = _normalize(action, args)
            return self._dispatch_validated(
                action, normalized, trusted_nav_id=trusted_nav_id
            )
        except NavigationBackendError as exc:
            return self._error(action, exc.code, str(exc))
        except Exception as exc:
            return self._error(
                action,
                "backend_error",
                f"{type(exc).__name__}: {exc}",
            )

    def stop(self) -> None:
        self._backend.stop()

    def _dispatch_validated(
        self, action: str, args: dict, *, trusted_nav_id: str | None
    ) -> dict:
        if action in _NAVIGATE_ACTIONS:
            return self._start_navigation(
                action, args, trusted_nav_id=trusted_nav_id
            )

        if action in _NAV_CONTROL_ACTIONS:
            return self._control_navigation(action, args)

        return self._result(action, self._backend.execute(action, args, nav_id=None))

    def _start_navigation(
        self, action: str, args: dict, *, trusted_nav_id: str | None
    ) -> dict:
        nav_id = trusted_nav_id or uuid.uuid4().hex
        with self._lock:
            if self._active_nav_id:
                raise NavigationBackendError(
                    "navigation_active",
                    f"navigation {self._active_nav_id} is already active",
                )
            self._active_nav_id = nav_id
        try:
            result = self._result(
                action,
                self._backend.execute(action, args, nav_id=nav_id),
                nav_id=nav_id,
            )
        except Exception:
            with self._lock:
                if self._active_nav_id == nav_id:
                    self._active_nav_id = None
            raise
        if result.get("status") in _TERMINAL_STATUSES:
            with self._lock:
                if self._active_nav_id == nav_id:
                    self._active_nav_id = None
        return result

    def _control_navigation(self, action: str, args: dict) -> dict:
        with self._lock:
            nav_id = self._active_nav_id

        if not nav_id:
            if action == "stop_nav":
                return {
                    "action": action,
                    "status": "stopped",
                    "nav_id": None,
                    "already_idle": True,
                }
            raise NavigationBackendError(
                "no_active_navigation", f"{action} requires an active navigation"
            )

        result = self._result(
            action,
            self._backend.execute(action, args, nav_id=nav_id),
            nav_id=nav_id,
        )
        if result.get("status") in _TERMINAL_STATUSES:
            with self._lock:
                if self._active_nav_id == nav_id:
                    self._active_nav_id = None
        return result

    @staticmethod
    def _result(action: str, raw: dict | None, *, nav_id: str | None = None) -> dict:
        result = dict(raw or {})
        result.setdefault("status", "ok")
        result.setdefault("action", action)
        if nav_id is not None:
            result.setdefault("nav_id", nav_id)
        return result

    @staticmethod
    def _error(action: str, code: str, message: str) -> dict:
        return {
            "action": action,
            "status": "error",
            "error_code": code,
            "error": message,
        }
