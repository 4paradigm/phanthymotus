"""ROS-independent validation and mapping lifecycle for FAST-LIVO2."""

from __future__ import annotations

import re
import threading
from typing import Protocol

from .contract import FAST_LIVO2_ACTIONS


_MAP_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class FastLivo2BackendError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


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


class FastLivo2Core:
    def __init__(self, backend: FastLivo2Backend):
        self._backend = backend
        self._lock = threading.Lock()
        self._active_map: str | None = None

    def info(self) -> dict:
        with self._lock:
            active_map = self._active_map
        result = dict(self._backend.info())
        result.setdefault("state", "idle")
        result["active_map"] = active_map
        result["actions"] = list(FAST_LIVO2_ACTIONS)
        return result

    def dispatch(self, args: dict) -> dict:
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
                result = dict(self._backend.execute(action, {"map_name": map_name}))
                if result.get("status") not in {"error", "rejected"}:
                    with self._lock:
                        self._active_map = map_name
                return self._result(action, result)

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
            return self._error(str(action), exc.code, str(exc))
        except Exception as exc:
            return self._error(str(action), "backend_error", f"{type(exc).__name__}: {exc}")

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
