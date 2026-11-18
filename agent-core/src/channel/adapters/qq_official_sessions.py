"""Durable, private QQ Gateway resume state for one AppID.

The official SDK implements WebSocket RESUME. This module persists the
opaque session ID and sequence across Core restarts without exposing it to
Channel configuration, Dashboard, logs, or outbound messages.
"""
from __future__ import annotations

import os
from pathlib import Path
from threading import RLock


class QQSessionState:
    def __init__(self, app_id: str, directory: str = "/work/resource/qq_sessions"):
        from qqbot_agent_sdk import WSSessionStore

        self.app_id = app_id
        self.path = Path(directory)
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.path, 0o700)
        self.filename = "gateway-" + app_id + ".json"
        self._store = WSSessionStore(str(self.path), filename=self.filename)
        self._lock = RLock()

    def read(self) -> tuple[str | None, int | None]:
        """Only resume an SDK-declared resumable, unexpired session."""
        with self._lock:
            state = self._store.get(self.app_id)
            if state.is_resumable and state.is_fresh():
                return state.session_id, state.seq
            return None, None

    def save(self, session_id: str | None, seq: int | None) -> None:
        with self._lock:
            if session_id:
                self._store.save(self.app_id, session_id, seq)
                os.chmod(self.path / self.filename, 0o600)
            else:
                self._store.clear(self.app_id)

    def clear(self) -> None:
        self.save(None, None)
