from __future__ import annotations

import json
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol


REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{5,127}$")


class BridgeFailure(RuntimeError):
    def __init__(self, code: str, message: str, *, uncertain: bool = False):
        super().__init__(message)
        self.code = code
        self.uncertain = uncertain


@dataclass(frozen=True)
class Visit:
    visit_id: str
    name: str
    phone_tail: str = ""
    host: str = ""
    visit_time: str = ""
    reason: str = ""
    status: str = ""
    booking_code: str = ""

    def public(self, ticket: str) -> dict:
        return {
            "ticket": ticket,
            "name": self.name,
            "tail": self.phone_tail[-4:],
            "host": self.host,
            "time": self.visit_time,
            "reason": self.reason,
            "status": self.status,
        }


class VisitorBackend(Protocol):
    def find_visits(
        self,
        name: str,
        host: str = "",
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> list[Visit]: ...

    def get_status(self, visit: Visit) -> str: ...

    def print_visit(self, visit: Visit) -> None: ...


class BridgeStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS tickets (
                ticket TEXT PRIMARY KEY,
                visit_json TEXT NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                event TEXT NOT NULL,
                request_id TEXT NOT NULL DEFAULT '',
                ticket_tail TEXT NOT NULL DEFAULT '',
                result TEXT NOT NULL
            );
            """
        )
        self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def issue_ticket(self, visit: Visit, ttl_seconds: float) -> str:
        ticket = secrets.token_urlsafe(24)
        with self._lock:
            self._db.execute(
                "INSERT INTO tickets(ticket, visit_json, expires_at) VALUES (?, ?, ?)",
                (ticket, json.dumps(asdict(visit), ensure_ascii=False), time.time() + ttl_seconds),
            )
            self._db.commit()
        return ticket

    def resolve_ticket(self, ticket: str) -> Visit | None:
        with self._lock:
            row = self._db.execute(
                "SELECT visit_json, expires_at FROM tickets WHERE ticket = ?", (ticket,)
            ).fetchone()
        if row is None or float(row["expires_at"]) <= time.time():
            return None
        return Visit(**json.loads(row["visit_json"]))

    def get_request(self, request_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT result_json FROM requests WHERE request_id = ?", (request_id,)
            ).fetchone()
        return json.loads(row["result_json"]) if row else None

    def save_request(self, request_id: str, result: dict) -> None:
        payload = json.dumps(result, ensure_ascii=False)
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO requests(request_id, result_json, created_at) VALUES (?, ?, ?)",
                (request_id, payload, time.time()),
            )
            self._db.commit()

    def audit(self, event: str, result: str, *, request_id: str = "", ticket: str = "") -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO audit(created_at, event, request_id, ticket_tail, result) VALUES (?, ?, ?, ?, ?)",
                (time.time(), event, request_id, ticket[-6:], result),
            )
            self._db.commit()


class VisitorBridge:
    def __init__(
        self,
        backend: VisitorBackend,
        store: BridgeStore,
        *,
        ticket_seconds: float = 300,
        printable_statuses: tuple[str, ...] = ("未入场",),
        success_statuses: tuple[str, ...] = ("已入场",),
    ):
        self.backend = backend
        self.store = store
        self.ticket_seconds = ticket_seconds
        self.printable_statuses = printable_statuses
        self.success_statuses = success_statuses
        self._print_lock = threading.Lock()

    def find_visit(self, name: str, host: str = "", start_time=None, end_time=None) -> dict:
        name = str(name or "").strip()
        host = str(host or "").strip()
        if not name:
            raise BridgeFailure("bad_request", "访客姓名不能为空")
        if len(name) > 32 or len(host) > 64:
            raise BridgeFailure("bad_request", "姓名或被访人信息不能过长")
        visits = self.backend.find_visits(name, host, start_time, end_time)
        public = []
        for visit in visits:
            ticket = self.store.issue_ticket(visit, self.ticket_seconds)
            public.append(visit.public(ticket))
        self.store.audit("find_visit", f"count={len(public)}")
        return {"count": len(public), "visits": public}

    def list_visits(self) -> dict:
        """Return today's printable visits; tickets stay inside the trusted card."""
        today = datetime.now().date().isoformat()
        visits = self.backend.find_visits(
            "", "", f"{today} 00:00:00", f"{today} 23:59:59"
        )
        printable = [visit for visit in visits if visit.status in self.printable_statuses]
        printable.sort(key=lambda visit: (visit.visit_time, visit.name, visit.host, visit.phone_tail))
        public = [
            visit.public(self.store.issue_ticket(visit, self.ticket_seconds))
            for visit in printable
        ]
        self.store.audit("list_visits", f"count={len(public)}")
        return {"date": today, "count": len(public), "visits": public}

    def print_tag(self, ticket: str, request_id: str, printer_name: str = "") -> dict:
        ticket = str(ticket or "").strip()
        request_id = str(request_id or "").strip()
        if not ticket or not REQUEST_ID_RE.fullmatch(request_id):
            raise BridgeFailure("bad_request", "缺少有效的 ticket 或 request_id")

        with self._print_lock:
            previous = self.store.get_request(request_id)
            if previous is not None:
                return previous

            visit = self.store.resolve_ticket(ticket)
            if visit is None:
                result = {"ok": False, "code": "ticket_expired", "error": "查询票据无效或已过期，请重新查询访客"}
                self._finish(request_id, ticket, result)
                return result

            status = self.backend.get_status(visit)
            if status not in self.printable_statuses:
                result = {
                    "ok": False,
                    "code": "status_not_printable",
                    "error": f"访客当前状态为“{status or '未知'}”，不能打印访客贴",
                }
                self._finish(request_id, ticket, result)
                return result

            try:
                if printer_name:
                    self.backend.print_visit(visit, printer_name=printer_name)
                else:
                    self.backend.print_visit(visit)
            except BridgeFailure as exc:
                # A click or network timeout is never retried. Re-read state once;
                # if the platform already moved to 已入场, the print action won.
                if exc.uncertain:
                    try:
                        after = self.backend.get_status(visit)
                    except Exception:
                        after = ""
                    if after in self.success_statuses:
                        result = {"ok": True}
                    else:
                        result = {"ok": False, "code": exc.code, "error": str(exc)}
                else:
                    result = {"ok": False, "code": exc.code, "error": str(exc)}
                self._finish(request_id, ticket, result)
                return result

            try:
                after = self.backend.get_status(visit)
            except Exception:
                after = ""
            if after not in self.success_statuses:
                result = {
                    "ok": False,
                    "code": "status_unconfirmed",
                    "error": f"打印动作已发出，但访客状态为“{after or '未知'}”；系统不会自动重试，请前台核对",
                }
            else:
                result = {"ok": True}
            self._finish(request_id, ticket, result)
            return result

    def _finish(self, request_id: str, ticket: str, result: dict) -> None:
        self.store.save_request(request_id, result)
        self.store.audit(
            "print_tag",
            "ok" if result.get("ok") else str(result.get("code", "failed")),
            request_id=request_id,
            ticket=ticket,
        )
