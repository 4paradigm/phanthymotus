"""Per-view simulation bindings; the remote service owns controller leases."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

from .client import ServiceError


class WorkbenchSession:
    def __init__(self, root, endpoint):
        self.path = Path(root).expanduser() / (hashlib.sha256(endpoint.encode()).hexdigest() + ".json")
        self.current = None
        self._views = {}
        self._grants = {}
        self._requests = {}
        self._seen_sessions = set()
        try:
            value = json.loads(self.path.read_text())
            if (value.get("endpoint") == endpoint and isinstance(value.get("session"), dict)
                    and isinstance(value["session"].get("id"), str)):
                self.current = value["session"]
        except (OSError, ValueError, AttributeError):
            pass
        self.endpoint = endpoint

    @property
    def id(self):
        return self.current.get("id") if self.current else None

    def remember(self, session):
        if session is not None and (not isinstance(session, dict) or not isinstance(session.get("id"), str)):
            raise ServiceError("服务器未返回仿真会话标识")
        # Persist ownership before exposing a newly created session to the UI.
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self.path.with_name(self.path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump({"endpoint": self.endpoint, "session": session}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self.current = session

    def for_view(self, view_id=None):
        if self.current:
            self._seen_sessions.add(self.current["id"])
        if view_id:
            return self._views.setdefault(view_id, self.current)
        return self.current

    def assert_switchable(self, client):
        """Stale viewers are not ownership; only the service can prove a lease ended."""
        bound = {item["id"] for item in self._views.values() if item}
        if self.id:
            bound.add(self.id)
        if not (bound or self._seen_sessions or self._grants):
            return
        sessions = client.call("simulation_list").get("sessions")
        if not isinstance(sessions, list) or any(not isinstance(item, dict) or not isinstance(item.get("id"), str) for item in sessions):
            raise ServiceError("服务器返回了无效会话列表")
        local_views = set(self._views) | set(self._grants)
        for session in sessions:
            if session.get("state") == "closed":
                continue
            control = session.get("control")
            if isinstance(control, dict):
                if control.get("owner_view_id") in local_views:
                    raise ServiceError("仍有窗口在控制仿真，请先释放控制后切换服务器")
            elif session["id"] in bound:
                # Old sessions have no lease, so their prior close requirement remains.
                raise ServiceError("请先结束当前试跑，再切换服务器")

    def inherit_disconnected_views(self, previous):
        # Explicit None prevents an old window from falling back to a session
        # previously saved on the new server, even if session IDs collide.
        self._views = dict.fromkeys(previous._views)
        self._requests = previous._requests


    def refresh(self, client, view_id=None):
        if not self.id and not any(self._views.values()):
            return
        result = client.call("simulation_list", **({"view_id": view_id} if view_id else {}))
        sessions = result.get("sessions")
        if not isinstance(sessions, list) or any(not isinstance(s, dict) or not isinstance(s.get("id"), str) for s in sessions):
            raise ServiceError("服务器返回了无效会话列表")
        active = {s["id"]: s for s in sessions if s.get("state") != "closed"}
        current = active.get(self.id)
        if current != self.current:
            self.remember(current)
        for view, bound in self._views.items():
            self._views[view] = active.get(bound["id"]) if bound else None
        self._grants = {view: grant for view, grant in self._grants.items() if grant["id"] in active}

    def pause_owned(self, client):
        if self._grants:
            for view, grant in list(self._grants.items()):
                try:
                    client.call("simulation_pause", session_id=grant["id"], view_id=view,
                                control_epoch=grant["epoch"])
                except ServiceError as exc:
                    if "控制权已变化" not in str(exc):
                        raise
                    self._grants.pop(view, None)
        elif self.current and not self.current.get("control"):
            client.call("simulation_pause", session_id=self.id)

    def call(self, client, action, values):
        values = dict(values)
        view = values.get("view_id")
        supplied = values.pop("session_id", None)
        bound = self.for_view(view)
        sid = bound.get("id") if bound else None
        if action == "simulation_list":
            return client.call(action, **values)
        if action == "simulation_detach":
            if view:
                self._views[view] = None  # Leaving a viewer never changes another view's binding.
                self._grants.pop(view, None)
                if self.id == sid and not any(item and item.get("id") == sid for item in self._views.values()):
                    self.remember(None)
            elif self.current and not self.current.get("control"):
                self.remember(None)
            return {"session": None}
        if action == "simulation_attach":
            if not supplied:
                raise ServiceError("请选择仿真会话")
            result = client.call("simulation_inspect", session_id=supplied,
                                 **({"view_id": view} if view else {}))
            session = result.get("session") or {}
            if (session.get("state") == "closed" or not values.get("model_id")
                    or any(session.get(k) != values.get(k) for k in ("profile_id", "model_id"))):
                raise ServiceError("此会话与当前动作版本或机器人不匹配")
            self._seen_sessions.add(session["id"])
            if view:
                self._views[view] = session
            else:
                self.remember(session)
            return result
        if supplied and supplied != sid:
            raise ServiceError("仿真会话与当前工作台不匹配")
        if action == "simulation_create":
            request_key = (view, values.get("request_id"))
            if view and isinstance(request_key[1], str):
                endpoint = self._requests.setdefault(request_key, self.endpoint)
                if endpoint != self.endpoint:
                    raise ServiceError("此创建请求属于之前的服务器，请重新发起试跑")
            if not values.get("model_id"):
                raise ServiceError("请先选择这个动作的训练模型")
            self.refresh(client, view)
            bound = self.for_view(view)
            if bound:
                if all(bound.get(k) == values.get(k) for k in ("profile_id", "model_id")):
                    return client.call("simulation_inspect", session_id=bound["id"],
                                       **({"view_id": view} if view else {}))
                raise ServiceError("请先离开当前试跑，再切换模型")
            if view:
                values["origin"] = "motion"
            result = client.call(action, **values)
            session = result.get("session")
            try:
                self.remember(session)
            except OSError:
                if isinstance(session, dict) and session.get("id"):
                    control = result.get("control")
                    fence = {"view_id": view, "control_epoch": control["epoch"]} if control else {}
                    client.call("simulation_close", session_id=session["id"], **fence)
                raise ServiceError("无法保存仿真会话，请检查工作台存储目录")
        else:
            if not sid:
                raise ServiceError("请先点击在仿真里试跑")
            result = client.call(action, session_id=sid, **values)
        session = result.get("session")
        if action == "simulation_close":
            if self.id == sid:
                self.remember(None)
            for key, item in self._views.items():
                if item and item.get("id") == sid:
                    self._views[key] = None
            self._grants = {key: item for key, item in self._grants.items() if item["id"] != sid}
        elif session:
            self._seen_sessions.add(session["id"])
            if view:
                self._views[view] = session
            elif not session.get("control"):
                self.current = session
            control = result.get("control") or session.get("control")
            if view and control and control.get("is_controller"):
                self._grants[view] = {"id": session["id"], "epoch": control["epoch"]}
            elif view:
                self._grants.pop(view, None)
        return result
