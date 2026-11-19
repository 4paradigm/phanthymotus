"""通用访客贴打印卡片（QL_printer），独立于机器人硬件驱动。

卡片本身不碰打印机，也不直接登录范式+：它只向前台电脑上的「桥接服务」发 HTTP
请求，由桥接服务去范式+ 查访客、调用前台电脑上的 USB 打印机打印访客贴。

    机器人 skill → QL_printer 卡片 ──HTTP──> 前台电脑桥接服务 → 范式+ / USB 打印机

公共服务部署和迁移说明见 docs/ql-printer.md。
"""

from __future__ import annotations

import json
import base64
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any

def action_schema(actions, properties):
    return {"type": "object", "properties": {"action": {"type": "string", "enum": list(actions)}, **properties},
            "required": ["action"], "x-action-params": {name: {"params": params, "description": desc}
            for name, (params, desc) in actions.items()}}


def tool(name, kind, description, schema):
    return {"name": name, "type": kind, "multiInstance": False,
            "description": description, "inputSchema": schema}


TOOL_NAME = "QL_printer"

_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


class BridgeError(Exception):
    """桥接服务调用失败。message 是可以直接转述给人的中文说明。"""

    def __init__(self, message: str, code: str = "bridge_error"):
        super().__init__(message)
        self.code = code


class BridgeClient:
    """前台电脑桥接服务的最小 HTTP 客户端（只用标准库）。"""

    def __init__(self, base_url: str, token: str = "", timeout_s: float = 8.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s

    def request(
        self,
        method: str,
        path: str,
        *,
        query: dict | None = None,
        body: dict | None = None,
        timeout_s: float | None = None,
        retry_refused: bool = True,
    ) -> dict:
        url = self.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v not in (None, "")})
        data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        raw = b""
        for attempt in range(2):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(
                    req, timeout=self.timeout_s if timeout_s is None else timeout_s
                ) as resp:
                    raw = resp.read()
                break
            except urllib.error.HTTPError as exc:
                payload = _safe_json(exc.read())
                if exc.code in (401, 403):
                    raise BridgeError("桥接服务拒绝了请求：令牌不对，请检查卡片配置里的访问令牌", "unauthorized")
                if exc.code == 404 and not payload:
                    raise BridgeError("桥接服务没有这个接口，请确认前台电脑上运行的是访客贴桥接服务", "not_found")
                message = str(payload.get("error") or f"桥接服务返回 HTTP {exc.code}")
                raise BridgeError(message, str(payload.get("code") or f"http_{exc.code}"))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                refused = getattr(reason, "errno", None) in (111, 10061) or "refused" in str(reason).lower()
                if attempt == 0 and refused and retry_refused:
                    # The Windows watchdog may be starting the bridge. A
                    # refused connection is known not to have reached the
                    # server, so this single retry cannot duplicate a print.
                    time.sleep(8)
                    continue
                raise BridgeError(
                    f"连不上前台电脑的桥接服务（{self.base_url}）：{reason}。请确认前台电脑开着、桥接服务在运行、网络通",
                    "unreachable",
                )
        payload = _safe_json(raw)
        if not isinstance(payload, dict):
            raise BridgeError("桥接服务返回的不是 JSON 对象", "bad_response")
        return payload


def _safe_json(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _visit(item: dict) -> dict:
    """只保留桥接协议允许交给 skill 的字段。"""
    return {
        "ticket": str(item.get("ticket", "")),
        "name": str(item.get("name", "")),
        "tail": str(item.get("tail", ""))[-4:],
        "host": str(item.get("host", "")),
        "time": str(item.get("time", "")),
        "reason": str(item.get("reason", "")),
        "status": str(item.get("status", "")),
    }


class QLPrinterPlugin:
    PREFIX = TOOL_NAME

    def __init__(self, plugin_config: dict | None = None, executor=None):
        plugin_config = plugin_config or {}
        self._lock = threading.Lock()
        self._cfg = {
            "printer_name": str(plugin_config.get("printer_name", "") or ""),
            "bridge_url": str(plugin_config.get("bridge_url", "") or ""),
            "token": str(plugin_config.get("token", "") or ""),
            "timeout_s": float(plugin_config.get("timeout_s", 8)),
            "print_timeout_s": float(plugin_config.get("print_timeout_s", 60)),
            "remote_start_enabled": bool(plugin_config.get("remote_start_enabled", False)),
            "remote_start_user": str(plugin_config.get("remote_start_user", "4paradigm") or "4paradigm"),
            "remote_start_port": int(plugin_config.get("remote_start_port", 22)),
            "remote_start_key_path": str(
                plugin_config.get(
                    "remote_start_key_path",
                    "/run/ql-printer-auth/frontdesk_bridge_ed25519",
                )
            ),
        }
        self._direct_request_ids: dict[tuple[str, str, str, str], str] = {}
        self._deploy_state = {"status": "idle"}

    # ── 卡片定义 ──────────────────────────────────────────────────────────
    def get_tool(self) -> dict:
        actions = {
            "start": ([], "启动卡片"),
            "stop": ([], "停止卡片"),
            "info": ([], "查看卡片配置是否完整"),
            "config": (["bridge_url", "token", "timeout_s", "print_timeout_s", "remote_start_enabled", "remote_start_user", "remote_start_port", "remote_start_key_path"], "保存桥接服务配置（平台在 start 前自动调用）"),
            "list": ([], "列出今天全部未入场访客，供打印前核对姓名、被访人、时间和手机号后四位"),
            "print": (["name", "host"], "从今天未入场访客列表中按姓名和被访人唯一匹配并打印；ticket 和防重复编号由卡片内部处理"),
        }
        properties = {
            "printer_name": {"type": "string"},
            "name": {"type": "string", "description": "访客姓名"},
            "host": {"type": "string", "description": "被访人姓名；同名访客时用于缩小范围"},
            "ticket": {"type": "string", "description": "桥接内部使用的短期票据，不能自己编，也不能改写"},
            "request_id": {
                "type": "string",
                "description": "本次打印的唯一编号；同一次请求重试时必须复用",
            },
            "bridge_url": {"type": "string"},
            "token": {"type": "string"},
            "timeout_s": {"type": "number"},
            "print_timeout_s": {"type": "number"},
            "remote_start_enabled": {"type": "boolean"},
            "remote_start_user": {"type": "string"},
            "remote_start_port": {"type": "integer"},
            "remote_start_key_path": {"type": "string"},
        }
        actions["config"][0].append("printer_name")
        actions["printers"] = (["bridge_url", "token"], "读取目标电脑已安装的打印机")
        actions["initialize"] = (["bridge_url", "token", "remote_start_user", "remote_start_port", "remote_start_key_path"], "通过管理员 SSH 初始化目标电脑，仅在安装设置时使用")
        actions["initialize_status"] = ([], "读取初始化进度")
        definition = tool(
            TOOL_NAME,
            "actuator",
            "访客贴打印：通过前台电脑上的桥接服务查询范式+ 里的访客预约，并用前台的打印机打印访客贴。"
            "先 list 查看今天全部未入场访客，再用 name 和 host 调用 print；不要按猜测的名字打印。",
            action_schema(actions, properties),
        )
        definition["inputSchema"]["properties"]["printer_name"] = {"type": "string"}
        definition["configSchema"] = {
            "type": "object",
            "properties": {
                # The printer belongs to this canvas card, not to the
                # perception process as a whole.  Marking it as instance
                # configuration also gives the placed card its own settings
                # button; the instance dialog deliberately includes the
                # shared bridge address/token above this field, so an operator
                # can finish the whole setup from the card in one place.
                "printer_name": {"type": "string", "title": "目标打印机", "scope": "instance", "format": "bridge-printer"},
                "bridge_url": {
                    "type": "string", "title": "桥接服务地址", "scope": "shared", "x-sensitive": True,
                    "description": "前台电脑上桥接服务的地址，例如 http://192.168.1.20:8765（前台电脑要设固定 IP）",
                },
                "token": {
                    "type": "string", "title": "访问令牌", "scope": "shared", "format": "password",
                    "description": "和桥接服务里设置的令牌一致，防止局域网里别人随便打印",
                },
                "timeout_s": {
                    "type": "number", "title": "查询超时（秒）", "scope": "shared",
                    "minimum": 2, "maximum": 30, "default": 8, "advanced": True,
                },
                "print_timeout_s": {
                    "type": "number", "title": "打印超时（秒）", "scope": "shared",
                    "minimum": 10, "maximum": 120, "default": 60, "advanced": True,
                    "description": "等待前台网页完成定位、打印并确认状态的最长时间",
                },
                "remote_start_enabled": {
                    "type": "boolean", "title": "机器人远程启动桥接", "scope": "shared",
                    "default": False, "advanced": True,
                    "description": "桥接未运行时，通过受限 SSH 密钥触发前台 Windows 启动任务",
                },
                "remote_start_user": {
                    "type": "string", "title": "前台 SSH 用户", "scope": "shared",
                    "default": "4paradigm", "advanced": True,
                },
                "remote_start_port": {
                    "type": "integer", "title": "前台 SSH 端口", "scope": "shared",
                    "minimum": 1, "maximum": 65535, "default": 22, "advanced": True,
                },
                "remote_start_key_path": {
                    "type": "string", "title": "机器人 SSH 私钥路径", "scope": "shared",
                    "default": "/run/ql-printer-auth/frontdesk_bridge_ed25519", "advanced": True,
                },
            },
            "required": ["bridge_url", "token"],
        }
        return definition

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    # ── 调度 ──────────────────────────────────────────────────────────────
    def get_tools(self) -> list:
        return [self.get_tool()]

    def dispatch(self, action: str, args: dict) -> dict | None:
        if action == self.PREFIX:
            action = args.get("action", "info")
        args = {k: v for k, v in args.items() if not k.startswith("_")}
        if action == "start":
            return {"state": "ready", "configured": self._configured()}
        if action == "stop":
            return {"state": "idle"}
        if action == "info":
            return {"state": "ready", "configured": self._configured(), "bridge_url": self._cfg["bridge_url"]}
        if action == "config":
            return self._config(args)
        handlers = {
            "initialize": self._initialize,
            "initialize_status": lambda args: dict(self._deploy_state),
            "printers": self._printers,
            "list": self._list,
            # Keep old saved canvas actions working during rollout; find is no
            # longer advertised and now returns the same safe daily list.
            "find": self._list,
            "find_and_print": self._find_and_print,
            "print": self._print_action,
        }
        handler = handlers.get(action)
        if handler is None:
            return {"status": "error", "code": "bad_action", "error": f"不支持的动作：{action}"}
        if not self._configured() and action not in ("printers", "initialize", "initialize_status"):
            return {"status": "error", "code": "not_configured", "error": "卡片还没配置桥接服务地址，请在卡片配置里填写"}
        try:
            return handler(args)
        except BridgeError as exc:
            print(f"[QL_printer] {action} failed: {exc.code}", flush=True)
            return {"status": "error", "code": exc.code, "error": str(exc)}

    def _configured(self) -> bool:
        return bool(self._cfg["bridge_url"] and self._cfg["token"])

    def _client(self) -> BridgeClient:
        with self._lock:
            cfg = dict(self._cfg)
        return BridgeClient(cfg["bridge_url"], cfg["token"], cfg["timeout_s"])

    def _config(self, args: dict) -> dict:
        update: dict = {}
        if "printer_name" in args:
            update["printer_name"] = str(args["printer_name"] or "").strip()
        if "bridge_url" in args:
            url = str(args["bridge_url"] or "").strip()
            if url:
                parsed = urllib.parse.urlparse(url)
                if parsed.scheme not in ("http", "https") or not parsed.hostname:
                    return {"status": "error", "adapter_ok": False,
                            "error": "桥接服务地址要写成 http://IP:端口 的形式"}
            update["bridge_url"] = url
        if "token" in args:
            update["token"] = str(args["token"] or "")
        if "remote_start_enabled" in args:
            update["remote_start_enabled"] = bool(args["remote_start_enabled"])
        if "remote_start_user" in args:
            update["remote_start_user"] = str(args["remote_start_user"] or "").strip()
        if "remote_start_key_path" in args:
            update["remote_start_key_path"] = str(args["remote_start_key_path"] or "").strip()
        if "remote_start_port" in args and args["remote_start_port"] not in (None, ""):
            try:
                update["remote_start_port"] = min(65535, max(1, int(args["remote_start_port"])))
            except (TypeError, ValueError):
                return {"status": "error", "adapter_ok": False, "error": "remote_start_port 要填整数"}
        for key, low, high in (("timeout_s", 2, 30), ("print_timeout_s", 10, 120)):
            if key in args and args[key] not in (None, ""):
                try:
                    update[key] = min(high, max(low, float(args[key])))
                except (TypeError, ValueError):
                    return {"status": "error", "adapter_ok": False, "error": f"{key} 要填数字"}
        with self._lock:
            self._cfg.update(update)
            configured = bool(self._cfg["bridge_url"] and self._cfg["token"])
        return {"status": "ok", "adapter_ok": configured, "configured": configured}

    # ── 动作 ──────────────────────────────────────────────────────────────
    def _bridge_is_healthy(self) -> bool:
        try:
            payload = BridgeClient(self._cfg["bridge_url"], self._cfg["token"], 2).request(
                "GET", "/health", timeout_s=2, retry_refused=False
            )
            return payload.get("state") == "running"
        except BridgeError:
            return False

    def _ensure_remote_bridge(self) -> dict | None:
        with self._lock:
            cfg = dict(self._cfg)
        if self._bridge_is_healthy() or not cfg["remote_start_enabled"]:
            return None

        parsed = urllib.parse.urlparse(cfg["bridge_url"])
        host = parsed.hostname or ""
        user = cfg["remote_start_user"]
        key_path = cfg["remote_start_key_path"]
        if not host or not user or not key_path or not os.path.isfile(key_path):
            return {
                "status": "error", "code": "remote_start_not_configured",
                "error": "桥接未运行，且机器人远程启动配置或 SSH 私钥不完整",
            }

        known_hosts = "/run/ql-printer-auth/known_hosts"
        command = [
            "ssh", "-F", "/dev/null", "-i", key_path,
            "-p", str(cfg["remote_start_port"]),
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={known_hosts}",
            f"{user}@{host}", "powershell.exe -NoProfile -NonInteractive -EncodedCommand " + base64.b64encode(
                "Start-ScheduledTask -TaskName 'Paradigm Visitor Bridge Watchdog'".encode('utf-16le')).decode(),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=10, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {"status": "error", "code": "remote_start_failed", "error": f"机器人无法远程启动前台桥接：{exc}"}
        if completed.returncode != 0:
            detail = " ".join((completed.stderr or completed.stdout or "SSH 启动失败").split())[:240]
            return {"status": "error", "code": "remote_start_failed", "error": f"机器人远程启动前台桥接失败：{detail}"}

        for _ in range(15):
            time.sleep(1)
            if self._bridge_is_healthy():
                return None
        return {"status": "error", "code": "remote_start_timeout", "error": "已通知前台电脑启动桥接，但 15 秒内未恢复"}

    def _initialize(self, args: dict) -> dict:
        with self._lock:
            if self._deploy_state.get("status") == "installing":
                return dict(self._deploy_state)
            config = dict(self._cfg)
            config.update({k: v for k, v in args.items() if k in config})
            self._deploy_state = {"status": "installing"}
        def work():
            try:
                from .bridge_deploy import deploy
                result = deploy(config)
            except Exception as exc:
                detail = str(exc).replace(str(config.get('token') or '\0'), '[REDACTED]')
                result = {"status": "error", "error": "初始化失败：" + detail[:1600]}
            with self._lock:
                self._deploy_state = result
        threading.Thread(target=work, daemon=True).start()
        return {"status": "installing"}

    def _printers(self, args: dict) -> dict:
        # Use unsaved dialog connection values without changing runtime config.
        client = BridgeClient(str(args.get("bridge_url") or self._cfg["bridge_url"]), str(args.get("token") or self._cfg["token"]), 20)
        return client.request("POST", "/printers", body={})

    def _daily_visits(self) -> tuple[str, list[dict]]:
        recovery = self._ensure_remote_bridge()
        if recovery is not None:
            raise BridgeError(str(recovery.get("error") or "桥接服务不可用"), str(recovery.get("code") or "bridge_error"))
        payload = self._client().request("POST", "/list_visits", body={})
        visits = [_visit(v) for v in payload.get("visits") or [] if isinstance(v, dict)]
        visits = [v for v in visits if v["ticket"]]
        return str(payload.get("date") or ""), visits

    def _list(self, args: dict) -> dict:
        day, visits = self._daily_visits()
        public = [{k: v for k, v in visit.items() if k != "ticket"} for visit in visits]
        if not visits:
            return {"status": "empty", "date": day, "count": 0, "visits": [],
                    "hint": "今天没有未入场访客"}
        return {"status": "listed", "date": day, "count": len(public), "visits": public}

    def _print_ticket(self, args: dict) -> dict:
        ticket = str(args.get("ticket", "")).strip()
        if not ticket or len(ticket) > 512:
            return {"status": "error", "code": "bad_args", "error": "旧版 print 调用缺少短期 ticket"}
        request_id = str(args.get("request_id", "")).strip()
        if not _ID_RE.match(request_id):
            return {"status": "error", "code": "bad_args",
                    "error": "print 需要唯一的 request_id；同一次请求重试时必须复用"}
        with self._lock:
            print_timeout_s = self._cfg["print_timeout_s"]
        payload = self._client().request(
            "POST",
            "/print_tag",
            body={"ticket": ticket, "request_id": request_id, "printer_name": self._cfg["printer_name"]},
            timeout_s=print_timeout_s,
        )
        if payload.get("ok") is True:
            return {"status": "printed", "request_id": request_id,
                    "say": "访客贴打好了，在前台打印机上"}
        return {
            "status": "failed",
            "code": str(payload.get("code") or "print_failed"),
            "error": str(payload.get("error") or "打印失败，请前台同事检查打印机"),
            "request_id": request_id,
        }

    def _find_and_print(self, args: dict) -> dict:
        name = str(args.get("name", "")).strip()
        host = str(args.get("host", "")).strip()
        if not name:
            return {"status": "error", "code": "bad_args", "error": "print 需要访客姓名 name"}
        if len(name) > 32 or len(host) > 64:
            return {"status": "error", "code": "bad_args", "error": "姓名或被访人信息太长"}
        _, visits = self._daily_visits()
        matches = [
            visit for visit in visits
            if visit.get("name") == name and (not host or host in str(visit.get("host", "")))
        ]
        if not matches:
            return {
                "status": "not_found", "name": name, "visits": [],
                "hint": "今天的未入场列表中没有匹配访客，不能打印访客贴",
            }
        if len(matches) != 1:
            return {
                "status": "error",
                "code": "visitor_ambiguous",
                "error": "今天有多条同名未入场预约，不能直接打印；请填写被访人 host 缩小范围",
            }

        visit = matches[0]

        key = (
            name,
            str(visit.get("host", "")),
            str(visit.get("tail", "")),
            str(visit.get("time", "")),
        )
        with self._lock:
            request_id = self._direct_request_ids.get(key)
            if not request_id:
                request_id = f"direct-{uuid.uuid4().hex}"
                if len(self._direct_request_ids) >= 64:
                    self._direct_request_ids.pop(next(iter(self._direct_request_ids)))
                self._direct_request_ids[key] = request_id
        return self._print_ticket({"ticket": visit["ticket"], "request_id": request_id})

    def _print_action(self, args: dict) -> dict:
        # Backward-compatible for an older Skill that still submits ticket/request_id.
        # The public schema only exposes name/host, so canvas users never handle them.
        if args.get("ticket") or args.get("request_id"):
            return self._print_ticket(args)
        return self._find_and_print(args)


# Keep the Python class alias so older local imports do not break during rollout.
HPPrinterPlugin = QLPrinterPlugin




