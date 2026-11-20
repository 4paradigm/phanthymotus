"""通用访客贴打印卡片（QL_printer），独立于机器人硬件驱动。

机器人与前台电脑之间只使用经过主机校验的 SSH。Windows 上的浏览器自动化运行时
只监听 127.0.0.1；卡片通过 SSH 命令让 Windows 本机调用它，外部不开放业务端口，
也不需要在卡片中保存访问令牌。

    机器人 skill → QL_printer 卡片 ──SSH──> Windows 自动化运行时 → 范式+ / 打印机

公共服务部署和迁移说明见 docs/ql-printer.md。
"""

from __future__ import annotations

import json
import base64
import ipaddress
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


TOOL_NAME = "Visitor_Badge_Printer"

_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def _normalize_bridge_url(value: object) -> str:
    """Normalize an operator-facing computer IP to an internal SSH target."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        if any(c.isspace() for c in raw):
            raise ValueError()
        explicit_scheme = "://" in raw
        if not explicit_scheme:
            try:
                if ipaddress.ip_address(raw).version == 6:
                    raw = f"[{raw}]"
            except ValueError:
                pass
        parsed = urllib.parse.urlsplit(raw if explicit_scheme else f"ssh://{raw}")
        host = parsed.hostname or ""
        port = parsed.port  # Validates numeric range as well.
        if parsed.scheme not in ("ssh", "http", "https") or not host or parsed.username is not None or parsed.password is not None:
            raise ValueError()
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment or not re.fullmatch(r"[A-Za-z0-9_.:%-]+", host):
            raise ValueError()
        if port == 0 or (parsed.scheme == "ssh" and port not in (None, 22)):
            raise ValueError()
        netloc = f"[{host}]" if ":" in host else host
        # Accept old http://IP:8765 values during migration but never use the
        # HTTP scheme or port for communication.
        return f"ssh://{netloc}"
    except ValueError:
        raise ValueError("前台电脑地址应填写 IP，不能包含账号、路径或无效端口") from None


class BridgeError(Exception):
    """前台 SSH 自动化失败；message 可以直接转述给操作人员。"""

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
                timed_out = (
                    isinstance(reason, TimeoutError)
                    or getattr(reason, "errno", None) in (110, 10060)
                    or "timed out" in str(reason).lower()
                )
                if timed_out:
                    message = (
                        "前台桥接请求超时，尚未收到结果；不能据此判断桥接未启动。"
                        "请检查前台专用 Edge 是否已登录访客网站、页面是否加载完成，以及网络连接。"
                    )
                    if path == "/print_tag":
                        message += "打印结果尚不确定，请先核对打印机和访客状态，不要立即重复打印。"
                    raise BridgeError(message, "request_timeout") from exc
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


class SSHBridgeClient:
    """通过唯一的 SSH 通道调用 Windows 本机回环接口。"""

    MARKER = "VISITOR_RESULT:"

    def __init__(self, host: str = "", user: str = "", key_path: str = "/run/ql-printer-auth/frontdesk_bridge_ed25519",
                 port: int = 22, timeout_s: float = 60):
        self.host = host
        self.user = user
        self.key_path = key_path
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.known_hosts = "/run/ql-printer-auth/known_hosts"

    def _connection(self) -> tuple[str, str]:
        if self.host and self.user:
            return "/dev/null", f"{self.user}@{self.host}"
        config = "/run/ql-printer-auth/config"
        if os.path.isfile(config):
            return config, "visitor-frontdesk"
        try:
            lines = open(self.known_hosts, encoding="utf-8").read().splitlines()
        except OSError as exc:
            raise BridgeError("机器人缺少 known_hosts，无法识别前台电脑", "ssh_not_configured") from exc
        hosts: set[str] = set()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            field = line.split(None, 1)[0]
            for host in field.split(","):
                if host.startswith("|"):
                    continue
                if host.startswith("[") and "]:" in host:
                    host = host[1:host.index("]:")]
                hosts.add(host)
        if len(hosts) != 1:
            raise BridgeError(
                "无法从 known_hosts 唯一识别前台电脑；请在密钥目录提供 SSH config（Host visitor-frontdesk）",
                "ssh_target_ambiguous",
            )
        # Standard deployments use this Windows account. Non-standard users
        # are declared once in the optional SSH config, never on every card.
        return "/dev/null", f"4paradigm@{next(iter(hosts))}"

    def _ssh(self, script: str, timeout_s: float | None = None) -> subprocess.CompletedProcess:
        if not os.path.isfile(self.key_path) or not os.path.isfile(self.known_hosts):
            raise BridgeError("机器人缺少 SSH 私钥或 known_hosts 挂载", "ssh_not_configured")
        ssh_config, target = self._connection()
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        command = [
            "ssh", "-F", ssh_config, "-T", "-i", self.key_path,
            "-p", str(self.port), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={self.known_hosts}", target,
            "powershell.exe -NoProfile -NonInteractive -EncodedCommand " + encoded,
        ]
        try:
            return subprocess.run(
                command, capture_output=True, timeout=timeout_s or self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise BridgeError(
                "SSH 请求超时；若刚执行打印，请先核对打印机和访客状态，不要立即重复打印",
                "request_timeout",
            ) from exc
        except OSError as exc:
            raise BridgeError(f"无法运行 SSH：{exc}", "ssh_failed") from exc

    def request(self, method: str, path: str, *, body: dict | None = None,
                timeout_s: float | None = None) -> dict:
        if not re.fullmatch(r"/[A-Za-z0-9_/-]+", path):
            raise BridgeError("内部请求路径无效", "bad_request")
        action = {
            "/health": "health", "/printers": "printers",
            "/list_visits": "list", "/print_tag": "print",
        }.get(path)
        if not action:
            raise BridgeError("内部请求路径无效", "bad_request")
        job_id = uuid.uuid4().hex
        payload = base64.b64encode(json.dumps(
            {"id": job_id, "action": action, "body": body or {}}, ensure_ascii=False
        ).encode("utf-8")).decode("ascii")
        script = rf"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$task = Get-ScheduledTask -TaskName 'Paradigm Visitor Automation' -ErrorAction Stop
$install = $task.Actions | Select-Object -First 1 -ExpandProperty WorkingDirectory
$jobs = Join-Path $install 'data\jobs'
New-Item -ItemType Directory -Force -Path $jobs | Out-Null
$request = Join-Path $jobs '{job_id}.request.json'
$temporary = $request + '.tmp'
$result = Join-Path $jobs '{job_id}.result.json'
[IO.File]::WriteAllBytes($temporary, [Convert]::FromBase64String('{payload}'))
Move-Item -LiteralPath $temporary -Destination $request -Force
Start-ScheduledTask -TaskName 'Paradigm Visitor Automation'
for ($i = 0; $i -lt 220 -and !(Test-Path -LiteralPath $result); $i++) {{ Start-Sleep -Milliseconds 500 }}
if (!(Test-Path -LiteralPath $result)) {{ throw 'Visitor automation job timed out' }}
$bytes = [IO.File]::ReadAllBytes($result)
Remove-Item -LiteralPath $result -Force -ErrorAction SilentlyContinue
Write-Output ('{self.MARKER}' + [Convert]::ToBase64String($bytes))
"""
        result = self._ssh(script, timeout_s)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).decode("utf-8", errors="replace")
            detail = " ".join(detail.split())[-400:]
            raise BridgeError(
                "SSH 无法执行前台自动化" + (f"：{detail}" if detail else ""),
                "ssh_failed",
            )
        markers = [
            line[len(self.MARKER):]
            for line in result.stdout.decode("utf-8", errors="replace").splitlines()
            if line.startswith(self.MARKER)
        ]
        if len(markers) != 1:
            raise BridgeError("SSH 返回内容无法识别", "bad_response")
        try:
            decoded = base64.b64decode(markers[0], validate=True)
            value = json.loads(decoded.decode("utf-8-sig"))
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise BridgeError("Windows 自动化返回的不是有效 JSON", "bad_response") from exc
        if not isinstance(value, dict):
            raise BridgeError("Windows 自动化返回格式错误", "bad_response")
        if value.get("ok") is False and value.get("error"):
            raise BridgeError(str(value["error"]), str(value.get("code") or "remote_error"))
        return value


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
    ALIASES = ("QL_printer",)

    def __init__(self, plugin_config: dict | None = None, executor=None):
        plugin_config = plugin_config or {}
        self._lock = threading.Lock()
        self._cfg = {
            "printer_name": str(plugin_config.get("printer_name", "") or ""),
            "bridge_url": _normalize_bridge_url(plugin_config.get("bridge_url", "")),
            "timeout_s": float(plugin_config.get("timeout_s", 8)),
            "print_timeout_s": float(plugin_config.get("print_timeout_s", 120)),
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
            "config": (["bridge_url", "remote_start_user", "printer_name"], "保存前台 SSH 配置"),
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
            "timeout_s": {"type": "number"},
            "print_timeout_s": {"type": "number"},
            "remote_start_user": {"type": "string"},
            "remote_start_port": {"type": "integer"},
            "remote_start_key_path": {"type": "string"},
            "deployment_id": {"type": "string"},
        }
        actions["config"][0].append("printer_name")
        actions["printers"] = (["bridge_url", "remote_start_user"], "通过 SSH 读取目标电脑已安装的打印机")
        actions["initialize"] = (["bridge_url", "remote_start_user"], "通过 SSH 准备目标电脑，仅在安装设置时使用")
        actions["initialize_status"] = (["deployment_id"], "读取本次初始化进度")
        definition = tool(
            TOOL_NAME,
            "actuator",
            "查询当天未入场访客并打印访客贴。",
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
                # shared SSH target above this field, so an operator
                # can finish the whole setup from the card in one place.
                "printer_name": {"type": "string", "title": "目标打印机", "scope": "instance", "format": "bridge-printer"},
                "bridge_url": {
                    "type": "string", "title": "前台电脑 IP", "scope": "shared",
                    "description": "例如 10.100.10.14",
                },
                "remote_start_user": {
                    "type": "string", "title": "Windows SSH 用户名", "scope": "shared",
                    "default": "4paradigm",
                },
            },
            "required": ["bridge_url", "remote_start_user"],
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
        if action == self.PREFIX or action in self.ALIASES:
            action = args.get("action", "info")
        args = {k: v for k, v in args.items() if not k.startswith("_")}
        if action == "start":
            return {"state": "ready", "configured": self._configured()}
        if action == "stop":
            return {"state": "idle"}
        if action == "info":
            return {"state": "ready", "configured": self._configured(), "frontdesk": self._cfg["bridge_url"]}
        if action == "config":
            return self._config(args)
        handlers = {
            "initialize": self._initialize,
            "initialize_status": self._initialize_status,
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
            return {"status": "error", "code": "not_configured", "error": "请先配置前台电脑 IP 和 Windows SSH 用户名"}
        try:
            return handler(args)
        except ValueError as exc:
            return {"status": "error", "code": "bad_args", "error": str(exc)}
        except BridgeError as exc:
            print(f"[QL_printer] {action} failed: {exc.code}", flush=True)
            return {"status": "error", "code": exc.code, "error": str(exc)}

    def _configured(self) -> bool:
        return bool(self._cfg["bridge_url"] and self._cfg["remote_start_user"])

    def _client(self, cfg: dict | None = None) -> SSHBridgeClient:
        with self._lock:
            effective = dict(self._cfg)
        if cfg:
            effective.update(cfg)
        parsed = urllib.parse.urlparse(effective["bridge_url"])
        return SSHBridgeClient(
            parsed.hostname or "", effective["remote_start_user"],
            effective["remote_start_key_path"], effective["remote_start_port"],
            effective["print_timeout_s"],
        )

    def _config(self, args: dict) -> dict:
        update: dict = {}
        if "printer_name" in args:
            update["printer_name"] = str(args["printer_name"] or "").strip()
        if "bridge_url" in args:
            try:
                url = _normalize_bridge_url(args["bridge_url"])
            except ValueError as exc:
                return {"status": "error", "adapter_ok": False, "error": str(exc)}
            update["bridge_url"] = url
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
            configured = self._configured()
        return {"status": "ok", "adapter_ok": configured, "configured": configured}

    # ── 动作 ──────────────────────────────────────────────────────────────
    def _initialize(self, args: dict) -> dict:
        with self._lock:
            if self._deploy_state.get("status") == "installing":
                return {"status": "error", "code": "installing", "error": "已有初始化任务正在运行，请等待完成后重试"}
            config = dict(self._cfg)
            config.update({k: v for k, v in args.items() if k in config})
            config["bridge_url"] = _normalize_bridge_url(config.get("bridge_url"))
            if not config["bridge_url"]:
                raise ValueError("请填写前台电脑 IP")
            deployment_id = uuid.uuid4().hex
            self._deploy_state = {"status": "installing", "deployment_id": deployment_id}
        def work():
            try:
                from .bridge_deploy import deploy
                result = deploy(config)
            except Exception as exc:
                detail = str(exc).replace(str(config.get('token') or '\0'), '[REDACTED]')
                result = {"status": "error", "error": "初始化失败：" + detail[:1600]}
            with self._lock:
                self._deploy_state = {**result, "deployment_id": deployment_id}
        threading.Thread(target=work, daemon=True).start()
        return {"status": "installing", "deployment_id": deployment_id}

    def _initialize_status(self, args: dict) -> dict:
        with self._lock:
            state = dict(self._deploy_state)
        requested = args.get("deployment_id")
        if requested and requested != state.get("deployment_id"):
            return {"status": "error", "code": "unknown_deployment", "error": "初始化任务已变化，请重新打开配置"}
        if not requested:
            # Legacy status polling can show progress, but cannot discover the
            # credential of another configuration dialog's installation.
            state.pop("token", None)
            state.pop("deployment_id", None)
        return state

    def _printers(self, args: dict) -> dict:
        # Use unsaved dialog SSH values without changing runtime config.
        with self._lock:
            cfg = dict(self._cfg)
        url = _normalize_bridge_url(args.get("bridge_url", cfg["bridge_url"]))
        user = str(args.get("remote_start_user", cfg["remote_start_user"]) or "").strip()
        if not url or not user:
            return {"status": "error", "code": "not_configured", "error": "请填写前台电脑 IP 和 Windows SSH 用户名"}
        client = self._client({"bridge_url": url, "remote_start_user": user})
        return client.request("POST", "/printers", body={})

    def _daily_visits(self) -> tuple[str, list[dict]]:
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




