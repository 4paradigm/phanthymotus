from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from bridge_core import BridgeFailure, Visit


_ALIASES = {
    "id": ("visitId", "visitorId", "visitorInfoId", "id", "code", "number"),
    "name": ("visitorName", "visitName", "name"),
    "phone": ("visitorPhone", "phone", "mobile", "contactWay", "contact"),
    "host": ("bookingPerson", "hostName", "receiverName", "interviewedName", "reserveName", "inviteName", "employeeName"),
    "time": ("visitStartDate", "visitTime", "arriveTime", "appointmentTime", "reserveTime", "startTime"),
    "reason": ("visitReasonName", "reasonName", "visitReason", "reason"),
    "status": ("visitorStatusName", "statusName", "visitStatusName", "visitorStatus", "status"),
}

_STATUS_LABELS = {
    "0": "未入场",
    "1": "已入场",
    "2": "已离场",
}


def _pick(value: dict, aliases: tuple[str, ...]) -> str:
    for key in aliases:
        item = value.get(key)
        if item not in (None, "") and not isinstance(item, (dict, list)):
            return str(item).strip()
    return ""


def normalize_status(value: Any) -> str:
    text = str(value).strip() if value not in (None, "") else ""
    return _STATUS_LABELS.get(text, text)


def extract_visits(payload: Any, expected_name: str, host: str = "") -> list[Visit]:
    """Extract visitor rows from the platform response without retaining raw PII."""
    found: dict[tuple[str, str, str], Visit] = {}

    def walk(node: Any, inherited: dict[str, str]):
        if isinstance(node, list):
            for item in node:
                walk(item, inherited)
            return
        if not isinstance(node, dict):
            return

        context = dict(inherited)
        if isinstance(node.get("visitGuestVOList"), list):
            context["booking_code"] = _pick(node, ("code",))
        for field in ("host", "time", "reason", "status", "id"):
            value = _pick(node, _ALIASES[field])
            if value:
                if field == "status":
                    value = normalize_status(value)
                context[field] = value

        name = _pick(node, _ALIASES["name"])
        if name and (expected_name == "" or name == expected_name):
            visit_host = _pick(node, _ALIASES["host"]) or context.get("host", "")
            if not host or host in visit_host:
                phone = _pick(node, _ALIASES["phone"])
                visit_id = _pick(node, _ALIASES["id"]) or context.get("id", "")
                if visit_id:
                    visit = Visit(
                        visit_id=visit_id,
                        name=name,
                        phone_tail=phone[-4:],
                        host=visit_host,
                        visit_time=_pick(node, _ALIASES["time"]) or context.get("time", ""),
                        reason=_pick(node, _ALIASES["reason"]) or context.get("reason", ""),
                        status=(normalize_status(_pick(node, _ALIASES["status"])) or
                                context.get("status", "")),
                        booking_code=context.get("booking_code", ""),
                    )
                    found[(visit.visit_id, visit.name, visit.phone_tail)] = visit

        for child in node.values():
            if isinstance(child, (dict, list)):
                walk(child, context)

    walk(payload, {})
    return list(found.values())


def schema_only(value: Any) -> Any:
    """Return keys and shapes only; never write response values to diagnostics."""
    if isinstance(value, dict):
        return {str(key): schema_only(item) for key, item in value.items()}
    if isinstance(value, list):
        return [schema_only(value[0])] if value else []
    return type(value).__name__


def replay_auth_headers(headers: dict[str, str]) -> dict[str, str]:
    """Keep only application authentication headers; never replay transport headers."""
    selected: dict[str, str] = {}
    for raw_name, raw_value in headers.items():
        name = str(raw_name).strip().lower()
        if not name or name == "cookie":
            continue
        if any(marker in name for marker in ("authorization", "token", "csrf", "xsrf", "tenant")):
            selected[name] = str(raw_value)
    return selected


def print_readiness_script(wait_ms: Any = 1500) -> str:
    """Delay automatic printing until images, fonts and late layout work are ready."""
    try:
        delay = max(250, min(5000, int(wait_ms)))
    except (TypeError, ValueError):
        delay = 1500
    return f"""
(() => {{
  if (window.__paradigmPrintReadinessInstalled) return;
  window.__paradigmPrintReadinessInstalled = true;
  const nativePrint = window.print.bind(window);
  const nativeClose = window.close.bind(window);
  let preparingPrint = false;
  let closeRequested = false;

  window.close = () => {{
    if (preparingPrint) {{
      closeRequested = true;
      return;
    }}
    nativeClose();
  }};

  window.print = () => {{
    if (preparingPrint) return;
    preparingPrint = true;
    const images = Array.from(document.images || []);
    const imageReady = images.map((image) => new Promise((resolve) => {{
      if (image.complete) return resolve();
      image.addEventListener('load', resolve, {{ once: true }});
      image.addEventListener('error', resolve, {{ once: true }});
    }}));
    const fontReady = document.fonts && document.fonts.ready
      ? document.fonts.ready.catch(() => undefined)
      : Promise.resolve();
    const maximumWait = new Promise((resolve) => setTimeout(resolve, 5000));

    Promise.race([Promise.all([...imageReady, fontReady]), maximumWait])
      .then(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))))
      .then(() => new Promise((resolve) => setTimeout(resolve, {delay})))
      .then(() => {{
        nativePrint();
        preparingPrint = false;
        if (closeRequested) setTimeout(nativeClose, 0);
      }});
  }};
}})();
"""


class EdgeVisitorBackend:
    def __init__(self, config: dict, data_dir: str | Path):
        self.config = dict(config)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.profile_dir = Path(self.config.get("profile_dir", self.data_dir / "edge-profile"))
        if not self.profile_dir.is_absolute():
            self.profile_dir = self.data_dir / self.profile_dir
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.list_url = str(self.config["list_url"])
        self.query_marker = str(self.config.get("query_endpoint_contains", "/visitInfo/page/query"))
        list_parts = urlsplit(self.list_url)
        default_query_url = (
            f"{list_parts.scheme}://{list_parts.netloc}"
            "/smart-admin-api/visitInfo/page/query"
        )
        self.query_url = str(self.config.get("query_url", default_query_url))
        self._lock = threading.RLock()
        self._pw = None
        self._context = None
        self._page = None
        self._jobs: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, name="edge-visitor-worker", daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                break
            method_name, args, done, result = job
            try:
                result["value"] = getattr(self, method_name)(*args)
            except BaseException as exc:
                result["error"] = exc
            finally:
                done.set()

    def _submit(self, method_name: str, *args):
        if threading.current_thread() is self._worker:
            return getattr(self, method_name)(*args)
        done = threading.Event()
        result: dict[str, Any] = {}
        self._jobs.put((method_name, args, done, result))
        if not done.wait(timeout=90):
            raise BridgeFailure(
                "browser_timeout",
                "Edge 操作超过 90 秒；系统不会自动重试打印",
                uncertain=method_name == "_print_visit_impl",
            )
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def list_printers(self):
        result = self._run_printer_powershell("ConvertTo-Json -Compress -InputObject @(Get-CimInstance Win32_Printer | Select-Object Name,Default,WorkOffline)")
        if result.returncode:
            raise BridgeFailure("printer_check_failed", self._printer_command_error(result))
        return {"printers": [{"name": p["Name"], "default": bool(p.get("Default")), "offline": bool(p.get("WorkOffline"))} for p in json.loads(result.stdout)]}

    def _prepare_default_printer(self, printer_name=None) -> None:
        expected = str(printer_name or self.config.get("printer_name", "")).strip()
        if not expected or os.name != "nt":
            return
        quoted = expected.replace("'", "''")
        auto_set = bool(self.config.get("auto_set_default_printer", True))

        # Keep each PowerShell invocation deliberately short.  Windows
        # PowerShell 5.1 can otherwise report the whole multiline command in a
        # CalledProcessError, hiding the useful failure and occasionally
        # tripping over the host's legacy output encoding.
        query = self._run_printer_powershell(
            f"""
$expected = '{quoted}'
$installed = Get-CimInstance Win32_Printer | Where-Object Name -eq $expected | Select-Object -First 1
if (-not $installed) {{ exit 3 }}
Get-CimInstance Win32_Printer | Where-Object Default | Select-Object -First 1 -ExpandProperty Name
"""
        )
        if query.returncode == 3:
            raise BridgeFailure("printer_not_installed", f"Windows 中未安装打印机“{expected}”")
        if query.returncode != 0:
            raise BridgeFailure("printer_check_failed", self._printer_command_error(query))
        actual = query.stdout.strip()

        if actual != expected and auto_set:
            switch = self._set_default_printer(expected)
            if switch.returncode != 0:
                raise BridgeFailure("printer_switch_failed", self._printer_command_error(switch))
            time.sleep(0.75)
            verify = self._run_printer_powershell(
                "Get-CimInstance Win32_Printer | Where-Object Default | "
                "Select-Object -First 1 -ExpandProperty Name"
            )
            if verify.returncode != 0:
                raise BridgeFailure("printer_check_failed", self._printer_command_error(verify))
            actual = verify.stdout.strip()

        if actual != expected:
            raise BridgeFailure(
                "printer_not_default",
                f"无法把 Windows 默认打印机切换为“{expected}”（当前为“{actual or '未设置'}”）",
            )

    @staticmethod
    def _set_default_printer(printer_name: str) -> subprocess.CompletedProcess:
        """Use the native PrintUI entry point; WScript.Network can hang headlessly."""
        try:
            return subprocess.run(
                [
                    "rundll32.exe",
                    "printui.dll,PrintUIEntry",
                    "/y",
                    "/n",
                    printer_name,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise BridgeFailure("printer_switch_failed", f"无法切换 Windows 默认打印机：{exc}") from exc

    @staticmethod
    def _run_printer_powershell(command: str) -> subprocess.CompletedProcess:
        preamble = (
            "$ErrorActionPreference = 'Stop'; "
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
        )
        try:
            return subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", preamble + command],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise BridgeFailure("printer_check_failed", f"无法运行 Windows 打印机检查：{exc}") from exc

    @staticmethod
    def _printer_command_error(result: subprocess.CompletedProcess) -> str:
        detail = (result.stderr or result.stdout or "未知错误").strip()
        # Do not return a huge PowerShell command/trace through MCP.
        detail = " ".join(detail.split())[:300]
        return f"Windows 打印机操作失败：{detail}"

    def _ensure_browser(self):
        if self._context is not None:
            try:
                if self._page is not None and not self._page.is_closed():
                    return self._page
                for candidate in self._context.pages:
                    if not candidate.is_closed():
                        self._page = candidate
                        return candidate
                self._page = self._context.new_page()
                return self._page
            except Exception:
                # Edge may have been closed manually while the bridge stayed
                # alive. Discard the stale Playwright objects and relaunch.
                self._discard_browser()
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise BridgeFailure("playwright_missing", "桥接程序缺少 Edge 自动化组件，请重新运行安装程序") from exc
        self._pw = sync_playwright().start()
        args = ["--no-first-run", "--disable-session-crashed-bubble"]
        if self.config.get("kiosk_printing", True):
            args.append("--kiosk-printing")
        try:
            self._context = self._pw.chromium.launch_persistent_context(
                str(self.profile_dir),
                channel="msedge",
                headless=bool(self.config.get("headless", False)),
                args=args,
                accept_downloads=False,
            )
            self._context.add_init_script(
                script=print_readiness_script(self.config.get("print_render_wait_ms", 1500))
            )
        except Exception as exc:
            raise BridgeFailure("edge_start_failed", f"无法启动专用 Edge：{exc}") from exc
        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        return self._page

    def _ensure_login(self, page):
        # Use the operator-selected tab URL.  Navigating to a vendor-specific
        # root here discarded the selection saved by the card and also broke
        # deployments whose visitor portal uses a different host/path.
        if page.url != self.list_url:
            page.goto(self.list_url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_timeout(1200)
        if "login" in page.url.lower():
            raise BridgeFailure("login_required", "访客平台尚未登录，请在前台 Edge 登录后重新选择网页")
        # 监听网页自身的查询请求，从中取得应用层认证头。只保存头名称和值在内存中，绝不写日志。
        captured_request = None
        try:
            with page.expect_request(
                lambda request: self.query_marker in request.url and request.method == "POST",
                timeout=20_000,
            ) as request_info:
                page.reload(wait_until="domcontentloaded", timeout=30_000)
            captured_request = request_info.value
        except Exception:
            # 部分页面进入列表后不会自动刷新；尝试点击一次查询按钮以触发真实请求。
            button = page.get_by_role("button", name=re.compile("查询|搜索")).first
            if button.count() and button.is_visible():
                try:
                    with page.expect_request(
                        lambda request: self.query_marker in request.url and request.method == "POST",
                        timeout=15_000,
                    ) as request_info:
                        button.click()
                    captured_request = request_info.value
                except Exception:
                    captured_request = None
        page.wait_for_timeout(1200)
        if "login" in page.url.lower():
            raise BridgeFailure("login_required", "访客平台登录状态已失效，请重新登录")
        if captured_request is None:
            raise BridgeFailure(
                "auth_capture_failed",
                "没有捕获到网页自身的访客查询请求，请确认列表页能正常加载并显示数据",
            )
        return replay_auth_headers(captured_request.all_headers())

    @staticmethod
    def _field(page, labels: tuple[str, ...]):
        for label in labels:
            for css in (".el-form-item", ".ant-form-item", "label"):
                item = page.locator(css).filter(has_text=re.compile(re.escape(label))).first
                if item.count():
                    box = item.locator("input").first
                    if box.count() and box.is_visible():
                        return box
            box = page.locator(f'input[placeholder*="{label}"]').first
            if box.count() and box.is_visible():
                return box
        return None

    @staticmethod
    def _target_was_closed(exc: BaseException) -> bool:
        return (
            exc.__class__.__name__ == "TargetClosedError"
            or "Target page, context or browser has been closed" in str(exc)
        )

    def _search(self, name: str, host: str = "", start_time=None, end_time=None):
        for attempt in range(2):
            try:
                return self._search_once(name, host, start_time, end_time)
            except Exception as exc:
                if attempt == 0 and self._target_was_closed(exc):
                    self._discard_browser()
                    continue
                raise
        raise BridgeFailure("edge_start_failed", "无法恢复已关闭的专用 Edge")

    def _search_once(self, name: str, host: str = "", start_time=None, end_time=None):
        page = self._ensure_browser()
        auth_headers = self._ensure_login(page)

        req_body = {
            "bookingPerson": host if host else "",
            "contacts": "",
            "orders": [{"asc": False, "column": "visit_start_date"}],
            "pageNum": 1,
            "pageSize": 100,
            "reasonId": None,
            "status": None,
            "visitTimeBegin": start_time if (start_time and start_time.strip()) else None,
            "visitTimeEnd": end_time if (end_time and end_time.strip()) else None
        }
        try:
            js_code = """
            async ([url, postBody, authHeaders]) => {
                const resp = await fetch(url, {
                    method: 'POST',
                    credentials: 'include',
                    headers: {
                        'Accept': 'application/json',
                        'Content-Type': 'application/json',
                        ...authHeaders
                    },
                    body: JSON.stringify(postBody)
                });
                const text = await resp.text();
                return {
                    ok: resp.ok,
                    status: resp.status,
                    contentType: resp.headers.get('content-type') || '',
                    responseUrl: resp.url,
                    text
                };
            }
            """
            result = page.evaluate(js_code, [self.query_url, req_body, auth_headers])
        except Exception as exc:
            raise BridgeFailure("query_failed", f"浏览器内查询接口调用失败：{exc}") from exc

        status = int(result.get("status", 0))
        response_url = str(result.get("responseUrl", ""))
        text = str(result.get("text", ""))
        if status in (401, 403) or "login" in response_url.lower():
            raise BridgeFailure("login_required", "访客平台登录状态已失效，请重新登录")
        if not result.get("ok"):
            raise BridgeFailure("query_http_error", f"访客查询接口返回 HTTP {status}")
        try:
            payload = json.loads(text.lstrip("\ufeff"))
        except json.JSONDecodeError as exc:
            content_type = str(result.get("contentType", ""))
            raise BridgeFailure(
                "query_not_json",
                f"访客查询接口未返回有效 JSON（HTTP {status}，Content-Type: {content_type or '未知'}）",
            ) from exc

        if isinstance(payload, dict):
            business_code = payload.get("code")
            message = str(payload.get("message") or payload.get("msg") or "")
            code_text = str(business_code).strip().lower() if business_code is not None else ""
            if code_text in {"401", "403", "unauthorized", "not_login", "notlogin"} or any(
                marker in message for marker in ("未登录", "登录失效", "重新登录")
            ):
                raise BridgeFailure("login_required", "访客平台登录状态已失效，请重新登录")
        return page, payload



    def find_visits(self, name: str, host: str = "", start_time=None, end_time=None) -> list[Visit]:
        return self._submit("_find_visits_impl", name, host, start_time, end_time)

    def _find_visits_impl(self, name: str, host: str = "", start_time=None, end_time=None) -> list[Visit]:
        with self._lock:
            _, payload = self._search(name, host, start_time, end_time)
            visits = extract_visits(payload, name, host)
            if not visits:
                diagnostic = self.data_dir / "last-query-schema.json"
                diagnostic.write_text(json.dumps(schema_only(payload), ensure_ascii=False, indent=2), encoding="utf-8")
            return visits

    def _row_for(self, page, visit: Visit):
        rows = page.locator("tr")
        matches = []
        for index in range(rows.count()):
            row = rows.nth(index)
            try:
                text = row.inner_text(timeout=1000)
            except Exception:
                continue
            if not row.get_by_text(visit.name, exact=True).count():
                continue
            if visit.phone_tail and visit.phone_tail not in text:
                continue
            matches.append(row)
        if len(matches) > 1:
            raise BridgeFailure("visitor_ambiguous", "预约详情中有多位同名且手机号后四位相同的访客，拒绝自动打印")
        return matches[0] if matches else None

    def _open_detail_if_needed(self, page, visit: Visit):
        if not visit.booking_code:
            raise BridgeFailure("booking_code_missing", "查询结果缺少预约编号，无法安全定位访客详情")
        candidates = page.locator("tr")
        matching = []
        for index in range(candidates.count()):
            candidate = candidates.nth(index)
            try:
                text = candidate.inner_text(timeout=1000)
            except Exception:
                continue
            if visit.booking_code in text and candidate.get_by_text(visit.booking_code, exact=True).count():
                matching.append(candidate)
        if len(matching) != 1:
            raise BridgeFailure("booking_row_missing", "页面上无法唯一定位预约编号，请前台核对列表")
        detail = matching[0].get_by_text("查看", exact=True)
        if detail.count() != 1:
            raise BridgeFailure("detail_action_missing", "对应预约行没有唯一的“查看”操作")
        detail.click(timeout=5000)
        for _ in range(10):
            row = self._row_for(page, visit)
            if row is not None:
                return row
            page.wait_for_timeout(500)
        raise BridgeFailure("visitor_row_missing", "已打开预约详情，但无法同时核对访客姓名和手机号后四位")

    def inspect_visit_detail(self, visit: Visit) -> dict:
        return self._submit("_inspect_visit_detail_impl", visit)

    def _inspect_visit_detail_impl(self, visit: Visit) -> dict:
        with self._lock:
            page, _ = self._search(visit.name, visit.host)
            row = self._open_detail_if_needed(page, visit)
            text = row.inner_text(timeout=2000)
            visible_status = next(
                (status for status in ("未入场", "已入场", "已离场") if status in text), ""
            )
            return {
                "booking_matched": True,
                "visitor_matched": True,
                "visible_status": visible_status,
                "print_action_available": row.get_by_text("打印访客贴", exact=True).count() == 1,
            }

    def get_status(self, visit: Visit) -> str:
        return self._submit("_get_status_impl", visit)

    def _get_status_impl(self, visit: Visit) -> str:
        with self._lock:
            page, payload = self._search(visit.name, visit.host)
            matches = extract_visits(payload, visit.name, visit.host)
            for item in matches:
                if item.visit_id == visit.visit_id or (item.phone_tail and item.phone_tail == visit.phone_tail):
                    if item.status in ("未入场", "已入场", "已离场"):
                        return item.status
            row = self._open_detail_if_needed(page, visit)
            text = row.inner_text(timeout=2000)
            for status in ("未入场", "已入场", "已离场"):
                if status in text:
                    return status
            return ""

    def print_visit(self, visit: Visit, printer_name=None) -> None:
        return self._submit("_print_visit_impl", visit, printer_name)

    def _print_visit_impl(self, visit: Visit, printer_name=None) -> None:
        with self._lock:
            self._prepare_default_printer(printer_name)
            page, _ = self._search(visit.name, visit.host)
            row = self._open_detail_if_needed(page, visit)
            text = row.inner_text(timeout=1000)
            if "未入场" not in text:
                raise BridgeFailure("status_not_printable", "访客当前不是未入场状态，拒绝点击打印")
            action = row.get_by_text("打印访客贴", exact=True)
            if not action.count():
                raise BridgeFailure("print_action_missing", "页面上没有找到“打印访客贴”操作")
            try:
                action.click(timeout=5000)
                page.wait_for_timeout(3000)
            except Exception as exc:
                raise BridgeFailure(
                    "print_outcome_unknown",
                    "点击打印后没有得到明确结果；系统不会自动重试，请前台核对打印机和访客状态",
                    uncertain=True,
                ) from exc

    def close(self):
        self._submit("_close_impl")
        self._jobs.put(None)
        self._worker.join(timeout=10)

    def _close_impl(self):
        self._discard_browser()

    def _discard_browser(self):
        context = self._context
        playwright = self._pw
        self._page = None
        self._context = None
        self._pw = None
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
        if playwright is not None:
            try:
                playwright.stop()
            except Exception:
                pass
