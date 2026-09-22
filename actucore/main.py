#!/usr/bin/env python3
"""
actucore/main.py — ActuCore bundle 统一入口。

ActuCore 是 Perception 在执行侧的对称层：Perception 把原始数据流变成语义，
ActuCore 把意图/目标变成运动指令。执行模型（VLA、导航、抓取策略、locomotion、
whole-body control）以卡片（插件）的形式挂在这里，聚合成一个 MCP HTTP server
对外暴露，由 Agent Core 通过 MCP JSON-RPC 调用。

当前提供 VLA 卡片及可选遥操卡片；遥操仅在站点配置检查通过后注册。
新增卡片的完整步骤见 README.md。

MCP 工具命名规则：{plugin_prefix}_{tool_name}
  例：vla_info, vla_start, nav_goto

MCP server 端口: config.mcp_port（默认 15730）
"""

from __future__ import annotations

# First, before anything can write to stdout: make every log line one atomic,
# control-character-free write, so concurrent writers cannot tear a Docker log
# record. Without this, actucore produced a log the daemon could not read back at
# all — `docker logs` failed outright with "log message is too large
# (1952739189 > 1000000)", i.e. the framing had come apart and a length prefix
# was being read out of garbage. Same module perception installs; the Dockerfile
# copies it out of perception/utils/ rather than keeping a fourth duplicate.
import logsafe
logsafe.install()

import json
import logging
import os
import signal
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
from pathlib import Path

import yaml

import rclpy
import rclpy.executors

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)
# suppress noisy third-party loggers
for _quiet in ('urllib3', 'httpcore', 'httpx'):
    logging.getLogger(_quiet).setLevel(logging.WARNING)

# Cap on how much of an MCP argument/result dict reaches the log. Card configs
# carry whole maps and voxel grids, so an unbounded repr here is how a single log
# line grows past the point where Docker can frame it — the result side was
# already capped, the argument side was not.
_LOG_ARG_CHARS = 500

# How often the register thread says it is still alive when nothing has changed.
# The heartbeat itself stays at 30s; this only governs how often that fact
# reaches the log, so "quiet" cannot mean both "healthy" and "thread died".
REGISTER_ALIVE_INTERVAL_S = 1800.0


def _brief(obj) -> str:
    """One-line, length-capped repr for logging an MCP payload."""
    text = repr(obj)
    if len(text) <= _LOG_ARG_CHARS:
        return text
    return f"{text[:_LOG_ARG_CHARS]}…[+{len(text) - _LOG_ARG_CHARS} chars]"


# ── ACP: SSE event bus (thread-safe) ─────────────────────────────────────────

import queue as _queue

_sse_clients: list[_queue.Queue] = []   # 每个 SSE 连接一个 queue
_sse_lock = threading.Lock()


def sse_push(event: dict):
    """线程安全地广播 SSE 事件到所有连接的客户端。

    长时执行动作（导航到点、抓取一次）用它把 ACP 完成事件推给订阅方。
    """
    data = json.dumps(event, ensure_ascii=False)
    with _sse_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait(data)
            except _queue.Full:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = os.environ.get("CONFIG_PATH", str(Path(__file__).parent / "config.yaml"))
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


# ── Bundle ────────────────────────────────────────────────────────────────────

class ActuCoreBundle:
    def __init__(self, cfg: dict, executor):
        self.server_name = cfg.get("name", "actucore-bundle")
        self._plugins: list = []
        self.required_site_config: dict = {}
        plugins_cfg = cfg.get("plugins") or {}

        # ── 卡片注册区 ────────────────────────────────────────────────────
        # 卡片是显式注册的（不扫目录），每个卡片一个 if 块。加一个新卡片：
        #
        #   1. 写 plugins/<name>.py（或 plugins/<name>/ 包，__init__.py 里 re-export）
        #   2. 在 config.yaml 的 plugins 下加 `<name>: {enabled: true, ...}`
        #   3. 在这里加：
        #
        #        if plugins_cfg.get("<name>", {}).get("enabled", False):
        #            from plugins.<name> import XPlugin
        #            self._plugins.append(XPlugin(plugins_cfg["<name>"], executor))
        #            log.info("XPlugin loaded")
        #
        # 需要 ROS 命名空间的卡片（topic 里要带机器人名）多一步，参照
        # perception/main.py 里 vop 的写法：namespace 为空时用
        # re.sub(r"[^a-zA-Z0-9_]", "_", socket.gethostname()) 兜底。
        #
        # 卡片契约（PREFIX 不能含下划线、action.enum 必须含 "info" 等）见 README.md。
        # ──────────────────────────────────────────────────────────────────

        if plugins_cfg.get("vla", {}).get("enabled", False):
            from plugins.vla import VLAPlugin
            self._plugins.append(VLAPlugin(plugins_cfg["vla"], executor))
            log.info("VLAPlugin loaded")

        if plugins_cfg.get("teleop", {}).get("enabled", False):
            from plugins.teleop.site import required_site_config
            try:
                from plugins.teleop import TeleopPlugin
                plugin = TeleopPlugin(plugins_cfg["teleop"], executor)
                # Validate the accepted restored configuration, not an obsolete
                # calibration path from the original deployment file.
                issues = required_site_config(plugin.cfg)
                if issues:
                    self.required_site_config['teleop'] = issues
                    log.error("teleop not advertised: required_site_config=%s", issues)
                else:
                    self._plugins.append(plugin)
                    log.info("TeleopPlugin loaded")
            except (ImportError, OSError, ValueError, RuntimeError):
                self.required_site_config['teleop'] = [{'field': 'runtime', 'code': 'teleop_initialization_unavailable'}]
                log.error("teleop unavailable; other cards retained")

        if not self._plugins:
            log.info("no cards enabled — ActuCore is running as an empty MCP host")

    def get_all_tools(self) -> list:
        """Every card's schema. One broken card must not take the rest with it.

        `tools/list` is not a call against one card — it is how agent-core learns
        that any card exists at all, and what its ports are. So an exception
        raised while building one card's schema used to empty the whole bundle:
        agent-core got an RPC error, kept the cards it already knew, and showed
        them with no input and no output ports. That reads as a canvas problem,
        and the traceback is in *this* process's log, which is not where anyone
        looks first. Seen on Tianyi.

        Nothing here can repair the broken card, and pretending otherwise would
        be worse — so it is dropped, loudly, and the others are served.
        """
        tools = []
        for p in self._plugins:
            try:
                built = list(p.get_tools())
            except Exception:      # noqa: BLE001 — one card's schema, not the bundle's
                log.exception("card %s failed to build its schema; serving the rest "
                              "without it", getattr(p, "PREFIX", p))
                continue
            for t in built:
                full_name = t['name'] if t['name'] == p.PREFIX else f"{p.PREFIX}_{t['name']}"
                tools.append({**t, "name": full_name})
        return tools

    def dispatch(self, full_name: str, args: dict) -> dict | None:
        prefix, sep, tool_name = full_name.partition("_")
        name = tool_name if sep else prefix
        if prefix in self.required_site_config:
            return {'state': 'unavailable', 'error': 'required_site_config',
                    'required_site_config': self.required_site_config[prefix],
                    'output_active': False}
        for p in self._plugins:
            if p.PREFIX == prefix:
                return p.dispatch(name, args)
        return None


# ── MCP HTTP server ───────────────────────────────────────────────────────────

_bundle: ActuCoreBundle | None = None


def make_handler():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if args and "/sse" in str(args[0]):
                return
            log.debug(f"{self.address_string()} {fmt % args}")

        def _send(self, status: int, body: str):
            encoded = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path.split("?")[0] == "/sse":
                # SSE streaming endpoint for ACP completion events
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                client_queue = _queue.Queue(maxsize=64)
                with _sse_lock:
                    _sse_clients.append(client_queue)
                try:
                    while True:
                        try:
                            data = client_queue.get(timeout=30)
                            self.wfile.write(f"data: {data}\n\n".encode())
                            self.wfile.flush()
                        except _queue.Empty:
                            # keep-alive ping
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with _sse_lock:
                        if client_queue in _sse_clients:
                            _sse_clients.remove(client_queue)
                return
            self.send_response(404)
            self.end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)

            try:
                rpc = json.loads(raw)
            except Exception as e:
                self._send(200, json.dumps({"jsonrpc": "2.0", "id": None,
                                            "error": {"code": -32700, "message": f"Parse error: {e}"}}))
                return

            rid    = rpc.get("id")
            method = rpc.get("method", "")
            params = rpc.get("params") or {}

            if rid is None:
                self.send_response(202); self.end_headers(); return

            def ok(result):
                self._send(200, json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}))

            def err(code, msg):
                self._send(200, json.dumps({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}))

            try:
                if method == "initialize":
                    log.debug(f"[mcp] initialize request from client")
                    ok({"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                        "serverInfo": {"name": _bundle.server_name, "version": "1.0.0"}})
                elif method == "tools/list":
                    ok({"tools": _bundle.get_all_tools(),
                        "_meta": {"required_site_config": _bundle.required_site_config}})
                elif method == "tools/call":
                    name   = params.get("name", "")
                    args   = params.get("arguments") or {}
                    if name.partition('_')[0] == 'teleop' and args.get('action', 'info') != 'info':
                        import hmac
                        import os
                        from pathlib import Path
                        try:
                            key = Path(os.environ['TELEOP_MANAGEMENT_KEY_FILE']).read_text().strip()
                        except (KeyError, OSError):
                            key = ''
                        if (len(key) < 32 or self.client_address[0] not in ('127.0.0.1', '::1')
                            or self.headers.get('Origin')
                            or not hmac.compare_digest(self.headers.get('X-Teleop-Management', ''), key)):
                            ok({'isError': True, 'content': [{'type': 'text', 'text': json.dumps(
                                {'state': 'error', 'error': 'teleop_management_unauthorized'})}]})
                            return
                    # info action is heartbeat probe — log at DEBUG to reduce noise
                    is_info = (args.get('action') == 'info')
                    if not is_info and name != 'teleop':
                        log.info(f"[mcp] tools/call: {name}({_brief(args)})")
                    result = _bundle.dispatch(name, args)
                    if result is None:
                        err(-32601, f"Unknown tool: {name}")
                    else:
                        if not is_info and name != 'teleop':
                            log.info(f"[mcp] tools/call result: {json.dumps(result)[:200]}")
                        reply = {"content": [{"type": "text", "text": json.dumps(result)}]}
                        if name == 'teleop' and isinstance(result, dict) and (result.get('error') or result.get('state') == 'error'):
                            reply['isError'] = True
                        ok(reply)
                else:
                    err(-32601, f"Method not found: {method}")
            except BrokenPipeError:
                log.debug(f"Client disconnected before response")
            except Exception as e:
                log.error(f"RPC error: {e}", exc_info=True)
                try:
                    err(-32603, str(e))
                except BrokenPipeError:
                    pass

    return Handler


# ── Entry point ───────────────────────────────────────────────────────────────


def _start_registration(mcp_port: int, name: str, category: str):
    """Register this service with agent-core in a background thread, then heartbeat every 30s."""
    import urllib.request as _urllib
    import ssl as _ssl
    agent_core_url = os.environ.get("AGENT_CORE_URL", "https://localhost:15678")
    _ctx = _ssl.create_default_context()
    _ctx.check_hostname = False
    _ctx.verify_mode = _ssl.CERT_NONE
    payload = json.dumps({
        "name": name,
        "url":  f"http://localhost:{mcp_port}/mcp",
        "category": category,
    }).encode()
    def _run():
        import time as _t
        # Log transitions, plus a slow keepalive. A 30s heartbeat that says "ok"
        # every time is 92 of this container's 115 log lines — it crowds out the
        # plugin errors that are the only reason to read this log at all.
        #
        # But edges alone are not enough either, and that was a real loss when
        # this first shipped: every "ok" line used to double as proof the thread
        # was alive, so a wedged register thread showed up as the log going
        # quiet. With edges only, quiet *is* the healthy state, and "fine" and
        # "dead" look identical. The slow line keeps that signal at 1/60th the
        # cost — two lines an hour instead of 120.
        healthy = None
        last_alive = 0.0
        while True:
            try:
                req = _urllib.Request(
                    f"{agent_core_url}/api/mcp", data=payload,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with _urllib.urlopen(req, timeout=3, context=_ctx):
                    now = _t.monotonic()
                    if healthy is not True:
                        log.info(f"[register] heartbeat ok → {agent_core_url}"
                                 + ("" if healthy is None else " (recovered)"))
                        healthy = True
                        last_alive = now
                    elif now - last_alive >= REGISTER_ALIVE_INTERVAL_S:
                        last_alive = now
                        log.info(f"[register] still registered → {agent_core_url}")
                _t.sleep(30)
            except Exception as e:
                # Every failure is logged: a flapping link is a real symptom and
                # collapsing it would hide how often it drops.
                log.warning(f"[register] failed: {e}, retrying in 5s")
                healthy = False
                _t.sleep(5)
    threading.Thread(target=_run, daemon=True, name="register").start()


def main():
    global _bundle

    cfg      = _load_config()
    mcp_port = int(cfg.get("mcp_port", 15730))

    plugins_cfg = cfg.get("plugins") or {}
    enabled = [k for k, v in plugins_cfg.items() if isinstance(v, dict) and v.get("enabled")]
    log.info(f"actucore bundle starting, mcp_port={mcp_port}")
    log.info(f"config: cards enabled={enabled or '(none)'}")

    os.environ.setdefault("RCUTILS_LOGGING_SEVERITY_THRESHOLD", "50")
    os.environ.setdefault("ROS_LOG_LEVEL", "WARN")

    rclpy.init()
    executor = rclpy.executors.MultiThreadedExecutor()
    _bundle  = ActuCoreBundle(cfg, executor)

    def _spin():
        executor.spin()

    spin_thread = threading.Thread(target=_spin, daemon=True, name="actucore_spin")
    spin_thread.start()

    _start_registration(mcp_port, "ActuCore", "actucore")

    server = ThreadingHTTPServer(("", mcp_port), make_handler())
    log.info(f"MCP server → http://0.0.0.0:{mcp_port}")

    def _shutdown(signum, frame):
        log.info(f"signal {signum}, shutting down")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        server.serve_forever()
    finally:
        # 关机顺序是有讲究的：spin 线程还停在 `executor.spin()` 里面（rclpy 的 C++
        # 代码里）时，解释器一旦开始 finalize，就会 `terminate called without an
        # active exception` → `Fatal Python error: Aborted`，容器退出码 134。
        # 天轶上每次 SIGTERM 都会这样，日志里累计 13 次。
        #
        # `executor.shutdown()` 会让 `spin()` 返回，所以在 `rclpy.shutdown()` 之前
        # 把线程 join 掉 —— 让 C++ 那边在解释器还活着的时候退干净。超时是兜底：
        # 关不干净也不能把关机卡死，daemon 线程本来就会被强制收走。
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        if spin_thread.is_alive():
            log.warning("spin thread did not stop within 5s; shutting down anyway")
        rclpy.shutdown()


if __name__ == "__main__":
    main()
