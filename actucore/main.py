#!/usr/bin/env python3
"""
actucore/main.py — ActuCore bundle 统一入口。

ActuCore 是 Perception 在执行侧的对称层：Perception 把原始数据流变成语义，
ActuCore 把意图/目标变成运动指令。执行模型（VLA、导航、抓取策略、locomotion、
whole-body control）以卡片（插件）的形式挂在这里，聚合成一个 MCP HTTP server
对外暴露，由 Agent Core 通过 MCP JSON-RPC 调用。

当前挂载的卡片：`navigation`（公开工具名 `ControlledSemanticSpatial`）——
FAST-LIVO2 建图/里程计 + Nav2 规划/控制 + 语义航点，三者作为子进程跑在本容器内，
对外只发布 bounded velocity proposal。新增卡片的完整步骤见 README.md。

MCP 工具命名规则：{plugin_prefix}_{tool_name}；工具名等于 PREFIX 时不加前缀
  例：vla_info、vla_start、ControlledSemanticSpatial

MCP server 端口: config.mcp_port（默认 15730）
"""

from __future__ import annotations

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

from utils.security import redact_sensitive

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)
# suppress noisy third-party loggers
for _quiet in ('urllib3', 'httpcore', 'httpx'):
    logging.getLogger(_quiet).setLevel(logging.WARNING)

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
        self._plugins: list = []
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
        # 需要 ROS 命名空间的卡片必须在自己的注册块里校验运行时支持范围。
        #
        # 卡片契约（PREFIX 不能含下划线、action.enum 必须含 "info" 等）见 README.md。
        # ──────────────────────────────────────────────────────────────────

        navigation_cfg = plugins_cfg.get("navigation", {})
        if not isinstance(navigation_cfg, dict):
            raise ValueError("plugins.navigation must be an object")
        if navigation_cfg.get("enabled", False):
            namespace = navigation_cfg.get("namespace", "ubuntu")
            if not isinstance(namespace, str):
                raise ValueError("plugins.navigation.namespace must be a string")
            namespace = namespace.strip().strip("/")
            if namespace != "ubuntu":
                raise ValueError(
                    "built-in navigation runtime currently requires namespace=ubuntu"
                )
            from plugins.navigation import NavigationPlugin
            plugin = NavigationPlugin(navigation_cfg, namespace, executor)
            self._plugins.append(plugin)
            log.info(
                "NavigationPlugin loaded (single-container runtime, namespace=%s)",
                namespace,
            )

        if not self._plugins:
            log.info("no cards enabled — ActuCore is running as an empty MCP host")

    def get_all_tools(self) -> list:
        tools = []
        for p in self._plugins:
            for t in p.get_tools():
                full_name = t['name'] if t['name'] == p.PREFIX else f"{p.PREFIX}_{t['name']}"
                tools.append({**t, "name": full_name})
        return tools

    def dispatch(self, full_name: str, args: dict) -> dict | None:
        prefix, sep, tool_name = full_name.partition("_")
        name = tool_name if sep else prefix
        for p in self._plugins:
            if p.PREFIX == prefix:
                return p.dispatch(name, args)
        return None

    def stop(self) -> None:
        """Best-effort release for every card that owns runtime resources.

        执行模型卡片常常持有真实资源（导航卡片在容器内跑 FAST-LIVO2 / Nav2
        子进程），进程退出时必须收回，否则重启会撞上残留的 ROS 节点。
        """

        for plugin in reversed(self._plugins):
            stop = getattr(plugin, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    log.exception("failed to stop card %s", plugin.PREFIX)


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
                        "serverInfo": {"name": "actucore-bundle", "version": "1.0.0"}})
                elif method == "tools/list":
                    ok({"tools": _bundle.get_all_tools()})
                elif method == "tools/call":
                    name   = params.get("name", "")
                    args   = params.get("arguments") or {}
                    # info action is heartbeat probe — log at DEBUG to reduce noise
                    is_info = (args.get('action') == 'info')
                    if not is_info:
                        log.info(
                            "[mcp] tools/call: %s(%s)",
                            name,
                            redact_sensitive(args),
                        )
                    result = _bundle.dispatch(name, args)
                    if result is None:
                        err(-32601, f"Unknown tool: {name}")
                    else:
                        if not is_info:
                            safe_result = redact_sensitive(result)
                            log.info(
                                "[mcp] tools/call result: %s",
                                json.dumps(safe_result)[:200],
                            )
                        ok({"content": [{"type": "text", "text": json.dumps(result)}]})
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
        while True:
            try:
                req = _urllib.Request(
                    f"{agent_core_url}/api/mcp", data=payload,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with _urllib.urlopen(req, timeout=3, context=_ctx):
                    log.info(f"[register] heartbeat ok → {agent_core_url}")
                _t.sleep(30)
            except Exception as e:
                log.warning(f"[register] failed: {e}, retrying in 5s")
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

    threading.Thread(target=_spin, daemon=True, name="actucore_spin").start()

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
        _bundle.stop()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
