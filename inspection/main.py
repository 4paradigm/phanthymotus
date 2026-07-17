#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import signal
import ssl
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

from plugins.audioinspector import AudioInspectorPlugin
from plugins.videoinspector import VideoInspectorPlugin


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("inspection")


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class InspectionBundle:
    def __init__(self, config: dict, executor=None) -> None:
        plugins = config.get("plugins", {})
        self._plugins = []
        if plugins.get("audioinspector", {}).get("enabled", True):
            self._plugins.append(AudioInspectorPlugin(plugins.get("audioinspector", {}), executor))
        if plugins.get("videoinspector", {}).get("enabled", True):
            self._plugins.append(VideoInspectorPlugin())

    def get_all_tools(self) -> list[dict]:
        tools: list[dict] = []
        for plugin in self._plugins:
            tools.extend(plugin.get_tools())
        return tools

    def dispatch(self, name: str, arguments: dict) -> dict | None:
        for plugin in self._plugins:
            result = plugin.dispatch(name, arguments)
            if result is not None:
                return result
        return None

    def shutdown(self) -> None:
        for plugin in self._plugins:
            plugin.shutdown()

    def runtime_modes(self) -> list[str]:
        return sorted({str(plugin._runtime_mode) for plugin in self._plugins})


def load_config() -> dict:
    import yaml

    config_path = Path(os.environ.get("CONFIG_PATH", Path(__file__).parent / "config.yaml"))
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def make_handler(bundle: InspectionBundle, server_name: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            log.debug("%s %s", self.address_string(), fmt % args)

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"ok": True, "name": server_name, "runtime_modes": bundle.runtime_modes()})
                return
            self._send_json(404, {"error": "not found"})

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.end_headers()

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._send_json(400, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
                return

            request_id = request.get("id")
            method = request.get("method", "")
            params = request.get("params") or {}
            try:
                if method == "initialize":
                    result = {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": server_name, "version": "0.1.0"},
                    }
                elif method == "tools/list":
                    result = {"tools": bundle.get_all_tools()}
                elif method == "tools/call":
                    tool_name = str(params.get("name", ""))
                    arguments = params.get("arguments") or {}
                    response = bundle.dispatch(tool_name, arguments)
                    if response is None:
                        raise LookupError(f"Unknown tool: {tool_name}")
                    result = {"content": [{"type": "text", "text": json.dumps(response, ensure_ascii=False)}]}
                else:
                    raise LookupError(f"Method not found: {method}")
                self._send_json(200, {"jsonrpc": "2.0", "id": request_id, "result": result})
            except LookupError as exc:
                self._send_json(200, {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": str(exc)}})
            except Exception as exc:
                log.warning("MCP call failed: %s", exc)
                self._send_json(200, {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32602, "message": str(exc)}})

    return Handler


def start_registration(mcp_port: int, name: str, category: str) -> None:
    agent_core_url = os.environ.get("AGENT_CORE_URL", "https://localhost:15678")
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    payload = json.dumps({
        "name": name,
        "url": f"http://localhost:{mcp_port}/mcp",
        "category": category,
    }).encode("utf-8")

    def run() -> None:
        while True:
            try:
                request = urllib.request.Request(
                    f"{agent_core_url}/api/mcp",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=3, context=context):
                    log.info("registration heartbeat ok -> %s", agent_core_url)
                time.sleep(30)
            except Exception as exc:
                log.warning("registration failed: %s; retrying in 5s", exc)
                time.sleep(5)

    threading.Thread(target=run, daemon=True, name="inspection-register").start()


def main() -> None:
    config = load_config()
    mcp_port = int(config.get("mcp_port", 15671))
    server_name = str(config.get("server_name", "inspection-bundle"))
    category = str(config.get("category", "inspection"))
    audio_config = config.get("plugins", {}).get("audioinspector", {})
    ros_enabled = audio_config.get("enabled", True) and audio_config.get("runtime_mode") == "ros2"
    executor = None
    rclpy_module = None
    if ros_enabled:
        import rclpy
        import rclpy.executors

        rclpy.init()
        rclpy_module = rclpy
        executor = rclpy.executors.MultiThreadedExecutor()

    bundle = InspectionBundle(config, executor)
    if executor is not None:
        threading.Thread(target=executor.spin, daemon=True, name="inspection-ros-spin").start()
    server = ThreadingHTTPServer(("", mcp_port), make_handler(bundle, server_name))

    def shutdown(signum, _frame) -> None:
        log.info("signal %s received; stopping inspector instances", signum)
        bundle.shutdown()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    start_registration(mcp_port, "Inspection Stack", category)
    log.info("MCP server -> http://0.0.0.0:%s/mcp", mcp_port)
    try:
        server.serve_forever()
    finally:
        bundle.shutdown()
        server.server_close()
        if executor is not None:
            executor.shutdown()
        if rclpy_module is not None:
            rclpy_module.shutdown()


if __name__ == "__main__":
    main()
