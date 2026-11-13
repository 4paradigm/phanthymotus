"""Standalone read-only ROI vision MCP, compatible with Perception/Core tools.

Default loopback binding permits a same-host Core without exposing evidence
submission to the LAN. This server never imports a robot driver or CAN SDK.
"""

import argparse
import json
import signal
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .contracts import NAME, Refusal, canonical, need


def loads(raw):
    def pairs(values):
        out = {}
        for k, v in values:
            need(k not in out, "DUPLICATE_JSON_KEY")
            out[k] = v
        return out

    def invalid(value):
        raise Refusal("INVALID_NUMBER")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def rpc(url, method, params=None, *, timeout=4):
    from urllib.parse import urlparse

    parsed = urlparse(url)
    need(
        parsed.scheme in ("http", "https") and parsed.hostname and not parsed.username,
        "INVALID_MCP_URL",
    )
    body = canonical(
        {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        result = loads(response.read())
    need("error" not in result, "MCP_UPSTREAM_ERROR", str(result.get("error")))
    return result["result"]


def handler(plugin):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, result, status=200):
            data = canonical(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/workbench"):
                data = Path(__file__).with_name("workbench.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if self.path == "/health":
                self.send(
                    {
                        "state": "running",
                        "service": "tabletop-roi-goal-verifier",
                        "motion_capability": False,
                    }
                )
            else:
                self.send({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/mcp":
                self.send({"error": "not found"}, 404)
                return
            rid = None
            try:
                from urllib.parse import urlparse

                host = self.headers.get("Host", "")
                need(
                    urlparse("http://" + host).hostname in ("127.0.0.1", "localhost"),
                    "INVALID_HOST",
                )
                origin = self.headers.get("Origin")
                need(not origin or origin == "http://" + host, "INVALID_ORIGIN")
                need(
                    self.headers.get_content_type() == "application/json",
                    "INVALID_CONTENT_TYPE",
                )
                length = int(self.headers.get("Content-Length", "0"))
                need(0 < length <= 1024 * 1024, "REQUEST_TOO_LARGE")
                request = loads(self.rfile.read(length))
                need(isinstance(request, dict), "INVALID_RPC")
                rid = request.get("id")
                need(request.get("jsonrpc") == "2.0", "INVALID_RPC")
                if rid is None:
                    self.send_response(202)
                    self.end_headers()
                    return
                method = request.get("method")
                params = request.get("params", {})
                if method == "initialize":
                    out = {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {
                            "name": "tabletop-roi-goal-verifier",
                            "version": "1.0.0",
                        },
                    }
                elif method == "tools/list":
                    out = {"tools": plugin.get_tools()}
                elif method == "resources/list":
                    out = {"resources": []}
                elif method == "ping":
                    out = {}
                elif method == "tools/call":
                    need(params.get("name") == NAME, "UNKNOWN_TOOL")
                    result = plugin.dispatch(NAME, params.get("arguments", {}))
                    out = {
                        "content": [{"type": "text", "text": canonical(result)}],
                        "isError": result.get("state") == "error",
                    }
                else:
                    raise Refusal("METHOD_NOT_FOUND")
                self.send({"jsonrpc": "2.0", "id": rid, "result": out})
            except Exception as exc:  # noqa: BLE001 — return transport/source diagnostics
                reason = getattr(exc, "code", "INTERNAL_ERROR")
                code = (
                    -32700
                    if isinstance(exc, json.JSONDecodeError)
                    or reason in ("DUPLICATE_JSON_KEY", "INVALID_NUMBER")
                    else -32601
                    if reason in ("UNKNOWN_TOOL", "METHOD_NOT_FOUND")
                    else -32600
                    if reason in ("INVALID_RPC", "REQUEST_TOO_LARGE")
                    else -32602
                    if isinstance(exc, (Refusal, KeyError, TypeError))
                    else -32603
                )
                self.send(
                    {
                        "jsonrpc": "2.0",
                        "id": rid,
                        "error": {
                            "code": code,
                            "message": str(exc),
                            "data": {"reason_code": reason},
                        },
                    }
                )

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1"])
    p.add_argument("--port", type=int, default=18084)
    args = p.parse_args()
    from .plugin import TabletopROIGoalVerifierPlugin

    plugin = TabletopROIGoalVerifierPlugin(loads(args.config.read_text()))
    server = ThreadingHTTPServer((args.host, args.port), handler(plugin))

    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    startup = plugin.dispatch(NAME, {"action": "start"})
    print(canonical({"startup": startup}), flush=True)
    print(
        canonical(
            {
                "mcp_url": f"http://{args.host}:{server.server_port}/mcp",
                "motion_capability": False,
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        plugin.stop()
        server.server_close()


if __name__ == "__main__":
    main()
