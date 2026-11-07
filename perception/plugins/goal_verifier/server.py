"""Standalone read-only PiPER vision MCP, compatible with Perception/Core tools.

Default loopback binding permits a same-host Core without exposing evidence
submission to the LAN. This server never imports a robot driver or CAN SDK.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import signal
import threading
import urllib.request

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
        dict(jsonrpc="2.0", id=1, method=method, params=params or {})
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
            if plugin.image_exchange:
                data = plugin.image_exchange.get(self.path)
                if data is not None:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
            if self.path == "/health":
                self.send(
                    dict(
                        state="running",
                        service="piper-goal-verifier",
                        motion_capability=False,
                    )
                )
            else:
                self.send(dict(error="not found"), 404)

        def do_POST(self):
            if self.path != "/mcp":
                self.send(dict(error="not found"), 404)
                return
            rid = None
            try:
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
                    out = dict(
                        protocolVersion="2024-11-05",
                        capabilities=dict(tools={}),
                        serverInfo=dict(name="piper-goal-verifier", version="1.0.0"),
                    )
                elif method == "tools/list":
                    out = dict(tools=plugin.get_tools())
                elif method == "resources/list":
                    out = dict(resources=[])
                elif method == "ping":
                    out = {}
                elif method == "tools/call":
                    need(params.get("name") == NAME, "UNKNOWN_TOOL")
                    result = plugin.dispatch(NAME, params.get("arguments", {}))
                    out = dict(
                        content=[dict(type="text", text=canonical(result))],
                        isError=result.get("state") == "error",
                    )
                else:
                    raise Refusal("METHOD_NOT_FOUND")
                self.send(dict(jsonrpc="2.0", id=rid, result=out))
            except Exception as exc:
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
                    dict(
                        jsonrpc="2.0",
                        id=rid,
                        error=dict(
                            code=code, message=str(exc), data=dict(reason_code=reason)
                        ),
                    )
                )

    return Handler


def main():
    # Before argument parsing, plugin imports or startup output; library imports
    # continue to use their host process's logging setup.
    from utils import logsafe

    logsafe.install()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1"])
    p.add_argument("--port", type=int, default=15785)
    args = p.parse_args()
    from .plugin import GoalVerifierPlugin

    plugin = GoalVerifierPlugin(loads(args.config.read_text()))
    server = ThreadingHTTPServer((args.host, args.port), handler(plugin))
    from .exchange import FrameExchange

    plugin.image_exchange = FrameExchange(f"http://{args.host}:{server.server_port}")

    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(
        canonical(
            dict(
                mcp_url=f"http://{args.host}:{server.server_port}/mcp",
                motion_capability=False,
            )
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
