from __future__ import annotations

import hmac
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bridge_core import BridgeFailure, VisitorBridge


def make_handler(bridge: VisitorBridge, shared_token: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ParadigmVisitorBridge/0.1"

        def log_message(self, fmt, *args):
            # Do not write request bodies, visitor names, cookies, or tokens.
            print(f"[http] {self.address_string()} {fmt % args}", flush=True)

        def send_json(self, status: int, value: dict):
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def authorized(self) -> bool:
            expected = f"Bearer {shared_token}"
            supplied = self.headers.get("Authorization", "")
            return bool(shared_token) and hmac.compare_digest(supplied, expected)

        def read_json(self) -> dict:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise BridgeFailure("bad_request", "Content-Length 无效") from exc
            if length <= 0 or length > 16_384:
                raise BridgeFailure("bad_request", "请求内容为空或过大")
            try:
                value = json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BridgeFailure("bad_request", "请求必须是 JSON") from exc
            if not isinstance(value, dict):
                raise BridgeFailure("bad_request", "请求必须是 JSON 对象")
            return value

        def do_GET(self):
            if self.path == "/health":
                self.send_json(200, {"state": "running"})
            else:
                self.send_json(404, {"ok": False, "code": "not_found"})

        def do_POST(self):
            if not self.authorized():
                self.send_json(401, {"ok": False, "code": "unauthorized", "error": "访问令牌不正确"})
                return
            try:
                body = self.read_json()
                if self.path == "/find_visit":
                    result = bridge.find_visit(
                        body.get("name", ""),
                        body.get("host", ""),
                        body.get("start_time"),
                        body.get("end_time")
                    )

                elif self.path == "/list_visits":
                    result = bridge.list_visits()

                elif self.path == "/print_tag":
                    result = bridge.print_tag(body.get("ticket", ""), body.get("request_id", ""), body.get("printer_name", ""))
                elif self.path == "/printers":
                    result = bridge.backend.list_printers()
                else:
                    self.send_json(404, {"ok": False, "code": "not_found"})
                    return
                self.send_json(200, result)
            except BridgeFailure as exc:
                self.send_json(400, {"ok": False, "code": exc.code, "error": str(exc)})
            except Exception as exc:
                print(f"[bridge] internal error: {type(exc).__name__}: {exc}", flush=True)
                self.send_json(500, {"ok": False, "code": "internal_error", "error": "桥接服务内部错误"})

    return Handler


def serve(bridge: VisitorBridge, shared_token: str, host: str, port: int):
    server = ThreadingHTTPServer((host, port), make_handler(bridge, shared_token))
    print(f"[bridge] listening on {host}:{port}", flush=True)
    server.serve_forever()
