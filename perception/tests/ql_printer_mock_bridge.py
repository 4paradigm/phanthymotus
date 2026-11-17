#!/usr/bin/env python3
"""访客贴桥接服务模拟版，用于联调 QL_printer 卡片。

它遵循生产桥接协议，但不连接范式+或 USB 打印机：
    POST /list_visits {}
    POST /print_tag  {"ticket": "...", "request_id": "..."}

运行：
    python3 visitor_badge_mock_bridge.py --port 8765 --token test-token
"""

from __future__ import annotations

import argparse
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

SAMPLE_VISITORS = [
    {"id": "V1001", "name": "王磊", "host": "张三", "time": "10:00", "tail": "1234", "reason": "面试"},
    {"id": "V1002", "name": "李娜", "host": "李臻", "time": "14:30", "tail": "5678", "reason": "商务拜访"},
    {"id": "V1003", "name": "李娜", "host": "行政部", "time": "15:00", "tail": "9012", "reason": "物业服务"},
]


class MockBridge:
    def __init__(self, token: str = "", ticket_seconds: float = 300, printer_ready: bool = True):
        self.token = token
        self.ticket_seconds = ticket_seconds
        self.printer_ready = printer_ready
        self.visitors = [dict(v) for v in SAMPLE_VISITORS]
        self.printed: set[str] = set()
        self.tickets: dict[str, dict] = {}
        self.requests: dict[str, dict] = {}
        self.print_log: list[str] = []
        self.lock = threading.Lock()

    def handler(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def reply(self, status: int, payload: dict):
                body = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def authorized(self) -> bool:
                if bridge.token and self.headers.get("Authorization") != f"Bearer {bridge.token}":
                    self.reply(401, {"ok": False, "error": "访问令牌不正确", "code": "unauthorized"})
                    return False
                return True

            def read_json(self) -> dict | None:
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    self.reply(400, {"ok": False, "error": "请求不是 JSON", "code": "bad_request"})
                    return None
                if not isinstance(body, dict):
                    self.reply(400, {"ok": False, "error": "请求必须是 JSON 对象", "code": "bad_request"})
                    return None
                return body

            def do_POST(self):
                if not self.authorized():
                    return
                body = self.read_json()
                if body is None:
                    return
                path = urlparse(self.path).path
                if path == "/find_visit":
                    self.find_visit(body)
                elif path == "/list_visits":
                    self.list_visits()
                elif path == "/print_tag":
                    self.print_tag(body)
                else:
                    self.reply(404, {"ok": False, "error": "没有这个接口", "code": "not_found"})

            def find_visit(self, body: dict):
                name = str(body.get("name", "")).strip()
                host = str(body.get("host", "")).strip()
                if not name:
                    self.reply(400, {"error": "缺少访客姓名", "code": "bad_request"})
                    return
                with bridge.lock:
                    hits = [v for v in bridge.visitors if v["name"] == name and (not host or host in v["host"])]
                    visits = []
                    for visitor in hits:
                        ticket = secrets.token_urlsafe(24)
                        bridge.tickets[ticket] = {
                            "visitor_id": visitor["id"],
                            "expires_at": time.monotonic() + bridge.ticket_seconds,
                        }
                        visits.append({
                            "ticket": ticket,
                            "name": visitor["name"],
                            "tail": visitor["tail"],
                            "host": visitor["host"],
                            "time": visitor["time"],
                            "reason": visitor["reason"],
                            "status": "已入场" if visitor["id"] in bridge.printed else "未入场",
                        })
                self.reply(200, {"count": len(visits), "visits": visits})

            def list_visits(self):
                with bridge.lock:
                    hits = [v for v in bridge.visitors if v["id"] not in bridge.printed]
                    visits = []
                    for visitor in hits:
                        ticket = secrets.token_urlsafe(24)
                        bridge.tickets[ticket] = {
                            "visitor_id": visitor["id"],
                            "expires_at": time.monotonic() + bridge.ticket_seconds,
                        }
                        visits.append({
                            "ticket": ticket,
                            "name": visitor["name"],
                            "tail": visitor["tail"],
                            "host": visitor["host"],
                            "time": visitor["time"],
                            "reason": visitor["reason"],
                            "status": "未入场",
                        })
                self.reply(200, {"date": "2026-11-17", "count": len(visits), "visits": visits})

            def print_tag(self, body: dict):
                ticket = str(body.get("ticket", "")).strip()
                request_id = str(body.get("request_id", "")).strip()
                if not ticket or not request_id:
                    self.reply(400, {"ok": False, "error": "缺少 ticket 或 request_id", "code": "bad_request"})
                    return
                with bridge.lock:
                    previous = bridge.requests.get(request_id)
                    if previous is not None:
                        self.reply(200, previous)
                        return
                    ticket_info = bridge.tickets.get(ticket)
                    if not ticket_info or time.monotonic() >= ticket_info["expires_at"]:
                        result = {"ok": False, "error": "查询票据无效或已过期，请重新查询访客", "code": "ticket_expired"}
                    elif not bridge.printer_ready:
                        result = {"ok": False, "error": "打印机缺纸，请前台同事检查", "code": "printer_not_ready"}
                    elif ticket_info["visitor_id"] in bridge.printed:
                        result = {"ok": False, "error": "这位访客已经打印过访客贴", "code": "already_printed"}
                    else:
                        visitor_id = ticket_info["visitor_id"]
                        bridge.printed.add(visitor_id)
                        bridge.print_log.append(visitor_id)
                        result = {"ok": True}
                    bridge.requests[request_id] = result
                self.reply(200, result)

        return Handler

    def serve(self, host: str, port: int) -> ThreadingHTTPServer:
        return ThreadingHTTPServer((host, port), self.handler())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    server = MockBridge(args.token).serve(args.host, args.port)
    print(f"[mock-bridge] 监听 {args.host}:{args.port}，示例访客：王磊、李娜（两位）", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()



