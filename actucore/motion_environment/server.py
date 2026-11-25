"""Private host bootstrap API; no ROS, SSH keys, or Core Docker socket needed."""
from __future__ import annotations

import argparse
import fcntl
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import threading

from .manager import EnvironmentManager, secret


def make_handler(manager, token):
    if len(token) < 32:
        raise ValueError("环境服务凭证至少需要 32 个字符")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def reply(self, code, value):
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def authorized(self):
            if not hmac.compare_digest(self.headers.get("Authorization", "").encode(),
                                       ("Bearer " + token).encode()):
                self.reply(401, {"state": "error", "message": "环境服务认证失败"})
                return False
            return True

        def do_GET(self):
            if self.authorized():
                self.reply(200, manager.health()) if self.path == "/health" else self.send_error(404)

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != "/v1/call":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192 or self.headers.get("Transfer-Encoding"):
                    raise ValueError("请求大小无效")
                args = json.loads(self.rfile.read(length))
                if not isinstance(args, dict):
                    raise ValueError("请求必须为对象")
                self.reply(200, manager.call(args))
            except (ValueError, KeyError) as exc:
                self.reply(400, {"state": "error", "message": str(exc)})
            except Exception:
                self.reply(500, {"state": "error", "message": "环境服务内部错误，请查看服务日志"})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15734)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--token-file", required=True)
    args = parser.parse_args()
    Path(args.root).mkdir(parents=True, exist_ok=True)
    owner = (Path(args.root) / "manager.lock").open("a")
    # Avoid two HTTP daemons independently submitting jobs to the same host state.
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manager = EnvironmentManager(args.root, args.config)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(manager, secret(args.token_file)))
    server.daemon_threads = True
    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        manager.close()
        owner.close()


if __name__ == "__main__":
    main()
