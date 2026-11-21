"""Local offline acceptance surface. Production uses PerceptionBundle's MCP."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import time
from .plugin import FaceEffectsLivePlugin


def make_server(config, port=0):
    plugin = FaceEffectsLivePlugin(config)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, content, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path == "/":
                self.send(200, Path(__file__).with_name("preview.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/frame.jpg":
                with plugin.lock:
                    service = plugin.instances.get("default")
                    if service:
                        with service.condition:
                            jpeg = service.jpeg if time.monotonic()-service.last_output_at < .25 else None
                    else:
                        jpeg = None
                self.send(200 if jpeg else 204, jpeg or b"", "image/jpeg")
            elif self.path == "/status":
                self.send(200, json.dumps(plugin.dispatch("info", {"instance_id": "default"})).encode())
            else:
                self.send(404, b"{}")

        def do_POST(self):
            if self.path != "/mcp":
                self.send(404, b"{}")
                return
            # Local UI is same-origin; prevent cross-origin control of local inputs.
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"):
                self.send(403, b"{}")
                return
            request_id = None
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("request length")
                rpc = json.loads(self.rfile.read(length))
                request_id = rpc.get("id")
                method, params = rpc.get("method"), rpc.get("params", {})
                if method == "initialize":
                    result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                              "serverInfo": {"name": "face-effects-offline", "version": "0.1.0"}}
                elif method == "tools/list":
                    result = {"tools": plugin.get_tools()}
                elif method == "tools/call" and params.get("name") == plugin.PREFIX:
                    response = plugin.dispatch(plugin.PREFIX, params.get("arguments", {}))
                    if response is None:
                        raise ValueError("unknown action")
                    result = {"content": [{"type": "text", "text": json.dumps(response)}]}
                else:
                    raise ValueError("unknown method/tool")
                self.send(200, json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}).encode())
            except (ValueError, TypeError, KeyError, AttributeError):
                self.send(400, json.dumps({"jsonrpc": "2.0", "id": request_id,
                                          "error": {"code": -32600, "message": "invalid request"}}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server, plugin


def serve(config, port=0):
    server, plugin = make_server(config, port)
    print(f"Offline Canvas preview: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        plugin.shutdown()
        server.server_close()
