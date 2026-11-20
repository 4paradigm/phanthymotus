#!/usr/bin/env python3
"""Loopback-only development host for the two real motion workbenches (no ROS)."""

from __future__ import annotations

import argparse
import hmac
import importlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL = "motus.workbench/1"
MAX_BODY = 40_000
FRAME_CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
             "img-src data: blob:; connect-src 'none'; form-action 'none'; base-uri 'none'")

PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phanthymotus 动作工作台</title>
<style>*{box-sizing:border-box}body{margin:0;background:#10151f;color:#e3e8f0;font:14px system-ui}
body{height:100vh;display:flex;flex-direction:column}nav{display:flex;gap:8px;padding:10px 18px;border-bottom:1px solid #293241;align-items:center}
strong{margin-right:18px;font-size:14px}button{border:1px solid #364154;border-radius:7px;padding:8px 18px;background:#1c2534;color:inherit;cursor:pointer}
button[aria-pressed=true]{background:#246be8;border-color:#4985ee}#error{color:#ff9b91;margin-left:12px}iframe{border:0;width:100%;flex:1;min-height:0}</style>
<nav><strong>Phanthymotus</strong><button data-card="motion" aria-pressed="true">全身动作</button><button data-card="simulation" aria-pressed="false">仿真环境</button><span id="error" role="status"></span></nav>
<script>
const bridgeToken=__BRIDGE_TOKEN__,protocol='motus.workbench/1',frameCsp=__FRAME_CSP__;
let generation=0,currentPort=null;
async function api(path,body){const options={headers:{'X-Workbench-Token':bridgeToken},cache:'no-store'};
if(body!==undefined){options.method='POST';options.headers['Content-Type']='application/json';options.body=JSON.stringify(body);}
const response=await fetch(path,options);let value;try{value=await response.json();}catch{throw Error('本机工作台返回了无效响应');}
if(!response.ok)throw Error(value.error||`HTTP ${response.status}`);return value;}
async function openCard(card){const opened=++generation;currentPort?.close();document.querySelector('iframe')?.remove();
document.querySelector('#error').textContent='';document.querySelectorAll('[data-card]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.card===card)));
try{const ui=await api('/ui/'+card);if(opened!==generation)return;
if(ui.protocol!==protocol||typeof ui.html!=='string'||ui.html.length>250000)throw Error('无效工作台页面');
const frame=document.createElement('iframe');frame.title=card==='motion'?'全身动作':'仿真环境';frame.setAttribute('sandbox','allow-scripts');frame.referrerPolicy='no-referrer';
frame.onload=()=>{if(opened!==generation)return;const channel=new MessageChannel(),port=channel.port1;currentPort=port;
port.onmessage=async event=>{const request=event.data;try{
if(!request||typeof request!=='object'||typeof request.id!=='string'||request.id.length>128||JSON.stringify(request).length>40000)throw Error('无效工作台请求');
if(opened!==generation)throw Error('工作台已关闭');
const result=await api('/api/'+card,{method:request.method,args:request.args||{}});
if(opened===generation)port.postMessage({id:request.id,result});
}catch(error){if(opened===generation&&request?.id)port.postMessage({id:request.id,error:error.message});}};
frame.contentWindow.postMessage({protocol},'*',[channel.port2]);};
frame.srcdoc='<meta http-equiv="Content-Security-Policy" content="'+frameCsp+'">'+ui.html;document.body.append(frame);
}catch(error){if(opened===generation)document.querySelector('#error').textContent=error.message;}}
document.querySelectorAll('[data-card]').forEach(b=>b.onclick=()=>openCard(b.dataset.card));openCard('motion');
</script></html>"""


class RequestError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def load_plugins(driver_root: Path, endpoint: str, token: str):
    """Import just plugin packages; neither repository's ROS entrypoint is loaded."""
    driver_root = driver_root.resolve()
    if not (driver_root / "simulator/mujoco/plugin.py").is_file():
        raise ValueError("--driver-root 必须指向包含 simulator/mujoco 的 Driver 仓库")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(driver_root))
    motion = importlib.import_module("plugins.motion").MotionPlugin
    simulation = importlib.import_module("simulator.mujoco.plugin").SimulationPlugin
    config = {"endpoint": endpoint, "api_key": token}
    plugins = {"motion": motion(config), "simulation": simulation(config)}
    for name, plugin in plugins.items():
        state = plugin.dispatch(name, {"action": "info"})
        if state.get("state") != "ready":
            raise ValueError(state.get("message") or "服务器地址或凭证未配置")
    return plugins


def read_token(token_file: Path | None):
    if token_file is None:
        token = os.environ.get("MOTION_API_TOKEN", "")
    else:
        with token_file.open("rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
                raise ValueError("凭证文件必须是权限为 0600 的普通文件")
            token = stream.read(4097).decode("ascii").strip()
    if not token or len(token) > 4096 or any(ord(c) < 33 or ord(c) > 126 for c in token):
        raise ValueError("请通过 --token-file 或 MOTION_API_TOKEN 提供有效服务凭证")
    return token


def validate_value(value, schema):
    kind = schema.get("type")
    valid = {"string": lambda: isinstance(value, str),
             "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
             "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
             "array": lambda: isinstance(value, list)}.get(kind, lambda: False)()
    if not valid:
        raise RequestError("参数类型不正确")
    if "enum" in schema and value not in schema["enum"]:
        raise RequestError("参数取值不正确")
    if kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", MAX_BODY):
            raise RequestError("数组长度不正确")
        for item in value:
            validate_value(item, schema.get("items", {}))


class Workbench:
    def __init__(self, plugins):
        self.plugins = plugins
        self.tools = {name: plugin.get_tools()[0] for name, plugin in plugins.items()}
        self.locks = {name: threading.RLock() for name in plugins}

    def dispatch(self, card, args):
        result = self.plugins[card].dispatch(card, {**args, "instance_id": "local-workbench"})
        if not isinstance(result, dict):
            raise RequestError("插件返回了无效响应", 502)
        if result.get("error") or result.get("state") == "error":
            raise RequestError(result.get("error") or result.get("message") or "服务器操作失败", 502)
        return result

    def ui(self, card):
        if card not in self.plugins:
            raise RequestError("工作台不存在", 404)
        result = self.dispatch(card, {"action": "ui"})
        if result.get("protocol") != PROTOCOL or not isinstance(result.get("html"), str) or len(result["html"]) > 250_000:
            raise RequestError("插件返回了无效工作台页面", 502)
        return result

    def request(self, card, body):
        if card not in self.plugins:
            raise RequestError("工作台不存在", 404)
        if not isinstance(body, dict) or set(body) != {"method", "args"} or not isinstance(body["args"], dict):
            raise RequestError("无效工作台请求")
        method, args = body["method"], body["args"]
        tool = self.tools[card]
        spec = tool["inputSchema"]["x-workbench"]
        with self.locks[card]:
            if method == "call":
                action = args.get("action")
                if not isinstance(action, str) or action not in spec["actions"] or action in {"config", "ui"}:
                    raise RequestError("工作台无权执行此操作", 403)
                properties = tool["inputSchema"]["properties"]
                for key, value in args.items():
                    if key not in properties:
                        raise RequestError("工作台参数未授权", 403)
                    validate_value(value, properties[key])
                return self.dispatch(card, args)
            if method != "config" or "config" not in spec["actions"]:
                raise RequestError("工作台方法未授权", 403)
            fields = {key: value for key, value in tool["configSchema"]["properties"].items()
                      if key in spec.get("config", []) and value.get("scope") != "instance"}
            if args:
                if set(args) != {"values"} or not isinstance(args["values"], dict):
                    raise RequestError("无效连接配置")
                values = args["values"]
                for key, value in values.items():
                    if key not in fields:
                        raise RequestError("连接配置未授权", 403)
                    validate_value(value, fields[key])
                self.dispatch(card, {"action": "config", **values})
                return {"saved": True}
            info = self.dispatch(card, {"action": "info"})
            sensitive = {key for key, value in fields.items() if value.get("x-sensitive") or value.get("format") == "password"}
            values = {key: info.get("config_values", {}).get(key, "") for key in fields if key not in sensitive}
            secret_set = {key: info.get("config_secrets", {}).get(key) is True for key in sensitive}
            return {"values": values, "secret_set": secret_set, "runtime_secret_set": secret_set}


class WorkbenchServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, workbench, port=15733):
        self.workbench = workbench
        self.bridge_token = secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", port), WorkbenchHandler)
        self.host = f"127.0.0.1:{self.server_port}"
        self.origin = f"http://{self.host}"


class WorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "MotionWorkbench"

    def log_message(self, *args):
        pass  # Do not log credentials, query strings, or request payloads.

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def check_access(self, *, api=False, mutation=False):
        if self.headers.get_all("Host") != [self.server.host]:
            raise RequestError("不允许的 Host", 403)
        origin = self.headers.get("Origin")
        if origin and origin != self.server.origin:
            raise RequestError("不允许的来源", 403)
        if mutation and origin != self.server.origin:
            raise RequestError("请求缺少同源 Origin", 403)
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise RequestError("不允许跨站请求", 403)
        if api and not hmac.compare_digest(self.headers.get("X-Workbench-Token", ""), self.server.bridge_token):
            raise RequestError("工作台会话无效，请刷新页面", 403)

    def respond(self, status, value, html=False):
        payload = value.encode() if html else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8" if html else "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; frame-src 'self'; connect-src 'self'; img-src data: blob:; frame-ancestors 'none'; form-action 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        try:
            self.check_access(api=self.path.startswith("/ui/"))
            if self.path == "/":
                page = PAGE.replace("__BRIDGE_TOKEN__", json.dumps(self.server.bridge_token)).replace("__FRAME_CSP__", json.dumps(FRAME_CSP))
                self.respond(200, page, html=True)
            elif self.path.startswith("/ui/"):
                self.respond(200, self.server.workbench.ui(self.path[4:]))
            else:
                raise RequestError("页面不存在", 404)
        except RequestError as exc:
            self.respond(exc.status, {"error": str(exc)})

    def do_POST(self):
        try:
            self.check_access(api=True, mutation=True)
            if not self.path.startswith("/api/"):
                raise RequestError("接口不存在", 404)
            if self.headers.get("Transfer-Encoding") or len(self.headers.get_all("Content-Length", [])) != 1:
                raise RequestError("请求必须提供唯一 Content-Length")
            length = self.headers.get("Content-Length", "")
            if not length.isdigit() or int(length) > MAX_BODY:
                raise RequestError("请求超过 40 KB 限制", 413)
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                raise RequestError("请求必须使用 application/json", 415)
            try:
                body = json.loads(self.rfile.read(int(length)))
            except (ValueError, UnicodeError) as exc:
                raise RequestError("无效 JSON 请求") from exc
            self.respond(200, self.server.workbench.request(self.path[5:], body))
        except RequestError as exc:
            self.respond(exc.status, {"error": str(exc)})
        except (TimeoutError, ConnectionError):
            self.close_connection = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver-root", type=Path, required=True)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--endpoint", default=os.environ.get("MOTION_SERVICE_URL", "http://127.0.0.1:15732"))
    options = parser.parse_args(argv)
    try:
        token = read_token(options.token_file)
        server = WorkbenchServer(Workbench(load_plugins(options.driver_root, options.endpoint, token)))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"无法启动工作台：{exc}\n")
    print(f"本机开发工作台：{server.origin}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
