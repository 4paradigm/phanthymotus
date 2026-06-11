#!/usr/bin/env python3
"""
perception/main.py — Perception Stack bundle 统一入口。

读取 config.yaml，按插件配置加载 ASRPlugin / TTSPlugin（以及未来的 VLM、SLAM 等），
聚合成一个 MCP HTTP server 对外暴露。

MCP 工具命名规则：{plugin_prefix}_{tool_name}
  例：asr_info, asr_start, asr_stop, tts_info, tts_start, tts_speak

MCP server 端口: config.mcp_port（默认 15720）
WebSocket ASR 端口: config.ws_port（默认 15721）
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import yaml

import rclpy
import rclpy.executors

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)
# suppress noisy third-party loggers
for _quiet in ('urllib3', 'websockets', 'httpcore', 'httpx', 'dashscope'):
    logging.getLogger(_quiet).setLevel(logging.WARNING)


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = os.environ.get("CONFIG_PATH", str(Path(__file__).parent / "config.yaml"))
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Bundle ────────────────────────────────────────────────────────────────────

class PerceptionBundle:
    def __init__(self, cfg: dict, executor):
        self._plugins: list = []
        plugins_cfg = cfg.get("plugins", {})

        if plugins_cfg.get("asr", {}).get("enabled", False):
            from plugins.asr import ASRPlugin
            self._plugins.append(ASRPlugin(plugins_cfg["asr"], executor))
            log.info("ASRPlugin loaded")

        if plugins_cfg.get("tts", {}).get("enabled", False):
            from plugins.tts import TTSPlugin
            self._plugins.append(TTSPlugin(plugins_cfg["tts"], executor))
            log.info("TTSPlugin loaded")

        # 未来扩展：VLM、SLAM、语义分割等插件在此添加

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

    def tts_synthesize_raw(self, text: str) -> bytes:
        for p in self._plugins:
            if getattr(p, 'PREFIX', None) == 'tts':
                return p.synthesize_raw(text)
        raise RuntimeError("TTS plugin not loaded or not enabled")


# ── MCP HTTP server ───────────────────────────────────────────────────────────

_bundle: PerceptionBundle | None = None


def make_handler():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if args and "/sse" in str(args[0]):
                return
            log.info(f"{self.address_string()} {fmt % args}")

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
            if self.path in ("/vad/test", "/tts/test"):
                self._send(405, '{"error":"Use POST"}')
                return
            if self.path.split("?")[0] == "/sse":
                self.send_response(405)
                self.send_header("Content-Type", "application/json")
                self.send_header("Allow", "POST, OPTIONS")
                self.end_headers()
                self.wfile.write(b'{"error":"SSE not supported. Use HTTP POST."}')
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

            if self.path == "/vad/test":
                try:
                    req = json.loads(raw)
                except Exception:
                    self._send(400, '{"ok":false,"info":"invalid JSON"}')
                    return
                try:
                    import base64, threading as _threading
                    audio_bytes = base64.b64decode(req.get("audio_b64", ""))
                    model      = req.get("model", "silero") or "silero"
                    threshold  = float(req.get("threshold", 0.5))
                    silence_ms = int(req.get("silence_ms", 800))
                    from plugins.asr import _vad_segment_sync
                    result = _vad_segment_sync(audio_bytes, model, threshold, silence_ms)
                    self._send(200, json.dumps({"ok": True, "segments": result}))
                except Exception as e:
                    log.error(f"[vad/test] {e}", exc_info=True)
                    self._send(200, json.dumps({"ok": False, "info": str(e)}))
                return

            if self.path == "/tts/test":
                try:
                    req = json.loads(raw)
                except Exception:
                    self._send(400, '{"ok":false,"info":"invalid JSON"}')
                    return
                try:
                    text = req.get("text", "").strip()
                    if not text:
                        self._send(200, json.dumps({"ok": False, "info": "text is required"}))
                        return
                    # Build ad-hoc adapter from inline credentials if provided
                    api_key = req.get("api_key", "")
                    if api_key:
                        from plugins.tts import AliyunDashScopeTTSAdapter
                        adapter = AliyunDashScopeTTSAdapter(
                            api_key=api_key,
                            model=req.get("model", ""),
                            voice=req.get("voice", ""),
                            url=req.get("url", ""),
                        )
                        pcm = adapter.synthesize(text)
                    else:
                        pcm = _bundle.tts_synthesize_raw(text)
                    import base64 as _b64, io, wave
                    buf = io.BytesIO()
                    with wave.open(buf, 'wb') as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(16000)
                        w.writeframes(pcm)
                    wav_b64 = _b64.b64encode(buf.getvalue()).decode()
                    self._send(200, json.dumps({"ok": True, "wav_b64": wav_b64}))
                except Exception as e:
                    log.error(f"[tts/test] {e}", exc_info=True)
                    self._send(200, json.dumps({"ok": False, "info": str(e)}))
                return

            try:
                rpc = json.loads(raw)
            except Exception:
                self._send(400, json.dumps({"jsonrpc":"2.0","id":None,"error":{"code":-32700,"message":"Parse error"}}))
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
                    log.info(f"[mcp] initialize request from client")
                    ok({"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                        "serverInfo": {"name": "perception-bundle", "version": "1.0.0"}})
                elif method == "tools/list":
                    ok({"tools": _bundle.get_all_tools()})
                elif method == "tools/call":
                    name   = params.get("name", "")
                    args   = params.get("arguments") or {}
                    log.info(f"[mcp] tools/call: {name}({args})")
                    result = _bundle.dispatch(name, args)
                    if result is None:
                        err(-32601, f"Unknown tool: {name}")
                    else:
                        log.info(f"[mcp] tools/call result: {json.dumps(result)[:200]}")
                        ok({"content": [{"type": "text", "text": json.dumps(result)}]})
                else:
                    err(-32601, f"Method not found: {method}")
            except Exception as e:
                log.error(f"RPC error: {e}", exc_info=True)
                err(-32603, str(e))

    return Handler


# ── WebSocket ASR server ───────────────────────────────────────────────────────

async def _ws_asr_handler(websocket):
    """Handle a /ws/asr WebSocket connection.

    Protocol:
      1. Client sends a JSON text frame with ASR config:
         {"provider":"openai","url":"...","key":"...","model":"...","language":"zh-CN"}
      2. Client sends binary frames: raw PCM16 chunks (512 samples @ 16kHz)
      3. Server sends JSON text frames with transcription:
         {"text": "识别结果"}
      4. Client sends text "flush" to force-flush remaining speech
    """
    from plugins.asr import VadSession, _build_asr_adapter, _pcm16_to_wav
    import websockets

    # 1. Receive config frame
    try:
        cfg_raw = await websocket.recv()
        cfg = json.loads(cfg_raw)
    except Exception as e:
        log.warning(f"[ws_asr] invalid config frame: {e}")
        return

    adapter = _build_asr_adapter(cfg)
    language = cfg.get('language', 'zh-CN')

    if adapter is None:
        await websocket.send(json.dumps({'type': 'asr_error', 'payload': {'error': 'ASR adapter not configured'}}))
        return

    session = VadSession()
    session.init()

    await websocket.send(json.dumps({'type': 'asr_ready', 'payload': {'language': language}}))
    log.info(f"[ws_asr] client connected, provider={cfg.get('provider','?')}")

    async def _transcribe_and_send(pcm: bytes):
        try:
            wav = _pcm16_to_wav(pcm)
            text = await asyncio.get_event_loop().run_in_executor(
                None, lambda: adapter.transcribe(wav, language)
            )
            if text and text.strip():
                await websocket.send(json.dumps({'type': 'asr_result', 'payload': {'text': text.strip()}}))
        except Exception as e:
            log.error(f"[ws_asr] transcribe error: {e}")
            try:
                await websocket.send(json.dumps({'type': 'asr_error', 'payload': {'error': str(e)}}))
            except Exception:
                pass

    try:
        async for msg in websocket:
            if isinstance(msg, bytes):
                result = session.process_chunk(msg, __import__('time').time())
                if result:
                    utterance, _, _ = result
                    await _transcribe_and_send(utterance)
            elif isinstance(msg, str):
                if msg == 'flush':
                    result = session.flush() if hasattr(session, 'flush') else None
                    if result:
                        utterance = result[0] if isinstance(result, tuple) else result
                        await _transcribe_and_send(utterance)
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        log.warning(f"[ws_asr] connection error: {e}")


async def _run_ws_server(ws_port: int):
    import websockets
    async with websockets.serve(_ws_asr_handler, "", ws_port):
        log.info(f"WebSocket ASR server → ws://0.0.0.0:{ws_port}")
        await asyncio.Future()  # run forever


def _start_ws_thread(ws_port: int):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_run_ws_server(ws_port))


# ── Entry point ───────────────────────────────────────────────────────────────


def _start_registration(mcp_port: int, name: str, category: str):
    """Register this driver with agent-core in a background thread, then heartbeat every 30s."""
    import urllib.request as _urllib
    agent_core_url = os.environ.get("AGENT_CORE_URL", "http://localhost:15678")
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
                with _urllib.urlopen(req, timeout=3):
                    log.info(f"[register] heartbeat ok → {agent_core_url}")
                _t.sleep(30)
            except Exception as e:
                log.warning(f"[register] failed: {e}, retrying in 5s")
                _t.sleep(5)
    threading.Thread(target=_run, daemon=True, name="register").start()


def main():
    global _bundle

    cfg      = _load_config()
    mcp_port = int(cfg.get("mcp_port", 15720))
    ws_port  = int(cfg.get("ws_port",  15721))

    log.info(f"perception bundle starting, mcp_port={mcp_port}, ws_port={ws_port}")
    log.info(f"config: plugins.asr.enabled={cfg.get('plugins',{}).get('asr',{}).get('enabled')}, "
             f"plugins.tts.enabled={cfg.get('plugins',{}).get('tts',{}).get('enabled')}")
    asr_cfg = cfg.get('plugins',{}).get('asr',{})
    tts_cfg = cfg.get('plugins',{}).get('tts',{})
    log.info(f"  asr: provider={asr_cfg.get('provider')}, url={asr_cfg.get('url','')[:40] or '(empty)'}, "
             f"model={asr_cfg.get('model','') or '(default)'}, key={'set' if asr_cfg.get('key') else 'MISSING'}")
    log.info(f"  tts: provider={tts_cfg.get('provider')}, model={tts_cfg.get('model','') or '(default)'}, "
             f"api_key={'set' if tts_cfg.get('api_key') else 'MISSING'}")

    os.environ.setdefault("RCUTILS_LOGGING_SEVERITY_THRESHOLD", "50")
    os.environ.setdefault("ROS_LOG_LEVEL", "WARN")

    rclpy.init()
    executor = rclpy.executors.MultiThreadedExecutor()
    _bundle  = PerceptionBundle(cfg, executor)

    def _spin():
        executor.spin()

    threading.Thread(target=_spin, daemon=True, name="perception_spin").start()

    # Start WebSocket ASR server in a separate thread
    threading.Thread(target=_start_ws_thread, args=(ws_port,), daemon=True, name="ws_asr").start()

    _start_registration(mcp_port, "Perception Stack", "perception")

    server = HTTPServer(("", mcp_port), make_handler())
    log.info(f"MCP server → http://0.0.0.0:{mcp_port}")

    def _shutdown(signum, frame):
        log.info(f"signal {signum}, shutting down")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        server.serve_forever()
    finally:
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
