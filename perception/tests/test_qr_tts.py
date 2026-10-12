"""Local native TTS request contract shared by U1 and Tianyi."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from utils.qr_tts import LocalTTS


@pytest.mark.parametrize('url', ['https://127.0.0.1:15740/mcp', 'http://10.100.129.28:15740/mcp',
    'http://127.0.0.1/mcp', 'http://127.0.0.1:15740/other', 'http://user:secret@127.0.0.1:15740/mcp'])
def test_requires_operator_configured_local_mcp(url):
    with pytest.raises(ValueError):
        LocalTTS(url)


def test_real_http_tts_request_never_executes_qr_content():
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body = json.dumps({'jsonrpc': '2.0', 'id': 1, 'result': {'content': [
                {'type': 'text', 'text': json.dumps({'state': 'queued'})}]}}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        tts = LocalTTS(f'http://127.0.0.1:{server.server_port}/mcp')
        assert tts.speak('扫码成功，识别到一个链接。', 'test-1')['state'] == 'submitted'
        request = calls[0]
        assert request['params']['name'] == 'tts'
        assert request['params']['arguments']['action'] == 'speak'
        assert request['params']['arguments']['force'] is False
        assert request['params']['arguments']['action_id'] == 'test-1'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
