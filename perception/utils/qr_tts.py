"""Optional local MCP TTS adapter; QR data cannot select tools or endpoints."""
import json
import urllib.request
from urllib.parse import urlsplit


class LocalTTS:
    def __init__(self, url):
        parsed = urlsplit(url)
        if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
                or parsed.username or parsed.password or parsed.path != '/mcp'
                or parsed.query or parsed.fragment or not parsed.port):
            raise ValueError('Native TTS requires a local HTTP /mcp endpoint with a port')
        self.url = url

    def speak(self, text, action_id):
        request = urllib.request.Request(self.url, data=json.dumps({
            'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
            'params': {'name': 'tts', 'arguments': {
                'action': 'speak', 'text': text, 'action_id': action_id, 'force': False,
            }},
        }).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=5) as response:
            data = response.read(1_000_001)
        if len(data) > 1_000_000:
            raise ValueError('TTS response too large')
        reply = json.loads(data)
        if reply.get('error') or not isinstance(reply.get('result'), dict):
            raise RuntimeError('Native TTS MCP request failed')
        result = reply['result']
        if result.get('isError'):
            raise RuntimeError('Native TTS rejected request')
        content = result.get('content', [])
        if content:
            result = json.loads(content[0]['text'])
        if isinstance(result, dict) and (result.get('state') == 'error' or result.get('error')):
            raise RuntimeError('Native TTS rejected request')
        return {'state': 'submitted', 'action_id': action_id}
