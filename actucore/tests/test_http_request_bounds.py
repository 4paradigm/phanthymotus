"""Actual MCP HTTP handler on loopback; no ROS, robot or card startup."""
import ast
from http.client import HTTPResponse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace

import pytest


@pytest.fixture
def http():
    path = Path(__file__).parents[1] / 'main.py'
    node = next(n for n in ast.parse(path.read_text()).body if getattr(n, 'name', '') == 'make_handler')
    ns = {'BaseHTTPRequestHandler': BaseHTTPRequestHandler, 'json': json,
          'log': logging.getLogger('http-bounds'), '_brief': str,
          '_bundle': SimpleNamespace(server_name='isolated-test')}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    handler = ns['make_handler']()
    assert handler.max_request_bytes == 16 * 1024 * 1024
    handler.max_request_bytes = 512
    handler.request_body_timeout_s = .2
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    worker.start()
    yield server.server_address
    server.shutdown(); server.server_close(); worker.join(1)
    assert not worker.is_alive()


def send(address, headers, body=b'', eof=False):
    with socket.create_connection(address, timeout=2) as wire:
        wire.sendall(b'POST /mcp HTTP/1.1\r\nHost: localhost\r\n' + headers + b'\r\n' + body)
        if eof:
            wire.shutdown(socket.SHUT_WR)
        reply = HTTPResponse(wire); reply.begin()
        return reply.status, json.loads(reply.read())


@pytest.mark.parametrize('header,status', [
    (b'', 411), (b'Content-Length: -1\r\n', 400),
    (b'Content-Length: abc\r\n', 400), (b'Content-Length: +1\r\n', 400),
    (b'Content-Length: 1.0\r\n', 400), (b'Content-Length: \r\n', 400),
    (b'Content-Length: 0\r\nContent-Length: 0\r\n', 400),
    (b'Transfer-Encoding: chunked\r\nContent-Length: 0\r\n', 400),
    (b'Content-Length: 513\r\n', 413),
    (b'Content-Length: ' + b'9' * 5000 + b'\r\n', 413),
])
def test_bad_framing_is_rejected_before_body_read(http, header, status):
    # Deliberately send no declared body: response must not wait for it.
    started = time.monotonic()
    result, payload = send(http, header)
    assert result == status and payload['error']
    assert time.monotonic() - started < 1
    assert initialize(http)[0] == 200  # Same service still accepts new requests.


def initialize(http, size=None):
    body = json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize'}).encode()
    if size:
        body += b' ' * (size - len(body))
    return send(http, f'Content-Length: {len(body)}\r\n'.encode(), body)


def test_request_at_bound_keeps_normal_mcp_response(http):
    status, body = initialize(http, 512)
    assert status == 200 and body['result']['serverInfo']['name'] == 'isolated-test'


@pytest.mark.parametrize('body,code', [(b'', -32700), (b'{', -32700), (b'[]', -32600)])
def test_invalid_json_preserves_rpc_error_contract(http, body, code):
    status, reply = send(http, f'Content-Length: {len(body)}\r\n'.encode(), body)
    assert status == 200 and reply['error']['code'] == code


def test_incomplete_body_has_explicit_error(http):
    assert send(http, b'Content-Length: 20\r\n', b'{', eof=True)[0] == 400


def test_body_timeout_is_absolute_even_with_trickle(http):
    with socket.create_connection(http, timeout=2) as wire:
        wire.sendall(b'POST /mcp HTTP/1.1\r\nHost: localhost\r\nContent-Length: 512\r\n\r\n{')
        done = threading.Event()
        def trickle():
            while not done.wait(.05):
                try:
                    wire.sendall(b' ')
                except OSError:
                    return
        sender = threading.Thread(target=trickle, daemon=True); sender.start()
        started = time.monotonic()
        try:
            reply = HTTPResponse(wire); reply.begin()
            assert reply.status == 408
            assert json.loads(reply.read())['error'] == 'Request body timed out'
            assert time.monotonic() - started < 1
        finally:
            done.set(); sender.join(1)
    assert initialize(http)[0] == 200
