"""Exercise ACP delivery against Core-style local HTTPS, including failures."""
import importlib.util
import base64
import json
import ssl
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from subprocess import CompletedProcess

from utils import acp
from plugins.ql_printer import QLPrinterPlugin, SSHBridgeClient

# Test-only certificate and key for a loopback server; no deployment identity.
CERT_PEM = '-----BEGIN CERTIFICATE-----\nMIICvTCCAaWgAwIBAgIBATANBgkqhkiG9w0BAQsFADAUMRIwEAYDVQQDDAlsb2Nh\nbGhvc3QwIBcNMjAwMTAxMDAwMDAwWhgPMjA5OTAxMDEwMDAwMDBaMBQxEjAQBgNV\nBAMMCWxvY2FsaG9zdDCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAMCS\nFIr8cOjk/Qb6GvvfAdXXOvuq5jzff4gtz/nHrQHuXETDtJ9tiLLO7NLD/SoyCo5A\nZbMj/O8XSgLWKsV8h2o+sIUWC0MH5ojnCtTxefmqIbIPLhuN4kx0roKlk11rapyN\noV8rAuQLzQ4LdbQ3oxPA9PF08XRCe9L7LdDGfhjTHf4IvX+QbXP8Oh6Bn8uVxJIk\nW9/pW6pWqZHwcmUzgUYq0iElz5zp3S0vIBz9g3Qj2qi4aW7EqLsS/0gDsN2Vv4Sw\neJPKNyK0qs+5kvrWZYCtU7kfRF0+oAgUTF89gqApRcnl/2a3I+XaCoh+YfTICccI\nGVrt23JaLkbAcfzt2ZMCAwEAAaMYMBYwFAYDVR0RBA0wC4IJbG9jYWxob3N0MA0G\nCSqGSIb3DQEBCwUAA4IBAQAYaDvHZ5tpGyHQ1b2nFj46AJ/X3ACCNigj5076OfZX\n5/AD2gyr5J1aXbq1a7g3OMOsD2Rpp0HGwBYURBljfgwJfIsOh319bzu9Rju/yOc2\n67Lsdhyoi2GIatZTclk8MezINLwH8ErDeNghKNtk2v6ak90ydLuzLYBB6dCafZKA\nGNWTMhN01AOsXlSO08HVn1I85RA5QhcTOZ/MfEE+gC9XjTzmBgHT7unpZwaEBDqD\n4WYYKNq77tG2bizCZVXribSXSmdTkrch9FpyRqseThj2wivZDA7O0YMXLSBvuDIo\naTUAfT/0Fho7hsCv8kDlaGzPCGcZ+vs/hxN/ccyLLepe\n-----END CERTIFICATE-----\n'
KEY_PEM = '-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDAkhSK/HDo5P0G\n+hr73wHV1zr7quY833+ILc/5x60B7lxEw7SfbYiyzuzSw/0qMgqOQGWzI/zvF0oC\n1irFfIdqPrCFFgtDB+aI5wrU8Xn5qiGyDy4bjeJMdK6CpZNda2qcjaFfKwLkC80O\nC3W0N6MTwPTxdPF0QnvS+y3Qxn4Y0x3+CL1/kG1z/DoegZ/LlcSSJFvf6VuqVqmR\n8HJlM4FGKtIhJc+c6d0tLyAc/YN0I9qouGluxKi7Ev9IA7Ddlb+EsHiTyjcitKrP\nuZL61mWArVO5H0RdPqAIFExfPYKgKUXJ5f9mtyPl2gqIfmH0yAnHCBla7dtyWi5G\nwHH87dmTAgMBAAECggEAEzF8Iq4ugsfwADYFYn1DrZCaLGNIN2D/UQ9f30FrmPms\n8J2g7xL3wPqm2Ar4vK/oJW19tXEs0Je9pLK76JsU1B4TcfBUhwXOkG6Wwcqfoaa3\nXQvuSIjHuV8X1aWzHmiKY7gqEVY0H46djXzO6ntxIZLl1vsU5cWKmei3FEv0XXzh\nAxPlp6tVQ26aKTM5ECUWHcosWyBQiOEGYXMhI4aZBjYF2Vae6WKEi5R2Hzn9EbMW\nDYMt2pBIEf3aZXUkRRBv5XK5+jINusS6aYd2hIaDXHsDvUoXTnBkOiVc20LZxAO2\nBxOh9BOcK+N/aVHL1OzE03mlC3UhxBiFpwdZX4CbAQKBgQDl1dP8cXjIlYSqmHle\nBLVeDZPI4aZLcoiaoJEEyrE3nq253DwREnYGYxlY5YfDM5eEye/S7FIefVCxZasm\n2Jc/yY+ehiCBH+mrrlYau6sUrazMG/1RNQoQhAXBtvIEm6IjJwDm3kLkY18TOVMv\nZrZg1+w9eMEjcknCj0D/Ur2QcwKBgQDWfjxADZVVl+1jAnNli2C0KVCvoW4H3NGx\nabXcpaZSjunjdr8YDex5RjRwEQw8RE3ztfnWZu34C6zP3upDs5rB+CofnyQmER6B\nyQ4v0i0LaoCZeSlLZVstb+2o45Bt6ko6021xRjy+MmKx1F+DLuNYWKw4fUnUU7AY\n9ipOz2vqYQKBgE9LtebL84Mo6uu8zWrdrwFzJ8DP8YLoUi8fJMmFcR0idwe/701o\nlCXh/PXJw0xauXUMAt0/wZlZXQTHcgRmB9DvJsjnt8E1FHayIGz6wKzSuuJ06bIh\n3YYTNrU9/EQQuiDsrQf6ORP5IO1OAuBAzL2e4ZfX2LL1FHO23xL4JgT3AoGAUc7r\n4/JKVs8Pd4fjw4MZax2m7JXBhbYvNF+BQZrciPiBNISB9ga5frfMIEIgTIwwvxdV\nJvUzJjwtuUroRtIl4H1qc40bGHnHDPFGO+8JxyTjz6VqnpsV34qlNpk06PdHQJCL\nWXzUhiZphjHeDAbCJ1wKs7D1i8ovjBW3pT9vhYECgYA3fVErvb0OwKyCPGLLzwtr\nhda0jOz+NK33H78loj8WaBw2z5KWp7f6rrYL81oxXICCbynnpHEiuQJg3g7WH9/H\n+MkyqVE8h4BWc6XTw+hEsLKn1E2m+5o+mXc4Iauy53AwdIkr/+63trR8yI9Bmb8D\n9SBOI+533HZlzbOLOlDxyw==\n-----END PRIVATE KEY-----\n'


@contextmanager
def callback_server(tmp_path, use_https):
    received = []
    arrived = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
            arrived.set()

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    if use_https:
        cert = tmp_path / 'callback-cert.pem'
        key = tmp_path / 'callback-key.pem'
        cert.write_text(CERT_PEM, encoding='ascii')
        key.write_text(KEY_PEM, encoding='ascii')
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(cert), str(key))
        server.socket = context.wrap_socket(server.socket, server_side=True)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    scheme = 'https' if use_https else 'http'
    try:
        yield f'{scheme}://127.0.0.1:{server.server_port}', received, arrived
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_failed_print_is_delivered_through_real_self_signed_https(tmp_path, monkeypatch, caplog):
    result = {'status': 'error', 'code': 'query_button_missing',
              'error': '未检测到访客查询功能，请打开正确的访客列表网址'}
    with callback_server(tmp_path, use_https=True) as (url, received, arrived):
        monkeypatch.setattr(acp, 'AGENT_CORE_URL', url)
        acp._post('visitor-print-failed', 'failed', result)
        assert arrived.wait(2), 'failure callback never reached local HTTPS Core'
    assert received == [('/api/acp/complete', {'action_id': 'visitor-print-failed',
                                             'status': 'failed', 'result': result})]
    assert 'did not reach agent-core' not in caplog.text


def test_explicit_http_configuration_still_delivers_completion(tmp_path, monkeypatch):
    with callback_server(tmp_path, use_https=False) as (url, received, arrived):
        monkeypatch.setattr(acp, 'AGENT_CORE_URL', url)
        acp._post('visitor-print-ok', 'completed', {'status': 'printed'})
        assert arrived.wait(2)
    assert received[0][1]['status'] == 'completed'
    assert received[0][1]['result']['status'] == 'printed'


def test_public_complete_delivers_failure_asynchronously(tmp_path, monkeypatch):
    with callback_server(tmp_path, use_https=True) as (url, received, arrived):
        monkeypatch.setattr(acp, 'AGENT_CORE_URL', url)
        acp.complete('visitor-print-async', 'failed', {'code': 'printer_not_ready', 'error': '打印机缺纸'})
        assert arrived.wait(2)
    assert received[0][1]['action_id'] == 'visitor-print-async'
    assert received[0][1]['result']['code'] == 'printer_not_ready'


def test_non_loopback_https_keeps_certificate_verification():
    context = acp._ssl_context('https://robot.example.test:15678')
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED


def test_default_core_url_matches_https_server(monkeypatch):
    monkeypatch.delenv('AGENT_CORE_URL', raising=False)
    spec = importlib.util.spec_from_file_location('acp_default_url_test', Path(acp.__file__))
    isolated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolated)
    assert isolated.AGENT_CORE_URL == 'https://localhost:15678'


def test_accepted_print_later_delivers_windows_failure_and_updated_message(tmp_path, monkeypatch):
    old_failure = {'ok': False, 'code': 'query_button_missing',
                   'error': '页面加载后未观察到查询请求，也未找到可见的查询或搜索按钮，请确认选择了访客列表页'}
    payload = base64.b64encode(json.dumps(old_failure, ensure_ascii=False).encode()).decode()
    monkeypatch.setattr(SSHBridgeClient, '_ssh', lambda *args, **kwargs: CompletedProcess(
        [], 0, ('VISITOR_RESULT:' + payload + '\n').encode(), b''))
    card = QLPrinterPlugin({'bridge_url': '192.168.1.2', 'remote_start_user': 'operator',
                            'remote_start_password': 'secret', 'printer_name': 'Brother',
                            'visitor_page_url': 'https://visitor.test/list'})
    with callback_server(tmp_path, use_https=True) as (url, received, arrived):
        monkeypatch.setattr(acp, 'AGENT_CORE_URL', url)
        accepted = card.dispatch('print', {'name': '王磊'})
        assert accepted['status'] == 'accepted'
        assert arrived.wait(3), 'accepted print failure was never reported'
    completion = received[0][1]
    assert completion['action_id'] == accepted['action_id']
    assert completion['status'] == 'failed'
    assert completion['result']['code'] == 'query_button_missing'
    assert completion['result']['error'] == '当前页面未找到访客查询按钮，请确认已打开正确的访客列表网址。'

