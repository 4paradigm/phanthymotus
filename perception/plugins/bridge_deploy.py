"""Explicit administrator-SSH provisioning; never used by list/print."""
import base64
import json
import hashlib
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse


def deploy(args, package_root=Path('/deploy/windows_bridge')):
    url = urlparse(args['bridge_url'])
    if url.scheme != 'http' or url.port != 8765 or url.path not in ('', '/'):
        raise ValueError('初始化地址必须为 http://电脑IP:8765')
    host = url.hostname or ''
    user = args.get('remote_start_user', '')
    if not re.fullmatch(r'[A-Za-z0-9_.:-]+', host) or not re.fullmatch(r'[A-Za-z0-9_.-]+', user):
        raise ValueError('电脑地址或 SSH 用户无效')
    port = int(args.get('remote_start_port', 22))
    if not 1 <= port <= 65535:
        raise ValueError('SSH 端口无效')
    key = Path(args.get('remote_start_key_path', ''))
    known = Path('/run/ql-printer-auth/known_hosts')
    if not key.is_file() or not known.is_file():
        raise ValueError('SSH 私钥或已核对的 known_hosts 缺失')
    token = str(args.get('token', ''))
    if len(token) < 24:
        raise ValueError('初始化需要至少 24 位访问令牌；升级请填写现有令牌')
    names = ['bridge_core.py', 'http_server.py', 'edge_backend.py', 'run_bridge.py',
             'login_edge.py', 'bridge_watchdog.ps1', 'remote_start_bridge.ps1', 'config.example.json']
    data = {'token': token, 'files': {n: base64.b64encode((package_root / n).read_bytes()).decode() for n in names}}
    offline = package_root / 'offline'
    manifest = json.loads((offline / 'manifest.json').read_text(encoding='utf-8'))
    if 'python-3.12.10-amd64.exe' not in manifest or not any(n.endswith('.whl') for n in manifest):
        raise ValueError('镜像缺少完整离线安装包，请重新构建 perception')
    data['offline'] = {}
    for name, digest in manifest.items():
        if not re.fullmatch(r'[A-Za-z0-9_.-]+\.(exe|whl)', name):
            raise ValueError('Invalid offline asset name')
        content = (offline / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError('离线安装包校验失败')
        data['offline'][name] = {'sha256': digest, 'content': base64.b64encode(content).decode('ascii')}
    script = (package_root / 'bootstrap.ps1').read_text(encoding='utf-8-sig')
    payload = json.dumps({'script': script, 'data': data})
    loader = "$ErrorActionPreference='Stop'; $p=[Console]::In.ReadToEnd() | ConvertFrom-Json; & ([ScriptBlock]::Create($p.script)) $p.data"
    encoded = base64.b64encode(loader.encode('utf-16le')).decode()
    command = ['ssh', '-F', '/dev/null', '-T', '-i', str(key), '-p', str(port),
               '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
               '-o', 'StrictHostKeyChecking=yes', '-o', f'UserKnownHostsFile={known}',
               f'{user}@{host}', 'powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand ' + encoded]
    # Windows may emit diagnostics in its local code page. Keep stdout/stderr
    # as bytes so decoding cannot mask the bootstrap result.
    result = subprocess.run(command, input=payload.encode('utf-8'), capture_output=True, timeout=900)
    if result.returncode or b'BRIDGE_BOOTSTRAP_OK' not in result.stdout:
        # Never return raw installer output: it may contain configuration values.
        raise RuntimeError('远程初始化失败，请检查电脑 ProgramData\\ParadigmVisitorBridge\\bootstrap.log；确认使用管理员 SSH 密钥而非仅启动密钥')
    return {'status': 'installed', 'hint': '桥接已安装并启动。请保存配置、刷新打印机；首次使用需登录访客网站。'}
