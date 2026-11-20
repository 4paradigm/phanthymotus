"""Explicit administrator-SSH provisioning; never used by list/print."""
import base64
import json
import hashlib
import re
import subprocess
import secrets
import tempfile
from pathlib import Path
from urllib.parse import urlparse

CHUNK_SIZE = 5 * 1024 * 1024  # 5MB per chunk
REMOTE_CHUNK_DIR = "D:/ParadigmVisitorBridge/upload"
REMOTE_BOOTSTRAP = f"{REMOTE_CHUNK_DIR}/bootstrap.ps1"


def sha256_file(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def split_file(src_path: Path, chunk_dir: Path) -> int:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_idx = 0
    with open(src_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            chunk_file = chunk_dir / f"{src_path.name}.{chunk_idx:04d}.chunk"
            with open(chunk_file, "wb") as cf:
                cf.write(chunk)
            chunk_idx += 1
    return chunk_idx


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
    token = str(args.get('token') or '')
    supplied_token = token

    names = ['bridge_core.py', 'http_server.py', 'edge_backend.py', 'run_bridge.py',
             'login_edge.py', 'bridge_watchdog.ps1', 'remote_start_bridge.ps1',
             'config.example.json', 'README.txt']
    offline = package_root / 'offline'
    manifest = json.loads((offline / 'manifest.json').read_text(encoding='utf-8'))
    if 'python-3.12.10-amd64.exe' not in manifest or not any(n.endswith('.whl') for n in manifest):
        raise ValueError('镜像缺少完整离线安装包，请重新构建 perception')

    for name, digest in manifest.items():
        if not re.fullmatch(r'[A-Za-z0-9_.-]+\.(exe|whl)', name):
            raise ValueError('Invalid offline asset name')
        if sha256_file(offline / name) != digest:
            raise ValueError('离线安装包校验失败')

    ssh_base_args = [
        "ssh", "-F", "/dev/null", "-T",
        "-i", str(key), "-p", str(port),
        "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={known}",
    ]
    scp_base_args = [
        "scp", "-F", "/dev/null",
        "-i", str(key), "-P", str(port),
        "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={known}",
    ]

    def run(command, *, sensitive=False, **kwargs):
        try:
            result = subprocess.run(command, capture_output=True, timeout=900)
        except subprocess.TimeoutExpired:
            raise RuntimeError('初始化步骤超时：' + command[0]) from None
        if result.returncode:
            detail = 'SSH 配置读取失败，请检查管理员权限及现有桥接配置' if sensitive else (result.stderr or result.stdout).decode('utf-8', errors='replace')
            for secret in (supplied_token, token):
                if secret:
                    detail = detail.replace(secret, '[REDACTED]')
            detail = detail[-1200:]
            raise RuntimeError('初始化 %s 失败（退出码 %s）：%s' % (command[0], result.returncode, detail))
        return result

    # Read an existing secret only through the already authorized, host-verified
    # admin SSH connection. Never rotate another robot's working bridge token.
    probe = "$ErrorActionPreference='Stop';\n" + (package_root / 'bridge_config.ps1').read_text(encoding='utf-8-sig')
    probe += "\ntry { $info = Get-BridgeTokenInfo; $json = $info | ConvertTo-Json -Compress; Write-Output ('BRIDGE_CONFIG:' + [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($json))); exit 0 } catch { Write-Error 'Bridge configuration preflight failed'; exit 1 }"
    encoded_probe = base64.b64encode(probe.encode('utf-16le')).decode('ascii')
    response = run(ssh_base_args + [f'{user}@{host}',
        'powershell.exe -NoProfile -NonInteractive -EncodedCommand ' + encoded_probe], sensitive=True)
    try:
        markers = [line.partition(b':')[2] for line in response.stdout.splitlines() if line.startswith(b'BRIDGE_CONFIG:')]
        if len(markers) != 1:
            raise ValueError()
        info = json.loads(base64.b64decode(markers[0], validate=True))
        if not isinstance(info, dict) or type(info.get('installed')) is not bool:
            raise ValueError()
        if info['installed']:
            existing_token = info.get('token')
            if not isinstance(existing_token, str) or len(existing_token) < 24 or any(c.isspace() for c in existing_token):
                raise ValueError()
            token = existing_token
        else:
            token = token or secrets.token_urlsafe(32)
    except (ValueError, TypeError, UnicodeError):
        raise RuntimeError('无法读取前台桥接配置，请检查管理员 SSH 权限及强制命令设置') from None
    if len(token) < 24 or any(c.isspace() for c in token):
        raise ValueError('自定义访问令牌需至少 24 位且不能包含空白；留空可自动生成')

    prepare = "$ErrorActionPreference='Stop'; if (!(Test-Path 'D:/')) { throw 'D drive missing' }; New-Item -ItemType Directory -Force -Path 'D:/ParadigmVisitorBridge/upload' | Out-Null; Add-Content -LiteralPath 'D:/ParadigmVisitorBridge/bootstrap.log' -Value 'Upload preparation ready'; Write-Output 'UPLOAD_READY'"
    encoded = base64.b64encode(prepare.encode('utf-16le')).decode('ascii')
    # Encode PowerShell to prevent cmd.exe from interpreting its pipeline.
    prepared = run(
        ssh_base_args + [
            f"{user}@{host}",
            'powershell.exe -NoProfile -NonInteractive -EncodedCommand ' + encoded
        ],
        check=True
    )
    if b'UPLOAD_READY' not in prepared.stdout:
        raise RuntimeError('SSH 未执行上传准备命令；请检查公钥是否仍绑定强制启动命令')

    # Also remove the credential-bearing payload if upload/bootstrap fails.
    with tempfile.TemporaryDirectory(prefix='ql_bridge_chunks_') as directory:
        local_chunk_dir = Path(directory)
        man = {"files": [], "chunk_size": CHUNK_SIZE}
        core_payload = {"token": token, "core_files": {}}
        for n in names:
            core_payload["core_files"][n] = base64.b64encode((package_root / n).read_bytes()).decode()
        core_manifest_path = local_chunk_dir / "core.json"
        core_manifest_path.write_text(json.dumps(core_payload, ensure_ascii=False), encoding='utf-8')
        run(scp_base_args + [str(core_manifest_path), f"{user}@{host}:{REMOTE_CHUNK_DIR}/core.json"])

        for name, digest in manifest.items():
            src_file = offline / name
            if sha256_file(src_file) != digest:
                raise ValueError('离线安装包校验失败')
            chunk_cnt = split_file(src_file, local_chunk_dir)
            for i in range(chunk_cnt):
                chunk_name = f"{name}.{i:04d}.chunk"
                run(scp_base_args + [str(local_chunk_dir / chunk_name), f"{user}@{host}:{REMOTE_CHUNK_DIR}/{chunk_name}"])
            man["files"].append({"name": name, "sha256": digest, "chunk_count": chunk_cnt})

        manifest_path = local_chunk_dir / "manifest.json"
        manifest_path.write_text(json.dumps(man), encoding='utf-8')
        run(scp_base_args + [str(manifest_path), f"{user}@{host}:{REMOTE_CHUNK_DIR}/manifest.json"])
        for script in ('bridge_config.ps1', 'bootstrap.ps1'):
            run(scp_base_args + [str(package_root / script), f'{user}@{host}:{REMOTE_CHUNK_DIR}/{script}'])
        result = run(ssh_base_args + [f"{user}@{host}",
            f"powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File {REMOTE_BOOTSTRAP}"])
    if result.returncode or b'BRIDGE_BOOTSTRAP_OK' not in result.stdout:
        raise RuntimeError('安装未返回成功标记，请检查 D:\\ParadigmVisitorBridge\\bootstrap.log')
    return {'status': 'installed', 'token': token, 'bridge_url': args['bridge_url'],
            'hint': '桥接已安装并启动。请选择打印机并保存；首次使用需登录访客网站。'}
