"""Provision the on-demand Windows automation runtime over SSH."""
import base64
import json
import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

CHUNK_SIZE = 5 * 1024 * 1024  # 5MB per chunk
REMOTE_CHUNK_DIR = "D:/ParadigmVisitorAutomation/upload"
REMOTE_BOOTSTRAP = f"{REMOTE_CHUNK_DIR}/bootstrap_ssh.ps1"


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
    raw_url = str(args.get('bridge_url') or '')
    url = urlparse(raw_url)
    host = url.hostname or ''
    user = str(args.get('remote_start_user') or '4paradigm')
    port = int(args.get('remote_start_port', 22))
    if not 1 <= port <= 65535:
        raise ValueError('SSH 端口无效')
    password = str(args.get('remote_start_password') or '')
    if not password:
        raise ValueError('请填写 Windows SSH 密码')
    if url.scheme != 'ssh' or url.port not in (None, 22) or url.path not in ('', '/'):
        raise ValueError('前台电脑地址无效')
    if not re.fullmatch(r'[A-Za-z0-9_.:-]+', host) or not re.fullmatch(r'[A-Za-z0-9_.-]+', user):
        raise ValueError('电脑地址或 SSH 用户无效')
    config_arg = '/dev/null'
    target = f'{user}@{host}'

    names = ['bridge_core.py', 'edge_backend.py', 'ssh_worker.py', 'login_edge.py',
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

    known = '/tmp/visitor-printer-known-hosts'
    ssh_base_args = [
        "ssh", "-F", config_arg, "-T",
        "-p", str(port), "-o", "PubkeyAuthentication=no",
        "-o", "PreferredAuthentications=password,keyboard-interactive",
        "-o", "NumberOfPasswordPrompts=1", "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={known}",
    ]
    scp_base_args = [
        "scp", "-F", config_arg,
        "-P", str(port), "-o", "PubkeyAuthentication=no",
        "-o", "PreferredAuthentications=password,keyboard-interactive",
        "-o", "NumberOfPasswordPrompts=1", "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={known}",
    ]

    askpass = tempfile.NamedTemporaryFile('w', prefix='visitor-deploy-askpass-', suffix='.sh', delete=False)
    askpass.write('#!/bin/sh\nprintf "%s\\n" "$VISITOR_SSH_PASSWORD"\n')
    askpass.close()
    os.chmod(askpass.name, 0o700)
    env = os.environ.copy()
    env.update({'DISPLAY': 'visitor-ssh', 'SSH_ASKPASS': askpass.name,
                'SSH_ASKPASS_REQUIRE': 'force', 'VISITOR_SSH_PASSWORD': password})

    def run(command, **kwargs):
        try:
            result = subprocess.run(command, capture_output=True, timeout=900,
                                    stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        except subprocess.TimeoutExpired:
            raise RuntimeError('初始化步骤超时：' + command[0]) from None
        if result.returncode:
            raw = result.stderr or result.stdout
            text = raw.decode('utf-8', errors='replace')
            if text and text.count('\ufffd') * 4 > len(text):
                # zh-CN Windows 输出 GBK 编码错误，UTF-8 解码会得到乱码。
                text = raw.decode('gbk', errors='replace')
            detail = ' '.join(text.replace('_x000D__x000A_', ' ').split())[-1200:]
            raise RuntimeError('初始化 %s 失败（退出码 %s）：%s' % (command[0], result.returncode, detail))
        return result

    try:
        prepare = "$ErrorActionPreference='Stop'; if (!(Test-Path 'D:/')) { throw 'D drive missing' }; New-Item -ItemType Directory -Force -Path 'D:/ParadigmVisitorAutomation/upload' | Out-Null; Write-Output 'UPLOAD_READY'"
        encoded = base64.b64encode(prepare.encode('utf-16le')).decode('ascii')
        prepared = run(ssh_base_args + [target, 'powershell.exe -NoProfile -NonInteractive -EncodedCommand ' + encoded])
        if b'UPLOAD_READY' not in prepared.stdout:
            raise RuntimeError('SSH 未执行上传准备命令，请确认该 Windows 账户具有管理员权限')

        with tempfile.TemporaryDirectory(prefix='visitor_ssh_chunks_') as directory:
            local_chunk_dir = Path(directory)
            man = {"files": [], "chunk_size": CHUNK_SIZE}
            core_payload = {"core_files": {}}
            for n in names:
                core_payload["core_files"][n] = base64.b64encode((package_root / n).read_bytes()).decode()
            core_manifest_path = local_chunk_dir / "core.json"
            core_manifest_path.write_text(json.dumps(core_payload, ensure_ascii=False), encoding='utf-8')
            run(scp_base_args + [str(core_manifest_path), f"{target}:{REMOTE_CHUNK_DIR}/core.json"])
            for name, digest in manifest.items():
                src_file = offline / name
                chunk_cnt = split_file(src_file, local_chunk_dir)
                for i in range(chunk_cnt):
                    chunk_name = f"{name}.{i:04d}.chunk"
                    run(scp_base_args + [str(local_chunk_dir / chunk_name), f"{target}:{REMOTE_CHUNK_DIR}/{chunk_name}"])
                man["files"].append({"name": name, "sha256": digest, "chunk_count": chunk_cnt})
            manifest_path = local_chunk_dir / "manifest.json"
            manifest_path.write_text(json.dumps(man), encoding='utf-8')
            run(scp_base_args + [str(manifest_path), f"{target}:{REMOTE_CHUNK_DIR}/manifest.json"])
            run(scp_base_args + [str(package_root / 'bootstrap_ssh.ps1'), f'{target}:{REMOTE_CHUNK_DIR}/bootstrap_ssh.ps1'])
            result = run(ssh_base_args + [target,
                f"powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File {REMOTE_BOOTSTRAP}"])
    finally:
        try:
            os.unlink(askpass.name)
        except OSError:
            pass
    if result.returncode or b'SSH_AUTOMATION_OK' not in result.stdout:
        raise RuntimeError('安装未返回成功标记，请检查 D:\\ParadigmVisitorAutomation\\install.log')
    return {'status': 'installed', 'bridge_url': raw_url,
            'hint': 'SSH 自动化已准备完成。请在前台打开的 Edge 中登录访客网站，登录完成后选择打印机并保存。'}
