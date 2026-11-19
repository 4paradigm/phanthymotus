"""Explicit administrator-SSH provisioning; never used by list/print."""
import base64
import json
import hashlib
import re
import subprocess
import shutil
from pathlib import Path
from urllib.parse import urlparse

CHUNK_SIZE = 5 * 1024 * 1024  # 5MB per chunk
REMOTE_CHUNK_DIR = "C:\\ParadigmChunks"
REMOTE_BOOTSTRAP = f"{REMOTE_CHUNK_DIR}\\bootstrap.ps1"


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
    token = str(args.get('token', ''))
    if len(token) < 24:
        raise ValueError('初始化需要至少 24 位访问令牌；升级请填写现有令牌')

    names = ['bridge_core.py', 'http_server.py', 'edge_backend.py', 'run_bridge.py',
             'login_edge.py', 'bridge_watchdog.ps1', 'remote_start_bridge.ps1', 'config.example.json']
    offline = package_root / 'offline'
    manifest = json.loads((offline / 'manifest.json').read_text(encoding='utf-8'))
    if 'python-3.12.10-amd64.exe' not in manifest or not any(n.endswith('.whl') for n in manifest):
        raise ValueError('镜像缺少完整离线安装包，请重新构建 perception')

    ssh_base_args = [
        "ssh", "-F", "/dev/null", "-T",
        "-i", str(key), "-p", str(port),
        "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={known}",
    ]
    scp_base_args = [
        "scp", "-F", "/dev/null",
        "-i", str(key), "-p", str(port),
        "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={known}",
    ]

    # 远程创建分片缓存目录
    subprocess.run(
        ssh_base_args + [
            f"{user}@{host}",
            f"powershell -NoProfile -Command New-Item -ItemType Directory -Path {REMOTE_CHUNK_DIR} -Force | Out-Null"
        ],
        check=True
    )

    local_chunk_dir = Path("/tmp/ql_bridge_chunks")
    local_chunk_dir.mkdir(parents=True, exist_ok=True)
    man = {"files": [], "chunk_size": CHUNK_SIZE}

    # 上传小的核心脚本文件
    core_payload = {"token": token, "core_files": {}}
    for n in names:
        fpath = package_root / n
        core_payload["core_files"][n] = base64.b64encode(fpath.read_bytes()).decode()
    core_manifest_path = local_chunk_dir / "core.json"
    with open(core_manifest_path, "w", encoding="utf-8") as f:
        json.dump(core_payload, f, ensure_ascii=False)
    subprocess.run(scp_base_args + [str(core_manifest_path), f"{user}@{host}:{REMOTE_CHUNK_DIR}\\core.json"], check=True)

    # 分片上传离线包
    for name, digest in manifest.items():
        if not re.fullmatch(r'[A-Za-z0-9_.-]+\.(exe|whl)', name):
            raise ValueError('Invalid offline asset name')
        src_file = offline / name
        if sha256_file(src_file) != digest:
            raise ValueError('离线安装包校验失败')
        chunk_cnt = split_file(src_file, local_chunk_dir)
        for i in range(chunk_cnt):
            chunk_name = f"{name}.{i:04d}.chunk"
            subprocess.run(scp_base_args + [str(local_chunk_dir / chunk_name), f"{user}@{host}:{REMOTE_CHUNK_DIR}\\{chunk_name}"], check=True)
        man["files"].append({"name": name, "sha256": digest, "chunk_count": chunk_cnt})

    # 上传分片清单
    manifest_path = local_chunk_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False)
    subprocess.run(scp_base_args + [str(manifest_path), f"{user}@{host}:{REMOTE_CHUNK_DIR}\\manifest.json"], check=True)

    # 上传bootstrap.ps1
    bootstrap_src = package_root / "bootstrap.ps1"
    subprocess.run(scp_base_args + [str(bootstrap_src), f"{user}@{host}:{REMOTE_BOOTSTRAP}"], check=True)

    # 远程执行bootstrap
    result = subprocess.run(
        ssh_base_args + [
            f"{user}@{host}",
            f"powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File {REMOTE_BOOTSTRAP}"
        ],
        capture_output=True, timeout=900
    )

    shutil.rmtree(local_chunk_dir, ignore_errors=True)
    if result.returncode or b'BRIDGE_BOOTSTRAP_OK' not in result.stdout:
        raise RuntimeError('远程初始化失败，请检查电脑 ProgramData\\ParadigmVisitorBridge\\bootstrap.log；确认使用管理员 SSH 密钥而非仅启动密钥')
    return {'status': 'installed', 'hint': '桥接已安装并启动。请保存配置、刷新打印机；首次使用需登录访客网站。'}
