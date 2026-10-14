#!/usr/bin/env python3
"""PiPER-only preparation for run-pr-image.sh; never sends hardware commands."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import socket
import ssl
import subprocess
import sys
import tempfile
import time
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener


DATA = Path('/opt/phanthy-motus/data/piper')
CONFIG = DATA / 'config.dual.yaml'
CORE_ENV = Path('/opt/phanthy-motus/.env')
CORE = 'https://127.0.0.1:15678'
TARGETS = {'embodied-agilex-piper': 15741, 'embodied-agilex-piper-2': 15742}
SERVICE = 'agilex-piper'
PRIMARY = 'embodied-agilex-piper'
SECONDARY = 'embodied-agilex-piper-2'
CARDS = {'piper_teleop', 'piper_motion'}


class CommandError(RuntimeError):
    def __init__(self, args, result):
        super().__init__(f'{args[0]} {args[1]} failed (exit {result.returncode})')
        self.diagnostics = result.stdout + result.stderr


def run(*args, data=None):
    result = subprocess.run(args, input=data, text=True, capture_output=True, timeout=120)
    if result.returncode:
        # Do not echo command output: compose/inspect can include credentials.
        raise CommandError(args, result)
    return result.stdout + result.stderr if args[:2] == ('docker', 'logs') else result.stdout


def docker(*args, data=None):
    return run('docker', *args, data=data)


def private_write(path, text):
    """Replace our generated files atomically, without a world-readable interval."""
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def yaml_config(image, saved):
    # Use the image's existing PyYAML, in isolation; no host pip installation.
    program = """import json, sys, yaml
from pathlib import Path
template = yaml.safe_load(Path('/work/agilex/piper/config.dual.example.yaml').read_text())
saved = sys.stdin.read()
print(json.dumps(yaml.safe_load(saved) if saved else template))
"""
    try:
        return json.loads(docker(
            'run', '--rm', '-i', '--network', 'none', '--read-only', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '-e', 'PYTHONDONTWRITEBYTECODE=1',
            '--entrypoint', 'python3', image, '-c', program, data=saved or ''))
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError('无法读取双臂配置。请使用包含 config.dual.example.yaml 的新版 PiPER 镜像。') from exc


def can_interfaces():
    return sorted(p.name for p in Path('/sys/class/net').iterdir()
                  if (p / 'type').read_text().strip() == '280')


def bindings(config, available):
    if not isinstance(config, dict) or set(config.get('arms', {})) != {'left', 'right'}:
        raise ValueError('双臂配置必须包含 arms.left 和 arms.right。')
    selected = []
    for side in ('left', 'right'):
        value = config['arms'][side].get('can_interface')
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,15}', value):
            raise ValueError(f'{side} CAN 接口名无效。')
        if value not in available:
            raise ValueError(f'CAN 接口 {value} 不存在；保留原绑定，请核对接线。')
        selected.append(value)
    if selected[0] == selected[1]:
        raise ValueError('左右臂不能使用同一个 CAN 接口。')
    if config.get('mcp_port') != 15741 or config.get('data_dir') != str(DATA):
        raise ValueError('双臂部署要求 MCP 15741 和现有 PiPER 数据目录。')
    return selected


def containers():
    names = docker('ps', '-a', '--format', '{{.Names}}').splitlines()
    found = {}
    for name in TARGETS:
        if name not in names:
            continue
        item, = json.loads(docker('inspect', name))
        if item['Name'] != '/' + name or item['State'].get('Paused') or item['State'].get('Restarting'):
            raise RuntimeError(f'{name} 状态不允许切换。')
        found[name] = item
    return found


def request(url, token='', payload=None):
    opener = build_opener(ProxyHandler({}), HTTPSHandler(context=ssl._create_unverified_context()))
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    req = Request(url, headers=headers, data=None if payload is None else json.dumps(payload).encode())
    with opener.open(req, timeout=5) as response:
        return json.load(response)


def rpc(port, method, params=None):
    result = request(f'http://127.0.0.1:{port}/mcp', payload={
        'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params or {}})
    if 'error' in result:
        raise RuntimeError(f'MCP {port} 查询失败。')
    return result['result']


def info(port, name):
    result = rpc(port, 'tools/call', {'name': name, 'arguments': {'action': 'info'}})
    if result.get('isError'):
        raise RuntimeError(f'{name} 状态查询失败。')
    return json.loads(result['content'][0]['text'])


def driver_idle(port):
    names = {tool['name'] for tool in rpc(port, 'tools/list')['tools']}
    name = 'arm_control' if 'arm_control' in names else 'piper_status'
    if name not in names:
        raise RuntimeError(f'MCP {port} 不是受支持的 PiPER Driver。')
    state = info(port, name)
    arms = state.get('arms')
    if arms is not None and set(arms) != {'left', 'right'}:
        raise RuntimeError('双臂状态不完整。')
    for arm in arms.values() if arms is not None else [state]:
        if arm.get('busy') is not False or arm.get('motion_enabled') is not False or arm.get('active_card'):
            raise RuntimeError(f'MCP {port} 忙碌或运动许可未关闭；未停止任何服务。')
    if 'piper_camera_live' in names:
        camera = info(port, 'piper_camera_live')
        if camera.get('state') not in {'idle', 'disabled'} or camera.get('publishing') is not False:
            raise RuntimeError('实时相机仍在使用；请先停止采集。')
    if 'piper_teleop' in names:
        teleop = info(port, 'piper_teleop')
        if teleop.get('state') != 'idle' or teleop.get('output') is not False or teleop.get('recording'):
            raise RuntimeError('双臂遥操或录制尚未停止。')
    return names


def require_idle(token, found):
    if request(CORE + '/api/config/project-running', token).get('running') is not False:
        raise RuntimeError('请先在画布停止项目。')
    editor = request(CORE + '/api/canvas/edit-status', token)
    if editor.get('code') != 200 or 'editor' not in editor or editor['editor'] is not None:
        raise RuntimeError('画布有人编辑或占用状态未知。')
    for name, port in TARGETS.items():
        if name in found and found[name]['State']['Running']:
            driver_idle(port)
        else:
            with socket.socket() as sock:
                sock.settimeout(1)
                if sock.connect_ex(('127.0.0.1', port)) == 0:
                    raise RuntimeError(f'端口 {port} 被其他进程占用。')


def prepare_compose(image, fragment, project):
    wrapped = 'services:\n' + ''.join('  ' + line + '\n' for line in fragment.read_text().splitlines())
    wrapped = re.sub(r'(?m)^(\s*image:)\s*.*$', lambda m: m[1] + ' ' + image, wrapped)
    compose = json.loads(docker('compose', '-p', project, '-f', '-', 'config', '--format', 'json', data=wrapped))
    if set(compose['services']) != {SERVICE}:
        raise ValueError('PiPER 镜像必须只声明 agilex-piper 服务。')
    service = compose['services'][SERVICE]
    if service.get('container_name') != PRIMARY or service.get('network_mode') != 'host':
        raise ValueError('PiPER 容器名称或网络模式不符合部署约定。')
    if not any(v.get('type') == 'bind' and v.get('source') == str(DATA) and v.get('target') == str(DATA)
               for v in service.get('volumes', [])):
        raise ValueError('PiPER 服务缺少持久数据目录挂载。')
    service.setdefault('environment', {}).update({
        'CONFIG_PATH': str(CONFIG), 'PIPER_MOTION_ENABLED': '0', 'PIPER_CAMERA_ENABLED': '0'})
    return compose


def filesystem_access():
    parent = DATA
    while not parent.exists():
        parent = parent.parent
    if os.access(parent, os.W_OK) and (not CORE_ENV.exists() or os.access(CORE_ENV, os.R_OK)):
        return []
    # Authenticate before stopping anything. Never run the whole deploy as root.
    result = subprocess.run(['sudo', '-v'] if sys.stdin.isatty() else ['sudo', '-n', 'true'])
    if result.returncode:
        raise RuntimeError('无法写入 PiPER 数据目录；未停止服务。')
    return ['sudo', '-n']


def read_file(path, privilege):
    try:
        return path.read_text()
    except PermissionError:
        return run(*privilege, 'cat', str(path)) if privilege else run('sudo', '-n', 'cat', str(path))


def save_file(path, text, staging, privilege):
    temporary = staging / 'file'
    private_write(temporary, text)
    run(*privilege, 'install', '-m', '600', str(temporary), str(path))


def backup(found, run_dir, staging, privilege, saved):
    if (DATA / 'deploy-backups').is_symlink():
        raise ValueError('部署备份目录不能是符号链接。')
    destination = DATA / 'deploy-backups' / (time.strftime('%Y%m%d-%H%M%S') + '-' + str(time.time_ns()))
    run(*privilege, 'install', '-d', '-m', '700', str(destination))
    metadata = {}
    paths = {run_dir / 'docker-compose.yml', run_dir / 'state'}
    for name, item in found.items():
        labels = item['Config'].get('Labels') or {}
        metadata[name] = {
            'id': item['Id'], 'image_id': item['Image'], 'image': item['Config']['Image'],
            'running': item['State']['Running'], 'restart_policy': item['HostConfig']['RestartPolicy'],
            'mounts': item['Mounts'], 'compose_labels': {k: v for k, v in labels.items() if k.startswith('com.docker.compose.')},
            'environment': [e for e in item['Config'].get('Env', [])
                            if e.partition('=')[0] in {'CONFIG_PATH', 'PIPER_MOTION_ENABLED', 'PIPER_CAMERA_ENABLED'}]}
        paths.update(Path(p) for p in labels.get('com.docker.compose.project.config_files', '').split(',') if p)
    files = {}
    for index, path in enumerate(sorted(paths)):
        if path.is_file():
            name = f'previous-{index}-{path.name}'
            save_file(destination / name, read_file(path, privilege), staging, privilege)
            files[str(path)] = name
    if saved is not None:
        save_file(destination / 'config.dual.yaml', saved, staging, privilege)
    save_file(destination / 'previous.json', json.dumps({'containers': metadata, 'files': files}, indent=2), staging, privilege)
    return destination


def health(token, image_id):
    deadline = time.monotonic() + 45
    while True:
        try:
            item = containers()[PRIMARY]
            if item['Image'] != image_id or not item['State']['Running'] or item['RestartCount']:
                raise RuntimeError('双臂容器未正常运行或发生重启。')
            if not CARDS <= driver_idle(15741):
                raise RuntimeError('MCP 缺少双臂遥操或录制卡片。')
            require_idle(token, {PRIMARY: item})
            devices = request(CORE + '/api/mcp', token)['data']
            if not any(d.get('server_name') == 'agilex-piper'
                       and d.get('url') in {'http://localhost:15741/mcp', 'http://127.0.0.1:15741/mcp'}
                       and CARDS <= {t['name'] for t in d.get('tools', [])} for d in devices):
                raise RuntimeError('Core 尚未注册双臂卡片。')
            return
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError('双臂部署检查失败：' + str(exc)) from exc
            time.sleep(1)


def deploy(image, fragment, project):
    run_dir = fragment.parent
    if CONFIG.is_symlink() or DATA.resolve() != DATA:
        raise ValueError('PiPER 配置和数据目录不能是符号链接。')
    privilege = filesystem_access()
    saved = read_file(CONFIG, privilege) if CONFIG.exists() else None
    if saved is not None and not saved.strip():
        raise ValueError('已保存的双臂配置为空；请修复配置，不能使用默认左右绑定。')
    image_id = docker('image', 'inspect', '--format', '{{.Id}}', image).strip()
    config = yaml_config(image_id, saved)
    available = can_interfaces()
    if saved is None:
        if not sys.stdin.isatty():
            raise RuntimeError('首次部署需要终端确认左右 CAN；未停止任何服务。')
        print('可用 CAN：' + ', '.join(available))
        if len(available) < 2:
            raise RuntimeError('需要两个独立 CAN 接口。')
        config['arms']['left']['can_interface'] = input('左臂 CAN：').strip()
        config['arms']['right']['can_interface'] = input('右臂 CAN：').strip()
    left, right = bindings(config, available)
    found = containers()
    print(f'双臂绑定：左={left}，右={right}。')
    print('将切换以下旧实例：' + (', '.join(found) or '无'))
    if saved is None and input('确认左右绑定并部署双臂？输入 yes：').strip().lower() != 'yes':
        raise RuntimeError('已取消；未保存配置或停止服务。')
    token_text = read_file(CORE_ENV, privilege) if CORE_ENV.exists() else ''
    token = next((s.strip().split('=', 1)[1].strip() for s in token_text.splitlines()
                  if s.strip().startswith('ACCESS_TOKEN=')), '')
    compose = prepare_compose(image_id, fragment, project)
    print('检查画布和旧 Driver 状态……')
    require_idle(token, found)
    with tempfile.TemporaryDirectory(prefix='.piper-', dir=run_dir) as directory:
        staging = Path(directory)
        destination = backup(found, run_dir, staging, privilege, saved)
        # JSON is also YAML; keep the host dependency limited to Python's stdlib.
        if saved is None:
            save_file(CONFIG, json.dumps(config, ensure_ascii=False, indent=2) + '\n', staging, privilege)
        elif read_file(CONFIG, privilege) != saved:
            raise RuntimeError('部署期间配置已变化；未停止服务。')
        fresh = containers()
        if {n: (i['Id'], i['State']['Running']) for n, i in found.items()} != {
                n: (i['Id'], i['State']['Running']) for n, i in fresh.items()}:
            raise RuntimeError('部署期间容器已变化；未停止服务。')
        require_idle(token, fresh)
        private_write(run_dir / 'docker-compose.yml', json.dumps(compose, ensure_ascii=False, indent=2) + '\n')
        private_write(run_dir / 'state', f'IMAGE={image}\nSERVICE={SERVICE}\nCONTAINER={PRIMARY}\nPROJECT={project}\n')
        print(f'恢复资料：{destination}')
        try:
            print('停止旧实例并启动双臂……')
            if SECONDARY in fresh:
                docker('update', '--restart=no', SECONDARY)
            for name, item in fresh.items():
                if item['State']['Running']:
                    docker('stop', '--time', '20', name)
            if PRIMARY in fresh:
                # Older run-pr-image versions used a shared project; remove only this stopped container.
                docker('rm', PRIMARY)
            docker('compose', '-p', project, '-f', str(run_dir / 'docker-compose.yml'),
                   'up', '-d', '--no-deps', SERVICE)
            print('检查 MCP 和 Core 卡片注册……')
            health(token, image_id)
        except Exception as exc:
            diagnostics = str(exc) + '\n' + getattr(exc, 'diagnostics', '')
            try:
                item = containers().get(PRIMARY)
                if item and item['Image'] == image_id:
                    try:
                        diagnostics += docker('logs', '--tail', '100', PRIMARY)
                    finally:
                        docker('update', '--restart=no', PRIMARY)
                        docker('stop', '--time', '20', PRIMARY)
            finally:
                save_file(destination / 'failed.log', diagnostics, staging, privilege)
                print(f'切换失败。旧服务不会自动恢复；请保留恢复资料：{destination}', file=sys.stderr)
            raise
        save_file(destination / 'result.json', json.dumps({
            'image': image, 'image_id': image_id, 'left': left, 'right': right,
            'cards': sorted(CARDS), 'motion_enabled': False, 'physical_acceptance': False}, indent=2), staging, privilege)
    print('双臂部署完成：MCP 和 Core 已注册遥操、录制卡片。运动许可关闭；标定与实体动作尚未验收。')


def main():
    image, fragment, project = sys.argv[1:]
    fragment = Path(fragment).resolve()
    with (fragment.parent / '.piper-deploy.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deploy(image, fragment, project)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('PiPER 双臂部署失败：' + str(error), file=sys.stderr)
        sys.exit(1)
