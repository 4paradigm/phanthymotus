"""
system.py — Core 自我版本检测与热更新。

GET  /api/system/update-check   → 对比当前运行镜像 tag 与 resource-center 最新 tag
POST /api/system/update         → pull 新镜像，启动 restart helper 容器完成无缝切换
GET  /api/system/update-status  → 查询当前升级进度
"""

import asyncio
import os
import socket
import time

import fastapi

router = fastapi.APIRouter(prefix='/system', tags=['system'])

# ── In-memory update progress ─────────────────────────────────────────────────

_update_state: dict = {'step': '', 'error': '', 'ts': 0}

def _set_step(msg: str) -> None:
    _update_state.update(step=msg, error='', ts=int(time.time()))
    print(f'[system] {msg}')

def _set_error(msg: str) -> None:
    _update_state.update(error=msg, ts=int(time.time()))
    print(f'[system] ERROR: {msg}')


def _get_current_tag() -> str:
    """从镜像内 VERSION 文件读取当前版本 tag。"""
    try:
        return open('/work/VERSION').read().strip()
    except Exception:
        return os.environ.get('IMAGE_TAG', 'unknown')


def _get_current_image() -> str:
    """返回当前容器运行的完整镜像引用，失败时返回空字符串。"""
    try:
        import docker as docker_sdk
        client = docker_sdk.from_env()

        # 1. 环境变量直接注入
        name = os.environ.get('CONTAINER_NAME', '')
        if name:
            return client.containers.get(name).attrs.get('Config', {}).get('Image', '')

        # 2. /proc/self/cgroup 解析容器 ID
        try:
            with open('/proc/self/cgroup') as f:
                for line in f:
                    parts = line.strip().split('/')
                    for part in reversed(parts):
                        if len(part) == 64 and all(c in '0123456789abcdef' for c in part):
                            return client.containers.get(part).attrs.get('Config', {}).get('Image', '')
                        if part.startswith('docker-') and part.endswith('.scope'):
                            cid = part[7:-6]
                            return client.containers.get(cid).attrs.get('Config', {}).get('Image', '')
        except Exception:
            pass

        # 3. hostname fallback
        return client.containers.get(socket.gethostname()).attrs.get('Config', {}).get('Image', '')
    except Exception as e:
        print(f'[system] get_current_image failed: {e}')
        return ''


def _tag_from_image(image: str) -> str:
    """从完整镜像引用提取 tag，如 'registry/.../core:release.260531.abc' → 'release.260531.abc'"""
    return image.rsplit(':', 1)[-1] if ':' in image else ''


def _check_update_sync() -> dict:
    from api.registry import _build_catalog_sync, _current_channel

    current_tag = _get_current_tag()

    catalog = _build_catalog_sync(_current_channel())
    core_items = catalog.get('core', [])

    if not core_items:
        return {'up_to_date': True, 'current_tag': current_tag, 'latest_tag': '', 'latest_image': ''}

    latest_item = core_items[0]
    tags = latest_item.get('tags', [])
    if not tags:
        return {'up_to_date': True, 'current_tag': current_tag, 'latest_tag': '', 'latest_image': ''}

    latest_tag_obj = tags[0]
    latest_tag = latest_tag_obj.get('tag', '')
    latest_image = latest_tag_obj.get('imageRef', '')
    if not latest_image:
        full_repo = latest_item.get('full_repo', '')
        latest_image = f'{full_repo}:{latest_tag}' if full_repo else ''

    up_to_date = (current_tag == latest_tag) if (current_tag and latest_tag) else True

    return {
        'up_to_date': up_to_date,
        'current_tag': current_tag,
        'latest_tag': latest_tag,
        'latest_image': latest_image,
    }


def _pull_and_restart_sync(image: str) -> None:
    """pull 新镜像，然后启动 restart helper 容器通过 docker compose 完成切换。"""
    import docker as docker_sdk
    try:
        client = docker_sdk.from_env()
    except Exception as e:
        _set_error(f'无法连接 Docker: {e}')
        return

    try:
        _set_step(f'正在拉取镜像 {image.rsplit(":", 1)[-1]}…')
        client.images.pull(image)
    except Exception as e:
        _set_error(f'镜像拉取失败: {e}')
        return

    restart_image = os.environ.get('RESTART_IMAGE', '')
    if not restart_image:
        # 从目标镜像（而非 current_image）推导 registry 前缀，确保使用正确仓库
        image_path = image.rsplit(':', 1)[0]  # strip tag
        parts = image_path.split('/')
        # registry/namespace/name → registry/namespace; 无 registry 则用 image 的前两段
        if len(parts) >= 3:
            base = '/'.join(parts[:2])  # registry/namespace
        elif len(parts) == 2:
            base = parts[0]  # 可能是 namespace/name，取 namespace
        else:
            base = ''
        restart_image = f'{base}/restart:latest' if base else 'restart:latest'

    try:
        _set_step(f'正在拉取 restart helper…')
        client.images.pull(restart_image)
    except Exception as e:
        _set_error(f'restart helper 镜像拉取失败: {e}')
        return

    try:
        _set_step(f'启动 restart helper，升级 agent-core → {image.rsplit(":", 1)[-1]}…')
        compose_dir = os.environ.get('COMPOSE_DIR', '/opt/phanthy-motus')
        container_name = os.environ.get('CONTAINER_NAME', 'phanthy-motus-agent-core-1')
        client.containers.run(
            restart_image,
            detach=True,
            remove=True,
            network_mode='host',
            volumes={
                '/var/run/docker.sock': {'bind': '/var/run/docker.sock', 'mode': 'rw'},
                compose_dir: {'bind': compose_dir, 'mode': 'rw'},
            },
            environment={
                # 兼容新旧两版 restart helper entrypoint
                'COMPOSE_DIR':    compose_dir,
                'SERVICE':        'agent-core',
                'NEW_IMAGE':      image,
                'CONTAINER_NAME': container_name,
            },
        )
        _set_step('restart helper 已启动，容器即将切换…')
    except Exception as e:
        _set_error(f'启动 restart helper 失败: {e}')


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get('/update-check')
async def update_check():
    loop = asyncio.get_event_loop()
    try:
        data = await loop.run_in_executor(None, _check_update_sync)
    except Exception as e:
        print(f'[system] update_check error: {e}')
        return {'code': 200, 'data': {'up_to_date': True}}
    return {'code': 200, 'data': data}


@router.post('/update')
async def update(body: dict = fastapi.Body(default={})):
    image = (body or {}).get('image', '')
    if not image:
        raise fastapi.HTTPException(status_code=400, detail='image is required')

    _update_state.update(step='升级任务已启动…', error='', ts=int(time.time()))

    async def _do_update():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _pull_and_restart_sync, image)

    asyncio.create_task(_do_update())
    return {'code': 200, 'data': {'message': '升级任务已启动'}}


@router.get('/update-status')
async def update_status():
    return {'code': 200, 'data': _update_state}
