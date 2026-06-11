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
    from api.registry import _build_catalog_sync

    current_tag = _get_current_tag()

    catalog = _build_catalog_sync()
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
    """pull 新镜像，然后启动 restart helper 容器完成 stop→remove→run。"""
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

    container_name = os.environ.get('CONTAINER_NAME', '')
    if not container_name:
        # Try to resolve from /proc/self/cgroup (container ID → name lookup)
        try:
            container_id = None
            with open('/proc/self/cgroup') as f:
                for line in f:
                    # cgroup v1: "12:devices:/docker/<id>"
                    # cgroup v2: "0::/system.slice/docker-<id>.scope"
                    parts = line.strip().split('/')
                    for part in reversed(parts):
                        if len(part) == 64 and all(c in '0123456789abcdef' for c in part):
                            container_id = part
                            break
                        if part.startswith('docker-') and part.endswith('.scope'):
                            container_id = part[7:-6]
                            break
                    if container_id:
                        break
            if container_id:
                container_name = client.containers.get(container_id).name
        except Exception:
            pass
    if not container_name:
        # Last resort: hostname (works when container --hostname matches container name)
        try:
            container_name = client.containers.get(socket.gethostname()).name
        except Exception as e:
            _set_error(f'无法获取当前容器名: {e}')
            return

    restart_image = os.environ.get('RESTART_IMAGE', '')
    if not restart_image:
        current_image = _get_current_image()
        # Image path: registry/org/category/name:tag — take only registry/org as base
        image_path = current_image.rsplit(':', 1)[0]  # strip tag
        parts = image_path.split('/')
        base = '/'.join(parts[:2]) if len(parts) >= 2 else parts[0]
        restart_image = f'{base}/restart:latest' if base else 'restart:latest'

    try:
        _set_step(f'启动 restart helper，替换容器 {container_name}…')
        client.containers.run(
            restart_image,
            detach=True,
            remove=True,
            network_mode='host',
            volumes={'/var/run/docker.sock': {'bind': '/var/run/docker.sock', 'mode': 'rw'}},
            environment={
                'CONTAINER_NAME': container_name,
                'NEW_IMAGE':      image,
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
