"""
system.py — Core 自我版本检测与热更新。

GET  /api/system/update-check   → 对比当前运行镜像 tag 与 resource-center 最新 tag
POST /api/system/update         → pull 新镜像，启动 restart helper 容器完成无缝切换
GET  /api/system/update-status  → 查询当前升级进度

升级进度走和驱动部署同一条 WebSocket（api/deploy_stream 的 /ws/deploy/{driver_id}）。
在此之前 core 只有 update-status 里的一个 step 字符串，前端因此被迫为 core 单开一套
「无预检、无百分比、按步数伪造进度条」的渲染，同一个升级动作长出两种形态。update-status
仍然保留：restart helper 换掉容器后进程就没了，重连的前端要靠它和镜像 tag 收尾。
"""

import asyncio
import os
import socket
import time

import fastapi

from api.deploy_stream import begin_run, emit_threadsafe, end_run

router = fastapi.APIRouter(prefix='/system', tags=['system'])

# ── In-memory update progress ─────────────────────────────────────────────────

_update_state: dict = {'step': '', 'error': '', 'ts': 0}

# 本次升级的事件流目标，由 /update 设置后供 _set_step/_set_error 用。
_stream_target: dict = {'loop': None, 'driver_id': '', 'run_id': ''}


def _emit(data: dict) -> None:
    loop = _stream_target.get('loop')
    if not loop:
        return
    emit_threadsafe(loop, _stream_target['driver_id'], data, _stream_target['run_id'])


def _set_step(msg: str, percent: float = None, stage: str = 'update', **extra) -> None:
    _update_state.update(step=msg, error='', ts=int(time.time()))
    print(f'[system] {msg}')
    data = {'type': 'progress', 'stage': stage, 'message': msg, **extra}
    if percent is not None:
        data['percent'] = round(percent, 1)
    _emit(data)

def _set_error(msg: str, suggestion: str = '') -> None:
    _update_state.update(error=msg, ts=int(time.time()))
    print(f'[system] ERROR: {msg}')
    data = {'type': 'error', 'error_type': 'update', 'message': msg}
    if suggestion:
        data['suggestion'] = suggestion
    _emit(data)


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


def _pull_with_progress(client, image: str, base: float, span: float) -> str:
    """拉取镜像并把层进度折算成 [base, base+span] 区间的百分比。

    走 api.pull 的流式接口而不是 images.pull：后者要等整个镜像拉完才返回，core 升级
    因此只有一句「正在拉取镜像…」挂在那里几分钟，和驱动部署的进度条完全不是一回事。
    返回空字符串表示成功，否则是错误信息。
    """
    layers: dict[str, dict] = {}
    pull_error = ''
    start = time.time()
    last_update = 0.0

    for line in client.api.pull(image, stream=True, decode=True):
        if line.get('error') or line.get('errorDetail'):
            pull_error = ((line.get('errorDetail') or {}).get('message')
                          or line.get('error') or 'unknown pull error')
            continue

        layer_id = line.get('id', '')
        detail = line.get('progressDetail') or {}
        if layer_id and detail.get('total'):
            layers[layer_id] = {
                'current': detail.get('current', 0),
                'total':   detail['total'],
                'status':  line.get('status', ''),
            }

        now = time.time()
        if now - last_update < 0.5 or not layers:
            continue
        last_update = now
        total_size = sum(l['total'] for l in layers.values())
        if total_size <= 0:
            continue
        done = sum(l['current'] for l in layers.values())
        frac = done / total_size
        speed = (done / (1 << 20)) / max(0.1, now - start)
        # 消息里不再自带百分比：日志行会由前端拼成「<消息> - <percent>% (<speed>)」，
        # 而 percent 是折算进 [base, base+span] 的整体进度，两个数字并排出现只会让人
        # 以为哪个错了。
        _set_step(
            f'拉取镜像：{len(layers)} 层',
            percent=base + span * frac,
            stage='pull',
            speed=f'{speed:.1f} MB/s',
        )

    return pull_error


def _pull_and_restart_sync(image: str) -> None:
    """pull 新镜像，然后启动 restart helper 容器通过 docker compose 完成切换。"""
    import docker as docker_sdk
    try:
        client = docker_sdk.from_env()
    except Exception as e:
        _set_error(f'无法连接 Docker: {e}', suggestion='确认容器挂载了 /var/run/docker.sock。')
        return

    try:
        _set_step(f'正在拉取镜像 {image.rsplit(":", 1)[-1]}…', percent=5, stage='pull')
        err = _pull_with_progress(client, image, base=5, span=65)
        if err:
            _set_error(f'镜像拉取失败: {err}')
            return
        _set_step('镜像拉取完成', percent=70, stage='pull')
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
        _set_step(f'正在拉取 restart helper…', percent=75, stage='pull')
        client.images.pull(restart_image)
    except Exception as e:
        _set_error(f'restart helper 镜像拉取失败: {e}')
        return

    try:
        _set_step(f'启动 restart helper，升级 agent-core → {image.rsplit(":", 1)[-1]}…',
                  percent=85, stage='start')
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
        # 这是本进程能发出的最后一条消息：restart helper 接下来就把它换掉。没有 done
        # 事件 —— 升级是否成功要由重连后的 tag 判断，见 web/js/deploy-panel.js。
        _set_step('restart helper 已启动，容器即将切换…', percent=90, stage='restart')
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
    body = body or {}
    image = body.get('image', '')
    if not image:
        raise fastapi.HTTPException(status_code=400, detail='image is required')

    # 前端用它自己的 core 条目 id 开进度窗口，这里就用同一个 id 建流，两端才对得上。
    driver_id = body.get('driver_id') or 'core'
    run_id = begin_run(driver_id, image)
    _stream_target.update(loop=asyncio.get_running_loop(), driver_id=driver_id, run_id=run_id)

    _update_state.update(step='升级任务已启动…', error='', ts=int(time.time()))
    _set_step('升级任务已启动…', percent=2)

    async def _do_update():
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, _pull_and_restart_sync, image)
        finally:
            # 失败时收流；成功时进程已被 restart helper 换掉，走不到这里。
            if _update_state.get('error'):
                end_run(driver_id)

    asyncio.create_task(_do_update())
    return {'code': 200, 'data': {'message': '升级任务已启动', 'run_id': run_id}}


@router.get('/update-status')
async def update_status():
    return {'code': 200, 'data': _update_state}
