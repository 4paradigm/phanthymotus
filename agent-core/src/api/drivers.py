"""
drivers.py — Driver catalog & lifecycle management via Docker socket.
Manifest stored in SQLite config DB (key: 'drivers'). Populated via registry sync.
"""

import asyncio
import os
from typing import Optional

import fastapi

import config as _config

router = fastapi.APIRouter(prefix='/drivers', tags=['drivers'])

# Fixed service endpoints for core/perception/inspection (not hardware drivers)
_SERVICE_ENDPOINTS: dict[str, dict] = {
    'core':       {'host_port': 15678},
    'perception': {'port': 15720, 'mcp_url': 'http://localhost:15720/mcp',
                   'volumes': {os.environ.get('MODELS_PATH', '/opt/embodied/models'):
                               {'bind': '/models', 'mode': 'rw'}}},
    'inspection': {'port': 15671},
}


# ── Manifest persistence ───────────────────────────────────────────────────

def _load_manifest() -> list:
    drivers = _config.main.get('drivers')
    return list(drivers) if drivers is not None else []


def _save_manifest(drivers: list) -> None:
    _config.main['drivers'] = drivers


# ── Docker helpers ─────────────────────────────────────────────────────────

def _docker():
    import docker
    return docker.from_env()


def _container_name(driver_id: str) -> str:
    return f'embodied-{driver_id}'


def _get_status_sync(driver_id: str) -> dict:
    try:
        client = _docker()
        name = _container_name(driver_id)
        containers = client.containers.list(all=True, filters={'name': f'^{name}$'})
        if not containers:
            return {'status': 'stopped'}
        c = containers[0]
        try:
            logs = c.logs(tail=30).decode('utf-8', errors='replace')
        except Exception:
            logs = ''
        running_image = c.attrs.get('Config', {}).get('Image', '')
        return {'status': c.status, 'logs': logs, 'running_image': running_image}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def _deploy_sync(driver: dict) -> dict:
    import docker as docker_sdk
    client = _docker()
    name = _container_name(driver['id'])
    target_image = driver['image']

    # Check if already running with the same image — skip re-deploy
    try:
        existing = client.containers.get(name)
        if existing.status == 'running':
            running_image = existing.attrs.get('Config', {}).get('Image', '')
            if running_image == target_image:
                return {'status': 'running', 'message': 'already running with same image', 'skipped': True}
            # Different image: stop and remove
            existing.stop(timeout=5)
            existing.remove(force=True)
        else:
            existing.remove(force=True)
    except docker_sdk.errors.NotFound:
        pass

    port = driver.get('port')
    host_port = driver.get('host_port')
    # ROS2 topic names cannot start with a number (Docker default hostname = short container ID).
    # Use a sanitized hostname derived from the container name.
    ros_hostname = name.replace('-', '_')  # e.g. embodied_unitree_g1

    run_kwargs = dict(
        image=target_image,
        detach=True,
        name=name,
        hostname=ros_hostname,
        remove=False,
        restart_policy={'Name': 'unless-stopped'},
    )

    # Jetson GPU: add nvidia runtime for images with '-jetson' tag
    if '-jetson' in target_image:
        run_kwargs['runtime'] = 'nvidia'
        run_kwargs['environment'] = {'NVIDIA_VISIBLE_DEVICES': 'all'}
    else:
        run_kwargs['privileged'] = True
    container_network = os.environ.get('CONTAINER_NETWORK', '')
    network_mode = driver.get('network_mode', '')
    if network_mode == 'host' or (not network_mode and not container_network):
        # host 模式：DDS 驱动容器需要共享宿主机网络才能收发 ROS2 DDS
        run_kwargs['network_mode'] = 'host'
    else:
        if container_network:
            run_kwargs['network'] = container_network
        # Map host_port (explicit host→container) or port (container port same as host)
        if host_port:
            run_kwargs['ports'] = {f'{host_port}/tcp': host_port}
        elif port and not container_network:
            run_kwargs['ports'] = {f'{port}/tcp': port}

    # Extra env vars from manifest
    env = dict(driver.get('environment') or {})
    # Inject ROS2 DDS env vars (inherit from agent-core) so containers join the same domain
    for key in ('ROS_DOMAIN_ID', 'RMW_IMPLEMENTATION', 'FASTDDS_BUILTIN_TRANSPORTS'):
        if key not in env and os.environ.get(key):
            env[key] = os.environ[key]
    if env:
        run_kwargs['environment'] = env

    # Volume mounts from manifest + auto-inject persistent data volume
    volumes = dict(driver.get('volumes') or {})
    # Ensure spatial data directory is always persisted to host
    # Docker daemon auto-creates host dirs that don't exist
    data_dir = '/opt/phanthy-motus/data'
    has_data_mount = any(
        (v == data_dir) or (isinstance(v, dict) and v.get('bind') == data_dir)
        for v in volumes.values()
    )
    if not has_data_mount:
        volumes[data_dir] = {'bind': data_dir, 'mode': 'rw'}
    if volumes:
        run_kwargs['volumes'] = volumes

    container = client.containers.run(**run_kwargs)
    return {'status': 'starting', 'container_id': container.id[:12]}


def _stop_sync(driver_id: str) -> dict:
    import docker as docker_sdk
    try:
        client = _docker()
        name = _container_name(driver_id)
        container = client.containers.get(name)
        container.stop(timeout=5)
        return {'status': 'stopped'}
    except docker_sdk.errors.NotFound:
        return {'status': 'already_stopped'}
    except Exception as e:
        raise RuntimeError(str(e))


async def _run_in_executor(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


# ── Registry sync helper ───────────────────────────────────────────────────

def _upsert_from_catalog(manifest: list, catalog: dict) -> tuple[int, int]:
    """Upsert drivers from registry catalog into manifest list (in-place).
    Returns (added, updated) counts.
    """
    added = 0
    updated = 0

    all_items = (
        catalog.get('core', []) +
        catalog.get('driver', []) +
        catalog.get('perception', []) +
        catalog.get('inspection', [])
    )

    for item in all_items:
        tags = item.get('tags', [])
        if not tags:
            continue

        image_name = item.get('image', '')   # e.g. "driver-unitree-g1"
        category   = item.get('category', 'driver')

        # Derive driver id (mirrors frontend _driverIdForItem)
        if category == 'driver':
            driver_id = f"{item.get('provider', '')}-{item.get('model', '')}".strip('-')
        else:
            driver_id = image_name

        latest_tag = tags[0]
        # Prefer imageRef from resource-center; fall back to building from full_repo
        if latest_tag.get('imageRef'):
            full_image = latest_tag['imageRef']
        else:
            full_repo = item.get('full_repo', '')
            full_image = f'{full_repo}:{latest_tag["tag"]}' if full_repo else image_name

        # Look for existing entry by id or registry_image
        existing = next(
            (d for d in manifest if d.get('id') == driver_id or d.get('registry_image') == image_name),
            None,
        )

        if existing:
            existing['image'] = full_image
            # Sync fixed endpoint fields (port, host_port, mcp_url) in case they were added later
            for k, v in _SERVICE_ENDPOINTS.get(image_name, {}).items():
                existing.setdefault(k, v)
            # Also sync port from registry catalog for hardware drivers
            if item.get('port') and not existing.get('port'):
                existing['port'] = item['port']
            updated += 1
        else:
            # Build human-readable name
            if category == 'driver':
                name = f"{item.get('provider', '').title()} {item.get('model', '').upper()}".strip()
            else:
                name = item.get('name', image_name)

            new_entry: dict = {
                'id':             driver_id,
                'name':           name,
                'category':       category,
                'registry_image': image_name,
                'image':          full_image,
                'description':    '',
                **_SERVICE_ENDPOINTS.get(image_name, {}),
            }
            # Preserve port from registry catalog for hardware drivers (used to derive mcp_url)
            if item.get('port') and 'port' not in new_entry:
                new_entry['port'] = item['port']
            manifest.append(new_entry)
            added += 1

    return added, updated


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get('')
async def drivers_list():
    manifest = _load_manifest()
    result = []
    try:
        self_tag = open('/work/VERSION').read().strip()
    except Exception:
        self_tag = os.environ.get('IMAGE_TAG', '')
    for d in manifest:
        if d.get('category') == 'core' and self_tag:
            # core 能响应请求说明自身正在运行，直接从 env 读 tag
            base = d.get('image', '').rsplit(':', 1)[0]
            result.append({
                'id':            d['id'],
                'name':          d['name'],
                'image':         d.get('image', ''),
                'port':          d.get('port'),
                'description':   d.get('description', ''),
                'category':      'core',
                'mcp_url':       d.get('mcp_url', ''),
                'running_image': f'{base}:{self_tag}',
                'last_deploy':   d.get('last_deploy'),
            })
        else:
            status_info = await _run_in_executor(_get_status_sync, d['id'])
            result.append({
                'id':            d['id'],
                'name':          d['name'],
                'image':         d.get('image', ''),
                'port':          d.get('port'),
                'description':   d.get('description', ''),
                'category':      d.get('category', 'driver'),
                'mcp_url':       d.get('mcp_url', ''),
                'running':       status_info.get('status') == 'running',
                'status':        status_info.get('status', 'stopped'),
                'running_image': status_info.get('running_image', ''),
                'last_deploy':   d.get('last_deploy'),
            })
    return {'code': 200, 'data': result}


@router.post('/sync')
async def drivers_sync():
    """Fetch registry catalog and upsert drivers in DB."""
    from api.registry import _build_catalog_sync, _cache as _registry_cache
    loop = asyncio.get_event_loop()
    try:
        catalog = await loop.run_in_executor(None, _build_catalog_sync)
    except Exception as e:
        return {'code': 500, 'message': str(e)}

    # Update registry cache with fresh data so next GET /registry/catalog is immediate
    _registry_cache['data'] = catalog
    _registry_cache['ts']   = __import__('time').time()

    manifest = _load_manifest()
    added, updated = _upsert_from_catalog(manifest, catalog)
    _save_manifest(manifest)
    return {'code': 200, 'data': {'added': added, 'updated': updated}}


@router.post('/{driver_id}/deploy')
async def driver_deploy(driver_id: str, body: dict = fastapi.Body(default={})):
    manifest = _load_manifest()
    driver = next((d for d in manifest if d['id'] == driver_id), None)
    if not driver:
        raise fastapi.HTTPException(status_code=404, detail='Driver not found in manifest')

    # Allow caller to override the image via:
    #   {"image": "full_image_ref"} — direct override
    #   {"registry_image": "namespace/image-name", "tag": "release.xxx"} — from registry catalog
    image_override = ''
    if isinstance(body, dict):
        if body.get('image'):
            image_override = body['image']
        elif body.get('registry_image') and body.get('tag'):
            ri = body['registry_image']
            tag = body['tag']
            image_override = f'{ri}:{tag}'
    if image_override:
        driver = {**driver, 'image': image_override}

    try:
        result = await _run_in_executor(_deploy_sync, driver)
    except Exception as e:
        return {'code': 500, 'message': str(e)}

    # Persist updated image into manifest (so DB remembers last deployed image)
    if not result.get('skipped'):
        import time as _time
        manifest = _load_manifest()
        for d in manifest:
            if d.get('id') == driver_id:
                d['image'] = driver['image']
                d['last_deploy'] = {
                    'image':  driver['image'],
                    'ts':     int(_time.time()),
                    'status': result.get('status', ''),
                }
                break
        _save_manifest(manifest)

    return {'code': 200, 'data': result}


@router.post('/{driver_id}/stop')
async def driver_stop(driver_id: str):
    try:
        result = await _run_in_executor(_stop_sync, driver_id)
        return {'code': 200, 'data': result}
    except Exception as e:
        return {'code': 500, 'message': str(e)}


@router.get('/{driver_id}/status')
async def driver_status(driver_id: str):
    status = await _run_in_executor(_get_status_sync, driver_id)
    return {'code': 200, 'data': status}

