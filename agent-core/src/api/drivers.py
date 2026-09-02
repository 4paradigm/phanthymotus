"""
drivers.py — Driver catalog & lifecycle management via Docker socket.
Manifest stored in SQLite config DB (key: 'drivers'). Populated via registry sync.
"""

import asyncio
import contextlib
import fcntl
import os
import threading
import time
from typing import Optional

import fastapi

import config as _config

router = fastapi.APIRouter(prefix='/drivers', tags=['drivers'])

# Fixed service endpoints for core/perception/actucore (not hardware drivers).
# Keyed by registryImage — the same string the build scripts register with.
_SERVICE_ENDPOINTS: dict[str, dict] = {
    'core':       {'host_port': 15678},
    'perception': {'port': 15720, 'mcp_url': 'http://localhost:15720/mcp',
                   'volumes': {os.environ.get('MODELS_PATH', '/opt/embodied/models'):
                               {'bind': '/models', 'mode': 'rw'}}},
    'actucore':   {'port': 15730, 'mcp_url': 'http://localhost:15730/mcp',
                   'volumes': {os.environ.get('MODELS_PATH', '/opt/embodied/models'):
                               {'bind': '/models', 'mode': 'rw'}}},
}


# ── Manifest persistence ───────────────────────────────────────────────────

def _load_manifest() -> list:
    drivers = _config.main.get('drivers')
    return list(drivers) if drivers is not None else []


def _save_manifest(drivers: list) -> None:
    _config.main['drivers'] = drivers


# ── Docker helpers ─────────────────────────────────────────────────────────

# Log rotation policy, kept here so the legacy `docker run` fallback below and
# the `logging:` block every deploy/service.yml declares stay visibly the same
# policy. Without the options the `local` driver falls back to its own defaults
# (20m x 5 = 100 MB), i.e. 3.3x what we intend, silently and per container.
LOG_MAX_SIZE = '10m'
LOG_MAX_FILE = '3'


def _log_config() -> dict:
    """docker-py log_config for a container. Values must be strings."""
    return {
        'type': 'local',
        'config': {'max-size': LOG_MAX_SIZE, 'max-file': LOG_MAX_FILE},
    }


def _docker():
    import docker
    return docker.from_env()


def _container_name(driver_id: str, override: str = '') -> str:
    """Return container name: use override from manifest if available, else embodied-{id}."""
    return override if override else f'embodied-{driver_id}'


# ── Deploy log buffer (in-memory, per driver) ─────────────────────────────

_deploy_logs: dict[str, str] = {}


def _log_deploy(driver_id: str, msg: str):
    _deploy_logs.setdefault(driver_id, '')
    _deploy_logs[driver_id] += msg + '\n'


def _clear_deploy_log(driver_id: str):
    _deploy_logs.pop(driver_id, None)


# ── Host compose file mutation ─────────────────────────────────────────────
#
# The host compose file has two independent writers: this module (merging each
# image's deploy/service.yml fragment) and deploy/restart/entrypoint.sh (swapping
# agent-core's image tag during a self-update). Neither used to lock, and both
# wrote with a plain open(...,'w') — which truncates at open and flushes at
# close, with no truncate in between. Two overlapping writers therefore left the
# shorter document at offset 0 followed by the tail of the longer one, i.e. a
# file that no YAML parser accepts and that nothing could repair afterwards
# (every retry died in safe_load before it could write).
#
# Two deploys landing in the same second is not hypothetical: it is what a
# double-clicked or retried deploy in the console does, since the
# already-running-same-image guard only short-circuits a container that is
# already up.

_COMPOSE_LOCK_NAME = '.compose.lock'


@contextlib.contextmanager
def _compose_lock(compose_dir: str, timeout: float = 120.0):
    """Hold an exclusive lock over the host compose file for the whole RMW cycle.

    The lock file lives in COMPOSE_DIR, which is bind-mounted into both this
    container and the restart helper, so both contend on one host inode.
    deploy/restart/entrypoint.sh takes the same lock by name — keep them in sync.
    """
    lock_path = os.path.join(compose_dir, _COMPOSE_LOCK_NAME)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f'timed out after {timeout:.0f}s waiting for {lock_path}; '
                        'another deploy or an agent-core self-update is in progress'
                    )
                time.sleep(0.2)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def _merge_service_into_compose(compose_file: str, service_def: dict) -> tuple[bool, str]:
    """Merge a service fragment into the host compose file under lock.

    Returns (ok, error). On any doubt about the existing file this refuses to
    write rather than guessing: a compose we cannot read is a compose whose other
    services we cannot preserve, and silently dropping them is how agent-core
    disappeared from the file on orin6.
    """
    import yaml

    compose_dir = os.path.dirname(compose_file) or '.'
    with _compose_lock(compose_dir):
        try:
            with open(compose_file) as f:
                raw = f.read()
        except FileNotFoundError:
            # Genuine fresh install — install.sh has not run yet.
            raw = ''
            existing: dict = {}
        else:
            # An existing-but-empty file is NOT a fresh install; it is a
            # truncated read or a damaged file. `safe_load(...) or {}` used to
            # turn this into "no other services exist" and write a compose
            # containing only the service being deployed.
            if not raw.strip():
                return False, f'{compose_file} exists but is empty — refusing to overwrite'
            try:
                loaded = yaml.safe_load(raw)
            except yaml.YAMLError as e:
                return False, f'{compose_file} is not valid YAML, refusing to overwrite: {e}'
            if not isinstance(loaded, dict):
                return False, f'{compose_file} is not a mapping, refusing to overwrite'
            existing = loaded

        services = existing.setdefault('services', {})
        if not isinstance(services, dict):
            return False, f'{compose_file}: services is not a mapping, refusing to overwrite'

        preserved = set(services)
        services.update(service_def)

        text = yaml.dump(existing, default_flow_style=False, allow_unicode=True, sort_keys=False)

        # Re-parse what we are about to write and confirm every service we
        # started with survived. Cheap insurance: corruption here is otherwise
        # invisible until the next `docker compose` call, by which point the
        # last good copy is gone.
        try:
            check = yaml.safe_load(text) or {}
        except yaml.YAMLError as e:
            return False, f'refusing to write unparseable compose: {e}'
        written = set((check.get('services') or {}))
        missing = preserved - written
        if missing:
            return False, f'refusing to write: would drop service(s) {sorted(missing)}'

        # Keep the last good copy before replacing.
        if raw:
            try:
                with open(compose_file + '.bak', 'w') as f:
                    f.write(raw)
            except OSError as e:
                return False, f'could not write backup {compose_file}.bak: {e}'

        # Atomic replace: a concurrent reader sees either the old or the new
        # file, never a half-written one.
        tmp = compose_file + '.tmp'
        with open(tmp, 'w') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, compose_file)

    return True, ''


# ── Per-driver in-flight guard ─────────────────────────────────────────────

_deploy_inflight: set[str] = set()
_deploy_inflight_lock = threading.Lock()


@contextlib.contextmanager
def _deploy_slot(driver_id: str):
    """Reject a second concurrent deploy of the same driver.

    Yields True if this call owns the slot, False if a deploy is already running.
    """
    with _deploy_inflight_lock:
        owned = driver_id not in _deploy_inflight
        if owned:
            _deploy_inflight.add(driver_id)
    try:
        yield owned
    finally:
        if owned:
            with _deploy_inflight_lock:
                _deploy_inflight.discard(driver_id)


def _get_status_sync(driver_id: str, container_name_override: str = '') -> dict:
    try:
        client = _docker()
        name = _container_name(driver_id, container_name_override)
        containers = client.containers.list(all=True, filters={'name': f'^{name}$'})
        if not containers:
            deploy_log = _deploy_logs.get(driver_id, '')
            return {'status': 'stopped', 'logs': deploy_log}
        c = containers[0]
        try:
            logs = c.logs(tail=100).decode('utf-8', errors='replace')
        except Exception:
            logs = ''
        # Prepend deploy logs if available
        deploy_log = _deploy_logs.get(driver_id, '')
        if deploy_log:
            logs = deploy_log + logs
        running_image = c.attrs.get('Config', {}).get('Image', '')
        return {'status': c.status, 'logs': logs, 'running_image': running_image}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def _deploy_sync(driver: dict) -> dict:
    """Deploy a driver, rejecting a second concurrent deploy of the same driver.

    The console retries and double-clicks; two deploys of one driver racing each
    other is what corrupted the host compose file on orin6.
    """
    with _deploy_slot(driver['id']) as owned:
        if not owned:
            return {
                'status':  'deploying',
                'message': 'a deploy for this driver is already in progress',
                'skipped': True,
            }
        return _deploy_sync_inner(driver)


def _deploy_sync_inner(driver: dict) -> dict:
    """Deploy a driver/perception container via docker compose.

    Extracts service.yml from the target image and merges it into the host
    compose file, then runs docker compose up for that service.
    """
    import subprocess
    import tarfile
    import io
    import docker as docker_sdk

    client = _docker()
    name = _container_name(driver['id'], driver.get('container_name', ''))
    target_image = driver['image']

    # Check if already running with the same image — skip re-deploy
    try:
        existing = client.containers.get(name)
        if existing.status == 'running':
            running_image = existing.attrs.get('Config', {}).get('Image', '')
            if running_image == target_image:
                return {'status': 'running', 'message': 'already running with same image', 'skipped': True}
    except docker_sdk.errors.NotFound:
        pass

    # Pull image
    _clear_deploy_log(driver['id'])
    _log_deploy(driver['id'], f'[pull] {target_image}')
    try:
        for line in client.api.pull(target_image, stream=True, decode=True):
            status = line.get('status', '')
            progress = line.get('progress', '')
            layer_id = line.get('id', '')
            if status:
                msg = f'  {layer_id} {status}' if layer_id else f'  {status}'
                if progress:
                    msg += f' {progress}'
                _log_deploy(driver['id'], msg)
    except Exception as e:
        _log_deploy(driver['id'], f'[pull] failed: {e}')
        return {'status': 'error', 'error': f'pull failed: {e}'}

    # Extract service.yml from image
    compose_dir = os.environ.get('COMPOSE_DIR', '/opt/phanthy-motus')
    compose_file = os.path.join(compose_dir, 'docker-compose.yml')

    # Ensure compose dir exists (may be a host-mounted volume)
    os.makedirs(compose_dir, exist_ok=True)

    container = client.containers.create(target_image)
    try:
        bits, _ = container.get_archive('/deploy/service.yml')
        tar_bytes = b''.join(bits)
        tf = tarfile.open(fileobj=io.BytesIO(tar_bytes))
        service_content = tf.extractfile('service.yml').read().decode()
    except Exception:
        # Fallback: image doesn't have service.yml — use legacy docker run
        try:
            container.remove(force=True)
        except Exception:
            pass
        return _deploy_sync_legacy(driver)
    try:
        container.remove(force=True)
    except Exception:
        pass

    # Parse service fragment and merge into compose
    import yaml
    service_def = yaml.safe_load(service_content)
    if not service_def or not isinstance(service_def, dict):
        return _deploy_sync_legacy(driver)

    service_name = list(service_def.keys())[0]
    service_def[service_name]['image'] = target_image
    # Default the rotation policy for images whose service.yml predates it (or
    # comes from a third party). Declared blocks win — this only fills a gap.
    service_def[service_name].setdefault('logging', {
        'driver': 'local',
        'options': {'max-size': LOG_MAX_SIZE, 'max-file': LOG_MAX_FILE},
    })

    # Merge into the host compose under lock, atomically. Bail out before
    # touching any container if the existing file cannot be read safely — a
    # failed merge that still removed the old container would leave the driver
    # both stopped and undeployable.
    ok, err = False, ''
    try:
        ok, err = _merge_service_into_compose(compose_file, service_def)
    except TimeoutError as e:
        err = str(e)
    if not ok:
        _log_deploy(driver['id'], f'[compose] {err}')
        return {'status': 'error', 'error': err}

    # Remove old container if it exists (may be from legacy docker run)
    try:
        old = client.containers.get(name)
        old.remove(force=True)
    except docker_sdk.errors.NotFound:
        pass
    # Also try the container_name from service.yml
    svc_container_name = service_def[service_name].get('container_name', '')
    if svc_container_name and svc_container_name != name:
        try:
            old = client.containers.get(svc_container_name)
            old.remove(force=True)
        except docker_sdk.errors.NotFound:
            pass

    # docker compose up
    _log_deploy(driver['id'], f'[compose] up -d {service_name}')
    result = subprocess.run(
        ['docker', 'compose', '-f', compose_file, 'up', '-d', '--no-deps', '--force-recreate', service_name],
        capture_output=True, text=True,
    )
    if result.stdout:
        _log_deploy(driver['id'], result.stdout.strip())
    if result.stderr:
        _log_deploy(driver['id'], result.stderr.strip())
    if result.returncode != 0:
        _log_deploy(driver['id'], f'[compose] exit code {result.returncode}')
        return {'status': 'error', 'error': f'compose up failed (rc={result.returncode})'}
    _log_deploy(driver['id'], '[deploy] done')
    return {'status': 'starting', 'service': service_name, 'container_name': svc_container_name}


def _deploy_sync_legacy(driver: dict) -> dict:
    """Fallback: deploy via docker run for images without /deploy/service.yml."""
    import docker as docker_sdk
    client = _docker()
    name = _container_name(driver['id'], driver.get('container_name', ''))
    target_image = driver['image']

    # Remove existing
    try:
        existing = client.containers.get(name)
        existing.stop(timeout=5)
        existing.remove(force=True)
    except docker_sdk.errors.NotFound:
        pass

    port = driver.get('port')
    host_port = driver.get('host_port')
    ros_hostname = name.replace('-', '_')

    run_kwargs = dict(
        image=target_image,
        detach=True,
        name=name,
        hostname=ros_hostname,
        remove=False,
        restart_policy={'Name': 'unless-stopped'},
    )

    # Second consumer of the `-jetson` tag suffix, alongside hostarch.py /
    # resource-center's lib/arch.ts — but a different question: does this image want
    # the nvidia runtime? Keep it even though the catalog is now arch-filtered:
    # POST /api/drivers/{id}/deploy still accepts an arbitrary image, so the UI being
    # unable to pick a mismatched one is not a guarantee. If tags ever stop carrying
    # `-jetson`, this silently falls back to privileged without the nvidia runtime.
    if '-jetson' in target_image:
        # Only use nvidia runtime if available on host
        try:
            runtimes = client.info().get('Runtimes', {})
        except Exception:
            runtimes = {}
        if 'nvidia' in runtimes:
            run_kwargs['runtime'] = 'nvidia'
            env_base = {'NVIDIA_VISIBLE_DEVICES': 'all'}
        else:
            run_kwargs['privileged'] = True
            env_base = {}
    else:
        run_kwargs['privileged'] = True
        env_base = {}

    container_network = os.environ.get('CONTAINER_NETWORK', '')
    network_mode = driver.get('network_mode', '')
    if network_mode == 'host' or (not network_mode and not container_network):
        run_kwargs['network_mode'] = 'host'
        run_kwargs['ipc_mode'] = 'host'
        run_kwargs['pid_mode'] = 'host'
    else:
        if container_network:
            run_kwargs['network'] = container_network
        if host_port:
            run_kwargs['ports'] = {f'{host_port}/tcp': host_port}
        elif port and not container_network:
            run_kwargs['ports'] = {f'{port}/tcp': port}

    env = dict(driver.get('environment') or {})
    for key in ('ROS_DOMAIN_ID', 'RMW_IMPLEMENTATION', 'FASTDDS_BUILTIN_TRANSPORTS'):
        if key not in env and os.environ.get(key):
            env[key] = os.environ[key]
    final_env = {**env_base, **env}
    if final_env:
        run_kwargs['environment'] = final_env

    if driver.get('volumes'):
        run_kwargs['volumes'] = driver['volumes']

    run_kwargs['log_config'] = _log_config()

    _log_deploy(driver['id'], f'[run] {name}')
    try:
        container = client.containers.run(**run_kwargs)
    except Exception as e:
        _log_deploy(driver['id'], f'[error] {e}')
        raise
    _log_deploy(driver['id'], f'[deploy] done, container={container.id[:12]}')
    return {'status': 'starting', 'container_id': container.id[:12], 'container_name': name}


def _stop_sync(driver_id: str, container_name_override: str = '') -> dict:
    import docker as docker_sdk
    try:
        client = _docker()
        name = _container_name(driver_id, container_name_override)
        container = client.containers.get(name)
        container.stop(timeout=5)
        return {'status': 'stopped'}
    except docker_sdk.errors.NotFound:
        return {'status': 'already_stopped'}
    except Exception as e:
        raise RuntimeError(str(e))


def _remove_sync(driver_id: str, container_name_override: str = '') -> dict:
    """Stop and remove container."""
    import docker as docker_sdk
    try:
        client = _docker()
        name = _container_name(driver_id, container_name_override)
        container = client.containers.get(name)
        container.remove(force=True)
        return {'status': 'removed'}
    except docker_sdk.errors.NotFound:
        return {'status': 'not_found'}
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

    from api.registry import CATEGORIES
    all_items = [item for c in CATEGORIES for item in catalog.get(c, [])]

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
            # Backfill provider/model on manifests written before these were stored.
            # The id alone can't be split back apart ('x-humanoid-tianyi2.0' — the
            # provider itself contains a hyphen), and 适用机型 is derived from model.
            if category == 'driver':
                if item.get('provider'):
                    existing['provider'] = item['provider']
                if item.get('model'):
                    existing['model'] = item['model']
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
            if category == 'driver':
                new_entry['provider'] = item.get('provider', '')
                new_entry['model'] = item.get('model', '')
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
            status_info = await _run_in_executor(_get_status_sync, d['id'], d.get('container_name', ''))
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
    from api.registry import (
        _build_catalog_sync, _current_channel, cache_key, _cache as _registry_cache,
    )
    channel = _current_channel()
    loop = asyncio.get_event_loop()
    try:
        catalog = await loop.run_in_executor(None, _build_catalog_sync, channel)
    except Exception as e:
        return {'code': 500, 'message': str(e)}

    # Update registry cache with fresh data so next GET /registry/catalog is immediate.
    # Must use cache_key(): the key includes the host arch facets, not just the channel.
    _registry_cache[cache_key(channel)] = {'data': catalog, 'ts': __import__('time').time()}

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
        _log_deploy(driver_id, f'[error] {e}')
        return {'code': 500, 'message': str(e)}

    # Persist updated image and container_name into manifest
    if not result.get('skipped'):
        import time as _time
        manifest = _load_manifest()
        for d in manifest:
            if d.get('id') == driver_id:
                d['image'] = driver['image']
                if result.get('container_name'):
                    d['container_name'] = result['container_name']
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
    manifest = _load_manifest()
    entry = next((d for d in manifest if d['id'] == driver_id), None)
    cn = entry.get('container_name', '') if entry else ''
    try:
        result = await _run_in_executor(_stop_sync, driver_id, cn)
        return {'code': 200, 'data': result}
    except Exception as e:
        return {'code': 500, 'message': str(e)}


@router.post('/{driver_id}/remove')
async def driver_remove(driver_id: str):
    """Stop + remove container, clear last_deploy from manifest."""
    manifest = _load_manifest()
    entry = next((d for d in manifest if d.get('id') == driver_id), None)
    cn = entry.get('container_name', '') if entry else ''
    try:
        result = await _run_in_executor(_remove_sync, driver_id, cn)
    except Exception as e:
        return {'code': 500, 'message': str(e)}
    manifest = _load_manifest()
    entry = next((d for d in manifest if d.get('id') == driver_id), None)
    if entry:
        entry.pop('last_deploy', None)
        _save_manifest(manifest)
    return {'code': 200, 'data': result}


@router.get('/{driver_id}/status')
async def driver_status(driver_id: str):
    manifest = _load_manifest()
    entry = next((d for d in manifest if d['id'] == driver_id), None)
    cn = entry.get('container_name', '') if entry else ''
    status = await _run_in_executor(_get_status_sync, driver_id, cn)
    return {'code': 200, 'data': status}

