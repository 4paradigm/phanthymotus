"""
drivers_async.py — Async deployment with progress streaming.

Provides enhanced deployment with:
- Pre-flight checks (disk, network, registry)
- Real-time progress via WebSocket
- Better error messages with recovery suggestions
- Pull progress tracking (%, speed, ETA)
"""

import asyncio
import os
import subprocess
import tarfile
import io
import time
import shutil
from typing import Optional

import docker as docker_sdk

from api.deploy_stream import DeployProgress
from api.preflight import run_preflight_checks
from api.drivers import (
    _container_name, _log_deploy, _clear_deploy_log,
    _explain_pull_error, _merge_service_into_compose,
    LOG_MAX_SIZE, LOG_MAX_FILE,
)


def _docker():
    return docker_sdk.from_env()


async def _deploy_with_progress(driver: dict) -> dict:
    """Deploy a driver with real-time progress streaming.

    This is an async version that integrates preflight checks and progress updates.
    Falls back to the sync version in drivers.py for legacy compatibility.
    """
    driver_id = driver['id']
    name = _container_name(driver_id, driver.get('container_name', ''))
    target_image = driver['image']

    async with DeployProgress(driver_id, target_image) as progress:
        # Check if already running with same image
        try:
            client = _docker()
            existing = client.containers.get(name)
            if existing.status == 'running':
                running_image = existing.attrs.get('Config', {}).get('Image', '')
                if running_image == target_image:
                    await progress.done('已经在运行相同版本，跳过部署')
                    return {'status': 'running', 'message': 'already running with same image', 'skipped': True}
        except docker_sdk.errors.NotFound:
            pass

        # Run preflight checks
        await progress.check('preflight', '执行部署前检查…')
        checks_result = await asyncio.get_event_loop().run_in_executor(
            None, run_preflight_checks, target_image, name
        )

        # Report each check result
        for check_name, check_data in checks_result['checks'].items():
            status = check_data.get('status', 'unknown')
            message = check_data.get('message', '')
            if status == 'pass':
                await progress.check(check_name, f'✓ {message}', status='pass')
            elif status == 'warning':
                await progress.check(check_name, f'⚠ {message}', status='warning')
                if check_data.get('suggestion'):
                    await progress.update('preflight', f'  建议: {check_data["suggestion"]}')
            elif status == 'fail':
                await progress.check(check_name, f'✗ {message}', status='fail')
                if check_data.get('suggestion'):
                    await progress.error(check_name, message, suggestion=check_data['suggestion'])

        # If critical failures, stop here
        if not checks_result['can_proceed']:
            await progress.error('preflight', '预检查失败，无法继续部署')
            return {
                'status': 'error',
                'error': '预检查失败',
                'checks': checks_result,
            }

        # If warnings, report them but continue
        if checks_result['overall_status'] == 'warning':
            await progress.update('preflight', '⚠ 检查通过但有警告，继续部署…')

        # Pull image with progress tracking
        await progress.update('pull', f'开始拉取镜像 {target_image}…', percent=0)
        # Pull image (this may take several minutes for large images)
        _log_deploy(driver_id, f'[pull] {target_image}')
        await progress.update('pull', '正在拉取镜像，请稍候...', percent=10)

        pull_result = await _pull_image_with_progress(driver_id, target_image, progress)

        # Debug: check result type
        if not isinstance(pull_result, dict):
            await progress.error('pull', f'拉取镜像返回值类型错误: {type(pull_result).__name__} = {pull_result}')
            return {'status': 'error', 'error': f'Internal error: unexpected result type {type(pull_result).__name__}'}

        if not pull_result.get('success'):
            detail = _explain_pull_error(pull_result.get('error', 'Unknown error'))
            _log_deploy(driver_id, f'[pull] failed: {detail}')
            await progress.error('pull', f'镜像拉取失败: {detail}')
            return {'status': 'error', 'error': f'镜像拉取失败: {detail}'}

        await progress.update('pull', '镜像拉取完成', percent=100)

        # Extract and merge service.yml
        await progress.update('compose', '准备容器配置…')
        compose_result = await _prepare_compose(driver, target_image, name, progress)

        # Debug: check result type
        if not isinstance(compose_result, dict):
            await progress.error('compose', f'准备配置返回值类型错误: {type(compose_result).__name__} = {compose_result}')
            return {'status': 'error', 'error': f'Internal error: unexpected result type {type(compose_result).__name__}'}

        if not compose_result.get('success'):
            return {'status': 'error', 'error': compose_result.get('error', 'Unknown error')}

        # Start container
        await progress.update('start', '启动容器…')
        start_result = await _start_container(driver_id, compose_result, progress)

        # Debug: check result type
        if not isinstance(start_result, dict):
            await progress.error('start', f'启动容器返回值类型错误: {type(start_result).__name__} = {start_result}')
            return {'status': 'error', 'error': f'Internal error: unexpected result type {type(start_result).__name__}'}

        if not start_result.get('success'):
            return {'status': 'error', 'error': start_result.get('error', 'Unknown error')}

        await progress.done(f'部署完成：{name}')
        return {
            'status': 'starting',
            'service': start_result.get('service'),
            'container_name': start_result.get('container_name'),
        }


async def _pull_image_with_progress(driver_id: str, image: str, progress: DeployProgress) -> dict:
    """Pull image and report progress via WebSocket."""
    loop = asyncio.get_event_loop()

    def schedule_update(percent, speed_mbps, status_msg):
        """Thread-safe progress update."""
        async def do_update():
            await progress.update(
                'pull', f'拉取进度: {status_msg}',
                percent=percent, speed=f'{speed_mbps:.1f} MB/s'
            )
        loop.call_soon_threadsafe(lambda: asyncio.create_task(do_update()))

    def schedule_layer_update(layer_id, status, progress_str=''):
        """Thread-safe layer status notification."""
        async def do_notify():
            await progress._push({
                'type': 'layer',
                'layer_id': layer_id,
                'status': status,
                'progress': progress_str,
                'message': f'{layer_id}: {status}',
            })
        loop.call_soon_threadsafe(lambda: asyncio.create_task(do_notify()))

    def _pull():
        client = _docker()
        pull_error = ''

        # Track progress by layer
        layers = {}
        last_update = time.time()
        start_time = time.time()

        # Track last logged status per layer to avoid spam
        layer_last_status = {}

        for line in client.api.pull(image, stream=True, decode=True):
            # Check for errors
            if line.get('error') or line.get('errorDetail'):
                pull_error = (
                    (line.get('errorDetail') or {}).get('message')
                    or line.get('error')
                    or 'unknown pull error'
                )
                _log_deploy(driver_id, f'[pull] error: {pull_error}')
                continue

            status = line.get('status', '')
            layer_id = line.get('id', '')
            progress_detail = line.get('progressDetail', {})
            progress_str = line.get('progress', '')

            # Push layer status changes to WebSocket (for logging)
            if layer_id and status:
                last_status = layer_last_status.get(layer_id, '')

                # Only push when status changes or is significant
                is_significant = (
                    status != last_status and (
                        'complete' in status.lower() or
                        'exists' in status.lower() or
                        'extracting' in status.lower() or
                        'verifying' in status.lower() or
                        'waiting' in status.lower() or
                        'pulling' in status.lower()
                    )
                )

                if is_significant:
                    schedule_layer_update(layer_id, status, progress_str)
                    layer_last_status[layer_id] = status

            # Track layer progress
            if layer_id and progress_detail:
                current = progress_detail.get('current', 0)
                total = progress_detail.get('total', 0)
                if total > 0:
                    layers[layer_id] = {'current': current, 'total': total, 'status': status}

            # Log to deploy log
            if status:
                msg = f'  {layer_id} {status}' if layer_id else f'  {status}'
                if line.get('progress'):
                    msg += f' {line["progress"]}'
                _log_deploy(driver_id, msg)

            # Calculate overall progress (throttle updates to every 0.5s)
            now = time.time()
            if now - last_update > 0.5 and layers:
                total_current = sum(l['current'] for l in layers.values())
                total_size = sum(l['total'] for l in layers.values())
                if total_size > 0:
                    percent = (total_current / total_size) * 100
                    # Estimate speed (rough approximation)
                    elapsed = now - start_time
                    speed_mbps = (total_current / (1 << 20)) / max(0.1, elapsed)

                    # Count layers by status
                    downloading = sum(1 for l in layers.values() if l.get('status') == 'Downloading')
                    extracting = sum(1 for l in layers.values() if l.get('status') == 'Extracting')

                    # Build status message
                    status_parts = []
                    if downloading > 0:
                        status_parts.append(f'{downloading} 层下载中')
                    if extracting > 0:
                        status_parts.append(f'{extracting} 层提取中')
                    status_msg = ', '.join(status_parts) if status_parts else f'{len(layers)} 层'

                    # Push progress update to WebSocket (thread-safe)
                    schedule_update(percent, speed_mbps, status_msg)

                    # Also log it
                    _log_deploy(driver_id, f'[pull] {percent:.1f}% ({len(layers)} layers, {speed_mbps:.1f} MB/s)')
                last_update = now

        return {'success': not pull_error, 'error': pull_error}

    try:
        result = await loop.run_in_executor(None, _pull)
        return result
    except Exception as e:
        return {'success': False, 'error': str(e)}


async def _prepare_compose(driver: dict, target_image: str, name: str, progress: DeployProgress) -> dict:
    """Extract service.yml and merge into host compose file."""
    loop = asyncio.get_event_loop()

    def _extract_and_merge():
        import yaml

        client = _docker()

        # Create temporary container to extract service.yml
        try:
            container = client.containers.create(target_image)
        except Exception as e:
            detail = _explain_pull_error(str(e))
            _log_deploy(driver['id'], f'[deploy] image unusable after pull: {detail}')
            return {
                'success': False,
                'error': f'镜像拉取后仍不可用: {detail}',
            }

        try:
            bits, _ = container.get_archive('/deploy/service.yml')
            tar_bytes = b''.join(bits)
            tf = tarfile.open(fileobj=io.BytesIO(tar_bytes))
            service_content = tf.extractfile('service.yml').read().decode()
        except Exception:
            # No service.yml - would need legacy fallback
            try:
                container.remove(force=True)
            except Exception:
                pass
            return {
                'success': False,
                'error': '镜像中未找到 /deploy/service.yml，需要使用旧版部署方式',
            }
        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass

        # Parse and merge
        service_def = yaml.safe_load(service_content)
        if not service_def or not isinstance(service_def, dict):
            return {
                'success': False,
                'error': 'service.yml 格式无效',
            }

        service_name = list(service_def.keys())[0]
        service_def[service_name]['image'] = target_image
        service_def[service_name].setdefault('logging', {
            'driver': 'local',
            'options': {'max-size': LOG_MAX_SIZE, 'max-file': LOG_MAX_FILE},
        })

        # Merge into host compose
        compose_dir = os.environ.get('COMPOSE_DIR', '/opt/phanthy-motus')
        compose_file = os.path.join(compose_dir, 'docker-compose.yml')
        os.makedirs(compose_dir, exist_ok=True)

        ok, err = False, ''
        try:
            ok, err = _merge_service_into_compose(compose_file, service_def)
        except TimeoutError as e:
            err = str(e)

        if not ok:
            _log_deploy(driver['id'], f'[compose] {err}')
            return {'success': False, 'error': err}

        # Remove old container
        try:
            old = client.containers.get(name)
            old.remove(force=True)
        except docker_sdk.errors.NotFound:
            pass

        svc_container_name = service_def[service_name].get('container_name', '')
        if svc_container_name and svc_container_name != name:
            try:
                old = client.containers.get(svc_container_name)
                old.remove(force=True)
            except docker_sdk.errors.NotFound:
                pass

        return {
            'success': True,
            'service_name': service_name,
            'container_name': svc_container_name,
            'compose_file': compose_file,
        }

    return await loop.run_in_executor(None, _extract_and_merge)


async def _start_container(driver_id: str, compose_result: dict, progress: DeployProgress) -> dict:
    """Start container via docker compose."""
    loop = asyncio.get_event_loop()

    def _run_compose():
        service_name = compose_result['service_name']
        compose_file = compose_result['compose_file']

        _log_deploy(driver_id, f'[compose] up -d {service_name}')
        result = subprocess.run(
            ['docker', 'compose', '-f', compose_file, 'up', '-d', '--no-deps', '--force-recreate', service_name],
            capture_output=True, text=True,
        )

        if result.stdout:
            _log_deploy(driver_id, result.stdout.strip())
        if result.stderr:
            _log_deploy(driver_id, result.stderr.strip())

        if result.returncode != 0:
            _log_deploy(driver_id, f'[compose] exit code {result.returncode}')
            return {
                'success': False,
                'error': f'compose up failed (rc={result.returncode})',
            }

        _log_deploy(driver_id, '[deploy] done')
        return {
            'success': True,
            'service': service_name,
            'container_name': compose_result['container_name'],
        }

    return await loop.run_in_executor(None, _run_compose)
