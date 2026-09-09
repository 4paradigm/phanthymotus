"""
drivers_v2_endpoint.py — Enhanced deployment endpoint to add to drivers.py

This module provides the /deploy-v2 endpoint with progress streaming.
Import and register this router in start.py alongside drivers.router.
"""

import time
import asyncio
import fastapi

from api.drivers import _load_manifest, _save_manifest, _deploy_sync, _run_in_executor, _log_deploy

router = fastapi.APIRouter(prefix='/drivers', tags=['drivers'])

# Track background deployment tasks
_background_tasks = {}


async def _deploy_background(driver_id: str, driver: dict):
    """Run deployment in background and update manifest when done."""
    try:
        from api.drivers_async import _deploy_with_progress
        result = await _deploy_with_progress(driver)

        # Persist updated image and container_name into manifest
        if not result.get('skipped'):
            manifest = _load_manifest()
            for d in manifest:
                if d.get('id') == driver_id:
                    d['image'] = driver['image']
                    if result.get('container_name'):
                        d['container_name'] = result['container_name']
                    d['last_deploy'] = {
                        'image':  driver['image'],
                        'ts':     int(time.time()),
                        'status': result.get('status', ''),
                    }
                    break
            _save_manifest(manifest)
    except Exception as e:
        _log_deploy(driver_id, f'[error] background deploy failed: {e}')
    finally:
        # Remove from tracking
        if driver_id in _background_tasks:
            del _background_tasks[driver_id]


@router.post('/{driver_id}/deploy-v2')
async def driver_deploy_v2(driver_id: str, body: dict = fastapi.Body(default={})):
    """Enhanced deployment with progress streaming and preflight checks.

    Use WebSocket /ws/deploy/{driver_id} to receive real-time progress.

    This endpoint starts deployment in background and returns immediately.
    The deployment continues even if the client disconnects.
    """
    manifest = _load_manifest()
    driver = next((d for d in manifest if d['id'] == driver_id), None)
    if not driver:
        raise fastapi.HTTPException(status_code=404, detail='Driver not found in manifest')

    # Check if already deploying
    if driver_id in _background_tasks:
        return {
            'code': 200,
            'data': {
                'status': 'deploying',
                'message': 'Deployment already in progress',
            }
        }

    # Update image if provided
    new_image = body.get('image')
    if new_image:
        driver['image'] = new_image

    _log_deploy(driver_id, f'[deploy-v2] starting background deployment: {driver["image"]}')

    # Start deployment in background (fire and forget)
    task = asyncio.create_task(_deploy_background(driver_id, driver))
    _background_tasks[driver_id] = task

    # Return immediately - client connects to WebSocket for progress
    return {
        'code': 200,
        'data': {
            'status': 'started',
            'message': 'Deployment started in background',
            'driver_id': driver_id,
        }
    }


@router.get('/{driver_id}/deploy-status')
async def driver_deploy_status(driver_id: str):
    """Check if a deployment is currently running for this driver."""
    is_deploying = driver_id in _background_tasks
    return {
        'code': 200,
        'data': {
            'deploying': is_deploying,
            'driver_id': driver_id,
        }
    }


@router.get('/{driver_id}/preflight')
async def driver_preflight(driver_id: str):
    """Run preflight checks without deploying.

    Returns disk space, network status, registry auth, etc.
    """
    from api.preflight import run_preflight_checks
    from api.drivers import _container_name

    manifest = _load_manifest()
    driver = next((d for d in manifest if d['id'] == driver_id), None)
    if not driver:
        raise fastapi.HTTPException(status_code=404, detail='Driver not found in manifest')

    image = driver['image']
    name = _container_name(driver_id, driver.get('container_name', ''))

    import asyncio
    loop = asyncio.get_event_loop()
    checks = await loop.run_in_executor(None, run_preflight_checks, image, name)

    return {'code': 200, 'data': checks}
