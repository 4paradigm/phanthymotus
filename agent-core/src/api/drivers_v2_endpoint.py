"""
drivers_v2_endpoint.py — Enhanced deployment endpoint to add to drivers.py

This module provides the /deploy-v2 endpoint with progress streaming.
Import and register this router in start.py alongside drivers.router.
"""

import time
import fastapi

from api.drivers import _load_manifest, _save_manifest, _deploy_sync, _run_in_executor, _log_deploy

router = fastapi.APIRouter(prefix='/drivers', tags=['drivers'])


@router.post('/{driver_id}/deploy-v2')
async def driver_deploy_v2(driver_id: str, body: dict = fastapi.Body(default={})):
    """Enhanced deployment with progress streaming and preflight checks.

    Use WebSocket /ws/deploy/{driver_id} to receive real-time progress.
    Falls back to legacy sync deployment if streaming fails.
    """
    manifest = _load_manifest()
    driver = next((d for d in manifest if d['id'] == driver_id), None)
    if not driver:
        raise fastapi.HTTPException(status_code=404, detail='Driver not found in manifest')

    # Allow image override (same as regular deploy)
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

    # Use async deployment with progress
    try:
        from api.drivers_async import _deploy_with_progress
        result = await _deploy_with_progress(driver)
    except Exception as e:
        _log_deploy(driver_id, f'[error] async deploy failed, falling back to sync: {e}')
        # Fallback to sync deployment
        try:
            result = await _run_in_executor(_deploy_sync, driver)
        except Exception as e2:
            _log_deploy(driver_id, f'[error] {e2}')
            return {'code': 500, 'message': str(e2)}

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

    return {'code': 200, 'data': result}


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
