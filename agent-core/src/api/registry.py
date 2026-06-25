"""
registry.py — 从 resource-center (motus.phanthy.com) 获取已审核的镜像列表。

GET /registry/catalog  → 返回按 category 分组的镜像列表及 release.* tags
"""

import json
import os
import time
import urllib.request

import fastapi

router = fastapi.APIRouter(prefix='/registry', tags=['registry'])

RESOURCE_CENTER_URL = os.environ.get('RESOURCE_CENTER_URL', 'https://motus.phanthy.com')

# ── Simple in-memory cache ──────────────────────────────────────────────────

_cache: dict = {'data': None, 'ts': 0.0}
_CACHE_TTL = 300  # 5 minutes


# ── Catalog fetch ─────────────────────────────────────────────────────────

def _build_catalog_sync() -> dict:
    url = f'{RESOURCE_CENTER_URL}/api/images'
    print(f'[registry] fetching catalog from resource-center: {url}')

    try:
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=15) as r:
            payload = json.load(r)
    except Exception as e:
        print(f'[registry] resource-center fetch failed: {e}')
        return {'core': [], 'driver': [], 'perception': [], 'inspection': []}

    if not payload.get('ok') or not isinstance(payload.get('data'), list):
        print(f'[registry] unexpected response: {str(payload)[:200]}')
        return {'core': [], 'driver': [], 'perception': [], 'inspection': []}

    result: dict = {'core': [], 'driver': [], 'perception': [], 'inspection': []}

    for item in payload['data']:
        category = item.get('category', '')
        tags_raw = item.get('tags', [])

        # Build full_repo from the first imageRef (strip the :tag part)
        full_repo = ''
        if tags_raw:
            first_ref = tags_raw[0].get('imageRef', '')
            full_repo = first_ref.rsplit(':', 1)[0] if ':' in first_ref else first_ref

        tags = []
        for t in sorted(tags_raw, key=lambda x: x.get('publishedAt', ''), reverse=True)[:20]:
            published = t.get('publishedAt', '') or ''
            # Format publishedAt ISO string to UTC+8 readable date
            created = ''
            if published:
                try:
                    from datetime import datetime, timezone, timedelta
                    _tz8 = timezone(timedelta(hours=8))
                    # "2026-05-31T14:22:00.000Z" or "2026-05-31T14:22:00Z"
                    dt = datetime.fromisoformat(published.replace('Z', '+00:00'))
                    created = dt.astimezone(_tz8).strftime('%Y-%m-%d %H:%M')
                except Exception:
                    created = published[:16].replace('T', ' ')
            tags.append({
                'tag': t.get('tag', ''),
                'created': created,
                'size': '',
                'imageRef': t.get('imageRef', ''),
            })

        entry = {
            'full_repo': full_repo,
            'image': item.get('registryImage', ''),
            'name': item.get('name', item.get('registryImage', '')),
            'description': item.get('description', ''),
            'port': item.get('port'),
            'tags': tags,
        }

        if category == 'driver':
            entry['category'] = 'driver'
            entry['provider'] = item.get('hardware_provider', '')
            entry['model'] = item.get('hardware_model', '')
            result['driver'].append(entry)
        elif category == 'core':
            entry['category'] = 'core'
            result['core'].append(entry)
        elif category == 'perception':
            entry['category'] = 'perception'
            result['perception'].append(entry)
        elif category == 'inspection':
            entry['category'] = 'inspection'
            result['inspection'].append(entry)
        else:
            print(f'[registry] unknown category {category!r} for {item.get("registryImage")}')

    print(
        f'[registry] catalog: core={len(result["core"])} driver={len(result["driver"])} '
        f'perception={len(result["perception"])} inspection={len(result["inspection"])}'
    )
    return result


# ── FastAPI endpoint ──────────────────────────────────────────────────────

@router.get('/catalog')
async def registry_catalog(refresh: bool = False):
    now = time.time()
    if not refresh and _cache['data'] and (now - _cache['ts']) < _CACHE_TTL:
        return {'code': 200, 'data': _cache['data'], 'cached': True}

    import asyncio
    loop = asyncio.get_event_loop()
    try:
        data = await loop.run_in_executor(None, _build_catalog_sync)
    except Exception as e:
        return {'code': 500, 'message': str(e), 'data': {'core': [], 'hardware': [], 'perception': [], 'inspection': []}}

    _cache['data'] = data
    _cache['ts'] = now
    return {'code': 200, 'data': data, 'cached': False}
