"""
registry.py — 从 resource-center (motus.phanthy.com) 获取已审核的镜像列表。

GET /registry/catalog  → 返回按 category 分组的镜像列表及 release.* tags
"""

import json
import os
import time
import urllib.parse
import urllib.request

import fastapi

router = fastapi.APIRouter(prefix='/registry', tags=['registry'])

RESOURCE_CENTER_URL = os.environ.get('RESOURCE_CENTER_URL', 'https://motus.phanthy.com')

# Service categories, in display order. A catalog item whose category is not in
# this tuple is dropped — adding a new kind of service means adding it here.
CATEGORIES = ('core', 'perception', 'actucore', 'driver')


def empty_catalog() -> dict:
    return {c: [] for c in CATEGORIES}


# ── Simple in-memory cache ──────────────────────────────────────────────────
# Keyed by channel + host arch facets: without this, switching channels while a
# stale cache entry from the previous channel is still within TTL would serve
# mismatched data to any caller that doesn't pass refresh=true. The facets are in
# the key for the same reason — they're constant per process today, but the
# ACC_ARCH / CPU_ARCH overrides make that a convention rather than a fact.
_cache: dict[str, dict] = {}
_CACHE_TTL = 300  # 5 minutes

# What the last successful fetch reported about arch filtering. `applied` is False when
# resource-center is an older build that ignored the params and returned everything —
# the dashboard must not then claim the list was filtered for this host. The hidden_*
# counts let it tell the user exactly how many variants the website has that this
# machine can't run.
_last_filter = {'applied': False, 'hidden_tags': 0, 'hidden_images': 0}


def _current_channel() -> str:
    try:
        import config as _cfg
        return _cfg.main.get('core', {}).get('update_channel', 'ga')
    except Exception:
        return 'ga'


def cache_key(channel: str) -> str:
    """Cache key for a catalog fetch. Callers that write into `_cache` must use this."""
    import hostarch
    return f'{channel}|{hostarch.acc_arch()}|{hostarch.cpu_arch()}'


# ── Catalog fetch ─────────────────────────────────────────────────────────

def _build_catalog_sync(channel: str) -> dict:
    # Host arch facets are a property of this process, not of the call site, so they
    # are resolved here rather than threaded through all three callers. resource-center
    # drops images that can't run on this host (e.g. a jp5.11 perception on a JetPack 6
    # robot); images with no arch marker are arch-agnostic and always returned.
    import hostarch
    query = {
        'channel': channel,
        'acc_arch': hostarch.acc_arch(),
        'cpu_arch': hostarch.cpu_arch(),
    }
    url = f'{RESOURCE_CENTER_URL}/api/images?' + urllib.parse.urlencode(query)

    print(f'[registry] fetching catalog from resource-center: {url}')

    try:
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=15) as r:
            payload = json.load(r)
    except Exception as e:
        print(f'[registry] resource-center fetch failed: {e}')
        return empty_catalog()

    if not payload.get('ok') or not isinstance(payload.get('data'), list):
        print(f'[registry] unexpected response: {str(payload)[:200]}')
        return empty_catalog()

    # An arch-aware resource-center echoes the filter it applied.
    flt = payload.get('filter')
    if isinstance(flt, dict):
        _last_filter.update(
            applied=True,
            hidden_tags=flt.get('hidden_tags', 0) or 0,
            hidden_images=flt.get('hidden_images', 0) or 0,
        )
        print(f'[registry] arch filter applied: {flt.get("acc_arch")}/{flt.get("cpu_arch")} '
              f'— hid {_last_filter["hidden_tags"]} versions, '
              f'{_last_filter["hidden_images"]} components')
    else:
        _last_filter.update(applied=False, hidden_tags=0, hidden_images=0)
        print('[registry] resource-center did not echo a filter — catalog is unfiltered')

    result: dict = empty_catalog()

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
                'channel': t.get('channel', ''),
                # Which host this version needs. Absent from older resource-centers.
                'acc_arch': t.get('accArch'),
                'cpu_arch': t.get('cpuArch'),
            })

        entry = {
            'full_repo': full_repo,
            'image': item.get('registryImage', ''),
            'name': item.get('name', item.get('registryImage', '')),
            'description': item.get('description', ''),
            'port': item.get('port'),
            'tags': tags,
        }

        if category not in CATEGORIES:
            print(f'[registry] unknown category {category!r} for {item.get("registryImage")}')
            continue

        entry['category'] = category
        if category == 'driver':
            entry['provider'] = item.get('hardware_provider', '')
            entry['model'] = item.get('hardware_model', '')
        result[category].append(entry)

    print('[registry] catalog: ' + ' '.join(f'{c}={len(result[c])}' for c in CATEGORIES))
    return result


# ── FastAPI endpoint ──────────────────────────────────────────────────────

@router.get('/catalog')
async def registry_catalog(refresh: bool = False):
    import hostarch
    channel = _current_channel()
    key = cache_key(channel)
    now = time.time()
    cached = _cache.get(key)
    if not refresh and cached and (now - cached['ts']) < _CACHE_TTL:
        return {'code': 200, 'data': cached['data'], 'cached': True,
                'facets': hostarch.facets(), 'filter': dict(_last_filter)}

    import asyncio
    loop = asyncio.get_event_loop()
    try:
        data = await loop.run_in_executor(None, _build_catalog_sync, channel)
    except Exception as e:
        return {'code': 500, 'message': str(e), 'data': empty_catalog()}

    _cache[key] = {'data': data, 'ts': now}
    return {'code': 200, 'data': data, 'cached': False,
            'facets': hostarch.facets(), 'filter': dict(_last_filter)}
