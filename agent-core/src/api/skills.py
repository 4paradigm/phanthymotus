"""
api/skills.py — 技能管理 API。

提供技能的安装、卸载、列表查询接口。
"""

import fastapi

import sys
import config

# 直接引用 event.skills 模块（而非 event.__init__ 中的 Tools 实例）
import event.skills
_skills_mod = sys.modules['event.skills']

router = fastapi.APIRouter(prefix='/skills', tags=['skills'])


def _get_rc_url() -> str:
    """获取 Resource Center URL。"""
    services = config.main.get('services', {})
    rc = services.get('resource_center', None)
    if rc and rc.get('url'):
        return rc['url'].rstrip('/')
    return 'https://motus.phanthy.com'


@router.get('')
async def list_skills():
    """列出已安装技能及其激活状态。"""
    installed = _skills_mod.installed_skills()
    result = []
    for s in installed:
        result.append({
            'slug': s['slug'],
            'name': s['name'],
            'icon': s.get('icon'),
            'oneLiner': s['oneLiner'],
            'description': s.get('description', ''),
            'category': s.get('category', ''),
            'version': s.get('version', ''),
            'author': s.get('author', ''),
            'requiredTools': s.get('requiredTools', []),
            'active': s.get('active', False),
            'installedAt': s.get('installedAt', ''),
        })
    return {'code': 200, 'data': result}


@router.post('/install')
async def install_skill(body: dict = fastapi.Body(...)):
    """从 Resource Center 安装技能。"""
    slug = body.get('slug', '').strip()
    if not slug:
        return {'code': 422, 'error': 'slug is required'}

    # 检查是否已安装
    installed = _skills_mod.installed_skills()
    if any(s['slug'] == slug for s in installed):
        return {'code': 409, 'error': f'技能 "{slug}" 已安装'}

    # 从 Resource Center 获取技能定义
    rc_url = _get_rc_url()
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(f'{rc_url}/api/skills/{slug}')
            if resp.status_code != 200:
                return {'code': 404, 'error': f'技能 "{slug}" 未在 Resource Center 找到 (HTTP {resp.status_code})'}
            data = resp.json()
            if not data.get('ok'):
                return {'code': 404, 'error': data.get('error', '获取失败')}
    except ImportError:
        # httpx 未安装时 fallback 到 aiohttp 或 urllib
        import urllib.request, json as _json
        try:
            req = urllib.request.Request(f'{rc_url}/api/skills/{slug}')
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read())
                if not data.get('ok'):
                    return {'code': 404, 'error': data.get('error', '获取失败')}
        except Exception as e:
            return {'code': 502, 'error': f'无法连接 Resource Center: {e}'}
    except Exception as e:
        return {'code': 502, 'error': f'无法连接 Resource Center: {e}'}

    skill_data = data['data']

    # 存储到 ConfigDB
    import datetime
    new_skill = {
        'slug': skill_data['slug'],
        'name': skill_data['name'],
        'description': skill_data.get('description', ''),
        'icon': skill_data.get('icon'),
        'oneLiner': skill_data['oneLiner'],
        'instruction': skill_data['instruction'],
        'category': skill_data.get('category', ''),
        'version': skill_data.get('version', '1.0.0'),
        'requiredTools': skill_data.get('requiredTools', []),
        'configSchema': skill_data.get('configSchema'),
        'author': skill_data.get('author', {}).get('name', ''),
        'installedAt': datetime.datetime.now().isoformat(),
        'active': True,
    }

    skills_cfg = config.main.get('skills', {'installed': []})
    skills_cfg['installed'].append(new_skill)
    config.main['skills'] = skills_cfg

    return {'code': 200, 'data': {'slug': slug, 'name': new_skill['name']}}


@router.post('/uninstall')
async def uninstall_skill(body: dict = fastapi.Body(...)):
    """卸载已安装的技能。"""
    slug = body.get('slug', '').strip()
    if not slug:
        return {'code': 422, 'error': 'slug is required'}

    skills_cfg = config.main.get('skills', {'installed': []})
    before_count = len(skills_cfg['installed'])
    skills_cfg['installed'] = [s for s in skills_cfg['installed'] if s['slug'] != slug]

    if len(skills_cfg['installed']) == before_count:
        return {'code': 404, 'error': f'技能 "{slug}" 未安装'}

    config.main['skills'] = skills_cfg

    # 同时停用
    _skills_mod.active_skills.discard(slug)

    return {'code': 200, 'data': {'slug': slug}}


@router.get('/active')
async def list_active():
    """列出当前 LLM 已加载指令的技能 slugs。"""
    return {'code': 200, 'data': list(_skills_mod._runtime_activated)}


@router.get('/{slug}')
async def get_skill_detail(slug: str):
    """获取技能完整详情（含 instruction）。"""
    skill = next((s for s in _skills_mod.installed_skills() if s['slug'] == slug), None)
    if not skill:
        return {'code': 404, 'error': f'技能 "{slug}" 未安装'}
    return {'code': 200, 'data': {
        **skill,
        'active': skill.get('active', False),
    }}


@router.post('/update')
async def update_skill(body: dict = fastapi.Body(...)):
    """更新已安装技能的字段（本地保存）。"""
    slug = body.get('slug', '').strip()
    if not slug:
        return {'code': 422, 'error': 'slug is required'}

    skills_cfg = config.main.get('skills', {'installed': []})
    skill = next((s for s in skills_cfg['installed'] if s['slug'] == slug), None)
    if not skill:
        return {'code': 404, 'error': f'技能 "{slug}" 未安装'}

    # 允许更新的字段
    editable = ('name', 'oneLiner', 'description', 'instruction', 'category',
                'icon', 'requiredTools', 'configSchema', 'version')
    for key in editable:
        if key in body:
            skill[key] = body[key]

    config.main['skills'] = skills_cfg
    return {'code': 200, 'data': {'slug': slug}}


@router.post('/publish')
async def publish_skill(body: dict = fastapi.Body(...)):
    """将技能发布到 Resource Center（创建或更新版本）。"""
    slug = body.get('slug', '').strip()
    if not slug:
        return {'code': 422, 'error': 'slug is required'}

    skill = next((s for s in _skills_mod.installed_skills() if s['slug'] == slug), None)
    if not skill:
        return {'code': 404, 'error': f'技能 "{slug}" 未安装'}

    # 版本处理
    version = body.get('version', '').strip()
    if not version:
        # 自动递增 patch 版本
        parts = skill.get('version', '1.0.0').split('.')
        parts[-1] = str(int(parts[-1]) + 1)
        version = '.'.join(parts)

    rc_url = _get_rc_url()
    api_key = config.main.get('services', {}).get('resource_center', {}).get('api_key', '')

    payload = {
        'slug': skill['slug'],
        'name': skill['name'],
        'description': skill.get('description', ''),
        'oneLiner': skill['oneLiner'],
        'instruction': skill.get('instruction', ''),
        'category': skill.get('category', 'utility'),
        'icon': skill.get('icon', ''),
        'version': version,
        'requiredTools': skill.get('requiredTools', []),
        'configSchema': skill.get('configSchema'),
    }

    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['X-API-Key'] = api_key

    try:
        import httpx
        async with httpx.AsyncClient(timeout=15) as http:
            resp = await http.post(f'{rc_url}/api/skills/mine', json=payload, headers=headers)
            data = resp.json()
            if resp.status_code in (200, 201):
                # 更新本地版本号
                skill['version'] = version
                skills_cfg = config.main.get('skills', {'installed': []})
                config.main['skills'] = skills_cfg
                return {'code': 200, 'data': {'slug': slug, 'version': version}}
            else:
                return {'code': resp.status_code, 'error': data.get('error', '发布失败')}
    except Exception as e:
        return {'code': 502, 'error': f'无法连接 Resource Center: {e}'}


@router.post('/activate')
async def activate(body: dict = fastapi.Body(...)):
    """手动激活技能（测试用）。"""
    slug = body.get('slug', '').strip()
    skill = next((s for s in _skills_mod.installed_skills() if s['slug'] == slug), None)
    if not skill:
        return {'code': 404, 'error': f'技能 "{slug}" 未安装'}
    _skills_mod.active_skills.add(slug)
    return {'code': 200, 'data': {'slug': slug, 'active': True}}


@router.post('/deactivate')
async def deactivate(body: dict = fastapi.Body(...)):
    """手动停用技能（测试用）。"""
    slug = body.get('slug', '').strip()
    _skills_mod.active_skills.discard(slug)
    return {'code': 200, 'data': {'slug': slug, 'active': False}}
