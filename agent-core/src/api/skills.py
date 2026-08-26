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


async def fetch_rc_skill(slug: str, rc_token: str | None = None) -> dict:
    """从 Resource Center 拉取技能定义。

    返回 {'ok': True, 'data': {...}} 或 {'ok': False, 'error': '...'}。
    rc_token 用于让作者取到自己尚未上架的技能。
    """
    rc_url = _get_rc_url()
    headers = {}
    if rc_token:
        headers['Authorization'] = f'Bearer {rc_token}'
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(f'{rc_url}/api/skills/{slug}', headers=headers)
            if resp.status_code != 200:
                return {'ok': False,
                        'error': f'技能 "{slug}" 未在 Resource Center 找到 (HTTP {resp.status_code})'}
            data = resp.json()
            if not data.get('ok'):
                return {'ok': False, 'error': data.get('error', '获取失败')}
            return {'ok': True, 'data': data['data']}
    except ImportError:
        # httpx 未安装时 fallback 到 urllib
        import urllib.request, json as _json
        try:
            req = urllib.request.Request(f'{rc_url}/api/skills/{slug}')
            for k, v in headers.items():
                req.add_header(k, v)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read())
                if not data.get('ok'):
                    return {'ok': False, 'error': data.get('error', '获取失败')}
                return {'ok': True, 'data': data['data']}
        except Exception as e:
            return {'ok': False, 'error': f'无法连接 Resource Center: {e}'}
    except Exception as e:
        return {'ok': False, 'error': f'无法连接 Resource Center: {e}'}


async def install_from_rc(slug: str, rc_token: str | None = None) -> dict:
    """安装一个 Resource Center 技能到本地 ConfigDB。

    返回 {'ok': True, 'data': {...}} / {'ok': False, 'error': ...}。
    已安装时视为成功并回 already=True —— 载入解决方案时会批量调用，
    重复安装不该让整包失败。
    """
    installed = _skills_mod.installed_skills()
    if any(s['slug'] == slug for s in installed):
        return {'ok': True, 'already': True, 'data': {'slug': slug}}

    fetched = await fetch_rc_skill(slug, rc_token)
    if not fetched.get('ok'):
        return fetched
    skill_data = fetched['data']

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
        'author': (skill_data.get('author') or {}).get('name', ''),
        'installedAt': datetime.datetime.now().isoformat(),
        'active': True,
    }

    skills_cfg = config.main.get('skills', {'installed': []})
    skills_cfg['installed'].append(new_skill)
    config.main['skills'] = skills_cfg

    return {'ok': True, 'already': False, 'data': {'slug': slug, 'name': new_skill['name']}}



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
async def install_skill(request: fastapi.Request, body: dict = fastapi.Body(...)):
    """从 Resource Center 安装技能。"""
    slug = body.get('slug', '').strip()
    if not slug:
        return {'code': 422, 'error': 'slug is required'}

    # 检查是否已安装（install_from_rc 对已安装是幂等的，这里保留显式 409
    # 是因为手动安装时用户需要知道"已经装过了"）
    installed = _skills_mod.installed_skills()
    if any(s['slug'] == slug for s in installed):
        return {'code': 409, 'error': f'技能 "{slug}" 已安装'}

    # 传递 Resource Center token 以允许作者安装自己的未发布技能
    result = await install_from_rc(slug, request.headers.get('x-rc-token'))
    if not result.get('ok'):
        return {'code': 404, 'error': result.get('error', '安装失败')}
    return {'code': 200, 'data': result['data']}


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


# ── Resource Center proxy endpoints ─────────────────────────────────────────

def _get_rc_token(request: fastapi.Request) -> str | None:
    """从请求头 X-RC-Token 中获取 Resource Center bearer token。"""
    return request.headers.get('x-rc-token')


def _rc_error(status: int, error: str) -> str:
    """Resource Center 报错中文化。鉴权类的原文是 Unauthorized，用户看不懂。"""
    from api.account import normalize_rc_error
    return normalize_rc_error(status, error)


@router.post('/rc/login')
async def rc_login(body: dict = fastapi.Body(...)):
    """代理登录 Resource Center，返回 bearer token。

    实现在 api/account.py —— 这里保留端点是为了兼容仍在调用它的前端代码，
    新代码请直接用 /api/account/login。
    """
    from api.account import login as account_login, LoginRequest
    return await account_login(LoginRequest(
        identifier=body.get('identifier', ''),
        password=body.get('password', ''),
    ))


@router.get('/rc/mine')
async def rc_my_skills(request: fastapi.Request):
    """代理获取用户在 Resource Center 上的技能列表。"""
    token = _get_rc_token(request)
    if not token:
        return {'code': 401, 'error': '未登录 Resource Center'}

    rc_url = _get_rc_url()
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(f'{rc_url}/api/skills/mine', headers={
                'Authorization': f'Bearer {token}'
            })
            data = resp.json()
            if resp.status_code == 200:
                return {'code': 200, 'data': data.get('data', [])}
            return {'code': resp.status_code,
                    'error': _rc_error(resp.status_code, data.get('error', '获取失败'))}
    except Exception as e:
        return {'code': 502, 'error': f'无法连接 Resource Center: {e}'}


@router.post('/rc/mine')
async def rc_create_skill(request: fastapi.Request, body: dict = fastapi.Body(...)):
    """代理在 Resource Center 上创建/更新技能。"""
    token = _get_rc_token(request)
    if not token:
        return {'code': 401, 'error': '未登录 Resource Center'}

    rc_url = _get_rc_url()
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15) as http:
            resp = await http.post(f'{rc_url}/api/skills/mine', json=body, headers={
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
            })
            data = resp.json()
            if resp.status_code == 200 and data.get('ok'):
                return {'code': 200, 'data': data.get('data')}
            return {'code': resp.status_code,
                    'error': _rc_error(resp.status_code, data.get('error', '操作失败'))}
    except Exception as e:
        return {'code': 502, 'error': f'无法连接 Resource Center: {e}'}


@router.put('/rc/mine/{skill_id}')
async def rc_update_skill(skill_id: str, request: fastapi.Request, body: dict = fastapi.Body(...)):
    """代理更新 Resource Center 上的技能。"""
    token = _get_rc_token(request)
    if not token:
        return {'code': 401, 'error': '未登录 Resource Center'}

    rc_url = _get_rc_url()
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15) as http:
            resp = await http.put(f'{rc_url}/api/skills/mine/{skill_id}', json=body, headers={
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
            })
            data = resp.json()
            if resp.status_code == 200 and data.get('ok'):
                return {'code': 200, 'data': data.get('data')}
            return {'code': resp.status_code,
                    'error': _rc_error(resp.status_code, data.get('error', '更新失败'))}
    except Exception as e:
        return {'code': 502, 'error': f'无法连接 Resource Center: {e}'}


@router.delete('/rc/mine/{skill_id}')
async def rc_delete_skill(skill_id: str, request: fastapi.Request):
    """代理删除 Resource Center 上的技能。"""
    token = _get_rc_token(request)
    if not token:
        return {'code': 401, 'error': '未登录 Resource Center'}

    rc_url = _get_rc_url()
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.delete(f'{rc_url}/api/skills/mine/{skill_id}', headers={
                'Authorization': f'Bearer {token}'
            })
            data = resp.json()
            if resp.status_code == 200 and data.get('ok'):
                return {'code': 200, 'data': {'id': skill_id}}
            return {'code': resp.status_code,
                    'error': _rc_error(resp.status_code, data.get('error', '删除失败'))}
    except Exception as e:
        return {'code': 502, 'error': f'无法连接 Resource Center: {e}'}


@router.post('/rc/mine/{skill_id}/submit')
async def rc_submit_skill(skill_id: str, request: fastapi.Request):
    """代理提交技能送审。"""
    token = _get_rc_token(request)
    if not token:
        return {'code': 401, 'error': '未登录 Resource Center'}

    rc_url = _get_rc_url()
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.post(f'{rc_url}/api/skills/mine/{skill_id}/submit', headers={
                'Authorization': f'Bearer {token}'
            })
            data = resp.json()
            if resp.status_code == 200 and data.get('ok'):
                return {'code': 200, 'data': data.get('data')}
            return {'code': resp.status_code,
                    'error': _rc_error(resp.status_code, data.get('error', '提交失败'))}
    except Exception as e:
        return {'code': 502, 'error': f'无法连接 Resource Center: {e}'}
