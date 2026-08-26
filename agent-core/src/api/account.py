"""
api/account.py — Resource Center 账号（登录 / 注册 / 当前身份）。

仪表盘的「我的」界面只跟这里打交道。账号态本身存在浏览器 localStorage 的
`rc_token` 里（bearer token），Agent Core 不持久化任何用户凭据 —— 它只是代理，
避免浏览器直连 Resource Center 时的跨域与自签证书问题。

为什么需要 /session：登录时只拿到一个 token，要显示"我是谁"、以及判断 token
是否已经过期，都得回问 Resource Center。没有这个校验的话，一个过期 token 会让下游每个需要
登录的操作各自回一句 Unauthorized，用户根本不知道该去哪重新登录。
"""

import fastapi
from pydantic import BaseModel
from typing import Optional

router = fastapi.APIRouter(prefix='/account', tags=['account'])


# ── Resource Center 交互 ─────────────────────────────────────────────────────────────────

def _rc_token(request: fastapi.Request) -> Optional[str]:
    """与技能 / 解决方案一致的 X-RC-Token 头。"""
    return request.headers.get('x-rc-token')


async def _rc(method: str, path: str, token: Optional[str] = None,
              json_body=None, timeout: float = 15) -> dict:
    """复用 api/solutions.py 的 Resource Center 请求封装，不再写第二份。"""
    from api.solutions import _rc_request
    return await _rc_request(method, path, token, json_body, timeout)


# Resource Center 的 401 响应体是 `{"ok":false,"error":"Unauthorized"}`。原样透给前端的话，
# 用户看到的就是一个英文单词，既不知道发生了什么也不知道去哪登录。
_UNAUTHORIZED_TEXTS = ('unauthorized', 'not authenticated', 'invalid token')
UNAUTHORIZED_MESSAGE = '未登录或登录已过期，请在「我的」中登录'


def normalize_rc_error(status: int, error: str) -> str:
    """把 Resource Center 的鉴权类报错换成能指向下一步动作的中文。"""
    if status == 401 or (error or '').strip().lower() in _UNAUTHORIZED_TEXTS:
        return UNAUTHORIZED_MESSAGE
    return error or '请求失败'


# ── 端点 ────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    identifier: str = ''
    password:   str = ''


@router.post('/login')
async def login(req: LoginRequest):
    """登录 Resource Center，返回 bearer token 与身份。"""
    identifier = req.identifier.strip()
    if not identifier or not req.password:
        return {'code': 422, 'error': '请输入账号和密码'}

    result = await _rc('POST', '/api/auth/login', None,
                       {'identifier': identifier, 'password': req.password})
    if not result['ok']:
        # 登录接口的 401 是"账号或密码错误"，不是"登录已过期"，别套用通用文案
        error = result['error']
        if result['status'] == 401:
            error = '账号或密码错误'
        return {'code': result['status'], 'error': error}

    # 登录响应把结果放在顶层：{ok, role, token, userId}
    data = result.get('payload') or {}
    return {'code': 200, 'data': {
        'token':  data.get('token', ''),
        'userId': data.get('userId', ''),
        'role':   data.get('role', ''),
    }}


class RegisterRequest(BaseModel):
    identifier: str = ''
    password:   str = ''
    name:       str = ''


@router.post('/register')
async def register(req: RegisterRequest):
    """注册 Resource Center 账号，成功后直接登录换取 token。

    Resource Center 的注册接口只种 cookie、不发 token（`app/api/auth/register/route.ts`），
    而仪表盘是跨域调用、拿不到那个 cookie，所以注册成功后必须再走一次登录。
    """
    identifier = req.identifier.strip()
    if not identifier or not req.password:
        return {'code': 422, 'error': '请输入账号和密码'}
    if len(req.password) < 8:
        return {'code': 422, 'error': '密码至少 8 位'}

    body = {'identifier': identifier, 'password': req.password}
    if req.name.strip():
        body['name'] = req.name.strip()

    result = await _rc('POST', '/api/auth/register', None, body)
    if not result['ok']:
        error = result['error']
        if result['status'] == 409:
            error = '该账号已注册，请直接登录'
        return {'code': result['status'], 'error': error}

    return await login(LoginRequest(identifier=identifier, password=req.password))


@router.get('/session')
async def session(request: fastapi.Request):
    """用浏览器存的 token 问 Resource Center "我是谁"。

    三种结果要分开，否则会误伤：
      200 + user   → 身份有效
      401          → token 真的失效了，前端应该清掉
      其它（含 200 但 ok=false）→ 无法确认。老版本 Resource Center 的 /api/auth/session 只认
        cookie、不认 Bearer，对 token 一律回 `{ok:false,user:null}`；要是把这种
        情况也当成失效，仪表盘在 Resource Center 升级前就永远登不上了。回 503 让前端保留
        token、退化用登录时拿到的身份显示。
    """
    token = _rc_token(request)
    if not token:
        return {'code': 401, 'error': '未登录'}

    result = await _rc('GET', '/api/auth/session', token)
    if not result['ok']:
        if result['status'] == 401:
            return {'code': 401, 'error': UNAUTHORIZED_MESSAGE}
        return {'code': 503,
                'error': f'无法确认登录状态（Resource Center 返回 {result["status"]}）'}

    # session 响应把身份放在顶层的 user 里：{ok, user:{id, role, name, email}}
    user = (result.get('payload') or {}).get('user') or {}
    if not user.get('id'):
        return {'code': 503, 'error': '无法确认登录状态：Resource Center 未返回身份'}
    return {'code': 200, 'data': user}


@router.get('/rc-url')
async def rc_url():
    """当前连的 Resource Center 地址（前端用来给"去网页版注册/管理"的链接）。"""
    from api.skills import _get_rc_url
    return {'code': 200, 'data': {'url': _get_rc_url()}}
