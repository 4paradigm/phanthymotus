"""
test_modelhub_link.py — agent-core 用本机 ModelHub 上的模型。

覆盖四件事：
  * 二选一：大模型是云端或本机；选本机不挑具体模型，用 ModelHub 当前加载的那个；
  * 自发现：从 ModelHub 卡片的 info.serving 解析模型和网关地址，存引用不存地址；
  * 选了本机就只问本机：不和云端竞速、也不悄悄转去云端；
  * 按本机的上下文容量提前压缩历史。
"""

import asyncio
import importlib
import json
import os
import pathlib
import sys
import tempfile

import openai
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import client.llm  # noqa: E402,F401
llm_mod = importlib.import_module('client.llm')

import config  # noqa: E402
import modelhub_link  # noqa: E402
from tool_config import missing_required_config  # noqa: E402

HUB = {'id': 'modelhub', 'name': 'ModelHub', 'category': 'modelhub',
       'url': 'http://localhost:15740/mcp', 'tools': [{'name': 'llm'}]}


def _info(loaded=(), installed=(), state='ready', message=''):
    return {'state': state, 'message': message,
            'serving': {'base_url': 'http://127.0.0.1:15741/v1',
                        'models': list(loaded), 'installed': list(installed)}}


QWEN4B = {'id': 'qwen3-4b', 'title': 'Qwen3 4B', 'state': 'ready',
          'max_model_len': 40960, 'kv_tokens': 109280, 'context_tokens': 40960}


@pytest.fixture(autouse=True)
def _state():
    saved = {k: config.main.get(k) for k in ('services', 'client')}
    saved_fetch = modelhub_link.fetch_info
    config.main['services'] = {'mcp': [HUB]}
    modelhub_link.invalidate()
    yield
    for k, v in saved.items():
        config.main[k] = v
    modelhub_link.fetch_info = saved_fetch
    modelhub_link.invalidate()


def _hub_answers(info):
    async def fetch(url):
        if isinstance(info, Exception):
            raise info
        return info
    modelhub_link.fetch_info = fetch
    modelhub_link.invalidate()


# ── 存引用 ────────────────────────────────────────────────────────────────────

def test_cloud_config_is_unchanged():
    assert modelhub_link.client_entries('cloud', 'https://x/v1', 'k', 'm', False) == [
        {'url': 'https://x/v1', 'key': 'k', 'model': 'm', 'think_mode': False}]


def test_local_is_local_only_and_names_no_model():
    """选本机就只有本机一条：云端的 Key 不拿来兜底，对话不会在不知情时离开这台机器。"""
    assert modelhub_link.client_entries('modelhub', 'https://x/v1', 'k', 'm', True) == [
        {'source': 'modelhub', 'model': '', 'think_mode': True}]


def test_cloud_url_is_only_required_when_cloud_is_chosen():
    tool = {'configSchema': {
        'properties': {
            'llm_source': {'type': 'string', 'default': 'cloud'},
            'llm_url': {'type': 'string', 'x-required-when': {'llm_source': 'cloud'}},
            'llm_key': {'type': 'string', 'x-required-when': {'llm_source': 'cloud'}},
        },
        'required': ['llm_url', 'llm_key']}}
    assert missing_required_config(tool, {}) == ['llm_url', 'llm_key']       # 默认云端
    assert missing_required_config(tool, {'llm_source': 'modelhub'}) == []


def test_decision_core_schema_declares_the_source_field():
    src = (pathlib.Path(__file__).resolve().parents[1] / 'src' / 'start.py').read_text()
    assert "'format': 'llm-source'" in src
    assert "'x-required-when': {'llm_source': 'cloud'}" in src
    # 选本机时云端那组隐藏、但照样保存，切回云端不用重填
    assert src.count("'x-keep-when-hidden': True") == 3


def test_search_is_offered_only_when_configured():
    """没配搜索引擎时 WebSearch 不进工具表 —— 否则模型一调就是「未配置」的报错。
    搜索和用云端还是本机模型无关。"""
    from event.desktop import search_configured
    saved = config.main.get('desktop_tools')
    try:
        config.main['desktop_tools'] = {'search': {'type': 'none'}}
        assert search_configured() is False
        config.main['desktop_tools'] = {}
        assert search_configured() is False
        config.main['desktop_tools'] = {'search': {'type': 'baidu_search', 'base_url': 'u'}}
        assert search_configured() is True
    finally:
        config.main['desktop_tools'] = saved
    src = (pathlib.Path(__file__).resolve().parents[1] / 'src' / 'start.py').read_text()
    line = next(l for l in src.splitlines() if "'search_type':" in l)
    assert 'llm_source' not in line          # 两种大模型都能配搜索


def test_card_config_in_local_mode_keeps_the_cloud_credentials():
    from api.mcp_manage import MCPCallRequest, _handle_agentcore_call
    asyncio.run(_handle_agentcore_call(MCPCallRequest(
        tool='decision_core',
        arguments={'action': 'config', 'llm_source': 'modelhub', 'llm_url': 'https://x/v1',
                   'llm_key': 'sk-1', 'llm_model': 'm', 'think_mode': False})))
    assert config.main['client']['llm'] == [
        {'source': 'modelhub', 'model': '', 'think_mode': False}]
    llm = config.main['services']['llm']
    assert (llm['source'], llm['url'], llm['key']) == ('modelhub', 'https://x/v1', 'sk-1')


def test_card_config_with_a_local_model_needs_no_cloud_url():
    from api.mcp_manage import MCPCallRequest, _handle_agentcore_call
    config.main['client'] = {'llm': []}
    asyncio.run(_handle_agentcore_call(MCPCallRequest(
        tool='decision_core',
        arguments={'action': 'config', 'llm_source': 'modelhub', 'think_mode': False})))
    assert config.main['client']['llm'] == [
        {'source': 'modelhub', 'model': '', 'think_mode': False}]
    assert config.main['services']['llm']['source'] == 'modelhub'


# ── 自发现 ────────────────────────────────────────────────────────────────────

def test_resolve_follows_whatever_is_loaded():
    _hub_answers(_info([QWEN4B]))
    target, why = asyncio.run(modelhub_link.resolve('modelhub'))
    assert why == '' and target['base_url'] == 'http://127.0.0.1:15741/v1'
    assert target['model'] == 'qwen3-4b' and target['context_tokens'] == 40960


def test_status_says_what_is_wrong_and_where_to_fix_it():
    _hub_answers(_info([{**QWEN4B, 'state': 'loading'}], state='loading',
                       message="正在加载模型 'Qwen3 4B' …"))
    st = asyncio.run(modelhub_link.status(force=True))
    assert st['state'] == 'loading' and st['message'].startswith('正在加载')

    _hub_answers(_info([], state='idle'))
    st = asyncio.run(modelhub_link.status(force=True))
    assert st['state'] == 'idle' and '模型管理' in st['message']

    _hub_answers(ConnectionRefusedError('refused'))
    st = asyncio.run(modelhub_link.status(force=True))
    assert st['state'] == 'unreachable' and '没有回应' in st['message']

    config.main['services'] = {'mcp': []}
    st = asyncio.run(modelhub_link.status(force=True))
    assert st['state'] == 'absent'


def test_status_when_ready_names_the_model_and_its_context():
    _hub_answers(_info([QWEN4B]))
    st = asyncio.run(modelhub_link.status(force=True))
    assert (st['state'], st['title'], st['context_tokens']) == ('ready', 'Qwen3 4B', 40960)


# ── 选了本机就只问本机 ────────────────────────────────────────────────────────

class _Endpoint:
    """最小的 OpenAI 流式假 endpoint：要么吐一句话，要么抛给定的异常。"""

    def __init__(self, name, answer=None, error=None):
        self.base_url = f'http://{name}/v1'
        self.name, self.answer, self.error = name, answer, error
        self.calls = []
        outer = self

        class _Completions:
            async def create(self, **kwargs):
                outer.calls.append(kwargs)
                if outer.error:
                    raise outer.error
                return _Stream([{'choices': [{'delta': {'content': outer.answer},
                                              'finish_reason': 'stop'}]},
                                {'choices': [], 'usage': {'prompt_tokens': 30000,
                                                          'completion_tokens': 3,
                                                          'total_tokens': 30003}}])
        self.chat = type('C', (), {'completions': _Completions()})()


class _Stream:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)

    async def close(self):
        pass


def _client(local, cloud=None):
    """装好本机那条，外加一条云端（模拟有人手改出两条的配置）。"""
    c = llm_mod.Client.__new__(llm_mod.Client)
    c.local_cfg = {'source': 'modelhub', 'model': '', 'think_mode': False}
    c._local_clients = {'http://127.0.0.1:15741/v1': local}
    c._local_unavailable = None
    c.client_list = [cloud] if cloud else []
    c.config_list = [{'url': cloud.base_url, 'key': 'k', 'model': 'cloud-model',
                      'think_mode': False}] if cloud else []
    c._endpoint_dead = [False] * len(c.client_list)
    return c


def _ask(c):
    return asyncio.run(c(message_list=[{'role': 'user', 'content': 'hi'}], tool_list=[]))


def test_local_answers_and_cloud_is_never_asked():
    _hub_answers(_info([QWEN4B]))
    local, cloud = _Endpoint('local', answer='本机'), _Endpoint('cloud', answer='云端')
    msg = _ask(_client(local, cloud))
    assert msg['content'] == '本机' and cloud.calls == []
    assert local.calls[0]['model'] == 'qwen3-4b'
    assert msg['_usage']['served_by'] == 'modelhub'
    assert msg['_usage']['context_tokens'] == 40960


def test_local_not_ready_fails_with_the_reason_and_never_asks_the_cloud():
    _hub_answers(_info([{**QWEN4B, 'state': 'loading'}], state='loading',
                       message="正在加载模型 'Qwen3 4B' …"))
    local, cloud = _Endpoint('local', answer='本机'), _Endpoint('cloud', answer='云端')
    with pytest.raises(llm_mod.LocalModelUnavailable, match='正在加载'):
        _ask(_client(local, cloud))
    assert local.calls == [] and cloud.calls == []


def test_local_failure_is_not_handed_to_the_cloud():
    _hub_answers(_info([QWEN4B]))
    import httpx
    down = openai.APIConnectionError(request=httpx.Request('POST', 'http://local/v1'))
    local, cloud = _Endpoint('local', error=down), _Endpoint('cloud', answer='云端')
    saved = llm_mod.LOCAL_MAX_RETRIES
    llm_mod.LOCAL_MAX_RETRIES = 0           # 别为了这个测试真的退避两秒
    try:
        with pytest.raises(openai.APIConnectionError):
            _ask(_client(local, cloud))
    finally:
        llm_mod.LOCAL_MAX_RETRIES = saved
    assert len(local.calls) == 1 and cloud.calls == []


def test_a_subagent_model_override_still_uses_the_local_model():
    """本机只有一个模型；子代理点名的模型名在这里没有意义，换模型在模型管理里做。"""
    _hub_answers(_info([QWEN4B]))
    local = _Endpoint('local', answer='本机')
    msg = asyncio.run(_client(local)(message_list=[], tool_list=[],
                                     model_override='cloud-model'))
    assert msg['content'] == '本机' and local.calls[0]['model'] == 'qwen3-4b'


def test_init_keeps_the_keyless_local_entry_out_of_the_cloud_list():
    config.main['client'] = {'llm': modelhub_link.client_entries(
        'modelhub', 'https://x/v1', 'k', 'm', False)}
    c = llm_mod.Client()
    assert c.local_cfg['source'] == 'modelhub' and c.config_list == []


# ── 按本机上下文提前压缩 ──────────────────────────────────────────────────────

def test_context_pressure_threshold():
    from event.llm import _context_pressured
    assert _context_pressured((31000, 40960), 0.75) is True
    assert _context_pressured((20000, 40960), 0.75) is False
    assert _context_pressured((99999, None), 0.75) is False     # 容量未知不算
    assert _context_pressured(None, 0.75) is False


# ── 没装 ModelHub 时怎么提示 ──────────────────────────────────────────────────

@pytest.mark.parametrize('catalog, expected', [
    ({'data': {'modelhub': [{'id': 'modelhub'}]}, 'facets': {'acc_arch': 'jetson-jp7'}},
     {'available': True, 'acc_arch': 'jetson-jp7'}),
    ({'data': {'modelhub': []}, 'facets': {'acc_arch': 'jetson-jp7'}},
     {'available': False, 'acc_arch': 'jetson-jp7', 'reason': 'not_published'}),
    ({'data': {'modelhub': []}, 'facets': {'acc_arch': 'jetson-jp6'}},
     {'available': False, 'acc_arch': 'jetson-jp6', 'reason': 'unsupported'}),
])
def test_absent_modelhub_says_whether_this_machine_can_install_it(monkeypatch, catalog, expected):
    """部署列表已按本机加速器过滤：有 ModelHub 就能装；没有时分「不是 Thor」和「没发布」。"""
    import api.config as cfg_api
    import api.registry as reg

    async def fake_catalog(refresh=False):
        return catalog
    monkeypatch.setattr(reg, 'registry_catalog', fake_catalog)
    config.main['services'] = {'mcp': []}
    res = asyncio.run(cfg_api.local_llm())
    assert res['data']['state'] == 'absent'
    assert res['data']['install'] == expected


# ── 开启智能控制前先看大模型能不能用 ──────────────────────────────────────────

def _readiness(entries):
    config.main['client'] = {'llm': entries}
    return asyncio.run(modelhub_link.llm_readiness(force=True))


def test_unconfigured_llm_blocks_the_start_and_says_what_is_missing():
    state, message = _readiness([{'url': '', 'key': '', 'model': ''}])
    assert state == 'error' and '还没有配置大模型' in message
    assert _readiness([{'url': 'https://x/v1', 'key': 'k', 'model': 'm'}]) == ('running', '')


def test_local_llm_readiness_by_modelhub_state():
    local = [{'source': 'modelhub', 'model': '', 'think_mode': False}]
    _hub_answers(_info([QWEN4B]))
    assert _readiness(local) == ('running', '')

    # 开机自启时模型通常还在加载：等它，不算失败
    _hub_answers(_info([{**QWEN4B, 'state': 'loading'}], state='loading', message='加载中 40%'))
    assert _readiness(local) == ('loading', '加载中 40%')
    # 两个容器一起开机，agent-core 常先起来：ModelHub 没回应也先等
    _hub_answers(ConnectionRefusedError('refused'))
    assert _readiness(local)[0] == 'loading'

    # 没加载模型不会自己好：不让启动，说去哪里处理
    _hub_answers(_info([], state='idle'))
    state, message = _readiness(local)
    assert state == 'error' and '模型管理' in message

    config.main['services'] = {'mcp': []}
    state, message = _readiness(local)
    assert state == 'error' and '没有 ModelHub' in message


def test_decision_core_start_and_info_report_llm_readiness():
    """启动流程看 start 的 state 决定成败，loading 时靠轮询 info 等它就绪。"""
    from api.mcp_manage import MCPCallRequest, _handle_agentcore_call
    config.main['client'] = {'llm': [{'url': '', 'key': '', 'model': ''}]}
    saved = config.main.get('tool_config:agentcore:decision_core')
    config.main['tool_config:agentcore:decision_core'] = None
    try:
        res = asyncio.run(_handle_agentcore_call(MCPCallRequest(
            tool='decision_core', arguments={'action': 'start'})))
        assert res['data']['state'] == 'error' and '还没有配置大模型' in res['data']['message']

        config.main['client'] = {'llm': [{'source': 'modelhub', 'model': '', 'think_mode': False}]}
        _hub_answers(_info([{**QWEN4B, 'state': 'loading'}], state='loading', message='加载中'))
        info = asyncio.run(_handle_agentcore_call(MCPCallRequest(
            tool='decision_core', arguments={'action': 'info'})))
        assert info['data']['state'] == 'loading'
    finally:
        config.main['tool_config:agentcore:decision_core'] = saved


def test_idle_tells_no_model_downloaded_apart_from_not_started():
    """装了 ModelHub 但一个模型都没下载时，要做的是下载，不是启动。"""
    _hub_answers(_info([], installed=[], state='idle'))
    st = asyncio.run(modelhub_link.status(force=True))
    assert (st['state'], st['downloaded']) == ('idle', 0) and '下载' in st['message']

    _hub_answers(_info([], installed=[{'id': 'qwen3-0.6b'}, {'id': 'qwen3-4b'}], state='idle'))
    st = asyncio.run(modelhub_link.status(force=True))
    assert st['downloaded'] == 2 and '已下载 2 个模型' in st['message']

    # 老版本 ModelHub 不报 installed：说不清，就不猜
    async def old_hub(url):
        return {'state': 'idle', 'serving': {'base_url': 'http://127.0.0.1:15741/v1', 'models': []}}
    modelhub_link.fetch_info = old_hub
    st = asyncio.run(modelhub_link.status(force=True))
    assert 'downloaded' not in st and st['state'] == 'idle'
