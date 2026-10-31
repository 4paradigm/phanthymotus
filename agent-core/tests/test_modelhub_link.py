"""
test_modelhub_link.py — agent-core 用本机 ModelHub 上的模型。

覆盖四件事：
  * 自发现：从 ModelHub 卡片的 info.serving 解析「用哪个模型、在哪个地址」；
  * 存引用：卡片 / 设置页写进 client.llm 的是 'modelhub' 这样的引用，不是地址；
  * 本机优先、云端兜底：选了本机就只问本机，用不了才问云端，而不是两边竞速；
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


def test_local_goes_first_and_cloud_becomes_a_fallback():
    entries = modelhub_link.client_entries('modelhub:qwen3-4b', 'https://x/v1', 'k', 'm', True)
    assert entries[0] == {'source': 'modelhub:qwen3-4b', 'model': 'qwen3-4b', 'think_mode': True}
    assert entries[1]['role'] == 'fallback' and entries[1]['url'] == 'https://x/v1'
    # 没填云端：只有本机那一条，不留一条没 key 的空壳
    assert modelhub_link.client_entries('modelhub', '', '', '', False) == [
        {'source': 'modelhub', 'model': '', 'think_mode': False}]


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
    assert missing_required_config(tool, {'llm_source': 'modelhub:qwen3-4b'}) == []


def test_decision_core_schema_declares_the_source_field():
    src = (pathlib.Path(__file__).resolve().parents[1] / 'src' / 'start.py').read_text()
    assert "'format': 'llm-source'" in src
    assert "'x-required-when': {'llm_source': 'cloud'}" in src


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


def test_resolve_says_why_it_cannot_serve():
    _hub_answers(_info([{**QWEN4B, 'state': 'loading'}]))
    target, why = asyncio.run(modelhub_link.resolve('modelhub'))
    assert target is None and '还没就绪' in why

    _hub_answers(_info([QWEN4B]))
    target, why = asyncio.run(modelhub_link.resolve('modelhub:qwen3-0.6b'))
    assert target is None and '不是 qwen3-0.6b' in why

    _hub_answers(_info([], state='idle', message='没有加载模型'))
    assert asyncio.run(modelhub_link.resolve('modelhub'))[0] is None

    _hub_answers(ConnectionRefusedError('refused'))
    target, why = asyncio.run(modelhub_link.resolve('modelhub'))
    assert target is None and '没有回应' in why

    config.main['services'] = {'mcp': []}
    modelhub_link.invalidate()
    assert asyncio.run(modelhub_link.resolve('modelhub')) == (None, '本机没有 ModelHub')


def test_options_list_cloud_follow_and_every_installed_model():
    _hub_answers(_info([QWEN4B], installed=[
        {'id': 'qwen3-4b', 'title': 'Qwen3 4B', 'unsupported': ''},
        {'id': 'qwen3-0.6b', 'title': 'Qwen3 0.6B', 'unsupported': ''},
        {'id': 'phanthymotus-0.1', 'title': 'PhanthyMotus 0.1',
         'unsupported': '需要 vLLM ≥ 0.19'}]))
    opts = asyncio.run(modelhub_link.source_options())
    by = {o['value']: o for o in opts}
    assert by['cloud']['label'].startswith('云端')
    assert 'Qwen3 4B' in by['modelhub']['label'] and '41k' in by['modelhub']['label']
    assert '已加载' in by['modelhub:qwen3-4b']['label']
    assert '未加载' in by['modelhub:qwen3-0.6b']['label']
    assert by['modelhub:phanthymotus-0.1']['disabled'] is True


def test_a_saved_choice_that_is_gone_still_shows():
    _hub_answers(_info([QWEN4B]))
    opts = asyncio.run(modelhub_link.source_options('modelhub:old-model'))
    assert opts[-1] == {'value': 'modelhub:old-model', 'label': 'modelhub:old-model（已不在本机）'}


# ── 本机优先、云端兜底 ────────────────────────────────────────────────────────

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


def _client(local, cloud=None, source='modelhub'):
    """装好一条本机 + 可选一条云端；本机的 OpenAI client 直接塞进按地址的缓存。"""
    c = llm_mod.Client.__new__(llm_mod.Client)
    c.local_cfg = {'source': source, 'model': modelhub_link.pinned_model(source),
                   'think_mode': False}
    c._local_clients = {'http://127.0.0.1:15741/v1': local}
    c._local_unavailable = None
    c.client_list = [cloud] if cloud else []
    c.config_list = [{'url': cloud.base_url, 'key': 'k', 'model': 'cloud-model',
                      'think_mode': False, 'role': 'fallback'}] if cloud else []
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


def test_local_still_loading_goes_straight_to_cloud():
    _hub_answers(_info([{**QWEN4B, 'state': 'loading'}]))
    local, cloud = _Endpoint('local', answer='本机'), _Endpoint('cloud', answer='云端')
    msg = _ask(_client(local, cloud))
    assert msg['content'] == '云端' and local.calls == []
    assert 'served_by' not in msg.get('_usage', {})


def test_local_failure_falls_back_to_cloud():
    _hub_answers(_info([QWEN4B]))
    import httpx
    down = openai.APIConnectionError(request=httpx.Request('POST', 'http://local/v1'))
    local, cloud = _Endpoint('local', error=down), _Endpoint('cloud', answer='云端')
    saved = llm_mod.LOCAL_MAX_RETRIES
    llm_mod.LOCAL_MAX_RETRIES = 0           # 别为了这个测试真的退避两秒
    try:
        msg = _ask(_client(local, cloud))
    finally:
        llm_mod.LOCAL_MAX_RETRIES = saved
    assert msg['content'] == '云端' and len(local.calls) == 1


def test_context_overflow_is_not_sent_to_the_cloud():
    """溢出要由主循环压缩历史；转给云端只是把同一个超长请求再发一遍。"""
    _hub_answers(_info([QWEN4B]))
    import httpx
    resp = httpx.Response(400, request=httpx.Request('POST', 'http://local/v1'))
    overflow = openai.BadRequestError(
        "This model's maximum context length is 40960 tokens", response=resp, body=None)
    local, cloud = _Endpoint('local', error=overflow), _Endpoint('cloud', answer='云端')
    with pytest.raises(openai.BadRequestError):
        _ask(_client(local, cloud))
    assert cloud.calls == []


def test_local_only_and_unavailable_says_so():
    _hub_answers(_info([], state='idle', message='没有加载模型'))
    with pytest.raises(RuntimeError, match='也没有配置云端后备'):
        _ask(_client(_Endpoint('local', answer='x')))


def test_a_subagent_naming_another_model_skips_local():
    _hub_answers(_info([QWEN4B]))
    local, cloud = _Endpoint('local', answer='本机'), _Endpoint('cloud', answer='云端')
    msg = asyncio.run(_client(local, cloud)(message_list=[], tool_list=[],
                                            model_override='cloud-model'))
    assert msg['content'] == '云端' and local.calls == []


def test_init_keeps_the_keyless_local_entry_out_of_the_cloud_list():
    config.main['client'] = {'llm': modelhub_link.client_entries(
        'modelhub', 'https://x/v1', 'k', 'm', False)}
    c = llm_mod.Client()
    assert c.local_cfg['source'] == 'modelhub'
    assert [cfg['url'] for cfg in c.config_list] == ['https://x/v1']


# ── 按本机上下文提前压缩 ──────────────────────────────────────────────────────

def test_context_pressure_threshold():
    from event.llm import _context_pressured
    assert _context_pressured((31000, 40960), 0.75) is True
    assert _context_pressured((20000, 40960), 0.75) is False
    assert _context_pressured((99999, None), 0.75) is False     # 容量未知不算
    assert _context_pressured(None, 0.75) is False
