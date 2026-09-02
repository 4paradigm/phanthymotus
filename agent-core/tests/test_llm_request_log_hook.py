"""调试用的请求日志钩子，绝不能把真实 LLM 请求打挂。

真实故障来自 orin6（10.100.121.16）：模型配成 PPIO 的 `zai-org/glm-5.2`，
_log_request 拼出 `resource/log/llm_request_zai-org/glm-5.2.json`，而 `zai-org/`
目录不存在 → FileNotFoundError。钩子跑在请求发出之前，openai SDK 又把所有非超时
异常包成 APIConnectionError('Connection error.')，于是一次写文件失败伪装成了
「网络不通」，还白白重试三次：

    [llm] zai-org/glm-5.2 @ https://api.ppio.com/openai/ failed after 0.03s:
          APIConnectionError: Connection error.
    [llm] connection failed after 3 attempts — LLM unreachable, check network

同样的 curl 请求是 200。凡是带厂商前缀（斜杠）的模型 id 都会踩到。
"""
import asyncio
import importlib
import json
import os
import sys

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# `client/__init__.py` 里的 `llm = client.llm.Client()` 把同名子模块属性盖掉了，
# 所以 `import client.llm as llm_mod` 拿到的是 Client 实例，只能走 sys.modules。
llm_mod = importlib.import_module('client.llm')  # noqa: E402


def _request(model: str) -> httpx.Request:
    return httpx.Request(
        'POST', 'https://api.ppio.com/openai/chat/completions',
        json={'model': model, 'messages': [{'role': 'user', 'content': 'hi'}]},
    )


def _run_hook(tmp_path, model: str):
    original = llm_mod.LOG_PATH
    llm_mod.LOG_PATH = tmp_path
    try:
        asyncio.run(llm_mod._log_request(_request(model)))
    finally:
        llm_mod.LOG_PATH = original


# ── 带斜杠的模型名 ────────────────────────────────────────────────────────────

def test_vendor_prefixed_model_does_not_raise(tmp_path):
    """`zai-org/glm-5.2` 不能因为缺目录而抛异常。"""
    _run_hook(tmp_path, 'zai-org/glm-5.2')


def test_vendor_prefixed_model_writes_flat_file(tmp_path):
    """斜杠被压平成下划线，落在 LOG_PATH 下而不是子目录里。"""
    _run_hook(tmp_path, 'zai-org/glm-5.2')

    path = tmp_path / 'llm_request_zai-org_glm-5.2.json'
    assert path.exists()
    assert not (tmp_path / 'zai-org').exists()
    assert json.loads(path.read_text())['model'] == 'zai-org/glm-5.2'


def test_plain_model_name_still_works(tmp_path):
    """不带斜杠的老模型名行为不变。"""
    _run_hook(tmp_path, 'glm-5.2')
    assert (tmp_path / 'llm_request_glm-5.2.json').exists()


# ── 钩子整体不可抛 ────────────────────────────────────────────────────────────

def test_unwritable_log_dir_is_swallowed(tmp_path, capsys):
    """LOG_PATH 根本不存在时也只记一行日志，不能中断请求。"""
    _run_hook(tmp_path / 'does' / 'not' / 'exist', 'glm-5.2')
    assert 'request log failed' in capsys.readouterr().out


def test_empty_body_is_ignored(tmp_path):
    """GET 之类没有 body 的请求直接跳过。"""
    original = llm_mod.LOG_PATH
    llm_mod.LOG_PATH = tmp_path
    try:
        asyncio.run(llm_mod._log_request(httpx.Request('GET', 'https://api.ppio.com/openai')))
    finally:
        llm_mod.LOG_PATH = original
    assert list(tmp_path.iterdir()) == []


def test_non_json_body_is_swallowed(tmp_path, capsys):
    """body 不是 JSON 也不能抛。"""
    original = llm_mod.LOG_PATH
    llm_mod.LOG_PATH = tmp_path
    try:
        asyncio.run(llm_mod._log_request(
            httpx.Request('POST', 'https://api.ppio.com/openai', content=b'not json')))
    finally:
        llm_mod.LOG_PATH = original
    assert 'request log failed' in capsys.readouterr().out
