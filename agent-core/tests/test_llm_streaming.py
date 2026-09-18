"""
test_llm_streaming.py — LLM 调用改为流式后，行为对上层必须完全不变，而卡死
必须能在几十秒内被发现。

背景（真实故障，日志逐行对齐）：

    12:17:38  大请求发出（prompt≈36k）
    12:18:16  narration 调同一个 router  ok 6.90s   ← 同一时刻同一模型，通的
    12:19:38  大请求 failed after 120.06s  APITimeoutError
    12:19:45  重试同一份 payload          ok 6.99s   prompt=36385

上游没挂、模型也不慢——是单个连接卡住，而 `stream=False` 让「卡死」和「正在
生成长回答」在 read timeout 之前无法区分，于是干等满 120 秒（×3 次重试，最坏
6 分钟）。流式之后判据变成首 token 延迟：健康时 2–3 秒，25 秒没首 token 即可
判死重试，重试成本实测 7 秒。

这里锁三件事：
  1. 聚合出来的 message 和非流式 `to_dict()` 的结构一致（上层零改动）；
  2. 卡死在「首 token」和「中途断流」两处都能按期限抛错，且被 `_classify_error`
     归为 TIMEOUT——直接复用既有重试策略；
  3. cancel_event / reconsider_event 抢占在流式下仍然生效，且会关流。

Run: cd agent-core && python3 -m pytest tests/test_llm_streaming.py
"""
import asyncio
import importlib
import os
import pathlib
import sys
import tempfile
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

# `client/__init__.py` 里的 `llm = client.llm.Client()` 把同名子模块属性盖掉了，
# 只能走 sys.modules 拿到真模块（和 test_llm_request_log_hook.py 同样的绕法）。
import client.llm  # noqa: E402,F401
llm_mod = importlib.import_module('client.llm')

import config  # noqa: E402
from event.llm import RoundReconsider, TurnCancelled  # noqa: E402


# ── 假的流式响应 ──────────────────────────────────────────────────────────────

def _chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    """OpenAI chunk 的最小 dict 形态；`_attr` 对 dict 和 pydantic 一视同仁。"""
    delta = {}
    if content is not None:
        delta['content'] = content
    if tool_calls is not None:
        delta['tool_calls'] = tool_calls
    out = {'choices': [{'delta': delta, 'finish_reason': finish_reason}]}
    if usage is not None:
        out['usage'] = usage
        out['choices'] = []
    return out


class _FakeStream:
    """按脚本吐 chunk。

    元素为 float   → 「这里永远卡住」（睡够就说明期限没生效）；
    元素为 ('wait', s) → 真的等 s 秒再继续下一条，用来模拟「慢但活着」的上游。
    """

    def __init__(self, script):
        self._script = list(script)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        while self._script:
            item = self._script.pop(0)
            if isinstance(item, tuple) and item and item[0] == 'wait':
                await asyncio.sleep(item[1])
                continue
            if isinstance(item, (int, float)):
                await asyncio.sleep(item)
                raise AssertionError('sleep should have been cut short by the deadline')
            return item
        raise StopAsyncIteration

    async def close(self):
        self.closed = True


class _FakeCompletions:
    def __init__(self, outer):
        self._outer = outer

    async def create(self, **kwargs):
        self._outer.kwargs = kwargs
        if self._outer.open_delay:
            await asyncio.sleep(self._outer.open_delay)
        self._outer.stream = _FakeStream(self._outer.script)
        return self._outer.stream


class _FakeClient:
    base_url = 'https://fake/v1'

    def __init__(self, script=(), open_delay=0.0):
        self.script = list(script)
        self.open_delay = open_delay
        self.stream = None
        self.kwargs = None
        self.chat = type('C', (), {'completions': _FakeCompletions(self)})()


def _client(fake):
    """绕过 _init_clients，只装一个假 endpoint。"""
    c = llm_mod.Client.__new__(llm_mod.Client)
    c.client_list = [fake]
    c._endpoint_dead = [False]
    return c


@pytest.fixture(autouse=True)
def _one_endpoint():
    """config.main 是进程级单例，显式钉住而不是指望默认值。"""
    saved = config.main.get('client')
    config.main['client'] = {'llm': [{'url': 'https://fake/v1', 'key': 'sk-x',
                                      'model': 'fake-model', 'think_mode': False}]}
    yield
    config.main['client'] = saved


@pytest.fixture(autouse=True)
def _fast_deadlines():
    """把期限压到毫秒级，否则测超时要等十几秒。阶梯保持递增形状。"""
    saved = (llm_mod.FIRST_TOKEN_TIMEOUTS_S, llm_mod.CHUNK_TIMEOUT_S,
             llm_mod.TOTAL_TIMEOUT_S, llm_mod.HEARTBEAT_S, llm_mod.STREAM_ENABLED)
    llm_mod.FIRST_TOKEN_TIMEOUTS_S = [0.2, 0.4, 0.8]
    llm_mod.CHUNK_TIMEOUT_S = 0.3
    llm_mod.TOTAL_TIMEOUT_S = 5.0
    llm_mod.HEARTBEAT_S = 0.05
    llm_mod.STREAM_ENABLED = True
    yield
    (llm_mod.FIRST_TOKEN_TIMEOUTS_S, llm_mod.CHUNK_TIMEOUT_S,
     llm_mod.TOTAL_TIMEOUT_S, llm_mod.HEARTBEAT_S, llm_mod.STREAM_ENABLED) = saved


def _call(fake, **kw):
    return asyncio.run(_client(fake)(message_list=[], tool_list=[], **kw))


# ── 聚合结果的形状 ────────────────────────────────────────────────────────────

def test_plain_text_is_reassembled():
    msg = _call(_FakeClient([_chunk('你'), _chunk('好'), _chunk(finish_reason='stop')]))
    assert msg == {'role': 'assistant', 'content': '你好'}


def test_tool_call_fragments_are_reassembled():
    """arguments 按帧切片是流式的常态，拼不回来就等于工具调用全废。"""
    msg = _call(_FakeClient([
        _chunk(tool_calls=[{'index': 0, 'id': 'call_1', 'type': 'function',
                            'function': {'name': 'speaker_play', 'arguments': '{"te'}}]),
        _chunk(tool_calls=[{'index': 0, 'function': {'arguments': 'xt":"hi"}'}}]),
        _chunk(finish_reason='tool_calls'),
    ]))
    assert msg['tool_calls'] == [{
        'id': 'call_1', 'type': 'function',
        'function': {'name': 'speaker_play', 'arguments': '{"text":"hi"}'},
    }]


def test_repeated_full_name_is_not_doubled():
    """有的网关每帧都重复完整 name；无脑 += 会拼出 speaker_playspeaker_play。"""
    msg = _call(_FakeClient([
        _chunk(tool_calls=[{'index': 0, 'id': 'c1',
                            'function': {'name': 'speaker_play', 'arguments': '{}'}}]),
        _chunk(tool_calls=[{'index': 0, 'function': {'name': 'speaker_play'}}]),
    ]))
    assert msg['tool_calls'][0]['function']['name'] == 'speaker_play'


def test_parallel_tool_calls_keep_index_order():
    msg = _call(_FakeClient([
        _chunk(tool_calls=[{'index': 1, 'id': 'b', 'function': {'name': 'g', 'arguments': '{}'}}]),
        _chunk(tool_calls=[{'index': 0, 'id': 'a', 'function': {'name': 'f', 'arguments': '{}'}}]),
    ]))
    assert [tc['id'] for tc in msg['tool_calls']] == ['a', 'b']


def test_empty_tool_call_is_dropped():
    """glm 会追加 id/name 全空的重复条目，进了历史之后每次请求都被 400 拒掉。"""
    msg = _call(_FakeClient([
        _chunk(tool_calls=[{'index': 0, 'id': 'c1', 'function': {'name': 'f', 'arguments': '{}'}}]),
        _chunk(tool_calls=[{'index': 1, 'id': '', 'function': {'name': '', 'arguments': '{}'}}]),
    ]))
    assert [tc['id'] for tc in msg['tool_calls']] == ['c1']


def test_think_tags_are_stripped():
    msg = _call(_FakeClient([_chunk('<think>'), _chunk('嗯'), _chunk('</think>好')]))
    assert msg['content'] == '嗯好'


def test_usage_from_final_chunk_is_attached():
    msg = _call(_FakeClient([
        _chunk('hi'),
        _chunk(usage={'prompt_tokens': 36385, 'completion_tokens': 12,
                      'total_tokens': 36397,
                      'prompt_tokens_details': {'cached_tokens': 30000}}),
    ]))
    assert msg['_usage']['prompt_tokens'] == 36385
    assert msg['_usage']['cached_tokens'] == 30000
    assert msg['_usage']['elapsed_s'] >= 0


def test_missing_usage_is_not_fatal():
    """不支持 include_usage 的网关只是少一行统计，不能让调用失败。"""
    msg = _call(_FakeClient([_chunk('hi')]))
    assert msg['content'] == 'hi' and '_usage' not in msg


def test_include_usage_is_requested():
    fake = _FakeClient([_chunk('hi')])
    _call(fake)
    assert fake.kwargs['stream'] is True
    assert fake.kwargs['stream_options'] == {'include_usage': True}


def test_think_mode_off_still_sends_extra_body():
    """think_mode=False 的 extra_body 是生产在用的，流式不能把它弄丢。"""
    fake = _FakeClient([_chunk('hi')])
    _call(fake)
    assert fake.kwargs['extra_body'] == {'chat_template_kwargs': {'enable_thinking': False}}


# ── 卡死必须被及时发现 ────────────────────────────────────────────────────────

def test_no_first_token_times_out_at_the_deadline():
    """核心回归：卡在首 token 上不能干等到 httpx 的 120s read timeout。"""
    fake = _FakeClient([30.0])  # 永远不出第一帧
    start = time.perf_counter()
    with pytest.raises(Exception) as ei:
        _call(fake)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f'首 token 期限没生效，等了 {elapsed:.1f}s'
    kind, _ = llm_mod._classify_error(ei.value)
    assert kind == llm_mod.LLMErrorKind.TIMEOUT
    assert 'first token' in str(ei.value)


def test_stall_midstream_times_out():
    """首 token 来了但生成中途断流，同样要判死，而不是让 chunk 间隔无限长。"""
    fake = _FakeClient([_chunk('开始'), 30.0])
    with pytest.raises(Exception) as ei:
        _call(fake)
    kind, _ = llm_mod._classify_error(ei.value)
    assert kind == llm_mod.LLMErrorKind.TIMEOUT
    assert 'stalled' in str(ei.value)


def test_hang_before_response_headers_times_out():
    """stream=True 下 create() 也要等响应头，卡在这里同样受首 token 预算管。"""
    fake = _FakeClient([_chunk('hi')], open_delay=30.0)
    with pytest.raises(Exception) as ei:
        _call(fake)
    kind, _ = llm_mod._classify_error(ei.value)
    assert kind == llm_mod.LLMErrorKind.TIMEOUT


def test_timeout_closes_the_stream():
    """超时不关流，连接会一直挂在池里。"""
    fake = _FakeClient([_chunk('开始'), 30.0])
    with pytest.raises(Exception):
        _call(fake)
    assert fake.stream.closed


# ── 首字期限阶梯 ──────────────────────────────────────────────────────────────

def test_ladder_gives_each_attempt_a_longer_budget():
    llm_mod.FIRST_TOKEN_TIMEOUTS_S = [10.0, 20.0, 60.0]
    assert [llm_mod._first_token_timeout(i) for i in range(3)] == [10.0, 20.0, 60.0]
    # 超出阶梯长度沿用最后一级，而不是 IndexError
    assert llm_mod._first_token_timeout(7) == 60.0


def test_single_env_value_degrades_to_a_flat_ladder():
    """LLM_FIRST_TOKEN_TIMEOUT=15 这种老写法仍然有效（一级不变的阶梯）。"""
    assert llm_mod._env_floats('_NOPE_', [1.0]) == [1.0]
    os.environ['_LADDER_TEST_'] = '15'
    try:
        assert llm_mod._env_floats('_LADDER_TEST_', [1.0]) == [15.0]
        os.environ['_LADDER_TEST_'] = '5, 10, 30'
        assert llm_mod._env_floats('_LADDER_TEST_', [1.0]) == [5.0, 10.0, 30.0]
        os.environ['_LADDER_TEST_'] = 'garbage'
        assert llm_mod._env_floats('_LADDER_TEST_', [1.0]) == [1.0]
    finally:
        del os.environ['_LADDER_TEST_']


def test_slow_but_alive_upstream_survives_on_the_second_rung():
    """核心：上游今天就是慢（不是卡死）时，阶梯必须救回来而不是三次都打断。

    这正是 Hermes Agent #7069 的死锁形状——首字超时在 prefill 期间打断请求，
    立刻重试同一个重请求，再次打断，永远出不来。首字 0.2s 拦不住 0.3s 的上游，
    第二级 0.4s 就该放它过去。
    """
    msg = _call(_FakeClient([('wait', 0.3), _chunk('ok')]))
    assert msg['content'] == 'ok'


def test_ladder_does_not_stall_three_full_budgets_on_a_dead_connection():
    """真卡死时三级加起来仍然远小于原来的单次 120s。"""
    fake = _FakeClient([30.0])
    start = time.perf_counter()
    with pytest.raises(Exception):
        _call(fake)
    elapsed = time.perf_counter() - start
    # 阶梯 0.2 + 0.4 + 0.8 = 1.4s
    assert 1.0 < elapsed < 2.5, f'阶梯没有按预期递增消耗，实际 {elapsed:.2f}s'


def test_timeout_is_retried_and_a_healthy_retry_wins():
    """实测里重试只要 7 秒——超时后必须真的重试，而不是直接把错抛给上层。"""
    fake = _FakeClient([30.0])

    calls = {'n': 0}
    orig_create = fake.chat.completions.create

    async def create(**kwargs):
        calls['n'] += 1
        if calls['n'] == 1:
            return await orig_create(**kwargs)
        fake.script = [_chunk('ok')]
        return await orig_create(**kwargs)

    fake.chat.completions.create = create
    msg = _call(fake)
    assert calls['n'] == 2 and msg['content'] == 'ok'


# ── 抢占语义不变 ──────────────────────────────────────────────────────────────

def test_cancel_event_still_raises_turn_cancelled_midstream():
    async def scenario():
        fake = _FakeClient([_chunk('开'), 30.0])
        cancel = asyncio.Event()

        async def _arm():
            await asyncio.sleep(0.05)
            cancel.set()

        asyncio.create_task(_arm())
        with pytest.raises(TurnCancelled):
            await _client(fake)(message_list=[], tool_list=[], cancel_event=cancel)
        # 抢占后让 shield 里的 close() 跑完
        await asyncio.sleep(0.05)
        assert fake.stream.closed, '抢占时没关流'

    asyncio.run(scenario())


def test_reconsider_event_still_raises_round_reconsider_midstream():
    async def scenario():
        fake = _FakeClient([_chunk('开'), 30.0])
        reconsider = asyncio.Event()

        async def _arm():
            await asyncio.sleep(0.05)
            reconsider.set()

        asyncio.create_task(_arm())
        with pytest.raises(RoundReconsider):
            await _client(fake)(message_list=[], tool_list=[], reconsider_event=reconsider)

    asyncio.run(scenario())


# ── 降级开关 ──────────────────────────────────────────────────────────────────

def test_stream_can_be_turned_off():
    """LLM_STREAM=0 回到非流式老路径，排查时可用。"""
    llm_mod.STREAM_ENABLED = False

    fake = _FakeClient([])

    async def create(**kwargs):
        fake.kwargs = kwargs
        msg = type('M', (), {'to_dict': lambda self: {'role': 'assistant', 'content': 'hi'}})()
        choice = type('C', (), {'message': msg})()
        return type('R', (), {'choices': [choice], 'usage': None})()

    fake.chat.completions.create = create
    msg = _call(fake)
    assert fake.kwargs['stream'] is False
    assert msg == {'role': 'assistant', 'content': 'hi'}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
