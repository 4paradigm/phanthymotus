import typing
import base64
import os
import pathlib
import asyncio
import re
import openai
import httpx
import time
import json

import config

LOG_PATH = pathlib.Path('./resource/log')


# ── 流式参数 ──────────────────────────────────────────────────────────────────
#
# 为什么必须流式：`stream=False` 时，在整个响应体到齐之前没有任何信号，于是
# 「请求卡死」和「答案很长正在生成」这两件事在 read timeout 之前根本无法区分。
# 实测现象是同一个 prompt（36k）在 12:19:38 报 APITimeoutError after 120.06s，
# 7 秒后重试同一份 payload 只花 6.99s 就回来了——上游没挂、模型也不慢，是单个
# 连接卡住，而我们干等了两分钟（×3 次重试，最坏 6 分钟）。
#
# 流式之后判据变成「首 token 延迟」和「chunk 间隔」，它们和总生成时长是两回事：
# 健康时首 token 在 2–3 秒内，所以十几秒没有首 token 就可以判死重试，而重试成本
# 只有几秒。总时长反而可以放宽，长回答不再被误杀。

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, '') or default)
    except ValueError:
        return default


def _env_floats(name: str, default: list) -> list:
    """逗号分隔的一串秒数；给单个值就退化成一级不变的阶梯。"""
    raw = os.environ.get(name, '').strip()
    if not raw:
        return default
    try:
        vals = [float(x) for x in raw.split(',') if x.strip()]
    except ValueError:
        return default
    return vals or default


# 关掉流式回到老路径的开关（仅用于排查，默认开）
STREAM_ENABLED = os.environ.get('LLM_STREAM', '1').strip().lower() not in ('0', 'false', 'no')
# 首 token 期限，按重试次数递增。
#
# 为什么是阶梯而不是一个固定值、更不是按输入长度算的公式：prefill 确实是
# O(input)，但托管 router 上 36k prompt 的 TTFT 实测 2–3 秒，4.5k 的播报请求整个
# 请求才 1.4–3 秒——输入涨 8 倍，首字只差几秒，而阈值要覆盖的是几十秒级的卡死。
# 公式只会把 10s 换成 12s，判据没变却多一个会漂移的参数。
#
# 真正的风险在反方向：Hermes Agent #7069 是首字超时在 prefill 期间打断请求、立刻
# 重试同一个重请求、再次打断的永久死锁。阶梯用经验替代对 prefill 成本的建模——
# 第一次 10s 快速探活（卡死时实测 7 秒就能恢复），真慢的话第二、三次自然放过去。
FIRST_TOKEN_TIMEOUTS_S = _env_floats('LLM_FIRST_TOKEN_TIMEOUT', [10.0, 20.0, 60.0])


def _first_token_timeout(attempt: int) -> float:
    """第 attempt 次尝试（从 0 起）的首 token 期限；超出阶梯长度就沿用最后一级。"""
    ladder = FIRST_TOKEN_TIMEOUTS_S or [10.0]
    return ladder[min(attempt, len(ladder) - 1)]
# chunk 间隔期限：生成中途断流。已经开始出字之后，token 是连着来的，间隔远小于
# 1 秒，所以 10 秒的静默只可能是断流。注意这一闸管的是「两帧之间」，不是总时长，
# 长回答不受影响；唯一的误伤场景是服务端在推理阶段一帧都不发（think_mode=True
# 且不吐 reasoning 增量），那种 endpoint 用 LLM_CHUNK_TIMEOUT 调大即可。
CHUNK_TIMEOUT_S = _env_float('LLM_CHUNK_TIMEOUT', 10.0)
# 总时长上限（放宽，只兜底）
TOTAL_TIMEOUT_S = _env_float('LLM_TOTAL_TIMEOUT', 600.0)
# 在飞请求的心跳日志间隔；0 关闭。事中可见，不必等失败后才蹦一行
HEARTBEAT_S = _env_float('LLM_HEARTBEAT', 15.0)

# 思考内容落盘时保留的字符数。整条留会把历史和 data.db 撑大（实测单轮 870 字，
# 长的更多），全丢又等于看不见——所以留一段，长度照完整值记。设 0 关掉保留，
# 只留长度。
REASONING_KEEP_CHARS = int(_env_float('LLM_REASONING_KEEP_CHARS', 4000))


async def _log_request(request: httpx.Request):
    """httpx event hook: dump the real HTTP request body to disk for debugging.

    这个钩子在请求发出前执行，任何异常都会中断发送；而 openai SDK 把非超时异常
    一律包成 APIConnectionError('Connection error.')，于是一个写文件失败会伪装成
    「网络不通」并触发三次重试。所以这里既要保证文件名合法（模型名可能带厂商前缀
    的斜杠，如 zai-org/glm-5.2），也要整体吞掉异常——调试日志不该拖垮真实请求。
    """
    try:
        if not request.content:
            return
        body = json.loads(request.content)
        model = str(body.get('model', 'unknown')).replace('/', '_')
        path = LOG_PATH / f'llm_request_{model}.json'
        path.write_text(json.dumps(body, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f'[llm] request log failed (ignored): {type(e).__name__}: {e}')


# ── 错误分类 ──────────────────────────────────────────────────────────────────

def _valid_tool_call(tc) -> bool:
    """tool_call 结构是否能被服务端接受：id 与 function.name 都必须是非空串。"""
    if not isinstance(tc, dict) or not tc.get('id'):
        return False
    fn = tc.get('function')
    return isinstance(fn, dict) and bool(fn.get('name'))


class LLMErrorKind:
    RATE_LIMIT      = 'rate_limit'       # 429
    BILLING         = 'billing'          # 402
    SERVER_ERROR    = 'server_error'     # 500/502/503/529
    CONTEXT_OVERFLOW = 'context_overflow' # 上下文溢出
    AUTH            = 'auth'             # 401/403
    TIMEOUT         = 'timeout'          # 超时
    CONNECTION      = 'connection'       # 网络连接失败
    UNKNOWN         = 'unknown'


def _classify_error(e: Exception) -> tuple[str, float | None]:
    """分类 LLM 调用错误，返回 (kind, retry_after_seconds | None)。"""
    status = getattr(e, 'status_code', None)
    body_msg = str(e).lower()

    if isinstance(e, (asyncio.TimeoutError, httpx.TimeoutException, openai.APITimeoutError)):
        return LLMErrorKind.TIMEOUT, 5.0

    if isinstance(e, openai.APIConnectionError):
        return LLMErrorKind.CONNECTION, 2.0

    if status == 429:
        # 尝试解析 retry-after
        retry_after = None
        if hasattr(e, 'response') and e.response is not None:
            ra = e.response.headers.get('retry-after')
            if ra:
                try:
                    retry_after = float(ra)
                except ValueError:
                    pass
        return LLMErrorKind.RATE_LIMIT, retry_after or 10.0

    if status == 402:
        return LLMErrorKind.BILLING, None

    if status in (401, 403):
        return LLMErrorKind.AUTH, None

    if status in (500, 502, 503, 529):
        return LLMErrorKind.SERVER_ERROR, 3.0

    # 模型生成了非法 tool call（arguments 非 JSON）：可重试，下次可能正确
    # 但如果服务端指名道姓地怪某条历史消息（messages[24].tool_calls[1]...），
    # payload 每次都一样，重试只是把同一个 400 再撞两遍。
    if status == 400 and 'messages[' in body_msg:
        return LLMErrorKind.UNKNOWN, None

    if status == 400 and any(kw in body_msg for kw in (
        'json format', 'invalid_parameter', 'must be in json',
    )):
        return LLMErrorKind.SERVER_ERROR, 1.0

    # 上下文溢出：从错误消息推断
    if any(kw in body_msg for kw in (
        'context length', 'context_length', 'too many tokens',
        'maximum context', 'token limit', 'max_tokens',
    )):
        return LLMErrorKind.CONTEXT_OVERFLOW, None

    return LLMErrorKind.UNKNOWN, None


# ── 流式聚合 ──────────────────────────────────────────────────────────────────

def _attr(obj, key, default=None):
    """chunk 既可能是 pydantic 模型也可能是裸 dict（部分网关/测试），统一取值。"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        v = obj.get(key, default)
    else:
        v = getattr(obj, key, default)
    return default if v is None else v


class _StreamAccumulator:
    """把 chat.completions 的 delta 流拼回一条完整 message。

    只做拼装，不做清洗——清洗交给 `_finalize_message`，和非流式路径共用同一份
    代码，避免两条路径的行为漂移。
    """

    def __init__(self):
        self.content_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.tool_calls: dict[int, dict] = {}
        self.finish_reason: str | None = None
        self.usage = None
        self.chunks = 0

    @property
    def chars(self) -> int:
        # 思考的字数也要算进来。它只进心跳行，不影响聚合结果——漏了的话，
        # think_mode=True 下的思考阶段会打出 `chunks=1145 chars=0`，看着像
        # 「流在动但一个字都没出」的卡死，实测 18.7s 里有 15s 长这样。
        return (sum(len(p) for p in self.content_parts)
                + sum(len(p) for p in self.reasoning_parts)
                + sum(len(tc['function']['arguments']) for tc in self.tool_calls.values()))

    def feed(self, chunk) -> None:
        self.chunks += 1

        usage = _attr(chunk, 'usage')
        if usage:
            self.usage = usage

        choices = _attr(chunk, 'choices') or []
        if not choices:
            # include_usage 的最后一帧没有 choices，正常
            return
        choice = choices[0]
        fr = _attr(choice, 'finish_reason')
        if fr:
            self.finish_reason = fr

        delta = _attr(choice, 'delta')
        if delta is None:
            return

        text = _attr(delta, 'content', '')
        if text:
            self.content_parts.append(text)

        # think_mode=True 时模型的思考逐帧到达（实测 126 帧 / 870 字），而这里原先
        # 只取 content 和 tool_calls，于是它在进入任何日志之前就被丢掉：整个容器
        # 日志和落盘的请求文件里 `reasoning_content` 命中数都是 0。开着 think 的
        # token 和首 token 延迟照付，内容却无从复盘。
        reasoning = _attr(delta, 'reasoning_content', '')
        if reasoning:
            self.reasoning_parts.append(reasoning)

        for tcd in (_attr(delta, 'tool_calls') or []):
            idx = _attr(tcd, 'index', 0) or 0
            slot = self.tool_calls.setdefault(idx, {
                'id': '', 'type': 'function',
                'function': {'name': '', 'arguments': ''},
            })
            tc_id = _attr(tcd, 'id', '')
            if tc_id:
                slot['id'] = tc_id
            tc_type = _attr(tcd, 'type', '')
            if tc_type:
                slot['type'] = tc_type
            fn = _attr(tcd, 'function')
            if fn is None:
                continue
            name = _attr(fn, 'name', '')
            if name:
                # OpenAI 只在第一帧给出完整 name，但有的网关会把它切片、有的会
                # 每帧重复同一个完整 name。无条件 += 会拼出 get_timeget_time，
                # 所以重复的整串直接忽略，其余按切片追加。
                if name != slot['function']['name']:
                    slot['function']['name'] += name
            args = _attr(fn, 'arguments', '')
            if args:
                slot['function']['arguments'] += args

    @property
    def reasoning(self) -> str:
        return ''.join(self.reasoning_parts)

    def message(self) -> dict:
        msg: dict = {'role': 'assistant', 'content': ''.join(self.content_parts)}
        if self.tool_calls:
            msg['tool_calls'] = [self.tool_calls[i] for i in sorted(self.tool_calls)]
        _attach_reasoning(msg, self.reasoning)
        return msg


def _attach_reasoning(msg: dict, text: str) -> dict:
    """把思考内容挂成内部字段，让它在日志和落盘文件里可见。

    下划线开头是本仓库既有的约定：`event/llm.py` 出门前会摘掉所有 `_` 字段，
    所以它能进历史、进 recent 文件，却不会随历史回到请求体里被服务端拒。
    （`_usage` 走的就是这条路。）

    正文截断而不是整条丢弃：思考可以很长，而历史是要落 data.db 的；长度始终
    按完整值记，截断了就说出来，免得把「被截断」读成「模型只想了这么多」。
    """
    if not text:
        return msg
    limit = REASONING_KEEP_CHARS
    kept = text if len(text) <= limit else text[:limit]
    msg['_reasoning'] = {'chars': len(text), 'text': kept,
                         'truncated': len(text) > limit}
    return msg


def _finalize_message(msg: dict) -> dict:
    """流式/非流式共用的消息清洗。"""
    # OpenAI SDK 可能生成 tool_calls: None，清理以避免下游迭代报错
    if 'tool_calls' in msg and msg['tool_calls'] is None:
        del msg['tool_calls']
    # glm 有时在正常 tool_call 之后追加一条 id/name 都是空串的重复条目
    # （{"id": "", "function": {"name": "", "arguments": "{}"}}）。字段是齐的，
    # 所以只判断 key 存在会放过去；一旦进历史并落盘，之后每次请求都会被
    # 服务端以 messages[N].tool_calls[M].function missing required field "name"
    # 拒掉，重启续跑也还在。这里按「值非空」筛。
    if 'tool_calls' in msg and isinstance(msg['tool_calls'], list):
        msg['tool_calls'] = [tc for tc in msg['tool_calls'] if _valid_tool_call(tc)]
        if not msg['tool_calls']:
            del msg['tool_calls']
    # 清理模型泄漏的 think 标签残留
    if msg.get('content'):
        msg['content'] = re.sub(r'</?think>', '', msg['content']).strip()
    return msg


def _usage_dict(usage, elapsed: float) -> dict | None:
    if not usage:
        return None
    cached = _attr(usage, 'prompt_tokens_details')
    return {
        'prompt_tokens': _attr(usage, 'prompt_tokens', 0),
        'completion_tokens': _attr(usage, 'completion_tokens', 0),
        'total_tokens': _attr(usage, 'total_tokens', 0),
        'cached_tokens': _attr(cached, 'cached_tokens', 0) if cached else 0,
        'elapsed_s': round(elapsed, 2),
    }


def _timeout(url: str, message: str) -> Exception:
    """流式期限超时。

    用 httpx.ReadTimeout 而不是自定义异常：`_classify_error` 已经把它归为
    TIMEOUT，上层的重试策略、错误上报一个字都不用改。
    """
    return httpx.ReadTimeout(message, request=httpx.Request('POST', url))


async def _drain_stream(stream, url: str, model: str, t0: float,
                        acc: _StreamAccumulator, first_timeout: float) -> None:
    """按「首 token 期限 / chunk 间隔期限 / 总时长」三道闸消费流。"""
    it = stream.__aiter__()
    while True:
        first = acc.chunks == 0
        elapsed = time.perf_counter() - t0
        # 首 token 的期限从请求发出算起（建连和响应头也花在这段里），之后按
        # chunk 间隔算。
        limit = (first_timeout - elapsed) if first else CHUNK_TIMEOUT_S
        remaining = TOTAL_TIMEOUT_S - elapsed
        if remaining <= 0:
            raise _timeout(url, f'{model}: stream exceeded total budget {TOTAL_TIMEOUT_S:.0f}s')
        if limit <= 0:
            raise _timeout(
                url, f'{model}: no first token within {first_timeout:.0f}s')
        try:
            chunk = await asyncio.wait_for(it.__anext__(), timeout=min(limit, remaining))
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            waited = time.perf_counter() - t0
            what = (f'no first token within {first_timeout:.0f}s'
                    if first else
                    f'stream stalled {CHUNK_TIMEOUT_S:.0f}s after {acc.chunks} chunks')
            raise _timeout(url, f'{model}: {what} (elapsed {waited:.1f}s)') from None
        acc.feed(chunk)


async def _heartbeat(model: str, url: str, t0: float, acc: _StreamAccumulator) -> None:
    """在飞请求的进度日志——没有它，只能在失败后才看见这次调用存在过。"""
    while True:
        await asyncio.sleep(HEARTBEAT_S)
        print(f'[llm] {model} @ {url} still waiting {time.perf_counter() - t0:.0f}s '
              f'(chunks={acc.chunks} chars={acc.chars})')


# ── 关闭 thinking ─────────────────────────────────────────────────────────────
#
# 同一个模型换一个供应商就换一种关法：vLLM/SGLang 自建的走 chat template，
# 智谱和 Anthropic 兼容层走 `thinking.type`，DashScope 把开关放在顶层，
# OpenRouter 又是另一个名字。单带 chat_template_kwargs 时实测会遇到「参数发了
# 但照样在想」——那不是模型不听话，是这个网关压根不看这个字段。
#
# 所以几种写法全带上：多出来的字段绝大多数网关直接忽略，而真有网关认了 400，
# 下面 _create() 会按报错点名的字段逐个撤掉并记在这个 client 上，不让一次
# 400 变成整次调用失败。
#
# 下面标的「实测」是 2026-10-14 在 Orin5 上对生产入口
# router.phanthy.com + zhipu/glm-5.2 逐个字段单独发、同一个 prompt 各 5 次的
# completion_tokens 中位数（2 = 只答数字，没思考；175 = 在想）：
#
#     不发任何字段                 175    3.09s
#     chat_template_kwargs 单发      2    0.81s   ← 管用
#     thinking.type 单发             2    0.71s   ← 管用
#     enable_thinking 单发           2    0.88s   ← 管用
#     thinking_budget 单发         207    3.66s   ← 这条线上无效
#     reasoning.enabled 单发       197    2.81s   ← 这条线上无效
#     reasoning_effort 单发          2    0.80s   ← 管用
#     六个一起发                     3    0.72s
#
# 两点结论。一是「发了但没关掉」是真的：这个入口对 thinking_budget 和
# reasoning.enabled 完全不响应，只发它们等于没发，而延迟差 4 倍。二是互不打架
# ——六个一起发和最好的单发同档，没有哪个字段把别的顶掉。
#
# 无效的那两个仍然留着：它们是给别的供应商的（DashScope、OpenRouter），在这个
# 入口只是被忽略，代价为零。别拿这张表当「哪些字段该删」的依据——它只测了一个
# 入口一个模型，换一条线结论就不一样，这正是要全发的原因。
THINKING_OFF_EXTRA = {
    # vLLM / SGLang 托管的 Qwen3、GLM 系列（经 chat template 透传）。实测管用
    "chat_template_kwargs": {"enable_thinking": False},
    # 智谱 GLM-4.5/4.6 的 OpenAI 接口、Anthropic 兼容网关。实测管用
    "thinking": {"type": "disabled"},
    # DashScope 及部分自建网关放在顶层的同名开关。实测管用
    "enable_thinking": False,
    # budget 形式：0 等价于不思考。实测对 router.phanthy.com 无效，留给别的线
    "thinking_budget": 0,
    # OpenRouter。同上，实测对 router.phanthy.com 无效
    "reasoning": {"enabled": False},
    # OpenAI 风格的 reasoning_effort，部分网关用它关。实测管用
    "reasoning_effort": "none",
}


# ── think_mode=True：开着思考，但只给最低限度 ─────────────────────────────────
#
# 「开 thinking」不等于「随它想多久」。这个 agent 的每一轮都卡在用户面前——
# 关与不关实测是 4 倍延迟差（见上表），而一轮里真正需要长链推理的只有少数。
# 所以 think_mode=True 时仍然把档位压到各家支持的最低：留下思考能力，不留
# 「想 2000 token 再开口」的尾巴。原先这里返回 {}，即完全不带参数，等于每家
# 网关各按自己的默认档跑——同一个配置在两个供应商上行为不同，且都是最贵的那档。
#
# 和关闭那组一样：没有通用参数，所以几种写法全发，多余的字段绝大多数网关忽略，
# 被 400 点名的由 _create() 逐个撤掉。两组走的是同一条降级路径。
#
# 几个取值上的坑：
#   * `reasoning_effort` 用 "low" 而不是 OpenAI gpt-5 才有的 "minimal"——
#     "low" 是各家都认的最低档，"minimal" 在别处会 400。
#   * `thinking.budget_tokens` 必须 >= 1024（Anthropic 兼容层的硬下限，更小
#     的值直接 400），所以 MIN_BUDGET 取 1024 而不是更激进的数。
#   * `reasoning` 里只给 effort，不同时给 max_tokens——OpenRouter 要求二选一。
#   * 智谱那条线只有 `thinking.type` 这一种开关，没有「低档」的说法，所以那边
#     能做到的就是 type=enabled 加一个它可能忽略的 budget_tokens。
#
# ── 实测，以及为什么这组参数的收益测不出来 ──────────────────────────────────
#
# 两轮都在 Orin5 上打生产入口 router.phanthy.com，2026-10-22。
#
# 第一轮是合成 prompt（一道水池题，baseline 与 MIN 交替采样，判据 completion_tokens
# 中位）：glm-5.3 各 12 次量到 129 → 100、3.23s → 2.38s；G53 各 25 次量到 128 → 99，
# 延迟 2.79s → 2.88s（无差别）。
#
# **第二轮用 llm_bench 回放本机真实请求日志，这组数字一个都没复现。** 15 条真实
# payload、8 组、串行轮换、逐条配对（判据是配对差的 95% bootstrap CI + 符号检验）：
#
#     组内对比（vs 不发任何字段）      延迟中位差    符号检验      判定
#     G52  这组六个一起发              +0.11s      p=0.302 (5:10)  不显著
#     G52  只发上限（不带 enable）     +0.11s      p=0.607 (6:9)   不显著
#     G52  只发 reasoning_effort:low   +0.51s      p=0.035 (3:12)  **显著更慢**
#     G53  这组六个一起发              -0.01s      p=0.607 (9:6)   不显著
#     G53  只发上限                    +0.47s      p=0.607 (6:9)   不显著
#     G53  只发 reasoning_effort:low   -0.05s      p=1.000 (8:7)   不显著
#
# **那一轮全程用的是 router 的别名模型名（G52/G53），而这些参数生效与否是按渠道
# 而定的——所以它量到的「不显著」是假阴性。** 重测时改成一个模型名一组，并把**思考
# token** 也记下来（llm_bench 原先只记 completion，而思考 token 已经算在里面，分不出
# 「答得长」和「想得久」——评测思考档位要分的恰恰是这两件事）。同样 10 条真实 payload
# 逐条配对：
#
#     模型名                参数         延迟 p50  延迟 p90  思考token均值  最大
#     G53（别名）           不加            5.59s   18.47s        186.6   1006
#     G53（别名）           effort=low      5.54s   18.25s        194.9    862  ← 无反应
#     G53（别名）           整组关闭        —— 400「该模型始终思考」，上游拒绝
#     zhipu/glm-5.3（直名） 不加            1.34s    9.47s        188.6   1401
#     zhipu/glm-5.3（直名） effort=low      1.12s    1.95s          6.7     58  ← 生效
#     G52（别名）           不加            1.98s    7.78s         86.8    475
#     G52（别名）           effort=low      3.08s    6.68s        100.6    388  ← 无反应
#     G52（别名）           整组关闭        2.24s    3.40s          0.0      0  ← 生效
#
# 读法：
#   * `reasoning_effort: low` 只在 `zhipu/glm-5.3` 这条渠道上生效，把思考 token 压掉
#     28 倍、p90 压掉 4.9 倍（最极端那条 payload：1401 token / 9.5s → 58 / 1.8s）。
#   * 同一个参数在 G52 上毫无反应——glm-5.2 这代没有档位的概念，只有开/关。而**整组
#     关闭参数在 G52 上是生效的**（思考 token 全 0），所以关闭和压档是两件独立的事，
#     各自在哪条渠道上管用要分别测。
#   * G53 这个别名既不吃 effort，又比直名慢一个量级（p50 5.59s vs 1.34s）；但它会把
#     关闭参数转给上游（上游回 400），所以不是「别名把参数吃掉了」那么简单。
#
# **收益全在尾部**：10 条里只有 3 条真的触发长推理，所以延迟中位几乎不动而 p90 和
# 均值差几倍——这件事要按 p90/均值判，看 p50 会得出「没用」。
#
# 三条由此而来的纪律：
#   * **用实际配置的那个模型名测，别用同族的另一个名字。** 生效与否取决于 new-api 侧
#     那条渠道，而不是模型本身；换个名字结论就变。
#   * 别把合成 prompt 的收益写成产品收益，也别把某条渠道上的「无差别」写成参数无效。
#   * 六个字段全发和只发 reasoning_effort 同档（思考 token 8.2 vs 8.6），那三个
#     enable 字段在这条线上既不抵消也不帮忙——所以全发是安全的。
#
# 另外，上游说「没收到 reasoning_effort」和「它生效了」可以同时为真：new-api 很可能
# 是**翻译**而不是透传，把它转成渠道自己的思考参数，所以上游日志里没有这个字面字段。
# 判据永远是思考 token，不是上游有没有看见这个键。
#
# 顺带量到的、比这组参数重要得多的一件事：同一批 payload 上 G52 比 G53 快 2.2–3.2s
# （多个组对 p<=0.035，其中一对 15:0）。选模型的收益比调 thinking 参数大一个量级。
#
# 另外两个采样上的坑，踩过所以记下来：
#   * 这个 router 的别名按请求分发到多个上游渠道，逐样本方差极大（同一组参数五次
#     采样拿到过 2 / 5 / 1227 / 1440 / 1485）。必须交替采样看中位，顺序跑完一组再
#     跑下一组量到的是时段漂移。
#   * n 不能小。G53 合成 prompt 各 12 次量到 125 → 138（「这组参数反而更费」），各
#     25 次量到 128 → 99；回放真实日志 12 条量到 MIN 慢 0.9s，15 条配对后变成无差别。
#     同机同入口，只是采样数和配对与否不同，结论的符号就反。n<20 且没配对，不要下
#     结论。
THINKING_MIN_BUDGET = 1024

THINKING_MIN_EXTRA = {
    # vLLM / SGLang 托管的 Qwen3、GLM 系列：chat template 里同时认开关和预算
    "chat_template_kwargs": {"enable_thinking": True,
                             "thinking_budget": THINKING_MIN_BUDGET},
    # 智谱 GLM-4.5/4.6/5.x 的 OpenAI 接口、Anthropic 兼容网关
    "thinking": {"type": "enabled", "budget_tokens": THINKING_MIN_BUDGET},
    # DashScope 及部分自建网关放在顶层的同名开关
    "enable_thinking": True,
    "thinking_budget": THINKING_MIN_BUDGET,
    # OpenRouter：effort 和 max_tokens 二选一，这里只给 effort
    "reasoning": {"enabled": True, "effort": "low"},
    # OpenAI 风格。"low" 是各家通认的最低档，"minimal" 只有 gpt-5 认
    "reasoning_effort": "low",
}


def _refuses_to_disable_thinking(err: str) -> bool:
    """400 的文案像不像「这个模型关不掉思考」。

    实测 2026-10-22，router.phanthy.com 的 G53 对整组关闭参数回
    `{"message":"该模型始终思考，不支持关闭思考；请使用 low、high 或 max。",
      "type":"upstream_error","param":"","code":"1210"}` —— **param 是空的**，
    所以按字段名撤的那条路匹配不到任何键，而 `_looks_like_unknown_param` 也不认
    这句中文（它找的是 parameter/unknown/unsupported 那类词，`"param":""` 里的
    param 不是 parameter 的子串）。结果是整个调用抛 BadRequestError —— 也就是说
    `model=G53` 配上 `think_mode=False`（而这是默认值）会让每一次 LLM 调用都失败。
    """
    if any(k in err for k in ('不支持关闭思考', '始终思考', '必须思考', '无法关闭思考')):
        return True
    low = err.lower()
    return any(k in low for k in ('always thinks', 'always reasoning',
                                  'cannot be disabled', "can't be disabled",
                                  'cannot disable thinking', 'does not support disabling',
                                  'thinking is always', 'reasoning is always'))


def _looks_like_unknown_param(err: str) -> bool:
    """400 的文案像不像「有个参数我不认识」。"""
    low = err.lower()
    if any(k in low for k in ('context length', 'context_length', 'maximum context',
                             'token limit', 'too many tokens')):
        return False
    return any(k in low for k in ('parameter', 'argument', 'field', 'property',
                                  'extra_body', 'unknown', 'unsupported',
                                  'unrecognized', 'not allowed', 'invalid request'))


# ── Client ────────────────────────────────────────────────────────────────────

class Client():
    def __init__(self):
        LOG_PATH.mkdir(parents=True, exist_ok=True)
        self._init_clients()

    def _init_clients(self):
        """从配置创建 OpenAI client 列表。"""
        self.client_list = [
            openai.AsyncOpenAI(
                base_url=config_it['url'],
                api_key=config_it['key'],
                max_retries=0,  # 由我们自己管理重试
                # 流式下 read 是「单次 socket 读」的上限，已经被上面那三道更紧的
                # 期限（首 token / chunk 间隔 / 总时长）盖住，留着只作兜底。
                timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=10.0),
                http_client=httpx.AsyncClient(
                    event_hooks={"request": [_log_request]},
                ),
            )
            for config_it in config.main['client']['llm']
            if config_it.get('key')  # 跳过未配置 credentials 的条目
        ]
        # 跟踪每个 endpoint 的健康状态
        self._endpoint_dead: list[bool] = [False] * len(self.client_list)

    async def __call__(self,
        message_list: list[dict],
        tool_list: list[dict],
        cancel_event: 'asyncio.Event | None' = None,
        reconsider_event: 'asyncio.Event | None' = None,
        model_override: 'str | None' = None,
    ) -> dict:

        def _extra(client, think_mode: bool) -> dict:
            # think_mode 只决定发哪一组：关掉，还是开着但压到最低档。两组共用
            # 同一条「被 400 点名就撤掉」的降级记账。
            #
            # 另一种情况是模型压根关不掉思考（G53 就是），这时它已经被下面的
            # _create() 记在 client 上，之后直接发最低档那组——既不再白吃一次
            # 400，也不至于退回「什么都不发」的默认档。
            if getattr(client, '_thinking_cannot_disable', False):
                think_mode = True
            base = THINKING_MIN_EXTRA if think_mode else THINKING_OFF_EXTRA
            # 之前被这个 endpoint 400 掉的字段不再发
            rejected = getattr(client, '_thinking_rejected', ())
            extra = {k: v for k, v in base.items() if k not in rejected}
            return {"extra_body": extra} if extra else {}

        async def _create(client, model, url, kwargs):
            """发起 create()，并对「网关不认某个字段」逐级降级后重试。

            每一轮要么严格减少 kwargs，要么走那个只做一次的 OFF→MIN 置换，
            所以循环必然收敛。
            """
            swapped_to_min = False
            while True:
                try:
                    return await client.chat.completions.create(**kwargs)
                except openai.BadRequestError as e:
                    err = str(e)
                    if 'stream_options' in kwargs and 'stream_options' in err:
                        print(f'[llm] {model} @ {url} rejects stream_options — '
                              f'retrying without usage')
                        client._no_stream_options = True
                        kwargs.pop('stream_options')
                        continue
                    extra = kwargs.get('extra_body') or {}
                    if extra and not swapped_to_min and _refuses_to_disable_thinking(err):
                        # 这个模型关不掉思考。退回「什么都不发」会让它按供应商
                        # 默认档跑（G53 上实测是最贵那档），所以换成最低档那组
                        # ——它接受的最低值正是报错里点的 low。记在 client 上，
                        # 这个 endpoint 之后不再试着关了。
                        swapped_to_min = True
                        client._thinking_cannot_disable = True
                        rejected = getattr(client, '_thinking_rejected', ())
                        kwargs['extra_body'] = {k: v for k, v in THINKING_MIN_EXTRA.items()
                                                if k not in rejected}
                        print(f'[llm] {model} @ {url} cannot disable thinking — '
                              f'retrying at the lowest reasoning setting')
                        continue
                    # 必须按词边界匹配：`thinking` 是 `thinking_budget` 的前缀，
                    # 子串匹配会因为后者被拒而把前者一起撤掉——而前者正是对智谱
                    # 那条线唯一管用的关法。
                    named = [k for k in extra
                             if re.search(rf'(?<![\w.]){re.escape(k)}(?![\w])', err)]
                    if named:
                        # 报错点了名：记在 client 上，这个 endpoint 以后都不发了
                        client._thinking_rejected = set(
                            getattr(client, '_thinking_rejected', ())) | set(named)
                        drop = named
                    elif extra and _looks_like_unknown_param(err):
                        # 没点名（有些网关只回一句 unsupported parameter）：把整组
                        # 关闭参数一起撤掉再试一次，但不记账——这个 400 也可能压根
                        # 与 thinking 无关，不该让该 endpoint 永久退回带 thinking。
                        #
                        # 要先确认它真像「参数不认识」：context overflow 在一些网关
                        # 上也是 400，无条件重发等于把一个 36k 的请求白发两遍。
                        drop = list(extra)
                    else:
                        raise
                    print(f'[llm] {model} @ {url} rejects {",".join(drop)} — retrying without')
                    extra = {k: v for k, v in extra.items() if k not in drop}
                    if extra:
                        kwargs['extra_body'] = extra
                    else:
                        kwargs.pop('extra_body', None)
                    continue

        async def _go_stream(client, model, think_mode: bool, attempt: int) -> dict:
            url = str(client.base_url)
            t0 = time.perf_counter()
            acc = _StreamAccumulator()
            first_timeout = _first_token_timeout(attempt)
            kwargs = dict(
                model=model,
                messages=message_list,
                tools=tool_list,
                max_tokens=10240,
                stream=True,
                **_extra(client, think_mode),
            )
            # include_usage 让最后一帧带 usage。个别网关不认这个字段，认了 400
            # 就记住并降级——usage 没了只是统计缺一行，不该让整次调用失败。
            if not getattr(client, '_no_stream_options', False):
                kwargs['stream_options'] = {'include_usage': True}

            hb = (asyncio.create_task(_heartbeat(model, url, t0, acc))
                  if HEARTBEAT_S > 0 else None)
            try:
                # create() 在 stream=True 下也要等到响应头才返回，这一段只有
                # httpx 的 120s read timeout 管着。不把它一起纳入首 token 预算，
                # 卡在响应头上的请求照样能干等两分钟——那正是这次要修的故障。
                async def _open():
                    return await _create(client, model, url, kwargs)

                try:
                    stream = await asyncio.wait_for(_open(), timeout=first_timeout)
                except asyncio.TimeoutError:
                    raise _timeout(
                        url, f'{model}: no response headers within '
                             f'{first_timeout:.0f}s') from None

                try:
                    await _drain_stream(stream, url, model, t0, acc, first_timeout)
                finally:
                    # 取消/超时时必须显式关流，否则连接一直挂在池里。
                    # shield：cancel_event 抢占时本任务正在被 cancel，裸 await
                    # 会立刻再抛 CancelledError，关流动作根本执行不到。
                    try:
                        await asyncio.shield(stream.close())
                    except Exception:
                        pass

                elapsed = time.perf_counter() - t0
                usage = _usage_dict(acc.usage, elapsed)
                think = f' | reasoning={len(acc.reasoning)}字' if acc.reasoning else ''
                if usage:
                    print(
                        f'[llm] {model} ok {elapsed:.2f}s | '
                        f'prompt={usage["prompt_tokens"]} completion={usage["completion_tokens"]} '
                        f'total={usage["total_tokens"]} cached={usage["cached_tokens"]} '
                        f'| stream chunks={acc.chunks}{think}'
                    )
                else:
                    print(f'[llm] {model} ok {elapsed:.2f}s | usage=N/A '
                          f'| stream chunks={acc.chunks}{think}')

                msg = _finalize_message(acc.message())
                if usage:
                    msg['_usage'] = usage
                return msg
            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f'[llm] {model} @ {url} failed after {elapsed:.2f}s: {type(e).__name__}: {e}')
                raise
            finally:
                # CancelledError 走不到上面的 except（它是 BaseException），
                # 心跳任务必须在 finally 里收掉，否则每次抢占都漏一个任务。
                if hb is not None:
                    hb.cancel()

        async def _go_blocking(client, model, think_mode: bool, attempt: int) -> dict:
            """非流式旧路径，保留给 LLM_STREAM=0 排查用。"""
            url = str(client.base_url)
            t0 = time.perf_counter()
            try:
                response = await _create(client, model, url, dict(
                    model=model,
                    messages=message_list,
                    tools=tool_list,
                    max_tokens=10240,
                    stream=False,
                    **_extra(client, think_mode),
                ))
                elapsed = time.perf_counter() - t0
                usage = _usage_dict(response.usage, elapsed)
                _r = _attr(response.choices[0].message, 'reasoning_content', '') or ''
                think = f' | reasoning={len(_r)}字' if _r else ''
                if usage:
                    print(
                        f'[llm] {model} ok {elapsed:.2f}s | '
                        f'prompt={usage["prompt_tokens"]} completion={usage["completion_tokens"]} '
                        f'total={usage["total_tokens"]} cached={usage["cached_tokens"]}{think}'
                    )
                else:
                    print(f'[llm] {model} ok {elapsed:.2f}s | usage=N/A{think}')
                try:
                    msg = response.choices[0].message.to_dict()
                except (KeyError, AttributeError, IndexError) as parse_err:
                    # Some models return non-standard response structures
                    # Fallback: extract what we can manually
                    m = response.choices[0].message
                    msg = {'role': 'assistant', 'content': getattr(m, 'content', '') or ''}
                    if hasattr(m, 'tool_calls') and m.tool_calls:
                        try:
                            msg['tool_calls'] = [tc.to_dict() for tc in m.tool_calls]
                        except Exception:
                            pass
                    print(f'[llm] WARNING: message.to_dict() failed ({parse_err}), using fallback parse')
                msg = _finalize_message(msg)
                _attach_reasoning(msg, _attr(response.choices[0].message,
                                             'reasoning_content', '') or '')
                # 附加 token 用量信息（内部字段，下划线前缀）
                if usage:
                    msg['_usage'] = usage
                return msg
            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f'[llm] {model} @ {url} failed after {elapsed:.2f}s: {type(e).__name__}: {e}')
                raise

        async def _go(client, model, think_mode: bool, attempt: int = 0) -> dict:
            if STREAM_ENABLED:
                return await _go_stream(client, model, think_mode, attempt)
            return await _go_blocking(client, model, think_mode, attempt)

        configs = config.main['client']['llm']
        last_error = None
        max_retries = 2  # 重试上限

        # model_override: select matching endpoints or override model on first endpoint
        _override_model = None
        if model_override:
            matched_indices = [i for i, c in enumerate(configs) if c.get('model') == model_override]
            if matched_indices:
                # Use only matching endpoints
                configs = [configs[i] for i in matched_indices]
                client_list = [self.client_list[i] for i in matched_indices]
                endpoint_dead = [self._endpoint_dead[i] for i in matched_indices]
            else:
                # Override model name, use all endpoints
                _override_model = model_override
                client_list = self.client_list
                endpoint_dead = self._endpoint_dead
        else:
            client_list = self.client_list
            endpoint_dead = self._endpoint_dead

        for attempt in range(max_retries + 1):
            # 筛选存活的 endpoint
            alive = [
                (i, client_list[i], configs[i])
                for i in range(len(client_list))
                if not endpoint_dead[i]
            ]
            if not alive:
                # 全部标记为 dead，重置后再试
                endpoint_dead = [False] * len(client_list)
                alive = [(i, client_list[i], configs[i]) for i in range(len(client_list))]

            # 竞速调用所有存活 endpoint
            task_list = [
                asyncio.create_task(_go(c, _override_model or cfg['model'],
                                        cfg.get('think_mode', False), attempt))
                for _, c, cfg in alive
            ]

            # 如果有 cancel_event / reconsider_event，加入哨兵 task 实现用户消息抢占
            cancel_task = None
            reconsider_task = None
            wait_tasks = list(task_list)
            if cancel_event:
                cancel_task = asyncio.create_task(cancel_event.wait())
                wait_tasks.append(cancel_task)
            if reconsider_event:
                reconsider_task = asyncio.create_task(reconsider_event.wait())
                wait_tasks.append(reconsider_task)

            done, pending = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()

            # 如果 cancel 先完成 → 中断当前 turn
            if cancel_task and cancel_task in done:
                for t in task_list:
                    t.cancel()
                from event.llm import TurnCancelled
                raise TurnCancelled("Interrupted by user message during LLM call")

            # reconsider 先完成 → 这次请求作废，但 turn 不结束，调用方原地重问
            if reconsider_task and reconsider_task in done:
                for t in task_list:
                    t.cancel()
                from event.llm import RoundReconsider
                raise RoundReconsider("Higher-priority input arrived during LLM call")

            # 检查是否有成功的
            for t in done:
                if not t.exception():
                    return t.result()

            # 所有 done 的都失败了，取第一个错误做分类
            error = next(iter(done)).exception()
            last_error = error
            kind, retry_after = _classify_error(error)

            print(f'[llm] error classified as {kind} (attempt {attempt + 1}/{max_retries + 1})')

            if kind == LLMErrorKind.BILLING:
                # 标记触发 402 的 endpoint 为 dead，切换到下一个
                for idx, _, cfg in alive:
                    endpoint_dead[idx] = True
                print(f'[llm] billing error — marked endpoint(s) dead, trying others')
                continue  # 立即重试剩余 endpoint

            if kind == LLMErrorKind.AUTH:
                # 认证错误不可恢复
                for idx, _, cfg in alive:
                    endpoint_dead[idx] = True
                print(f'[llm] auth error — marked endpoint(s) dead')
                continue

            if kind == LLMErrorKind.RATE_LIMIT:
                if attempt < max_retries:
                    wait = min(retry_after or 10.0, 30.0)
                    print(f'[llm] rate limited — waiting {wait:.1f}s before retry')
                    await asyncio.sleep(wait)
                    continue

            if kind == LLMErrorKind.SERVER_ERROR:
                if attempt < max_retries:
                    wait = retry_after or (3.0 * (attempt + 1))  # 递增退避
                    print(f'[llm] server error — waiting {wait:.1f}s before retry')
                    await asyncio.sleep(wait)
                    continue

            if kind == LLMErrorKind.TIMEOUT:
                if attempt < max_retries:
                    print(f'[llm] timeout — retrying immediately '
                          f'(first-token budget {_first_token_timeout(attempt + 1):.0f}s)')
                    continue

            if kind == LLMErrorKind.CONNECTION:
                if attempt < max_retries:
                    wait = retry_after or 2.0
                    print(f'[llm] connection failed — retrying in {wait:.1f}s (check network)')
                    await asyncio.sleep(wait)
                    continue
                else:
                    print(f'[llm] connection failed after {max_retries + 1} attempts — LLM unreachable, check network connectivity')

            if kind == LLMErrorKind.CONTEXT_OVERFLOW:
                # 上下文溢出：不重试，由调用方处理（需要压缩历史）
                print(f'[llm] context overflow — caller should compress history')
                raise error

            # UNKNOWN：不重试
            break

        raise last_error
