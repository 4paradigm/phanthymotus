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
        self.tool_calls: dict[int, dict] = {}
        self.finish_reason: str | None = None
        self.usage = None
        self.chunks = 0

    @property
    def chars(self) -> int:
        return sum(len(p) for p in self.content_parts) + sum(
            len(tc['function']['arguments']) for tc in self.tool_calls.values())

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

    def message(self) -> dict:
        msg: dict = {'role': 'assistant', 'content': ''.join(self.content_parts)}
        if self.tool_calls:
            msg['tool_calls'] = [self.tool_calls[i] for i in sorted(self.tool_calls)]
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

        def _extra(think_mode: bool) -> dict:
            if think_mode:
                return {}
            return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

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
                **_extra(think_mode),
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
                    try:
                        return await client.chat.completions.create(**kwargs)
                    except openai.BadRequestError as e:
                        if 'stream_options' not in kwargs or 'stream_options' not in str(e):
                            raise
                        print(f'[llm] {model} @ {url} rejects stream_options — '
                              f'retrying without usage')
                        client._no_stream_options = True
                        kwargs.pop('stream_options')
                        return await client.chat.completions.create(**kwargs)

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
                if usage:
                    print(
                        f'[llm] {model} ok {elapsed:.2f}s | '
                        f'prompt={usage["prompt_tokens"]} completion={usage["completion_tokens"]} '
                        f'total={usage["total_tokens"]} cached={usage["cached_tokens"]} '
                        f'| stream chunks={acc.chunks}'
                    )
                else:
                    print(f'[llm] {model} ok {elapsed:.2f}s | usage=N/A | stream chunks={acc.chunks}')

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
                response = await client.chat.completions.create(
                    model=model,
                    messages=message_list,
                    tools=tool_list,
                    max_tokens=10240,
                    stream=False,
                    **_extra(think_mode),
                )
                elapsed = time.perf_counter() - t0
                usage = _usage_dict(response.usage, elapsed)
                if usage:
                    print(
                        f'[llm] {model} ok {elapsed:.2f}s | '
                        f'prompt={usage["prompt_tokens"]} completion={usage["completion_tokens"]} '
                        f'total={usage["total_tokens"]} cached={usage["cached_tokens"]}'
                    )
                else:
                    print(f'[llm] {model} ok {elapsed:.2f}s | usage=N/A')
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
