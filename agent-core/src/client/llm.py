import typing
import base64
import pathlib
import asyncio
import re
import openai
import httpx
import time
import json

import config

LOG_PATH = pathlib.Path('./resource/log')


async def _log_request(request: httpx.Request):
    """httpx event hook: dump the real HTTP request body to disk for debugging."""
    if request.content:
        body = json.loads(request.content)
        model = body.get('model', 'unknown')
        path = LOG_PATH / f'llm_request_{model}.json'
        path.write_text(json.dumps(body, ensure_ascii=False, indent=2))


# ── 错误分类 ──────────────────────────────────────────────────────────────────

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
        model_override: 'str | None' = None,
    ) -> dict:

        async def _go(client, model, think_mode: bool) -> dict:
            url = str(client.base_url)
            t0 = time.perf_counter()
            try:
                extra = {}
                if not think_mode:
                    extra["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": False},
                    }

                response = await client.chat.completions.create(
                    model=model,
                    messages=message_list,
                    tools=tool_list,
                    max_tokens=10240,
                    stream=False,
                    **extra,
                )
                elapsed = time.perf_counter() - t0
                # Performance log: latency + token usage
                usage = response.usage
                if usage:
                    cached = getattr(usage, 'prompt_tokens_details', None)
                    cached_tokens = getattr(cached, 'cached_tokens', 0) if cached else 0
                    print(
                        f'[llm] {model} ok {elapsed:.2f}s | '
                        f'prompt={usage.prompt_tokens} completion={usage.completion_tokens} '
                        f'total={usage.total_tokens} cached={cached_tokens}'
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
                # OpenAI SDK 可能生成 tool_calls: None，清理以避免下游迭代报错
                if 'tool_calls' in msg and msg['tool_calls'] is None:
                    del msg['tool_calls']
                # glm 有时返回 tool_calls 内部缺少必要字段，清理无效条目
                if 'tool_calls' in msg and isinstance(msg['tool_calls'], list):
                    msg['tool_calls'] = [
                        tc for tc in msg['tool_calls']
                        if isinstance(tc, dict) and 'function' in tc and 'name' in tc.get('function', {})
                    ]
                    if not msg['tool_calls']:
                        del msg['tool_calls']
                # 清理模型泄漏的 think 标签残留
                if msg.get('content'):
                    msg['content'] = re.sub(r'</?think>', '', msg['content']).strip()
                # 附加 token 用量信息（内部字段，下划线前缀）
                if usage:
                    msg['_usage'] = {
                        'prompt_tokens': usage.prompt_tokens,
                        'completion_tokens': usage.completion_tokens,
                        'total_tokens': usage.total_tokens,
                        'cached_tokens': cached_tokens,
                        'elapsed_s': round(elapsed, 2),
                        'model': model,
                    }
                else:
                    msg['_usage'] = {'elapsed_s': round(elapsed, 2), 'model': model}
                return msg
            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f'[llm] {model} @ {url} failed after {elapsed:.2f}s: {type(e).__name__}: {e}')
                raise

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
                asyncio.create_task(_go(c, _override_model or cfg['model'], cfg.get('think_mode', False)))
                for _, c, cfg in alive
            ]

            # 如果有 cancel_event，加入哨兵 task 实现用户消息抢占
            cancel_task = None
            wait_tasks = list(task_list)
            if cancel_event:
                cancel_task = asyncio.create_task(cancel_event.wait())
                wait_tasks.append(cancel_task)

            done, pending = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()

            # 如果 cancel 先完成 → 中断当前 turn
            if cancel_task and cancel_task in done:
                for t in task_list:
                    t.cancel()
                from event.llm import TurnCancelled
                raise TurnCancelled("Interrupted by user message during LLM call")

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
                    print(f'[llm] timeout — retrying immediately')
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
