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
    """httpx event hook: dump the real HTTP request body to disk."""
    if request.content:
        body = json.loads(request.content)
        model = body.get('model', 'unknown')
        path = LOG_PATH / f'llm_request_{model}.json'
        path.write_text(json.dumps(body, ensure_ascii=False, indent=2))
        # Print curl equivalent for debugging
        headers = ' '.join(f"-H '{k}: {v}'" for k, v in request.headers.items()
                          if k.lower() not in ('host', 'content-length'))
        curl = f"curl -X {request.method} '{request.url}' {headers} -d @{path}"
        print(f'[llm-curl] {curl}')


# ── 错误分类 ──────────────────────────────────────────────────────────────────

class LLMErrorKind:
    RATE_LIMIT      = 'rate_limit'       # 429
    BILLING         = 'billing'          # 402
    SERVER_ERROR    = 'server_error'     # 500/502/503/529
    CONTEXT_OVERFLOW = 'context_overflow' # 上下文溢出
    AUTH            = 'auth'             # 401/403
    TIMEOUT         = 'timeout'          # 超时
    UNKNOWN         = 'unknown'


def _classify_error(e: Exception) -> tuple[str, float | None]:
    """分类 LLM 调用错误，返回 (kind, retry_after_seconds | None)。"""
    status = getattr(e, 'status_code', None)
    body_msg = str(e).lower()

    if isinstance(e, (asyncio.TimeoutError, httpx.TimeoutException, openai.APITimeoutError)):
        return LLMErrorKind.TIMEOUT, 5.0

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
                timeout=120.0,
                http_client=httpx.AsyncClient(
                    event_hooks={"request": [_log_request]},
                ),
            )
            for config_it in config.main['client']['llm']
        ]
        # 跟踪每个 endpoint 的健康状态
        self._endpoint_dead: list[bool] = [False] * len(self.client_list)

    async def __call__(self,
        message_list: list[dict],
        tool_list: list[dict],
    ) -> dict:

        async def _go(client, model) -> dict:
            url = str(client.base_url)
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=message_list,
                    tools=tool_list,
                    max_tokens=10240,
                    stream=False,
                    extra_body={"thinking": {"type": "disabled"}, "enable_thinking": False},
                )
                print(f'[llm] {model} ok ({time.perf_counter():.1f}s)')
                msg = response.choices[0].message.to_dict()
                # 清理模型泄漏的 think 标签残留
                if msg.get('content'):
                    msg['content'] = re.sub(r'</?think>', '', msg['content']).strip()
                return msg
            except Exception as e:
                print(f'[llm] {model} @ {url} failed: {type(e).__name__}: {e}')
                raise

        configs = config.main['client']['llm']
        last_error = None
        max_retries = 2  # 重试上限

        for attempt in range(max_retries + 1):
            # 筛选存活的 endpoint
            alive = [
                (i, self.client_list[i], configs[i])
                for i in range(len(self.client_list))
                if not self._endpoint_dead[i]
            ]
            if not alive:
                # 全部标记为 dead，重置后再试
                self._endpoint_dead = [False] * len(self.client_list)
                alive = [(i, self.client_list[i], configs[i]) for i in range(len(self.client_list))]

            # 竞速调用所有存活 endpoint
            task_list = [
                asyncio.create_task(_go(c, cfg['model']))
                for _, c, cfg in alive
            ]

            done, pending = await asyncio.wait(task_list, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()

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
                    self._endpoint_dead[idx] = True
                print(f'[llm] billing error — marked endpoint(s) dead, trying others')
                continue  # 立即重试剩余 endpoint

            if kind == LLMErrorKind.AUTH:
                # 认证错误不可恢复
                for idx, _, cfg in alive:
                    self._endpoint_dead[idx] = True
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

            if kind == LLMErrorKind.CONTEXT_OVERFLOW:
                # 上下文溢出：不重试，由调用方处理（需要压缩历史）
                print(f'[llm] context overflow — caller should compress history')
                raise error

            # UNKNOWN：不重试
            break

        raise last_error
