"""HTTP 传输 + 错误分类。

有 httpx 就用 httpx（带连接复用，更贴近生产的 openai SDK 路径）；没有就退回
stdlib urllib，这样系统 python3（容器里那个没装 httpx 的 3.10）也能跑。

错误分类刻意沿用生产 `src/client/llm.py:_classify_error`（llm.py:46-96）的词汇，
这样报告里的 server_error 和 `docker logs` 里的
`[llm] error classified as server_error` 说的是同一件事，不用做心算翻译。
"""
import json
import time
import urllib.error
import urllib.request

try:
    import httpx
except ImportError:
    httpx = None


class Kind:
    OK = 'ok'
    RATE_LIMIT = 'rate_limit'
    BILLING = 'billing'
    SERVER_ERROR = 'server_error'
    CONTEXT_OVERFLOW = 'context_overflow'
    AUTH = 'auth'
    TIMEOUT = 'timeout'
    CONNECTION = 'connection'
    BAD_REQUEST = 'bad_request'
    MODEL_UNAVAILABLE = 'model_unavailable'
    UNKNOWN = 'unknown'


def classify(status: int | None, body: str, exc: Exception | None = None) -> str:
    low = (body or '').lower()

    if exc is not None and status is None:
        name = type(exc).__name__.lower()
        if 'timeout' in name:
            return Kind.TIMEOUT
        return Kind.CONNECTION

    if status == 429:
        return Kind.RATE_LIMIT
    if status == 402:
        return Kind.BILLING
    if status in (401, 403):
        return Kind.AUTH

    # 「模型不存在 / 无可用渠道」单独成类：它和普通 5xx 的处置完全不同——
    # 前者是配置或供应商侧的能力缺失，重试没有意义。orin6 上 router 对
    # glm-5.3 连续返回 503「无可用渠道」，混进 server_error 会被误读成过载。
    if any(k in low for k in ('model_not_found', 'model_not_available',
                              '无可用渠道', 'not available', 'unknown model')):
        return Kind.MODEL_UNAVAILABLE

    if status in (500, 502, 503, 504, 529):
        return Kind.SERVER_ERROR

    if any(k in low for k in ('context length', 'context_length', 'too many tokens',
                              'maximum context', 'token limit')):
        return Kind.CONTEXT_OVERFLOW

    if status == 400:
        return Kind.BAD_REQUEST
    if status and status >= 400:
        return Kind.UNKNOWN
    return Kind.UNKNOWN


class Transport:
    """同步、串行的 POST。串行是刻意的：并发会让各组互相争带宽和配额，
    污染的正是我们要测的单请求延迟。"""

    def __init__(self, timeout_s: float = 180.0):
        self.timeout_s = timeout_s
        self._client = None
        if httpx is not None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(connect=10.0, read=timeout_s,
                                      write=30.0, pool=10.0),
            )

    @property
    def backend(self) -> str:
        return 'httpx' if self._client is not None else 'urllib'

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    def post_json(self, url: str, key: str, payload: dict) -> dict:
        """发一次请求，返回结构化结果。**永不抛异常** —— 失败也是数据。

        评测跑到一半因为一个 500 就整批中断，是这类工具最没用的失败方式：
        既没有结果，也没有失败样本可分析。
        """
        headers = {'Content-Type': 'application/json',
                   'Authorization': f'Bearer {key}'}
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        t0 = time.perf_counter()

        try:
            if self._client is not None:
                r = self._client.post(url, content=body, headers=headers)
                status, text = r.status_code, r.text
            else:
                req = urllib.request.Request(url, data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    status = resp.status
                    text = resp.read().decode('utf-8', 'replace')
        except urllib.error.HTTPError as e:
            text = e.read().decode('utf-8', 'replace')
            return self._fail(e.code, text, None, time.perf_counter() - t0)
        except Exception as e:  # 连接失败、超时、TLS 错误……
            return self._fail(None, str(e), e, time.perf_counter() - t0)

        elapsed = time.perf_counter() - t0
        if status != 200:
            return self._fail(status, text, None, elapsed)

        try:
            data = json.loads(text)
        except ValueError:
            return self._fail(200, text[:400], None, elapsed,
                              kind=Kind.UNKNOWN, note='200 但响应体不是 JSON')

        # 有些网关把错误塞进 200 的响应体里。当成功统计会凭空拉高成功率，
        # 而且这些「成功」通常极快，还会把延迟中位数拉低。
        if isinstance(data, dict) and data.get('error'):
            return self._fail(200, json.dumps(data['error'], ensure_ascii=False)[:400],
                              None, elapsed, note='200 响应体内含 error')

        usage = (data.get('usage') or {}) if isinstance(data, dict) else {}
        details = usage.get('prompt_tokens_details') or {}
        choices = data.get('choices') or [{}] if isinstance(data, dict) else [{}]
        return {
            'ok': True, 'kind': Kind.OK, 'status': 200, 'elapsed': elapsed,
            'prompt': usage.get('prompt_tokens', 0) or 0,
            'completion': usage.get('completion_tokens', 0) or 0,
            'cached': details.get('cached_tokens', 0) or 0,
            'finish': (choices[0] or {}).get('finish_reason'),
            'err': None, 'detail': None,
        }

    @staticmethod
    def _fail(status, text, exc, elapsed, kind=None, note=None) -> dict:
        detail = (text or '')[:300]
        if note:
            detail = f'{note}: {detail}'
        return {
            'ok': False, 'kind': kind or classify(status, text, exc),
            'status': status, 'elapsed': elapsed,
            'prompt': 0, 'completion': 0, 'cached': 0, 'finish': None,
            'err': f'HTTP {status}' if status else type(exc).__name__ if exc else 'error',
            'detail': detail,
        }

    def list_models(self, url: str, key: str) -> list[str] | None:
        """GET /models。取不到返回 None（表示「未知」，不等于「空」）。"""
        req = urllib.request.Request(
            f'{url}/models', headers={'Authorization': f'Bearer {key}'})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8', 'replace'))
            return [m['id'] for m in data.get('data', []) if isinstance(m, dict) and 'id' in m]
        except Exception:
            return None
