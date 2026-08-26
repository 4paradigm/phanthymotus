"""
llm_logger.py — LLM 请求/回复持久化日志。

功能：
  - 每次 LLM 调用立即持久化请求和回复（防异常关机丢数据）
  - 每 500 条自动切分 JSONL 文件（训练数据准备）
  - 按 agent 类型分目录保存最新记录（快速查阅）
  - 循环删除管理存储空间

目录结构：
  resource/llm_data/              → JSONL 打包文件
  resource/llm_recent_request/    → 最新记录（按 agent_type 分子目录）
"""

import json
import os
import pathlib
import threading
import time
from datetime import datetime, timezone, timedelta

import config

_TZ_CN = timezone(timedelta(hours=8))
_SEPARATOR = '=' * 64


def _get_config() -> dict:
    defaults = {
        'enabled': True,
        'data_dir': './resource/llm_data',
        'recent_dir': './resource/llm_recent_request',
        'batch_size': 500,
        'max_records': 50000,
        'recent_max_per_dir': 100,
    }
    cfg = config.main.get('llm_logger', {})
    return {**defaults, **cfg}


class LLMLogger:
    def __init__(self):
        self._cfg = _get_config()
        self._lock = threading.Lock()
        self._ensure_dirs()
        # Current active JSONL files (append mode)
        self._req_file: pathlib.Path | None = None
        self._resp_file: pathlib.Path | None = None
        self._req_count = 0
        self._resp_count = 0
        self._resume_current_files()

    def _ensure_dirs(self):
        pathlib.Path(self._cfg['data_dir']).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self._cfg['recent_dir']).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _scan_and_repair(path: pathlib.Path) -> 'int | None':
        """完整记录条数；顺便截掉尾部残缺记录。不可读时返回 None（不续写）。

        The docstring at the top of this module promises the JSONL survives an
        unclean shutdown, but the tail record is exactly what an unclean
        shutdown loses: R1 ended up with a 6,561,792-byte file (1602 × 4096 —
        the last page never reached disk) stopping mid UTF-8 sequence. Two
        things went wrong and both are fixed here:

        1. Counting used `open('r', encoding='utf-8')`, so a partial multi-byte
           character raised UnicodeDecodeError. We count `b'\\n'` in bytes
           instead — the count never depends on the contents being decodable.
        2. The broken tail stayed on disk forever. We now truncate back to the
           last *complete, parseable* record, which makes the file valid JSONL
           again and safe to append to. Appending after a partial line would
           otherwise silently produce one unparseable record in the training
           data — the failure mode that matters once this file is consumed
           rather than just written.

        Note a truncated write is not always undecodable: a record cut at an
        ASCII byte (`{"a": 1`) decodes fine and is still not JSON. So the last
        complete line is parsed, not merely decoded.
        """
        try:
            data = path.read_bytes()
        except OSError as e:
            print(f'[llm_logger] cannot read {path.name} ({type(e).__name__}: {e}) '
                  f'— starting a new file')
            return None
        if not data:
            return 0

        cut = data.rfind(b'\n')
        # Bytes after the last newline are an unterminated record by definition.
        dropped = len(data) - (cut + 1)
        lines = data[:cut + 1].split(b'\n')[:-1] if cut >= 0 else []

        # Walk back over trailing records that are not valid JSON.
        while lines:
            try:
                json.loads(lines[-1].decode('utf-8'))
                break
            except (UnicodeDecodeError, ValueError):
                dropped += len(lines.pop()) + 1

        if dropped:
            keep = sum(len(ln) + 1 for ln in lines)
            try:
                with path.open('r+b') as f:
                    f.truncate(keep)
            except OSError as e:
                print(f'[llm_logger] {path.name} has a {dropped}-byte partial tail and '
                      f'cannot be repaired ({type(e).__name__}: {e}) — starting a new file')
                return None
            print(f'[llm_logger] repaired {path.name}: dropped {dropped} bytes of partial '
                  f'tail, {len(lines)} complete records kept')
        return len(lines)

    def _resume_current_files(self):
        """启动时检查是否有未满的 JSONL 文件可以续写。"""
        data_dir = pathlib.Path(self._cfg['data_dir'])
        batch_size = self._cfg['batch_size']

        # Find latest request file
        req_files = sorted(data_dir.glob('llm_request_*.jsonl'))
        if req_files:
            last = req_files[-1]
            count = self._scan_and_repair(last)
            if count is not None and count < batch_size:
                self._req_file = last
                self._req_count = count

        # Find latest response file
        resp_files = sorted(data_dir.glob('llm_response_*.jsonl'))
        if resp_files:
            last = resp_files[-1]
            count = self._scan_and_repair(last)
            if count is not None and count < batch_size:
                self._resp_file = last
                self._resp_count = count

    def _new_file(self, prefix: str) -> pathlib.Path:
        ts = datetime.now(_TZ_CN).strftime('%Y%m%d_%H%M%S')
        return pathlib.Path(self._cfg['data_dir']) / f'{prefix}{ts}.jsonl'

    # ── Public API ────────────────────────────────────────────────────────────

    async def log_request(self, request_id: str, trace_id: str,
                          caller_info: dict | None, message_list: list[dict],
                          tool_list: list[dict], model: str):
        if not self._cfg.get('enabled'):
            return
        record = {
            'request_id': request_id,
            'trace_id': trace_id,
            'agent_type': (caller_info or {}).get('agent_type', 'unknown'),
            'model': model,
            'messages': message_list,
            'tools': tool_list,
            'ts': time.time(),
        }
        line = json.dumps(record, ensure_ascii=False, separators=(',', ':'))
        self._append_request(line)

        # 暂存到实例，供 log_response 写 recent 文件
        self._last_request = record

    async def log_response(self, request_id: str, trace_id: str,
                           caller_info: dict | None, response: dict):
        if not self._cfg.get('enabled'):
            return
        record = {
            'request_id': request_id,
            'trace_id': trace_id,
            'agent_type': (caller_info or {}).get('agent_type', 'unknown'),
            'role': response.get('role', 'assistant'),
            'content': response.get('content'),
            'tool_calls': response.get('tool_calls'),
            'usage': response.get('_usage'),
            'ts': time.time(),
        }
        line = json.dumps(record, ensure_ascii=False, separators=(',', ':'))
        self._append_response(line)

        # Write recent file
        self._write_recent(request_id, caller_info, response)

    # ── Immediate append to JSONL ─────────────────────────────────────────────

    @staticmethod
    def _append_line(path: pathlib.Path, line: str) -> None:
        """Append one record, durably, as a single write.

        `line` must not contain a newline — one record is one line, and an
        embedded newline would split it into two unparseable ones. `json.dumps`
        escapes newlines by default, so this only guards against a future caller
        passing pre-formatted text.

        The `fsync` is what the module docstring's "防异常关机丢数据" actually
        requires: closing the file only hands the bytes to the page cache, which
        is why a power loss on R1 left a record cut at a 4096-byte boundary.
        One fsync per LLM call is a few milliseconds against a multi-second
        request, so the cost is not measurable here.
        """
        payload = line.replace('\n', '\\n') + '\n'
        with path.open('a', encoding='utf-8') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())

    def _append_request(self, line: str):
        with self._lock:
            batch_size = self._cfg['batch_size']
            if self._req_file is None or self._req_count >= batch_size:
                self._req_file = self._new_file('llm_request_')
                self._req_count = 0
                self._rotate_data_files('llm_request_')
            self._append_line(self._req_file, line)
            self._req_count += 1

    def _append_response(self, line: str):
        with self._lock:
            batch_size = self._cfg['batch_size']
            if self._resp_file is None or self._resp_count >= batch_size:
                self._resp_file = self._new_file('llm_response_')
                self._resp_count = 0
                self._rotate_data_files('llm_response_')
            self._append_line(self._resp_file, line)
            self._resp_count += 1

    def _rotate_data_files(self, prefix: str):
        """文件数 * batch_size > max_records 时删除最早文件。"""
        data_dir = pathlib.Path(self._cfg['data_dir'])
        files = sorted(data_dir.glob(f'{prefix}*.jsonl'))
        max_files = self._cfg['max_records'] // self._cfg['batch_size']
        while len(files) > max_files:
            files[0].unlink()
            files.pop(0)

    # ── Recent Files ──────────────────────────────────────────────────────────

    def _write_recent(self, request_id: str, caller_info: dict | None, response: dict):
        agent_type = (caller_info or {}).get('agent_type', 'unknown')
        recent_dir = pathlib.Path(self._cfg['recent_dir']) / agent_type
        recent_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(_TZ_CN).strftime('%y%m%d_%H%M%S')
        short_id = request_id[:8]
        filename = f'{ts}_{short_id}.txt'
        filepath = recent_dir / filename

        # Build file content: request + separator + response
        req_data = getattr(self, '_last_request', None)
        req_json = json.dumps(req_data, ensure_ascii=False, indent=2) if req_data else '{}'
        resp_json = json.dumps(response, ensure_ascii=False, indent=2, default=str)

        filepath.write_text(f'{req_json}\n{_SEPARATOR}\n{resp_json}\n', encoding='utf-8')

        # Rotate: keep max recent_max_per_dir files
        self._rotate_recent(recent_dir)

    def _rotate_recent(self, directory: pathlib.Path):
        """保持目录内最多 N 个文件，删除最早的。"""
        max_files = self._cfg['recent_max_per_dir']
        files = sorted(directory.glob('*.txt'))
        while len(files) > max_files:
            files[0].unlink()
            files.pop(0)


# ── Module-level singleton ────────────────────────────────────────────────────

_instance: LLMLogger | None = None


def get_logger() -> LLMLogger:
    global _instance
    if _instance is None:
        _instance = LLMLogger()
    return _instance
