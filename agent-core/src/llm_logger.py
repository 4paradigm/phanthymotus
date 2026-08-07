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

    def _resume_current_files(self):
        """启动时检查是否有未满的 JSONL 文件可以续写。"""
        data_dir = pathlib.Path(self._cfg['data_dir'])
        batch_size = self._cfg['batch_size']

        # Find latest request file
        req_files = sorted(data_dir.glob('llm_request_*.jsonl'))
        if req_files:
            last = req_files[-1]
            count = sum(1 for _ in last.open('r', encoding='utf-8'))
            if count < batch_size:
                self._req_file = last
                self._req_count = count

        # Find latest response file
        resp_files = sorted(data_dir.glob('llm_response_*.jsonl'))
        if resp_files:
            last = resp_files[-1]
            count = sum(1 for _ in last.open('r', encoding='utf-8'))
            if count < batch_size:
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

    def _append_request(self, line: str):
        with self._lock:
            batch_size = self._cfg['batch_size']
            if self._req_file is None or self._req_count >= batch_size:
                self._req_file = self._new_file('llm_request_')
                self._req_count = 0
                self._rotate_data_files('llm_request_')
            with self._req_file.open('a', encoding='utf-8') as f:
                f.write(line + '\n')
            self._req_count += 1

    def _append_response(self, line: str):
        with self._lock:
            batch_size = self._cfg['batch_size']
            if self._resp_file is None or self._resp_count >= batch_size:
                self._resp_file = self._new_file('llm_response_')
                self._resp_count = 0
                self._rotate_data_files('llm_response_')
            with self._resp_file.open('a', encoding='utf-8') as f:
                f.write(line + '\n')
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
