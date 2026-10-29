"""
event_bus.py — 全局事件队列。

所有外部触发（MCP 推送、定时任务、用户直接调用）都经由此队列入，
Agent Loop 从此队列出，保证单一消费者串行处理。

事件格式：
    {
        'source':  str,    # 'mcp:<id>' / 'scheduler:<name>' / 'user'
        'text':    str,    # 已转化为自然语言的描述，LLM 直接看
        'payload': dict,   # 原始数据（可选），供工具调用或前端可视化用
        'ts':      float,  # unix timestamp
    }
"""

import asyncio
import time

_queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
_recent: list[dict]   = []    # 最近 N 条事件，用于 L2 环境快照


async def enqueue(source: str, text: str, payload: dict | None = None) -> None:
    event = {
        'source':  source,
        'text':    text,
        'payload': payload or {},
        'ts':      time.time(),
    }
    await _queue.put(event)
    _recent.append(event)
    if len(_recent) > 100:
        _recent.pop(0)


async def dequeue() -> dict:
    return await _queue.get()


def recent(n: int = 10) -> list[dict]:
    """返回最近 n 条事件（从旧到新）"""
    return list(_recent[-n:])
