"""
collector.py — 信息整理器。

职责：
  - 从 event_bus 持续消费事件，在 buffer 中积累
  - 同一 source 限流（1 条/秒，保留最新），超过 max_window 的事件 FIFO 丢弃
  - agent loop 空闲时通过 next_trigger() 按需快照当前 buffer：
    取出此刻累积的全部事件、清空 buffer、格式化为 XML 返回。
    这样大模型每次拿到的都是「当下」的状态快照，不会因推理耗时
    而读到很久以前积压的旧批次。
"""

import asyncio
import datetime
import time
from collections import deque

import config
import event_bus


_buffer: deque = deque()
# Per-source throttle: source → timestamp of last accepted event
_last_accepted: dict[str, float] = {}
_THROTTLE_INTERVAL = 1.0  # 每个 source 最多 1 条/秒
_last_trigger_ts: float = 0.0  # 上次 next_trigger 触发时间（用于最小间隔防抖）
_data_event: asyncio.Event = asyncio.Event()  # buffer 有数据时置位，唤醒等待的 next_trigger


def _get_interval_ms() -> int:
    return config.main.get('event', {}).get('llm', {}).get('trigger_interval_ms', 1000)


def _get_max_window() -> int:
    return config.main.get('event', {}).get('llm', {}).get('collector_max_window', 20)


def _format_batch(events: list[dict]) -> str:
    """将事件列表格式化为堆叠的 <event> XML。"""
    lines = []
    for ev in events:
        ts = datetime.datetime.fromtimestamp(ev['ts']).strftime('%Y-%m-%dT%H:%M:%S')
        source = ev.get('source', 'unknown')
        text = ev.get('text', '')
        lines.append(f'<event source="{source}" ts="{ts}">\n{text}\n</event>')
    return '\n'.join(lines)


async def _drain_loop():
    """后台任务：持续从 event_bus 消费事件存入 buffer，per-source 限流（1条/秒，保留最新）。"""
    max_window = _get_max_window()
    while True:
        ev = await event_bus.dequeue()
        source = ev.get('source', 'unknown')
        now = ev.get('ts', time.time())

        last_ts = _last_accepted.get(source, 0)
        if now - last_ts < _THROTTLE_INTERVAL:
            # 同 source 在 1s 内：替换 buffer 中该 source 的最后一条（保留最新）
            for i in range(len(_buffer) - 1, -1, -1):
                if _buffer[i].get('source') == source:
                    _buffer[i] = ev
                    break
            else:
                # 未找到（极端情况），直接 append
                _buffer.append(ev)
        else:
            _last_accepted[source] = now
            _buffer.append(ev)

        # FIFO 丢弃超过窗口的旧事件
        while len(_buffer) > max_window:
            _buffer.popleft()

        # 通知等待中的 next_trigger：buffer 有数据可消费
        if _buffer:
            _data_event.set()


def start():
    """启动 collector 后台任务（在 lifespan 中调用）。"""
    asyncio.ensure_future(_drain_loop())
    print(f'[collector] started: interval={_get_interval_ms()}ms, max_window={_get_max_window()}')


async def next_trigger() -> dict:
    """阻塞等待并按需快照。

    当 buffer 有内容、且距上次触发已达最小间隔时，取出此刻累积的
    全部事件、清空 buffer，格式化为合成 trigger event 返回。

    与旧的固定频率定时生产不同：这里在 agent 空闲时才快照，因此
    大模型每次看到的都是最新状态，不会积压、回放陈旧批次。
    """
    global _last_trigger_ts
    while True:
        if not _buffer:
            _data_event.clear()
            await _data_event.wait()
            continue

        # 防抖：保证两次触发的最小间隔；agent 推理慢于 interval 时此值为负，立即触发
        interval = _get_interval_ms() / 1000.0
        wait = _last_trigger_ts + interval - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
            continue  # 醒来后重新读取最新 buffer

        # 快照 + 清空（中间无 await，单线程 asyncio 下天然原子，无需加锁）
        batch = list(_buffer)
        _buffer.clear()
        _last_trigger_ts = time.time()

        return {
            'source': 'collector',
            'text': _format_batch(batch),
            'payload': {'event_count': len(batch), 'sources': [e['source'] for e in batch]},
            'ts': max(e['ts'] for e in batch),  # 取最新事件时间，反映「当下」
        }
