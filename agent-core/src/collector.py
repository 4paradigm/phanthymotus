"""
collector.py — 信息整理器。

职责：
  - 从 event_bus 持续消费事件，在触发间隔内积累
  - 超过 max_window 的事件 FIFO 丢弃
  - 间隔到达时，将积累的事件批量格式化为 XML 并输出
  - agent loop 通过 next_trigger() 获取下一批格式化事件
"""

import asyncio
import datetime
import time
from collections import deque

import config
import event_bus


_buffer: deque = deque()
_output: asyncio.Queue = asyncio.Queue(maxsize=64)
_task: asyncio.Task | None = None
# Per-source throttle: source → timestamp of last accepted event
_last_accepted: dict[str, float] = {}
_THROTTLE_INTERVAL = 1.0  # 每个 source 最多 1 条/秒


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


async def _trigger_loop():
    """后台任务：每隔 trigger_interval 检查 buffer，有内容则格式化并放入 output。"""
    while True:
        interval = _get_interval_ms() / 1000.0
        await asyncio.sleep(interval)

        if not _buffer:
            continue

        # 取出当前所有积累的事件
        batch = list(_buffer)
        _buffer.clear()

        formatted = _format_batch(batch)
        # 构造一个合成的 trigger 对象供 agent loop 使用
        trigger = {
            'source': 'collector',
            'text': formatted,
            'payload': {'event_count': len(batch), 'sources': [e['source'] for e in batch]},
            'ts': batch[-1]['ts'],  # 使用最后一个事件的时间戳
        }
        await _output.put(trigger)


def start():
    """启动 collector 后台任务（在 lifespan 中调用）。"""
    global _task
    loop = asyncio.get_event_loop()
    asyncio.ensure_future(_drain_loop())
    asyncio.ensure_future(_trigger_loop())
    print(f'[collector] started: interval={_get_interval_ms()}ms, max_window={_get_max_window()}')


async def next_trigger() -> dict:
    """阻塞等待下一批格式化事件。返回合成的 trigger event dict。"""
    return await _output.get()
