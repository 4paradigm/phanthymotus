"""
collector.py — 双队列事件收集器。

架构：
  - P>0 事件（ASR/message/channel）→ 立即送 main agent，或 busy 时按模式处理：
    - steer: 推入 steering_queue，agent loop 在 tool batch 间消费
    - interrupt: 触发 cancel_event，中止当前 turn
    - followup: 暂存 _priority_pending，等 turn 结束后 drain
  - P=0 事件（sensor/scheduler 等）→ 独立节奏送 bg subagent，main agent 永远不看到
  - Ring buffer 保留所有事件（供 raw_input_info 按需查询）
  - 语音 barge-in 检测：ASR 事件 duration_ms < 阈值时视为 backchannel 丢弃
"""

import asyncio
import datetime
import json as _json
import time
from collections import deque

import config
import event_bus


# ── P>0 管道 ─────────────────────────────────────────────────────────────────
_priority_pending: deque = deque()   # P>0 事件（followup 模式：busy 时暂存）
_output: asyncio.Queue = asyncio.Queue(maxsize=64)  # main agent 消费端
_steering_queue: asyncio.Queue = asyncio.Queue(maxsize=32)  # steer 模式：busy 时推入

# ── P=0 管道 ─────────────────────────────────────────────────────────────────
_bg_buffer: deque = deque()          # P=0 事件（按节奏送 bg subagent）
_bg_last_accepted: dict[str, float] = {}  # per-source throttle for bg
_BG_THROTTLE_INTERVAL = 1.0

# ── 共享状态 ──────────────────────────────────────────────────────────────────
_busy: bool = False
_cancel_event: asyncio.Event | None = None
_current_turn_priority: int = 0
_source_ring: dict[str, deque] = {}  # per-source ring buffer（所有事件）

# 优先级判定规则
_PRIORITY_SOURCES = {'asr', 'message', 'channel', 'subagent', 'acp', 'scheduler'}

# 打断模式：steer(默认) | interrupt | followup
_interrupt_mode: str = "steer"
# barge-in 阈值（ms），ASR 事件 duration_ms 低于此值时视为 backchannel 丢弃
_barge_in_threshold_ms: int = 500


def _extract_priority(ev: dict) -> int:
    """从事件中解析 priority。JSON text 中的 priority 字段优先，否则按 source 匹配。"""
    text = ev.get('text', '')
    if text and text.startswith('{'):
        try:
            data = _json.loads(text)
            p = data.get('priority')
            if p is not None:
                return int(p)
            # ACP: action_complete 事件自动为 P>0（需要 steering 注入 LLM turn）
            if data.get('type') == 'action_complete':
                return 1
        except (ValueError, TypeError):
            pass
    # ACP: payload 中的 action_complete 也处理
    payload = ev.get('payload', {})
    if isinstance(payload, dict):
        if payload.get('type') == 'action_complete':
            return 1
        # Subagent 完成事件：继承 subagent 的 priority（反转映射回 event priority）
        sub_p = payload.get('priority')
        if sub_p is not None:
            return max(1, 3 - int(sub_p))  # sub P=0(紧急) → event P=3, sub P=2 → event P=1
    source = ev.get('source', '').lower()
    for key in _PRIORITY_SOURCES:
        if key in source:
            return 1
    return 0


def _extract_perf_timestamps(ev: dict):
    """从 ASR 事件 JSON 中提取性能 span 数据。"""
    text = ev.get('text', '')
    if not text or not text.startswith('{'):
        return
    try:
        data = _json.loads(text)
    except (ValueError, TypeError):
        return
    if 'spans' in data:
        ev['_perf_spans'] = data['spans']
        return
    spans = []
    audio_start = data.get('audio_start_ts')
    audio_end = data.get('audio_end_ts')
    asr_complete = data.get('asr_complete_ts')
    if audio_start and audio_start > 1e9 and audio_end and audio_end > 1e9:
        spans.append({'span': 'vad_collect', 'start_ts': audio_start, 'end_ts': audio_end,
                      'meta': {'audio_ms': data.get('audio_duration_ms')}})
    if audio_end and audio_end > 1e9 and asr_complete and asr_complete > 1e9:
        spans.append({'span': 'asr_inference', 'start_ts': audio_end, 'end_ts': asr_complete,
                      'meta': {'text_length': data.get('text_length')}})
    if spans:
        ev['_perf_spans'] = spans


def _extract_asr_text_field(ev: dict) -> str:
    """从 ASR 事件中提取纯文本（用于去重比较）。"""
    text = ev.get('text', '')
    if text.startswith('{'):
        try:
            return _json.loads(text).get('text', '')
        except (ValueError, TypeError):
            pass
    return text


def _pending_has_same_asr_text(asr_text: str) -> bool:
    """检查 steering_queue 或 priority_pending 中是否已有相同 ASR 文本。"""
    for item in list(_steering_queue._queue):
        if _extract_asr_text_field(item) == asr_text:
            return True
    for item in _priority_pending:
        if _extract_asr_text_field(item) == asr_text:
            return True
    return False


# ── 公开接口 ──────────────────────────────────────────────────────────────────

def set_busy(busy: bool):
    """由 agent loop 调用：标记当前是否正在执行 turn。"""
    global _busy
    _busy = busy
    if not busy:
        # turn 结束时立即排空所有 pending 队列，避免消息跨 turn 滞留
        _flush_all_pending()


def set_cancel_event(ev: asyncio.Event | None):
    """由 agent loop 调用：注册/清除当前 turn 的取消信号。"""
    global _cancel_event
    _cancel_event = ev


def set_turn_priority(priority: int):
    """由 agent loop 调用：设置当前 turn 的 priority。"""
    global _current_turn_priority
    _current_turn_priority = priority


def set_interrupt_mode(mode: str):
    """设置打断模式：steer | interrupt | followup。"""
    global _interrupt_mode
    if mode in ('steer', 'interrupt', 'followup'):
        _interrupt_mode = mode


def set_barge_in_threshold(ms: int):
    """设置 barge-in 阈值（毫秒）。"""
    global _barge_in_threshold_ms
    _barge_in_threshold_ms = max(0, ms)


def get_interrupt_mode() -> str:
    """返回当前打断模式。"""
    return _interrupt_mode


async def drain_steering() -> list[dict]:
    """非阻塞地 drain steering_queue 中的所有待处理消息。由 agent loop 在 tool batch 间调用。"""
    items = []
    while not _steering_queue.empty():
        try:
            items.append(_steering_queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return items


async def next_trigger() -> dict:
    """阻塞等待下一批 P>0 事件（main agent 消费端）。"""
    # Fallback: 每次等待前再检查一次 pending 队列，防止遗漏
    _flush_all_pending()
    return await _output.get()


def get_source_detail(source: str, limit: int = 20) -> list[dict]:
    """获取指定 source 的原始事件详情（从 ring buffer）。"""
    ring = _source_ring.get(source)
    if not ring:
        return []
    return list(ring)[-limit:]


def get_available_sources() -> list[str]:
    """返回当前有数据的所有 source 名称列表。"""
    return list(_source_ring.keys())


# ── 内部：P>0 管道 ────────────────────────────────────────────────────────────

def _flush_all_pending():
    """同步排空 _steering_queue 和 _priority_pending 到 _output。
    在 turn 结束时和 next_trigger 前调用，确保消息不跨 turn 滞留。"""
    # 先把 steering_queue 里的消息移到 _priority_pending
    while not _steering_queue.empty():
        try:
            _priority_pending.append(_steering_queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    # 再把 _priority_pending 全部 emit 到 _output
    if _priority_pending:
        batch = list(_priority_pending)
        _priority_pending.clear()
        # 同步放入 _output（非 async，避免 fire-and-forget 丢失）
        formatted = _format_priority_batch(batch)
        trigger = {
            'source': 'collector',
            'text': formatted,
            'payload': {'event_count': len(batch), 'sources': [e['source'] for e in batch]},
            'ts': batch[-1]['ts'],
            '_perf_trigger_emit_ts': time.time(),
            '_urgent': True,
        }
        for ev in reversed(batch):
            if '_perf_spans' in ev:
                trigger['_perf_spans'] = ev['_perf_spans']
                break
        try:
            _output.put_nowait(trigger)
        except asyncio.QueueFull:
            print('[collector] WARNING: _output queue full, pending messages dropped')


async def _emit_priority():
    """busy 结束后，立即 emit 暂存的 P>0 事件。"""
    if not _priority_pending:
        return
    batch = list(_priority_pending)
    _priority_pending.clear()
    await _emit_batch(batch, urgent=True)


async def _emit_batch(batch: list[dict], urgent: bool = False):
    """将 P>0 事件格式化并放入 output。"""
    formatted = _format_priority_batch(batch)
    trigger = {
        'source': 'collector',
        'text': formatted,
        'payload': {'event_count': len(batch), 'sources': [e['source'] for e in batch]},
        'ts': batch[-1]['ts'],
        '_perf_trigger_emit_ts': time.time(),
        '_urgent': urgent,
    }
    for ev in reversed(batch):
        if '_perf_spans' in ev:
            trigger['_perf_spans'] = ev['_perf_spans']
            break
    await _output.put(trigger)


# 对 LLM 决策无意义的 perf/trace 字段（已独立存入 perf_spans 表）
_LLM_IRRELEVANT_KEYS = frozenset({
    'audio_start_ts', 'audio_end_ts', 'asr_complete_ts',
    'spans', 'priority', 'text_length',
})


def _slim_event_text(text: str) -> str:
    """剥离对 LLM 决策无意义的 perf/trace 字段，只保留语义信息。"""
    if not text or not text.startswith('{'):
        return text
    try:
        data = _json.loads(text)
    except (ValueError, TypeError):
        return text
    changed = False
    # 移除 perf trace 字段
    for key in _LLM_IRRELEVANT_KEYS:
        if key in data:
            del data[key]
            changed = True
    # ACP 完成事件：去掉 result 中的冗余内容（LLM 已知自己发出了什么）
    if data.get('type') == 'action_complete' and 'result' in data:
        del data['result']
        changed = True
    if not changed:
        return text
    return _json.dumps(data, ensure_ascii=False)


def _format_priority_batch(events: list[dict]) -> str:
    """格式化 P>0 事件为 XML（精简 perf 字段后的文本）。"""
    parts = []
    for ev in events:
        ts = datetime.datetime.fromtimestamp(ev['ts']).strftime('%Y-%m-%dT%H:%M:%S')
        channel = _infer_channel(ev)
        source = ev.get('source', '')
        text = _slim_event_text(ev.get('text', ''))
        parts.append(f'<event source="{source}" channel="{channel}" ts="{ts}">\n{text}\n</event>')
    return '\n'.join(parts)


# ── 内部：P=0 管道 ────────────────────────────────────────────────────────────

def _bg_buffer_add(ev: dict):
    """将 P=0 事件加入 bg buffer（per-source throttle）。"""
    source = ev.get('source', 'unknown')
    now = ev.get('ts', time.time())
    last_ts = _bg_last_accepted.get(source, 0)

    if now - last_ts < _BG_THROTTLE_INTERVAL:
        # 替换同 source 最后一条
        for i in range(len(_bg_buffer) - 1, -1, -1):
            if _bg_buffer[i].get('source') == source:
                _bg_buffer[i] = ev
                return
    _bg_last_accepted[source] = now
    _bg_buffer.append(ev)

    # FIFO 限制
    max_window = config.main.get('event', {}).get('llm', {}).get('collector_max_window', 20)
    while len(_bg_buffer) > max_window:
        _bg_buffer.popleft()


def _format_bg_batch(events: list[dict]) -> str:
    """格式化 P=0 事件为摘要（按 source 分组）。"""
    groups: dict[str, list[dict]] = {}
    for ev in events:
        source = ev.get('source', 'unknown')
        groups.setdefault(source, []).append(ev)

    parts = []
    for source, evs in groups.items():
        ts = datetime.datetime.fromtimestamp(evs[-1]['ts']).strftime('%Y-%m-%dT%H:%M:%S')
        last_text = evs[-1].get('text', '')
        if len(evs) == 1:
            parts.append(f'<source name="{source}" ts="{ts}">\n{last_text}\n</source>')
        else:
            parts.append(f'<source name="{source}" count="{len(evs)}" ts="{ts}">\n{last_text}\n(共 {len(evs)} 条，显示最新)\n</source>')
    return '\n'.join(parts)


async def _route_to_bg_subagent(batch: list[dict]) -> bool:
    """将 P=0 事件批次路由到 bg subagent。"""
    try:
        from subagent import _manager_instance
    except ImportError:
        return False

    if not _manager_instance:
        return False

    bg_config = config.main.get('subagent', {})
    if not bg_config.get('bg_route_enabled', True):
        return False

    summary = _format_bg_batch(batch)

    # 同步 main agent 最近对话上下文（精简，subagent 可自行 memory_recall）
    try:
        from event.llm import get_recent_context
        recent_context = get_recent_context(max_turns=2)
    except (ImportError, AttributeError):
        recent_context = ''

    if recent_context:
        message = f'[主代理最近决策]\n{recent_context}\n\n[新数据]\n{summary}'
    else:
        message = summary

    # 检查是否有活跃的 bg subagent
    active = _manager_instance.list_active()
    bg_agents = [a for a in active if a.goal.startswith('[bg]')]

    if bg_agents:
        _manager_instance.send_message(bg_agents[0].id, message)
    else:
        from subagent.protocol import SubagentSpec, P_LOW
        spec = SubagentSpec(
            goal=(
                '[bg] 后台监控：分析传入的信息。\n'
                '- 需要历史对比时，用 memory_recall 检索之前的结论\n'
                '- 无变化 → subagent_finish\n'
                '- 有变化但非紧急 → subagent_report(progress=结论)\n'
                '- 安全/硬件告警（SOC<10%、温度>50°C、碰撞） → subagent_report(progress=..., urgent=true)\n'
                '不要主动调用 Bash/Read 等工具，只分析传入内容或通过 memory_recall 检索历史。'
            ),
            priority=P_LOW,
            model=bg_config.get('bg_model'),
            tool_deny=['mcp__*', 'Bash', 'Read', 'Write', 'Edit', 'Glob', 'Grep', 'WebFetch', 'WebSearch'],
            max_rounds=50,
            timeout_s=3600,
            context_seed=message,
        )
        await _manager_instance.spawn(spec)

    return True


# ── Channel 推断 ──────────────────────────────────────────────────────────────

def _infer_channel(ev: dict) -> str:
    """从事件 source 推断渠道标签。"""
    source = ev.get('source', '')
    if '/channel/' in source or source.startswith('channel:'):
        text = ev.get('text', '')
        if text and text.startswith('{'):
            try:
                data = _json.loads(text)
                platform = data.get('platform', '')
                if platform:
                    return f'channel:{platform}'
            except (ValueError, TypeError):
                pass
        return 'channel'
    if '/remote_control/message' in source:
        return 'remote_web'
    if 'asr' in source.lower() or '/mic' in source:
        if 'remote' in source:
            return 'remote_mic'
        return 'local_mic'
    return 'sensor'


# ── 主循环 ────────────────────────────────────────────────────────────────────

async def _drain_loop():
    """持续从 event_bus 消费事件，按 priority 分流到两个管道。"""
    ring_size = config.main.get('event', {}).get('llm', {}).get('source_ring_size', 50)
    while True:
        ev = await event_bus.dequeue()
        source = ev.get('source', 'unknown')

        _extract_perf_timestamps(ev)
        priority = _extract_priority(ev)

        # KWS wake-word hook (fires regardless of busy state)
        if 'asr' in source.lower():
            payload = ev.get('payload', {})
            if isinstance(payload, str):
                try:
                    import json as _json
                    payload = _json.loads(payload)
                except Exception:
                    payload = {}
            if payload.get('kws_triggered'):
                import hooks
                asyncio.create_task(hooks.fire('on_kws_wakeup'))

        # Ring buffer 始终存储（所有事件，供 raw_input_info 查询）
        if source not in _source_ring:
            _source_ring[source] = deque(maxlen=ring_size)
        _source_ring[source].append(ev)

        if priority > 0:
            # ── P>0: 送 main agent ──
            if not _busy:
                await _emit_batch([ev], urgent=True)
            else:
                # Barge-in 检测：ASR 事件 duration 不足时视为 backchannel，丢弃
                if 'asr' in source.lower() and _barge_in_threshold_ms > 0:
                    duration_ms = ev.get('payload', {}).get('duration_ms', 0)
                    if 0 < duration_ms < _barge_in_threshold_ms:
                        continue  # backchannel，不打断

                # ASR 去重：busy 时如果相同 text 已在排队中，丢弃重复
                if 'asr' in source.lower():
                    _asr_text = _extract_asr_text_field(ev)
                    if _asr_text and _pending_has_same_asr_text(_asr_text):
                        print(f'[collector] dedup: ASR text "{_asr_text[:30]}" already pending, skip')
                        continue

                # 按模式处理
                if _interrupt_mode == 'steer':
                    # Scheduler 去重：如果 steering_queue 中已有相同 source 的 scheduler 事件，跳过
                    if 'scheduler:' in source.lower():
                        _dedup = False
                        for item in list(_steering_queue._queue):
                            if item.get('source', '') == ev.get('source', ''):
                                _dedup = True
                                break
                        if _dedup:
                            continue  # 已有相同 task 的 check 事件，跳过
                    # Steer: 推入 steering_queue，agent loop 在 tool batch 间消费
                    try:
                        _steering_queue.put_nowait(ev)
                    except asyncio.QueueFull:
                        # queue 满时退化为 followup
                        _priority_pending.append(ev)
                elif _interrupt_mode == 'interrupt':
                    # Interrupt: 缓存事件并触发 cancel
                    _priority_pending.append(ev)
                    if _cancel_event:
                        _cancel_event.set()
                else:
                    # Followup: 暂存，等 turn 结束后 drain（原有行为）
                    _priority_pending.append(ev)
                    if priority > _current_turn_priority and _cancel_event:
                        _cancel_event.set()
        else:
            # ── P=0: 送 bg buffer ──
            _bg_buffer_add(ev)


def _bg_buffer_has_substance(batch: list[dict]) -> bool:
    """检查事件批次是否包含有意义的传感器数据。空文本或无数值的事件不值得 spawn bg_monitor。"""
    for ev in batch:
        text = ev.get('text', '').strip()
        if not text:
            continue
        if text.startswith('{'):
            try:
                data = _json.loads(text)
                # 含数值字段 = 有传感器数据（SOC/温度/电压/IMU 等）
                if any(isinstance(v, (int, float)) for v in data.values()):
                    return True
            except (ValueError, TypeError):
                pass
        elif len(text) > 5:
            # 非 JSON 但有文本内容
            return True
    return False


async def _bg_trigger_loop():
    """独立节奏：每 interval 把 bg_buffer 送给 bg subagent。"""
    while True:
        interval = config.main.get('event', {}).get('llm', {}).get('trigger_interval_ms', 1000) / 1000.0
        await asyncio.sleep(interval)
        if not _bg_buffer:
            continue
        batch = list(_bg_buffer)
        _bg_buffer.clear()
        # 只在 buffer 含有实质性传感器数据时才路由到 bg subagent
        if not _bg_buffer_has_substance(batch):
            continue
        await _route_to_bg_subagent(batch)


def start():
    """启动 collector 后台任务。"""
    # 从配置加载打断模式和 barge-in 阈值
    event_cfg = config.main.get('event', {}).get('llm', {})
    mode = event_cfg.get('interrupt_mode', 'steer')
    set_interrupt_mode(mode)
    threshold = event_cfg.get('barge_in_threshold_ms', 500)
    set_barge_in_threshold(threshold)

    asyncio.ensure_future(_drain_loop())
    asyncio.ensure_future(_bg_trigger_loop())
    interval = event_cfg.get('trigger_interval_ms', 1000)
    print(f'[collector] started: dual-queue mode, bg_interval={interval}ms, interrupt_mode={_interrupt_mode}, barge_in={_barge_in_threshold_ms}ms')
