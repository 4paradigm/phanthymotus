"""
manager.py — SubagentManager: lifecycle management, scheduling, and concurrency control.

Responsibilities:
- Spawn/cancel/pause/resume subagents
- Priority queue scheduling with preemption
- Concurrent execution within max_concurrent limit
- State persistence (checkpoint on pause/shutdown, restore on startup)
- Result delivery via event_bus
"""

from __future__ import annotations
import asyncio
import time

import config
import event_bus
from api.motus_stream import push_event
from .protocol import (
    SubagentSpec, SubagentResult, SubagentStatus,
    STATUS_PENDING, STATUS_RUNNING, STATUS_PAUSED, STATUS_SUSPENDED,
    STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED,
    TERMINAL_STATUSES, ACTIVE_STATUSES,
)
from .agent import Subagent
from .queue import SubagentPriorityQueue
from .store import SubagentStore


def _get_config() -> dict:
    """Get subagent configuration with defaults."""
    defaults = {
        'max_concurrent': 5,
        'max_concurrent_bg': 1,
        'max_total': 10,
        'default_max_rounds': 10,
        'default_timeout_s': 300,
        'preemption_enabled': True,
        'checkpoint_interval': 5,
        'compress_threshold_chars': 20000,
        'cleanup_age_hours': 24,
    }
    cfg = config.main.get('subagent', {})
    return {**defaults, **cfg}


class SubagentManager:
    """Manages subagent lifecycle, scheduling, and communication."""

    def __init__(self, llm_client):
        self._llm_client = llm_client
        self._agents: dict[str, Subagent] = {}          # all non-terminal agents
        self._completed: dict[str, SubagentResult] = {} # recent results (ring buffer)
        self._queue = SubagentPriorityQueue()
        self._running: dict[str, asyncio.Task] = {}     # agent_id -> asyncio.Task
        self._store = SubagentStore()
        self._scheduler_task: asyncio.Task | None = None
        self._timeout_tasks: dict[str, asyncio.Task] = {}
        self._cfg = _get_config()

    async def start(self) -> None:
        """Start the scheduler loop and restore persisted state."""
        await self._restore()
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        # Cleanup old terminal records
        cleaned = self._store.cleanup_old(self._cfg['cleanup_age_hours'])
        if cleaned:
            print(f'[subagent] cleaned up {cleaned} old records')

    async def shutdown(self) -> None:
        """Graceful shutdown: checkpoint all running agents."""
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        # Cancel timeout tasks
        for t in self._timeout_tasks.values():
            t.cancel()

        # Checkpoint running agents
        for agent_id, task in list(self._running.items()):
            agent = self._agents.get(agent_id)
            if agent:
                agent.cancel('agent-core shutting down')
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                # Save checkpoint as suspended (interrupted by shutdown)
                if agent.status == STATUS_RUNNING:
                    agent.status = STATUS_SUSPENDED
                self._checkpoint(agent)

        print(f'[subagent] shutdown: checkpointed {len(self._running)} agents')

    # ── Public API ────────────────────────────────────────────────────────────

    async def spawn(self, spec: SubagentSpec) -> str:
        """Create and queue a subagent. Returns agent_id."""
        if len(self._agents) >= self._cfg['max_total']:
            raise RuntimeError(f'Maximum subagent count ({self._cfg["max_total"]}) reached')

        agent = Subagent(
            spec=spec,
            compress_threshold=self._cfg['compress_threshold_chars'],
        )
        agent.status = STATUS_PENDING
        self._agents[agent.id] = agent
        self._queue.push(agent.id, spec.priority)

        print(f'[subagent] spawned {agent.id} P{spec.priority}: {spec.goal[:60]}')
        await push_event({
            'type': 'subagent_spawn',
            'payload': {
                'id': agent.id,
                'goal': spec.goal,
                'priority': spec.priority,
                'model': spec.model,
            },
        })

        # Kick scheduler
        self._queue._notify.set()
        return agent.id

    async def spawn_and_wait(self, spec: SubagentSpec, timeout: float = 120) -> SubagentResult:
        """Spawn a subagent and wait for its result (synchronous from caller's perspective)."""
        agent_id = await self.spawn(spec)
        return await self._wait_for_result(agent_id, timeout)

    async def cancel(self, agent_id: str, reason: str = '') -> bool:
        """Cancel a subagent."""
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        if agent.status == STATUS_PENDING:
            self._queue.remove(agent_id)
            agent.status = STATUS_CANCELLED
            agent.result = SubagentResult(
                agent_id=agent_id, status=STATUS_CANCELLED,
                output='', error=reason or 'Cancelled while pending',
            )
            self._finalize(agent)
            return True

        if agent.status == STATUS_RUNNING:
            agent.cancel(reason or 'cancelled by request')
            # The running task will handle the rest
            return True

        return False

    async def pause(self, agent_id: str) -> bool:
        """Pause a running subagent (checkpoint + suspend)."""
        agent = self._agents.get(agent_id)
        if not agent or agent.status != STATUS_RUNNING:
            return False
        agent.pause()
        return True

    async def resume(self, agent_id: str) -> bool:
        """Resume a paused/suspended subagent."""
        agent = self._agents.get(agent_id)
        if not agent or agent.status not in (STATUS_PAUSED, STATUS_SUSPENDED):
            return False
        agent.status = STATUS_PENDING
        agent._pause_event.clear()
        agent._cancel_event.clear()
        self._queue.push(agent_id, agent.spec.priority)
        self._queue._notify.set()
        return True

    def send_message(self, agent_id: str, text: str) -> bool:
        """Send a message to a running subagent."""
        agent = self._agents.get(agent_id)
        if not agent or agent.status != STATUS_RUNNING:
            return False
        agent.send_message(text)
        return True

    def get_status(self, agent_id: str) -> SubagentStatus | None:
        """Get status of a specific subagent."""
        agent = self._agents.get(agent_id)
        if agent:
            return agent.get_status()
        # Check completed
        result = self._completed.get(agent_id)
        if result:
            return SubagentStatus(
                id=agent_id, goal='(completed)', status=result.status,
                priority=0, model=None, rounds_completed=result.rounds_used,
                created_at=0, updated_at=0,
            )
        return None

    def get_result(self, agent_id: str) -> SubagentResult | None:
        """Get result of a completed subagent."""
        return self._completed.get(agent_id)

    def list_active(self) -> list[SubagentStatus]:
        """List all non-terminal subagents."""
        return [a.get_status() for a in self._agents.values()]

    # ── Scheduler ─────────────────────────────────────────────────────────────

    async def _scheduler_loop(self):
        """Background loop: schedule pending agents when slots available."""
        while True:
            try:
                await self._schedule_next()
                # Small sleep to avoid tight loop
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                return
            except Exception as e:
                print(f'[subagent] scheduler error: {e}')
                await asyncio.sleep(1)

    async def _schedule_next(self):
        """Try to schedule the next pending agent."""
        max_concurrent = self._cfg['max_concurrent']

        # Check if we have capacity
        running_count = len(self._running)
        if running_count >= max_concurrent:
            # Check preemption
            if self._cfg['preemption_enabled']:
                running_priorities = {
                    aid: self._agents[aid].spec.priority
                    for aid in self._running
                    if aid in self._agents
                }
                preempt_id = self._queue.should_preempt(running_priorities)
                if preempt_id:
                    print(f'[subagent] preempting {preempt_id}')
                    await self._preempt(preempt_id)
                else:
                    # Wait for a slot
                    await self._queue.wait_for_item()
                    return
            else:
                await self._queue.wait_for_item()
                return

        # Pop next from queue
        entry = self._queue.pop()
        if not entry:
            await self._queue.wait_for_item()
            return

        agent = self._agents.get(entry.agent_id)
        if not agent or agent.status in TERMINAL_STATUSES:
            return  # Skip stale entries

        # BG concurrency limit: bg agents (goal starts with [bg]) max 1 concurrent
        if agent.spec.goal.startswith('[bg]'):
            max_bg = self._cfg.get('max_concurrent_bg', 1)
            bg_running = sum(
                1 for aid in self._running
                if aid in self._agents and self._agents[aid].spec.goal.startswith('[bg]')
            )
            if bg_running >= max_bg:
                # Re-push to queue and wait
                self._queue.push(agent.id, agent.spec.priority)
                await self._queue.wait_for_item()
                return

        # Launch agent
        task = asyncio.create_task(self._run_agent(agent))
        self._running[agent.id] = task

        # Setup timeout
        if agent.spec.timeout_s > 0:
            self._timeout_tasks[agent.id] = asyncio.create_task(
                self._timeout_watchdog(agent.id, agent.spec.timeout_s)
            )

    async def _run_agent(self, agent: Subagent):
        """Execute one subagent's LLM loop."""
        agent_id = agent.id
        try:
            result = await agent.run(self._llm_client)

            if result is None:
                # Paused/suspended — checkpoint but don't finalize
                self._checkpoint(agent)
                print(f'[subagent:{agent_id}] paused at round {agent.rounds_completed}')
                return

            # Terminal result
            self._finalize(agent)

        except asyncio.CancelledError:
            # External cancellation (shutdown or preemption)
            if agent.status == STATUS_RUNNING:
                agent.status = STATUS_SUSPENDED
            self._checkpoint(agent)
        except Exception as e:
            print(f'[subagent:{agent_id}] unexpected error: {e}')
            agent.result = SubagentResult(
                agent_id=agent_id, status=STATUS_FAILED,
                output='', error=f'{type(e).__name__}: {e}',
                rounds_used=agent.rounds_completed,
                duration_s=time.time() - agent.created_at,
            )
            agent.status = STATUS_FAILED
            self._finalize(agent)
        finally:
            self._running.pop(agent_id, None)
            self._timeout_tasks.pop(agent_id, None)

    async def _preempt(self, agent_id: str):
        """Preempt a running agent: pause and checkpoint."""
        agent = self._agents.get(agent_id)
        if not agent:
            return
        agent.pause()
        task = self._running.get(agent_id)
        if task:
            try:
                await asyncio.wait_for(task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()
        agent.status = STATUS_SUSPENDED
        self._checkpoint(agent)
        print(f'[subagent:{agent_id}] preempted and suspended')

    async def _timeout_watchdog(self, agent_id: str, timeout_s: float):
        """Cancel agent if idle (no progress) for timeout_s seconds."""
        idle_timeout = timeout_s  # timeout_s 现在是"无进展超时"而非绝对超时
        while True:
            await asyncio.sleep(30)  # 每 30 秒检查一次
            agent = self._agents.get(agent_id)
            if not agent or agent.status != STATUS_RUNNING:
                return
            idle_time = time.time() - agent.updated_at
            if idle_time >= idle_timeout:
                print(f'[subagent:{agent_id}] idle timeout: no progress for {idle_time:.0f}s')
                agent.cancel(f'idle timeout ({idle_time:.0f}s without progress)')
                return

    # ── Lifecycle Helpers ─────────────────────────────────────────────────────

    def _finalize(self, agent: Subagent):
        """Handle terminal state: store result, notify, cleanup, save history."""
        result = agent.result
        if result:
            # Store in completed ring (keep last 50)
            self._completed[agent.id] = result
            if len(self._completed) > 50:
                oldest = next(iter(self._completed))
                del self._completed[oldest]

            # Persist terminal state
            self._store.save_result(agent.id, result)

            # Save subagent conversation to chat history
            self._save_subagent_history(agent, result)

            # Notify via event_bus
            asyncio.create_task(self._notify_completion(agent, result))

        # Remove from active
        self._agents.pop(agent.id, None)

    def _save_subagent_history(self, agent: Subagent, result: SubagentResult):
        """Save subagent turns to chat_history for visibility in history modal."""
        try:
            import chat_history
            session_id = chat_history.create_session()
            chat_history.update_summary(session_id, f'[subagent:{agent.id}] {agent.spec.goal[:80]}')
            for i, turn in enumerate(agent.context.turns):
                chat_history.save_turn(session_id, i, turn)
        except Exception as e:
            print(f'[subagent:{agent.id}] save history failed: {e}')

    async def _notify_completion(self, agent: Subagent, result: SubagentResult):
        """Push completion event to event_bus and motus stream.

        策略：
        - BG subagent 正常完成 → 结论存 DB，不触发 main agent
        - BG subagent fail/timeout → 仍触发 main agent
        - 用户任务 subagent → 结论存 DB + 精简通知触发 main agent
        """
        is_bg = agent.spec.goal.startswith('[bg]')

        # 所有 subagent 完成时结论存 DB
        if result.status == 'completed' and result.output.strip():
            try:
                import time as _time
                from config import _get_conn
                source_type = 'bg_monitor' if is_bg else 'user_task'
                with _get_conn() as conn:
                    conn.execute(
                        'INSERT INTO subagent_conclusions (agent_id, goal, conclusion, source_type, created_at) '
                        'VALUES (?, ?, ?, ?, ?)',
                        (agent.id, agent.spec.goal[:200], result.output, source_type, _time.time())
                    )
                    conn.commit()
            except Exception as e:
                print(f'[subagent:{agent.id}] save conclusion to DB failed: {e}')

        # BG subagent 正常完成 → 不触发 main agent
        if is_bg and result.status == 'completed':
            await push_event({
                'type': 'subagent_complete',
                'payload': {'id': agent.id, 'status': 'completed', 'output': result.output[:100], 'rounds': result.rounds_used},
            })
            return

        # 非 bg 或 fail/timeout → 触发 main agent（精简通知）
        status_emoji = {'completed': '✓', 'failed': '✗', 'timeout': '⏱', 'cancelled': '⊘'}
        emoji = status_emoji.get(result.status, '?')

        # 精简通知：goal 摘要 + output 前 100 字符
        notify_text = f'子代理 [{agent.id}] {emoji} {result.status}: {agent.spec.goal[:40]}'
        if result.output:
            notify_text += f'\n摘要: {result.output[:100]}'
        elif result.error:
            notify_text += f'\n错误: {result.error[:100]}'

        await event_bus.enqueue(
            source=f'subagent:{agent.id}',
            text=notify_text,
            payload={
                'agent_id': agent.id,
                'status': result.status,
                'output': result.output[:200] if result.output else '',
                'error': result.error,
                'rounds_used': result.rounds_used,
                'priority': agent.spec.priority,
            },
        )

        await push_event({
            'type': 'subagent_complete',
            'payload': {
                'id': agent.id,
                'status': result.status,
                'output': result.output[:200],
                'error': result.error,
                'rounds': result.rounds_used,
                'duration_s': result.duration_s,
            },
        })

    def _checkpoint(self, agent: Subagent):
        """Persist agent state to SQLite."""
        self._store.save_checkpoint(
            agent_id=agent.id,
            spec=agent.spec,
            status=agent.status,
            turns=agent.context.turns,
            summary=agent.context.summary or '',
            rounds_completed=agent.rounds_completed,
            created_at=agent.created_at,
            updated_at=agent.updated_at,
        )

    # ── Restore ───────────────────────────────────────────────────────────────

    async def _restore(self):
        """Restore persisted subagents on startup."""
        records = self._store.load_active()
        if not records:
            return

        for rec in records:
            agent = Subagent(
                spec=rec['spec'],
                agent_id=rec['id'],
                compress_threshold=self._cfg['compress_threshold_chars'],
            )
            agent.status = rec['status']
            agent.rounds_completed = rec['rounds_completed']
            agent.created_at = rec['created_at']
            agent.updated_at = rec['updated_at']
            agent.context.restore_from_checkpoint(
                turns=rec['turns'],
                summary=rec['summary'],
            )

            self._agents[agent.id] = agent

            # Re-queue pending/suspended agents
            if agent.status in (STATUS_PENDING, STATUS_SUSPENDED):
                agent.status = STATUS_PENDING
                self._queue.push(agent.id, agent.spec.priority)

        print(f'[subagent] restored {len(records)} agents from checkpoint')

    # ── Wait helper (for spawn_and_wait) ──────────────────────────────────────

    async def _wait_for_result(self, agent_id: str, timeout: float) -> SubagentResult:
        """Poll for result with timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            # Check if already completed
            result = self._completed.get(agent_id)
            if result:
                return result
            # Check if agent finished
            agent = self._agents.get(agent_id)
            if agent and agent.result:
                return agent.result
            await asyncio.sleep(0.5)

        # Timeout — cancel the agent
        await self.cancel(agent_id, 'spawn_sync timeout')
        return SubagentResult(
            agent_id=agent_id,
            status='timeout',
            output='',
            error=f'spawn_and_wait timed out after {timeout}s',
        )
