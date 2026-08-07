"""
queue.py — Priority queue with preemption support for subagent scheduling.
"""

from __future__ import annotations
import asyncio
import heapq
import time
from dataclasses import dataclass, field


@dataclass(order=True)
class QueueEntry:
    """Sortable queue entry: lower priority number = higher priority, FIFO within same level."""
    priority: int
    created_at: float = field(compare=True)
    agent_id: str = field(compare=False)


class SubagentPriorityQueue:
    """Priority queue for pending subagents with preemption detection."""

    def __init__(self):
        self._heap: list[QueueEntry] = []
        self._notify: asyncio.Event = asyncio.Event()
        self._removed: set[str] = set()  # agent_ids that were cancelled while queued

    def push(self, agent_id: str, priority: int) -> None:
        """Add a subagent to the queue."""
        entry = QueueEntry(priority=priority, created_at=time.time(), agent_id=agent_id)
        heapq.heappush(self._heap, entry)
        self._notify.set()

    def pop(self) -> QueueEntry | None:
        """Pop highest priority (lowest number) entry, skipping removed."""
        while self._heap:
            entry = heapq.heappop(self._heap)
            if entry.agent_id not in self._removed:
                return entry
            self._removed.discard(entry.agent_id)
        return None

    def peek(self) -> QueueEntry | None:
        """Look at the head without removing, skipping removed entries."""
        while self._heap and self._heap[0].agent_id in self._removed:
            heapq.heappop(self._heap)
            # don't need to track removed anymore since it's gone
        return self._heap[0] if self._heap else None

    def remove(self, agent_id: str) -> None:
        """Mark an agent as removed (lazy deletion)."""
        self._removed.add(agent_id)

    def should_preempt(self, running_priorities: dict[str, int]) -> str | None:
        """Check if head of queue should preempt a running agent.

        Preemption triggers when queued priority is 2+ levels higher than
        the worst (highest number) running agent.

        Args:
            running_priorities: {agent_id: priority} of currently running agents

        Returns:
            agent_id to preempt, or None
        """
        head = self.peek()
        if not head or not running_priorities:
            return None

        # Find the lowest-priority (highest number) running agent
        worst_id = max(running_priorities, key=running_priorities.get)
        worst_priority = running_priorities[worst_id]

        # Preempt if queued is 2+ levels higher priority
        if head.priority <= worst_priority - 2:
            return worst_id
        return None

    @property
    def size(self) -> int:
        """Approximate queue size (may include removed entries)."""
        return len(self._heap) - len(self._removed)

    @property
    def empty(self) -> bool:
        return self.peek() is None

    async def wait_for_item(self) -> None:
        """Block until queue is non-empty."""
        if not self.empty:
            return
        self._notify.clear()
        await self._notify.wait()

    def clear(self) -> None:
        """Remove all entries."""
        self._heap.clear()
        self._removed.clear()
