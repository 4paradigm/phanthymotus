"""Data structures for the subagent system."""

from __future__ import annotations
from dataclasses import dataclass, field
import time


# Priority constants
P_CRITICAL = 0
P_HIGH = 1
P_NORMAL = 2
P_LOW = 3

# Status constants
STATUS_PENDING = 'pending'
STATUS_RUNNING = 'running'
STATUS_PAUSED = 'paused'
STATUS_SUSPENDED = 'suspended'
STATUS_COMPLETED = 'completed'
STATUS_FAILED = 'failed'
STATUS_TIMEOUT = 'timeout'
STATUS_CANCELLED = 'cancelled'

TERMINAL_STATUSES = {STATUS_COMPLETED, STATUS_FAILED, STATUS_TIMEOUT, STATUS_CANCELLED}
ACTIVE_STATUSES = {STATUS_PENDING, STATUS_RUNNING, STATUS_PAUSED, STATUS_SUSPENDED}

# Tools that only talk *about* the task rather than carry it out. A subagent whose
# entire run consists of these did nothing observable, yet still reports
# STATUS_COMPLETED with whatever prose it chose to write.
#
# This is not hypothetical. Measured on Orin5+Orin6: of six inbound peer
# delegations each asking the receiver to speak one line, four ran a single round,
# called `subagent_finish` without ever touching tts, and returned "completed" —
# and `peer_delegate` handed that self-report back to the delegator as success.
# Both robots then announced a sixteen-line performance that never happened.
# `substantive_tool_calls` is how a caller tells the two apart.
BOOKKEEPING_TOOLS = frozenset({'subagent_finish', 'subagent_fail', 'subagent_report'})


@dataclass
class SubagentSpec:
    """Task specification for spawning a subagent."""
    goal: str
    priority: int = P_NORMAL
    model: str | None = None
    tool_filter: list[str] | None = None
    tool_deny: list[str] | None = None
    max_rounds: int = 10
    timeout_s: float = 300.0
    hop_count: int = 0  # Incremented on each delegation, prevents infinite chains
    system_prompt_extra: str = ''
    context_seed: str = ''
    checkpoint_interval: int = 5
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'goal': self.goal,
            'priority': self.priority,
            'model': self.model,
            'tool_filter': self.tool_filter,
            'tool_deny': self.tool_deny,
            'max_rounds': self.max_rounds,
            'timeout_s': self.timeout_s,
            'system_prompt_extra': self.system_prompt_extra,
            'context_seed': self.context_seed,
            'checkpoint_interval': self.checkpoint_interval,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SubagentSpec:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SubagentResult:
    """Result returned when a subagent reaches terminal state."""
    agent_id: str
    status: str
    output: str
    tool_calls_made: list[dict] = field(default_factory=list)
    rounds_used: int = 0
    duration_s: float = 0.0
    error: str | None = None
    # ACP actions this run started and how each ended. Starting an async action is
    # not evidence it happened: `speak` returns when the audio is *queued*, so a
    # run that only ever queued and a run whose audio actually played produce
    # identical prose. Entries are {action_id, status, tool}; status 'pending' means
    # it never resolved before the run ended.
    actions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'agent_id': self.agent_id,
            'status': self.status,
            'output': self.output,
            'tool_calls_made': self.tool_calls_made,
            'substantive_tool_calls': self.substantive_tool_calls(),
            'actions': self.actions,
            'confirmed_actions': self.confirmed_actions(),
            'rounds_used': self.rounds_used,
            'duration_s': self.duration_s,
            'error': self.error,
        }

    def substantive_tool_calls(self) -> list[str]:
        """Names of tools this run called that actually did something.

        Excludes BOOKKEEPING_TOOLS. An empty list on a `completed` result means the
        subagent reported success without acting — see BOOKKEEPING_TOOLS.
        """
        return [
            name for tc in self.tool_calls_made
            if (name := (tc.get('name') or '')) and name not in BOOKKEEPING_TOOLS
        ]

    def confirmed_actions(self) -> list[dict]:
        """Actions that reached a terminal 'completed' state.

        Anything else — 'timeout', 'cancelled', 'barge_in', 'pending' — did not
        demonstrably happen. A barrier timeout clears the pending and lets the
        caller proceed exactly as success does, so this distinction is the only
        thing standing between "the robot spoke" and "the robot was asked to".
        """
        return [a for a in self.actions if a.get('status') == 'completed']

    def acted(self) -> bool:
        """Whether this run did something observable.

        True when it either completed an async action or called a non-bookkeeping
        tool. A run that only called `subagent_finish` is False no matter how
        confidently its output says otherwise.
        """
        return bool(self.confirmed_actions() or self.substantive_tool_calls())

    @classmethod
    def from_dict(cls, d: dict) -> SubagentResult:
        if not d:
            return None
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SubagentStatus:
    """Lightweight status for listing subagents."""
    id: str
    goal: str
    status: str
    priority: int
    model: str | None
    rounds_completed: int
    created_at: float
    updated_at: float

    def to_display(self) -> str:
        elapsed = time.time() - self.created_at
        if elapsed < 60:
            elapsed_str = f'{elapsed:.0f}s'
        else:
            elapsed_str = f'{elapsed / 60:.1f}min'
        model_str = f' [{self.model}]' if self.model else ''
        return (
            f'[{self.id}] P{self.priority}{model_str} {self.status} '
            f'({self.rounds_completed}轮, {elapsed_str}) — {self.goal[:60]}'
        )
