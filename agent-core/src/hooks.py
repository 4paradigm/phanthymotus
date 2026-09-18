"""
System Hooks — bypass-LLM actions triggered by system events.

Hooks are registered by drivers via `x-hooks` in MCP tool schemas. When fired,
they execute tool calls directly, without going back to the LLM (see `fire`'s
`barrier_aware` param for whether they also bypass the barrier/ACP — true
interrupts should, narration-style hooks like on_notify should not).
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# ── Types ────────────────────────────────────────────────────────────────────


@dataclass
class HookBinding:
    mcp_id: str
    tool: str
    action: str
    params: dict = field(default_factory=dict)


# ── Registry ─────────────────────────────────────────────────────────────────

_registry: dict[str, list[HookBinding]] = {}
_fire_log: deque = deque(maxlen=30)


def register(mcp_id: str, tool_name: str, x_hooks: dict):
    """Register hook bindings from a tool's x-hooks schema field.

    Args:
        mcp_id: MCP device ID (e.g. "mcp-1785810828")
        tool_name: Tool name (e.g. "smart_motion")
        x_hooks: Dict of hook_id → {"action": str, "params": dict}
    """
    for hook_id, spec in x_hooks.items():
        action = spec.get('action', '')
        params = spec.get('params', {})
        binding = HookBinding(mcp_id=mcp_id, tool=tool_name, action=action, params=params)
        if hook_id not in _registry:
            _registry[hook_id] = []
        # Avoid duplicate bindings (same mcp_id + tool + action)
        existing = [(b.mcp_id, b.tool, b.action) for b in _registry[hook_id]]
        if (mcp_id, tool_name, action) not in existing:
            _registry[hook_id].append(binding)
            print(f'[hooks] registered: {hook_id} → {mcp_id}/{tool_name}.{action}')


def unregister_device(mcp_id: str):
    """Remove all bindings for a device (e.g. when it goes offline)."""
    for hook_id in list(_registry.keys()):
        _registry[hook_id] = [b for b in _registry[hook_id] if b.mcp_id != mcp_id]
        if not _registry[hook_id]:
            del _registry[hook_id]


def list_hooks() -> dict[str, list[dict]]:
    """Return all registered hooks and their bindings."""
    return {
        hook_id: [
            {'mcp_id': b.mcp_id, 'tool': b.tool, 'action': b.action, 'params': b.params}
            for b in bindings
        ]
        for hook_id, bindings in _registry.items()
    }


def is_interrupt_binding(mcp_id: str, tool: str, action: str) -> bool:
    """Check if tool+action is registered under an on_interrupt_* hook.
    Used by barrier logic to exempt interrupt actions from blocking."""
    for hook_id, bindings in _registry.items():
        if not hook_id.startswith('on_interrupt'):
            continue
        for b in bindings:
            if b.mcp_id == mcp_id and b.tool == tool and b.action == action:
                return True
    return False


def get_hook_for_binding(mcp_id: str, tool: str, action: str) -> str | None:
    """Find the hook_id that a tool+action is registered under."""
    for hook_id, bindings in _registry.items():
        for b in bindings:
            if b.mcp_id == mcp_id and b.tool == tool and b.action == action:
                return hook_id
    return None


def has_bindings(hook_id: str) -> bool:
    """Is anything registered under this hook at all.

    A deployment with no `on_notify` binding has no channel to the user, so
    anything that would narrate through it should no-op rather than spend an
    LLM call producing text nobody can hear.
    """
    return bool(_registry.get(hook_id))


def notify_resources(hook_id: str = 'on_notify') -> frozenset:
    """这个 hook 的绑定一共会占用哪些物理通道。

    用来判断一个刚完成的 action「是不是面向用户的输出」—— 不能按工具名猜，
    机器人的嘴叫 tts / speaker / audio_play 各有各的叫法（见 peer/tools.py 的教训）。
    """
    import mcp_client
    out: set = set()
    for b in _registry.get(hook_id) or []:
        entry = mcp_client.registry.get(b.mcp_id) or {}
        tool_meta = entry.get('tool_meta', {})
        meta = (tool_meta.get(f'mcp__{b.mcp_id}__{b.tool}__{b.action}')
                or tool_meta.get(f'mcp__{b.mcp_id}__{b.tool}'))
        res = (meta or {}).get('resource')
        if res:
            out |= set(res)
    return frozenset(out)


def notify_resource_busy(hook_id: str = 'on_notify') -> bool:
    """True if *every* binding of `hook_id` would be skipped as resource-busy.

    Same tool_meta lookup `mcp_client.call_tool_hook` does (schema name first
    with the action suffix, then without) — doing it up front lets a caller
    avoid paying for an LLM call whose output would be dropped on the floor.

    A binding that declares no `x-resource` is never "busy": call_tool_hook
    dispatches it unconditionally, so claiming otherwise here would suppress a
    narration that would in fact have been heard.

    用 `resource_actually_busy` 而不是 `conflicting_pending` —— 后者会把「已经播完、
    只是还没被 barrier 回收」的 action 也算成占用，那会让一句播报把嘴锁住好几分钟，
    期间所有自动播报被静默跳过（Orin5 上实测 2 分 34 秒）。
    """
    import mcp_client
    bindings = _registry.get(hook_id) or []
    if not bindings:
        return False
    for b in bindings:
        entry = mcp_client.registry.get(b.mcp_id) or {}
        tool_meta = entry.get('tool_meta', {})
        meta = (tool_meta.get(f'mcp__{b.mcp_id}__{b.tool}__{b.action}')
                or tool_meta.get(f'mcp__{b.mcp_id}__{b.tool}'))
        resource = (meta or {}).get('resource')
        if not resource:
            return False
        if not mcp_client.resource_actually_busy(resource):
            return False
    return True


def get_status() -> dict:
    """Return hook registry and recent fire log for diagnostics."""
    return {
        'registry': list_hooks(),
        'recent_fires': list(_fire_log),
    }


# ── Executor ─────────────────────────────────────────────────────────────────

async def fire(hook_id: str, extra_params: dict | None = None, exclude_mcp_id: str | None = None,
               barrier_aware: bool = False) -> list[dict]:
    """Fire a hook: execute all bound tool calls, bypassing the LLM.

    Args:
        hook_id: Hook identifier (e.g. "on_interrupt_all")
        extra_params: Additional params merged into each tool call
        exclude_mcp_id: Skip bindings from this mcp_id (avoid double-fire)
        barrier_aware: False (default) bypasses the barrier too — correct for
            true interrupts (on_interrupt_*, e-stop), which must preempt
            immediately no matter what else is in flight. True routes through
            `mcp_client.call_tool_hook`'s barrier-aware path instead: skip a
            binding if its resource is already busy, and register whatever it
            starts as an ACP pending so normal tool calls wait for it too.
            Use this for hooks that narrate/inform rather than interrupt
            (on_notify) — otherwise they either talk over something already
            playing, or get talked over by the very next tool call themselves.

    Returns:
        List of results from each binding execution.
    """
    import mcp_client  # deferred to avoid circular import

    bindings = _registry.get(hook_id, [])
    if exclude_mcp_id:
        bindings = [b for b in bindings if b.mcp_id != exclude_mcp_id]
    if not bindings:
        return []

    results = []
    tasks = []

    for binding in bindings:
        args = {**binding.params, **(extra_params or {})}
        if binding.action:
            args['action'] = binding.action
        tasks.append(_fire_one(mcp_client, binding, args, barrier_aware=barrier_aware))

    # Execute all bindings in parallel
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    for binding, outcome in zip(bindings, outcomes):
        if isinstance(outcome, Exception):
            results.append({'hook': hook_id, 'mcp_id': binding.mcp_id,
                           'tool': binding.tool, 'error': str(outcome)})
            print(f'[hooks] {hook_id} → {binding.tool}.{binding.action} FAILED: {outcome}')
        else:
            results.append({'hook': hook_id, 'mcp_id': binding.mcp_id,
                           'tool': binding.tool, 'result': outcome})

    if bindings:
        print(f'[hooks] fired {hook_id}: {len(bindings)} binding(s)')
    _fire_log.append({
        'hook': hook_id, 'ts': time.time(),
        'bindings': len(bindings),
        'errors': [r for r in results if 'error' in r],
    })
    return results


async def _fire_one(mcp_client, binding: HookBinding, args: dict, barrier_aware: bool = False) -> Any:
    """Execute a single hook binding via direct tool call."""
    if barrier_aware:
        return await mcp_client.call_tool_hook(binding.mcp_id, binding.tool, args, barrier_aware=True)
    return await mcp_client.call_tool_direct(binding.mcp_id, binding.tool, args)
