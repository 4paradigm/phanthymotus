from .manager import SubagentManager
from .protocol import SubagentSpec, SubagentResult, SubagentStatus

__all__ = ['SubagentManager', 'SubagentSpec', 'SubagentResult', 'SubagentStatus']

# Module-level reference to the active manager (set by event/llm.py on init)
_manager_instance: SubagentManager | None = None


def _set_manager(mgr: SubagentManager) -> None:
    global _manager_instance
    _manager_instance = mgr


def _get_active_subagents() -> list[SubagentStatus]:
    """Get active subagent statuses (used by prompt.py for L2 dynamic)."""
    if _manager_instance is None:
        return []
    return _manager_instance.list_active()


def _get_active_digests(max_turns: int = 3, since: dict | None = None) -> list[dict]:
    """running 子代理的**近况**：目标 + 最近几轮的真实消息。

    进度播报专用。只给 SubagentStatus（目标 + 轮数）的话，汇报器手里除了目标本身什么
    都没有，只能把目标换个说法念一遍。

    `since` 是 {agent_id: 上次汇报时它跑到第几轮}，用来判断"有没有新进展"并只取新的那
    几条。

    **水位用 rounds_completed，不用 turns 列表长度。** 后者会被子代理自己的上下文压缩
    改短（Orin5 实测 round 6 时 msgs 从 21 掉到 18），一压缩水位就大于列表长度、切片为
    空，于是轮数明明在涨（6→7→8）却被判成"没出新结果"，播报连着几条都说"还没有新结果"。
    rounds_completed 单调递增，"轮数涨了"就是"有新进展"最直接的定义。

    turns 仍然用来取**内容**：新跑了 n 轮就取最后 n 条（上限 max_turns）。列表被压缩过
    也没关系 —— `turns[-n:]` 有多少取多少。
    """
    if _manager_instance is None:
        return []
    out = []
    for agent in getattr(_manager_instance, '_agents', {}).values():
        if agent.status != 'running':
            continue
        try:
            turns = list(agent.context.turns)
        except Exception:
            turns = []
        rounds = agent.rounds_completed
        fresh = rounds - since.get(agent.id, 0) if since is not None else None
        stalled = fresh is not None and fresh <= 0
        if max_turns <= 0:
            picked = []
        elif stalled or fresh is None:
            picked = turns[-max_turns:]
        else:
            picked = turns[-min(fresh, max_turns):]
        out.append({
            'id': agent.id,
            'goal': agent.spec.goal,
            'rounds': rounds,
            'turn_count': len(turns),
            'turns': picked,
            'stalled': stalled,
        })
    return out
