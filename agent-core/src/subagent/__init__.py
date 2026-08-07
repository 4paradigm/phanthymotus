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
