"""PiPER operation-result vision; no robot control imports."""

__all__ = ["GoalVerifierPlugin"]


def __getattr__(name):
    # Bootstrap/diagnostics must work before optional vision packages exist.
    if name == "GoalVerifierPlugin":
        from .plugin import GoalVerifierPlugin

        return GoalVerifierPlugin
    raise AttributeError(name)
