"""
utils/log_sampling.py — Transition-plus-sampling gate for hot-path logs.

Vision plugins process frames at camera rate; logging every result floods the
container logs (and with docker's json driver, dominates CPU). The repo rule
is: log state *transitions* unthrottled, and within a steady state log only a
sample (the first occurrence and every Nth thereafter).
"""

from __future__ import annotations


class SampledLogGate:
    """Decide whether a hot-path event is worth a log line.

    check(outcome) returns (should_log, is_transition, occurrence):
    - is_transition: outcome differs from the previous call — always loggable,
      callers may attach detail (e.g. a traceback) only here.
    - occurrence: 1-based count of consecutive same-outcome events; sampling
      passes the first and every `every`-th.
    """

    def __init__(self, every: int = 100):
        self.every = max(1, int(every))
        self._outcome: str | None = None
        self._count = 0

    def check(self, outcome: str) -> tuple[bool, bool, int]:
        transition = outcome != self._outcome
        if transition:
            self._outcome = outcome
            self._count = 1
        else:
            self._count += 1
        should = transition or self._count % self.every == 0
        return should, transition, self._count


def escape_log_text(value: object, cap: int = 200) -> str:
    """Render externally-influenced text safe for a single log line.

    Error strings derived from remotely supplied payloads (camera frames,
    decoder output) may contain control characters or be arbitrarily long;
    both corrupt or bloat the container log stream. Escape C0/ANSI bytes and
    cap the length before logging.
    """
    text = str(value).encode("unicode_escape").decode("ascii")
    if len(text) > cap:
        return text[:cap] + "..."
    return text


__all__ = ["SampledLogGate", "escape_log_text"]
