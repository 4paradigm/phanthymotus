"""Records what the review loop did, so the dashboard can show how it reviewed.

Written as JSONL, one event per line, appended and flushed as the review runs —
the same shape as a build log (`builder.py` streams to disk so the dashboard can
tail an in-progress build), and stored in the same per-job directory so it is
served by the same endpoint style and deleted by the same prune.

Why a file rather than rows in SQLite: a 30-call review is ~100 KB of tool
output, the writes come from a worker thread mid-loop, and the read pattern is
"give me everything after byte N" — which is what a file already does well and
what `store.read_log` already implements for builds.

Nothing here may raise. A trace is an observability aid; losing a line is
acceptable, losing a review because a disk write failed is not.
"""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Per-field cap. Deliberately above *both* review_agent.MAX_TOOL_RESULT (12000)
# and MAX_REVIEW_IN_TRACE (20000), so the trace is never the narrower cap: when
# it was, the trace showed less than the model was actually given, and the traces
# were the only evidence anyone had. That is how a truncation bug survived
# several passes over them — the debugging instrument was lying the same way the
# model's input was. At 8000 it silently overrode MAX_REVIEW_IN_TRACE too, which
# had been dead code for as long as it had existed.
MAX_FIELD_CHARS = 24_000


class ReviewTrace:
    """Append-only JSONL sink for one review.

    Constructed with `None` to disable — tests and any caller without a log
    directory get a working no-op rather than a conditional at every call site.
    """

    def __init__(self, path: Path | None):
        self._path = path
        self._t0 = time.monotonic()
        self._failed = False

    @property
    def path(self) -> Path | None:
        return self._path

    def event(self, kind: str, **fields) -> None:
        """Append one event. `t` is seconds since this trace was created."""
        if self._path is None or self._failed:
            return
        record = {"kind": kind, "t": round(time.monotonic() - self._t0, 3)}
        record.update({k: _trim(v) for k, v in fields.items() if v is not None})
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as e:
            logger.warning(f"Unserialisable trace event {kind!r}: {e}")
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Append per event and let the OS flush: the dashboard tails this
            # file while the review is still running, so buffering would hold
            # events back until the review finished — exactly when they stop
            # being interesting.
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:
            # Log once. A trace that cannot be written must not turn into an
            # error per round for the rest of the review.
            self._failed = True
            logger.warning(f"Review trace disabled ({self._path}): {e}")


def _trim(value):
    """Bound any string that reaches the trace, at any nesting depth."""
    if isinstance(value, str):
        if len(value) <= MAX_FIELD_CHARS:
            return value
        return value[:MAX_FIELD_CHARS] + f"\n… (trimmed, {len(value)} chars)"
    if isinstance(value, dict):
        return {k: _trim(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_trim(v) for v in value]
    return value
