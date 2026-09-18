"""
utils/model_progress.py — one status line for a model fetch, shared by plugins.

Every plugin that downloads weights has the same problem and used to solve it
zero or one times: the card shows one status string, a cold fetch takes from two
seconds (a 4 MB tflite) to a minute and a half (437 MB of fp32 ASR weights over
a rig's uplink), and a static "loading" line for that whole span is
indistinguishable from a hang. ASR was the only plugin that had wired the
downloader's callbacks up to that line; this is that code, lifted so the other
six render identically and the wording lives in one place.

The two callbacks mirror the two waits a fetch actually has:

  progress_cb(pct, mb_done, mb_total)  bytes are moving        -> "… 40% (180/449 MB)"
  stage_cb("extract")                  bytes are in, unpacking -> "正在解压模型 …"

The second exists because a percentage alone lies at the end of an archive
download: a 515 MB Kokoro tarball still has to be decompressed and merged, and
"100% (515/515 MB)" frozen on the card for tens of seconds reads as stuck.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def fetch_status(on_status, label: str):
    """Return ``(progress_cb, stage_cb)`` rendering one model fetch onto a card.

    `on_status(text)` is whatever puts a line on the card — plugins spell this
    differently (an attribute, a dict entry, a setter), so it is passed in
    rather than assumed. `label` names the model to a human, not a path.

    Passing `on_status=None` returns `(None, None)`: a caller that does not
    report status should hand the downloader nothing at all, so the per-chunk
    bookkeeping is skipped rather than computed and thrown away.

    Note these only fire while work is happening. An already-verified model
    returns without a single call, so a warm start never flashes a percentage —
    which is why the caller must still set its own "preparing" line first.
    """
    if on_status is None:
        return None, None

    def _emit(text: str) -> None:
        try:
            on_status(text)
        except Exception as error:  # pragma: no cover - defensive
            # A status line failing must never abort a download that is fine.
            log.debug(f"[model_progress] status callback failed: {error}")

    def progress_cb(pct, mb_done, mb_total) -> None:
        _emit(f"正在下载模型 '{label}' … {pct}% ({mb_done:.0f}/{mb_total:.0f} MB)")

    def stage_cb(stage) -> None:
        if stage == "extract":
            _emit(f"正在解压模型 '{label}' …")

    return progress_cb, stage_cb
