"""Regression tests for the bounded log buffer and its SSE consumers.

The trap these guard: /api/logging/*_stream used to read handler.record_list by
absolute index. Bounding that list with a deque makes len() stop growing once
full, so an index-based reader silently stops delivering records forever. These
tests fail loudly if that regresses.

Run: python3 -m pytest agent-core/tests/test_log_buffer.py -q
  or: python3 agent-core/tests/test_log_buffer.py
"""
import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import log as log_mod  # noqa: E402
from api.logging import _record_stream  # noqa: E402


def _make(maxlen):
    h = log_mod.LoggingHandler(maxlen=maxlen)
    h.setFormatter(logging.Formatter('%(message)s'))
    return h


def _emit(handler, n, start=0):
    for i in range(start, start + n):
        handler.handle(logging.LogRecord(
            name='main', level=logging.INFO, pathname=__file__, lineno=1,
            msg='msg-%d' % i, args=(), exc_info=None,
        ))


def _drain(gen, expect_at_least):
    """Pull items from the async generator until we have enough or it idles."""
    out = []

    async def run():
        agen = gen()
        while len(out) < expect_at_least:
            try:
                out.append(await asyncio.wait_for(agen.__anext__(), timeout=1.0))
            except (asyncio.TimeoutError, StopAsyncIteration):
                break
        await agen.aclose()

    asyncio.run(run())
    return [json.loads(b.decode()) for b in out]


def test_buffer_is_bounded_and_counts_drops():
    h = _make(5)
    _emit(h, 20)
    assert len(h.record_list) == 5, len(h.record_list)
    assert h.dropped == 15, h.dropped
    # seq keeps counting past the bound
    seqs = [r.seq for r in h.record_list]
    assert seqs == [16, 17, 18, 19, 20], seqs


def test_seq_strictly_increasing():
    h = _make(100)
    _emit(h, 50)
    seqs = [r.seq for r in h.record_list]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_stream_keeps_delivering_after_buffer_saturates():
    """THE regression: a naive deque swap stalls here forever."""
    h = _make(5)
    _emit(h, 20)                      # buffer already saturated before we read
    gen = _record_stream(h)
    first = _drain(gen, 5)
    assert [d['message'] for d in first] == ['msg-%d' % i for i in range(15, 20)], first

    # New records after saturation must still be delivered.
    async def run():
        agen = gen()
        got = []
        for _ in range(5):            # drain the backlog
            got.append(await asyncio.wait_for(agen.__anext__(), timeout=1.0))
        _emit(h, 3, start=100)        # append while the stream is live
        for _ in range(3):
            got.append(await asyncio.wait_for(agen.__anext__(), timeout=2.0))
        await agen.aclose()
        return [json.loads(b.decode())['message'] for b in got]

    msgs = asyncio.run(run())
    assert msgs[-3:] == ['msg-100', 'msg-101', 'msg-102'], msgs


def test_default_level_is_not_below_debug():
    assert log_mod.logger.level >= logging.DEBUG, log_mod.logger.level


def test_both_routes_use_the_shared_helper():
    import api.logging as api_log
    src = open(api_log.__file__).read()
    # count call sites, not the def line
    assert src.count('_record_stream(handler)()') == 2, 'both routes must share the helper'
    assert 'last_index' not in src, 'index-based consumption must be gone'


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('PASS', name)
            except Exception as e:
                failures += 1
                print('FAIL', name, '->', repr(e))
    sys.exit(1 if failures else 0)
