"""
test_peer_backoff.py — a peer that authoritatively rejects us is not re-polled
every five seconds forever.

Background (Tianyi, 2026-09-18): Tianyi's `peers` table was empty while Orin5 and
R1 still had it paired, so every signed background request came back
`403 signature verification failed: unknown_peer`. Nothing reacted to that, so
the pollers kept their normal cadence for seven days:

    POST /api/peer/inbox/state    12363
    POST /api/peer/inbox/ping      1779
    GET  /api/peer/tools/list      1067
    ────────────────────────────────────
                                  15209

Each one costs the receiver a full Ed25519 verification, and together they buried
its log deeply enough that a real, unrelated burst of LLM 400s was invisible in it.

The line drawn here is narrow on purpose:
  - 401/403 — the peer is telling us it does not know us. Back off.
  - connection refused / timeout — "switched off" is the normal state of a robot
    and the existing cadence is deliberate. Untouched.
  - interactive paths (send a message, peer_call, peer_delegate) — a human or the
    LLM asked for it; try for real and return the real error.

Run: cd agent-core && python3 -m pytest tests/test_peer_backoff.py
"""
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from peer import backoff  # noqa: E402

PEER = 'a' * 32
REJECTED = 'https://10.100.129.72:15678 → HTTP 403 {"detail":"signature verification failed: unknown_peer"}'
UNREACHABLE = 'https://10.100.129.72:15678 → ClientConnectorError: Cannot connect to host'


@pytest.fixture(autouse=True)
def _clean():
    backoff.clear()
    yield
    backoff.clear()


def test_a_healthy_peer_is_never_skipped():
    assert not backoff.should_skip(PEER)
    backoff.note_result(PEER, True)
    assert not backoff.should_skip(PEER)


def test_a_403_starts_a_backoff():
    backoff.note_result(PEER, False, REJECTED)
    assert backoff.should_skip(PEER)
    assert backoff.state()[PEER]['retry_in_s'] <= backoff.INITIAL_S


def test_a_401_counts_too():
    backoff.note_result(PEER, False, 'https://x → HTTP 401 unauthorized')
    assert backoff.should_skip(PEER)


def test_connection_failure_does_not_back_off():
    """A robot that is switched off is the normal case; keep the old cadence."""
    backoff.note_result(PEER, False, UNREACHABLE)
    assert not backoff.should_skip(PEER)
    assert backoff.state() == {}


def test_timeout_does_not_back_off():
    backoff.note_result(PEER, False, 'https://x → TimeoutError: ')
    assert not backoff.should_skip(PEER)


def test_mixed_endpoints_do_not_back_off():
    """One endpoint 403s, another is merely unreachable — that second link may
    still come good, and gating would take it out along with the first."""
    backoff.note_result(PEER, False, f'{REJECTED}; {UNREACHABLE}')
    assert not backoff.should_skip(PEER)


def test_every_endpoint_rejecting_does_back_off():
    backoff.note_result(PEER, False, f'{REJECTED}; https://other → HTTP 403 nope')
    assert backoff.should_skip(PEER)


def test_backoff_doubles_and_is_capped():
    delays = []
    for _ in range(12):
        backoff.note_result(PEER, False, REJECTED)
        delays.append(backoff._gated[PEER]['delay'])
        # expire it so the next rejection escalates rather than being skipped
        backoff._gated[PEER]['until'] = 0.0
    assert delays[0] == backoff.INITIAL_S
    assert delays[1] == backoff.INITIAL_S * 2
    assert delays[-1] == backoff.MAX_S
    assert max(delays) <= backoff.MAX_S


def test_success_clears_the_backoff_immediately():
    backoff.note_result(PEER, False, REJECTED)
    assert backoff.should_skip(PEER)
    backoff.note_result(PEER, True)
    assert not backoff.should_skip(PEER)
    assert backoff.state() == {}


def test_repairing_resets_without_waiting():
    """A human just fixed the one-sided pairing; they should not wait 15 minutes."""
    backoff.note_result(PEER, False, REJECTED)
    assert backoff.should_skip(PEER)
    backoff.reset(PEER)
    assert not backoff.should_skip(PEER)


def test_backoff_is_per_peer():
    other = 'b' * 32
    backoff.note_result(PEER, False, REJECTED)
    assert backoff.should_skip(PEER)
    assert not backoff.should_skip(other)


def test_state_reports_why_for_diagnosis():
    backoff.note_result(PEER, False, REJECTED)
    assert 'unknown_peer' in backoff.state()[PEER]['reason']


def test_empty_reason_is_not_a_rejection():
    backoff.note_result(PEER, False, '')
    assert not backoff.should_skip(PEER)
