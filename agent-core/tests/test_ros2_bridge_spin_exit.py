"""
test_ros2_bridge_spin_exit.py — the spin loop stops when the rcl context is gone
instead of hammering a dead context ten times a second.

Background (Tianyi, 2026-09-18): `_spin_loop` caught every exception, printed it,
slept 0.1s and tried again. `rcl_shutdown()` is not only called by this module's
own `stop()` — a plugin teardown or another module on the shutdown path can
invalidate the context while `_running` is still True. From that moment the loop
printed

    [ros2_bridge] spin error: failed to create timer: the given context is not
    valid, either rcl_init() was not called or rcl_shutdown() was called.

exactly ten times a second — the 0.1s sleep — for as long as shutdown took,
about 100 lines each time and 876 in the container's log. A context does not
become valid again, so retrying it can never succeed.

The check is `rclpy.ok()` rather than a match on the message text: the text is an
rcl implementation detail that changes between versions, while `ok()` asks the
question we actually mean.

Run: cd agent-core && python3 -m pytest tests/test_ros2_bridge_spin_exit.py
"""
import os
import pathlib
import sys
import tempfile
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import ros2_bridge  # noqa: E402


class _Executor:
    """Raises on every spin_once, like an executor whose context died."""

    def __init__(self, error='failed to create timer: the given context is not valid'):
        self.calls = 0
        self._error = error

    def spin_once(self, timeout_sec=None):
        self.calls += 1
        raise RuntimeError(self._error)


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setattr(ros2_bridge, '_running', True)
    yield ros2_bridge
    monkeypatch.setattr(ros2_bridge, '_running', False)


def _fake_rclpy(ok):
    return types.SimpleNamespace(ok=lambda: ok)


def test_dead_context_exits_the_loop(bridge, monkeypatch, capsys):
    ex = _Executor()
    monkeypatch.setattr(bridge, '_executor', ex)
    # raising=False: on a machine without ROS2 the name is never bound at all
    # (the import is guarded), and `_spin_loop` treats that NameError as a dead
    # context too — which is the right read there.
    monkeypatch.setattr(bridge, 'rclpy', _fake_rclpy(ok=False), raising=False)

    bridge._spin_loop()          # must return on its own, not spin

    assert ex.calls == 1, f'kept spinning a dead context ({ex.calls} times)'
    assert 'context is gone' in capsys.readouterr().err


def test_a_live_context_still_retries(bridge, monkeypatch):
    """A transient error with a healthy context is the case the retry is for."""
    ex = _Executor()
    monkeypatch.setattr(bridge, '_executor', ex)
    monkeypatch.setattr(bridge, 'rclpy', _fake_rclpy(ok=True), raising=False)

    # Stop after a few rounds so the test terminates; the point is that it did
    # NOT give up on its own after the first error.
    calls = {'n': 0}
    orig = ex.spin_once

    def counted(timeout_sec=None):
        calls['n'] += 1
        if calls['n'] >= 3:
            monkeypatch.setattr(bridge, '_running', False)
        return orig(timeout_sec)

    ex.spin_once = counted
    bridge._spin_loop()
    assert calls['n'] >= 3, 'gave up while the context was still valid'


def test_rclpy_ok_itself_blowing_up_is_treated_as_dead(bridge, monkeypatch, capsys):
    """During interpreter teardown even `ok()` can throw. Exiting is the safe read."""
    ex = _Executor()

    def _boom():
        raise RuntimeError('context handle is invalid')

    monkeypatch.setattr(bridge, '_executor', ex)
    monkeypatch.setattr(bridge, 'rclpy', types.SimpleNamespace(ok=_boom), raising=False)

    bridge._spin_loop()

    assert ex.calls == 1
    assert 'context is gone' in capsys.readouterr().err
