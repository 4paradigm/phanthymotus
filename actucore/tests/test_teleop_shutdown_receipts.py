"""Real runtime/dispatcher shutdown receipts; only a zero-output adapter is used."""
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'plugins'))
from teleop.dispatch import AdapterAck, RecordingAdapter
from teleop.plugin import TeleopPlugin
from teleop.protocol import ProtocolError
from teleop.runtime import TeleopRuntime


class RetryCloseAdapter(RecordingAdapter):
    def __init__(self):
        super().__init__()
        self.failing = True
        self.close_calls = []
        self.stop_calls = []
        self.apply_calls = []
        self.block_close = False
        self.entered = threading.Event()
        self.release = threading.Event()

    def apply(self, intent):
        self.apply_calls.append(intent)
        return super().apply(intent)

    def safe_stop(self, request):
        self.stop_calls.append((request.reason, threading.get_ident()))
        return super().safe_stop(request)

    def close(self):
        self.close_calls.append(threading.get_ident())
        if self.block_close:
            self.entered.set()
            assert self.release.wait(3)
        if self.failing:
            return AdapterAck(False, 'stop_unconfirmed')
        super().close()
        return AdapterAck(True, 'stop_confirmed')


def test_real_close_nack_prevents_config_write_until_fresh_cleanup_succeeds(tmp_path):
    adapter = RetryCloseAdapter()
    runtime = TeleopRuntime(mode='shadow', adapter=adapter, auto_watchdog=False)
    card = TeleopPlugin({'mode':'shadow', 'position_scale':.5,
                        'capture':{'state_file':str(tmp_path/'capture.json')}}, None)
    card._save_configuration(card.cfg)
    saved = card._config_file.read_bytes()
    closed_links = []
    link = SimpleNamespace(lease=None, close=lambda:closed_links.append(True))
    card.runtime, card.adapter, card.link = runtime, adapter, link

    for _ in range(2):
        reply = card.dispatch('teleop', {'action':'config', 'position_scale':.6})
        assert reply == {'state':'error', 'error':'stop_unconfirmed', 'code':'stop_unconfirmed'}
        assert card.runtime is runtime and card.adapter is adapter and card.link is link
        assert card.cfg['position_scale'] == .5 and card._config_file.read_bytes() == saved
        assert not closed_links and not adapter.apply_calls
        assert not runtime.status()['authority_valid']
        assert runtime.status()['dispatch']['state'] == 'fault_latched'
    assert len(adapter.close_calls) == len(adapter.stop_calls) == 2

    adapter.failing = False
    reply = card.dispatch('teleop', {'action':'config', 'position_scale':.6})
    assert reply['state'] == 'idle' and card.cfg['position_scale'] == .6
    assert json.loads(card._config_file.read_text())['values']['position_scale'] == .6
    assert card.runtime is card.adapter is card.link is None and closed_links == [True]
    assert len(adapter.close_calls) == len(adapter.stop_calls) == 3
    assert not adapter.apply_calls and not runtime._dispatcher._worker.is_alive()
    assert runtime.close().ok  # Repeated confirmed close is idempotent, no fourth cleanup.
    assert len(adapter.close_calls) == 3


@pytest.mark.parametrize('released', [False, True])
def test_real_runtime_nack_still_attempts_independent_driver_release(monkeypatch, released):
    adapter = RetryCloseAdapter()
    runtime = TeleopRuntime(mode='shadow', adapter=adapter, auto_watchdog=False)
    card = TeleopPlugin({}, None)
    card.runtime, card.adapter = runtime, adapter
    card.link = SimpleNamespace(lease={'test':'only'}, close=lambda:pytest.fail('link lost before confirmed close'))
    attempts = []
    monkeypatch.setattr(card, '_release_driver', lambda:(attempts.append(True) or released))
    with pytest.raises(ProtocolError, match='stop_unconfirmed'):
        card.stop()
    assert attempts == [True] and card.runtime is runtime and card.link.lease
    assert not adapter.apply_calls and not runtime._dispatcher._worker.is_alive()


def test_live_close_owner_is_not_replaced_and_retry_never_rearms():
    adapter = RetryCloseAdapter()
    adapter.block_close = True
    runtime = TeleopRuntime(mode='shadow', adapter=adapter, auto_watchdog=False)
    try:
        first = runtime.close()
        assert not first.ok and adapter.entered.is_set()
        owner = runtime._dispatcher._worker
        assert owner.is_alive()
        for _ in range(2):
            started = time.monotonic()
            assert not runtime.close().ok
            assert time.monotonic()-started < .5
            assert runtime._dispatcher._worker is owner and owner.is_alive()
            assert len(adapter.close_calls) == 1 and not adapter.apply_calls
        with pytest.raises(ProtocolError):
            runtime.prepare_local_session()
        adapter.release.set()
        owner.join(1)
        assert not owner.is_alive()
        adapter.block_close, adapter.failing = False, False
        assert runtime.close().ok
        assert runtime._dispatcher._worker is not owner
        assert not runtime._dispatcher._worker.is_alive()
        assert [reason for reason, _ in adapter.stop_calls] == ['service_close', 'service_close_retry']
        assert not adapter.apply_calls and not runtime.status()['authority_valid']
    finally:
        adapter.release.set()
        runtime._dispatcher._worker.join(1)


def test_concurrent_close_retry_has_one_cleanup_owner():
    adapter = RetryCloseAdapter()
    runtime = TeleopRuntime(mode='shadow', adapter=adapter, auto_watchdog=False)
    assert not runtime.close().ok
    old_owner = runtime._dispatcher._worker
    assert not old_owner.is_alive()
    adapter.failing = False
    adapter.block_close = True
    replies = []
    workers = [threading.Thread(target=lambda:replies.append(runtime.close())) for _ in range(2)]
    for worker in workers:worker.start()
    try:
        assert adapter.entered.wait(1)
        cleanup_owner = runtime._dispatcher._worker
        assert cleanup_owner is not old_owner and cleanup_owner.is_alive()
        assert len(adapter.close_calls) == 2 and not adapter.apply_calls
        adapter.release.set()
        for worker in workers:worker.join(1)
        assert all(not worker.is_alive() for worker in workers)
        assert len(replies) == 2 and all(reply.ok for reply in replies)
        assert len(adapter.close_calls) == len(adapter.stop_calls) == 2
        assert not runtime.status()['authority_valid'] and not adapter.apply_calls
    finally:
        adapter.release.set()
        for worker in workers:worker.join(1)
