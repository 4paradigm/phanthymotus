"""Regression tests for QQ turn isolation and steering privilege boundaries."""
import asyncio
import json
import pathlib
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import collector  # noqa: E402
from event.llm import Event, _shared_history_allowed, _isolate_private_turn_steering  # noqa: E402


def _qq_event(user_id: str, message_id: str, *, channel_id: str = "qq_main") -> dict:
    payload = {
        "platform": "qq", "channel_id": channel_id, "chat_id": user_id,
        "user_id": user_id, "message_id": message_id, "sender_type": "user",
        "user_role": "viewer", "text": "private message",
    }
    return {
        "source": f"dds:/channel/request/{channel_id}",
        "text": json.dumps(payload), "ts": time.time(),
    }


class QQSessionIsolationTests(unittest.TestCase):
    def test_different_qq_senders_and_sources_never_share_batch(self):
        a = _qq_event("123", "1")
        b = _qq_event("456", "2")
        b_second = _qq_event("456", "3")
        local = {"source": "dds:/sensor/imu", "text": "{}", "ts": time.time()}
        groups = collector._split_by_bot_trust([a, b, b_second, local, a])
        self.assertEqual([len(g) for g in groups], [1, 2, 1, 1])
        self.assertEqual([collector._build_trigger(g, True)["_qq_channel_turn"]
                          for g in groups], [True, True, False, True])
        self.assertEqual(
            collector._build_trigger(groups[1], True)["_channel_principals"],
            [("qq", "456")],
        )

    def test_private_qq_turn_defers_steering_in_all_llm_retry_paths(self):
        events = [{"source": "dds:/local/mic", "text": "private local event"}]
        with mock.patch.object(collector, "defer_priority") as defer:
            self.assertEqual(_isolate_private_turn_steering(events, True), [])
            defer.assert_called_once_with(events)
            defer.reset_mock()
            self.assertEqual(_isolate_private_turn_steering(events, False), events)
            defer.assert_not_called()

    def test_qq_turn_cannot_read_shared_agent_history(self):
        self.assertFalse(_shared_history_allowed({"_qq_channel_turn": True}))
        self.assertFalse(_shared_history_allowed({
            "_bot_channel_event": True, "_trusted_bot_channel_event": False}))
        self.assertTrue(_shared_history_allowed({"source": "local"}))

    def test_qq_in_progress_turn_never_flushes_to_global_sqlite(self):
        agent = Event()
        agent._current_turn = [{"role": "user", "content": "private QQ conversation"}]
        with mock.patch("chat_history.save_turn", side_effect=AssertionError(
                "QQ content must not be saved in global chat_history")):
            agent._flush_current_turn({"_qq_channel_turn": True})
        self.assertIsNone(agent._session_id)


class QQSteeringIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_busy_qq_message_queues_new_turn_without_steering_or_cancel(self):
        pending_before = list(collector._priority_pending)
        old_busy, old_mode = collector._busy, collector._interrupt_mode
        reconsider = asyncio.Event()
        cancel = asyncio.Event()
        try:
            collector._priority_pending.clear()
            while not collector._steering_queue.empty():
                collector._steering_queue.get_nowait()
            collector._busy = True
            collector._interrupt_mode = "steer"
            collector.set_reconsider_event(reconsider)
            collector.set_cancel_event(cancel)
            event = _qq_event("123", "1")
            with (mock.patch.object(collector, "project_running", return_value=True),
                  mock.patch.object(collector.event_bus, "dequeue",
                                    new_callable=mock.AsyncMock,
                                    side_effect=[event, asyncio.CancelledError()])):
                with self.assertRaises(asyncio.CancelledError):
                    await collector._drain_loop()
            self.assertEqual(list(collector._priority_pending), [event])
            self.assertTrue(collector._steering_queue.empty())
            self.assertFalse(reconsider.is_set())
            self.assertFalse(cancel.is_set())
        finally:
            collector._priority_pending.clear()
            collector._priority_pending.extend(pending_before)
            collector._busy, collector._interrupt_mode = old_busy, old_mode
            collector.set_reconsider_event(None)
            collector.set_cancel_event(None)


if __name__ == "__main__":
    unittest.main()


class TestQQVoidedRoundBarrierIsolation(unittest.TestCase):
    def test_voided_round_defers_foreign_priority_events(self):
        import inspect
        source=inspect.getsource(Event._one_turn_impl)
        branch=source.split("if _round_voided:",1)[1].split(
            "# ── 把工具结果加入本轮消息",1)[0]
        self.assertIn("_isolate_private_turn_steering(",branch)
        external=[{"source":"dds:/local/voice","text":"LOCAL_USER_PRIVATE_EVENT"}]
        with mock.patch.object(collector,"defer_priority") as deferred:
            from event.llm import _isolate_private_turn_steering
            safe=_isolate_private_turn_steering(external,True)
            self.assertEqual(safe,[])
            deferred.assert_called_once_with(external)
        self.assertNotIn("LOCAL_USER_PRIVATE_EVENT","".join(str(x) for x in safe))
