"""Core admission/dispatch integration, isolated from ROS, hardware and real API."""
import asyncio
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

_TEMP = tempfile.TemporaryDirectory(prefix='core-routing-')
os.environ['DB_PATH'] = str(Path(_TEMP.name) / 'config.db')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import config
import collector
import event_bus
import semantic_routing as routing

_REAL_REQUEST_JEV = routing.request_jev


class MemoryConfig(dict):
    def update_atomic(self, values, *, delete_keys=(), delete_prefix=None):
        keys = set(delete_keys)
        if delete_prefix is not None:
            keys.update(k for k in self if k.startswith(delete_prefix))
        removed = sum(k in self for k in keys)
        for key in keys:
            self.pop(key, None)
        self.update(copy.deepcopy(values))
        return removed


def response(mode='steer', confidence=0.99):
    return {'model': 'jev-test', 'answers': {
        'route': {'type': 'choice', 'choice': mode, 'confidence': confidence,
                  'probabilities': {k: float(k == mode) for k in routing.DECISIONS}}}}


class RoutingTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.identity = Path(_TEMP.name) / 'identity.md'
        self.identity.write_text('你是机器人小范。\n' * 300, encoding='utf-8')
        self.cfg = MemoryConfig({'core': {'project_running': True},
                    'semantic_routing': {**routing.DEFAULTS, 'jev_enabled': True,
                                         'jev_identity_path': str(self.identity)}})
        self.patchers = [patch.object(config, 'main', self.cfg),
                         patch.object(routing, '_saved_api_key', ''),
                         patch.object(routing, '_settings', self.cfg['semantic_routing']),
                         patch.object(routing, '_configure_lock', asyncio.Lock()),
                         patch.object(routing, '_commit_lock', asyncio.Lock()),
                         patch.object(routing, '_activity_task', None),
                         patch.object(routing, '_activity_pending', None),
                         patch.object(routing, '_activity_count', 0),
                         patch.object(routing, '_activity_next_at', 0.0),
                         patch.dict(os.environ, {'TYPESAFE_API_KEY': 'fake-test-key'}),
                         patch.object(routing, 'runtime_snapshot', side_effect=self.snapshot),
                         patch.object(routing, 'request_jev', new_callable=AsyncMock)]
        self.mocks = [p.start() for p in self.patchers]
        self.api = self.mocks[-1]
        self.api.return_value = response()
        self.version = ('session', 0)
        routing._queue.clear()
        routing._worker = None
        routing._diagnostics.clear()
        routing._robot_speech.clear()
        event_bus._queue = asyncio.Queue(maxsize=1024)
        event_bus._recent.clear()
        collector._output = asyncio.Queue(maxsize=64)
        collector._steering_queue = asyncio.Queue(maxsize=32)
        collector._priority_pending.clear()
        collector._source_ring.clear()
        collector._busy = False
        collector._turn_epoch = 0
        collector._cancel_event = asyncio.Event()
        collector._reconsider_event = asyncio.Event()
        collector._interrupt_mode = 'steer'
        self.consumer = None

    async def asyncTearDown(self):
        self.cfg['core']['project_running'] = False
        await routing.invalidate()
        if routing._activity_task:
            routing._activity_task.cancel()
            await asyncio.gather(routing._activity_task, return_exceptions=True)
        if self.consumer:
            self.consumer.cancel()
            try:
                await self.consumer
            except asyncio.CancelledError:
                pass
        for p in reversed(self.patchers):
            p.stop()

    def snapshot(self, event=None):
        return {'history': [{'role': 'user', 'content': '网页：介绍一下机器人'}],
                'runtime': {'main_loop_busy': collector._busy}, 'version': self.version}

    async def send(self, voice=True, text='你好小范'):
        await event_bus.enqueue('dds:/remote_control/mic/asr' if voice else 'message',
                                json.dumps({'text': text, 'audio_duration_ms': 1000,
                                            'priority': 1}) if voice else text)
        if routing._worker:
            await routing._worker

    async def test_asr_burst_activity_is_rate_limited_with_bounded_diagnostics(self):
        from api import motus_stream
        self.cfg['core']['project_running'] = False
        with patch.object(motus_stream, 'push_event', new_callable=AsyncMock) as push:
            for _ in range(1000):
                await event_bus.enqueue('asr', 'rejected burst')
            first_task = routing._activity_task
            await asyncio.wait_for(first_task, 1)
            self.assertEqual(push.await_count, 1)
            self.assertEqual(push.call_args.args[0]['payload']['coalesced'], 999)
            self.assertEqual(len(routing._diagnostics), 100)
            for _ in range(1000):
                await event_bus.enqueue('asr', 'second burst')
            await asyncio.sleep(0.02)
            self.assertEqual(push.await_count, 1)  # global one-second cooldown
            await asyncio.wait_for(routing._activity_task, 2)
            self.assertEqual(push.await_count, 2)
            self.assertEqual(push.call_args.args[0]['payload']['coalesced'], 999)
            self.assertIsNone(routing._activity_pending)
            self.assertEqual(len(routing._diagnostics), 100)
        self.api.assert_not_called()

    async def test_slow_activity_sink_keeps_only_one_pending_sample(self):
        from api import motus_stream
        started, release = asyncio.Event(), asyncio.Event()
        async def slow(*args):
            started.set()
            await release.wait()
        with patch.object(motus_stream, 'push_event', side_effect=slow) as push:
            routing.note({'source': 'asr'}, 'first')
            await asyncio.wait_for(started.wait(), 1)
            task = routing._activity_task
            for n in range(1000):
                routing.note({'source': f'asr:{n}'}, 'dispatch', actual='followup')
                self.assertIs(routing._activity_task, task)
            self.assertEqual(push.await_count, 1)
            self.assertEqual(routing._activity_pending['source'], 'asr:999')
            self.assertEqual(routing._activity_count, 1000)
            self.assertEqual(len(routing._diagnostics), 100)
            release.set()
            await asyncio.wait_for(task, 2)
            self.assertEqual(push.await_count, 2)
            self.assertEqual(push.call_args.args[0]['payload']['source'], 'asr:999')

    async def test_activity_failure_does_not_break_routing_or_retry_forever(self):
        from api import motus_stream
        with patch.object(motus_stream, 'push_event', side_effect=OSError('offline')) as push:
            await self.send(False)
            await asyncio.wait_for(routing._activity_task, 1)
            self.assertEqual(push.await_count, 1)
            self.assertEqual(event_bus._queue.qsize(), 1)
            self.assertTrue(routing._diagnostics)

    async def test_ignore_does_not_enter_context_or_steering(self):
        collector._busy = True
        for text in ('小李，我想到了一个问题', '我可以帮你拍照'):
            self.api.return_value = response('ignore')
            await self.send(True, text)
            self.assertTrue(event_bus._queue.empty())
            self.assertFalse(event_bus.recent())
            self.assertTrue(collector._steering_queue.empty())
            self.assertTrue(any(d['reason'] == 'ignore' and d.get('actual') == 'reject'
                                for d in routing._diagnostics))

    async def test_route_missing_or_malformed_fails_closed_for_voice(self):
        for value in (None, {}, {'type': 'choice', 'choice': 'steer'},
                      {**response()['answers']['route'], 'confidence': float('nan')}):
            self.api.return_value = response()
            self.api.return_value['answers']['route'] = value
            await self.send(True)
            self.assertTrue(event_bus._queue.empty())

    async def test_exact_speech_fragment_rejected_before_jev_not_mixed_or_short(self):
        import hooks
        with patch.object(hooks, 'list_hooks', return_value={'on_notify': [
                {'mcp_id': 'local', 'tool': 'voice', 'action': 'say'}]}):
            ref = routing.begin_robot_speech('local', 'voice', {
                'action': 'say', 'text': '我可以介绍展厅的机器人，还能帮你拍照。'})
        self.assertIsNotNone(ref)
        await self.send(True, '介绍展厅的机器人。')
        self.api.assert_not_called()
        self.assertTrue(event_bus._queue.empty())
        for text in ('停一下', '介绍展厅的机器人，先别讲了', '帮我联系小李'):
            await self.send(True, text)
            self.assertEqual((await event_bus.dequeue())['payload']['text'], text)
        self.assertTrue(self.api.call_args.args[0]['recent_robot_speech'])
        ref['expires'] = time.monotonic() - 1
        await self.send(True, '介绍展厅的机器人。')
        self.assertEqual(event_bus._queue.qsize(), 1)

    async def test_speech_references_are_bounded_and_failed_calls_removed(self):
        import hooks
        with patch.object(hooks, 'list_hooks', return_value={'on_notify': [
                {'mcp_id': 'local', 'tool': 'voice', 'action': 'say'}]}):
            self.assertIsNone(routing.begin_robot_speech('local', 'search', {'text': '普通文本工具'}))
            self.assertIsNone(routing.begin_robot_speech('peer', 'voice', {'action': 'say', 'text': '远程'}))
            for n in range(40):
                ref = routing.begin_robot_speech('local', 'voice', {'action': 'say', 'text': f'第{n}次播报'})
            self.assertEqual(len(routing._robot_speech), 16)
            routing.finish_robot_speech(ref, {'content': [{'text': '{"status":"error"}'}]})
            self.assertEqual(len(routing._robot_speech), 15)
            self.assertIsNone(routing.begin_robot_speech('local', 'voice', {'action': 'say', 'text': 'x' * 4097}))
            self.cfg['semantic_routing']['jev_enabled'] = False
            self.assertIsNone(routing.begin_robot_speech('local', 'voice', {'action': 'say', 'text': '关闭'}))

    async def test_real_mcp_dispatch_records_reference_before_request_and_removes_failure(self):
        import hooks, mcp_client
        schema = {'mcp__local__voice__say': {'tool': 'voice', 'action': 'say'}}
        entry = {'url': 'http://unused', 'online': True, 'tools': ['voice'], 'split_map': schema}
        async def rpc(*args):
            self.assertEqual(len(routing._robot_speech), 1)
            return {'content': [{'type': 'text', 'text': '{"status":"error"}'}]}
        with patch.object(hooks, 'list_hooks', return_value={'on_notify': [
                {'mcp_id': 'local', 'tool': 'voice', 'action': 'say'}]}), \
                patch.object(mcp_client, 'registry', {'local': entry}), \
                patch.object(mcp_client, '_jrpc', side_effect=rpc):
            await mcp_client.call_tool('mcp__local__voice__say', {'text': '播报工具返回失败'})
        self.assertFalse(routing._robot_speech)

    async def test_direct_hook_dispatch_records_reference(self):
        import hooks, mcp_client
        from unittest.mock import MagicMock
        session = MagicMock()
        session.__aenter__.return_value = session
        resp = MagicMock()
        resp.__aenter__.return_value = resp
        async def result():
            self.assertEqual(len(routing._robot_speech), 1)
            return {'result': {'content': [{'type': 'text', 'text': '{"status":"accepted"}'}]}}
        resp.json = AsyncMock(side_effect=result)
        session.post.return_value = resp
        with patch.object(hooks, 'list_hooks', return_value={'on_notify': [
                {'mcp_id': 'local', 'tool': 'voice', 'action': 'say'}]}), \
                patch.object(mcp_client, 'registry', {'local': {'url': 'http://unused', 'online': True}}), \
                patch.object(mcp_client.aiohttp, 'ClientSession', return_value=session):
            result = await mcp_client.call_tool_hook('local', 'voice', {'action': 'say', 'text': '系统主动播报'})
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(len(routing._robot_speech), 1)

    async def test_disabled_is_exact_passthrough(self):
        self.cfg['semantic_routing']['jev_enabled'] = False
        await self.send()
        self.api.assert_not_called()
        ev = await event_bus.dequeue()
        self.assertNotIn('_semantic_route', ev)
        self.assertEqual(len(event_bus.recent()), 1)

    async def test_voice_rejected_before_all_context(self):
        self.api.return_value = response('ignore')
        await self.send()
        self.assertTrue(event_bus._queue.empty())
        self.assertEqual(event_bus.recent(), [])
        self.assertEqual(collector._source_ring, {})

    async def test_ingestion_bypass_never_reads_config_db(self):
        from unittest.mock import Mock
        db = Mock()
        db.get.side_effect = AssertionError('event ingestion must not read SQLite')
        with patch.object(config, 'main', db):
            for enabled in (False, True):
                routing._settings['jev_enabled'] = enabled
                for _ in range(100):
                    await event_bus.enqueue('dds:/sensor/imu', '{}')
                if not enabled:
                    await event_bus.enqueue('asr', 'hello')
                    await event_bus.enqueue('message', 'hello')
            snapshot = routing.settings()
            snapshot['jev_enabled'] = False
            self.assertTrue(routing.settings()['jev_enabled'])
        db.get.assert_not_called()
        self.api.assert_not_called()

    async def test_failed_persistence_does_not_publish_settings(self):
        from unittest.mock import MagicMock
        db = MagicMock()
        db.update_atomic.side_effect = OSError('disk unavailable')
        before = routing.settings()
        with patch.object(config, 'main', db):
            with self.assertRaises(OSError):
                await routing.configure({'jev_enabled': False})
        self.assertEqual(routing.settings(), before)

    async def test_status_file_io_runs_off_event_loop(self):
        import fastapi
        import httpx
        import threading
        from api import canvas
        app = fastapi.FastAPI()
        app.include_router(canvas.router, prefix='/api')
        main_thread = threading.get_ident()
        threads = []
        original = routing.identity
        def observed(cfg):
            threads.append(threading.get_ident())
            return original(cfg)
        with patch.object(routing, 'identity', side_effect=observed):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                result = await client.get('/api/canvas/semantic-routing')
        self.assertEqual(result.status_code, 200)
        self.assertTrue(threads)
        self.assertNotIn(main_thread, threads)

    async def test_full_identity_history_and_audio_contract(self):
        await self.send()
        state, voice, _ = self.api.call_args.args
        self.assertTrue(voice)
        self.assertEqual(state['identity'], self.identity.read_text())
        self.assertEqual(state['history'][0]['content'], '网页：介绍一下机器人')
        ev = await event_bus.dequeue()
        self.assertEqual(ev['payload']['duration_ms'], 1000)
        self.assertEqual(ev['_semantic_route']['mode'], 'steer')

    async def test_text_receives_single_route(self):
        self.api.return_value = response('interrupt')
        await self.send(False, '取消当前任务')
        self.assertFalse(self.api.call_args.args[1])
        ev = await event_bus.dequeue()
        self.assertEqual(ev['_semantic_route']['mode'], 'interrupt')

    async def test_uncertain_uses_live_default(self):
        self.api.return_value = response('uncertain')
        await self.send(False)
        ev = await event_bus.dequeue()
        collector._busy = True
        for mode in routing.MODES:
            collector.set_interrupt_mode(mode)
            self.assertEqual(routing.consume_mode(ev), mode)

    async def test_single_choice_has_no_confidence_or_legacy_threshold_gate(self):
        self.assertEqual(set(routing.QUESTIONS), {'route'})
        self.assertEqual(set(routing.QUESTIONS['route']['criteria']), set(routing.DECISIONS))
        self.assertNotIn('jev_addressed_threshold', routing.SCHEMA)
        self.assertNotIn('jev_route_threshold', routing.SCHEMA)
        cfg = {**routing.DEFAULTS, 'jev_addressed_threshold': 1, 'jev_route_threshold': 1}
        for mode in routing.DECISIONS:
            result = routing.parse_result(response(mode, confidence=0.01), True, cfg)
            self.assertEqual(result[0], mode != 'ignore')
            self.assertEqual(result[1], mode if mode in routing.MODES else None)
        self.api.return_value = response('uncertain', confidence=0.01)
        await self.send(True)
        self.assertIsNone((await event_bus.dequeue())['_semantic_route']['mode'])

    async def test_text_ignore_falls_back_without_losing_message(self):
        self.api.return_value = response('ignore')
        await self.send(False)
        self.assertIsNone((await event_bus.dequeue())['_semantic_route']['mode'])

    async def test_invalid_route_text_falls_back(self):
        self.api.return_value['answers']['route'] = {'type': 'choice'}
        await self.send(False)
        ev = await event_bus.dequeue()
        self.assertIsNone(ev['_semantic_route']['mode'])

    async def test_invalid_confidence_rejects(self):
        for val in (True, float('nan'), 2, '0.9'):
            self.api.return_value = response(confidence=val)
            await self.send()
            self.assertTrue(event_bus._queue.empty())

    async def test_api_failure_voice_reject_text_fallback(self):
        self.api.side_effect = OSError('provider unavailable')
        await self.send()
        self.assertTrue(event_bus._queue.empty())
        await self.send(False)
        self.assertEqual(event_bus._queue.qsize(), 1)

    async def test_timeout_uses_live_default_for_voice_and_text(self):
        self.api.side_effect = TimeoutError()
        for voice in (True, False):
            await self.send(voice)
            ev = await event_bus.dequeue()
            self.assertIsNone(ev['_semantic_route']['mode'])
            self.assertIsNone(ev['_semantic_route']['version'])
            collector._busy = True
            for mode in routing.MODES:
                collector.set_interrupt_mode(mode)
                self.assertEqual(routing.consume_mode(ev), mode)
            self.assertTrue(event_bus._queue.empty())
        self.assertFalse(any(d.get('actual') == 'reject' for d in routing._diagnostics))

    async def test_timeout_does_not_cross_stop_or_generation_change(self):
        for stop in (True, False):
            self.cfg['core']['project_running'] = True
            async def timeout(*args):
                if stop:
                    self.cfg['core']['project_running'] = False
                else:
                    routing._generation += 1
                raise TimeoutError()
            self.api.side_effect = timeout
            await self.send()
            self.assertTrue(event_bus._queue.empty())

    async def test_preparation_timeout_defaults_without_api(self):
        self.cfg['semantic_routing']['jev_timeout_s'] = 0.01
        def slow_snapshot(*args):
            time.sleep(0.02)
            return self.snapshot()
        with patch.object(routing, 'runtime_snapshot', side_effect=slow_snapshot):
            await self.send()
        self.api.assert_not_called()
        self.assertIsNone((await event_bus.dequeue())['_semantic_route']['mode'])

    async def test_late_result_cannot_override_timeout_default(self):
        self.cfg['semantic_routing']['jev_timeout_s'] = 0.01
        async def late(*args):
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                return response('interrupt')
        self.api.side_effect = late
        await self.send()
        ev = await event_bus.dequeue()
        self.assertIsNone(ev['_semantic_route']['mode'])
        self.assertTrue(event_bus._queue.empty())

    async def test_missing_identity_voice_reject_text_fallback(self):
        self.identity.unlink()
        await self.send()
        await self.send(False)
        self.assertEqual(event_bus._queue.qsize(), 1)
        self.api.assert_not_called()

    async def test_control_events_bypass_while_api_waits(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def waiting(*args):
            started.set()
            await release.wait()
            return response()
        self.api.side_effect = waiting
        await event_bus.enqueue('asr', '小范')
        await started.wait()
        for source in ('acp:1', 'scheduler:1', 'subagent:1', 'dds:/sensor/temp'):
            await event_bus.enqueue(source, '{"type":"action_complete"}')
        self.assertEqual(event_bus._queue.qsize(), 4)
        release.set()
        await routing._worker

    async def test_switch_off_delivers_text_once_and_drops_voice(self):
        started = asyncio.Event()
        async def waiting(*args):
            started.set()
            await asyncio.Event().wait()
        self.api.side_effect = waiting
        await event_bus.enqueue('message', 'first')
        await started.wait()
        await event_bus.enqueue('asr', '小范')
        await event_bus.enqueue('message', 'second')
        await routing.configure({'jev_enabled': False})
        self.assertEqual([e['text'] for e in event_bus.recent()], ['first', 'second'])
        await event_bus.enqueue('message', 'third')
        self.assertEqual(event_bus._queue.qsize(), 3)

    async def test_stop_drops_all_pending(self):
        started = asyncio.Event()
        async def waiting(*args):
            started.set()
            await asyncio.Event().wait()
        self.api.side_effect = waiting
        await event_bus.enqueue('message', 'first')
        await started.wait()
        await event_bus.enqueue('message', 'second')
        self.cfg['core']['project_running'] = False
        await routing.invalidate()
        self.assertTrue(event_bus._queue.empty())

    async def test_stale_turn_rejudged_once(self):
        calls = 0
        async def changing(*args):
            nonlocal calls
            calls += 1
            self.version = ('session', calls)
            return response('interrupt')
        self.api.side_effect = changing
        await self.send(False)
        self.assertEqual(calls, 2)
        ev = await event_bus.dequeue()
        self.assertIsNone(ev['_semantic_route']['mode'])

    async def test_stale_after_admission_does_not_use_old_interrupt(self):
        self.api.return_value = response('interrupt')
        await self.send(False)
        ev = await event_bus.dequeue()
        self.version = ('different', 1)
        self.assertEqual(routing.consume_mode(ev), 'steer')

    async def test_retry_shares_absolute_budget_and_timeout_defaults(self):
        for voice in (False, True):
            with self.subTest(voice=voice):
                self.cfg['semantic_routing']['jev_timeout_s'] = 0.2
                budgets = []
                cancelled = asyncio.Event()
                async def changing(state, is_voice, cfg):
                    budgets.append(cfg['jev_timeout_s'])
                    if len(budgets) == 1:
                        await asyncio.sleep(0.12)
                        self.version = ('changed', self.version[1] + 1)
                        return response('interrupt')
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cancelled.set()
                self.api.side_effect = changing
                await asyncio.wait_for(self.send(voice), 0.7)
                self.assertEqual(len(budgets), 2)
                self.assertLessEqual(budgets[0], 0.2)
                self.assertLess(budgets[1], 0.1)
                self.assertTrue(cancelled.is_set())
                self.assertEqual(routing.settings()['jev_timeout_s'], 0.2)
                self.assertEqual(event_bus._queue.qsize(), 1)
                self.assertIsNone((await event_bus.dequeue())['_semantic_route']['mode'])

    async def test_failed_rejudge_does_not_reuse_old_voice_acceptance(self):
        calls = 0
        async def changing(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                self.version = ('changed', 1)
                return response('interrupt')
            raise OSError('provider unavailable')
        self.api.side_effect = changing
        await self.send(True)
        self.assertEqual(calls, 2)
        self.assertTrue(event_bus._queue.empty())

    async def test_queue_time_consumes_configured_budget(self):
        self.cfg['semantic_routing']['jev_timeout_s'] = 0.1
        for kind in ('voice', 'text'):
            event = {'source': 'asr' if kind == 'voice' else 'message',
                     'text': 'expired while queued', 'ts': time.time(), 'payload': {}}
            await routing._judge(event, kind, time.monotonic() - 0.2)
        self.api.assert_not_called()
        self.assertEqual(event_bus._queue.qsize(), 2)
        self.assertEqual((await event_bus.dequeue())['source'], 'asr')
        self.assertEqual((await event_bus.dequeue())['source'], 'message')

    async def test_expired_queued_echo_still_rejected(self):
        ev = {'source': 'asr', 'text': '我能介绍展厅的机器人',
              'ts': time.time(), 'payload': {}}
        with patch.object(routing, 'recent_robot_speech',
                          return_value=[{'text': ev['text']}]):
            await routing._judge(ev, 'voice', time.monotonic() - 6)
        self.api.assert_not_called()
        self.assertTrue(event_bus._queue.empty())
        self.assertEqual(routing._diagnostics[-1]['reason'], 'robot_echo')

    async def test_cancellation_after_real_commit_does_not_duplicate_fallback(self):
        original = event_bus.enqueue_accepted
        inserted = asyncio.Event()
        async def pause_after_insertion(event):
            await original(event)
            inserted.set()
            await asyncio.Event().wait()
        with patch.object(event_bus, 'enqueue_accepted', side_effect=pause_after_insertion) as push:
            await event_bus.enqueue('message', 'only once')
            await asyncio.wait_for(inserted.wait(), 1)
            await asyncio.wait_for(routing.invalidate(deliver_text=True), 1)
            self.assertEqual(push.await_count, 1)
        self.assertEqual(event_bus._queue.qsize(), 1)
        self.assertEqual(len(event_bus.recent()), 1)

    async def test_cancellation_before_insertion_retries_without_losing_text(self):
        event_bus._queue = asyncio.Queue(maxsize=1)
        event_bus._queue.put_nowait({'source': 'filler'})
        entered = asyncio.Event()
        original = event_bus.enqueue_accepted
        calls = 0
        async def observed(event):
            nonlocal calls
            calls += 1
            entered.set()
            await original(event)
        with patch.object(event_bus, 'enqueue_accepted', side_effect=observed):
            await event_bus.enqueue('message', 'survives cancellation')
            await asyncio.wait_for(entered.wait(), 1)
            entered.clear()
            invalidation = asyncio.create_task(routing.invalidate(deliver_text=True))
            await asyncio.wait_for(entered.wait(), 1)
            self.assertEqual(calls, 2)
            await event_bus.dequeue()
            await asyncio.wait_for(invalidation, 1)
        self.assertEqual(event_bus._queue.qsize(), 1)
        self.assertEqual(len(event_bus.recent()), 1)

    async def test_competing_commits_share_one_insertion(self):
        event = {'source': 'message', 'text': 'one', 'ts': time.time(), 'payload': {}}
        await asyncio.gather(routing.commit(event), routing.commit(event))
        self.assertEqual(event_bus._queue.qsize(), 1)
        self.assertEqual(len(event_bus.recent()), 1)

    async def test_disable_cutover_allows_new_legacy_text_before_old_fallback(self):
        started, cancelling, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def pending(*args):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelling.set()
                await release.wait()
                raise
        self.api.side_effect = pending
        await event_bus.enqueue('message', 'old in-flight', {'user_role': 'viewer'})
        await asyncio.wait_for(started.wait(), 1)
        await event_bus.enqueue('message', 'old waiting')
        change = asyncio.create_task(routing.configure({'jev_enabled': False}))
        try:
            await asyncio.wait_for(cancelling.wait(), 1)
            await event_bus.enqueue('message', 'new legacy')
            first = await asyncio.wait_for(event_bus.dequeue(), 1)
            self.assertEqual(first['text'], 'new legacy')
            self.assertNotIn('_semantic_route', first)
        finally:
            release.set()
            await asyncio.wait_for(change, 1)
        old = [await event_bus.dequeue(), await event_bus.dequeue()]
        self.assertEqual([e['text'] for e in old], ['old in-flight', 'old waiting'])
        self.assertEqual(old[0]['payload']['user_role'], 'viewer')
        self.assertEqual(self.api.await_count, 1)
        self.assertEqual(len(event_bus.recent()), 3)

    async def test_queue_full_text_survives_voice_dropped(self):
        started = asyncio.Event()
        async def waiting(*args):
            started.set()
            await asyncio.Event().wait()
        self.api.side_effect = waiting
        await event_bus.enqueue('asr', 'initial')
        await started.wait()
        for _ in range(8):
            await event_bus.enqueue('asr', 'waiting')
        await event_bus.enqueue('asr', 'overflow')
        await event_bus.enqueue('message', 'text-overflow')
        self.assertEqual(event_bus._queue.qsize(), 1)
        self.assertEqual(event_bus.recent()[0]['text'], 'text-overflow')

    async def test_expired_text_not_lost(self):
        ev = {'source': 'message', 'text': 'late', 'ts': time.time(), 'payload': {}}
        await routing._judge(ev, 'text', time.monotonic() - 6)
        self.assertEqual(event_bus._queue.qsize(), 1)
        self.api.assert_not_called()

    async def test_external_route_envelope_cannot_bypass(self):
        self.api.return_value = response('ignore')
        await event_bus.enqueue('asr', '旁人聊天', {'_semantic_route': {'mode': 'interrupt'}, 'approved': True})
        await routing._worker
        self.assertTrue(event_bus._queue.empty())

    async def test_real_collector_modes(self):
        collector._busy = True
        self.consumer = asyncio.create_task(collector._drain_loop())
        for mode in routing.MODES:
            self.api.return_value = response(mode)
            await self.send(False, mode)
            for _ in range(5):
                await asyncio.sleep(0)
            if mode == 'steer':
                self.assertEqual(collector._steering_queue.qsize(), 1)
                self.assertTrue(collector._reconsider_event.is_set())
            elif mode == 'interrupt':
                self.assertTrue(collector._cancel_event.is_set())
            else:
                self.assertEqual(len(collector._priority_pending), 2)

    async def test_timeout_voice_reaches_real_collector_default_modes(self):
        collector._busy = True
        self.consumer = asyncio.create_task(collector._drain_loop())
        self.api.side_effect = TimeoutError()
        for mode in routing.MODES:
            collector.set_interrupt_mode(mode)
            await self.send(True, mode)
            for _ in range(5):
                await asyncio.sleep(0)
            if mode == 'steer':
                self.assertEqual(collector._steering_queue.qsize(), 1)
                self.assertTrue(collector._reconsider_event.is_set())
            elif mode == 'interrupt':
                self.assertTrue(collector._cancel_event.is_set())
            else:
                self.assertEqual(len(collector._priority_pending), 2)

    async def test_bot_cannot_interrupt(self):
        collector._busy = True
        self.consumer = asyncio.create_task(collector._drain_loop())
        self.api.return_value = response('interrupt')
        await event_bus.enqueue('dds:/channel/request/demo', json.dumps({'sender_type': 'bot', 'text': 'stop'}))
        await routing._worker
        for _ in range(5):
            await asyncio.sleep(0)
        self.assertFalse(collector._cancel_event.is_set())
        self.assertEqual(len(collector._priority_pending), 1)

    async def test_config_validation_is_atomic(self):
        old = copy.deepcopy(self.cfg)
        with self.assertRaises(ValueError):
            await routing.configure({'jev_identity_path': '/does-not-exist'})
        self.assertEqual(old, self.cfg)
        for v in (-1, 1.1, True, float('nan')):
            with self.assertRaises(ValueError):
                routing.validate({'jev_route_threshold': v})

    async def test_no_secret_or_binary_in_projection(self):
        data = [{'content': [{'type': 'image_url', 'image_url': 'data:xxx'},
                             {'type': 'text', 'text': 'keep'}]},
                {'content': '{"api_key":"sensitive", "result":"okay"}'}]
        cleaned = routing.text_only(data)
        self.assertNotIn('sensitive', json.dumps(cleaned))
        self.assertNotIn('data:xxx', json.dumps(cleaned))
        self.assertIn('okay', json.dumps(cleaned))
        self.assertIn('sensitive', json.dumps(data))

    async def test_http_config_invalid_does_not_persist_and_status_has_no_key(self):
        import fastapi
        import httpx
        from api import canvas
        app = fastapi.FastAPI()
        app.include_router(canvas.router, prefix='/api')
        transport = httpx.ASGITransport(app=app)
        with patch.object(canvas, 'apply_tool_config', new_callable=AsyncMock) as apply:
            async with httpx.AsyncClient(transport=transport, base_url='http://test') as c:
                url = '/api/canvas/tool-config/agentcore/decision_core'
                bad = await c.put(url, json={'jev_enabled': True, 'jev_identity_path': '/missing'})
                self.assertEqual(bad.status_code, 400)
                self.assertNotIn('tool_config:agentcore:decision_core', self.cfg)
                good = await c.put(url, json={'jev_enabled': True, 'jev_identity_path': str(self.identity)})
                self.assertEqual(good.status_code, 200)
                self.assertFalse(set(apply.call_args.args[2]) & set(routing.DEFAULTS),
                                 'deferred MCP push must not replay an already committed Jev config')
                saved = (await c.get(url)).json()['data']
                self.assertTrue(saved['jev_enabled'])
                status = (await c.get('/api/canvas/semantic-routing')).json()['data']
                self.assertEqual(status['identity_status'], 'readable')
                self.assertNotIn('fake-test-key', json.dumps(status))
                self.assertNotIn(self.identity.read_text(), json.dumps(status))
                disabled = await c.put(url, json={'jev_enabled': False})
                self.assertEqual(disabled.status_code, 200)
                self.assertFalse(routing.settings()['jev_enabled'])

    async def test_real_history_snapshot_preserves_context_and_isolation(self):
        # Module initialization needs the normal seeded client config, even
        # when this file runs alone rather than after other suite imports.
        with patch.object(config, 'main', config.ConfigDB()):
            llm = importlib.import_module('event.llm')
        inst = llm.Event()
        inst._turns = [[{'role': 'user', 'content': '网页里的问题'},
                        {'role': 'assistant', 'content': '网页里的回答'}]]
        inst._current_turn = [{'role': 'user', 'content': '当前飞书问题'}]
        inst._summary = '之前的历史摘要'
        before = copy.deepcopy(inst._turns)
        with patch.object(llm, '_event_instance', inst), patch.object(llm, '_last_turn_restricted', False):
            collector._busy = True
            snapshot = llm.routing_snapshot({'source': 'asr'})
            text = json.dumps(snapshot, ensure_ascii=False)
            self.assertIn('网页里的问题', text)
            self.assertIn('当前飞书问题', text)
            self.assertIn('之前的历史摘要', text)
            self.assertEqual(before, inst._turns)
            bot = {'source': 'dds:/channel/request/demo',
                   'text': json.dumps({'sender_type': 'bot', 'text': 'stop'})}
            self.assertEqual(llm.routing_snapshot(bot)['history'], [])

    async def test_short_semantically_admitted_interrupt_not_vetoed_by_duration(self):
        collector._busy = True
        self.consumer = asyncio.create_task(collector._drain_loop())
        self.api.return_value = response('interrupt')
        await event_bus.enqueue('asr', json.dumps({'text': '停', 'audio_duration_ms': 200,
                                                  'priority': 1, 'spans': [{'span': 'asr'}]}))
        await routing._worker
        for _ in range(5):
            await asyncio.sleep(0)
        self.assertTrue(collector._cancel_event.is_set())
        spans = collector._source_ring['asr'][0]['_perf_spans']
        self.assertEqual([s['span'] for s in spans], ['asr', 'jev_route'])

    async def test_identity_edit_rejudges_with_full_new_identity(self):
        calls = []
        async def changing(state, *args):
            calls.append(state['identity'])
            if len(calls) == 1:
                self.identity.write_text('新身份：机器人小新', encoding='utf-8')
            return response()
        self.api.side_effect = changing
        await self.send()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1], self.identity.read_text())
        self.assertEqual(event_bus._queue.qsize(), 1)

    async def test_config_persists_in_isolated_sqlite(self):
        with patch.object(config, 'main', config.ConfigDB()), patch.object(routing, '_settings', dict(routing.DEFAULTS)):
            await routing.configure({'jev_enabled': True, 'jev_identity_path': str(self.identity)})
            self.assertTrue(config.ConfigDB()['semantic_routing']['jev_enabled'])
            self.assertTrue(routing.settings()['jev_enabled'])
            # Fresh module initialization simulates restart, not the live cache.
            spec = importlib.util.spec_from_file_location('routing_restart', routing.__file__)
            restarted = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(restarted)
            self.assertEqual(restarted.settings(), routing.settings())
            await routing.configure({'jev_enabled': False})
            self.assertFalse(config.ConfigDB()['semantic_routing']['jev_enabled'])
            self.assertFalse(routing.settings()['jev_enabled'])

    async def test_clear_identity_path_restores_default(self):
        with patch.object(routing, 'DEFAULT_IDENTITY_PATH', str(self.identity)):
            await routing.configure({'jev_identity_path': ''})
            self.assertEqual(routing.settings()['jev_identity_path'], '')
            self.assertEqual(routing.identity(routing.settings())[1], self.identity.read_text())

    async def test_delete_failure_rolls_back_database_and_runtime(self):
        import fastapi
        import httpx
        from api import canvas
        db = config.ConfigDB()
        before = routing.settings()
        key = 'tool_config:agentcore:decision_core'
        db.update_atomic({'semantic_routing': before, key: before})
        conn = config._get_conn()
        conn.execute("CREATE TRIGGER reject_core_delete BEFORE DELETE ON config "
                     "WHEN OLD.key = 'tool_config:agentcore:decision_core' "
                     "BEGIN SELECT RAISE(ABORT, 'test locked delete'); END")
        conn.commit()
        conn.close()
        app = fastapi.FastAPI()
        app.include_router(canvas.router, prefix='/api')
        try:
            with patch.object(config, 'main', db):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                    result = await client.delete('/api/canvas/tool-config/agentcore/decision_core')
                self.assertEqual(result.status_code, 503)
                self.assertEqual(db[key], before)
                self.assertEqual(db['semantic_routing'], before)
                self.assertEqual(routing.settings(), before)
        finally:
            conn = config._get_conn()
            conn.execute('DROP TRIGGER reject_core_delete')
            conn.commit()
            conn.close()
        with patch.object(config, 'main', db):
            await canvas.delete_tool_config('agentcore', 'decision_core')
        self.assertNotIn(key, db)
        self.assertEqual(db['semantic_routing'], routing.DEFAULTS)
        self.assertEqual(routing.settings(), routing.DEFAULTS)

    async def test_concurrent_writes_merge_without_blocking_ingestion(self):
        import threading
        started, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        main_thread = threading.get_ident()
        threads = []
        original = self.cfg.update_atomic
        def delayed(*args, **kwargs):
            threads.append(threading.get_ident())
            loop.call_soon_threadsafe(started.set)
            if not release.wait(3):
                raise TimeoutError('test worker release')
            return original(*args, **kwargs)
        with patch.object(self.cfg, 'update_atomic', side_effect=delayed):
            first = asyncio.create_task(routing.configure({'jev_addressed_threshold': 0.8}))
            try:
                await asyncio.wait_for(started.wait(), 2)
                second = asyncio.create_task(routing.configure({'jev_route_threshold': 0.9}))
                await event_bus.enqueue('dds:/sensor/imu', '{}')
                self.assertEqual(event_bus._queue.qsize(), 1)
                self.assertEqual(routing.settings()['jev_addressed_threshold'], 0.5)
            finally:
                release.set()
            await asyncio.gather(first, second)
        self.assertNotIn(main_thread, threads)
        self.assertEqual(routing.settings()['jev_addressed_threshold'], 0.8)
        self.assertEqual(routing.settings()['jev_route_threshold'], 0.9)
        self.assertEqual(routing.settings(), self.cfg['semantic_routing'])

    async def test_cancelled_request_finishes_committed_cache_publication(self):
        import threading
        started, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        original = self.cfg.update_atomic
        def delayed(*args, **kwargs):
            loop.call_soon_threadsafe(started.set)
            if not release.wait(3):
                raise TimeoutError('test worker release')
            return original(*args, **kwargs)
        with patch.object(self.cfg, 'update_atomic', side_effect=delayed):
            request = asyncio.create_task(routing.configure({'jev_enabled': False}))
            try:
                await asyncio.wait_for(started.wait(), 2)
                request.cancel()
                await asyncio.sleep(0)
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await request
        self.assertEqual(routing.settings(), self.cfg['semantic_routing'])
        self.assertFalse(routing.settings()['jev_enabled'])

    async def test_solution_marks_identity_path_local_secret_reference(self):
        from api.solutions import _must_clear_props
        sensitive, _ = _must_clear_props({'properties': routing.SCHEMA})
        self.assertIn('jev_identity_path', sensitive)
        self.assertIn('jev_api_key', sensitive)
        self.assertNotIn('TYPESAFE_API_KEY', routing.SCHEMA)

    async def test_schema_registration_real_function(self):
        import ast
        from api import mcp_manage
        src = Path(__file__).resolve().parents[1] / 'src' / 'start.py'
        tree = ast.parse(src.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_register_core_mcp')
        ns = {'semantic_routing': routing}
        with patch.object(mcp_manage, '_get_mcp_list', return_value=[]), patch.object(mcp_manage, '_save_mcp_list') as save:
            exec(compile(ast.Module(body=[fn], type_ignores=[]), str(src), 'exec'), ns)
            ns['_register_core_mcp'](silent=True)
        core = next(m for m in save.call_args.args[0] if m['id'] == 'agentcore')
        schema = core['tools'][0]['configSchema']
        self.assertNotIn('x-status-url', schema)
        self.assertEqual(schema['properties']['jev_api_key']['format'], 'password')
        self.assertTrue(schema['properties']['jev_api_key']['writeOnly'])
        self.assertEqual(schema['properties']['jev_enabled']['default'], False)
        self.assertEqual(schema['properties']['jev_model']['x-show-when'], {'jev_enabled': 'true'})
        self.assertIn('narration_silence_seconds', schema['properties'])

    async def test_ui_key_is_private_atomic_and_blank_preserves_it(self):
        from api import canvas
        import fastapi
        import httpx
        app = fastapi.FastAPI()
        app.include_router(canvas.router, prefix='/api')
        secret = 'fixture-ui-private-key'
        url = '/api/canvas/tool-config/agentcore/decision_core'
        with patch.dict(os.environ, {'TYPESAFE_API_KEY': ''}), \
                patch.object(canvas, 'apply_tool_config', new_callable=AsyncMock) as apply:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as c:
                missing = await c.put(url, json={'jev_enabled': True})
                self.assertEqual(missing.status_code, 400)
                result = await c.put(url, json={'jev_enabled': True, 'jev_api_key': secret})
                self.assertEqual(result.status_code, 200)
                self.assertEqual(routing.api_key(), secret)
                self.assertNotIn('jev_api_key', apply.call_args.args[2])
                for endpoint in (url, '/api/canvas/tool-configs', '/api/canvas/semantic-routing'):
                    self.assertNotIn(secret, (await c.get(endpoint)).text)
                self.assertNotIn(secret, json.dumps(routing.settings()))
                self.assertNotIn(secret, json.dumps(canvas.all_tool_configs()))
                self.assertEqual(self.cfg[routing._CREDENTIAL_ROW]['api_key'], secret)
                for preserved in ('', '****'):
                    self.assertEqual((await c.put(url, json={'jev_api_key': preserved})).status_code, 200)
                    self.assertEqual(routing.api_key(), secret)
                before = copy.deepcopy(self.cfg)
                bad = await c.put(url, json={'jev_api_key': 'replacement', 'jev_identity_path': '/missing'})
                self.assertEqual(bad.status_code, 400)
                self.assertEqual(self.cfg, before)
                with patch.object(self.cfg, 'update_atomic', side_effect=OSError('disk full')):
                    failed = await c.put(url, json={'jev_api_key': 'replacement'})
                self.assertEqual(failed.status_code, 503)
                self.assertEqual(routing.api_key(), secret)
                self.assertNotIn(secret, routing.text_only('secret=' + secret))

    async def test_instance_jev_fields_rejected_before_any_write(self):
        from api import canvas, mcp_manage
        import fastapi
        import httpx
        app = fastapi.FastAPI()
        app.include_router(canvas.router, prefix='/api')
        before = copy.deepcopy(self.cfg)
        url = '/api/canvas/tool-config/agentcore/decision_core/card-fixture'
        with patch.object(canvas, 'apply_tool_config') as apply:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as c:
                for field in routing.SCHEMA:
                    result = await c.put(url, json={field: 'private-fixture'})
                    self.assertEqual(result.status_code, 400, field)
                    self.assertEqual(self.cfg, before)
                    result = await mcp_manage._handle_agentcore_call(mcp_manage.MCPCallRequest(
                        tool='decision_core', arguments={'action': 'config',
                        'instance_id': 'card-fixture', field: 'private-fixture'}))
                    self.assertEqual(result['code'], 400, field)
                    self.assertEqual(self.cfg, before)
                self.assertEqual((await c.put(url, json=[])).status_code, 400)
                apply.assert_not_called()
                for endpoint in (url, '/api/canvas/tool-configs'):
                    self.assertNotIn('private-fixture', (await c.get(endpoint)).text)

    async def test_instance_delete_is_atomic_and_preserves_shared_credentials(self):
        from api import canvas
        import fastapi
        await routing.configure({'jev_api_key': 'machine-private-key'})
        key = canvas.tool_config_key('agentcore', 'decision_core', 'fixture')
        self.cfg[key] = {'trigger_interval_ms': 500}
        before = copy.deepcopy(self.cfg)
        with patch.object(self.cfg, 'update_atomic', side_effect=OSError('disk full')):
            with self.assertRaises(fastapi.HTTPException) as error:
                await canvas.delete_instance_config('agentcore', 'decision_core', 'fixture')
        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(self.cfg, before)
        await canvas.delete_instance_config('agentcore', 'decision_core', 'fixture')
        before.pop(key)
        self.assertEqual(self.cfg, before)
        self.assertEqual(routing.api_key(), 'machine-private-key')

    async def test_post_commit_apply_failure_is_explicit_and_retryable(self):
        from api import canvas, mcp_manage
        # Exercise real application helper, both synchronous preparation errors
        # and asynchronous failures/error results, without touching hardware.
        body = {'jev_api_key': 'private-retry-key', 'trigger_interval_ms': 700}
        with patch('tool_config.find_tool', side_effect=RuntimeError('private-retry-key')):
            result = await canvas.save_tool_config('agentcore', 'decision_core', body)
        self.assertTrue(result['persisted'])
        self.assertFalse(result['runtime_applied'])
        self.assertNotIn('private-retry-key', json.dumps(result))
        self.assertEqual(routing.api_key(), 'private-retry-key')
        calls = [({'trigger_interval_ms': 700}, {})]
        with patch('tool_config.find_tool'), patch('tool_config.plan_config_calls', return_value=calls), \
                patch.object(mcp_manage, 'mcp_call_tool', new_callable=AsyncMock) as call:
            call.side_effect = RuntimeError('private-retry-key')
            failed = await canvas.save_tool_config('agentcore', 'decision_core', body)
            self.assertFalse(failed['runtime_applied'])
            self.assertNotIn('private-retry-key', json.dumps(failed))
            call.side_effect = None
            for error in ({'code': 503}, {'isError': True}):
                call.return_value = error
                failed = await canvas.save_tool_config('agentcore', 'decision_core', body)
                self.assertFalse(failed['runtime_applied'])
            call.return_value = {'code': 200}
            success = await canvas.save_tool_config('agentcore', 'decision_core', body)
            self.assertTrue(success['persisted'])
            self.assertTrue(success['runtime_applied'])
            self.assertNotIn('jev_api_key', call.call_args.args[1].arguments)
        saved = self.cfg[canvas.tool_config_key('agentcore', 'decision_core')]
        self.assertEqual(saved['trigger_interval_ms'], 700)
        self.assertNotIn('jev_api_key', saved)

    async def test_key_validation_and_solution_cannot_replace_machine_key(self):
        from api import canvas, solutions
        from api import config as config_api
        await routing.configure({'jev_api_key': 'machine-private-key'})
        for value in (None, 12, 'invalid\nheader', 'x' * 4097, '中文'):
            with self.assertRaises(ValueError):
                await routing.configure({'jev_api_key': value})
        package = {'cards': [{'id': 'core', 'deviceRef': 'core', 'toolName': 'decision_core'}],
                   'toolConfigs': {'core:decision_core': {'jev_enabled': False, 'jev_api_key': 'injected-key'}}}
        with patch.object(canvas, 'apply_tool_config') as apply, \
                patch.object(canvas, 'notify_layout_changed'), \
                patch.object(config_api, 'stop_removed_cards', new_callable=AsyncMock):
            await solutions._apply_canvas(package, {'core': 'agentcore'})
        self.assertEqual(routing.api_key(), 'machine-private-key')
        self.assertNotIn('injected-key', json.dumps(self.cfg))
        self.assertNotIn('jev_api_key', apply.call_args.args[2])
        await routing.reset_settings(delete_prefix='tool_config:')
        self.assertEqual(routing.api_key(), 'machine-private-key')

    async def test_key_persisted_and_used_by_real_request_function(self):
        # Exercise the production HTTP function with a local fake transport;
        # no credentials or conversation leave the test process.
        from unittest.mock import MagicMock
        db = config.ConfigDB()
        with patch.object(config, 'main', db):
            await routing.configure({'jev_api_key': 'persisted-fixture-key'})
            self.assertEqual(config.ConfigDB()[routing._CREDENTIAL_ROW]['api_key'], 'persisted-fixture-key')
        import subprocess
        env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src'),
               'DB_PATH': str(Path(config.DB_PATH).resolve()), 'TYPESAFE_API_KEY': ''}
        subprocess.run([sys.executable, '-c',
                        'import semantic_routing as r; assert r.api_key() == "persisted-fixture-key"'],
                       env=env, check=True, capture_output=True, timeout=10)
        http_response = MagicMock()
        http_response.status = 200
        http_response.json = AsyncMock(return_value=response())
        session = MagicMock()
        session.post.return_value.__aenter__ = AsyncMock(return_value=http_response)
        client = MagicMock()
        client.return_value.__aenter__ = AsyncMock(return_value=session)
        with patch.object(routing.aiohttp, 'ClientSession', client):
            await _REAL_REQUEST_JEV({}, True, routing.settings())
        self.assertEqual(session.post.call_args.kwargs['headers']['Authorization'],
                         'Bearer persisted-fixture-key')

    async def test_provider_presets_use_real_request_contract(self):
        from unittest.mock import MagicMock
        reply = MagicMock(status=200)
        reply.json = AsyncMock(return_value=response())
        session = MagicMock()
        session.post.return_value.__aenter__ = AsyncMock(return_value=reply)
        client = MagicMock()
        client.return_value.__aenter__ = AsyncMock(return_value=session)
        with patch.object(routing.aiohttp, 'ClientSession', client), patch.dict(os.environ, {
                'TYPESAFE_API_KEY': 'typesafe-fixture', 'OPEN_ROUTER_KEY': 'router-fixture',
                'OPENROUTER_API_KEY': ''}):
            for base, suffix, model, key in (
                    (routing.TYPESAFE_BASE_URL, '/systemone', 'jev-latest', 'typesafe-fixture'),
                    (routing.OPENROUTER_BASE_URL, '/decisions', 'typesafe/jev-1.13', 'router-fixture')):
                cfg = {**routing.DEFAULTS, 'jev_base_url': base}
                await _REAL_REQUEST_JEV({'message': 'synthetic'}, True, cfg)
                args = session.post.call_args
                self.assertEqual(args.args[0], base + suffix)
                self.assertEqual(args.kwargs['headers']['Authorization'], 'Bearer ' + key)
                self.assertEqual(args.kwargs['json']['model'], model)
                self.assertEqual(args.kwargs['json']['questions'], routing.QUESTIONS)
                self.assertFalse(args.kwargs['allow_redirects'])
                await _REAL_REQUEST_JEV({}, False, cfg)
                self.assertEqual(set(session.post.call_args.kwargs['json']['questions']), {'route'})

    async def test_provider_switch_reuses_one_key_and_persists(self):
        await routing.configure({'jev_api_key': 'one-private-key'})
        await routing.configure({'jev_base_url': routing.OPENROUTER_BASE_URL})
        self.assertEqual(routing.api_key(), 'one-private-key')
        await routing.configure({'jev_api_key': 'new-router-key'})
        self.assertEqual(self.cfg[routing._CREDENTIAL_ROW], {'api_key': 'new-router-key'})
        spec = importlib.util.spec_from_file_location('routing_provider_restart', routing.__file__)
        restarted = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(restarted)
        self.assertEqual(restarted.settings()['jev_base_url'], routing.OPENROUTER_BASE_URL)
        self.assertEqual(restarted.api_key(), 'new-router-key')
        await routing.configure({'jev_base_url': routing.TYPESAFE_BASE_URL, 'jev_api_key': ''})
        self.assertEqual(routing.api_key(), 'new-router-key')  # user replaces key on switch
        self.assertNotIn('new-router-key', routing.text_only('new-router-key'))

    async def test_provider_canvas_save_refresh_and_failure_atomicity(self):
        import fastapi, httpx
        from api import canvas
        app = fastapi.FastAPI()
        app.include_router(canvas.router, prefix='/api')
        url = '/api/canvas/tool-config/agentcore/decision_core'
        with patch.object(canvas, 'apply_tool_config', new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as c:
                saved = await c.put(url, json={'jev_base_url': routing.OPENROUTER_BASE_URL,
                                               'jev_api_key': 'private-router-fixture'})
                self.assertEqual(saved.status_code, 200)
                readback = await c.get(url)
                self.assertIn(routing.OPENROUTER_BASE_URL, readback.text)
                self.assertNotIn('private-router-fixture', readback.text)
                before = copy.deepcopy(self.cfg)
                with patch.object(self.cfg, 'update_atomic', side_effect=OSError('disk full')):
                    failed = await c.put(url, json={'jev_base_url': routing.TYPESAFE_BASE_URL,
                                                    'jev_api_key': 'replacement-key'})
                self.assertEqual(failed.status_code, 503)
                self.assertEqual(self.cfg, before)
                self.assertEqual(routing.settings()['jev_base_url'], routing.OPENROUTER_BASE_URL)
                self.assertEqual(routing.api_key(), 'private-router-fixture')

    async def test_identity_deleted_during_request_rejects_voice(self):
        async def deleted(*args):
            self.identity.unlink()
            return response()
        self.api.side_effect = deleted
        await self.send()
        self.assertTrue(event_bus._queue.empty())

    async def test_custom_asr_contract_cannot_bypass_admission(self):
        self.api.return_value = response('ignore')
        await event_bus.enqueue('mcp:custom', json.dumps({'text': '旁人聊天',
                                                        'audio_duration_ms': 1000, 'asr_complete_ts': time.time()}))
        await routing._worker
        self.assertTrue(event_bus._queue.empty())

    async def test_core_start_rejects_failed_config_replay_before_subscribing(self):
        from api import mcp_manage
        import topic_subscriber
        self.cfg['tool_config:agentcore:decision_core'] = {'jev_enabled': True}
        before = copy.deepcopy(self.cfg)
        with patch.object(routing, 'configure', new_callable=AsyncMock) as configure, \
                patch.object(topic_subscriber, 'subscribe') as subscribe:
            for error, code in ((ValueError('invalid identity'), 400), (OSError('disk'), 503)):
                configure.side_effect = error
                result = await mcp_manage._handle_agentcore_call(mcp_manage.MCPCallRequest(
                    tool='decision_core', arguments={'action': 'start', 'input_topic': '/fixture/asr'}))
                self.assertEqual(result['code'], code)
                subscribe.assert_not_called()
                self.assertEqual(self.cfg, before)

    async def test_accepted_custom_asr_preserves_original_payload_metadata(self):
        original = {'channel_id': 'fixture-channel', 'user_role': 'viewer',
                    'routing': {'reply_to': 'fixture-message'}, 'producer_id': 'fixture-source'}
        event = {'source': 'mcp:custom', 'text': json.dumps({
            'text': '你好小范', 'audio_duration_ms': 800, 'asr_complete_ts': time.time()}),
            'payload': copy.deepcopy(original)}
        await event_bus.enqueue(event['source'], event['text'], event['payload'])
        await routing._worker
        accepted = await asyncio.wait_for(event_bus.dequeue(), timeout=1)
        for key, value in original.items():
            self.assertEqual(accepted['payload'][key], value)
        self.assertEqual(accepted['payload']['duration_ms'], 800)
        self.assertTrue(accepted['_semantic_voice'])

    async def test_solution_replace_disables_old_invisible_setting(self):
        from api.canvas import delete_all_tool_configs
        await delete_all_tool_configs()
        self.assertFalse(routing.settings()['jev_enabled'])

    async def test_solution_core_config_is_validated_and_applied_before_return(self):
        from api import canvas, solutions
        from api import config as config_api
        import fastapi
        db = config.ConfigDB()
        old_layout = {'cards': [], 'marker': 'old'}
        old = routing.settings()
        old_key = 'tool_config:old:tts'
        db.update_atomic({'canvas_layout': old_layout, 'semantic_routing': old, old_key: {}})
        package = {'cards': [{'id': 'core-card', 'deviceRef': 'core', 'toolName': 'decision_core'}],
                   'toolConfigs': {'core:decision_core': {'jev_enabled': True, 'jev_identity_path': '/missing'}}}
        with patch.object(config, 'main', db), patch.object(canvas, 'apply_tool_config') as apply, \
                patch.object(canvas, 'notify_layout_changed'), patch.object(config_api, 'stop_removed_cards', new_callable=AsyncMock) as stop:
            with self.assertRaises(fastapi.HTTPException) as error:
                await solutions._apply_canvas(package, {'core': 'agentcore'})
            self.assertEqual(error.exception.status_code, 400)
            self.assertEqual(db['canvas_layout'], old_layout)
            self.assertIn(old_key, db)
            self.assertEqual(routing.settings(), old)
            apply.assert_not_called()
            stop.assert_not_called()
            package['toolConfigs']['core:decision_core']['jev_identity_path'] = str(self.identity)
            result = await solutions._apply_canvas(package, {'core': 'agentcore'})
            self.assertEqual(result['toolConfigsWritten'], 1)
            self.assertNotIn(old_key, db)
            self.assertTrue(routing.settings()['jev_enabled'])
            self.assertEqual(db['semantic_routing'], routing.settings())
            self.assertEqual(db['tool_config:agentcore:decision_core'], routing.settings())
            self.assertFalse(set(apply.call_args.args[2]) & set(routing.DEFAULTS))
            # A Solution without Jev config must synchronously turn it off.
            await solutions._apply_canvas({'cards': [], 'toolConfigs': {}}, {})
            self.assertEqual(routing.settings(), routing.DEFAULTS)

    async def test_solution_database_failure_rolls_back_deleted_rows_and_layout(self):
        from api import canvas, solutions
        from api import config as config_api
        import fastapi
        db = config.ConfigDB()
        old = routing.settings()
        layout = {'cards': [], 'marker': 'must survive'}
        key = 'tool_config:survivor:tts'
        db.update_atomic({'semantic_routing': old, 'canvas_layout': layout, key: {'voice': 'old'}})
        conn = config._get_conn()
        conn.execute("CREATE TRIGGER reject_semantic_write BEFORE INSERT ON config "
                     "WHEN NEW.key = 'semantic_routing' "
                     "BEGIN SELECT RAISE(ABORT, 'test failed write after delete'); END")
        conn.commit()
        conn.close()
        try:
            with patch.object(config, 'main', db), patch.object(canvas, 'apply_tool_config') as apply, \
                    patch.object(config_api, 'stop_removed_cards', new_callable=AsyncMock) as stop:
                with self.assertRaises(fastapi.HTTPException) as error:
                    await solutions._apply_canvas({'cards': [], 'toolConfigs': {}}, {})
                self.assertEqual(error.exception.status_code, 503)
                self.assertEqual(db[key], {'voice': 'old'})
                self.assertEqual(db['canvas_layout'], layout)
                self.assertEqual(db['semantic_routing'], old)
                self.assertEqual(routing.settings(), old)
                apply.assert_not_called()
                stop.assert_not_called()
        finally:
            conn = config._get_conn()
            conn.execute('DROP TRIGGER reject_semantic_write')
            conn.commit()
            conn.close()

    async def test_candidate_arriving_during_invalidation_is_not_stranded(self):
        routing._invalidating = True
        try:
            await event_bus.enqueue('message', 'must survive')
            await event_bus.enqueue('asr', 'not confirmed')
            self.assertEqual(event_bus._queue.qsize(), 1)
            self.assertEqual(len(routing._queue), 0)
        finally:
            routing._invalidating = False

    async def test_legacy_priority_is_preserved(self):
        await event_bus.enqueue('message', '{"text":"urgent", "priority":3}')
        await routing._worker
        self.assertEqual(collector._extract_priority(await event_bus.dequeue()), 3)

    async def test_attachment_only_message_is_not_dropped(self):
        for payload in ({'text': '', 'files': [{'name': 'example.png'}]},
                        {'files': [{'name': 'example.png'}]},
                        {'attachments': [{'name': 'example.png'}]},
                        {'images': ['fixture-image']}):
            original = json.dumps(payload)
            await event_bus.enqueue('dds:/channel/request/demo', original)
            self.api.assert_not_called()
            self.assertEqual((await event_bus.dequeue())['text'], original)

    async def test_text_with_attachment_still_uses_jev(self):
        original = json.dumps({'text': '解释一下这张图', 'files': [{'name': 'example.png'}]})
        await event_bus.enqueue('dds:/channel/request/demo', original)
        await routing._worker
        self.api.assert_awaited_once()
        self.assertEqual((await event_bus.dequeue())['text'], original)

    async def test_solution_runtime_failures_keep_durable_retry_journal(self):
        from api import canvas, solutions, mcp_manage
        db = config.ConfigDB()
        old_card = {'id': 'old-asr', 'mcpId': 'fixture-device', 'toolName': 'asr'}
        db['canvas_layout'] = {'cards': [old_card]}
        package = {'cards': [{'id': 'core', 'deviceRef': 'core', 'toolName': 'decision_core'}],
                   'toolConfigs': {'core:decision_core': {'trigger_interval_ms': 600}}}
        with patch.object(config, 'main', db), \
                patch.object(solutions, '_canvas_runtime_lock', asyncio.Lock()), \
                patch.object(canvas, 'notify_layout_changed'), \
                patch.object(canvas, 'apply_tool_config', new_callable=AsyncMock) as apply, \
                patch.object(mcp_manage, 'mcp_call_tool', new_callable=AsyncMock) as call:
            call.return_value = {'code': 503}
            result = await solutions._apply_canvas(package, {'core': 'agentcore'})
            self.assertTrue(result['persisted'])
            self.assertFalse(result['runtime_applied'])
            apply.assert_not_called()
            self.assertEqual(config.ConfigDB()['canvas_runtime_pending']['stop_cards'], [old_card])
            call.return_value = {'code': 200, 'data': {'state': 'idle'}}
            apply.side_effect = RuntimeError('private-fixture')
            self.assertFalse(await solutions.reconcile_canvas_runtime())
            self.assertIn('canvas_runtime_pending', db)
            # Reapplying the same solution must not lose the now-removed card.
            result = await solutions._apply_canvas(package, {'core': 'agentcore'})
            self.assertFalse(result['runtime_applied'])
            self.assertEqual(db['canvas_runtime_pending']['stop_cards'], [old_card])
            self.assertNotIn('private-fixture', json.dumps(result))
            apply.side_effect = None
            self.assertTrue(await solutions.reconcile_canvas_runtime())
            self.assertNotIn('canvas_runtime_pending', config.ConfigDB())
            self.assertEqual(apply.call_args.args[2]['trigger_interval_ms'], 600)
            self.assertTrue(apply.call_args.kwargs['wait'])

    async def test_solution_rejects_all_instance_jev_fields_before_changes(self):
        from api import canvas, solutions
        from api import config as config_api
        import fastapi
        await routing.configure({'jev_api_key': 'machine-fixture-key'})
        before = copy.deepcopy(self.cfg)
        with patch.object(canvas, 'apply_tool_config') as apply, \
                patch.object(canvas, 'notify_layout_changed') as notify, \
                patch.object(config_api, 'stop_removed_cards', new_callable=AsyncMock) as stop:
            for field in routing.SCHEMA:
                package = {'cards': [{'id': 'core', 'deviceRef': 'core', 'toolName': 'decision_core'}],
                           'toolConfigs': {'core:decision_core:core': {field: 'injected-fixture'}}}
                with self.assertRaises(fastapi.HTTPException) as error:
                    await solutions._apply_canvas(package, {'core': 'agentcore'})
                self.assertEqual(error.exception.status_code, 400, field)
                self.assertEqual(self.cfg, before)
                self.assertEqual(routing.api_key(), 'machine-fixture-key')
                self.assertEqual(package['toolConfigs']['core:decision_core:core'][field], 'injected-fixture')
            apply.assert_not_called()
            stop.assert_not_called()
            notify.assert_not_called()


if __name__ == '__main__':
    unittest.main()
