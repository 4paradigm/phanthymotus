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


class MemoryConfig(dict):
    def update_atomic(self, values, *, delete_keys=(), delete_prefix=None):
        self.update(copy.deepcopy(values))
        keys = set(delete_keys)
        if delete_prefix is not None:
            keys.update(k for k in self if k.startswith(delete_prefix))
        removed = sum(k in self for k in keys)
        for key in keys:
            self.pop(key, None)
        return removed


def response(mode='steer', addressed=0.99, confidence=0.99):
    return {'model': 'jev-test', 'answers': {
        'addressed': {'type': 'noul', 'noul': addressed},
        'route': {'type': 'choice', 'choice': mode, 'confidence': confidence,
                  'probabilities': {k: float(k == mode) for k in (*routing.MODES, 'uncertain')}}}}


class RoutingTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.identity = Path(_TEMP.name) / 'identity.md'
        self.identity.write_text('你是机器人小范。\n' * 300, encoding='utf-8')
        self.cfg = MemoryConfig({'core': {'project_running': True},
                    'semantic_routing': {**routing.DEFAULTS, 'jev_enabled': True,
                                         'jev_identity_path': str(self.identity)}})
        self.patchers = [patch.object(config, 'main', self.cfg),
                         patch.object(routing, '_settings', self.cfg['semantic_routing']),
                         patch.object(routing, '_configure_lock', asyncio.Lock()),
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

    async def test_disabled_is_exact_passthrough(self):
        self.cfg['semantic_routing']['jev_enabled'] = False
        await self.send()
        self.api.assert_not_called()
        ev = await event_bus.dequeue()
        self.assertNotIn('_semantic_route', ev)
        self.assertEqual(len(event_bus.recent()), 1)

    async def test_voice_rejected_before_all_context(self):
        self.api.return_value = response(addressed=0.1)
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

    async def test_text_not_subject_to_addressed(self):
        self.api.return_value = response('interrupt', addressed=0)
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

    async def test_invalid_route_but_admitted_voice_falls_back(self):
        self.api.return_value['answers']['route'] = {'type': 'choice'}
        await self.send()
        ev = await event_bus.dequeue()
        self.assertIsNone(ev['_semantic_route']['mode'])

    async def test_invalid_addressed_rejects(self):
        for val in (True, float('nan'), 2, '0.9'):
            self.api.return_value = response(addressed=val)
            await self.send()
            self.assertTrue(event_bus._queue.empty())

    async def test_api_failure_voice_reject_text_fallback(self):
        self.api.side_effect = TimeoutError()
        await self.send()
        self.assertTrue(event_bus._queue.empty())
        await self.send(False)
        self.assertEqual(event_bus._queue.qsize(), 1)

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
        self.api.return_value = response(addressed=0)
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
        with patch.object(canvas, 'apply_tool_config') as apply:
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
        self.assertEqual(schema['properties']['jev_enabled']['default'], False)
        self.assertEqual(schema['properties']['jev_model']['x-show-when'], {'jev_enabled': 'true'})
        self.assertIn('narration_silence_seconds', schema['properties'])

    async def test_identity_deleted_during_request_rejects_voice(self):
        async def deleted(*args):
            self.identity.unlink()
            return response()
        self.api.side_effect = deleted
        await self.send()
        self.assertTrue(event_bus._queue.empty())

    async def test_custom_asr_contract_cannot_bypass_admission(self):
        self.api.return_value = response(addressed=0.01)
        await event_bus.enqueue('mcp:custom', json.dumps({'text': '旁人聊天',
                                                        'audio_duration_ms': 1000, 'asr_complete_ts': time.time()}))
        await routing._worker
        self.assertTrue(event_bus._queue.empty())

    async def test_solution_replace_disables_old_invisible_setting(self):
        from api.canvas import delete_all_tool_configs
        await delete_all_tool_configs()
        self.assertFalse(routing.settings()['jev_enabled'])

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
        original = json.dumps({'text': '', 'files': [{'name': 'example.png'}]})
        await event_bus.enqueue('dds:/channel/request/demo', original)
        self.api.assert_not_called()
        self.assertEqual((await event_bus.dequeue())['text'], original)


if __name__ == '__main__':
    unittest.main()
