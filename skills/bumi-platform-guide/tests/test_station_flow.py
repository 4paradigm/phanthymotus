"""Simulated sequencing/transport failures, not robot or audio acceptance."""
import asyncio
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import guide
from station_flow import FlowError, Interrupted, PresenceGate, REMINDER, StationFlow, UNCONFIRMED, WELCOME, carriage, validate_flow
from run_flow import EOF, Hardware, prepare, preflight


def site():
    value = guide.load_json(ROOT / 'site_config.example.json')
    value.update(staff_confirmed=True, confirmed_scene_id=value['scene_id'])
    value['confirmed_config_sha256'] = guide.confirmation_hash(value)
    return value


def settings():
    raw = guide.load_json(ROOT / 'flow.example.json')
    raw.update(driver_url='http://127.0.0.1:15704/mcp', perception_url='http://localhost:15702/mcp',
               vision_topic='/ubuntu/camera/color/objects', asr_topic='/ubuntu/mic/asr', audio_topic='/perception/tts')
    # Synthetic test windows, never deploy as measured robot data.
    for key in ('wave', 'point_a'):
        raw[key].update(window_s=0.1, measured=True)
    return validate_flow(raw)


class FakeIO:
    def __init__(self):
        self.events, self.on_speak, self.fail_motion = [], None, False

    async def motion(self, item, stop, during=None):
        self.events.append(('start', item['recording_id']))
        if self.fail_motion:
            raise FlowError('synthetic motion failure')
        if during:
            await during()
        self.events.append(('exit', item['recording_id']))

    async def speak(self, text, stop, cancel=None):
        self.events.append(('say', text))
        if self.on_speak:
            self.on_speak(text)
        if stop.is_set() or (cancel and cancel.is_set()):
            raise Interrupted()
        await asyncio.sleep(0)

    async def pause(self, seconds, stop, cancel=None):
        if stop.is_set() or (cancel and cancel.is_set()):
            raise Interrupted()
        await asyncio.sleep(0)

    async def close(self):
        self.events.append(('closed', None))


def packet(stamp, present=True, **override):
    return {'timestamp': stamp, 'objects': [{'name': 'person', 'confidence': 0.9}] if present else [], **override}


class GateTests(unittest.TestCase):
    def test_stable_person_once_and_empty_rearms(self):
        gate = PresenceGate()
        self.assertFalse(gate.update(packet(100), 100))
        self.assertTrue(gate.update(packet(101), 101))
        for stamp in (102, 103, 104):
            self.assertFalse(gate.update(packet(stamp), stamp))
        for stamp in range(105, 111):
            self.assertFalse(gate.update(packet(stamp, False), stamp))
        self.assertFalse(gate.update(packet(111), 111))
        self.assertTrue(gate.update(packet(112), 112))

    def test_stale_replayed_future_and_non_person_frames(self):
        for obj in ([], [{'name': 'chair', 'confidence': 1}], [{'name': 'person', 'confidence': float('inf')}], [{'name': 'person', 'confidence': 0.2}]):
            gate = PresenceGate()
            self.assertFalse(gate.update(packet(100, objects=obj), 100))
            self.assertFalse(gate.update(packet(101, objects=obj), 101))
        gate = PresenceGate()
        gate.update(packet(100), 100)
        self.assertFalse(gate.update(packet(100), 101))
        self.assertFalse(gate.update(packet(101), 110))
        self.assertFalse(gate.update(packet(120), 110))

    def test_camera_gap_does_not_mean_empty(self):
        gate = PresenceGate()
        gate.update(packet(100), 100)
        gate.update(packet(101), 101)
        self.assertFalse(gate.update(packet(200), 200))
        self.assertFalse(gate.update(packet(201), 201))
        self.assertTrue(gate.occupied)


class ConfigTests(unittest.TestCase):
    def test_measured_windows_and_local_endpoints_required(self):
        for key, value in (('wave', {'recording_id': 101, 'measured': False, 'window_s': 5}),
                           ('point_a', {'recording_id': 104, 'measured': True, 'window_s': float('nan')}),
                           ('point_a', {'recording_id': 102, 'measured': True, 'window_s': 1}),
                           ('driver_url', 'https://remote.example/mcp'), ('asr_topic', '/mic/asr_background')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_flow({**settings(), key: value})

    def test_carriage_numbers(self):
        for text, expected in [('请问五号车厢在哪边？', 5), ('5车怎么走', 5), ('十二号车厢', 12),
                               ('第十七车', 17), ('二十号车', 20), ('一百号车', 100), ('现在有5个人', None), ('您好', None)]:
            with self.subTest(text=text):
                self.assertEqual(carriage(text), expected)

    def test_unconfirmed_cannot_start(self):
        with self.assertRaises(ValueError):
            StationFlow(guide.load_json(ROOT / 'site_config.example.json'), settings(), FakeIO())


class SequenceTests(unittest.IsolatedAsyncioTestCase):
    def make(self):
        self.now, self.io = 100.0, FakeIO()
        return StationFlow(site(), settings(), self.io, clock=lambda: self.now)

    async def test_full_101_welcome_104_speech_and_three_reminders(self):
        flow = self.make()
        await flow._welcome()
        self.assertEqual(self.io.events[:3], [('start', 101), ('exit', 101), ('say', WELCOME)])
        await flow._answer('请问五号车厢在哪边？')
        self.assertEqual(self.io.events[3:6], [('start', 104), ('say', '模拟场景，请沿A标识方向前往。'), ('exit', 104)])
        self.assertEqual(self.io.events[6:], [('say', REMINDER)] * 3)
        self.assertEqual(flow.broadcasts, 3)

    async def test_b_no_104_and_out_of_range_no_reminders(self):
        flow = self.make()
        await flow._answer('十二号车厢在哪边')
        self.assertFalse(any(e[0] == 'start' for e in self.io.events))
        self.assertIn(('say', '模拟场景，请沿B标识方向前往。'), self.io.events)
        self.io.events.clear()
        await flow._answer('十七号车厢呢')
        self.assertEqual(len(self.io.events), 1)
        self.assertIn('超出配置', self.io.events[0][1])

    async def test_question_interrupts_reminders(self):
        flow = self.make()
        def question(text):
            if text == REMINDER:
                self.now += 1
                flow.speech({'text': '十二号车厢在哪边', 'audio_end_ts': self.now})
        self.io.on_speak = question
        await flow._answer('五号车厢')
        self.assertEqual(sum(e == ('say', REMINDER) for e in self.io.events), 1)
        self.assertEqual(flow.broadcasts, 0)
        self.assertEqual(flow.queue.get_nowait(), ('question', '十二号车厢在哪边'))

    async def test_old_duplicate_empty_and_echo_asr_ignored(self):
        flow = self.make()
        await flow._say(WELCOME)
        for text, stamp in [('', 100), ('五号车厢', 99), (WELCOME, 100), ('五号车厢', 100)]:
            flow.speech({'text': text, 'audio_end_ts': stamp})
        self.assertTrue(flow.queue.empty())
        self.now = 101
        flow.speech({'text': '五号车厢', 'audio_end_ts': 101})
        flow.speech({'text': '五号车厢', 'audio_end_ts': 101})
        self.assertEqual(flow.queue.qsize(), 1)

    async def test_scene_change_latches_direction_off(self):
        flow = self.make()
        flow.speech({'text': '现在换车次了', 'audio_end_ts': 100})
        await flow._answer('五号车厢')
        self.assertEqual(self.io.events, [('say', UNCONFIRMED)])

    async def test_stop_cancels_remaining_and_new_triggers(self):
        flow = self.make()
        def stop(text):
            if text == REMINDER:
                self.now += 1
                flow.speech({'text': '结束站台引导', 'audio_end_ts': self.now})
        self.io.on_speak = stop
        await flow._answer('五号车厢')
        self.assertTrue(flow.stopped.is_set())
        flow.vision(packet(102))
        self.assertTrue(flow.queue.empty())
        self.assertEqual(sum(e == ('say', REMINDER) for e in self.io.events), 1)

    async def test_motion_failure_stops_worker_without_welcome_or_retry(self):
        flow = self.make()
        self.io.fail_motion = True
        flow.vision(packet(100))
        self.now = 101
        flow.vision(packet(101))
        await asyncio.wait_for(flow.run(), 1)
        self.assertEqual(flow.state, 'error')
        self.assertEqual(self.io.events, [('start', 101), ('closed', None)])

    async def test_worker_idle_after_service_and_no_orphan_on_stop(self):
        flow = self.make()
        worker = asyncio.create_task(flow.run())
        self.now = 101
        flow.speech({'text': '五号车厢', 'audio_end_ts': 101})
        await asyncio.wait_for(flow.queue.join(), 1)
        self.assertEqual(flow.state, 'idle')
        self.assertEqual(flow.broadcasts, 3)
        await flow.stop()
        await worker
        self.assertEqual(flow.state, 'stopped')


class FakeMCP:
    def __init__(self):
        self.calls, self.answer = [], {'state': 'running', 'confirmed_started': True}
    def call(self, tool, **args):
        self.calls.append((tool, args))
        if args['action'] == 'stop_playback':
            return {'confirmed': True, 'workmode': 2}
        return dict(self.answer)


class HardwareTests(unittest.IsolatedAsyncioTestCase):
    def make(self):
        self.driver, self.per = FakeMCP(), FakeMCP()
        self.hw = Hardware(self.driver, self.per, settings())
        return self.hw

    async def test_queued_does_not_complete_until_pcm_and_eof(self):
        hw = self.make()
        self.per.answer = {'status': 'queued', 'action_id': 'synthetic'}
        task = asyncio.create_task(hw.speak('测试', asyncio.Event()))
        await asyncio.sleep(0.02)
        self.assertFalse(task.done())
        hw.audio(b'pcm', 1)
        self.assertEqual(hw.speech_slot['frames'], 0)
        hw.audio(b'pcm', hw.speech_slot['since'])
        hw.audio(EOF, hw.speech_slot['since'])
        await asyncio.wait_for(task, 1)
        self.assertIsNone(hw.speech_slot)

    async def test_eof_without_audio_fails_and_interrupts(self):
        hw = self.make()
        self.per.answer = {'status': 'queued', 'action_id': 'synthetic'}
        task = asyncio.create_task(hw.speak('测试', asyncio.Event()))
        await asyncio.sleep(0.02)
        hw.audio(EOF, hw.speech_slot['since'])
        with self.assertRaises(FlowError):
            await task
        self.assertEqual(self.per.calls[-1][1]['action'], 'interrupt')

    async def test_loading_does_not_advance(self):
        hw = self.make()
        self.per.answer = {'state': 'loading'}
        with self.assertRaises(FlowError):
            await hw.speak('测试', asyncio.Event())
        self.assertEqual(len(self.per.calls), 2)

    async def test_motion_speech_exit_order(self):
        hw = self.make()
        called = []
        async def speaking():
            self.assertEqual(self.driver.calls[0][1]['recording_id'], 104)
            called.append('speaking')
        await hw.motion(settings()['point_a'], asyncio.Event(), speaking)
        self.assertEqual(called, ['speaking'])
        self.assertEqual([a[1]['action'] for a in self.driver.calls], ['play_recording', 'stop_playback'])
        self.assertFalse(hw.motion_owned)

    async def test_stop_reaps_parallel_speech_and_exits_motion(self):
        hw = self.make()
        stop, reaped = asyncio.Event(), asyncio.Event()
        async def speaking():
            try:
                await hw.pause(5, stop)
            finally:
                reaped.set()
        task = asyncio.create_task(hw.motion(settings()['point_a'], stop, speaking))
        await asyncio.sleep(0.02)
        stop.set()
        with self.assertRaises(Interrupted):
            await asyncio.wait_for(task, 1)
        self.assertTrue(reaped.is_set())
        self.assertEqual(self.driver.calls[-1][1]['action'], 'stop_playback')

    async def test_rejected_start_never_speaks_or_stops_unknown_action(self):
        hw = self.make()
        self.driver.answer = {'state': 'accepted', 'confirmed_started': False, 'command_sent': False}
        async def forbidden():
            self.fail('must not speak')
        with self.assertRaises(FlowError):
            await hw.motion(settings()['point_a'], asyncio.Event(), forbidden)
        self.assertEqual(len(self.driver.calls), 1)


class PreparationTests(unittest.TestCase):
    def test_real_speaker_playing_contract_and_read_only_preflight(self):
        outer = self
        class Endpoint:
            def __init__(self, url):
                pass
            def tools(self):
                return {'action_recording': {'inputSchema': {'properties': {'action': {'enum': ['play_recording', 'stop_playback']}}}},
                        'speaker': {}, 'tts': {}, 'asr': {}, 'vop': {}}
            def call(self, tool, **args):
                outer.assertEqual(args, {'action': 'info'})
                if tool == 'speaker':
                    return {'state': 'playing', 'last_error': None, 'topic_in': [{'topic': '/perception/tts'}]}
                topic = {'tts': '/perception/tts', 'asr': '/ubuntu/mic/asr', 'vop': '/ubuntu/camera/color/objects'}[tool]
                return {'state': 'running', 'topic_out': [{'topic': topic}]}
        with patch('run_flow.MCP', Endpoint), patch('run_flow.read_db', return_value={}):
            self.assertEqual(len(preflight(settings(), 'unused')), 2)

    def test_mcp_error_content_is_not_treated_as_a_success(self):
        from run_flow import MCP
        with patch.object(MCP, 'request', return_value={'isError': True, 'content': [{'type': 'text', 'text': '{}'}]}):
            with self.assertRaises(FlowError):
                MCP('http://localhost/mcp').call('tts', action='speak', text='测试')

    def test_multiple_tts_outputs_are_rejected(self):
        class Endpoint:
            def __init__(self, url):
                pass
            def tools(self):
                return {'action_recording': {'inputSchema': {'properties': {'action': {'enum': ['play_recording', 'stop_playback']}}}},
                        'speaker': {}, 'tts': {}, 'asr': {}, 'vop': {}}
            def call(self, tool, **args):
                return {'state': 'running', 'topic_out': [{'topic': '/perception/tts'}, {'topic': '/other/tts'}]}
        with patch('run_flow.MCP', Endpoint), patch('run_flow.read_db', return_value={}):
            with self.assertRaises(FlowError):
                preflight(settings(), 'unused')

    def test_prepare_reads_info_only_and_never_invents_windows(self):
        outer = self
        class ReadOnlyMCP:
            def __init__(self, url):
                pass
            def tools(self):
                return {'exercise_routine': {}}
            def call(self, tool, **args):
                outer.assertEqual(args, {'action': 'info'})
                if tool == 'exercise_routine':
                    return {'robot_id': 'test', 'motions': [{'recording_id': 101, 'duration_s': 5, 'reported_duration_s': 6}]}
                return {'topic_out': [{'topic': '/' + tool}]}
        with patch('run_flow.MCP', ReadOnlyMCP), patch('run_flow.discover', return_value={'driver_url': 'http://localhost/mcp', 'perception_url': 'http://localhost/mcp'}):
            raw = prepare('unused', guide.load_json(ROOT / 'flow.example.json'))
        self.assertIsNone(raw['wave']['window_s'])
        self.assertFalse(raw['wave']['measured'])
        self.assertIsNone(raw['point_a']['window_s'])

    def test_preflight_refuses_active_core_without_device_calls(self):
        with patch('run_flow.read_db', return_value={'subscribe_topics': ['/asr']}), patch('run_flow.MCP') as client:
            with self.assertRaises(FlowError):
                preflight(settings(), 'unused')
            client.assert_not_called()


if __name__ == '__main__':
    unittest.main()
