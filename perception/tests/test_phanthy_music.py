"""One-card MCP scenarios with ROS stubs, controlled catalogue and real dispatch."""
import io
import json
import logging
import threading
import unittest
import urllib.request
from unittest.mock import Mock, patch

from vision_stubs import _FakeExecutor
from plugins.phanthy_music import PhanthyMusicPlugin, TOOLS
from plugins.music_catalog.contracts import SCHEMA


def detail():
    return {'schema': SCHEMA, 'track': {'id': 'fictional-1', 'title': 'Fictional',
        'audio': {'url': 'https://audio.example/song.mp3', 'format': 'mp3', 'expires_at': None},
        'usage': {'scope': 'personal_playback', 'attribution': 'Fictional artist'}}}


class CardTests(unittest.TestCase):
    def setUp(self):
        self.plugin = PhanthyMusicPlugin({'catalogue_type': 'mock'}, _FakeExecutor())

    def tearDown(self):
        self.plugin.stop()

    def call(self, action, **args):
        return self.plugin.dispatch('phanthy-music', {'action': action, **args})

    def test_search_works_idle_and_never_starts_audio(self):
        result = self.call('search', language='zh', vocal='female')
        self.assertTrue(result['tracks'])
        self.assertTrue(all(t['vocal'] == 'female' for t in result['tracks']))
        self.assertIsNone(self.plugin._node)
        self.assertEqual(len(self.plugin.get_tools()), 1)
        self.assertEqual(self.plugin.get_tools()[0]['name'], 'phanthy-music')

    def test_id_play_refreshes_details_and_forwards_usage(self):
        self.call('start')
        with patch.object(self.plugin._catalogue, 'execute', return_value=detail()) as query, \
             patch.object(self.plugin._node.player, 'play') as play:
            result = self.call('play', track_id='fictional-1')
        self.assertFalse(result.get('error'))
        self.assertEqual(query.call_args.kwargs['track_id'], 'fictional-1')
        self.assertEqual(play.call_args.args, ('https://audio.example/song.mp3', 'mp3', 'fictional-1',
                                              None, detail()['track']['usage']))
        self.assertFalse(result['resolving_track'])

    def test_mock_never_claims_playable_audio_and_explicit_url_needs_no_catalogue(self):
        self.call('start')
        self.assertEqual(self.call('play', track_id='mock-01')['error']['code'], 'not_playable')
        self.call('config', catalogue_type='none')
        with patch.object(self.plugin._node.player, 'play') as play:
            self.assertFalse(self.call('play', url='https://audio.example/a.mp3').get('error'))
        play.assert_called_once()

    def test_network_request_never_blocks_controls_or_resurrects_cancelled_play(self):
        for control in ('interrupt', 'stop', 'config', 'replacement'):
            with self.subTest(control=control):
                self.call('start')
                entered, release = threading.Event(), threading.Event()
                result = []
                def delayed(**args):
                    entered.set(); release.wait(3)
                    return detail()  # Deliberately ignores cancellation: final guard must hold.
                node = self.plugin._node
                with patch.object(self.plugin._catalogue, 'execute', side_effect=delayed), \
                     patch.object(node.player, 'play') as play:
                    worker = threading.Thread(target=lambda: result.append(self.call('play', track_id='fictional-1')))
                    worker.start()
                    try:
                        self.assertTrue(entered.wait(1))
                        self.assertTrue(self.call('status')['resolving_track'])
                        self.call('duck'); self.call('unduck')
                        if control == 'replacement':
                            self.call('play', url='https://audio.example/new.mp3')
                        elif control == 'config':
                            self.call('config', timeout_ms=4000 if self.plugin._cfg.get('timeout_ms') != 4000 else 3000)
                        else:
                            self.call(control)
                    finally:
                        release.set(); worker.join(2)
                    self.assertFalse(worker.is_alive())
                    self.assertEqual(result[0]['error']['code'], 'cancelled')
                    self.assertEqual(play.call_count, 1 if control == 'replacement' else 0)
                    if control == 'replacement':
                        self.assertEqual(play.call_args.args[0], 'https://audio.example/new.mp3')

    def test_stop_signals_old_player_before_waiting_for_lifecycle_cleanup(self):
        self.call('start')
        node = self.plugin._node
        with self.plugin._lifecycle_lock:
            worker = threading.Thread(target=lambda: self.call('stop'))
            worker.start()
            self.assertTrue(node.player._stop.wait(1))
            self.assertIsNone(self.plugin._node)
            self.assertEqual(self.call('status')['state'], 'idle')
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertTrue(node.destroyed)

    def test_closed_configuration_never_accepts_credentials_or_redirects_key(self):
        for field in ('api_key', 'endpoint', 'api_key_env', 'api_key_file'):
            result = self.call('config', **{field: 'fake-private-value'})
            self.assertIn('error', result)
            self.assertNotIn('fake-private-value', json.dumps(result))
            self.assertNotIn(field, self.plugin._cfg)
        self.assertNotIn('api_key', TOOLS[0]['configSchema']['properties'])

    def test_search_timeout_does_not_interrupt_existing_music_or_tts(self):
        self.call('start')
        with patch.object(self.plugin._catalogue, 'execute', return_value={'error': {'code': 'unavailable'}}), \
             patch.object(self.plugin._node.player, 'interrupt') as interrupt:
            self.assertEqual(self.call('search')['error']['code'], 'unavailable')
            self.assertEqual(self.call('status')['state'], 'running')
            interrupt.assert_not_called()


class MCPTests(unittest.TestCase):
    def test_real_http_tools_list_dispatch_and_log_redaction(self):
        import main
        bundle = main.PerceptionBundle({'plugins': {'phanthy-music': {'enabled': True, 'catalogue_type': 'mock'}}}, _FakeExecutor())
        captured = io.StringIO()
        handler = logging.StreamHandler(captured)
        main.log.addHandler(handler)
        old_level = main.log.level
        main.log.setLevel(logging.INFO)
        server = main.ThreadingHTTPServer(('127.0.0.1', 0), main.make_handler())
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        def rpc(method, params):
            body = json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}).encode()
            req = urllib.request.Request('http://127.0.0.1:' + str(server.server_port), body,
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=2) as response:
                return json.load(response)['result']
        try:
            with patch.object(main, '_bundle', bundle):
                self.assertEqual([t['name'] for t in rpc('tools/list', {})['tools']], ['phanthy-music'])
                self.assertIsNone(bundle.dispatch('phanthy-music_config', {'volume': 10}))
                self.assertIsNone(bundle.dispatch('phanthy-music_start', {}))
                result = rpc('tools/call', {'name': 'phanthy-music', 'arguments': {'action': 'search', 'query': 'fake-private-value'}})
                self.assertTrue(json.loads(result['content'][0]['text'])['tracks'])
                result = rpc('tools/call', {'name': 'phanthy-music', 'arguments': {'action': 'config', 'api_key': 'fake-private-value'}})
                self.assertIn('error', json.loads(result['content'][0]['text']))
                self.assertNotIn('fake-private-value', captured.getvalue())
        finally:
            server.shutdown(); server.server_close(); worker.join(2)
            bundle._plugins[0].stop()
            main.log.removeHandler(handler); main.log.setLevel(old_level)
