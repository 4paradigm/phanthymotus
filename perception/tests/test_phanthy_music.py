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
        fixed = patch.multiple('plugins.phanthy_music', CATALOGUE_ENDPOINT='https://catalogue.example',
                               CATALOGUE_API_KEY='fake-internal-key')
        fixed.start()
        self.addCleanup(fixed.stop)
        self.plugin = PhanthyMusicPlugin({'catalogue_type': 'mock'}, _FakeExecutor())

    def tearDown(self):
        self.plugin.stop()

    def call(self, action, **args):
        return self.plugin.dispatch('phanthy_music', {'action': action, **args})

    def test_search_works_idle_and_never_starts_audio(self):
        result = self.call('search', language='zh', vocal='female')
        self.assertTrue(result['tracks'])
        self.assertTrue(all(t['vocal'] == 'female' for t in result['tracks']))
        self.assertIsNone(self.plugin._node)
        self.assertEqual(len(self.plugin.get_tools()), 1)
        self.assertEqual(self.plugin.get_tools()[0]['name'], 'phanthy_music')

    def test_id_play_refreshes_details_and_forwards_usage(self):
        self.call('start')
        with patch.object(self.plugin._catalogue, 'execute', return_value=detail()) as query, \
             patch.object(self.plugin._node.player, 'play') as play:
            result = self.call('play_by_id', track_id='fictional-1')
        self.assertFalse(result.get('error'))
        self.assertEqual(query.call_args.kwargs['track_id'], 'fictional-1')
        self.assertEqual(play.call_args.args, ('https://audio.example/song.mp3', 'mp3', 'fictional-1',
                                              None, detail()['track']['usage']))
        self.assertFalse(result['resolving_track'])
        self.assertEqual(result['track']['id'], 'fictional-1')

    def test_mock_never_claims_playable_audio(self):
        self.call('start')
        self.assertEqual(self.call('play_by_id', track_id='mock-01')['error']['code'], 'not_playable')
        self.assertEqual(self.call('play_by_genre', genre='pop')['error']['code'], 'not_playable')
        self.call('config', catalogue_type='none')
        with patch.object(self.plugin._node.player, 'play') as play:
            self.assertEqual(self.call('play_by_id', track_id='fictional-1')['error']['code'], 'not_configured')
        play.assert_not_called()

    def test_play_selects_one_track_and_returns_relaxation_and_attribution(self):
        self.call('start')
        found = {'schema': SCHEMA, 'tracks': [detail()['track']], 'relaxed': ['artist'],
                 'reply_notice': 'Tell the user which conditions were relaxed.'}
        with patch.object(self.plugin._catalogue, 'execute', return_value=found) as query, \
             patch.object(self.plugin._node.player, 'play') as play:
            result = self.call('play', genre='pop', language='zh',
                               artist='Fictional artist')
        self.assertEqual(query.call_args.kwargs['top_k'], 1)
        self.assertEqual(query.call_args.kwargs['genre'], 'pop')
        self.assertEqual(query.call_args.kwargs['language'], 'zh')
        self.assertEqual(result['relaxed'], ['artist'])
        self.assertEqual(result['reply_notice'], found['reply_notice'])
        self.assertEqual(result['track'], detail()['track'])
        self.assertEqual(play.call_args.args[-1], detail()['track']['usage'])
        # A play opens one ACP session.
        self.assertTrue(result['action_id'].startswith('music-'))
        # This fixture carries no duration, so none is claimed. Declaring a
        # duration we do not know would be worse than declaring none: it sizes
        # a barrier.
        self.assertNotIn('expected_duration_s', result)

    def test_a_known_duration_is_declared_so_the_pending_is_not_reaped(self):
        """A track has no `text` for agent-core to estimate from, so without
        this every track shares one static ceiling and an overrun silently
        reaps the pending mid-playback.
        """
        self.call('start')
        track = {**detail()['track'], 'duration_s': 214}
        found = {'schema': SCHEMA, 'tracks': [track], 'relaxed': []}
        with patch.object(self.plugin._catalogue, 'execute', return_value=found), \
             patch.object(self.plugin._node.player, 'play'):
            result = self.call('play', genre='pop')
        self.assertEqual(result['expected_duration_s'], 214)

    def test_next_continues_the_same_session_rather_than_opening_another(self):
        """One action_id per listening session. A second pending per skip would
        have the barrier believing two songs play at once.
        """
        self.call('start')
        found = {'schema': SCHEMA, 'tracks': [detail()['track']], 'relaxed': []}
        with patch.object(self.plugin._catalogue, 'execute', return_value=found), \
             patch.object(self.plugin._node.player, 'play'):
            first = self.call('play', genre='pop')['action_id']
            second = self.call('next')
        self.assertEqual(second.get('action_id'), first)

    def test_next_needs_no_arguments_and_excludes_what_already_played(self):
        """`exclude_ids` used to be a list the model had to maintain by hand."""
        self.call('start')
        found = {'schema': SCHEMA, 'tracks': [detail()['track']], 'relaxed': []}
        with patch.object(self.plugin._catalogue, 'execute', return_value=found), \
             patch.object(self.plugin._node.player, 'play'):
            self.call('play', genre='pop')
        with patch.object(self.plugin._catalogue, 'execute',
                          return_value=found) as query, \
             patch.object(self.plugin._node.player, 'play'):
            self.call('next')
        # The station's criteria are reused, and the track just played is excluded.
        self.assertEqual(query.call_args.kwargs['genre'], 'pop')
        self.assertIn(detail()['track']['id'],
                      query.call_args.kwargs['exclude_ids'])

    def test_an_unknown_genre_becomes_free_text_instead_of_an_error(self):
        """A model that says "city pop" should get music, not invalid_filter.

        Before this, the only way to learn a real code was to fail a search and
        read `error.allowed`.
        """
        self.call('start')
        found = {'schema': SCHEMA, 'tracks': [detail()['track']], 'relaxed': []}
        with patch.object(self.plugin, '_known_values',
                          return_value=('pop', 'jazz')), \
             patch.object(self.plugin._catalogue, 'execute',
                          return_value=found) as query, \
             patch.object(self.plugin._node.player, 'play'):
            result = self.call('play', genre='city pop')
        self.assertNotIn('genre', query.call_args.kwargs)
        self.assertIn('city pop', query.call_args.kwargs['query'])
        self.assertEqual(result['genre_as_query'], ['city pop'])

    def test_a_failed_lookup_preserves_the_current_song(self):
        """`play` with no criteria is now legal — "放点歌" means anything, and
        the catalogue picks. What must not happen is a failed lookup killing
        whatever is already playing.
        """
        self.call('start')
        for result in ({'schema': SCHEMA, 'request_id': 'fictional-request',
                        'tracks': [], 'relaxed': ['genre']},
                       {'error': {'code': 'unavailable'}}):
            with self.subTest(result=result), \
                 patch.object(self.plugin._catalogue, 'execute',
                              return_value=result) as query, \
                 patch.object(self.plugin._node.player, 'play') as play, \
                 patch.object(self.plugin._node.player, 'interrupt') as interrupt:
                reply = self.call('play', genre='jazz')
                self.assertIn('error', reply)
                self.assertIsInstance(reply['error'], dict)
                query.assert_called()
                if 'request_id' in result:
                    self.assertEqual(reply['request_id'], 'fictional-request')
                    self.assertEqual(reply['relaxed'], ['genre'])
                play.assert_not_called(); interrupt.assert_not_called()

    def test_network_request_never_blocks_controls_or_resurrects_cancelled_play(self):
        for action, arguments in [('play', {'track_id': 'fictional-1'}),
                                  ('play', {'genre': 'pop'})]:
            for control in ('stop_music', 'stop', 'config', 'replacement'):
                with self.subTest(action=action, control=control):
                    self.call('start')
                    entered, release = threading.Event(), threading.Event()
                    result = []
                    def delayed(**args):
                        entered.set(); release.wait(3)
                        # Deliberately ignores cancellation: final guard must hold.
                        return detail() if action == 'play_by_id' else {'tracks': [detail()['track']]}
                    node = self.plugin._node
                    with patch.object(self.plugin._catalogue, 'execute', side_effect=delayed), \
                         patch.object(node.player, 'play') as play:
                        worker = threading.Thread(target=lambda: result.append(self.call(action, **arguments)))
                        worker.start()
                        try:
                            self.assertTrue(entered.wait(1))
                            self.assertTrue(self.call('status')['resolving_track'])
                            # 控制路径在网络卡住时必须照常应答。duck/unduck 已随混音搬到
                            # audio_mixer，这里改用同样不碰网络的只读动作。
                            self.call('now_playing'); self.call('list_genres')
                            if control == 'replacement':
                                replacement = self.call('play', track_id='fictional-2')
                                self.assertEqual(replacement['error']['code'], 'busy')
                            elif control == 'config':
                                # timeout_ms 不再是设置项；换曲库才是会让在途请求作废的
                                # 那种改动。来回切（而不是切到 none）：subTest 之间
                                # 共用同一张卡，切到 none 会让后面几轮的 _query 直接
                                # 抛 not_configured，连 delayed 都不会被调用。
                                self.call('config', catalogue_type=(
                                    'motus_music'
                                    if self.plugin._cfg.get('catalogue_type') == 'mock'
                                    else 'mock'))
                            else:
                                self.call(control)
                        finally:
                            release.set(); worker.join(2)
                        self.assertFalse(worker.is_alive())
                        self.assertEqual(result[0]['error']['code'], 'cancelled')
                        play.assert_not_called()

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

    def test_card_uses_fixed_credentials_without_exposing_configuration(self):
        result = self.call('config', catalogue_type='motus_music')
        self.assertTrue(result['catalogue_configured'])
        self.assertEqual(self.plugin._catalogue._provider.endpoint, 'https://catalogue.example')
        self.assertEqual(self.plugin._catalogue._provider._key, 'fake-internal-key')
        # 换走再换回来：重建出的 service 依然用源码里内置的连接 —— 凭据不是从某处
        # 记住的，而是每次构造时重新从卡片源码里取。（这里原本改 timeout_ms 来触发
        # 重建，而 timeout_ms 已经不再是设置项。）
        before = self.plugin._catalogue
        self.call('config', catalogue_type='mock')
        self.assertIsNot(before, self.plugin._catalogue)
        result = self.call('config', catalogue_type='motus_music')
        self.assertTrue(result['catalogue_configured'])
        self.assertEqual(self.plugin._catalogue._provider.endpoint, 'https://catalogue.example')
        self.assertEqual(self.plugin._catalogue._provider._key, 'fake-internal-key')
        for reply in (result, self.call('info'), self.plugin.get_tools()):
            self.assertNotIn('fake-internal-key', json.dumps(reply))
            self.assertNotIn('https://catalogue.example', json.dumps(reply))
        for field in ('endpoint', 'api_key', 'api_key_env', 'api_key_file'):
            result = self.call('config', **{field: 'fake-private-value'})
            self.assertIn('error', result)
            self.assertNotIn('fake-private-value', json.dumps(result))
            self.assertNotIn(field, self.plugin._cfg)
            # 这张卡已经完全没有 configSchema —— 没有任何需要操作者回答的问题。
            self.assertNotIn('configSchema', TOOLS[0])
            self.assertNotIn(field, TOOLS[0]['inputSchema']['properties'])
        self.assertNotIn('config', TOOLS[0]['inputSchema']['x-action-params'])

    def test_default_card_ignores_external_credential_settings_and_environment(self):
        with patch.dict('os.environ', {'PHANTHY_MUSIC_ENDPOINT': 'https://ignored.example',
                                       'PHANTHY_MUSIC_API_KEY': 'fake-ignored-env-key'}):
            plugin = PhanthyMusicPlugin({'endpoint': 'https://ignored.example',
                                        'api_key': 'fake-ignored-config-key'}, _FakeExecutor())
        try:
            self.assertEqual(plugin._catalogue.kind, 'motus_music')
            self.assertTrue(plugin.dispatch('phanthy_music', {'action': 'info'})['catalogue_configured'])
            self.assertEqual(plugin._catalogue._provider.endpoint, 'https://catalogue.example')
            self.assertEqual(plugin._catalogue._provider._key, 'fake-internal-key')
            self.assertNotIn('api_key', plugin._cfg)
            self.assertNotIn('endpoint', plugin._cfg)
        finally:
            plugin.stop()

    def test_search_timeout_does_not_interrupt_existing_music_or_tts(self):
        self.call('start')
        with patch.object(self.plugin._catalogue, 'execute', return_value={'error': {'code': 'unavailable'}}), \
             patch.object(self.plugin._node.player, 'interrupt') as interrupt:
            self.assertEqual(self.call('search')['error']['code'], 'unavailable')
            self.assertEqual(self.call('status')['state'], 'running')
            interrupt.assert_not_called()


class AsmrCardTests(unittest.TestCase):
    """The ASMR half of the card: separate actions, one shared station."""

    def setUp(self):
        fixed = patch.multiple('plugins.phanthy_music',
                               CATALOGUE_ENDPOINT='https://catalogue.example',
                               CATALOGUE_API_KEY='fake-internal-key')
        fixed.start()
        self.addCleanup(fixed.stop)
        self.plugin = PhanthyMusicPlugin({'catalogue_type': 'mock'}, _FakeExecutor())

    def tearDown(self):
        self.plugin.stop()

    def call(self, action, **args):
        return self.plugin.dispatch('phanthy_music', {'action': action, **args})

    def found(self, **overrides):
        track = {'id': 'fictional-asmr-1', 'kind': 'asmr', 'category': 'bedtime_story',
                 'title': '睡前故事 第 1 期', 'genre': 'other', 'language': 'zh',
                 'vocal': None, 'artist': {'id': None, 'name': ''}, 'duration_s': 150,
                 'audio': {'url': 'https://audio.example/story.wav', 'format': 'wav',
                           'expires_at': None},
                 'usage': {'scope': 'personal_playback', 'attribution': '虚构样例'},
                 **overrides}
        return {'schema': SCHEMA, 'tracks': [track], 'relaxed': []}

    def test_search_asmr_asks_the_catalogue_for_asmr_only(self):
        result = self.call('search_asmr', category='bedtime_story')
        self.assertTrue(result['tracks'])
        self.assertTrue(all(t['kind'] == 'asmr' for t in result['tracks']))
        self.assertTrue(all(t['category'] == 'bedtime_story' for t in result['tracks']))
        self.assertIn('bedtime_story', result['available_categories'])
        self.assertIsNone(self.plugin._node)

    def test_play_asmr_forwards_the_wav_and_opens_a_session(self):
        self.call('start')
        with patch.object(self.plugin._catalogue, 'execute',
                          return_value=self.found()) as query, \
             patch.object(self.plugin._node.player, 'play') as play:
            result = self.call('play_asmr', category='bedtime_story', query='星空')
        self.assertEqual(query.call_args.kwargs['kind'], 'asmr')
        self.assertEqual(query.call_args.kwargs['category'], 'bedtime_story')
        self.assertEqual(play.call_args.args[:3],
                         ('https://audio.example/story.wav', 'wav', 'fictional-asmr-1'))
        # ASMR is as much a playback session as a song is: a two-minute story
        # left without a pending would have `finish` land in the middle of it.
        self.assertTrue(result['action_id'].startswith('music-'))
        self.assertEqual(result['expected_duration_s'], 150)

    def test_the_asmr_actions_cannot_express_a_song_filter(self):
        """The parameter set comes from the action, not from `args`. That is
        what stops a `genre` the model volunteered from reaching the catalogue
        and returning 422 — the failure mode the split exists to remove.
        """
        self.call('start')
        with patch.object(self.plugin._catalogue, 'execute',
                          return_value=self.found()) as query, \
             patch.object(self.plugin._node.player, 'play'):
            self.call('play_asmr', category='meditation', genre='pop',
                      vocal='female', artist='史叶茜', language='zh')
        for field in ('genre', 'vocal', 'artist', 'language'):
            self.assertNotIn(field, query.call_args.kwargs)
        self.assertEqual(query.call_args.kwargs['kind'], 'asmr')

    def test_next_stays_on_the_side_the_user_asked_for(self):
        """A station remembers `kind`, so "换一个" after an ASMR request does
        not drift into songs."""
        self.call('start')
        with patch.object(self.plugin._catalogue, 'execute',
                          return_value=self.found()), \
             patch.object(self.plugin._node.player, 'play'):
            first = self.call('play_asmr', category='chat')['action_id']
        with patch.object(self.plugin._catalogue, 'execute',
                          return_value=self.found(id='fictional-asmr-2')) as query, \
             patch.object(self.plugin._node.player, 'play'):
            following = self.call('next')
        self.assertEqual(query.call_args.kwargs['kind'], 'asmr')
        self.assertEqual(query.call_args.kwargs['category'], 'chat')
        self.assertIn('fictional-asmr-1', query.call_args.kwargs['exclude_ids'])
        # One session across the skip, exactly as for songs.
        self.assertEqual(following['action_id'], first)

    def test_no_category_is_a_valid_request(self):
        """「来点助眠的」names no category, and the catalogue picks from all of
        them. The model must not have to guess one."""
        self.call('start')
        with patch.object(self.plugin._catalogue, 'execute',
                          return_value=self.found()) as query, \
             patch.object(self.plugin._node.player, 'play'):
            self.assertFalse(self.call('play_asmr').get('error'))
        # `kind` and nothing else: no category guessed on the model's behalf.
        self.assertEqual(query.call_args.kwargs['kind'], 'asmr')
        for field in ('category', 'query', 'tags'):
            self.assertNotIn(field, query.call_args.kwargs)

    def test_an_unknown_category_becomes_free_text_rather_than_an_error(self):
        """Same rule as `genre`: 「想听嚼冰块的声音」should get an answer, not a
        vocabulary error."""
        self.call('start')
        with patch.object(self.plugin._catalogue, 'execute',
                          return_value=self.found()) as query, \
             patch.object(self.plugin._node.player, 'play'):
            result = self.call('play_asmr', category='ice_chewing')
        self.assertNotIn('category', query.call_args.kwargs)
        self.assertIn('ice_chewing', query.call_args.kwargs['query'])
        self.assertEqual(result['category_as_query'], ['ice_chewing'])

    def test_now_playing_does_not_describe_an_asmr_entry_as_a_song(self):
        """The protocol shares one track shape, so an ASMR entry arrives with
        `genre: 'other'` and an empty artist. Reported verbatim that is what
        makes a robot announce 「为你播放 other 曲风的歌」over a bedtime story.
        """
        self.call('start')
        with patch.object(self.plugin._catalogue, 'execute',
                          return_value=self.found()), \
             patch.object(self.plugin._node.player, 'play'):
            self.call('play_asmr', category='bedtime_story')
        track = self.call('now_playing')['track']
        self.assertEqual(track['kind'], 'asmr')
        self.assertEqual(track['category'], 'bedtime_story')
        self.assertEqual(track['title'], '睡前故事 第 1 期')
        self.assertEqual((track['genre'], track['artist'], track['vocal']), ('', '', None))

    def test_list_genres_publishes_the_category_codes(self):
        vocabulary = self.call('list_genres')
        self.assertIn('bedtime_story',
                      [x['value'] for x in vocabulary['asmr_categories']])
        self.assertEqual([x['value'] for x in vocabulary['kinds']], ['song', 'asmr'])

    def test_the_declared_asmr_actions_describe_where_sleep_intent_goes(self):
        """The routing lives in the descriptions, so pin the words that carry
        it: a model only reaches `play_asmr` if the card says so.
        """
        schema = self.plugin.get_tools()[0]['inputSchema']
        params = schema['x-action-params']
        self.assertEqual(set(params['play_asmr']['params']),
                         {'query', 'category', 'tags'})
        asmr_text = params['play_asmr']['description']
        for phrase in ('助眠', '哄睡', '睡前故事', '冥想', '白噪音', '咀嚼'):
            self.assertIn(phrase, asmr_text)
        # And `play` must point sleep intent away from itself.
        self.assertIn('play_asmr', params['play']['description'])
        categories = schema['properties']['category']['description']
        for code in ('bedtime_story', 'meditation', 'companion', 'chat',
                     'poetry', 'trigger'):
            self.assertIn(code, categories)


class MCPTests(unittest.TestCase):
    def test_real_http_tools_list_dispatch_and_log_redaction(self):
        import main
        # echo_cancellation is default-on, so it has to be switched off
        # explicitly for the bundle to hold just this card — the tool list
        # below is asserted exactly.
        bundle = main.PerceptionBundle({'plugins': {'phanthy_music': {'enabled': True, 'catalogue_type': 'mock'},
                                                   'echo_cancellation': {'enabled': False},
                                                   'audio_mixer': {'enabled': False}}}, _FakeExecutor())
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
                self.assertEqual([t['name'] for t in rpc('tools/list', {})['tools']], ['phanthy_music'])
                self.assertIsNone(bundle.dispatch('phanthy_music_config', {'volume': 10}))
                self.assertIsNone(bundle.dispatch('phanthy_music_start', {}))
                result = rpc('tools/call', {'name': 'phanthy_music', 'arguments': {'action': 'search', 'query': 'fake-private-value'}})
                self.assertTrue(json.loads(result['content'][0]['text'])['tracks'])
                result = rpc('tools/call', {'name': 'phanthy_music', 'arguments': {'action': 'config', 'api_key': 'fake-private-value'}})
                self.assertIn('error', json.loads(result['content'][0]['text']))
                self.assertNotIn('fake-private-value', result['content'][0]['text'])
                self.assertNotIn('fake-private-value', captured.getvalue())
        finally:
            server.shutdown(); server.server_close(); worker.join(2)
            # Selected by type, not by position: the bundle also loads any
            # other default-on plugin (e.g. echo_cancellation), so an index
            # here would clean up the wrong one.
            for plugin in bundle._plugins:
                if isinstance(plugin, PhanthyMusicPlugin):
                    plugin.stop()
            main.log.removeHandler(handler); main.log.setLevel(old_level)
