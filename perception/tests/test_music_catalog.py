"""Offline motus.music/1 contracts and bounded HTTP provider; no real credentials."""
import io
import json
import threading
import time
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch
import urllib.error

from plugins.music_catalog.contracts import (
    CATEGORIES, FALLBACK, MusicError, SCHEMA, capabilities, search_request,
    search_result)
from plugins.music_catalog.providers import discover
from plugins.music_catalog.providers.motus_music import MotusMusicProvider, retry_seconds, _NoRedirect
from plugins.music_catalog.service import MusicService

def track(**overrides):
    return {'id': 'fictional-1', 'title': 'Fictional', 'language': 'zh', 'genre': 'pop',
            'vocal': 'female', 'artist': {'id': 'artist-1', 'name': 'Fictional artist'},
            'duration_s': 120, 'audio': {'url': 'https://audio.example/test.mp3'}, **overrides}


def asmr(**overrides):
    """An ASMR entry in the shape the catalogue actually returns it.

    Not a song with a field renamed: no artist name, no voice type, `genre`
    pinned to `other`, and a wav. Those are the three fields whose handling the
    card has to get right, so a fixture that filled them in would hide the bug.
    """
    return {'id': 'fictional-asmr-1', 'kind': 'asmr', 'category': 'bedtime_story',
            'title': '睡前故事 第 1 期', 'language': 'zh', 'genre': 'other',
            'vocal': None, 'artist': {'id': None, 'name': ''}, 'duration_s': 150,
            'audio': {'url': 'https://audio.example/test.wav', 'format': 'wav'},
            'lyrics_excerpt': '嘘……', **overrides}


class AsmrContractTests(unittest.TestCase):
    """motus.music/1's second kind of content, over the same endpoint and key."""

    def test_kind_and_category_reach_the_request(self):
        request = search_request('星空', kind='asmr', category='bedtime_story,chat',
                                 tags='睡前')
        self.assertEqual(request['filters']['kind'], 'asmr')
        self.assertEqual(request['filters']['category'], ['bedtime_story', 'chat'])
        self.assertEqual(request['filters']['tags'], ['睡前'])

    def test_a_category_alone_means_asmr(self):
        """The server infers it; inferring it here too is what keeps the local
        cross-kind check reading the same conclusion the server will."""
        self.assertEqual(search_request(category='meditation')['filters']['kind'], 'asmr')

    def test_a_song_request_is_unchanged_by_the_split(self):
        """A caller that only ever wanted songs sends exactly what it used to —
        no `kind`, so no behaviour change and nothing to migrate."""
        self.assertNotIn('kind', search_request('下雨天', genre='pop')['filters'])
        self.assertNotIn('category', search_request('下雨天', genre='pop')['filters'])

    def test_crossed_fields_fail_here_rather_than_as_a_422(self):
        for kwargs in ({'kind': 'asmr', 'genre': 'pop'},
                       {'kind': 'asmr', 'vocal': 'female'},
                       {'kind': 'asmr', 'artist': '史叶茜'},
                       {'kind': 'song', 'category': 'trigger'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(MusicError) as ctx:
                search_request(**kwargs)
            self.assertEqual(ctx.exception.code, 'invalid_filter')

    def test_unknown_kind_or_category_lists_what_is_allowed(self):
        for kwargs, expected in (({'kind': 'podcast'}, ['song', 'asmr']),
                                 ({'category': 'xyz'}, list(CATEGORIES))):
            with self.subTest(kwargs=kwargs), self.assertRaises(MusicError) as ctx:
                search_request(**kwargs)
            self.assertEqual(ctx.exception.allowed, expected)

    def test_capabilities_carry_both_vocabularies_and_tolerate_their_absence(self):
        caps = capabilities({**deepcopy(FALLBACK),
                             'features': ['query', 'asmr'],
                             'filters': {**deepcopy(FALLBACK['filters']),
                                         'category': [{'value': 'bedtime_story',
                                                       'label': '睡前故事'}]}})
        self.assertEqual([x['value'] for x in caps['filters']['category']],
                         ['bedtime_story'])
        # A provider with no ASMR simply omits both, and keeps the built-in
        # fallback — the same tolerance `genre` already had.
        without = {k: v for k, v in FALLBACK['filters'].items()
                   if k not in ('kind', 'category')}
        caps = capabilities({**deepcopy(FALLBACK), 'filters': without})
        self.assertEqual([x['value'] for x in caps['filters']['kind']], ['song', 'asmr'])

    def test_kind_is_never_relaxed_so_asmr_cannot_come_back_as_a_song(self):
        """Asking for ASMR and getting a song is not a loosened filter, it is
        the wrong content — so there is no `relaxed` value that excuses it."""
        request = search_request(kind='asmr')
        self.assertEqual(search_result({'schema': SCHEMA, 'tracks': [asmr()],
                                        'relaxed': []}, request)['tracks'][0]['kind'], 'asmr')
        for relaxed in ([], ['query'], ['category']):
            with self.subTest(relaxed=relaxed), self.assertRaises(MusicError):
                search_result({'schema': SCHEMA, 'tracks': [track()],
                               'relaxed': relaxed}, request)

    def test_a_missing_kind_reads_as_song(self):
        """A provider predating the split labels nothing, and everything it has
        is a song. That must satisfy a song request and fail an ASMR one."""
        bare = track()
        self.assertNotIn('kind', bare)
        search_result({'schema': SCHEMA, 'tracks': [bare], 'relaxed': []},
                      search_request(kind='song'))
        with self.assertRaises(MusicError):
            search_result({'schema': SCHEMA, 'tracks': [bare], 'relaxed': []},
                          search_request(kind='asmr'))

    def test_category_is_hard_unless_the_catalogue_says_it_relaxed_it(self):
        request = search_request(kind='asmr', category='trigger')
        wrong = {'schema': SCHEMA, 'tracks': [asmr(category='chat')], 'relaxed': []}
        with self.assertRaises(MusicError):
            search_result(wrong, request)
        self.assertEqual(search_result({**wrong, 'relaxed': ['category']},
                                       request)['relaxed'], ['category'])

    def test_a_trigger_entry_without_a_script_is_valid(self):
        """Trigger sounds have no voice and no transcript. An empty
        `lyrics_excerpt` and a null `vocal` are the correct shape, not missing
        data to reject."""
        result = search_result(
            {'schema': SCHEMA, 'relaxed': [],
             'tracks': [asmr(category='trigger', title='咬/咀嚼·薯片',
                             duration_s=22, lyrics_excerpt='')]},
            search_request(kind='asmr', category='trigger'))
        self.assertEqual(result['tracks'][0]['lyrics_excerpt'], '')
        self.assertIsNone(result['tracks'][0]['vocal'])


class ContractTests(unittest.TestCase):
    def test_discovery_does_not_connect(self):
        self.assertEqual(set(discover()), {'mock', 'motus_music'})

    def test_query_filters_and_unknown_tags(self):
        req = search_request('下雨天', genre='pop,folk', vocal='female', tags='新标签,慵懒',
                             language='zh', duration_min_s=60, duration_max_s=300,
                             exclude_ids='a,b,a')
        self.assertEqual(req['filters']['genre'], ['pop', 'folk'])
        self.assertEqual(req['filters']['tags'], ['新标签', '慵懒'])
        self.assertEqual(req['exclude_ids'], ['a', 'b'])
        self.assertEqual(req['filters']['duration_s'], {'min': 60, 'max': 300})

    def test_invalid_inputs(self):
        for kwargs in ({'query': 'x' * 201}, {'top_k': 11}, {'top_k': True},
                       {'genre': 'invented'}, {'duration_min_s': 20, 'duration_max_s': 1},
                       {'exclude_ids': ','.join(str(i) for i in range(51))}):
            with self.subTest(kwargs=kwargs), self.assertRaises(MusicError):
                search_request(**kwargs)

    def test_capabilities_supply_new_codes(self):
        caps = deepcopy(FALLBACK)
        caps['filters']['genre'] = [{'value': 'provider-genre'}]
        caps['limits']['max_top_k'] = 2
        self.assertEqual(search_request(genre='provider-genre', top_k=2, caps=capabilities(caps))['top_k'], 2)
        with self.assertRaises(MusicError):
            search_request(top_k=3, caps=caps)

    def test_relaxation_is_preserved_and_language_never_relaxed(self):
        request = search_request(genre='jazz', language='zh')
        data = {'schema': SCHEMA, 'tracks': [track()], 'relaxed': ['genre']}
        self.assertEqual(search_result(data, request)['relaxed'], ['genre'])
        for bad in ({**data, 'relaxed': []}, {**data, 'tracks': [track(language='en')]},
                    {**data, 'relaxed': ['language']}):
            with self.assertRaises(MusicError):
                search_result(bad, request)

    def test_excluded_song_and_unsafe_audio_rejected(self):
        with self.assertRaises(MusicError):
            search_result({'schema': SCHEMA, 'tracks': [track()]}, search_request(exclude_ids='fictional-1'))
        with self.assertRaises(MusicError):
            search_result({'schema': SCHEMA, 'tracks': [track(audio={'url': 'http://audio.example/a'})]}, search_request())

    def test_unknown_vocal_is_preserved_but_does_not_satisfy_a_filter(self):
        data = {'schema': SCHEMA, 'tracks': [track(vocal=None)], 'relaxed': []}
        self.assertIsNone(search_result(data, search_request())['tracks'][0]['vocal'])
        for voice in ('female', 'male', 'mixed', 'instrumental'):
            with self.subTest(voice=voice), self.assertRaises(MusicError):
                search_result(data, search_request(vocal=voice))
        relaxed = {**data, 'relaxed': ['vocal']}
        self.assertIsNone(search_result(relaxed, search_request(vocal='female'))['tracks'][0]['vocal'])

    def test_unknown_vocal_does_not_expand_request_values_or_hide_malformed_data(self):
        for voice in (None, 'unknown', 'null'):
            with self.subTest(voice=voice), self.assertRaises(MusicError):
                search_request(vocal=voice)
        missing = track()
        del missing['vocal']
        for invalid in (track(vocal='unknown'), track(vocal=[]), missing):
            with self.assertRaises(MusicError):
                search_result({'schema': SCHEMA, 'tracks': [invalid]}, search_request())

    def test_malformed_containers_return_contract_errors(self):
        for field in ('artist', 'audio'):
            with self.subTest(field=field), self.assertRaises(MusicError):
                search_result({'schema': SCHEMA, 'tracks': [track(**{field: []})]}, search_request())
        with self.assertRaises(MusicError):
            capabilities({**FALLBACK, 'limits': []})

    def test_invalid_provider_config_and_retry_header(self):
        for value in ('bad', None, True, -10):
            with self.assertRaises(MusicError):
                MotusMusicProvider({'endpoint': 'https://catalog.example', 'api_key': 'fake', 'timeout_ms': value})
        for value in ('nan', 'inf', 'bad'):
            self.assertEqual(retry_seconds(value), 1)



class Response(io.BytesIO):
    def __init__(self, data, code=200, headers=None):
        super().__init__(data if isinstance(data, bytes) else json.dumps(data).encode())
        self.code, self.headers = code, headers or {}


class ProviderTests(unittest.TestCase):
    def provider(self):
        return MotusMusicProvider({'endpoint': 'https://catalog.example/open', 'api_key': 'test-placeholder'})

    def test_malformed_header_credential_never_reaches_http_or_error_text(self):
        for key in ('fictional\nkey', 'fictional\x00key', 'fictional 密钥'):
            with self.subTest(), self.assertRaises(MusicError) as ctx:
                MotusMusicProvider({'endpoint': 'https://catalog.example', 'api_key': key})
            self.assertNotIn('fictional', str(ctx.exception))

    def test_authorization_only_in_header_and_detail_id_encoded(self):
        opener = Mock()
        opener.open.return_value = Response({'schema': SCHEMA, 'track': track()})
        with patch('urllib.request.build_opener', return_value=opener):
            self.provider().track('a/b?c')
        req = opener.open.call_args.args[0]
        self.assertEqual(req.full_url, 'https://catalog.example/open/v1/music/tracks/a%2Fb%3Fc')
        self.assertEqual(req.get_header('Authorization'), 'Bearer test-placeholder')
        self.assertNotIn('test-placeholder', req.full_url)
        with self.assertRaises(MusicError):
            _NoRedirect().redirect_request(None, None, 302, '', {}, 'https://other.example')

    def test_error_mapping_does_not_echo_remote_secrets_or_retry_4xx(self):
        for code, expected in [(400, 'schema_mismatch'), (401, 'unauthorized'), (403, 'forbidden'),
                               (404, 'not_found'), (422, 'invalid_filter'), (302, 'http_error')]:
            opener = Mock()
            opener.open.return_value = Response({'schema': SCHEMA, 'request_id': 'test-placeholder',
                'error': {'message': 'test-placeholder', 'allowed': ['test-placeholder']}}, code)
            with self.subTest(code=code), patch('urllib.request.build_opener', return_value=opener):
                with self.assertRaises(MusicError) as ctx:
                    self.provider().capabilities()
                self.assertEqual(ctx.exception.code, expected)
                self.assertNotIn('test-placeholder', json.dumps(ctx.exception.result()))
                opener.open.assert_called_once()

    def test_retry_quota_cooldown_and_cancel(self):
        provider, opener = self.provider(), Mock()
        opener.open.return_value = Response({'schema': SCHEMA}, 429, {'Retry-After': '60'})
        with patch('urllib.request.build_opener', return_value=opener):
            for _ in range(2):
                with self.assertRaises(MusicError) as ctx:
                    provider.capabilities()
                self.assertEqual(ctx.exception.code, 'rate_limited')
            opener.open.assert_called_once()
        opener.open.side_effect = [Response({}, 503), Response({'schema': SCHEMA})]
        with patch('urllib.request.build_opener', return_value=opener), patch('time.sleep'):
            self.assertEqual(self.provider().capabilities()['schema'], SCHEMA)
        cancel = threading.Event(); cancel.set()
        with patch('urllib.request.build_opener') as factory, self.assertRaises(MusicError) as ctx:
            self.provider().capabilities(cancel=cancel)
        self.assertEqual(ctx.exception.code, 'cancelled')
        factory.assert_not_called()

    def test_timeout_oversize_bad_version_and_http_error_close(self):
        for response in (Response(b'x' * (1024 * 1024 + 1)), Response({'schema': 'motus.music/2'})):
            with patch('urllib.request.build_opener') as factory:
                factory.return_value.open.return_value = response
                with self.assertRaises(MusicError):
                    self.provider().capabilities()
            self.assertTrue(response.closed)
        with patch('urllib.request.build_opener') as factory:
            factory.return_value.open.side_effect = TimeoutError('private-url-or-key')
            with self.assertRaises(MusicError) as ctx:
                self.provider().capabilities()
            self.assertNotIn('private-url-or-key', str(ctx.exception))
        body = Response({'schema': SCHEMA})
        with patch('urllib.request.build_opener') as factory:
            factory.return_value.open.side_effect = urllib.error.HTTPError('https://catalog.example', 401, '', {}, body)
            with self.assertRaises(MusicError): self.provider().capabilities()
        self.assertTrue(body.closed)


class ServiceTests(unittest.TestCase):
    def test_none_mock_filter_exclude_and_detail(self):
        self.assertEqual(MusicService({}).execute()['error']['code'], 'not_configured')
        service = MusicService({'catalogue_type': 'mock'})
        result = service.execute(language='zh', vocal='female')
        self.assertTrue(result['tracks'])
        ids = ','.join(t['id'] for t in result['tracks'])
        following = service.execute(language='zh', vocal='female', exclude_ids=ids)
        self.assertFalse(set(ids.split(',')) & {t['id'] for t in following['tracks']})
        self.assertIsNone(service.execute(track_id=result['tracks'][0]['id'])['track']['audio'])
        self.assertEqual(service.execute(track_id='missing')['error']['code'], 'not_found')

    def test_cache_and_transient_fallback_do_not_hide_auth_or_version_error(self):
        service = MusicService({'catalogue_type': 'mock'})
        service.execute()
        with patch.object(service._provider, 'capabilities') as caps:
            service.execute(); caps.assert_not_called()
        for code, blocks in [('unavailable', False), ('unauthorized', True), ('schema_mismatch', True)]:
            service._expires = 0
            with patch.object(service._provider, 'capabilities', side_effect=MusicError(code, 'test')):
                result = service.execute()
                self.assertEqual('error' in result, blocks)

    def test_credentials_come_only_from_card_and_unknown_voice_preserved(self):
        with patch.dict('os.environ', {'PHANTHY_MUSIC_ENDPOINT': 'https://ignored.example',
                                       'PHANTHY_MUSIC_API_KEY': 'fake-ignored-key'}):
            service = MusicService({'catalogue_type': 'motus_music'})
            self.assertFalse(service.configured())
            self.assertEqual(service.execute()['error']['code'], 'not_configured')
            service = MusicService({'catalogue_type': 'motus_music',
                                    'endpoint': 'https://catalogue.example', 'api_key': 'fake-card-key'})
            self.assertTrue(service.configured())
            self.assertEqual(service._provider.endpoint, 'https://catalogue.example')
            self.assertEqual(service._provider._key, 'fake-card-key')
        service = MusicService({'catalogue_type': 'mock'})
        service.execute()
        with patch.object(service._provider, 'track', return_value={'schema': SCHEMA, 'track': track(vocal=None)}):
            result = service.execute(track_id='fictional-1')
        self.assertIsNone(result['track']['vocal'])
        self.assertIn('人声未知', result['reply_notice'])

    def test_asmr_lookups_offer_the_vocabulary_that_applies_to_them(self):
        """An ASMR reply must not carry a genre list. Offering one is how a
        model learns to put `genre` on the next ASMR call — which is a 422."""
        service = MusicService({'catalogue_type': 'mock'})
        songs = service.execute(genre='pop')
        self.assertIn('available_genres', songs)
        self.assertNotIn('available_categories', songs)
        entries = service.execute(kind='asmr')
        self.assertTrue(entries['tracks'])
        self.assertTrue(all(t['kind'] == 'asmr' for t in entries['tracks']))
        self.assertIn('bedtime_story', entries['available_categories'])
        self.assertNotIn('available_genres', entries)
        self.assertIn('ASMR 没有歌手和曲风', entries['reply_notice'])

    def test_an_asmr_detail_lookup_is_recognised_from_the_result(self):
        """`previous` refreshes by id and sends no filters at all, so the kind
        can only be read off what came back."""
        service = MusicService({'catalogue_type': 'mock'})
        found = service.execute(kind='asmr', category='bedtime_story')
        detail = service.execute(track_id=found['tracks'][0]['id'])
        self.assertEqual(detail['track']['kind'], 'asmr')
        self.assertIn('available_categories', detail)
        self.assertNotIn('available_genres', detail)

    def test_vocabulary_publishes_the_asmr_categories(self):
        """Before this, the only way to learn a category code was to send a
        wrong one and read `allowed` off the 422."""
        vocabulary = MusicService({'catalogue_type': 'mock'}).vocabulary()
        self.assertEqual([x['value'] for x in vocabulary['kinds']], ['song', 'asmr'])
        self.assertIn('bedtime_story',
                      [x['value'] for x in vocabulary['asmr_categories']])
        self.assertTrue(vocabulary['asmr_supported'])
        self.assertEqual(vocabulary['source'], 'catalogue')

    def test_a_filter_from_the_wrong_kind_never_reaches_the_network(self):
        service = MusicService({'catalogue_type': 'mock'})
        service.execute()  # warm the capability cache
        with patch.object(service._provider, 'search') as search:
            self.assertEqual(service.execute(kind='asmr', genre='pop')['error']['code'],
                             'invalid_filter')
        search.assert_not_called()

    def test_concurrent_catalogue_call_is_bounded_not_queued(self):
        service = MusicService({'catalogue_type': 'mock'})
        service._gate.acquire()
        try:
            self.assertEqual(service.execute()['error']['code'], 'busy')
        finally:
            service._gate.release()
