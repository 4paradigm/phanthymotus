"""Offline motus.music/1 contracts and bounded HTTP provider; no real credentials."""
import io
import json
import threading
import time
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch
import urllib.error

from plugins.music_catalog.contracts import FALLBACK, MusicError, SCHEMA, capabilities, search_request, search_result
from plugins.music_catalog.providers import discover
from plugins.music_catalog.providers.motus_music import MotusMusicProvider, retry_seconds, _NoRedirect
from plugins.music_catalog.service import MusicService

def track(**overrides):
    return {'id': 'fictional-1', 'title': 'Fictional', 'language': 'zh', 'genre': 'pop',
            'vocal': 'female', 'artist': {'id': 'artist-1', 'name': 'Fictional artist'},
            'duration_s': 120, 'audio': {'url': 'https://audio.example/test.mp3'}, **overrides}


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

    def test_concurrent_catalogue_call_is_bounded_not_queued(self):
        service = MusicService({'catalogue_type': 'mock'})
        service._gate.acquire()
        try:
            self.assertEqual(service.execute()['error']['code'], 'busy')
        finally:
            service._gate.release()
