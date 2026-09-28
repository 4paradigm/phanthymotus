"""Music tool/provider contracts: offline, no real catalogue or credentials."""
import asyncio
import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('DB_PATH', str(Path(tempfile.mkdtemp()) / 'music-test.db'))

from music.contracts import FALLBACK, MusicError, SCHEMA, capabilities, search_request, search_result
from music.providers import discover
from music.providers.motus_music import MotusMusicProvider, retry_seconds
from music.service import MusicService
from music.settings import save_settings
import config


class Response:
    def __init__(self, status, data, headers=None):
        self.status, self.headers = status, headers or {}
        self.raw = json.dumps(data).encode() if not isinstance(data, bytes) else data
        self.content = self

    async def iter_chunked(self, size):
        for start in range(0, len(self.raw), size):
            yield self.raw[start:start + size]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class Session(Response):
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


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


class ConfigExportTests(unittest.TestCase):
    def test_solution_pack_clears_music_endpoint_and_key(self):
        from api import canvas, solutions
        # Exercise the real packer using the shipped decision-core schema,
        # without starting the application/DDS or external clients.
        schemas = []
        for node in ast.walk(ast.parse((ROOT / 'src/start.py').read_text())):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value == 'configSchema':
                        try:
                            schemas.append(ast.literal_eval(value))
                        except (ValueError, SyntaxError):
                            pass
        schema = next(s for s in schemas if 'music_type' in s.get('properties', {}))
        tool = {'name': 'decision_core', 'configSchema': schema}
        saved = {'agentcore:decision_core': {'music_type': 'motus_music',
                  'music_endpoint': 'https://catalog.example/private-deployment',
                  'music_api_key': 'fake-secret-for-export-test', 'music_timeout_ms': 3000}}
        with patch.object(solutions, '_mcp_list', return_value=[{'id': 'agentcore', 'tools': [tool]}]), \
             patch.object(solutions, '_layout', return_value={}), \
             patch.object(canvas, 'all_tool_configs', return_value=saved):
            packed, redacted = solutions._pack_canvas({'agentcore': 'd0'}, set())
        exported = packed['toolConfigs']['d0:decision_core']
        self.assertEqual(exported['music_endpoint'], '')
        self.assertEqual(exported['music_api_key'], '')
        self.assertEqual(exported['music_type'], 'motus_music')
        self.assertEqual(exported['music_timeout_ms'], 3000)
        self.assertEqual(len(redacted), 2)
        self.assertNotIn('fake-secret', json.dumps(packed))


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    def provider(self):
        return MotusMusicProvider({'endpoint': 'https://catalog.example/open', 'api_key': 'test-placeholder'})

    async def test_bearer_header_path_and_no_redirects(self):
        session = Session([Response(200, FALLBACK)])
        with patch('aiohttp.ClientSession', return_value=session) as constructor:
            await self.provider().capabilities()
        self.assertEqual(constructor.call_args.kwargs['headers']['Authorization'], 'Bearer test-placeholder')
        args, kwargs = session.calls[0]
        self.assertEqual(args, ('GET', 'https://catalog.example/open/v1/capabilities'))
        self.assertFalse(kwargs['allow_redirects'])

    async def test_errors_not_retried_and_credentials_not_returned(self):
        for status in (400, 401, 403, 404, 422, 302):
            session = Session([Response(status, {'request_id': 'req-test', 'error': {'message': 'test-placeholder'}})])
            with patch('aiohttp.ClientSession', return_value=session), self.assertRaises(MusicError) as raised:
                await self.provider().search(search_request())
            self.assertEqual(len(session.calls), 1)
            self.assertEqual(raised.exception.request_id, 'req-test')
            self.assertNotIn('test-placeholder', str(raised.exception))

    async def test_429_respects_retry_after_and_cooldown(self):
        session = Session([Response(429, {'request_id': 'limited'}, {'Retry-After': '60'})])
        provider = self.provider()
        with patch('aiohttp.ClientSession', return_value=session):
            for _ in range(2):
                with self.assertRaises(MusicError) as raised:
                    await provider.capabilities()
                self.assertEqual(raised.exception.code, 'rate_limited')
                self.assertGreater(raised.exception.retry_after, 59)
        self.assertEqual(len(session.calls), 1)

    async def test_transient_retry_and_deadline(self):
        session = Session([Response(503, {}), Response(200, FALLBACK)])
        with patch('aiohttp.ClientSession', return_value=session), patch('asyncio.sleep', new_callable=AsyncMock) as sleep:
            self.assertEqual((await self.provider().capabilities())['schema'], SCHEMA)
        sleep.assert_awaited_once_with(0.2)
        self.assertLessEqual(session.calls[-1][1]['timeout'].total, 3)

    async def test_timeout_and_bad_responses(self):
        for response, code in ((asyncio.TimeoutError(), 'unavailable'),
                               (Response(200, {'schema': 'motus.music/2'}), 'schema_mismatch'),
                               (Response(200, b'<html>'), 'schema_mismatch'),
                               (Response(200, b'x' * (1024 * 1024 + 1)), 'invalid_response')):
            with patch('aiohttp.ClientSession', return_value=Session([response])), self.assertRaises(MusicError) as raised:
                await self.provider().capabilities()
            self.assertEqual(raised.exception.code, code)

    async def test_track_id_is_path_escaped(self):
        session = Session([Response(200, {'schema': SCHEMA, 'track': track()})])
        with patch('aiohttp.ClientSession', return_value=session):
            await self.provider().track('a/b?c')
        self.assertTrue(session.calls[0][0][1].endswith('a%2Fb%3Fc'))


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.settings = config.main.get('desktop_tools', {})
        config.main['desktop_tools'] = {**self.settings, 'music': {'type': 'mock'}}
        self.service = MusicService()

    async def asyncTearDown(self):
        config.main['desktop_tools'] = self.settings

    async def test_mock_search_next_and_detail(self):
        result = await self.service.execute(language='zh', vocal='female')
        self.assertEqual(len(result['tracks']), 3)
        self.assertIsNone(result['tracks'][0]['audio'])
        song = result['tracks'][0]['id']
        next_result = await self.service.execute(exclude_ids=song)
        self.assertNotIn(song, [t['id'] for t in next_result['tracks']])
        self.assertEqual((await self.service.execute(track_id=song))['track']['id'], song)
        self.assertEqual((await self.service.execute(track_id='absent'))['error']['code'], 'not_found')

    async def test_capabilities_cached_and_config_change_invalidates(self):
        provider, _ = await self.service.prepare()
        with patch.object(provider, 'capabilities', new_callable=AsyncMock) as fetch:
            await self.service.execute()
            await self.service.execute()
            fetch.assert_not_awaited()
        config.main['desktop_tools'] = {**self.settings, 'music': {'type': 'none'}}
        self.assertEqual((await self.service.execute())['error']['code'], 'not_configured')

    async def test_capability_outage_fallback_but_version_failure_is_visible(self):
        provider, _ = await self.service.prepare()
        self.service._expires = 0
        with patch.object(provider, 'capabilities', side_effect=MusicError('unavailable', 'offline')):
            _, caps = await self.service.prepare()
            self.assertEqual(caps['filters']['tags'], {})
        self.service._expires = 0
        with patch.object(provider, 'capabilities', side_effect=MusicError('schema_mismatch', 'wrong version')):
            self.assertEqual((await self.service.execute())['error']['code'], 'schema_mismatch')

    async def test_real_desktop_entry(self):
        # Import the real entry without booting the event package's LLM clients.
        spec = importlib.util.spec_from_file_location('music_desktop_test', ROOT / 'src/event/desktop.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch('music.service.service', self.service):
            result = json.loads(await module.DesktopTools().MusicSearch(query='睡前', top_k=2))
        self.assertEqual(len(result['tracks']), 2)
        self.assertIn('不得', result['usage_notice'])

    async def test_malformed_detail_is_visible_not_an_uncaught_exception(self):
        provider, _ = await self.service.prepare()
        with patch.object(provider, 'track', return_value={'schema': SCHEMA, 'track': []}):
            result = await self.service.execute(track_id='bad')
        self.assertEqual(result['error']['code'], 'invalid_response')

    async def test_settings_save_preserves_mask_and_supports_clear(self):
        await asyncio.to_thread(save_settings, {'music_type': 'motus_music',
            'music_endpoint': 'https://catalog.example/open', 'music_api_key': 'test-placeholder'})
        await asyncio.to_thread(save_settings, {'music_type': 'motus_music', 'music_api_key': '****'})
        self.assertEqual(config.main.get('desktop_tools')['music']['api_key'], 'test-placeholder')
        await asyncio.to_thread(save_settings, {'music_type': 'none', 'music_api_key': ''})
        self.assertEqual(config.main.get('desktop_tools')['music']['api_key'], '')


if __name__ == '__main__':
    unittest.main()
