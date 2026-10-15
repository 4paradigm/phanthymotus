"""Offline media, mixer and MCP lifecycle tests. No GPU, ROS or catalogue needed."""
import io
import json
import queue
import socket
import subprocess
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit
import wave

from vision_stubs import _FakeExecutor, _FakeAudioChunk
import numpy as np

from plugins.phanthy_music import PhanthyMusicPlugin, TOOLS, OUTPUT_TOPIC, FORMAT
from plugins.music_decoder import (Decoder, AudioAddressError, _public_connection,
                                  _public_dns_addresses, validate_url, _HTTPSRedirect,
                                  _DNSNoRedirect, _PublicHTTPSConnection)
from plugins.music_player import MusicPlayer, AUDIO_EOF, FRAME_BYTES, MAX_VOICE_BYTES


def pcm(value, samples=1600):
    return np.full(samples, value, dtype='<i2').tobytes()


def ffmpeg_path(test):
    import shutil
    path = shutil.which('ffmpeg')
    if not path:
        try:
            import imageio_ffmpeg
            path = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            test.skipTest('ffmpeg not available; run in the Perception image')
    return path


def wav_bytes(samples):
    source = io.BytesIO()
    with wave.open(source, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples)
    return source.getvalue()


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeDecoder:
    def __init__(self, *args):
        self.frames = queue.Queue(maxsize=20)
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.error = ''

    def start(self):
        pass

    def cancel(self):
        self.cancelled.set()
        self.done.set()

    def close(self):
        self.cancel()

    def is_closed(self):
        return self.done.is_set()


class SessionTests(unittest.TestCase):
    """One music session: decoder lifecycle, transport states, rapid-skip bound.

    The speech/duck half of these moved to test_audio_mixer.py along with the
    mixing itself — this card no longer takes speech in.
    """

    def setUp(self):
        self.output = []
        self.clock = FakeClock()
        self.player = MusicPlayer(self.output.append, decoder_factory=FakeDecoder,
                                  volume=100, clock=self.clock)

    def tearDown(self):
        self.player.close()

    def step(self, music=None):
        if music is not None:
            self.player._decoder.frames.put(music)
        self.player.tick()
        self.clock.now += 0.1

    def play(self):
        return self.player.play('https://audio.example/fictional.mp3')

    def test_background_session_and_terminal_state(self):
        started = self.play()
        self.assertEqual(started['playback_state'], 'buffering')
        self.assertIn('playback_id', started)
        self.assertNotIn('action_id', started)
        self.step(pcm(12000))
        self.assertEqual(self.player.state, 'playing')
        self.player._decoder.done.set()
        self.step()
        self.assertEqual(self.player.state, 'completed')
        self.assertEqual(self.output[-1], AUDIO_EOF)

    def test_pause_preserves_music_buffer_but_speech_continues(self):
        self.play()
        self.player._decoder.frames.put(pcm(10000))
        self.player.pause()
        self.step()
        # Pause must not drain what was already decoded: resuming has to
        # continue the song, not skip the buffered 100 ms.
        self.assertEqual(self.player._decoder.frames.qsize(), 1)
        self.player.resume()
        self.step()
        self.assertEqual(self.player._decoder.frames.qsize(), 0)

    def test_interrupt_music_preserves_tts_and_replacement_discards_old_frames(self):
        first = self.play()['playback_id']
        old = self.player._decoder
        old.frames.put(pcm(25000))
        self.player.interrupt()
        self.assertTrue(old.cancelled.is_set())
        self.assertNotEqual(self.play()['playback_id'], first)
        self.assertTrue(self.player._decoder.frames.empty())

    def test_finished_decoder_with_pending_reader_still_limits_rapid_skip(self):
        for _ in range(2):
            self.play()
            decoder = self.player._decoder
            decoder.is_closed = lambda: False
            decoder.error = 'Download failed'
            decoder.done.set()
            self.step()
        with self.assertRaisesRegex(ValueError, 'still closing'):
            self.play()
        self.assertEqual(len(self.player._retired), 2)

    def test_no_reader_and_stalled_stream_fail_visibly(self):
        self.play()
        self.player._has_reader = lambda: False
        self.clock.now += 6
        self.step()
        self.assertEqual(self.player.state, 'error')
        self.assertEqual(self.output, [])
        self.player._has_reader = lambda: True
        self.play()
        self.clock.now += 16
        self.step()
        self.assertEqual(self.player.error, 'Audio stream stalled')

    def test_decode_error_remains_error(self):
        self.play()
        self.player._decoder.error = 'broken media'
        self.player._decoder.done.set()
        self.step()
        self.assertEqual(self.player.state, 'error')
        self.assertEqual(self.player.status()['error'], 'broken media')

    def test_invalid_expiry_usage_and_volume(self):
        for kwargs in ({'expires_at': '2000-01-01T00:00:00Z'}, {'expires_at': 'bad'},
                       {'usage': {'scope': 'redistribute'}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.player.play('https://audio.example/a.mp3', **kwargs)
        for volume in (-1, 101, float('nan'), True):
            with self.assertRaises(ValueError):
                self.player.set_volume(volume)


class PluginTests(unittest.TestCase):
    def setUp(self):
        self.executor = _FakeExecutor()
        self.plugin = PhanthyMusicPlugin({}, self.executor)

    def tearDown(self):
        self.plugin.stop()

    def test_schema_declares_an_actuator_that_only_emits_music(self):
        import jsonschema
        tool = self.plugin.get_tools()[0]
        self.assertEqual(tool['name'], 'phanthy_music')
        # An actuator with an output topic and no input: mixing is audio_mixer's
        # job now, so this card has nothing to subscribe to.
        self.assertEqual(tool['type'], 'actuator')
        self.assertEqual(tool['topic_in'], [])
        self.assertEqual(tool['topic_out'][0]['topic'], OUTPUT_TOPIC)
        schema = tool['inputSchema']
        # Its own channel, not the mouth: music and speech are summed downstream
        # and must not make each other wait.
        self.assertEqual(schema['x-resource'], 'music')
        self.assertEqual(set(schema['x-action-params']),
                         {'list_genres', 'play', 'next', 'previous', 'pause',
                          'resume', 'stop_music', 'now_playing', 'set_volume',
                          'set_autoplay', 'search'})
        jsonschema.validate({'action': 'play', 'genre': 'pop'}, schema)
        jsonschema.validate({'action': 'next'}, schema)
        jsonschema.validate({'action': 'list_genres'}, schema)

    def test_play_only_takes_what_a_person_actually_says(self):
        """Pins the param set against drift back toward the REST shape.

        Every one of these has to be reachable from a spoken sentence. The
        catalogue can filter on more than this — duration is genuinely indexed,
        measured on the live service — but nobody utters a duration bound, and
        a filter that exists only because SearchRequest has it is a field the
        model can get wrong for no benefit.
        """
        params = self.plugin.get_tools()[0]['inputSchema']['x-action-params']
        self.assertEqual(set(params['play']['params']),
                         {'query', 'genre', 'language', 'artist', 'tags'})
        # `tags` in particular: the catalogue's scene vocabulary is
        # 驾驶/通勤/工作/夜晚, so "放点适合开车的歌" has somewhere to land. query
        # cannot substitute for it — query is relaxable, tags are not.
        self.assertIn('tags', params['play']['params'])

    def test_search_keeps_the_precision_filters_play_dropped(self):
        """Nothing is lost, it just moves: search stays the precise path."""
        params = self.plugin.get_tools()[0]['inputSchema']['x-action-params']
        self.assertIn('vocal', params['search']['params'])
        self.assertIn('top_k', params['search']['params'])

    def test_the_dropped_filters_are_gone_from_the_schema_entirely(self):
        """Declared nowhere means the model never sees them, not even as a
        property it could pass to the wrong action.
        """
        schema = self.plugin.get_tools()[0]['inputSchema']
        for gone in ('duration_min_s', 'duration_max_s', 'track_id'):
            self.assertNotIn(gone, schema['properties'])
            for action in schema['x-action-params'].values():
                self.assertNotIn(gone, action['params'])

    def test_playing_a_specific_track_still_works_for_stored_callers(self):
        """`track_id` left the declared surface, not the dispatch path — the
        play_by_id alias and any saved canvas call keep working.
        """
        self.plugin.dispatch('phanthy_music', {'action': 'start'})
        reply = self.plugin.dispatch('phanthy_music',
                                     {'action': 'play_by_id', 'track_id': 'fictional-1'})
        # mock catalogue has no audio, so it reaches the player and is refused
        # there — the point is that it got that far rather than being ignored.
        self.assertIn('error', reply)
        self.assertIn(reply['error']['code'], ('not_playable', 'not_found'))

    def test_the_card_has_nothing_to_configure(self):
        """四项设置没有一项经得起「谁比默认值更有资格回答」的追问：连接内置在卡里，
        超时是实现细节，音量和自动续播是听的人当场说的（set_volume/set_autoplay）。
        """
        self.assertNotIn('configSchema', self.plugin.get_tools()[0])

    def test_autoplay_is_off_until_asked_for(self):
        """放一首就停。自己接着往下放是个决定，应该被明确要求。"""
        self.plugin.dispatch('phanthy_music', {'action': 'start'})
        self.assertFalse(self.plugin.dispatch(
            'phanthy_music', {'action': 'now_playing'})['autoplay'])
        self.plugin.dispatch('phanthy_music',
                             {'action': 'set_autoplay', 'autoplay': True})
        self.assertTrue(self.plugin.dispatch(
            'phanthy_music', {'action': 'now_playing'})['autoplay'])

    def test_autoplay_refuses_a_value_it_would_have_to_guess_at(self):
        """猜错的后果是歌自己一直放下去，比报错糟。"""
        self.plugin.dispatch('phanthy_music', {'action': 'start'})
        for bad in ('true', 1, 'on', None):
            reply = self.plugin.dispatch('phanthy_music',
                                         {'action': 'set_autoplay', 'autoplay': bad})
            self.assertEqual(reply['error']['code'], 'invalid_argument')

    def test_the_system_actions_are_not_offered_to_the_model(self):
        """`present_to_llm` only filters them for `processor` cards, and this is
        an `actuator` — so leaving them in the enum would hand the model a
        `stop` that tears the ROS node down, next to the `stop_music` it means.
        Exactly the confusion measured on Orin5 with `tts(action="stop")`.
        """
        enum = self.plugin.get_tools()[0]['inputSchema']['properties']['action']['enum']
        for hidden in ('start', 'stop', 'info', 'config'):
            self.assertNotIn(hidden, enum)
        # Still dispatchable — the canvas calls them directly and never reads
        # the enum.
        self.assertEqual(self.plugin.dispatch(
            'phanthy_music', {'action': 'start'})['state'], 'running')

    def test_the_session_is_declared_background_and_silent(self):
        """A four-minute track must not block `finish`, the agent's next call,
        or suppress anything — and its end is a receipt, not a message.
        """
        completion = self.plugin.get_tools()[0]['inputSchema']['x-completion']
        self.assertEqual(completion['actions'], ['play'])
        self.assertTrue(completion['background'])
        self.assertTrue(completion['silent_completion'])
        # 0 = no deadline: a track has no `text` to estimate from, and a static
        # ceiling would reap the pending mid-playback.
        self.assertEqual(completion['timeout'], 0)

    def test_barge_in_must_not_stop_the_music(self):
        """on_interrupt_all fires on every new user turn. Binding it here would
        mean the song dies whenever somebody speaks.
        """
        hooks = self.plugin.get_tools()[0]['inputSchema']['x-hooks']
        self.assertEqual(set(hooks), {'on_interrupt_music'})
        self.assertEqual(hooks['on_interrupt_music']['action'], 'stop_music')

    def test_start_stop_and_restart_have_one_node_and_no_autoplay(self):
        for _ in range(3):
            result = self.plugin.dispatch('phanthy_music', {'action': 'start'})
            self.assertEqual(result['state'], 'running')
            self.assertEqual(result['playback_state'], 'idle')
            self.assertEqual(result['topic_in'], [])
            self.assertEqual(result['topic_out'][0]['topic'], OUTPUT_TOPIC)
            self.assertEqual(len(self.executor.nodes), 1)
        node = self.plugin._node
        self.plugin.dispatch('phanthy_music', {'action': 'stop'})
        self.assertTrue(node.destroyed)
        self.assertEqual(len(self.executor.nodes), 0)
        self.assertFalse(node.player._thread.is_alive())
        self.plugin.dispatch('phanthy_music', {'action': 'start'})
        self.assertEqual(len(self.executor.nodes), 1)

    def test_music_pcm_reaches_the_output_topic(self):
        self.plugin.dispatch('phanthy_music', {'action': 'start'})
        node = self.plugin._node
        node.player._factory = FakeDecoder
        node.player.play('https://audio.example/fictional.mp3')
        node.player._decoder.frames.put(pcm(1000))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not node.publisher.messages:
            time.sleep(.01)
        self.assertTrue(node.publisher.messages)
        # Volume defaults to 50 and the gain ramps in from silence, so the first
        # frame is attenuated — what matters is that music, and only music,
        # reaches the topic.
        self.assertTrue(any(bytes(m) for m in node.publisher.messages))

    def test_legacy_mix_wiring_fails_loudly_instead_of_going_mute(self):
        """An old canvas has `TTS → phanthy_music → Speaker`. Ignoring that
        input would leave TTS publishing into nothing — a mute robot reporting
        已就绪, which `_dropped_inputs` cannot catch at a single topic.
        """
        for args in ({'action': 'start', 'input_topic': '/speech/tts'},
                     {'action': 'start', 'input_topics': ['/speech/tts']}):
            result = self.plugin.dispatch('phanthy_music', args)
            self.assertEqual(result['error']['code'], 'legacy_wiring')
            self.assertIn('audio_mixer', result['error']['message'])
            self.assertEqual(self.executor.nodes, [])

    def test_an_interrupt_hook_while_idle_is_not_an_error(self):
        self.plugin.dispatch('phanthy_music', {'action': 'stop_music'})
        self.assertIsNone(self.plugin._node)

    def test_closed_or_invalid_play_returns_one_error_shape(self):
        """`error` is always absent or a dict with a `code`. It used to be a
        plain string on this path and a dict on the MusicError path, so a
        caller could not read `error.code` without checking the type first.
        """
        reply = self.plugin.dispatch('phanthy_music',
                                     {'action': 'play', 'track_id': 'fictional-1'})
        self.assertIsInstance(reply['error'], dict)
        self.assertIn('code', reply['error'])
        self.plugin.dispatch('phanthy_music', {'action': 'start'})
        reply = self.plugin.dispatch('phanthy_music', {'action': 'previous'})
        self.assertIsInstance(reply['error'], dict)
        self.assertEqual(self.plugin._node.player.state, 'idle')


class TransportTests(unittest.TestCase):
    @staticmethod
    def dns_rows(*ips):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, 443)) for ip in ips]

    def test_url_validation_and_redirect_downgrade(self):
        for url in ('http://audio.example/a', 'file:///etc/passwd', 'https://user:pass@audio.example/a', 'https://audio.example/a#x'):
            with self.assertRaises(ValueError):
                validate_url(url)
        with self.assertRaises(ValueError):
            _HTTPSRedirect().redirect_request(None, None, 302, '', {}, 'http://audio.example/a')

    def test_rejects_private_mixed_and_loopback_dns_before_connect(self):
        for ip in ('127.0.0.1', '10.0.0.1', '169.254.169.254', '::1'):
            with patch('socket.getaddrinfo', return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, 443))]), patch('socket.socket') as sock:
                with self.assertRaises(ValueError):
                    _public_connection(('audio.example', 443))
                sock.assert_not_called()

    def test_connect_uses_validated_ip_without_second_dns_lookup(self):
        with patch('socket.getaddrinfo', return_value=self.dns_rows('1.1.1.1')) as resolve, \
             patch('socket.socket') as sock, patch('plugins.music_decoder._public_dns_addresses') as fallback:
            _public_connection(('audio.example', 443))
            fallback.assert_not_called()
        resolve.assert_called_once()
        sock.return_value.connect.assert_called_once_with(('1.1.1.1', 443))

    def test_fake_ip_uses_verified_resolver_result_and_pins_media_connection(self):
        opener = Mock()
        opener.open.return_value = io.BytesIO(json.dumps({'Status': 0, 'Answer': [
            {'type': 5, 'data': 'cdn.example.'}, {'type': 1, 'data': '1.1.1.1'}]}).encode())
        with patch('socket.getaddrinfo', return_value=self.dns_rows('198.18.0.7')) as resolve, \
             patch('urllib.request.build_opener', return_value=opener), patch('socket.socket') as sock:
            _public_connection(('audio.example', 443))
        resolve.assert_called_once_with('audio.example', 443, type=socket.SOCK_STREAM)
        sock.return_value.connect.assert_called_once_with(('1.1.1.1', 443))
        request = opener.open.call_args.args[0]
        url = urlsplit(request.full_url)
        self.assertEqual((url.scheme, url.netloc, url.path), ('https', 'cloudflare-dns.com', '/dns-query'))
        self.assertEqual(parse_qs(url.query), {'name': ['audio.example'], 'type': ['A'],
                                              'edns_client_subnet': ['0.0.0.0/0']})
        self.assertFalse(request.has_header('Authorization'))
        self.assertLessEqual(opener.open.call_args.kwargs['timeout'], 3)

    def test_fake_ip_fallback_never_exempts_private_mixed_or_literal_targets(self):
        for host, ips in (
            ('198.18.0.7', ('198.18.0.7',)), ('audio.example', ('127.0.0.1',)),
            ('audio.example', ('10.0.0.1',)), ('audio.example', ('169.254.169.254',)),
            ('audio.example', ('198.18.0.7', '1.1.1.1')),
            ('audio.example', ('198.18.0.7', '10.0.0.1')),
        ):
            with self.subTest(host=host, ips=ips), \
                 patch('socket.getaddrinfo', return_value=self.dns_rows(*ips)), \
                 patch('plugins.music_decoder._public_dns_addresses') as fallback, patch('socket.socket') as sock:
                with self.assertRaises(AudioAddressError):
                    _public_connection((host, 443))
                fallback.assert_not_called()
                sock.assert_not_called()

    def test_fallback_addresses_are_checked_again_at_connection_boundary(self):
        with patch('socket.getaddrinfo', return_value=self.dns_rows('198.18.0.7')), \
             patch('plugins.music_decoder._public_dns_addresses', return_value=['10.0.0.1']), \
             patch('socket.socket') as sock:
            with self.assertRaises(AudioAddressError):
                _public_connection(('audio.example', 443))
            sock.assert_not_called()

    def test_dns_bad_responses_are_bounded_and_fail_closed(self):
        payloads = [[], {'Status': 2}, {'Status': 0, 'Answer': []},
                    {'Status': 0, 'TC': True, 'Answer': [{'type': 1, 'data': '1.1.1.1'}]},
                    {'Status': 0, 'Answer': [{'type': 1, 'data': '1.1.1.1'}, {'type': 1, 'data': '10.0.0.1'}]},
                    {'Status': 0, 'Answer': [{'type': 1, 'data': '198.18.0.8'}]},
                    {'Status': 0, 'Answer': [{'type': 1, 'data': 'not-an-ip'}]}, b'x' * 16385]
        for payload in payloads:
            opener = Mock()
            opener.open.return_value = io.BytesIO(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
            with self.subTest(payload_type=type(payload).__name__), patch('urllib.request.build_opener', return_value=opener):
                with self.assertRaisesRegex(AudioAddressError, 'proxy fake-IP'):
                    _public_dns_addresses('audio.example', 10)
                opener.open.assert_called_once()  # Invalid answers never try another resolver.

    def test_dns_deadline_and_failures_have_at_most_two_resolver_requests(self):
        with patch('urllib.request.build_opener') as opener:
            with self.assertRaises(AudioAddressError):
                _public_dns_addresses('audio.example', 0)
            opener.assert_not_called()
        opener = Mock()
        opener.open.side_effect = TimeoutError()
        with patch('urllib.request.build_opener', return_value=opener):
            with self.assertRaisesRegex(AudioAddressError, 'configure real DNS'):
                _public_dns_addresses('audio.example', 10)
        self.assertEqual(opener.open.call_count, 2)
        with self.assertRaises(AudioAddressError):
            _DNSNoRedirect().redirect_request(None, None, 302, '', {}, 'https://other.example/')

    def test_dns_second_resolver_recovers_from_first_transport_timeout(self):
        opener = Mock()
        opener.open.side_effect = [TimeoutError(), io.BytesIO(json.dumps({
            'Status': 0, 'Answer': [{'type': 1, 'data': '1.1.1.1'}]}).encode())]
        with patch('urllib.request.build_opener', return_value=opener):
            self.assertEqual(_public_dns_addresses('audio.example', 10), ['1.1.1.1'])
        self.assertEqual([urlsplit(call.args[0].full_url).hostname for call in opener.open.call_args_list],
                         ['cloudflare-dns.com', 'dns.google'])

    def test_media_tls_still_checks_certificate_and_hostname(self):
        import ssl
        connection = _PublicHTTPSConnection('audio.example', timeout=10)
        try:
            self.assertTrue(connection._context.check_hostname)
            self.assertEqual(connection._context.verify_mode, ssl.CERT_REQUIRED)
        finally:
            connection.close()

    def test_address_failure_is_actionable_without_leaking_url(self):
        message = 'Audio DNS returned a proxy fake-IP, but public DNS lookup failed; configure real DNS'
        with patch('plugins.music_decoder.open_audio', side_effect=AudioAddressError(message)):
            decoder = Decoder('https://audio.example/a.mp3?signature=fake-secret', ffmpeg=ffmpeg_path(self))
            decoder.start()
            try:
                self.assertTrue(decoder.done.wait(3))
                self.assertEqual(decoder.error, message)
                self.assertNotIn('fake-secret', decoder.error)
            finally:
                decoder.close()

    def test_real_ffmpeg_decodes_generated_wav_without_saving_audio(self):
        ffmpeg = ffmpeg_path(self)
        with patch('plugins.music_decoder.open_audio', return_value=io.BytesIO(wav_bytes(pcm(1234, 8000)))):
            decoder = Decoder('https://audio.example/generated.wav', 'wav', ffmpeg=ffmpeg)
            decoder.start()
            self.assertTrue(decoder.done.wait(5))
            data = b''.join(list(decoder.frames.queue))
            self.assertEqual(decoder.error, '')
            self.assertEqual(data, pcm(1234, 8000))
            decoder.close()

    def test_real_mp3_and_full_queue_cancellation(self):
        ffmpeg = ffmpeg_path(self)
        samples = (8000 * np.sin(np.arange(80000) * 2 * np.pi * 440 / 16000)).astype('<i2').tobytes()
        encoded = subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error',
                                  '-f', 'wav', '-i', 'pipe:0', '-f', 'mp3', 'pipe:1'],
                                 input=wav_bytes(samples), capture_output=True, check=True, timeout=5).stdout
        with patch('plugins.music_decoder.open_audio', return_value=io.BytesIO(encoded)):
            decoder = Decoder('https://audio.example/generated.mp3', ffmpeg=ffmpeg)
            decoder.start()
            try:
                deadline = time.monotonic() + 5
                while not decoder.frames.full() and not decoder.done.is_set() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(decoder.frames.full())
                self.assertFalse(decoder.done.is_set())  # Backpressure, not an unbounded buffer.
                self.assertGreater(np.abs(np.frombuffer(decoder.frames.get(), dtype='<i2')).max(), 100)
                decoder.cancel()
                self.assertTrue(decoder.done.wait(2))
                self.assertEqual(decoder.error, '')
            finally:
                decoder.close()

    def test_expired_download_error_does_not_echo_signed_url(self):
        import urllib.error
        url = 'https://audio.example/expired.mp3?signature=fake-secret'
        with patch('plugins.music_decoder.open_audio', side_effect=urllib.error.HTTPError(url, 403, 'denied', {}, None)):
            decoder = Decoder(url, ffmpeg=ffmpeg_path(self))
            decoder.start()
            try:
                self.assertTrue(decoder.done.wait(3))
                self.assertIn('refresh', decoder.error)
                self.assertNotIn('fake-secret', decoder.error)
            finally:
                decoder.close()

    def test_decoder_failure_and_cancellation_are_bounded(self):
        decoder = Decoder('https://audio.example/a.mp3', ffmpeg='/nonexistent/ffmpeg')
        decoder.start()
        self.assertTrue(decoder.done.wait(2))
        self.assertIn('not installed', decoder.error)
        self.assertEqual(decoder.frames.maxsize, 20)
        decoder.close()


if __name__ == '__main__':
    unittest.main()
