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


class MixerTests(unittest.TestCase):
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

    def test_speech_passthrough_without_catalogue_or_song(self):
        voice = pcm(14000)
        self.player.feed_voice(voice)
        self.player.feed_voice(AUDIO_EOF)
        self.step()
        self.step()
        self.assertEqual(self.output, [voice, AUDIO_EOF])

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

    def test_speech_ducks_music_and_restores_without_forwarding_speech_eof(self):
        self.play()
        for _ in range(12):
            self.step(pcm(10000))
        full = np.frombuffer(self.output[-1], dtype='<i2').mean()
        self.player.feed_voice(pcm(1000, 16000))
        self.player.feed_voice(AUDIO_EOF)
        for _ in range(10):
            self.step(pcm(10000))
        ducked = np.frombuffer(self.output[-1], dtype='<i2').mean()
        self.assertLess(ducked, full / 2)
        self.assertNotIn(AUDIO_EOF, self.output)
        for _ in range(12):
            self.step(pcm(10000))
        self.assertGreater(np.frombuffer(self.output[-1], dtype='<i2').mean(), full * .95)

    def test_user_speech_end_restores_music(self):
        self.play()
        self.player.hearing(True)
        for _ in range(10):
            self.step(pcm(10000))
        self.assertLess(np.frombuffer(self.output[-1], dtype='<i2').mean(), 2000)
        self.player.hearing(False)
        for _ in range(15):
            self.step(pcm(10000))
        self.assertGreater(np.frombuffer(self.output[-1], dtype='<i2').mean(), 9500)

    def test_pause_preserves_music_buffer_but_speech_continues(self):
        self.play()
        self.player._decoder.frames.put(pcm(10000))
        self.player.pause()
        self.player.feed_voice(pcm(3000))
        self.step()
        self.assertEqual(self.output[-1], pcm(3000))
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
        self.player.feed_voice(pcm(3000))
        self.step()
        self.assertEqual(self.output[-1], pcm(3000))
        self.assertNotEqual(self.play()['playback_id'], first)
        self.assertTrue(self.player._decoder.frames.empty())

    def test_voice_interrupt_does_not_cancel_song_or_swallow_next_utterance(self):
        self.play()
        self.player.feed_voice(pcm(9000))
        self.player.interrupt_voice()
        self.player.feed_voice(pcm(9000))  # old late frame
        self.assertEqual(len(self.player._voice), 0)
        self.player.feed_voice(AUDIO_EOF)
        self.player.feed_voice(pcm(2000))
        self.player.feed_voice(AUDIO_EOF)
        self.step()
        self.assertEqual(self.output[-1], pcm(2000))
        self.assertIsNotNone(self.player._decoder)

    def test_idle_interrupt_does_not_mute_future_speech(self):
        self.player.interrupt_voice()
        self.player.feed_voice(pcm(3000))
        self.step()
        self.assertEqual(self.output[-1], pcm(3000))

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

    def test_clip_and_volume_apply_only_to_music(self):
        self.play()
        self.player._music_gain = 1
        self.player.duck_gain = 1
        self.player.feed_voice(pcm(30000))
        self.step(pcm(30000))
        self.assertEqual(np.frombuffer(self.output[-1], dtype='<i2').max(), 32767)
        self.player.set_volume(0)
        for _ in range(20):
            self.player.feed_voice(pcm(3000))
            self.step(pcm(30000))
        self.assertLess(abs(np.frombuffer(self.output[-1], dtype='<i2').mean() - 3000), 2)

    def test_voice_buffer_has_a_limit(self):
        self.player.feed_voice(b'\0' * (MAX_VOICE_BYTES + 2))
        self.assertIn('bounded', self.player.error)
        self.assertEqual(len(self.player._voice), 0)

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

    def test_schema_and_background_hooks(self):
        import jsonschema
        tool = self.plugin.get_tools()[0]
        self.assertEqual(tool['name'], 'phanthy_music')
        schema = tool['inputSchema']
        self.assertNotIn('x-completion', schema)
        self.assertNotEqual(schema['x-resource'], 'mouth')
        self.assertEqual(set(schema['x-action-params']),
                         {'search', 'play_by_id', 'play_by_genre', 'pause', 'resume', 'interrupt', 'status', 'set_volume'})
        self.assertEqual(schema['x-hooks']['on_interrupt_all']['action'], 'interrupt_voice')
        jsonschema.validate({'action': 'play_by_id', 'track_id': 'fictional-1'}, schema)
        jsonschema.validate({'action': 'play_by_genre', 'genre': 'pop'}, schema)

    def test_start_stop_and_restart_have_one_node_and_no_autoplay(self):
        for _ in range(3):
            result = self.plugin.dispatch('phanthy_music', {'action': 'start', 'input_topic': '/speech/tts'})
            self.assertEqual(result['state'], 'running')
            self.assertEqual(result['playback_state'], 'idle')
            self.assertEqual(result['topic_in'][0]['topic'], '/speech/tts')
            self.assertEqual(result['topic_out'][0]['topic'], OUTPUT_TOPIC)
            self.assertEqual(len(self.executor.nodes), 1)
        node = self.plugin._node
        self.plugin.dispatch('phanthy_music', {'action': 'stop'})
        self.assertTrue(node.destroyed)
        self.assertEqual(len(self.executor.nodes), 0)
        self.assertFalse(node.player._thread.is_alive())
        self.plugin.dispatch('phanthy_music', {'action': 'start'})
        self.assertEqual(len(self.executor.nodes), 1)

    def test_passes_real_audio_messages_to_output(self):
        self.plugin.dispatch('phanthy_music', {'action': 'start', 'input_topic': '/speech'})
        node = self.plugin._node
        msg = _FakeAudioChunk()
        msg.format, msg.data = FORMAT, pcm(1000)
        node.subscription.callback(msg)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and not node.publisher.messages:
            time.sleep(.01)
        self.assertTrue(node.publisher.messages)
        self.assertEqual(bytes(node.publisher.messages[0]), pcm(1000))

    def test_no_hook_autostart_and_no_feedback_connection(self):
        self.plugin.dispatch('phanthy_music', {'action': 'duck'})
        self.assertIsNone(self.plugin._node)
        result = self.plugin.dispatch('phanthy_music', {'action': 'start', 'input_topic': OUTPUT_TOPIC})
        self.assertEqual(result['state'], 'error')
        self.assertEqual(self.executor.nodes, [])
        self.assertIn('error', self.plugin.dispatch('phanthy_music', {'action': 'start', 'input_topics': ['/a', '/b']}))

    def test_closed_or_invalid_play_returns_error(self):
        self.assertIn('error', self.plugin.dispatch('phanthy_music', {'action': 'play_by_id', 'track_id': 'fictional-1'}))
        self.plugin.dispatch('phanthy_music', {'action': 'start'})
        self.assertIn('error', self.plugin.dispatch('phanthy_music', {'action': 'play_by_id'}))
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
