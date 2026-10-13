"""The mixer: speech always audible, background ducked under it, no clicks.

These assertions used to live in test_music.py's MixerTests, because the music
card was the mixer. They moved with the code. Everything here runs on a fake
clock against `audio_mix.Mixer` — no ROS, no catalogue, no ffmpeg.
"""
import unittest
from unittest.mock import patch

import numpy as np

from vision_stubs import _FakeExecutor
from plugins.audio_mix import AUDIO_EOF, MAX_VOICE_BYTES, Mixer, paced_loop


def pcm(value, samples=1600):
    return np.full(samples, value, dtype='<i2').tobytes()


def mean(frame):
    return abs(np.frombuffer(frame, dtype='<i2').astype(np.float32)).mean()


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class MixerTests(unittest.TestCase):
    def setUp(self):
        self.output = []
        self.clock = FakeClock()
        # prefill 关掉：这一组用例测的是**每帧的混音数学**（原音量语音、渐变、裁剪、
        # EOF 边界），每次只喂一帧就 tick —— 而抖动缓冲按设计就会把单独一帧先攒着。
        # 默认 prefill 的行为由下面三个专门的用例覆盖，它们各自建自己的 Mixer。
        self.mixer = Mixer(self.output.append, clock=self.clock, speech_prefill_ms=0)

    def tearDown(self):
        self.mixer.close()

    def step(self, *, speech=None, background=None):
        if speech is not None:
            self.mixer.feed(speech, speech=True)
        if background is not None:
            self.mixer.feed(background, speech=False)
        self.mixer.tick()
        self.clock.now += 0.1

    def test_speech_passes_through_with_no_background_at_all(self):
        self.step(speech=pcm(3000))
        self.assertEqual(self.output[-1], pcm(3000))

    def test_speech_is_never_attenuated(self):
        """duck_gain and background_gain must not touch the robot's own voice."""
        self.mixer.configure(duck_gain=0.0, background_gain=0)
        self.step(speech=pcm(3000))
        self.assertAlmostEqual(mean(self.output[-1]), 3000, delta=2)

    def test_background_alone_plays_at_full_level(self):
        for _ in range(8):
            self.step(background=pcm(10000))
        self.assertGreater(mean(self.output[-1]), 9500)

    def test_speech_ducks_the_background_and_it_restores_after(self):
        for _ in range(8):
            self.step(background=pcm(10000))
        full = mean(self.output[-1])
        for _ in range(4):
            self.step(speech=pcm(1), background=pcm(10000))
        ducked = mean(self.output[-1])
        self.assertLess(ducked, full / 2)
        # The utterance ends the way a real one does — with an EOF. Letting the
        # buffer merely run dry is not enough: the stream stays open for another
        # two seconds, and the duck correctly stays on for all of it.
        self.mixer.feed(AUDIO_EOF, speech=True)
        self.clock.now += 0.5           # past duck_hold_ms
        for _ in range(12):
            self.step(background=pcm(10000))
        self.assertGreater(mean(self.output[-1]), full * 0.95)

    def test_the_gain_moves_gradually_rather_than_stepping(self):
        """A step change in gain is audible as a click at every utterance."""
        for _ in range(8):
            self.step(background=pcm(10000))
        levels = []
        for _ in range(4):
            self.step(speech=pcm(1), background=pcm(10000))
            levels.append(mean(self.output[-1]))
        self.assertTrue(all(b <= a for a, b in zip(levels, levels[1:])),
                        f'gain should fall monotonically, got {levels}')
        self.assertGreater(levels[0], levels[-1])

    def test_a_speech_eof_is_not_forwarded_while_background_is_live(self):
        """Only the *combined* stream's end is a boundary for the speaker."""
        for _ in range(4):
            self.step(background=pcm(8000))
        self.mixer.feed(AUDIO_EOF, speech=True)
        for _ in range(3):
            self.step(background=pcm(8000))
        self.assertNotIn(AUDIO_EOF, self.output)

    def test_the_combined_stream_emits_one_eof_when_everything_stops(self):
        self.step(speech=pcm(3000))
        self.mixer.feed(AUDIO_EOF, speech=True)
        self.step()
        self.assertEqual(self.output[-1], AUDIO_EOF)

    def test_flush_speech_drops_buffered_speech_but_not_the_next_utterance(self):
        """Barge-in: synthesis is told to stop, but up to 500 ms of it is
        already buffered here (tts.py bursts a 5-frame prebuffer).
        """
        self.mixer.feed(pcm(3000), speech=True)
        self.mixer.flush_speech()
        self.step(background=pcm(9000))
        self.assertLess(mean(self.output[-1]), 9000)   # no speech in it
        # Stale frames of the interrupted utterance stay dropped...
        self.step(speech=pcm(5000))
        self.assertEqual(self.mixer.status()['speech_buffered_ms'], 0)
        # ...until the next EOF, after which the next utterance is heard.
        self.mixer.feed(AUDIO_EOF, speech=True)
        self.step(speech=pcm(5000))
        self.assertGreater(mean(self.output[-1]), 100)

    def test_speech_does_not_break_up_when_music_is_playing(self):
        """实测到的 bug：一边放音乐一边说话，说话断断续续。

        根因不是解码也不是音量，是**输出节奏由谁驱动**。背景一在，
        `if speech or background` 就每帧必发，于是语音每一次晚到都被固化成输出流
        里一个 100ms 的静音洞。没有背景时同样的晚到不会有洞 —— 那一拍两个源都空，
        什么都不发，节奏自然跟着语音走。所以这个 bug 只在两者同时存在时出现，
        这个用例也必须同时断言这两种情形。
        """
        gaps = {}
        for label, with_music in (('music', True), ('silence', False)):
            out = []
            clock = FakeClock()
            mixer = Mixer(out.append, clock=clock, duck_gain=1.0, background_gain=0)
            self.addCleanup(mixer.close)
            for i in range(8):
                if i != 4:                      # 第 5 帧晚到一拍
                    mixer.feed(pcm(3000), speech=True)
                if with_music:
                    mixer.feed(pcm(0), speech=False)
                mixer.tick()
                clock.now += 0.1
            levels = [mean(f) for f in out if f != AUDIO_EOF]
            # 开头的攒包不算断：那是延迟，不是中断。只看语音开始之后。
            started = next((i for i, v in enumerate(levels) if v > 0), len(levels))
            gaps[label] = [v for v in levels[started:]].count(0)
        self.assertEqual(gaps['music'], 0,
                         '音乐在播时语音被切出了洞 —— 正是这个 bug')
        self.assertEqual(gaps['silence'], 0)

    def test_the_prefill_does_not_delay_a_real_utterance(self):
        """tts.py 先突发 5 帧再按节奏发，所以余量几乎立刻攒满。

        抖动缓冲换来的延迟直接影响打断响应，所以它必须便宜。
        """
        out = []
        clock = FakeClock()
        mixer = Mixer(out.append, clock=clock, duck_gain=1.0, background_gain=0)
        self.addCleanup(mixer.close)
        for _ in range(5):                      # PREBUF_FRAMES
            mixer.feed(pcm(3000), speech=True)
        mixer.feed(pcm(0), speech=False)
        mixer.tick()
        self.assertGreater(mean(out[0]), 0, '第一帧就该有语音')

    def test_an_utterance_shorter_than_the_prefill_still_plays(self):
        """EOF 到了就说明这段话只有这么多，再等也不会更多。"""
        out = []
        clock = FakeClock()
        mixer = Mixer(out.append, clock=clock, duck_gain=1.0, background_gain=0)
        self.addCleanup(mixer.close)
        mixer.feed(pcm(3000), speech=True)
        mixer.feed(AUDIO_EOF, speech=True)
        mixer.feed(pcm(0), speech=False)
        mixer.tick()
        self.assertGreater(mean(out[-1]), 0)

    def test_flush_speech_does_not_cancel_the_background(self):
        for _ in range(8):
            self.step(background=pcm(10000))
        self.mixer.flush_speech()
        self.step(background=pcm(10000))
        self.assertGreater(mean(self.output[-1]), 9000)

    def test_flush_speech_while_idle_does_not_mute_future_speech(self):
        self.mixer.flush_speech()
        self.step(speech=pcm(4000))
        self.assertAlmostEqual(mean(self.output[-1]), 4000, delta=2)

    def test_an_explicit_duck_lifts_itself_after_a_failsafe(self):
        """A lost end-of-speech must not leave music permanently inaudible."""
        self.mixer.duck(True)
        for _ in range(6):
            self.step(background=pcm(10000))
        self.assertLess(mean(self.output[-1]), 3000)
        self.clock.now += 61            # past DUCK_FAILSAFE_S
        for _ in range(12):
            self.step(background=pcm(10000))
        self.assertGreater(mean(self.output[-1]), 9000)

    def test_the_sum_is_clipped_rather_than_wrapped(self):
        self.step(speech=pcm(30000), background=pcm(30000))
        self.assertLessEqual(np.frombuffer(self.output[-1], dtype='<i2').max(), 32767)

    def test_each_source_buffer_is_bounded(self):
        for speech in (True, False):
            with self.subTest(speech=speech):
                mixer = Mixer(lambda _p: None, clock=FakeClock())
                mixer.feed(b'\x00' * (MAX_VOICE_BYTES + 2), speech=speech)
                self.assertIn('bounded', mixer.status()['error'])
                mixer.close()

    def test_a_background_backlog_is_trimmed_instead_of_growing(self):
        """Orin5 实测：background_buffered_ms 涨到 9875，音频一卡一卡。

        背景是实时流 —— 听众要的是「现在」的音乐。攒着只会让延迟一直涨到撞
        MAX_VOICE_BYTES 的 20 秒上限、整个清空、报错、重来。
        """
        for _ in range(60):                       # 6 秒音频，只喂不消费
            self.mixer.feed(pcm(9000), speech=False)
        status = self.mixer.status()
        self.assertLessEqual(status['background_buffered_ms'], 1000)
        self.assertGreater(status['background_dropped_ms'], 0,
                           '丢弃必须是可见的事实，不是悄悄的自我修正')
        self.assertEqual(status['error'], '', '裁剪是正常止损，不该报成错误')

    def test_speech_is_never_trimmed_that_way(self):
        """丢背景只是过时；丢语音就是少说字。"""
        for _ in range(8):
            self.mixer.feed(pcm(3000), speech=True)
        self.assertGreater(self.mixer.status()['speech_buffered_ms'], 700)

    def test_odd_length_pcm_is_rejected_not_misread(self):
        self.mixer.feed(b'\x01\x02\x03', speech=True)
        self.assertIn('S16LE', self.mixer.status()['error'])

    def test_a_silent_source_stops_ducking_without_an_eof(self):
        """A source that went away without saying so is not still speaking."""
        self.step(speech=pcm(3000), background=pcm(10000))
        self.clock.now += 3
        for _ in range(12):
            self.step(background=pcm(10000))
        self.assertFalse(self.mixer.ducked())

    def test_no_subscriber_is_reported_rather_than_buffered_forever(self):
        mixer = Mixer(self.output.append, has_reader=lambda: False, clock=self.clock)
        self.addCleanup(mixer.close)
        for _ in range(70):
            mixer.feed(pcm(3000), speech=True)
            mixer.tick()
            self.clock.now += 0.1
        self.assertIn('Speaker', mixer.status()['error'])


class PacedLoopTests(unittest.TestCase):
    def test_a_late_frame_is_dropped_rather_than_burst(self):
        """Chasing a missed deadline publishes several frames back to back, and
        a speaker fed a burst plays it as a chirp.
        """
        import threading
        clock = FakeClock()
        stop = threading.Event()
        calls = []

        def tick():
            calls.append(clock.now)
            clock.now += 5          # every tick overruns its 100 ms budget
            if len(calls) >= 3:
                stop.set()

        waits = []
        with patch.object(stop, 'wait', side_effect=lambda t: waits.append(t)):
            paced_loop(stop, tick, clock=clock)
        self.assertEqual(waits, [0, 0, 0])   # never negative, never catching up


class MixerCardTests(unittest.TestCase):
    """The card around the mixer: roles, and what `info()` has to report."""

    def _plugin(self, **cfg):
        from plugins.audio_mixer import AudioMixerPlugin
        return AudioMixerPlugin(cfg, _FakeExecutor())

    def test_the_declaration_offers_the_model_nothing_to_call(self):
        """Mixing is wiring, not a decision an agent makes."""
        tool = self._plugin().get_tools()[0]
        self.assertNotIn('x-action-params', tool['inputSchema'])
        self.assertEqual(tool['inputSchema']['x-resource'], 'speaker_mix')
        self.assertEqual(set(tool['inputSchema']['x-hooks']),
                         {'on_interrupt_speak', 'on_interrupt_all'})

    def test_auto_finds_the_tts_topic_through_its_info_action(self):
        """TTSPlugin 是个 facade，它自己没有 _nodes —— 节点字典在当前加载的引擎里。

        猜属性名会一无所获，于是 auto 解析出空的语音集合，**每一个**输入都被当成
        背景音乐，包括 TTS 本身。Orin5 上实测到的就是这个。
        """
        class FakeTTS:
            def dispatch(self, name, args):
                return {'topic_out': [{'topic': '/perception/tts',
                                       'format': 'audio/pcm-16k'}]}
        from plugins.audio_mixer import AudioMixerPlugin
        plugin = AudioMixerPlugin({}, _FakeExecutor(), tts=FakeTTS())
        plugin.dispatch('audio_mixer', {
            'action': 'start',
            'input_topics': ['/perception/tts', '/perception/music/audio']})
        self.addCleanup(plugin.stop)
        info = plugin.dispatch('audio_mixer', {'action': 'info'})
        self.assertEqual(info['speech_topics'], ['/perception/tts'])
        self.assertNotIn('degraded', info)
        roles = {t['topic']: t['role'] for t in info['topic_in']}
        self.assertEqual(roles['/perception/tts'], 'speech')
        self.assertEqual(roles['/perception/music/audio'], 'background')

    def test_a_broken_tts_does_not_stop_the_card_answering(self):
        class AngryTTS:
            def dispatch(self, name, args):
                raise RuntimeError('engine is loading')
        from plugins.audio_mixer import AudioMixerPlugin
        plugin = AudioMixerPlugin({}, _FakeExecutor(), tts=AngryTTS())
        self.assertTrue(plugin.get_tools())

    def test_one_in_port_because_port_identity_does_not_survive_to_start(self):
        """api/config.py's _resolve_input_topics discards toPortIdx, so a
        'speech' port and a 'background' port would be resolved by a coin flip.
        """
        tool = self._plugin().get_tools()[0]
        self.assertEqual(len(tool['topic_in']), 1)

    def test_info_reports_every_bound_topic_with_its_role(self):
        """_dropped_inputs fails a card's start when it was handed >=2 topics
        and info() does not account for all of them.
        """
        plugin = self._plugin(speech_topics='/speech/tts')
        plugin.dispatch('audio_mixer', {
            'action': 'start',
            'input_topics': ['/speech/tts', '/perception/music/audio']})
        self.addCleanup(plugin.stop)
        info = plugin.dispatch('audio_mixer', {'action': 'info'})
        roles = {t['topic']: t['role'] for t in info['topic_in']}
        self.assertEqual(roles, {'/speech/tts': 'speech',
                                 '/perception/music/audio': 'background'})
        self.assertNotIn('degraded', info)

    def test_the_dropdown_lists_the_topics_actually_wired_in(self):
        """选「哪些算说话」应该在**接进来的话题**之间选。

        原先列的是本进程 TTS 卡在发什么 —— 那是在猜谁是语音源，而别的容器里的
        TTS、驱动自带的语音卡、第二个音源，都不在那份猜测里。输入是确切已知的。
        """
        plugin = self._plugin(speech_topics='/speech/tts')
        plugin.dispatch('audio_mixer', {
            'action': 'start',
            'input_topics': ['/speech/tts', '/perception/music/audio']})
        self.addCleanup(plugin.stop)
        options = (plugin.get_tools()[0]['configSchema']['properties']
                   ['speech_topics']['oneOf'])
        consts = [o['const'] for o in options]
        self.assertEqual(set(consts),
                         {'auto', '/speech/tts', '/perception/music/audio'})

    def test_before_wiring_there_is_nothing_to_choose_among(self):
        """没接线就没有候选，只有 auto —— 这是诚实的，不该凭空列出猜来的话题。"""
        options = (self._plugin().get_tools()[0]['configSchema']['properties']
                   ['speech_topics']['oneOf'])
        self.assertEqual([o['const'] for o in options], ['auto'])

    def test_no_identifiable_speech_source_is_reported_as_degraded(self):
        """Otherwise every input is background and the robot's own voice gets
        ducked by its own music, with nothing saying why.
        """
        plugin = self._plugin()        # `auto`, and no TTS handle
        plugin.dispatch('audio_mixer', {'action': 'start',
                                        'input_topics': ['/a', '/b']})
        self.addCleanup(plugin.stop)
        info = plugin.dispatch('audio_mixer', {'action': 'info'})
        self.assertIn('degraded', info)
        self.assertEqual(info['speech_topics'], [])

    def test_its_own_output_is_refused_as_an_input(self):
        plugin = self._plugin()
        from plugins.audio_mixer import OUTPUT_TOPIC
        reply = plugin.dispatch('audio_mixer', {'action': 'start',
                                                'input_topics': [OUTPUT_TOPIC]})
        self.assertIn('error', reply)

    def test_starting_with_no_input_says_what_to_connect(self):
        reply = self._plugin().dispatch('audio_mixer', {'action': 'start'})
        self.assertIn('phanthy_music', reply['error']['message'])

    def test_an_interrupt_hook_while_idle_is_not_an_error(self):
        plugin = self._plugin()
        self.assertEqual(plugin.dispatch(
            'audio_mixer', {'action': 'flush_speech'})['state'], 'idle')


if __name__ == '__main__':
    unittest.main()
