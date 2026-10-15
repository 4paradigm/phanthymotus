"""Wake/command state transitions with scripted VAD events and a capture clock.

No model, microphone, phonemizer installation, network or wall-clock sleeps.
"""
import json
import queue
import urllib.request

import pytest

from plugins import asr


PCM = b'\x00\x01' * 16000
WAKE = '小星小星'


@pytest.fixture
def run_pipeline(monkeypatch):
    units = {'小': ['x', 'j', 'au'], '星': ['s', 'i', 'ng']}

    def phonemes(text, with_positions=False):
        phones, ends = [], []
        for index, char in enumerate(text):
            if char.isspace() or char in '，。！？,.!?':
                continue
            part = units.get(char, [char.lower()])
            phones.extend(part)
            ends.extend([index + 1] * len(part))
        return (phones, ends) if with_positions else phones

    monkeypatch.setattr(asr, '_text_to_ipa', phonemes)
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: None)

    def run(texts, events, keyword=WAKE, identities=None):
        class Adapter(asr.ASRAdapter):
            responses = iter(texts)

            def transcribe(self, wav_bytes, language):
                response = next(self.responses)
                return response(node) if callable(response) else response

        node = asr._ASRNode('/test/audio', Adapter(), 'zh-CN',
                            kws_cfg={'trigger_mode': 'asr_kws',
                                     'asr_kws_keyword': keyword})
        clock = [0.0]
        node._audio_now = lambda: clock[0]
        if identities is not None:
            speakers = iter(identities)
            node._collect_identity = lambda handle: next(speakers)
        items = iter([*events, ('idle', 100.0), ('stop',)])

        class Inputs:
            def get(self, timeout=None):
                try:
                    event = next(items)
                except StopIteration:
                    node.request_stop()
                    raise queue.Empty
                if callable(event):
                    event(node)
                    raise queue.Empty
                if event[0] == 'stop':
                    node.request_stop()
                    raise queue.Empty
                if event[0] == 'idle':
                    clock[0] = event[1]
                    raise queue.Empty
                clock[0] = event[2]
                return event

        node._utterance_queue = Inputs()
        node._worker_inner()
        return ([json.loads(x) for x in node._pub.messages],
                [json.loads(x) for x in node._background_pub.messages])

    return run


@pytest.mark.parametrize('text,expected', [
    ('小星小 go forward', 'go forward'),
    ('小星小星，先说几句。小星小星，打开窗户', '打开窗户'),
    ('小星小星，介绍小星小星的功能', '介绍小星小星的功能'),
    ('小星小星，打开窗户。小星小星', '打开窗户。小星小星'),
])
def test_wake_removal_preserves_body_and_distinguishes_invocations(run_pipeline, text, expected):
    primary, background = run_pipeline([text], [(PCM, 1., 2., 0)])
    assert [x['text'] for x in primary] == [expected]
    assert primary[0]['kws_triggered'] is True
    assert background == []


@pytest.mark.parametrize('onset,accepted', [(3., True), (7., True), (7.1, False)])
def test_wake_only_deadline_uses_onset_without_pre_roll(run_pipeline, onset, accepted):
    primary, background = run_pipeline([WAKE, 'open the window'],
        [(PCM, 1., 2., 0), (PCM, onset - .5, onset + .5, 16000)])
    assert [x['text'] for x in primary] == (['open the window'] if accepted else [])
    assert [x['text'] for x in background] == ([] if accepted else ['open the window'])
    if accepted:
        assert primary[0]['audio_start_ts'] == onset - .5


@pytest.mark.parametrize('texts,keyword', [
    (['小星', '小星 go forward'], WAKE),
    (['hello', 'robot open the window'], 'hello robot'),
])
def test_split_wake_keeps_the_following_command(run_pipeline, texts, keyword):
    primary, background = run_pipeline(texts,
        [(PCM, 1., 2., 0), (PCM, 2.1, 3.6, 16000)], keyword)
    assert [x['text'] for x in primary] == [texts[1].split(' ', 1)[1]]
    assert background == []


def test_partial_wakes_expire_by_speech_onset(run_pipeline):
    primary, background = run_pipeline(['小星', '小星 go forward'],
        [(PCM, 1., 2., 0), (PCM, 2.6, 4., 16000)])
    assert primary == []
    assert [x['text'] for x in background] == ['小星', '小星 go forward']


def test_four_short_fragments_do_not_publish_the_final_wake_character(run_pipeline):
    primary, background = run_pipeline(['小', '星', '小', '星', 'go forward'],
        [(PCM, 1., 1.4, 0), (PCM, 1.8, 2.2, 0), (PCM, 2.6, 3., 0),
         (PCM, 3.4, 3.8, 0), (PCM, 4.2, 5.2, 0)])
    assert [x['text'] for x in primary] == ['go forward']
    assert background == []


def test_completed_command_does_not_admit_later_background(run_pipeline):
    primary, background = run_pipeline([WAKE, 'go forward', '旁人聊天'],
        [(PCM, 1., 2., 0), (PCM, 3., 4., 0), (PCM, 6., 7., 0)])
    assert [x['text'] for x in primary] == ['go forward']
    assert [x['text'] for x in background] == ['旁人聊天']
    assert background[0]['priority'] == 0


def test_multiple_body_segments_publish_once_with_full_interval(run_pipeline):
    primary, background = run_pipeline([WAKE, '往前走', 'two steps', '然后停下'],
        [(PCM, 1., 2., 0), (PCM, 3., 4., 0),
         (PCM, 4.1, 5.6, 16000), (PCM, 5.6, 7.1, 16000)])
    assert [x['text'] for x in primary] == ['往前走 two steps 然后停下']
    assert primary[0]['audio_start_ts'] == 3.
    assert primary[0]['audio_end_ts'] == 7.1
    assert primary[0]['audio_duration_ms'] == 4100
    assert background == []


@pytest.mark.parametrize('gap,joined', [(.62, True), (.64, False), (.81, False)])
def test_body_join_window_includes_one_vad_frame_tolerance(run_pipeline, gap, joined):
    primary, background = run_pipeline([WAKE + ' 往前走', 'two steps'],
        [(PCM, 1., 2., 0), (PCM, 2. + gap - .5, 3. + gap, 16000)])
    assert [x['text'] for x in primary] == (['往前走 two steps'] if joined else ['往前走'])
    assert [x['text'] for x in background] == ([] if joined else ['two steps'])


def test_onset_grace_and_active_speech_hold_publication(run_pipeline):
    counts = []

    def observe(node):
        counts.append(len(node._pub.messages))

    primary, background = run_pipeline([WAKE + ' 往前走', 'two steps'],
        [(PCM, 1., 2., 0), ('idle', 2.6), observe,
         ('speech_start', 2.7, 2.7), ('idle', 5.), observe,
         (PCM, 2.1, 6.6, 16000), ('speech_end', 6.6, 6.6),
         ('idle', 7.5), observe])
    assert counts == [0, 0, 1]
    assert [x['text'] for x in primary] == ['往前走 two steps']
    assert background == []


def test_stop_cancels_a_pending_command(run_pipeline):
    primary, background = run_pipeline([WAKE + ' open the window'],
                                       [(PCM, 1., 2., 0), ('stop',)])
    assert primary == []
    assert background == []


def test_stop_during_decode_cannot_publish_an_older_command(run_pipeline):
    interrupted = []

    def stop_while_decoding(node):
        node.request_stop()
        interrupted.append(True)
        return WAKE + ' stop'

    primary, background = run_pipeline([WAKE + ' go forward', stop_while_decoding],
        [(PCM, 1., 2., 0), (PCM, 2.5, 3.5, 0)])
    assert interrupted == [True]
    assert primary == []
    assert background == []


def test_a_known_different_speaker_does_not_extend_the_command(run_pipeline):
    primary, background = run_pipeline([WAKE + ' 往前走', '旁人聊天'],
        [(PCM, 1., 2., 0), (PCM, 2.5, 3.5, 0)],
        identities=[{'speaker_id': 'one'}, {'speaker_id': 'two'}])
    assert [x['text'] for x in primary] == ['往前走']
    assert primary[0]['speaker_id'] == 'one'
    assert background[0]['text'] == '旁人聊天'
    assert background[0]['speaker_id'] == 'two'
