"""压缩历史时，说出口的原话不能被压成 `前N字…(略)`。

天轶导览里 tts.speak 的 text 先被 turn 内压缩（前 30 字）、再被 tier2 降级（前 20 字）
压成省略稿。省略稿留在历史里就是一条调用样例，模型照着写出了新的
`{"text": "我们现在到达具身智能展位。前面介绍的AI更多生活在软件里，具…(略)"}`
—— 这一条本身没被压缩过，`…(略)` 是模型自己生成的，深圳现场被 TTS 念了出来。
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import mcp_client  # noqa: E402
from event.llm import (  # noqa: E402
    _ELIDED,
    _compact_turn_messages,
    _degrade_turn,
    _is_utterance,
)

SPEECH = ('我们现在到达具身智能展位。前面介绍的AI更多生活在软件里，'
          '具身智能则让AI拥有身体，能看、能听、能动，真正走进物理世界。') * 2
SPEAK = 'mcp__perc__tts__speak'
NAV = 'mcp__body__navigate_to_tag'
REPLY = 'mcp__channel__channel_reply'


def _call(cid, name, args):
    return {'id': cid, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}}


def _args(msg):
    return [json.loads(tc['function']['arguments']) for tc in msg['tool_calls']]


class UtteranceCompactionTest(unittest.TestCase):

    def setUp(self):
        self._saved = {k: mcp_client.registry.get(k) for k in ('perc', 'body')}
        mcp_client.registry['perc'] = {'tool_meta': {
            SPEAK: {'type': 'actuator', 'resource': frozenset({'mouth'})}}}
        mcp_client.registry['body'] = {'tool_meta': {
            NAV: {'type': 'actuator', 'resource': frozenset({'base'})}}}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                mcp_client.registry.pop(k, None)
            else:
                mcp_client.registry[k] = v

    def _turn(self):
        return [
            {'role': 'assistant', 'content': '', 'tool_calls': [
                _call('c1', SPEAK, {'text': SPEECH}),
                _call('c2', NAV, {'tag_name': 'x' * 200}),
            ]},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'},
            {'role': 'tool', 'tool_call_id': 'c2', 'content': 'ok'},
        ]

    def test_is_utterance_follows_declared_channel_not_name(self):
        self.assertTrue(_is_utterance(SPEAK))
        self.assertTrue(_is_utterance(REPLY))
        self.assertFalse(_is_utterance(NAV))
        self.assertFalse(_is_utterance('mcp__gone__tts__speak'))   # 未注册：不猜
        self.assertFalse(_is_utterance('WebSearch'))

    def test_turn_compaction_keeps_speech_verbatim(self):
        msgs = self._turn() + [{'role': 'user', 'content': f'{i}'} for i in range(12)]
        _compact_turn_messages(msgs, keep_recent=12)
        speak, nav = _args(msgs[0])
        self.assertEqual(speak['text'], SPEECH)
        self.assertTrue(nav['tag_name'].endswith(_ELIDED))     # 其它工具照旧压

    def test_tier2_degrade_keeps_speech_verbatim(self):
        speak, nav = _args(_degrade_turn(self._turn())[0])
        self.assertEqual(speak['text'], SPEECH)
        self.assertNotEqual(nav['tag_name'], 'x' * 200)

    def test_channel_reply_text_kept(self):
        turn = [{'role': 'assistant', 'content': '', 'tool_calls': [
                    _call('c1', REPLY, {'action': 'send', 'text': SPEECH})]},
                {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'}]
        self.assertEqual(_args(_degrade_turn(turn)[0])[0]['text'], SPEECH)

    def test_no_elision_marker_reaches_history_after_both_passes(self):
        msgs = self._turn() + [{'role': 'user', 'content': f'{i}'} for i in range(12)]
        _compact_turn_messages(msgs, keep_recent=12)
        out = _degrade_turn(msgs)
        self.assertNotIn(_ELIDED, out[0]['tool_calls'][0]['function']['arguments'])


if __name__ == '__main__':
    unittest.main()
