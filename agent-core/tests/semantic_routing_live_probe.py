"""Opt-in real Jev probe: synthetic text only; no ROS, tools or control loop.

Set TYPESAFE_API_KEY and DB_PATH to an isolated database before running.
Timeouts are reported separately, never counted as successful classifications.
"""
import asyncio
import json
import os
from pathlib import Path
import sys
import time

if not os.environ.get('DB_PATH') or not os.environ.get('TYPESAFE_API_KEY'):
    raise SystemExit('Set isolated DB_PATH and TYPESAFE_API_KEY first')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import semantic_routing as routing

CASES = (
    ('other_named', '小李，我想到一个问题。', False),
    ('other_homophone', '晓李，你中午吃什么？', False),
    ('other_unknown_name', '小张，我想到一个问题。', False),
    ('other_explicit', '我在跟旁边的人说话，不是在问机器人。', False),
    ('echo', '我能介绍展厅的机器人，还可以帮你拍照。', False),
    ('echo_asr', '我能介绍展听的机器仁，还可以帮你拍照。', False),
    ('echo_fragment', '还可以帮你拍照。', False),
    ('direct', '小范，你能做什么？', True),
    ('contact_other', '小范，帮我联系小李。', True),
    ('stop', '停一下，先别讲了。', True),
    ('mixed', '你刚说可以介绍展厅的机器人，先别讲了。', True),
)


async def main():
    # A separate five-second diagnostic budget reveals model judgments even
    # when slower than production's default two-second admission deadline.
    cfg = {**routing.DEFAULTS, 'jev_timeout_s': 5}
    errors = mismatches = over_budget = 0
    for name, text, expected in CASES:
        state = {
            'identity': '你是展厅服务机器人小范。',
            'history': [{'role': 'user', 'content': '小范，介绍一下你能做什么。'},
                        {'role': 'assistant', 'content': '我能介绍展厅的机器人，还可以帮你拍照。'}],
            'runtime': {'main_loop_busy': True},
            'recent_robot_speech': [{'text': '我能介绍展厅的机器人，还可以帮你拍照。',
                                    'status': 'dispatched_playback_unconfirmed'}],
            'message': {'text': text, 'source': 'dds:/remote_control/mic/asr',
                        'kind': 'voice', 'ts': time.time()},
        }
        started = time.monotonic()
        try:
            # Intentionally bypass exact-text shortcut to verify Jev itself.
            body = await asyncio.wait_for(routing.request_jev(state, True, cfg), 5)
            accepted, mode, reason, _, _ = routing.parse_result(body, True, cfg)
            elapsed = round((time.monotonic() - started) * 1000)
            over_budget += elapsed > 2000
            mismatches += accepted != expected
            print(json.dumps({'case': name, 'expected_admitted': expected,
                              'admitted': accepted, 'reason': reason, 'mode': mode,
                              'answers': body.get('answers'), 'model': body.get('model'),
                              'ms': elapsed, 'within_default_budget': elapsed <= 2000},
                             ensure_ascii=False), flush=True)
        except Exception as exc:
            errors += 1
            print(json.dumps({'case': name, 'error': type(exc).__name__}), flush=True)
    print(json.dumps({'cases': len(CASES), 'classification_mismatches': mismatches,
                      'request_errors': errors, 'responses_over_default_budget': over_budget}))
    return int(bool(errors or mismatches))


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
