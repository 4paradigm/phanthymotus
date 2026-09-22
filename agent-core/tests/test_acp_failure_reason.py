"""失败的动作要带着原因到 LLM 面前。

真机上一次导航失败，送进 LLM 的只有

    {"type": "action_complete", "action_id": "navi_fab254c049c7", "status": "failed"}

「目标丢失」和「被挡住十几秒动不了」在这行字里长得一模一样，而这两件事该做的下一步
完全相反：前者要转头重新找，后者要换条路或者叫人。驱动一直把原因放在 `result.reason`
里（`actucore/plugins/navi/plugin.py` 的 `_maybe_complete`），是 collector 在送进 LLM
之前把整个 `result` 删掉了 —— 注释写的理由是「LLM 已知自己发出了什么」，对回声字段
成立，对 `reason` 不成立。
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import collector  # noqa: E402


def _slim(payload: dict) -> dict:
    text = collector._slim_event_text(json.dumps(payload, ensure_ascii=False))
    return json.loads(text)


def test_a_failed_action_keeps_the_reason():
    out = _slim({'type': 'action_complete', 'action_id': 'navi_fab254c049c7',
                 'status': 'failed',
                 'result': {'status': 'failed', 'target': 'chair',
                            'reason': "swept 6.3 rad without seeing 'chair': target lost"}})

    assert 'target lost' in out['result']['reason']


def test_two_different_failures_do_not_look_alike():
    """这条才是问题本身：两种失败在 LLM 眼里必须不一样。"""
    lost = _slim({'type': 'action_complete', 'action_id': 'a', 'status': 'failed',
                  'result': {'reason': "swept 6.3 rad without seeing 'chair': target lost"}})
    stuck = _slim({'type': 'action_complete', 'action_id': 'b', 'status': 'failed',
                   'result': {'reason': 'commanded 0.30 m/s for 12.0s but '
                                        'measured 0.01 m/s: blocked, stopping'}})

    assert lost['result']['reason'] != stuck['result']['reason']


def test_a_successful_action_still_drops_the_echo():
    """成功那一份基本是回声，LLM 知道自己发出了什么 —— 原来的精简是对的，保留。"""
    out = _slim({'type': 'action_complete', 'action_id': 'navi_1', 'status': 'completed',
                 'result': {'status': 'arrived', 'target': 'chair'}})

    assert 'result' not in out


def test_a_successful_action_keeps_the_quantities_it_could_not_know():
    """走了多远、用了多久不是回声 —— LLM 发出的是「去 A」，这些它猜不到，
    而下一步要不要调整节奏就看这个。"""
    out = _slim({'type': 'action_complete', 'action_id': 'navi_1', 'status': 'completed',
                 'result': {'target': 'chair', 'elapsed_s': 31.2, 'distance_m': 12.4}})

    assert out['result'] == {'elapsed_s': 31.2, 'distance_m': 12.4}


def test_an_overlong_reason_is_truncated_and_says_so():
    """不写明截断，LLM 会把半句话当成全部。"""
    out = _slim({'type': 'action_complete', 'action_id': 'a', 'status': 'error',
                 'result': {'reason': 'x' * 5000}})

    assert 'truncated' in out['result']
    assert len(out['result']['head']) == collector._ACP_RESULT_CHARS


def test_cancelled_counts_as_a_non_success():
    """被取消也不是成功 —— 「为什么被取消」同样只有驱动知道。"""
    out = _slim({'type': 'action_complete', 'action_id': 'a', 'status': 'cancelled',
                 'result': {'reason': 'superseded by new navigate'}})

    assert out['result']['reason'] == 'superseded by new navigate'
