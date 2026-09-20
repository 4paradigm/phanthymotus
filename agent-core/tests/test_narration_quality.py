"""定时播报说什么、怎么收尾。

这段话每 15 秒才有一次，用户看不到屏幕，只能靠它知道事情在不在往前走。三件事在真机
上出过问题：编出来的时长、被砍成半截的名字、以及把十几秒前做完的事说成刚刚完成。

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_narration_quality.py -q
"""

import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'narration-test.db'))

# `event.llm` 这个名字被包里的单例实例占了（event/__init__.py 里 `llm = Event()`），
# 所以 `import event.llm as x` 拿到的是实例。模块级的函数要从模块里直接导。
from event.llm import (  # noqa: E402
    _NARRATION_MAX_CHARS, _NARRATION_PROMPT, _trim_narration,
)


# ── 收尾：不从字中间砍 ────────────────────────────────────────────────────────

def test_a_short_line_is_left_alone():
    assert _trim_narration('正在前往P3展区。') == '正在前往P3展区。'


def test_an_overlong_line_ends_at_a_sentence_break():
    """真机上把展位名砍成半截播了出去 —— 听的人只会以为机器人坏了。"""
    text = '第一家店关门了。' + '我正在查附近还有哪几家开着，顺便看了一下营业时间。' * 8

    trimmed = _trim_narration(text, limit=60)

    assert trimmed.endswith(('。', '，'))
    assert len(trimmed) <= 60


def test_a_name_is_never_cut_in_half():
    text = '我们马上到算力工厂展区，' * 6

    trimmed = _trim_narration(text, limit=50)

    assert not trimmed.endswith('算力工') and not trimmed.endswith('算力')


def test_a_line_with_no_punctuation_still_gets_trimmed():
    """找不到句读也不能无限长 —— 它会变成一个同样长的 ACP 超时。"""
    trimmed = _trim_narration('啊' * 400, limit=60)

    assert len(trimmed) <= 60


def test_the_cap_is_wide_enough_to_hold_a_real_sentence():
    """上限是 ACP 超时的来源，但砍得太狠就会砍掉名字。"""
    assert _NARRATION_MAX_CHARS >= 200


# ── 说什么 ────────────────────────────────────────────────────────────────────

def test_the_prompt_leads_with_what_happens_next():
    """这句话是承上启下：用户要知道的是接下来会怎样，不是复述刚才。"""
    prompt = _NARRATION_PROMPT

    assert '承上启下' in prompt
    assert prompt.index('正在做什么') < prompt.index('已完成')


def test_the_prompt_gives_the_duration_instead_of_forbidding_it():
    """一开始的修法是「不许说时长」。但导览里「还要等多久」正是用户想听的 ——
    所以改成由框架把真数交给它，并禁止它自己算。"""
    assert '当前等待' in _NARRATION_PROMPT
    assert '{waiting}' in _NARRATION_PROMPT


def test_the_prompt_does_not_trade_names_for_brevity():
    assert '不要为了短而把名字' in _NARRATION_PROMPT


# ── 「等了多久」这个数 ────────────────────────────────────────────────────────

def test_the_waiting_number_comes_from_the_framework(monkeypatch):
    """天轶上播出过「大约等了 25 秒」，而那个数没有任何依据：播报这次调用里既没有
    当前时刻，也没有 pending 列表，过程记录里只有几条历史时间戳。让模型自己减，
    减出来的就是这种数。所以框架算好给它。"""
    import event.llm as _m
    import mcp_client
    module = sys.modules['event.llm']
    monkeypatch.setattr(module, '_silence_since', __import__('time').time() - 18)
    monkeypatch.setattr(mcp_client, 'pending_waits',
                        lambda: [{'tool': 'controlled_spatial', 'seconds': 25, 'timeout': 180}])

    facts = module._waiting_facts()

    assert '距你上次开口：18 秒' in facts
    assert '正在等 controlled_spatial 完成：已等 25 秒' in facts


def test_nothing_pending_says_so_rather_than_guessing(monkeypatch):
    """没在等就明说没有，不给模型留一个可以脑补的空位。"""
    import mcp_client
    module = sys.modules['event.llm']
    monkeypatch.setattr(module, '_silence_since', None)
    monkeypatch.setattr(mcp_client, 'pending_waits', lambda: [])

    assert '没有在等任何东西' in module._waiting_facts()


def test_the_prompt_forbids_computing_durations_itself():
    assert '一个字都不要自己算' in _NARRATION_PROMPT
    assert '{waiting}' not in _NARRATION_PROMPT or '当前等待' in _NARRATION_PROMPT


def test_a_pending_records_when_it_started():
    """先前没有任何地方记这个 —— 所以「已经等了多久」根本无从算起。"""
    import mcp_client

    assert hasattr(mcp_client, '_pending_started')
    assert callable(mcp_client.pending_waits)
