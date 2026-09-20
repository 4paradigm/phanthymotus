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


def test_the_prompt_forbids_making_up_durations():
    """播报这条路拿不到任何时长 —— 17 个真机请求里没有一个带 active_tasks。
    所以播报里出现的秒数一定是编的。"""
    assert '不要说时长' in _NARRATION_PROMPT


def test_the_prompt_does_not_trade_names_for_brevity():
    assert '不要为了短而把名字' in _NARRATION_PROMPT
