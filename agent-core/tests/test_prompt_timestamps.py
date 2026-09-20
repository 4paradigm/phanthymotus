"""送进 prompt 的时刻只能有一套钟。

真机上不是：`<status time=...>` 走北京时间，collector 里的 `<event ts=...>` 走裸
`fromtimestamp()`，而线上容器是 `TZ=Etc/UTC`。同一秒在同一段 prompt 里写成相差 8
小时的两个数，模型被问「这件事进行了多久」就拿它们相减 —— 定时播报里那句
「已经等了四分钟」「大约等了一分半」就是这么算出来的。

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_prompt_timestamps.py -q
"""

import datetime
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'ts-test.db'))

import collector  # noqa: E402
import prompt  # noqa: E402

# 2026-09-20 18:28:39 北京时间
MOMENT = datetime.datetime(2026, 9, 20, 18, 28, 39,
                           tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()


def test_an_event_and_the_status_line_agree_on_the_hour():
    """两个数并排出现在同一段 prompt 里。差 8 小时，模型就会拿它们做减法。"""
    status = prompt.format_ts(MOMENT, sep=' ')
    event = prompt.format_ts(MOMENT)

    assert status.startswith('2026-09-20 18:28:39')
    assert event.startswith('2026-09-20T18:28:39')


def test_the_formatter_does_not_follow_the_container_timezone(monkeypatch):
    """线上容器是 TZ=Etc/UTC —— 裸 `fromtimestamp()` 会跟着它走，这正是出事的那步。"""
    monkeypatch.setenv('TZ', 'Etc/UTC')
    try:
        import time as _time
        _time.tzset()
    except AttributeError:
        pass

    assert prompt.format_ts(MOMENT).endswith('18:28:39')


def test_the_batch_formatter_uses_that_one_formatter():
    """collector 自己格式化过时刻，于是两套钟并存。现在它只能从一处拿。"""
    batch = collector._format_priority_batch([
        {'source': 'dds:/mic/asr', 'text': '你好', 'ts': MOMENT},
    ])

    assert '18:28:39' in batch
    assert '10:28:39' not in batch


def test_the_background_batch_uses_it_too():
    text = collector._format_bg_batch([
        {'source': 'dds:/camera/objects', 'text': '{"count": 2}', 'ts': MOMENT},
    ])

    assert '18:28:39' in text
