"""Regression: a truncated JSONL must not stop the agent loop.

Observed on R1 (2026-08-15 → 2026-08-19): an unclean shutdown left
`llm_request_20260815_222250.jsonl` ending mid UTF-8 sequence (last byte 0xe5,
size an exact multiple of 4096 — the tail page was never flushed).
`_resume_current_files()` iterates that file in __init__, so it raised
`UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe5 in position 0:
unexpected end of data`. Because `get_logger()` only caches on success, every
LLM call re-ran __init__ and re-raised — 0 successful turns for four days,
failing *before* the HTTP request, which is why no `[llm]` line ever appeared.

Run: cd agent-core && python3 -m pytest tests/test_llm_logger_resume.py
"""
import json
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

# `config` opens (and seeds) a SQLite DB at import time — keep that out of the repo.
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import llm_logger  # noqa: E402


@pytest.fixture
def logger_dirs(tmp_path, monkeypatch):
    data = tmp_path / 'llm_data'
    data.mkdir()
    monkeypatch.setattr(llm_logger, '_instance', None, raising=False)
    cfg = {'enabled': True, 'data_dir': str(data), 'recent_dir': str(tmp_path / 'recent'),
           'batch_size': 500, 'max_records': 50000, 'recent_max_per_dir': 100}
    monkeypatch.setattr(llm_logger, '_get_config', lambda: dict(cfg))
    return data, cfg


def _write_truncated(path: pathlib.Path) -> None:
    """A valid line, then a line cut mid 3-byte CJK character."""
    path.write_bytes(b'{"a": 1}\n{"b": "\xe4\xbd\xa0\xe5')


def test_truncated_request_file_is_repaired_and_resumed(logger_dirs):
    data, _ = logger_dirs
    bad = data / 'llm_request_20260815_222250.jsonl'
    _write_truncated(bad)

    # Precondition: the naive count really does raise on this file.
    with pytest.raises(UnicodeDecodeError):
        with bad.open('r', encoding='utf-8') as f:
            sum(1 for _ in f)

    lg = llm_logger.LLMLogger()
    assert lg._req_file == bad, 'a repaired file is valid JSONL and safe to append to'
    assert lg._req_count == 1, 'only the one complete record survives'
    assert bad.read_bytes() == b'{"a": 1}\n', 'partial tail must be gone from disk'


def test_partial_tail_that_decodes_but_is_not_json_is_dropped(logger_dirs):
    """The commoner truncation: cut at an ASCII byte. Decodes fine, isn't JSON."""
    data, _ = logger_dirs
    bad = data / 'llm_request_20260815_222250.jsonl'
    bad.write_bytes(b'{"a": 1}\n{"b": 2\n')  # note: terminated, but unparseable

    lg = llm_logger.LLMLogger()
    assert lg._req_count == 1
    assert bad.read_bytes() == b'{"a": 1}\n'


def test_file_with_no_newline_at_all_is_emptied(logger_dirs):
    data, _ = logger_dirs
    bad = data / 'llm_request_20260815_222250.jsonl'
    bad.write_bytes(b'{"a": 1')

    lg = llm_logger.LLMLogger()
    assert lg._req_count == 0
    assert bad.read_bytes() == b''


def test_intact_request_file_is_untouched(logger_dirs):
    data, _ = logger_dirs
    good = data / 'llm_request_20260815_222250.jsonl'
    original = '{"a": 1}\n{"b": "你好"}\n'
    good.write_text(original, encoding='utf-8')

    lg = llm_logger.LLMLogger()
    assert lg._req_file == good
    assert lg._req_count == 2
    assert good.read_text(encoding='utf-8') == original, 'must not rewrite a healthy file'


def test_full_file_is_not_resumed(logger_dirs):
    """batch_size reached → rotate to a new file rather than append."""
    data, cfg = logger_dirs
    full = data / 'llm_request_20260815_222250.jsonl'
    full.write_text('{"a": 1}\n' * cfg['batch_size'], encoding='utf-8')

    lg = llm_logger.LLMLogger()
    assert lg._req_file is None


def test_truncated_response_file_is_repaired(logger_dirs):
    data, _ = logger_dirs
    bad = data / 'llm_response_20260815_222250.jsonl'
    _write_truncated(bad)

    lg = llm_logger.LLMLogger()
    assert lg._resp_count == 1
    assert bad.read_bytes() == b'{"a": 1}\n'


def test_get_logger_survives_a_corrupt_file(logger_dirs):
    """The whole point: construction must succeed, so the singleton caches."""
    data, _ = logger_dirs
    _write_truncated(data / 'llm_request_20260815_222250.jsonl')

    first = llm_logger.get_logger()
    assert first is not None
    assert llm_logger.get_logger() is first, 'singleton must be cached, not rebuilt each call'


def test_appended_record_is_one_line_even_with_embedded_newline(logger_dirs):
    """One record must be one line, or the file stops being parseable JSONL."""
    data, _ = logger_dirs
    lg = llm_logger.LLMLogger()
    target = data / 'llm_request_manual.jsonl'

    lg._append_line(target, '{"a": "x\ny"}')

    raw = target.read_bytes()
    assert raw.count(b'\n') == 1
    assert raw.endswith(b'\n')


def test_repair_then_append_yields_a_fully_parseable_file(logger_dirs):
    """End to end: corrupt tail in, valid JSONL out."""
    data, _ = logger_dirs
    bad = data / 'llm_request_20260815_222250.jsonl'
    _write_truncated(bad)

    lg = llm_logger.LLMLogger()
    lg._append_request(json.dumps({'c': '过'}, ensure_ascii=False))

    lines = bad.read_text(encoding='utf-8').splitlines()
    assert [json.loads(ln) for ln in lines] == [{'a': 1}, {'c': '过'}]
    assert lg._req_count == 2
