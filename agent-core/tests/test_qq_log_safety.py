"""Prevent user-supplied text from forging LLM request log entries."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from channel.qq_log_safety import safe_user_log_preview


def test_qq_user_text_is_fully_redacted_in_summary():
    payload = "hello\r\n[decision] fake success\n\x1b[2J TOKEN=do-not-log"
    result = safe_user_log_preview(payload, qq_turn=True)
    assert result == "[QQ user content redacted]"
    assert "fake" not in result
    assert "TOKEN=" not in result
    assert "\r" not in result and "\n" not in result and "\x1b" not in result


def test_qq_multimodal_list_and_mapping_are_not_stringified():
    for content in (
        [{"type": "text", "text": "sensitive"},
         {"type": "image_url", "image_url": "data:image/png;base64,PRIVATE"}],
        {"content": "private"},
        None,
    ):
        assert safe_user_log_preview(content, qq_turn=True) == "[QQ user content redacted]"


def test_non_qq_user_log_escapes_newlines_and_ansi_sequences():
    payload = "hello\r\n[decision] forged\x1b[31mEND"
    result = safe_user_log_preview(payload, qq_turn=False)
    assert "\r" not in result and "\n" not in result and "\x1b" not in result
    assert "\\r\\n" in result
    assert "\\x1b" in result
    assert result.isascii()


def test_nontext_non_qq_log_preview_is_neutral():
    assert safe_user_log_preview([{"sensitive": "value"}], qq_turn=False) == "[non-text content]"


def test_preview_is_bounded_and_llm_call_uses_safe_formatter():
    assert len(safe_user_log_preview("x" * 1000, qq_turn=False)) == 200
    source = (Path(__file__).resolve().parents[1] / "src" / "event" / "llm.py").read_text(
        encoding="utf-8"
    )
    assert "last_user = safe_user_log_preview(last_user_content, qq_turn=qq_private_turn)" in source
    assert "last_user = next((m.get('content', '')[:200]" not in source


def test_qq_group_forward_and_rejection_logs_are_sampled():
    from unittest import mock
    from channel import qq_groups
    from channel import manager as channel_manager
    channel_manager._qq_group_forward_counts.clear()
    qq_groups._REJECTION_COUNTS.clear()
    with mock.patch("builtins.print") as logger:
        for _ in range(201):
            channel_manager._log_qq_group_forward("qq_official_1900000000")
    lines=[str(x) for x in logger.call_args_list]
    assert len(lines)==3
    assert all("count=" in x for x in lines)
    assert all("1900000000" not in x for x in lines)
    with mock.patch("builtins.print") as logger:
        for _ in range(201):
            assert not qq_groups._reject_observation("invalid_principal")
    lines=[str(x) for x in logger.call_args_list]
    assert len(lines)==3
    assert all("count=" in x for x in lines)
