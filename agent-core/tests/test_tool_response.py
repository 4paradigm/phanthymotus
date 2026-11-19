"""MCP business failures must not be presented as successful tool calls."""
import json

from tool_response import tool_failure


def wrap(value):
    return {"content": [{"type": "text", "text": json.dumps(value)}]}


def test_rejects_business_error_and_state_error():
    assert tool_failure(wrap({"ok": False, "error_code": "APPROVAL_REQUIRED",
                              "message": "需要新的开播授权"})) == "APPROVAL_REQUIRED: 需要新的开播授权"
    assert tool_failure(wrap({"state": "error", "error_code": "MIC_UNAVAILABLE"})) == "MIC_UNAVAILABLE"


def test_preserves_success_and_ignores_non_json():
    assert tool_failure(wrap({"ok": True, "state": "ready"})) is None
    assert tool_failure({"content": [{"type": "text", "text": "plain text"}]}) is None
    assert tool_failure({"isError": True}) == "MCP_TOOL_ERROR"


def test_error_text_is_bounded():
    detail = "X" * 1000
    error = tool_failure(wrap({"ok": False, "error_code": "DENIED", "message": detail}))
    assert len(error) < 270
