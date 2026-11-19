"""Interpret structured MCP tool failures without exposing raw configuration."""

import json


def tool_failure(result: dict) -> str | None:
    """Return a bounded operator-facing error for a failed MCP tool result."""
    if not isinstance(result, dict):
        return None
    for item in result.get("content") or []:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        try:
            payload = json.loads(item.get("text", ""))
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("ok") is False or payload.get("state") == "error":
            code = str(payload.get("error_code") or "TOOL_REJECTED")[:80]
            message = str(payload.get("message") or "")[:240]
            return f"{code}: {message}" if message else code
    if result.get("isError"):
        return "MCP_TOOL_ERROR"
    return None
