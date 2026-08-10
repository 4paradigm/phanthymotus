"""Strict helpers for the MCP 2024-11-05 HTTP/JSON-RPC boundary."""

from __future__ import annotations

import json


def content_block_valid(item: object) -> bool:
    """Validate one MCP 2024-11-05 ContentBlock without interpreting it."""

    if not isinstance(item, dict):
        return False
    block_type = item.get('type')
    if block_type == 'text':
        return isinstance(item.get('text'), str)
    if block_type == 'image':
        return (
            isinstance(item.get('data'), str)
            and isinstance(item.get('mimeType'), str)
            and bool(item['mimeType'])
        )
    if block_type == 'resource':
        resource = item.get('resource')
        if not isinstance(resource, dict) or not isinstance(resource.get('uri'), str):
            return False
        text = resource.get('text')
        blob = resource.get('blob')
        return (isinstance(text, str) and blob is None) or (
            isinstance(blob, str) and text is None
        )
    return False


def _reject_non_finite(value: str):
    raise ValueError(f'non-finite JSON constant: {value}')


def tool_result_error(
    result: object,
    *,
    require_structured_ack: bool = False,
) -> str | None:
    """Validate CallToolResult and optionally require a JSON-object ack."""

    if not isinstance(result, dict):
        return 'Driver returned an invalid tool result'
    content = result.get('content')
    if not isinstance(content, list) or any(
        not content_block_valid(item) for item in content
    ):
        return 'Driver returned an invalid tool result'
    is_error = result.get('isError', False)
    if not isinstance(is_error, bool):
        return 'Driver returned an invalid tool result'

    parsed_items: list[dict] = []
    text_messages: list[str] = []
    for item in content:
        if item.get('type') != 'text':
            continue
        text = item['text']
        try:
            parsed = json.loads(text, parse_constant=_reject_non_finite)
        except (json.JSONDecodeError, ValueError, RecursionError):
            if text:
                text_messages.append(text[:200])
            continue
        if isinstance(parsed, dict):
            parsed_items.append(parsed)

    if is_error:
        for parsed in parsed_items:
            message = parsed.get('message') or parsed.get('error')
            if isinstance(message, str) and message:
                return message[:200]
        if text_messages:
            return text_messages[0]
        return 'Driver reported a tool error'

    if require_structured_ack:
        if not parsed_items:
            return 'Driver did not return a structured acknowledgement'
        for parsed in parsed_items:
            explicit_error = parsed.get('error')
            if explicit_error not in (None, '', False):
                return (
                    explicit_error[:200]
                    if isinstance(explicit_error, str)
                    else 'Driver returned an explicit error'
                )
            if parsed.get('adapter_ok') is False:
                message = parsed.get('message')
                return (
                    message[:200]
                    if isinstance(message, str) and message
                    else 'Driver rejected the configuration'
                )
            if parsed.get('state') == 'error':
                message = parsed.get('message')
                return (
                    message[:200]
                    if isinstance(message, str) and message
                    else 'Driver reported an error state'
                )
            if (
                parsed.get('ok') is False
                or parsed.get('success') is False
                or parsed.get('configured') is False
                or (
                    isinstance(parsed.get('status'), str)
                    and parsed['status'] in {'error', 'failed', 'failure'}
                )
            ):
                message = parsed.get('message')
                return (
                    message[:200]
                    if isinstance(message, str) and message
                    else 'Driver rejected the operation'
                )
    return None


def rpc_response_error(
    payload: object,
    *,
    request_id: int | str,
    http_status: int,
) -> str | None:
    """Return a safe error for an invalid or failed JSON-RPC response."""

    if http_status < 200 or http_status >= 300:
        return f'Driver returned HTTP {http_status}'
    if not isinstance(payload, dict):
        return 'Driver returned an invalid JSON-RPC response'
    if payload.get('jsonrpc') != '2.0':
        return 'Driver returned an invalid JSON-RPC version'
    response_id = payload.get('id')
    if type(response_id) is not type(request_id) or response_id != request_id:
        return 'Driver returned a mismatched JSON-RPC id'
    has_result = 'result' in payload
    has_error = 'error' in payload
    if has_result == has_error:
        return 'Driver returned an invalid JSON-RPC envelope'
    if has_error:
        error = payload.get('error')
        if not isinstance(error, dict):
            return 'Driver returned an invalid JSON-RPC error'
        message = error.get('message')
        return (
            message[:200]
            if isinstance(message, str) and message
            else 'Driver tool call failed'
        )
    return None
