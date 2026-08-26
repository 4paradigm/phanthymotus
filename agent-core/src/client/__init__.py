"""
client/__init__.py — LLM client singleton with unified usage tracking.

All LLM calls should go through `client.call()` to ensure usage is recorded.
"""

import client.llm
import perf_log
import llm_logger as _llm_logger
from uuid import uuid4

# Raw client instance (for direct access if needed)
llm = client.llm.Client()


def _get_logger_safely():
    """LLMLogger, or None if it cannot be constructed.

    `get_logger()` only caches the singleton on success, so anything that makes
    __init__ raise repeats on every single call. That is how a truncated
    `llm_request_*.jsonl` took the whole agent loop down.
    """
    try:
        return _llm_logger.get_logger()
    except Exception as e:
        print(f'[client] LLM logging unavailable ({type(e).__name__}: {e}) — continuing without it')
        return None


async def _log_safely(coro, what: str) -> None:
    """Await a logging coroutine, swallowing (but reporting) any failure."""
    try:
        await coro
    except Exception as e:
        print(f'[client] {what} failed ({type(e).__name__}: {e}) — ignored')


async def call(
    message_list: list[dict],
    tool_list: list[dict],
    cancel_event=None,
    model_override: str | None = None,
    trace_id: str = '',
    caller_info: dict | None = None,
) -> dict:
    """Unified LLM call with automatic usage recording.

    All components (main agent, subagent, etc.) should use this function
    instead of calling client.llm directly, to ensure token usage is tracked.

    Args:
        message_list: Messages for the LLM
        tool_list: Available tools
        cancel_event: Optional asyncio.Event to cancel the call
        model_override: Optional model name override
        trace_id: Optional trace ID for usage attribution
        caller_info: Optional caller metadata {'agent_type': 'main_agent'|'subagent'}

    Returns:
        LLM response dict (with _usage field attached)
    """
    import config
    request_id = str(uuid4())

    # Resolve model name for logging
    configs = config.main.get('client', {}).get('llm', [])
    model_name = model_override or (configs[0]['model'] if configs else 'unknown')

    # Log request. Diagnostics must never be able to stop the agent loop — a
    # corrupt JSONL file used to raise here and kill every turn before the HTTP
    # call was even attempted.
    logger = _get_logger_safely()
    if logger is not None:
        await _log_safely(
            logger.log_request(request_id, trace_id, caller_info,
                               message_list, tool_list, model_name),
            'log_request')

    response = await llm(
        message_list=message_list,
        tool_list=tool_list,
        cancel_event=cancel_event,
        model_override=model_override,
    )

    # Log response
    if logger is not None:
        await _log_safely(
            logger.log_response(request_id, trace_id, caller_info, response),
            'log_response')

    # Record usage
    usage = response.get('_usage')
    if usage:
        try:
            perf_log.record_usage(trace_id or 'unknown', usage)
        except Exception as e:
            print(f'[client] usage recording failed ({type(e).__name__}: {e}) — ignored')

    return response
