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

    # Log request
    logger = _llm_logger.get_logger()
    await logger.log_request(request_id, trace_id, caller_info, message_list, tool_list, model_name)

    response = await llm(
        message_list=message_list,
        tool_list=tool_list,
        cancel_event=cancel_event,
        model_override=model_override,
    )

    # Log response
    await logger.log_response(request_id, trace_id, caller_info, response)

    # Record usage
    usage = response.get('_usage')
    if usage:
        perf_log.record_usage(trace_id or 'unknown', usage)

    return response
