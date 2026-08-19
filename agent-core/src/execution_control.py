"""Generic manual MCP execution-control orchestration.

The controller consumes tool metadata rather than robot-specific code.  It is
used only by the HTTP endpoint behind a manual Canvas tool call; internal
callers such as topic actions continue to call the raw MCP helper directly.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable


RawCall = Callable[[str, str, dict], Awaitable[dict]]


@dataclass(frozen=True)
class _Lease:
    nav_id: str
    target_mcp_id: str
    target_tool: str
    revoke_action: str
    nav_id_argument: str


def _content_payload(response: dict) -> dict:
    data = response.get("data")
    if isinstance(data, dict):
        return data
    if not isinstance(data, list):
        return {}
    for item in data:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        try:
            parsed = json.loads(item.get("text", ""))
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _response_failed(response: dict) -> bool:
    if response.get("code") != 200:
        return True
    payload = _content_payload(response)
    status = str(payload.get("status") or payload.get("state") or "").lower()
    if status in {"error", "failed", "rejected"}:
        return True
    if payload.get("error_code"):
        return True
    return bool(payload.get("error"))


def _domain_error(action: str, error_code: str, message: str) -> dict:
    payload = {
        "action": action,
        "status": "error",
        "error_code": error_code,
        "error": message,
    }
    return {
        "code": 200,
        "data": [
            {
                "type": "text",
                "text": json.dumps(payload, ensure_ascii=False),
            }
        ],
    }


def _response_error_message(response: dict) -> str:
    payload = _content_payload(response)
    return str(
        payload.get("error")
        or payload.get("message")
        or response.get("message")
        or "execution target rejected the request"
    )


class ManualExecutionController:
    """Authorize one manual action before forwarding it to its source tool."""

    def __init__(self) -> None:
        self._leases: dict[tuple[str, str, str], _Lease] = {}
        self._locks: dict[tuple[str, str, str], asyncio.Lock] = {}

    @staticmethod
    def _source_tool(mcps: list, mcp_id: str, tool_name: str) -> dict | None:
        source = next((item for item in mcps if item.get("id") == mcp_id), None)
        if not source:
            return None
        return next(
            (
                tool
                for tool in source.get("tools", [])
                if isinstance(tool, dict) and tool.get("name") == tool_name
            ),
            None,
        )

    @staticmethod
    def _proposal_output(source_tool: dict, control: dict) -> dict | None:
        output_port = control.get("output_port")
        return next(
            (
                item
                for item in source_tool.get("topic_out", [])
                if isinstance(item, dict) and item.get("port") == output_port
            ),
            None,
        )

    @staticmethod
    def _execution_target(
        mcps: list,
        *,
        target_tool: str,
        proposal_topic: str,
        proposal_schema: str,
    ) -> tuple[str, dict] | None:
        matches: list[tuple[str, dict]] = []
        for mcp in mcps:
            for tool in mcp.get("tools", []):
                if not isinstance(tool, dict) or tool.get("name") != target_tool:
                    continue
                for topic_in in tool.get("topic_in", []):
                    if not isinstance(topic_in, dict):
                        continue
                    if topic_in.get("topic") != proposal_topic:
                        continue
                    if topic_in.get("schema") != proposal_schema:
                        continue
                    matches.append((str(mcp.get("id", "")), tool))
                    break
        if len(matches) != 1 or not matches[0][0]:
            return None
        return matches[0]

    async def call_manual(
        self,
        *,
        mcps: list,
        source_mcp_id: str,
        source_tool_name: str,
        arguments: dict,
        raw_call: RawCall,
    ) -> dict:
        source_tool = self._source_tool(mcps, source_mcp_id, source_tool_name)
        control = (source_tool or {}).get("x-execution-control")
        action = str(arguments.get("action", ""))
        if not isinstance(control, dict):
            return await raw_call(source_mcp_id, source_tool_name, arguments)

        instance_id = str(arguments.get("instance_id", ""))
        source_key = (source_mcp_id, source_tool_name, instance_id)
        lock = self._locks.setdefault(source_key, asyncio.Lock())
        async with lock:
            if action in set(control.get("start_actions", [])):
                return await self._start(
                    mcps=mcps,
                    source_key=source_key,
                    source_mcp_id=source_mcp_id,
                    source_tool_name=source_tool_name,
                    source_tool=source_tool,
                    control=control,
                    action=action,
                    arguments=arguments,
                    raw_call=raw_call,
                )
            if action in set(control.get("stop_actions", [])):
                return await self._stop(
                    source_key=source_key,
                    source_mcp_id=source_mcp_id,
                    source_tool_name=source_tool_name,
                    action=action,
                    arguments=arguments,
                    raw_call=raw_call,
                )
            return await raw_call(source_mcp_id, source_tool_name, arguments)

    async def _start(
        self,
        *,
        mcps: list,
        source_key: tuple[str, str, str],
        source_mcp_id: str,
        source_tool_name: str,
        source_tool: dict,
        control: dict,
        action: str,
        arguments: dict,
        raw_call: RawCall,
    ) -> dict:
        proposal = self._proposal_output(source_tool, control)
        proposal_topic = str((proposal or {}).get("topic", ""))
        proposal_schema = str(
            control.get("proposal_schema") or (proposal or {}).get("schema", "")
        )
        target_tool_name = str(control.get("target_tool", ""))
        target = self._execution_target(
            mcps,
            target_tool=target_tool_name,
            proposal_topic=proposal_topic,
            proposal_schema=proposal_schema,
        )
        if not proposal_topic or not proposal_schema or target is None:
            return _domain_error(
                action,
                "execution_target_unavailable",
                "no unique execution target matches the proposal topic and schema",
            )

        target_mcp_id, _ = target
        nav_id = uuid.uuid4().hex
        authorize_action = str(
            control.get("authorize_action", "authorize_navigation")
        )
        revoke_action = str(control.get("revoke_action", "revoke_navigation"))
        nav_id_argument = str(control.get("nav_id_argument", "nav_id"))
        proposal_topic_argument = str(
            control.get("proposal_topic_argument", "proposal_topic")
        )
        proposal_schema_argument = str(
            control.get("proposal_schema_argument", "proposal_schema")
        )
        authorize_response = await raw_call(
            target_mcp_id,
            target_tool_name,
            {
                "action": authorize_action,
                nav_id_argument: nav_id,
                proposal_topic_argument: proposal_topic,
                proposal_schema_argument: proposal_schema,
            },
        )
        if _response_failed(authorize_response):
            return _domain_error(
                action,
                "execution_authorization_failed",
                _response_error_message(authorize_response),
            )

        source_arguments = dict(arguments)
        source_arguments[str(control.get("lease_argument", "_control_nav_id"))] = (
            nav_id
        )
        source_response = await raw_call(
            source_mcp_id, source_tool_name, source_arguments
        )
        if _response_failed(source_response):
            await raw_call(
                target_mcp_id,
                target_tool_name,
                {"action": revoke_action, nav_id_argument: nav_id},
            )
            return source_response

        self._leases[source_key] = _Lease(
            nav_id=nav_id,
            target_mcp_id=target_mcp_id,
            target_tool=target_tool_name,
            revoke_action=revoke_action,
            nav_id_argument=nav_id_argument,
        )
        return source_response

    async def _stop(
        self,
        *,
        source_key: tuple[str, str, str],
        source_mcp_id: str,
        source_tool_name: str,
        action: str,
        arguments: dict,
        raw_call: RawCall,
    ) -> dict:
        source_response = await raw_call(
            source_mcp_id, source_tool_name, arguments
        )
        lease = self._leases.get(source_key)
        if lease is None:
            return source_response
        revoke_response = await raw_call(
            lease.target_mcp_id,
            lease.target_tool,
            {
                "action": lease.revoke_action,
                lease.nav_id_argument: lease.nav_id,
            },
        )
        if not _response_failed(revoke_response):
            self._leases.pop(source_key, None)
        if _response_failed(source_response):
            return source_response
        if _response_failed(revoke_response):
            return _domain_error(
                action,
                "execution_revoke_failed",
                _response_error_message(revoke_response),
            )
        return source_response


__all__ = ["ManualExecutionController"]
