"""Canvas-scoped ROS topic to MCP action dispatch.

Tools opt in with an ``inputSchema.x-topic-actions`` declaration.  A
declaration is activated only while the Canvas project is running and only
when the declared input port has an actual Canvas connection.
"""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import jsonschema

import config
import mcp_client
import ros2_bridge


_MAX_SEEN_IDS = 1024
_LOG_SAMPLE_EVERY = 100


@dataclass(frozen=True)
class TopicActionRoute:
    key: str
    topic: str
    fmt: str
    mcp_id: str
    tool_name: str
    instance_id: str
    port: str
    action: str
    schema: str
    id_field: str
    allowed_fields: tuple[str, ...]
    required_fields: tuple[str, ...]
    input_schema: dict[str, Any]


@dataclass
class RouteRuntime:
    route: TopicActionRoute
    seen_ids: OrderedDict[str, None] = field(default_factory=OrderedDict)
    dispatch_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    received: int = 0
    dispatched: int = 0
    duplicates: int = 0
    invalid: int = 0
    errors: int = 0
    last_error: str | None = None
    log_outcome: str | None = None
    log_occurrence: int = 0


def _log_hot_path(runtime: RouteRuntime, outcome: str, detail: object) -> None:
    if outcome != runtime.log_outcome:
        runtime.log_outcome = outcome
        runtime.log_occurrence = 1
    else:
        runtime.log_occurrence += 1
    if runtime.log_occurrence != 1 and runtime.log_occurrence % _LOG_SAMPLE_EVERY:
        return
    safe_detail = str(detail).encode("unicode_escape").decode("ascii")[:200]
    print(
        f"[topic-actions] {outcome} count={runtime.log_occurrence}: "
        f"{safe_detail}"
    )


def _tool_definition(mcp_id: str, tool_name: str) -> dict[str, Any] | None:
    """Return the raw MCP tool definition, preferring the live registry."""

    live_tools = mcp_client.registry.get(mcp_id, {}).get("tool_definitions", [])
    for tool in live_tools:
        if isinstance(tool, dict) and tool.get("name") == tool_name:
            return tool

    mcps = config.main.get("services", {}).get("mcp", [])
    mcp = next((item for item in mcps if item.get("id") == mcp_id), None)
    for tool in (mcp or {}).get("tools", []):
        if isinstance(tool, dict) and tool.get("name") == tool_name:
            return tool
    return None


def _connection_topic(connection: dict, cards_by_id: dict[str, dict]) -> str:
    topic = str(connection.get("fromTopic") or "").strip()
    if topic:
        return topic

    source = cards_by_id.get(connection.get("fromCardId"), {})
    outputs = source.get("topicOut") or []
    try:
        port_idx = int(connection.get("fromPortIdx", 0))
    except (TypeError, ValueError):
        return ""
    if 0 <= port_idx < len(outputs):
        return str(outputs[port_idx].get("topic") or "").strip()
    return ""


def build_routes(layout: dict[str, Any]) -> list[TopicActionRoute]:
    """Resolve active topic-action declarations from the saved Canvas graph."""

    cards = layout.get("cards", [])
    cards_by_id = {card.get("id"): card for card in cards if card.get("id")}
    routes: list[TopicActionRoute] = []
    route_keys: set[str] = set()

    for connection in layout.get("connections", []):
        target = cards_by_id.get(connection.get("toCardId"))
        if not target:
            continue

        mcp_id = str(target.get("mcpId") or "")
        tool_name = str(target.get("toolName") or "")
        tool = _tool_definition(mcp_id, tool_name)
        input_schema = (tool or {}).get("inputSchema") or {}
        declarations = input_schema.get("x-topic-actions")
        if not isinstance(declarations, list) or not declarations:
            continue

        topic_inputs = (tool or {}).get("topic_in") or target.get("topicIn") or []
        try:
            target_port_idx = int(connection.get("toPortIdx", 0))
        except (TypeError, ValueError):
            continue
        if not 0 <= target_port_idx < len(topic_inputs):
            continue

        topic_input = topic_inputs[target_port_idx]
        port = str(topic_input.get("port") or "")
        declaration = next(
            (
                item
                for item in declarations
                if isinstance(item, dict) and item.get("port") == port
            ),
            None,
        )
        if declaration is None:
            continue

        topic = _connection_topic(connection, cards_by_id)
        if not topic:
            topic = str(topic_input.get("topic") or "").strip()
        fmt = str(connection.get("format") or topic_input.get("format") or "")
        if not topic:
            raise ValueError(f"x-topic-actions port {port!r} has no resolved topic")
        if fmt not in ("json", "data/json"):
            raise ValueError(
                f"x-topic-actions port {port!r} requires data/json, got {fmt!r}"
            )

        action = declaration.get("action")
        schema = declaration.get("schema")
        id_field = declaration.get("id_field")
        allowed_fields = declaration.get("allowed_fields")
        required_fields = declaration.get("required_fields", [])
        if not all(isinstance(value, str) and value for value in (action, schema, id_field)):
            raise ValueError(f"invalid x-topic-actions declaration for port {port!r}")
        if not isinstance(allowed_fields, list) or not all(
            isinstance(value, str) and value for value in allowed_fields
        ):
            raise ValueError(f"invalid allowed_fields for x-topic-actions port {port!r}")
        if not isinstance(required_fields, list) or not all(
            isinstance(value, str) and value in allowed_fields
            for value in required_fields
        ):
            raise ValueError(f"invalid required_fields for x-topic-actions port {port!r}")

        key = f"__topic_action__#{target['id']}#{port}"
        if key in route_keys:
            raise ValueError(f"multiple Canvas connections target topic-action port {port!r}")
        route_keys.add(key)
        routes.append(
            TopicActionRoute(
                key=key,
                topic=topic,
                fmt=fmt,
                mcp_id=mcp_id,
                tool_name=tool_name,
                instance_id=str(target["id"]),
                port=port,
                action=action,
                schema=schema,
                id_field=id_field,
                allowed_fields=tuple(allowed_fields),
                required_fields=tuple(required_fields),
                input_schema=input_schema,
            )
        )

    return routes


class TopicActionManager:
    def __init__(self) -> None:
        self._runtimes: dict[str, RouteRuntime] = {}
        self._inflight: set[asyncio.Task] = set()

    async def start(self, layout: dict[str, Any]) -> None:
        await self.stop()
        loop = asyncio.get_running_loop()
        for route in build_routes(layout):
            runtime = RouteRuntime(route=route)
            self._runtimes[route.key] = runtime

            async def _callback(data: bytes, _fmt: str, *, key: str = route.key) -> None:
                await self._handle(key, data)

            ros2_bridge.subscribe(route.key, route.topic, route.fmt, loop, _callback)
            print(
                f"[topic-actions] active topic={route.topic} "
                f"target={route.mcp_id}/{route.tool_name}:{route.action}"
            )

    async def stop(self) -> None:
        runtimes = list(self._runtimes.values())
        self._runtimes.clear()
        for runtime in runtimes:
            ros2_bridge.unsubscribe(runtime.route.key)
        current = asyncio.current_task()
        inflight = [task for task in self._inflight if task is not current]
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)
        if runtimes:
            print(f"[topic-actions] stopped {len(runtimes)} route(s)")

    async def _handle(self, key: str, data: bytes) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._inflight.add(task)
        try:
            runtime = self._runtimes.get(key)
            if runtime is None:
                return
            runtime.received += 1

            if not config.main.get("core", {}).get("project_running", False):
                return

            try:
                payload = json.loads(
                    data.decode("utf-8"),
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"invalid JSON constant {value}")
                    ),
                )
                args, message_id = self._validate(runtime.route, payload)
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
                runtime.invalid += 1
                runtime.last_error = str(exc)
                _log_hot_path(runtime, "rejected", exc)
                return

            async with runtime.dispatch_lock:
                if self._runtimes.get(key) is not runtime:
                    return
                if message_id in runtime.seen_ids:
                    runtime.duplicates += 1
                    return
                try:
                    from api.mcp_manage import MCPCallRequest, mcp_call_tool

                    result = await mcp_call_tool(
                        runtime.route.mcp_id,
                        MCPCallRequest(tool=runtime.route.tool_name, arguments=args),
                    )
                    if result.get("code") != 200:
                        raise RuntimeError(
                            result.get("message")
                            or result.get("detail")
                            or "MCP call failed"
                        )
                    runtime.dispatched += 1
                    runtime.seen_ids[message_id] = None
                    while len(runtime.seen_ids) > _MAX_SEEN_IDS:
                        runtime.seen_ids.popitem(last=False)
                    runtime.last_error = None
                    _log_hot_path(
                        runtime,
                        "dispatched",
                        f"target={runtime.route.mcp_id}/{runtime.route.tool_name}:"
                        f"{runtime.route.action}",
                    )
                except Exception as exc:
                    runtime.errors += 1
                    runtime.last_error = str(exc)
                    _log_hot_path(runtime, "dispatch_failed", exc)
        finally:
            if task is not None:
                self._inflight.discard(task)

    @staticmethod
    def _validate(route: TopicActionRoute, payload: Any) -> tuple[dict[str, Any], str]:
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
        if payload.get("schema") != route.schema:
            raise ValueError(f"schema must be {route.schema!r}")

        message_id = payload.get(route.id_field)
        if not isinstance(message_id, str) or not message_id.strip():
            raise ValueError(f"{route.id_field} must be a non-empty string")
        if len(message_id) > 256:
            raise ValueError(f"{route.id_field} exceeds 256 characters")

        permitted = {"schema", route.id_field, *route.allowed_fields}
        unknown = sorted(set(payload) - permitted)
        if unknown:
            raise ValueError(f"unsupported fields: {', '.join(unknown)}")
        missing = [field for field in route.required_fields if field not in payload]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")

        args = {
            "action": route.action,
            "instance_id": route.instance_id,
            **{field: payload[field] for field in route.allowed_fields if field in payload},
        }
        if route.input_schema:
            jsonschema.validate(instance=args, schema=route.input_schema)
        return args, message_id

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            key: {
                "topic": runtime.route.topic,
                "target": (
                    f"{runtime.route.mcp_id}/{runtime.route.tool_name}:"
                    f"{runtime.route.action}"
                ),
                "received": runtime.received,
                "dispatched": runtime.dispatched,
                "duplicates": runtime.duplicates,
                "invalid": runtime.invalid,
                "errors": runtime.errors,
                "last_error": runtime.last_error,
            }
            for key, runtime in self._runtimes.items()
        }


manager = TopicActionManager()
