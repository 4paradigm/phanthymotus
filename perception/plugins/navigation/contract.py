"""One public MCP contract for mapping, planning and semantic navigation."""

from __future__ import annotations

from copy import deepcopy

from .mapping.contract import (
    FAST_LIVO2_ACTION_PARAMS,
    FAST_LIVO2_ACTIONS,
    fast_livo2_tool_definition,
)
from .planning.contract import (
    NAV2_ACTION_PARAMS,
    NAV2_ACTIONS,
    nav2_tool_definition,
)
from .semantic.manifest import build_manifest


NAVIGATION_LIFECYCLE_ACTIONS = ("info", "config", "start", "stop")
SEMANTIC_ACTIONS = ("capture", "navigate")
NAVIGATION_ACTIONS = FAST_LIVO2_ACTIONS + NAV2_ACTIONS + SEMANTIC_ACTIONS
NAVIGATION_PUBLIC_ACTIONS = NAVIGATION_LIFECYCLE_ACTIONS + NAVIGATION_ACTIONS
CONTROLLED_SEMANTIC_SPATIAL_TOOL_NAME = "controlled_semantic_spatial"
NAVIGATION_PUBLIC_OUTPUT_PORTS = (
    "map_view",
    "status",
    "velocity_proposal",
    "plan",
    "costmap",
)


def _config_properties() -> dict:
    mapping = fast_livo2_tool_definition("ubuntu")["configSchema"]["properties"]
    planning = nav2_tool_definition("ubuntu")["configSchema"]["properties"]
    semantic = build_manifest(
        "/ubuntu/camera/rgb",
        "/ubuntu/navigation/odom",
    )["configSchema"]["properties"]
    result = {
        "mapping_request_timeout_sec": deepcopy(mapping["request_timeout_sec"]),
        "mapping_discovery_timeout_sec": deepcopy(mapping["discovery_timeout_sec"]),
        "map_voxel_size_m": deepcopy(mapping["map_voxel_size_m"]),
        "obstacle_min_height_m": deepcopy(mapping["obstacle_min_height_m"]),
        "obstacle_max_height_m": deepcopy(mapping["obstacle_max_height_m"]),
        "collection_enabled": deepcopy(mapping["collection_enabled"]),
        "collection_directory": deepcopy(mapping["collection_directory"]),
        "planning_request_timeout_sec": deepcopy(planning["request_timeout_sec"]),
        "planning_discovery_timeout_sec": deepcopy(planning["discovery_timeout_sec"]),
    }
    for key in (
        "min_x_mps",
        "max_x_mps",
        "min_y_mps",
        "max_y_mps",
        "min_yaw_rps",
        "max_yaw_rps",
    ):
        result[key] = deepcopy(planning[key])
    result.update(
        {
            "vlm_base_url": deepcopy(semantic["base_url"]),
            "vlm_api_key": deepcopy(semantic["api_key"]),
            "vlm_model": deepcopy(semantic["model"]),
            "vlm_timeout_sec": deepcopy(semantic["timeout_sec"]),
        }
    )
    return result


NAVIGATION_CONFIG_SCHEMA = {
    "type": "object",
    "properties": _config_properties(),
    "required": ["vlm_base_url", "vlm_api_key", "vlm_model"],
    "additionalProperties": False,
}


def _action_properties() -> dict:
    mapping = fast_livo2_tool_definition("ubuntu")["inputSchema"]["properties"]
    planning = nav2_tool_definition("ubuntu")["inputSchema"]["properties"]
    semantic = build_manifest(
        "/ubuntu/camera/rgb",
        "/ubuntu/navigation/odom",
    )["inputSchema"]["properties"]
    result = {"action": {"type": "string", "enum": list(NAVIGATION_PUBLIC_ACTIONS)}}
    business_params = {
        param
        for action in FAST_LIVO2_ACTIONS
        for param in FAST_LIVO2_ACTION_PARAMS[action]["params"]
    }
    business_params.update(
        param
        for action in NAV2_ACTIONS
        for param in NAV2_ACTION_PARAMS[action]["params"]
    )
    business_params.update({"query"})
    lifecycle_params = {"instance_id", "input_topic", "input_topics", "input_bindings"}
    for source in (mapping, planning, semantic):
        for key, value in source.items():
            if key in business_params or key in lifecycle_params:
                result.setdefault(key, deepcopy(value))
    for key, value in NAVIGATION_CONFIG_SCHEMA["properties"].items():
        result[key] = deepcopy(value)
    return result


NAVIGATION_ACTION_PARAMS = {
    **deepcopy(FAST_LIVO2_ACTION_PARAMS),
    **deepcopy(NAV2_ACTION_PARAMS),
    "capture": {
        "params": [],
        "description": "Capture a visual waypoint at the current map pose",
    },
    "navigate": {
        "params": ["query"],
        "description": "Match a visual waypoint and start Nav2 with the same lease",
    },
    "info": {"params": [], "description": "Read the unified navigation state"},
    "config": {
        "params": list(NAVIGATION_CONFIG_SCHEMA["properties"]),
        "description": "Configure mapping, planning and semantic navigation",
    },
    "start": {
        "params": ["instance_id", "input_topics", "input_bindings"],
        "description": "Start the in-container runtime and acquire card resources",
    },
    "stop": {
        "params": [],
        "description": "Stop navigation, mapping and every owned child process",
    },
}


def navigation_tool_definition(namespace: str) -> dict:
    mapping = fast_livo2_tool_definition(namespace)
    planning = nav2_tool_definition(namespace)
    semantic = build_manifest(
        f"/{namespace.strip('/')}/camera/rgb",
        f"/{namespace.strip('/')}/navigation/odom",
        f"/{namespace.strip('/')}/navigation/goal_pose",
        f"/{namespace.strip('/')}/navigation/fast_livo2/status",
    )
    external_inputs = [deepcopy(item) for item in mapping["topic_in"]]
    external_inputs.append(deepcopy(semantic["topic_in"][0]))
    goal_input = next(
        item for item in planning["topic_in"] if item["port"] == "goal_pose"
    )
    external_inputs.append(deepcopy(goal_input))

    component_outputs = {
        str(item.get("port", "")): item
        for item in [*mapping["topic_out"], *planning["topic_out"]]
        if item.get("port")
    }
    outputs = [
        deepcopy(component_outputs[port]) for port in NAVIGATION_PUBLIC_OUTPUT_PORTS
    ]

    return {
        "name": CONTROLLED_SEMANTIC_SPATIAL_TOOL_NAME,
        "displayName": CONTROLLED_SEMANTIC_SPATIAL_TOOL_NAME,
        "type": "processor",
        "multiInstance": False,
        "description": (
            "Unified G1 mapping, localization, semantic waypoint and Nav2 "
            "planner/controller card. FAST-LIVO2 and Nav2 run as child processes "
            "inside the Perception container; only bounded velocity proposals "
            "leave the card."
        ),
        "x-topic-actions": deepcopy(planning["x-topic-actions"]),
        "topic_in": external_inputs,
        "topic_out": outputs,
        "inputSchema": {
            "type": "object",
            "properties": _action_properties(),
            "required": ["action"],
            "additionalProperties": False,
            "x-action-params": deepcopy(NAVIGATION_ACTION_PARAMS),
        },
        "configSchema": deepcopy(NAVIGATION_CONFIG_SCHEMA),
    }


__all__ = [
    "CONTROLLED_SEMANTIC_SPATIAL_TOOL_NAME",
    "NAVIGATION_ACTIONS",
    "NAVIGATION_ACTION_PARAMS",
    "NAVIGATION_CONFIG_SCHEMA",
    "NAVIGATION_PUBLIC_OUTPUT_PORTS",
    "NAVIGATION_PUBLIC_ACTIONS",
    "navigation_tool_definition",
]
