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


CONTROLLED_SEMANTIC_SPATIAL_LIFECYCLE_ACTIONS = ("info", "config", "start", "stop")
SEMANTIC_ACTIONS = ("capture", "navigate")
CONTROLLED_SEMANTIC_SPATIAL_ACTIONS = FAST_LIVO2_ACTIONS + NAV2_ACTIONS + SEMANTIC_ACTIONS
CONTROLLED_SEMANTIC_SPATIAL_PUBLIC_ACTIONS = (
    CONTROLLED_SEMANTIC_SPATIAL_LIFECYCLE_ACTIONS + CONTROLLED_SEMANTIC_SPATIAL_ACTIONS
)
CONTROLLED_SEMANTIC_SPATIAL_TOOL_NAME = "controlled_semantic_spatial"
CONTROLLED_SEMANTIC_SPATIAL_PUBLIC_OUTPUT_PORTS = (
    "map_view",
    "control_velocity",
)


# Accepted only by config for saved-card compatibility, never shown as options.
LEGACY_CONFIG_FIELDS = (
    "lidar_topic", "rgb_topic", "imu_topic",
    "mapping_request_timeout_sec", "mapping_discovery_timeout_sec",
    "planning_request_timeout_sec", "planning_discovery_timeout_sec",
    "lidar_qos_depth", "imu_qos_depth", "map_voxel_size_m", "fault_capture_enabled",
)


def _config_properties() -> dict:
    mapping = fast_livo2_tool_definition("ubuntu")["configSchema"]["properties"]
    planning = nav2_tool_definition("ubuntu")["configSchema"]["properties"]
    semantic = build_manifest(
        "/ubuntu/camera/rgb",
        "/ubuntu/navigation/odom",
    )["configSchema"]["properties"]
    result = {
        "inflation_radius_m": {
            "type": "number", "exclusiveMinimum": 0, "maximum": 2.0,
            "default": 0.55, "title": "共享膨胀半径（m，下次启动生效）",
            "description": (
                "局部与全局共用，下次启动生效；不是 footprint 或安全净空。"
                "小于机器人外接尺寸可能不保守；2 m 为卡片防误填限制，0 不用于关闭膨胀。"
            ),
        },
        "obstacle_min_height_m": deepcopy(mapping["obstacle_min_height_m"]),
        "obstacle_max_height_m": deepcopy(mapping["obstacle_max_height_m"]),
        "map_view_enabled": deepcopy(mapping["map_view_enabled"]),
        "rotate_reengage_rad": deepcopy(planning["rotate_reengage_rad"]),
    }
    for key in (
        "min_x_mps",
        "max_x_mps",
        "min_y_mps",
        "max_y_mps",
    ):
        result[key] = deepcopy(planning[key])
    result["rotate_speed_rps"] = {
        "type": "number",
        "minimum": 0.3,
        "maximum": 2.0,
        "default": 0.3,
        "description": "Fixed magnitude of every nonzero yaw proposal",
    }
    result.update(
        {
            "vlm_base_url": deepcopy(semantic["base_url"]),
            "vlm_api_key": deepcopy(semantic["api_key"]),
            "vlm_model": deepcopy(semantic["model"]),
            "vlm_timeout_sec": deepcopy(semantic["timeout_sec"]),
        }
    )
    for key, title in {
        "obstacle_min_height_m": "障碍最低高度（m）",
        "obstacle_max_height_m": "障碍最高高度（m）",
        "map_view_enabled": "地图预览",
        "rotate_reengage_rad": "重新转向偏差（rad）",
        "min_x_mps": "X 输出死区（m/s）", "max_x_mps": "X 最大速度（m/s）",
        "min_y_mps": "Y 输出死区（m/s）", "max_y_mps": "Y 最大速度（m/s）",
        "rotate_speed_rps": "固定转向角速度幅值（rad/s）",
        "vlm_base_url": "VLM 服务地址", "vlm_api_key": "VLM API 密钥",
        "vlm_model": "VLM 模型", "vlm_timeout_sec": "VLM 超时（s）",
    }.items():
        result[key]["title"] = title
    return result


CONTROLLED_SEMANTIC_SPATIAL_CONFIG_SCHEMA = {
    "type": "object",
    "properties": _config_properties(),
    "required": [],
    "additionalProperties": False,
}


def _action_properties() -> dict:
    mapping = fast_livo2_tool_definition("ubuntu")["inputSchema"]["properties"]
    planning = nav2_tool_definition("ubuntu")["inputSchema"]["properties"]
    semantic = build_manifest(
        "/ubuntu/camera/rgb",
        "/ubuntu/navigation/odom",
    )["inputSchema"]["properties"]
    result = {"action": {"type": "string", "enum": list(CONTROLLED_SEMANTIC_SPATIAL_PUBLIC_ACTIONS)}}
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
    for key, value in CONTROLLED_SEMANTIC_SPATIAL_CONFIG_SCHEMA["properties"].items():
        result[key] = deepcopy(value)
    for key in LEGACY_CONFIG_FIELDS:
        # Ignore even malformed saved values; only exact names are accepted.
        result[key] = {"description": "Legacy config only; ignored in favor of internal defaults"}
    result["camera_info"] = {
        "type": "object",
        "description": (
            "Upstream camera declarations keyed by topic; accepted for mainline "
            "compatibility but not adopted as navigation calibration."
        ),
        "additionalProperties": {"type": "object"},
    }
    result["control_interface"] = {
        "type": "object",
        "description": (
            "Optional downstream motus.control/1 descriptor for the velocity "
            "output; absent when no control consumer is connected."
        ),
    }
    return result


CONTROLLED_SEMANTIC_SPATIAL_ACTION_PARAMS = {
    **deepcopy(FAST_LIVO2_ACTION_PARAMS),
    **deepcopy(NAV2_ACTION_PARAMS),
    "capture": {
        "params": [],
        "description": "Capture a visual waypoint at the current map pose",
    },
    "navigate": {
        "params": ["query"],
        "description": (
            "Match a visual waypoint and start Nav2 with the same lease. "
            "System automatically waits for arrival via ACP barrier."
        ),
    },
    "info": {"params": [], "description": "Read the unified navigation state"},
    "config": {
        "params": list(CONTROLLED_SEMANTIC_SPATIAL_CONFIG_SCHEMA["properties"]),
        "description": "Configure mapping, planning and semantic navigation",
    },
    "start": {
        "params": ["instance_id", "input_topic", "input_topics", "input_bindings", "control_interface"],
        "description": "Start the in-container runtime and acquire card resources",
    },
    "stop": {
        "params": [],
        "description": "Stop navigation, mapping and every owned child process",
    },
}


def controlled_semantic_spatial_tool_definition(namespace: str) -> dict:
    mapping = fast_livo2_tool_definition(namespace)
    planning = nav2_tool_definition(namespace)
    semantic = build_manifest(
        f"/{namespace.strip('/')}/camera/rgb",
        f"/{namespace.strip('/')}/navigation/odom",
        status_topic=f"/{namespace.strip('/')}/navigation/fast_livo2/status",
    )
    rgb_input = deepcopy(semantic["topic_in"][0])
    rgb_input["required"] = False
    lidar_input = deepcopy(next(item for item in mapping["topic_in"] if item["port"] == "lidar"))
    lidar_input["required"] = True
    imu_input = deepcopy(next(item for item in mapping["topic_in"] if item["port"] == "imu"))
    imu_input["required"] = True
    external_inputs = [rgb_input, lidar_input, imu_input]
    component_outputs = {
        str(item.get("port", "")): item
        for item in [*mapping["topic_out"], *planning["topic_out"]]
        if item.get("port")
    }
    component_outputs["control_velocity"] = {
        "port": "control_velocity",
        "topic": "/actucore/navi/cmd",
        "format": "control/velocity",
        "ros_type": "std_msgs/msg/String",
        "schema": "motus.control/1",
        "desc": "Nav2 planar twist [vx, vy, vz, wx, wy, wz]",
    }
    outputs = [
        deepcopy(component_outputs[port]) for port in CONTROLLED_SEMANTIC_SPATIAL_PUBLIC_OUTPUT_PORTS
    ]
    auxiliary_topics = [
        deepcopy(
            next(item for item in planning["topic_out"] if item["port"] == "plan")
        ),
        deepcopy(
            next(
                item for item in planning["topic_in"] if item["port"] == "livo_odom"
            )
        ),
    ]

    completion = deepcopy(planning["inputSchema"]["x-completion"])
    completion["actions"] = [*completion.get("actions", []), "navigate"]

    return {
        "name": CONTROLLED_SEMANTIC_SPATIAL_TOOL_NAME,
        "displayName": CONTROLLED_SEMANTIC_SPATIAL_TOOL_NAME,
        "type": "processor",
        "multiInstance": False,
        "description": (
            "Unified mapping, localization, semantic waypoint and Nav2 "
            "planner/controller card. The runtime consumes a robot-independent "
            "REP-103 sensor frame and Driver-owned static TF. FAST-LIVO2 and "
            "Nav2 run as child "
            "processes inside the ActuCore container; only bounded velocity "
            "proposals leave the card."
        ),
        "topic_in": external_inputs,
        "topic_out": outputs,
        "topic_aux": auxiliary_topics,
        "inputSchema": {
            "type": "object",
            "properties": _action_properties(),
            "required": ["action"],
            "additionalProperties": False,
            "x-action-params": deepcopy(CONTROLLED_SEMANTIC_SPATIAL_ACTION_PARAMS),
            "x-completion": completion,
            # Finish an in-flight start and the bounded 360 s map save before release.
            "x-stop": {"timeout_s": 600},
            "x-hooks": deepcopy(planning["inputSchema"]["x-hooks"]),
        },
        "configSchema": deepcopy(CONTROLLED_SEMANTIC_SPATIAL_CONFIG_SCHEMA),
    }


__all__ = [
    "CONTROLLED_SEMANTIC_SPATIAL_TOOL_NAME",
    "CONTROLLED_SEMANTIC_SPATIAL_ACTIONS",
    "CONTROLLED_SEMANTIC_SPATIAL_ACTION_PARAMS",
    "CONTROLLED_SEMANTIC_SPATIAL_CONFIG_SCHEMA",
    "CONTROLLED_SEMANTIC_SPATIAL_PUBLIC_OUTPUT_PORTS",
    "CONTROLLED_SEMANTIC_SPATIAL_PUBLIC_ACTIONS",
    "controlled_semantic_spatial_tool_definition",
]
