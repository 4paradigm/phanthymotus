"""Public MCP contract for the Nav2 Perception card."""

from __future__ import annotations

from copy import deepcopy


NAV2_ACTIONS = (
    "start_mapping",
    "stop_mapping",
    "switch_runtime_mode",
    "tag_place",
    "untag_place",
    "list_tags",
    "list_maps",
    "delete_map",
    "load_map",
    "navigate_to_tag",
    "navigate_to_pose",
    "wait_navigation_done",
    "pause_nav",
    "resume_nav",
    "stop_nav",
)

NAV2_LIFECYCLE_ACTIONS = ("info", "config", "start", "stop")
NAV2_PUBLIC_ACTIONS = NAV2_LIFECYCLE_ACTIONS + NAV2_ACTIONS

NAV2_CONFIG_DEFAULTS = {
    "namespace": "ubuntu",
    "backend": "ros_topic",
    "shadow_only": True,
    "request_timeout_sec": 30.0,
    "runtime_switch_timeout_sec": 120.0,
    "discovery_timeout_sec": 5.0,
    "input_max_age_ms": 500,
    "max_forward_mps": 0.15,
    "max_reverse_mps": 0.15,
    "max_lateral_mps": 0.0,
    "max_yaw_rps": 0.35,
    "max_planar_mps": 0.18,
    "proposal_ttl_ms": 250,
    "map_storage_dir": "/maps",
}

NAV2_FULL_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "namespace": {
            "type": "string",
            "const": "ubuntu",
            "default": "ubuntu",
            "description": "First-release G1 Driver namespace",
        },
        "backend": {
            "type": "string",
            "enum": ["ros_topic", "disabled"],
            "default": "ros_topic",
        },
        "shadow_only": {
            "type": "boolean",
            "const": True,
            "default": True,
            "description": "Nav2 only emits proposals; Driver owns execution",
        },
        "request_timeout_sec": {
            "type": "number",
            "minimum": 1.0,
            "maximum": 120.0,
            "default": 30.0,
        },
        "runtime_switch_timeout_sec": {
            "type": "number",
            "minimum": 10.0,
            "maximum": 300.0,
            "default": 120.0,
        },
        "discovery_timeout_sec": {
            "type": "number",
            "minimum": 0.5,
            "maximum": 30.0,
            "default": 5.0,
        },
        "input_max_age_ms": {
            "type": "integer",
            "const": 500,
            "minimum": 100,
            "maximum": 2000,
            "default": 500,
        },
        "max_forward_mps": {
            "type": "number",
            "const": 0.15,
            "minimum": 0.01,
            "maximum": 0.15,
            "default": 0.15,
        },
        "max_reverse_mps": {
            "type": "number",
            "const": 0.15,
            "minimum": 0.0,
            "maximum": 0.15,
            "default": 0.15,
        },
        "max_lateral_mps": {
            "type": "number",
            "const": 0.0,
            "minimum": 0.0,
            "maximum": 0.12,
            "default": 0.0,
        },
        "max_yaw_rps": {
            "type": "number",
            "const": 0.35,
            "minimum": 0.01,
            "maximum": 0.35,
            "default": 0.35,
        },
        "max_planar_mps": {
            "type": "number",
            "const": 0.18,
            "minimum": 0.01,
            "maximum": 0.18,
            "default": 0.18,
        },
        "proposal_ttl_ms": {
            "type": "integer",
            "const": 250,
            "minimum": 50,
            "maximum": 250,
            "default": 250,
        },
        "map_storage_dir": {
            "type": "string",
            "const": "/maps",
            "default": "/maps",
        },
    },
    "additionalProperties": False,
}

_CANVAS_CONFIG_FIELDS = (
    "backend",
    "request_timeout_sec",
    "runtime_switch_timeout_sec",
    "discovery_timeout_sec",
)
NAV2_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        key: deepcopy(NAV2_FULL_CONFIG_SCHEMA["properties"][key])
        for key in _CANVAS_CONFIG_FIELDS
    },
    "additionalProperties": False,
}

NAV2_ACTION_PARAMS = {
    "start_mapping": {
        "params": ["map_name"],
        "description": "Start SLAM mapping with given map name",
    },
    "stop_mapping": {
        "params": [],
        "description": "Stop mapping and save the map",
    },
    "switch_runtime_mode": {
        "params": ["runtime_mode", "map_name"],
        "description": (
            "Switch the Nav2 container between mapping and localization. "
            "Localization requires an existing map_name."
        ),
    },
    "tag_place": {
        "params": ["name", "description"],
        "description": "Tag current position with a semantic name",
    },
    "untag_place": {
        "params": ["name"],
        "description": "Remove a place tag",
    },
    "list_tags": {
        "params": [],
        "description": "List all tags in current map with relative positions",
    },
    "list_maps": {"params": [], "description": "List all saved maps"},
    "delete_map": {
        "params": ["map_name"],
        "description": "Delete a map and its associated data",
    },
    "load_map": {
        "params": ["map_name"],
        "description": "Load a map (robot must be at map origin)",
    },
    "navigate_to_tag": {
        "params": ["tag_name", "speed"],
        "description": (
            "Navigate to a tagged place with obstacle detouring (non-blocking). "
            "MUST be followed by a "
            "separate wait_navigation_done call in the same turn to wait for "
            "arrival before proceeding."
        ),
    },
    "navigate_to_pose": {
        "params": ["x", "y", "yaw", "speed"],
        "description": (
            "Navigate to coordinates with obstacle detouring (non-blocking). "
            "MUST be followed by a "
            "separate wait_navigation_done call in the same turn to wait for "
            "arrival before proceeding."
        ),
    },
    "wait_navigation_done": {
        "params": ["stall_timeout"],
        "description": (
            "Block until the previous navigate_to_tag or navigate_to_pose "
            "completes. Returns on arrival, timeout, or error. Always call "
            "after navigate_to_tag/navigate_to_pose."
        ),
    },
    "pause_nav": {"params": [], "description": "Pause navigation"},
    "resume_nav": {"params": [], "description": "Resume navigation"},
    "stop_nav": {"params": [], "description": "Stop and cancel navigation"},
}

NAV2_PUBLIC_ACTION_PARAMS = {
    "info": {"params": [], "description": "Read card and backend state"},
    "config": {
        "params": list(NAV2_CONFIG_SCHEMA["properties"]),
        "description": "Validate and store idle-state configuration",
    },
    "start": {
        "params": ["instance_id", "input_topics", "input_bindings"],
        "description": "Validate Canvas wiring and acquire ROS resources",
    },
    "stop": {"params": [], "description": "Stop navigation and release resources"},
    **NAV2_ACTION_PARAMS,
}


def _root(namespace: str) -> str:
    normalized = namespace.strip("/")
    return f"/{normalized}" if normalized else ""


def nav2_tool_definition(namespace: str) -> dict:
    """Return an isolated tool definition for one robot namespace."""

    root = _root(namespace)
    tool = {
        "name": "nav2",
        "displayName": "Nav2",
        "type": "processor",
        "multiInstance": False,
        "description": (
            "Nav2 — mapping, saved-map localization, semantic "
            "place tags and Nav2 navigation. This Perception card only emits "
            "bounded velocity proposals; an explicitly authorized Driver loco "
            "actuator owns any physical execution."
        ),
        "x-execution-control": {
            "version": 1,
            "proposal_schema": "phanthy.navigation.velocity_proposal.v1",
            "output_port": "velocity_proposal",
            "target_tool": "loco",
            "lease_argument": "_control_nav_id",
            "start_actions": ["navigate_to_tag", "navigate_to_pose"],
            "wait_actions": ["wait_navigation_done"],
            "stop_actions": ["stop_nav"],
            "pause_actions": ["pause_nav"],
            "resume_actions": ["resume_nav"],
            "terminal_statuses": [
                "arrived",
                "succeeded",
                "cancelled",
                "stopped",
                "timeout",
                "error",
                "aborted",
                "rejected",
            ],
        },
        "x-topic-actions": [
            {
                "port": "goal_pose",
                "action": "navigate_to_pose",
                "wait_action": "wait_navigation_done",
                "stop_action": "stop_nav",
                "schema": "phanthy.navigation.goal.v1",
                "id_field": "goal_id",
                "allowed_fields": ["x", "y", "yaw", "speed"],
            }
        ],
        "topic_in": [
            {
                "port": "loco_state",
                "topic": f"{root}/loco/state",
                "format": "data/json",
                "ros_type": "std_msgs/msg/String",
                "qos": "BEST_EFFORT + KEEP_LAST(depth=10) + VOLATILE",
                "schema": "unitree.g1.loco_state.legacy",
                "compatible_schemas": ["phanthy.g1.loco_state.v2"],
                "rate_hz": 10,
                "timestamp": (
                    "legacy uses adapter receive time; v2 declares Driver callback "
                    "receive time in source_stamp_ns using ROS system/Unix clock"
                ),
                "frame_id": "odom_source (v2 payload or legacy adapter contract)",
                "axes": "ROS REP-103 right-handed: x forward, y left, z up",
                "units": "position=m, velocity=m/s, yaw_speed=rad/s",
                "max_age_ms": 500,
                "desc": (
                    "Legacy Driver locomotion JSON remains supported; v2 Driver "
                    "receive timestamps are freshness-checked before odom/TF"
                ),
            },
            {
                "port": "lidar_cloud",
                "topic": "/utlidar/cloud",
                "format": "sensor/pointcloud",
                "ros_type": "sensor_msgs/msg/PointCloud2",
                "qos": "BEST_EFFORT + KEEP_LAST(depth=1) + VOLATILE",
                "schema": "sensor_msgs/msg/PointCloud2",
                "rate_hz": 10,
                "timestamp": (
                    "native PointCloud2 header.stamp; companion maps an independent "
                    "LiDAR clock into ROS system time before Nav2 consumption"
                ),
                "frame_id": "native PointCloud2 header.frame_id (utlidar_lidar)",
                "axes": "ROS REP-103 right-handed: x forward, y left, z up",
                "units": "native PointCloud2 fields; x/y/z in meters",
                "max_age_ms": 500,
                "desc": (
                    "Robot-native ROS2 PointCloud2 stream. The companion preserves "
                    "frame_id, fields and point bytes, retains the raw stamp in "
                    "diagnostics, and only rewrites the output stamp when clock-domain "
                    "normalization is required"
                ),
            },
            {
                "port": "goal_pose",
                "topic": f"{root}/navigation/goal_pose",
                "format": "data/json",
                "ros_type": "std_msgs/msg/String",
                "qos": "RELIABLE + KEEP_LAST(depth=10) + VOLATILE",
                "schema": "phanthy.navigation.goal.v1",
                "required": False,
                "frame_id": "map",
                "axes": "ROS REP-103 right-handed: x forward, y left",
                "units": "x/y=m, yaw=rad, speed=m/s",
                "desc": (
                    "Optional target input. Each JSON message needs a unique "
                    "goal_id plus x/y/yaw and is executed through the same "
                    "Agent Core Driver lease as navigate_to_pose."
                ),
            },
        ],
        "topic_out": [
            {
                "port": "velocity_proposal",
                "topic": f"{root}/navigation/nav2/velocity_proposal",
                "format": "data/json",
                "ros_type": "std_msgs/msg/String",
                "qos": "RELIABLE + KEEP_LAST(depth=10) + VOLATILE",
                "schema": "phanthy.navigation.velocity_proposal.v1",
                "rate_hz": 20,
                "timestamp": "issued_at_unix_ms; TTL uses Driver receive monotonic time",
                "frame_id": "base_link",
                "axes": "ROS REP-103: x forward, y left, yaw counter-clockwise",
                "units": "linear=m/s, angular=rad/s",
                "max_age_ms": 250,
                "desc": (
                    "Structured Nav2 proposal for the existing Driver loco "
                    "actuator; never a physical command by itself"
                ),
            },
            {
                "port": "map_view",
                "topic": f"{root}/navigation/nav2/map_view",
                "format": "sensor/mapping",
                "ros_type": "std_msgs/msg/UInt8MultiArray",
                "qos": "BEST_EFFORT + KEEP_LAST(depth=5) + VOLATILE",
                "schema": "phanthy.navigation.map_view.v1",
                "rate_hz": 1,
                "frame_id": "map",
                "axes": "ROS REP-103 right-handed: x forward, y left",
                "units": "x/y=m, yaw=rad, occupancy=percent",
                "default_preview": True,
                "desc": (
                    "Default read-only Canvas preview containing occupied cells "
                    "and the current robot pose; it cannot issue navigation commands"
                ),
            },
        ],
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(NAV2_PUBLIC_ACTIONS),
                    "description": "Action to perform",
                },
                "map_name": {
                    "type": "string",
                    "description": (
                        "Map name (required for localization mode, start_mapping, "
                        "delete_map and load_map)"
                    ),
                },
                "runtime_mode": {
                    "type": "string",
                    "enum": ["mapping", "localization"],
                    "default": "localization",
                    "description": "Target Nav2 container runtime mode",
                },
                "name": {"type": "string", "description": "POI tag name"},
                "description": {
                    "type": "string",
                    "description": "POI description",
                },
                "tag_name": {
                    "type": "string",
                    "description": "Target tag name for navigation",
                },
                "x": {
                    "type": "number",
                    "description": "Target X coordinate (meters)",
                },
                "y": {
                    "type": "number",
                    "description": "Target Y coordinate (meters)",
                },
                "yaw": {
                    "type": "number",
                    "description": "Target yaw (radians)",
                },
                "speed": {
                    "type": "number",
                    "minimum": 0.05,
                    "maximum": 0.15,
                    "default": 0.15,
                    "description": "Navigation speed 0.05-0.15 m/s (default 0.15)",
                },
                "stall_timeout": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 3600.0,
                    "default": 90.0,
                    "description": (
                        "Seconds without movement before declaring timeout (default 90)"
                    ),
                },
                "instance_id": {
                    "type": "string",
                    "description": "Canvas card instance identifier",
                },
                "input_topic": {"type": "string"},
                "input_topics": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "input_bindings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "port": {"type": "string"},
                            "topic": {"type": "string"},
                        },
                        "required": ["port", "topic"],
                    },
                },
                **deepcopy(NAV2_CONFIG_SCHEMA["properties"]),
            },
            "required": ["action"],
            "x-action-params": NAV2_PUBLIC_ACTION_PARAMS,
        },
        "configSchema": deepcopy(NAV2_CONFIG_SCHEMA),
    }
    return deepcopy(tool)
