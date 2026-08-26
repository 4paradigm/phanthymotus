"""Internal Nav2 contract owned by ControlledSemanticSpatial."""

from __future__ import annotations

from copy import deepcopy


NAV2_ACTIONS = (
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
    "discovery_timeout_sec": 5.0,
    "input_max_age_ms": 500,
    "min_x_mps": 0.30,
    "max_x_mps": 1.0,
    "min_y_mps": 0.0,
    "max_y_mps": 0.0,
    "min_yaw_rps": 1.0,
    "max_yaw_rps": 2.0,
    "proposal_ttl_ms": 250,
    "proposal_frequency_hz": 5.0,
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
        "min_x_mps": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.30,
            "description": "X deadband; smaller nonzero proposals are suppressed",
        },
        "max_x_mps": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 1.0,
            "description": "Maximum magnitude of an X proposal",
        },
        "min_y_mps": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.0,
            "description": "Y deadband; smaller nonzero proposals are suppressed",
        },
        "max_y_mps": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.0,
            "description": "Maximum magnitude of a Y proposal; zero disables lateral motion",
        },
        "min_yaw_rps": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 2.0,
            "default": 1.0,
            "description": "Yaw deadband; smaller nonzero proposals are suppressed",
        },
        "max_yaw_rps": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 2.0,
            "default": 2.0,
            "description": "Maximum magnitude of a yaw proposal; zero disables rotation",
        },
        "proposal_ttl_ms": {
            "type": "integer",
            "const": 250,
            "minimum": 50,
            "maximum": 250,
            "default": 250,
        },
        "proposal_frequency_hz": {
            "type": "number",
            "const": 5.0,
            "default": 5.0,
            "description": "Latest-only G1 velocity proposal cadence",
        },
    },
    "additionalProperties": False,
}

_CANVAS_CONFIG_FIELDS = (
    "backend",
    "request_timeout_sec",
    "discovery_timeout_sec",
    "min_x_mps",
    "max_x_mps",
    "min_y_mps",
    "max_y_mps",
    "min_yaw_rps",
    "max_yaw_rps",
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
            "Block until the previous navigate_to_pose "
            "completes. Returns on arrival, timeout, or error. Always call "
            "after navigate_to_pose."
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
            "Nav2 planner and controller consuming FAST-LIVO2 localization and "
            "directly accumulated static occupancy plus live clearing "
            "obstacles. This "
            "ActuCore card only emits "
            "bounded velocity proposals with a fresh task ID; the Driver loco "
            "actuator owns any physical execution and task adoption."
        ),
        "x-topic-actions": [
            {
                "port": "goal_pose",
                "action": "navigate_to_pose",
                "wait_action": "wait_navigation_done",
                "stop_action": "stop_nav",
                "schema": "phanthy.navigation.goal.v1",
                "id_field": "goal_id",
                "allowed_fields": ["x", "y", "yaw", "speed"],
                "required_fields": ["x", "y", "yaw"],
            }
        ],
        "topic_in": [
            {
                "port": "livo_odom",
                "topic": f"{root}/navigation/odom",
                "format": "sensor/odometry",
                "ros_type": "nav_msgs/msg/Odometry",
                "qos": "BEST_EFFORT + KEEP_LAST(depth=5) + VOLATILE",
                "rate_hz": 10,
                "timestamp": "FAST-LIVO2 corrected ROS system time",
                "frame_id": "map -> base_link",
                "axes": "ROS REP-103 right-handed: x forward, y left, z up",
                "units": "position=m, velocity=m/s, angular=rad/s",
                "max_age_ms": 500,
                "desc": (
                    "Canonical planar navigation odometry produced by the "
                    "in-container FAST-LIVO2 adapter"
                ),
            },
            {
                "port": "registered_cloud",
                "topic": f"{root}/navigation/cloud_registered",
                "format": "sensor/pointcloud",
                "ros_type": "sensor_msgs/msg/PointCloud2",
                "qos": "BEST_EFFORT + KEEP_LAST(depth=1) + VOLATILE",
                "rate_hz": 10,
                "timestamp": "FAST-LIVO2 corrected ROS system time",
                "frame_id": "map",
                "axes": "ROS REP-103 right-handed: x forward, y left, z up",
                "units": "x/y/z in meters",
                "max_age_ms": 500,
                "desc": (
                    "Motion-compensated registered cloud from FAST-LIVO2; Nav2 "
                    "uses it only for rolling obstacle costmaps"
                ),
            },
            {
                "port": "static_map",
                "topic": f"{root}/navigation/static_map",
                "format": "sensor/occupancy-grid",
                "ros_type": "nav_msgs/msg/OccupancyGrid",
                "qos": "RELIABLE + KEEP_LAST(depth=1) + TRANSIENT_LOCAL",
                "rate_hz": 1,
                "timestamp": "FAST-LIVO2 adapter publish time",
                "frame_id": "map",
                "axes": "ROS REP-103 right-handed: x forward, y left, z up",
                "units": "resolution=m/cell; data=-1 unknown, 0 free, 100 occupied",
                "desc": (
                    "Directly accumulated static occupancy source for "
                    "the Nav2 global StaticLayer"
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
                    "goal_id plus x/y/yaw; the planner creates a fresh nav_id "
                    "for the accepted navigation task."
                ),
            },
        ],
        "topic_out": [
            {
                "port": "velocity_proposal",
                "topic": f"{root}/navigation/nav2/velocity_proposal",
                "format": "data/json",
                "ros_type": "std_msgs/msg/String",
                "qos": "RELIABLE + KEEP_LAST(depth=1) + VOLATILE",
                "schema": "phanthy.navigation.velocity_proposal.v1",
                "rate_hz": 5,
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
                "port": "plan",
                "topic": "/plan",
                "format": "sensor/path",
                "ros_type": "nav_msgs/msg/Path",
                "qos": "RELIABLE + KEEP_LAST(depth=1) + VOLATILE",
                "schema": "phanthy.navigation.path.v1",
                "frame_id": "map",
                "axes": "ROS REP-103 right-handed: x forward, y left",
                "units": "x/y/z=m; yaw=rad",
                "desc": "Current Nav2 global plan rendered as a 2D path in Canvas",
            },
            {
                "port": "costmap",
                "topic": "/global_costmap/costmap",
                "format": "sensor/costmap",
                "ros_type": "nav_msgs/msg/OccupancyGrid",
                "qos": "RELIABLE + KEEP_LAST(depth=1) + TRANSIENT_LOCAL",
                "schema": "phanthy.navigation.costmap.v1",
                "frame_id": "map",
                "axes": "ROS REP-103 right-handed: x forward, y left",
                "units": "resolution=m/cell; data=-1 unknown, 0 free, 100 occupied",
                "default_preview": True,
                "desc": (
                    "Live Nav2 global costmap rendered with the current plan, "
                    "robot pose, inflated obstacles, and goal in Canvas"
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
                    "minimum": 0.30,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": "Navigation speed 0.30-1.00 m/s (default 0.50)",
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
