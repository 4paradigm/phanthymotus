"""MCP tool definition for vision-and-language navigation."""

from __future__ import annotations

from copy import deepcopy


DEFAULT_CAMERA_TOPIC = "/ubuntu/camera/rgb"
DEFAULT_ODOMETRY_TOPIC = "/ubuntu/navigation/odom"
DEFAULT_STATUS_TOPIC = "/ubuntu/navigation/fast_livo2/status"
DEFAULT_GOAL_TOPIC = "/ubuntu/navigation/goal_pose"
DEFAULT_VLM_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_VLM_MODEL = "doubao-seed-2-1-pro-260628"
DEFAULT_VLM_TIMEOUT_SEC = 18.0


_MANIFEST = {
    "name": "vln",
    "type": "processor",
    "multiInstance": False,
    "description": (
        "Vision-and-language navigation. capture records the current JPEG image and "
        "FAST-LIVO2 map pose as a visual waypoint. navigate matches "
        "a natural-language destination against every recorded waypoint. When a "
        "match is found, it publishes phanthy.navigation.goal.v1 to the downstream "
        "Nav2 ROS2 goal topic; when not_found, it publishes nothing and tells the "
        "caller that this place has not been recorded."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "stop", "info", "config", "capture", "navigate"],
                "description": "Action to perform",
            },
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "Natural-language description of the destination",
            },
        },
        # Phanthy Agent Core builds the action-specific required list from this
        # top-level list. x-action-params removes query from capture's schema.
        "required": ["action", "query"],
        "x-action-params": {
            "capture": {
                "params": [],
                "description": (
                    "Record the robot's current view and FAST-LIVO2 map pose. "
                    "This action takes no parameters."
                ),
            },
            "navigate": {
                "params": ["query"],
                "description": (
                    "Match the requested destination against all recorded points. "
                    "If matched, publish the recorded map pose to the downstream "
                    "Nav2 card over ROS2. If not_found, publish nothing and tell "
                    "the user that the place has not been recorded."
                ),
            },
        },
    },
    "configSchema": {
        "type": "object",
        "properties": {
            "base_url": {
                "type": "string",
                "title": "VLM API URL",
                "description": "VLM API base URL",
                "default": DEFAULT_VLM_BASE_URL,
                "scope": "shared",
            },
            "api_key": {
                "type": "string",
                "title": "VLM API Key",
                "description": "Credential for the configured VLM service",
                "format": "password",
                "scope": "shared",
            },
            "model": {
                "type": "string",
                "title": "VLM model",
                "description": "Model name accepted by the VLM service",
                "default": DEFAULT_VLM_MODEL,
                "scope": "shared",
            },
            "timeout_sec": {
                "type": "number",
                "title": "VLM timeout (seconds)",
                "description": "Maximum duration of one VLM request",
                "default": DEFAULT_VLM_TIMEOUT_SEC,
                "minimum": 1.0,
                "maximum": 120.0,
                "scope": "shared",
            },
        },
        "required": ["base_url", "api_key", "model"],
    },
    "topic_in": [
        {
            "port": "rgb",
            "topic": DEFAULT_CAMERA_TOPIC,
            "format": "image/jpeg",
            "ros_type": "sensor_msgs/msg/CompressedImage",
            "desc": "JPEG image from the camera_rgb card",
        },
        {
            "port": "livo_odom",
            "topic": DEFAULT_ODOMETRY_TOPIC,
            "format": "sensor/odometry",
            "ros_type": "nav_msgs/msg/Odometry",
            "desc": "FAST-LIVO2 pose in the map frame",
        },
        {
            "port": "livo_status",
            "topic": DEFAULT_STATUS_TOPIC,
            "format": "data/json",
            "ros_type": "std_msgs/msg/String",
            "schema": "phanthy.navigation.fast_livo2_status.v1",
            "required": False,
            "desc": "FAST-LIVO2 mapping/session heartbeat used as a safety guard",
        },
    ],
    "topic_out": [
        {
            "port": "goal_pose",
            "topic": DEFAULT_GOAL_TOPIC,
            "format": "data/json",
            "ros_type": "std_msgs/msg/String",
            "schema": "phanthy.navigation.goal.v1",
            "desc": "Matched map-frame goal for the downstream Nav2 card",
        }
    ],
}


def build_manifest(
    camera_topic: str,
    odometry_topic: str,
    goal_topic: str = DEFAULT_GOAL_TOPIC,
    status_topic: str = DEFAULT_STATUS_TOPIC,
) -> dict:
    """Return an isolated manifest containing the configured input topics."""

    manifest = deepcopy(_MANIFEST)
    manifest["topic_in"][0]["topic"] = camera_topic
    manifest["topic_in"][1]["topic"] = odometry_topic
    manifest["topic_in"][2]["topic"] = status_topic
    manifest["topic_out"][0]["topic"] = goal_topic
    return manifest


MANIFEST = build_manifest(DEFAULT_CAMERA_TOPIC, DEFAULT_ODOMETRY_TOPIC)
