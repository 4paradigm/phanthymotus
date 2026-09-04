"""Internal FAST-LIVO2 contract owned by ControlledSemanticSpatial."""

from __future__ import annotations

from copy import deepcopy


FAST_LIVO2_ACTIONS = (
    "start_mapping",
    "stop_mapping",
    "load_map",
    "relocalize",
)
FAST_LIVO2_LIFECYCLE_ACTIONS = ("info", "config", "start", "stop")
FAST_LIVO2_PUBLIC_ACTIONS = FAST_LIVO2_LIFECYCLE_ACTIONS + FAST_LIVO2_ACTIONS

FAST_LIVO2_CONFIG_DEFAULTS = {
    "namespace": "ubuntu",
    "backend": "ros_topic",
    "request_timeout_sec": 130.0,
    "discovery_timeout_sec": 5.0,
    "input_max_age_ms": 500,
    "lidar_qos_depth": 2,
    "imu_qos_depth": 400,
    "map_voxel_size_m": 0.10,
    "obstacle_min_height_m": -0.30,
    "obstacle_max_height_m": 0.30,
    "map_view_enabled": True,
    "fault_capture_enabled": True,
    "collection_enabled": False,
    "collection_directory": "/opt/phanthy-motus/data/fast_livo2/recordings",
}

FAST_LIVO2_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "backend": {
            "type": "string",
            "enum": ["ros_topic", "disabled"],
            "default": "ros_topic",
        },
        "request_timeout_sec": {
            "type": "number",
            "minimum": 5.0,
            "maximum": 180.0,
            "default": 130.0,
        },
        "discovery_timeout_sec": {
            "type": "number",
            "minimum": 0.5,
            "maximum": 30.0,
            "default": 5.0,
        },
        "lidar_qos_depth": {
            "type": "integer",
            "minimum": 1,
            "maximum": 32,
            "default": 2,
            "description": (
                "FAST-LIVO2 LiDAR subscription queue depth; applied on the next "
                "card start"
            ),
        },
        "imu_qos_depth": {
            "type": "integer",
            "minimum": 1,
            "maximum": 4000,
            "default": 400,
            "description": (
                "FAST-LIVO2 IMU subscription queue depth; applied on the next "
                "card start"
            ),
        },
        "map_voxel_size_m": {
            "type": "number",
            "minimum": 0.05,
            "maximum": 0.50,
            "default": 0.10,
        },
        "obstacle_min_height_m": {
            "type": "number",
            "minimum": -3.0,
            "maximum": 3.0,
            "default": -0.30,
            "description": (
                "Lowest map-frame Z retained as a 2D navigation obstacle; "
                "lower points remain visible in map_view"
            ),
        },
        "obstacle_max_height_m": {
            "type": "number",
            "minimum": -3.0,
            "maximum": 3.0,
            "default": 0.30,
            "description": (
                "Highest map-frame Z retained as a 2D navigation obstacle; "
                "higher points remain visible in map_view"
            ),
        },
        "map_view_enabled": {
            "type": "boolean",
            "default": True,
            "description": (
                "Publish the Canvas map view; disable on the next card start "
                "for a CPU A/B measurement"
            ),
        },
        "fault_capture_enabled": {
            "type": "boolean",
            "default": True,
            "description": (
                "Keep a short full-rate ring buffer and save it only on the "
                "first raw odometry discontinuity"
            ),
        },
        "collection_enabled": {
            "type": "boolean",
            "default": False,
            "description": (
                "Automatically record LiDAR, IMU, versioned RGB, depth and odom "
                "while this Canvas card is running, then generate obstacle "
                "annotations in the background"
            ),
        },
        "collection_directory": {
            "type": "string",
            "default": "/opt/phanthy-motus/data/fast_livo2/recordings",
            "description": (
                "Persistent in-container directory; it must remain under the "
                "mounted FAST-LIVO2 recordings root"
            ),
        },
    },
    "additionalProperties": False,
}

FAST_LIVO2_ACTION_PARAMS = {
    "info": {"params": [], "description": "Read mapping and adapter state"},
    "config": {
        "params": list(FAST_LIVO2_CONFIG_SCHEMA["properties"]),
        "description": "Validate idle-state configuration",
    },
    "start": {
        "params": ["instance_id", "input_topics", "input_bindings"],
        "description": "Validate LiDAR/IMU wiring and acquire ROS resources",
    },
    "stop": {"params": [], "description": "Stop mapping and release resources"},
    "start_mapping": {
        "params": ["map_name"],
        "description": "Start a new session-local FAST-LIVO2 map",
    },
    "stop_mapping": {
        "params": [],
        "description": (
            "Stop the active mapping session and transactionally finalize raw "
            "and confirmed-static PCD files; retry the same action after a "
            "retryable persistence error"
        ),
    },
    "load_map": {
        "params": ["map_name"],
        "description": (
            "Replace any loaded map, then load a saved PCD session and start "
            "a fresh localization frontend"
        ),
    },
    "relocalize": {
        "params": [
            "initial_x",
            "initial_y",
            "initial_z",
            "initial_yaw",
            "search_xy_m",
            "search_yaw_rad",
        ],
        "description": (
            "Align the current FAST-LIVO2 session to the loaded map near an "
            "operator-provided pose guess"
        ),
    },
}


def _root(namespace: str) -> str:
    normalized = namespace.strip("/")
    return f"/{normalized}" if normalized else ""


def fast_livo2_tool_definition(namespace: str) -> dict:
    root = _root(namespace)
    tool = {
        "name": "fast_livo2",
        "displayName": "FAST-LIVO2",
        "type": "processor",
        "multiInstance": False,
        "description": (
            "FAST-LIVO2 session mapping and localization adapter. It consumes "
            "Driver LiDAR/IMU topics and publishes canonical map-frame odometry, "
            "registered clouds and a Canvas map view for Nav2."
        ),
        "topic_in": [
            {
                "port": "lidar",
                "topic": f"{root}/navigation/lidar",
                "format": "sensor/pointcloud",
                "ros_type": "sensor_msgs/msg/PointCloud2",
                "qos": "RELIABLE + KEEP_LAST(depth=2) + VOLATILE",
                "rate_hz": 10,
                "timestamp": "ROS system time; per-point timestamp is absolute ns",
                "frame_id": "Driver-provided REP-103 sensor_frame",
                "units": "x/y/z=m; timestamp=absolute ns",
                "max_age_ms": 500,
                "desc": "Rigid navigation cloud prepared by the Driver sensor adapter",
            },
            {
                "port": "imu",
                "topic": f"{root}/navigation/imu",
                "format": "sensor/imu",
                "ros_type": "sensor_msgs/msg/Imu",
                "qos": "RELIABLE + KEEP_LAST(depth=200) + VOLATILE",
                "rate_hz": 200,
                "timestamp": "same ROS system clock as lidar",
                "frame_id": "same Driver-provided REP-103 sensor_frame as lidar",
                "units": "linear_acceleration=m/s^2; angular_velocity=rad/s",
                "max_age_ms": 500,
                "desc": "Navigation IMU aligned with the LiDAR stream",
            },
        ],
        "topic_out": [
            {
                "port": "livo_odom",
                "topic": f"{root}/navigation/odom",
                "format": "sensor/odometry",
                "ros_type": "nav_msgs/msg/Odometry",
                "qos": "BEST_EFFORT + KEEP_LAST(depth=5) + VOLATILE",
                "rate_hz": 10,
                "timestamp": "FAST-LIVO2 output stamp, ROS system time",
                "frame_id": "map -> base_link",
                "units": "position=m; velocity=m/s; angular=rad/s",
                "max_age_ms": 500,
                "desc": "Canonical base pose derived from Driver-owned base_link to sensor_frame TF",
            },
            {
                "port": "registered_cloud",
                "topic": f"{root}/navigation/cloud_registered",
                "format": "sensor/pointcloud",
                "ros_type": "sensor_msgs/msg/PointCloud2",
                "qos": "BEST_EFFORT + KEEP_LAST(depth=1) + VOLATILE",
                "rate_hz": 10,
                "timestamp": "FAST-LIVO2 output stamp, ROS system time",
                "frame_id": "map",
                "units": "x/y/z=m",
                "max_age_ms": 500,
                "desc": (
                    "Motion-compensated current scan in the session map frame, "
                    "filtered by the card-configured obstacle height band"
                ),
            },
            {
                "port": "obstacle_map",
                "topic": f"{root}/navigation/obstacle_map",
                "format": "sensor/pointcloud",
                "ros_type": "sensor_msgs/msg/PointCloud2",
                "qos": "BEST_EFFORT + KEEP_LAST(depth=1) + VOLATILE",
                "rate_hz": 1,
                "timestamp": "adapter publish time, ROS system time",
                "frame_id": "map",
                "units": "x/y=m; z=0 projected obstacle plane",
                "desc": (
                    "Directly accumulated static XY obstacle projection kept "
                    "for diagnostics and backward-compatible inspection"
                ),
            },
            {
                "port": "static_map",
                "topic": f"{root}/navigation/static_map",
                "format": "sensor/occupancy-grid",
                "ros_type": "nav_msgs/msg/OccupancyGrid",
                "qos": "RELIABLE + KEEP_LAST(depth=1) + TRANSIENT_LOCAL",
                "rate_hz": 1,
                "timestamp": "adapter publish time, ROS system time",
                "frame_id": "map",
                "units": "resolution=m/cell; data=-1 unknown, 0 free, 100 occupied",
                "desc": (
                    "Bounded static occupancy snapshot accumulated directly from "
                    "valid navigation-height registered-cloud voxels."
                ),
            },
            {
                "port": "map_view",
                "topic": f"{root}/navigation/fast_livo2/map_view",
                "format": "sensor/mapping",
                "ros_type": "std_msgs/msg/UInt8MultiArray",
                "qos": "BEST_EFFORT + KEEP_LAST(depth=1) + VOLATILE",
                "schema": "phanthy.navigation.map_view.v1",
                "rate_hz": 1,
                "frame_id": "map",
                "units": "x/y/z=m; yaw=rad",
                "desc": (
                    "Bounded height-balanced Canvas view of static obstacles, "
                    "visual-only out-of-band surfaces, the latest fresh scan, "
                    "current base pose and height metadata"
                ),
            },
            {
                "port": "status",
                "topic": f"{root}/navigation/fast_livo2/status",
                "format": "data/json",
                "ros_type": "std_msgs/msg/String",
                "qos": "RELIABLE + KEEP_LAST(depth=10) + TRANSIENT_LOCAL",
                "schema": "phanthy.navigation.fast_livo2_status.v1",
                "rate_hz": 1,
                "desc": "Lifecycle, source freshness, frame validation and artifact status",
            },
            {
                "port": "collection_status",
                "topic": f"{root}/navigation/fast_livo2/collection_preview",
                "format": "image/jpeg",
                "ros_type": "sensor_msgs/msg/CompressedImage",
                "qos": "RELIABLE + KEEP_LAST(depth=1) + TRANSIENT_LOCAL",
                "schema": "phanthy.navigation.collection_preview.v1",
                "rate_hz": 1,
                "desc": (
                    "Latest synchronized RGB sample with its frame number and "
                    "LiDAR nearest-visible-point distance markers while recording; "
                    "switches to export progress after the card stops"
                ),
            },
        ],
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(FAST_LIVO2_PUBLIC_ACTIONS),
                },
                "map_name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 64,
                    "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$",
                    "description": "Logical name recorded in the PCD session manifest",
                },
                "initial_x": {
                    "type": "number",
                    "minimum": -1000.0,
                    "maximum": 1000.0,
                    "description": "Approximate robot X in the loaded map, metres",
                },
                "initial_y": {
                    "type": "number",
                    "minimum": -1000.0,
                    "maximum": 1000.0,
                    "description": "Approximate robot Y in the loaded map, metres",
                },
                "initial_z": {
                    "type": "number",
                    "minimum": -10.0,
                    "maximum": 10.0,
                    "default": 0.0,
                    "description": "Approximate robot Z in the loaded map, metres",
                },
                "initial_yaw": {
                    "type": "number",
                    "minimum": -3.141592653589793,
                    "maximum": 3.141592653589793,
                    "description": "Approximate robot yaw in the loaded map, radians",
                },
                "search_xy_m": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 3.0,
                    "default": 1.0,
                    "description": "Bounded XY search radius around the initial guess",
                },
                "search_yaw_rad": {
                    "type": "number",
                    "minimum": 0.05,
                    "maximum": 1.5707963267948966,
                    "default": 0.35,
                    "description": "Bounded yaw search radius around the initial guess",
                },
                "instance_id": {"type": "string"},
                "input_topic": {"type": "string"},
                "input_topics": {"type": "array", "items": {"type": "string"}},
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
                **deepcopy(FAST_LIVO2_CONFIG_SCHEMA["properties"]),
            },
            "required": ["action"],
            "x-action-params": FAST_LIVO2_ACTION_PARAMS,
        },
        "configSchema": deepcopy(FAST_LIVO2_CONFIG_SCHEMA),
    }
    return deepcopy(tool)


__all__ = [
    "FAST_LIVO2_ACTIONS",
    "FAST_LIVO2_CONFIG_DEFAULTS",
    "FAST_LIVO2_CONFIG_SCHEMA",
    "fast_livo2_tool_definition",
]
