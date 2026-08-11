"""Nav2-only bringup for G1 mapping or saved-map localization."""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, SetRemap


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("g1_nav2")
    nav2_share = get_package_share_directory("nav2_bringup")
    slam_share = get_package_share_directory("slam_toolbox")

    mode = LaunchConfiguration("mode")
    map_yaml = LaunchConfiguration("map")
    map_name = LaunchConfiguration("map_name")
    maps_root = LaunchConfiguration("maps_root")
    params_file = LaunchConfiguration("params_file")
    slam_params_file = LaunchConfiguration("slam_params_file")
    wrapped_cloud_topic = LaunchConfiguration("wrapped_cloud_topic")
    cloud_topic = LaunchConfiguration("cloud_topic")
    scan_topic = LaunchConfiguration("scan_topic")
    odom_input_topic = LaunchConfiguration("odom_input_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    cmd_vel_raw_topic = LaunchConfiguration("cmd_vel_raw_topic")
    cmd_vel_shadow_topic = LaunchConfiguration("cmd_vel_shadow_topic")
    velocity_proposal_topic = LaunchConfiguration("velocity_proposal_topic")
    command_topic = LaunchConfiguration("command_topic")
    status_topic = LaunchConfiguration("status_topic")

    common_nav_args = {
        "use_sim_time": "false",
        "params_file": params_file,
        "autostart": "true",
        "use_composition": "False",
    }

    sensor_adapters = [
        Node(
            package="g1_nav2",
            executable="canvas_pointcloud_bridge",
            name="g1_canvas_pointcloud_bridge",
            output="screen",
            parameters=[
                {
                    "input_topic": wrapped_cloud_topic,
                    "output_topic": cloud_topic,
                    "legacy_frame_id": "livox_frame",
                }
            ],
        ),
        Node(
            package="g1_nav2",
            executable="loco_odom_bridge",
            name="g1_loco_odom_bridge",
            output="screen",
            parameters=[
                {
                    "input_topic": odom_input_topic,
                    "odom_topic": odom_topic,
                    "status_topic": "/ubuntu/navigation/nav2/odom_status",
                    "odom_frame": "odom",
                    "base_frame": "base_link",
                    "reset_origin": True,
                    "velocity_frame": "body",
                    "publish_tf": True,
                    "source_timeout": 0.5,
                }
            ],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="g1_livox_static_tf",
            arguments=[
                "--x", LaunchConfiguration("lidar_x"),
                "--y", LaunchConfiguration("lidar_y"),
                "--z", LaunchConfiguration("lidar_z"),
                "--roll", LaunchConfiguration("lidar_roll"),
                "--pitch", LaunchConfiguration("lidar_pitch"),
                "--yaw", LaunchConfiguration("lidar_yaw"),
                "--frame-id", "base_link",
                "--child-frame-id", "livox_frame",
            ],
        ),
        Node(
            package="pointcloud_to_laserscan",
            executable="pointcloud_to_laserscan_node",
            name="g1_pointcloud_to_laserscan",
            remappings=[
                ("cloud_in", cloud_topic),
                ("scan", scan_topic),
            ],
            parameters=[
                {
                    "target_frame": "livox_frame",
                    "transform_tolerance": 0.05,
                    "min_height": -0.20,
                    "max_height": 0.25,
                    "angle_min": -3.141592653589793,
                    "angle_max": 3.141592653589793,
                    "angle_increment": 0.008726646259971648,
                    "scan_time": 0.1,
                    "range_min": 0.35,
                    "range_max": 12.0,
                    "use_inf": True,
                    "inf_epsilon": 1.0,
                    "queue_size": 10,
                }
            ],
        ),
        Node(
            package="g1_nav2",
            executable="navigation_command_bridge",
            name="g1_nav2_navigation_command",
            output="screen",
            parameters=[
                {
                    "command_topic": command_topic,
                    "status_topic": status_topic,
                    "runtime_switch_topic": "/ubuntu/navigation/nav2/runtime_switch",
                    "action_name": "/navigate_to_pose",
                    "shadow_topic": cmd_vel_shadow_topic,
                    "proposal_topic": velocity_proposal_topic,
                    "proposal_ttl_ms": 250,
                    "enforce_shadow_isolation": True,
                    "max_shadow_speed": 0.15,
                    "supported_mode": 0,
                    "goal_response_timeout": 8.0,
                    "runtime_mode": mode,
                    "maps_root": maps_root,
                    "startup_map_name": map_name,
                    "service_timeout": 20.0,
                    "pose_lookup_timeout": 2.0,
                    "odom_status_topic": "/ubuntu/navigation/nav2/odom_status",
                    "scan_topic": scan_topic,
                    "sensor_max_age_sec": 0.5,
                }
            ],
        ),
    ]

    mapping = GroupAction(
        condition=IfCondition(
            PythonExpression(["'", mode, "' == 'mapping'"])
        ),
        actions=[
            SetRemap(src="/cmd_vel", dst=cmd_vel_raw_topic),
            SetRemap(src="cmd_vel", dst=cmd_vel_raw_topic),
            SetRemap(src="/cmd_vel_smoothed", dst=cmd_vel_shadow_topic),
            SetRemap(src="cmd_vel_smoothed", dst=cmd_vel_shadow_topic),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(nav2_share, "launch", "navigation_launch.py")
                ),
                launch_arguments=common_nav_args.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(slam_share, "launch", "online_async_launch.py")
                ),
                launch_arguments={
                    "use_sim_time": "false",
                    "slam_params_file": slam_params_file,
                }.items(),
            ),
        ],
    )

    localization = GroupAction(
        condition=IfCondition(
            PythonExpression(["'", mode, "' == 'localization'"])
        ),
        actions=[
            SetRemap(src="/cmd_vel", dst=cmd_vel_raw_topic),
            SetRemap(src="cmd_vel", dst=cmd_vel_raw_topic),
            SetRemap(src="/cmd_vel_smoothed", dst=cmd_vel_shadow_topic),
            SetRemap(src="cmd_vel_smoothed", dst=cmd_vel_shadow_topic),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(nav2_share, "launch", "bringup_launch.py")
                ),
                launch_arguments={
                    **common_nav_args,
                    "slam": "False",
                    "map": map_yaml,
                }.items(),
            ),
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "mode",
                default_value="mapping",
                description="mapping or localization",
            ),
            DeclareLaunchArgument(
                "map",
                default_value="",
                description="saved map.yaml path required in localization mode",
            ),
            DeclareLaunchArgument(
                "map_name",
                default_value="",
                description="saved map name required in localization mode",
            ),
            DeclareLaunchArgument("maps_root", default_value="/maps"),
            DeclareLaunchArgument(
                "params_file",
                default_value=os.path.join(
                    package_share, "config", "nav2_params.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "slam_params_file",
                default_value=os.path.join(
                    package_share, "config", "slam_toolbox.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "wrapped_cloud_topic",
                default_value="/ubuntu/lidar/cloud",
            ),
            DeclareLaunchArgument(
                "cloud_topic",
                default_value="/ubuntu/navigation/nav2/cloud",
            ),
            DeclareLaunchArgument(
                "scan_topic",
                default_value="/ubuntu/navigation/nav2/scan",
            ),
            DeclareLaunchArgument(
                "odom_input_topic", default_value="/ubuntu/loco/state"
            ),
            DeclareLaunchArgument(
                "odom_topic", default_value="/ubuntu/navigation/nav2/odom"
            ),
            DeclareLaunchArgument(
                "cmd_vel_raw_topic",
                default_value="/ubuntu/navigation/nav2/cmd_vel_raw",
            ),
            DeclareLaunchArgument(
                "cmd_vel_shadow_topic",
                default_value="/ubuntu/navigation/nav2/cmd_vel_shadow",
            ),
            DeclareLaunchArgument(
                "velocity_proposal_topic",
                default_value="/ubuntu/navigation/nav2/velocity_proposal",
            ),
            DeclareLaunchArgument(
                "command_topic",
                default_value="/ubuntu/navigation/nav2/command",
            ),
            DeclareLaunchArgument(
                "status_topic",
                default_value="/ubuntu/navigation/nav2/status",
            ),
            DeclareLaunchArgument(
                "lidar_x", description="Required measured base_link -> lidar x (m)"
            ),
            DeclareLaunchArgument(
                "lidar_y", description="Required measured base_link -> lidar y (m)"
            ),
            DeclareLaunchArgument(
                "lidar_z", description="Required measured base_link -> lidar z (m)"
            ),
            DeclareLaunchArgument(
                "lidar_roll", description="Required measured base_link -> lidar roll (rad)"
            ),
            DeclareLaunchArgument(
                "lidar_pitch", description="Required measured base_link -> lidar pitch (rad)"
            ),
            DeclareLaunchArgument(
                "lidar_yaw", description="Required measured base_link -> lidar yaw (rad)"
            ),
            *sensor_adapters,
            mapping,
            localization,
        ]
    )
