"""Planner/controller-only Nav2 bringup consuming FAST-LIVO2 outputs."""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("g1_nav2")
    nav2_share = get_package_share_directory("nav2_bringup")

    params_file = LaunchConfiguration("params_file")
    odom_topic = LaunchConfiguration("odom_topic")
    obstacle_cloud_topic = LaunchConfiguration("obstacle_cloud_topic")
    cmd_vel_raw_topic = LaunchConfiguration("cmd_vel_raw_topic")
    cmd_vel_shadow_topic = LaunchConfiguration("cmd_vel_shadow_topic")
    velocity_proposal_topic = LaunchConfiguration("velocity_proposal_topic")
    command_topic = LaunchConfiguration("command_topic")
    status_topic = LaunchConfiguration("status_topic")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=os.path.join(
                    package_share, "config", "nav2_params.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "odom_topic", default_value="/ubuntu/navigation/odom"
            ),
            DeclareLaunchArgument(
                "obstacle_cloud_topic",
                default_value="/ubuntu/navigation/cloud_registered",
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
            SetRemap(src="/cmd_vel", dst=cmd_vel_raw_topic),
            SetRemap(src="cmd_vel", dst=cmd_vel_raw_topic),
            SetRemap(src="/cmd_vel_smoothed", dst=cmd_vel_shadow_topic),
            SetRemap(src="cmd_vel_smoothed", dst=cmd_vel_shadow_topic),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(nav2_share, "launch", "navigation_launch.py")
                ),
                launch_arguments={
                    "use_sim_time": "false",
                    "params_file": params_file,
                    "autostart": "true",
                    "use_composition": "False",
                }.items(),
            ),
            Node(
                package="g1_nav2",
                executable="planner_command_bridge",
                name="g1_nav2_planner_command",
                output="screen",
                parameters=[
                    {
                        "command_topic": command_topic,
                        "status_topic": status_topic,
                        "action_name": "/navigate_to_pose",
                        "shadow_topic": cmd_vel_shadow_topic,
                        "proposal_topic": velocity_proposal_topic,
                        "controller_speed_limit_topic": (
                            "/ubuntu/navigation/nav2/speed_limit"
                        ),
                        "speed_limit_timeout": 3.0,
                        "behavior_tree_path": os.path.join(
                            package_share,
                            "behavior_trees",
                            "navigate_to_pose_w_replanning_and_recovery.xml",
                        ),
                        "proposal_ttl_ms": 250,
                        "enforce_shadow_isolation": True,
                        "max_shadow_speed": 0.15,
                        "supported_mode": 0,
                        "goal_response_timeout": 8.0,
                        "global_frame": "map",
                        "base_frame": "base_link",
                        "odom_topic": odom_topic,
                        "obstacle_cloud_topic": obstacle_cloud_topic,
                        "sensor_max_age_sec": 0.5,
                    }
                ],
            ),
        ]
    )
