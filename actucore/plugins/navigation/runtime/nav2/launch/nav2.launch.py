"""Planner/controller-only Nav2 bringup consuming FAST-LIVO2 outputs."""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("nav2")
    nav2_share = get_package_share_directory("nav2_bringup")

    params_file = LaunchConfiguration("params_file")
    odom_topic = LaunchConfiguration("odom_topic")
    obstacle_cloud_topic = LaunchConfiguration("obstacle_cloud_topic")
    cmd_vel_raw_topic = LaunchConfiguration("cmd_vel_raw_topic")
    velocity_proposal_topic = LaunchConfiguration("velocity_proposal_topic")
    command_topic = LaunchConfiguration("command_topic")
    status_topic = LaunchConfiguration("status_topic")
    segment_status_topic = LaunchConfiguration("segment_status_topic")
    speed_limit_topic = LaunchConfiguration("speed_limit_topic")

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
                "segment_status_topic",
                default_value="/ubuntu/navigation/nav2/segment_status",
            ),
            DeclareLaunchArgument(
                "speed_limit_topic",
                default_value="/ubuntu/navigation/nav2/speed_limit",
            ),
            GroupAction(
                scoped=True,
                actions=[
                    SetRemap(src="/cmd_vel", dst=cmd_vel_raw_topic),
                    SetRemap(src="cmd_vel", dst=cmd_vel_raw_topic),
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            os.path.join(
                                nav2_share, "launch", "navigation_launch.py"
                            )
                        ),
                        launch_arguments={
                            "use_sim_time": "false",
                            "params_file": params_file,
                            "autostart": "true",
                            "use_composition": "False",
                        }.items(),
                    ),
                ],
            ),
            Node(
                package="nav2",
                executable="planner_command_bridge",
                name="nav2_planner_command",
                output="screen",
                parameters=[
                    {
                        "command_topic": command_topic,
                        "status_topic": status_topic,
                        "segment_status_topic": segment_status_topic,
                        "action_name": "/navigate_to_pose",
                        # The actuator consumes discrete 5 Hz proposals. Feeding the
                        # bridge from the OPEN_LOOP smoother adds a second,
                        # unobserved motion model and delays crossing the
                        # actuator's effective velocity deadbands.
                        "shadow_topic": cmd_vel_raw_topic,
                        "proposal_topic": velocity_proposal_topic,
                        "controller_speed_limit_topic": speed_limit_topic,
                        "speed_limit_timeout": 3.0,
                        "behavior_tree_path": os.path.join(
                            package_share,
                            "behavior_trees",
                            "navigate_to_pose_w_replanning_and_recovery.xml",
                        ),
                        "proposal_ttl_ms": 250,
                        "proposal_frequency_hz": 5.0,
                        "enforce_shadow_isolation": True,
                        "max_shadow_speed": 1.0,
                        "supported_mode": 0,
                        "goal_response_timeout": 8.0,
                        "global_frame": "map",
                        "base_frame": "base_link",
                        "odom_topic": odom_topic,
                        "obstacle_cloud_topic": obstacle_cloud_topic,
                        "global_costmap_topic": "/global_costmap/costmap",
                        "goal_costmap_max_age_sec": 2.0,
                        "sensor_max_age_sec": 0.8,
                        "sensor_source_max_age_sec": 1.0,
                        "control_odom_max_age_sec": 0.60,
                        "control_odom_source_max_age_sec": 0.80,
                    }
                ],
            ),
        ]
    )
