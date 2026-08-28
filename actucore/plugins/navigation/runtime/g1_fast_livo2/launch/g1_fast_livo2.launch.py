from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    defaults = {
        "lidar_topic": "/ubuntu/navigation/lidar",
        "imu_topic": "/ubuntu/navigation/imu",
        "rgb_topic": "/ubuntu/camera/rgb_frame",
        "depth_topic": "/ubuntu/camera/depth_frame",
        "raw_odom_topic": "/ubuntu/navigation/fast_livo2/raw/odom",
        "raw_cloud_topic": "/ubuntu/navigation/fast_livo2/raw/cloud_registered",
        "odom_topic": "/ubuntu/navigation/odom",
        "cloud_topic": "/ubuntu/navigation/cloud_registered",
        "obstacle_map_topic": "/ubuntu/navigation/obstacle_map",
        "static_map_topic": "/ubuntu/navigation/static_map",
        "map_view_topic": "/ubuntu/navigation/fast_livo2/map_view",
        "diagnostics_topic": "/ubuntu/navigation/fast_livo2/diagnostics",
        "reset_topic": "/ubuntu/navigation/fast_livo2/reset_map",
        "map_control_topic": "/ubuntu/navigation/fast_livo2/map_control",
        "map_control_status_topic": (
            "/ubuntu/navigation/fast_livo2/map_control_status"
        ),
        "command_topic": "/ubuntu/navigation/fast_livo2/command",
        "status_topic": "/ubuntu/navigation/fast_livo2/status",
        "collection_status_topic": (
            "/ubuntu/navigation/fast_livo2/collection_status_raw"
        ),
    }
    topics = {name: LaunchConfiguration(name) for name in defaults}
    return LaunchDescription(
        [
            *[
                DeclareLaunchArgument(name, default_value=value)
                for name, value in defaults.items()
            ],
            Node(
                package="g1_fast_livo2",
                executable="frame_adapter",
                name="g1_fast_livo2_adapter",
                output="screen",
                parameters=[
                    {
                        key: topics[key]
                        for key in (
                            "raw_odom_topic",
                            "raw_cloud_topic",
                            "odom_topic",
                            "cloud_topic",
                            "obstacle_map_topic",
                            "static_map_topic",
                            "map_view_topic",
                            "diagnostics_topic",
                            "reset_topic",
                            "map_control_topic",
                            "map_control_status_topic",
                        )
                    }
                ],
            ),
            Node(
                package="g1_fast_livo2",
                executable="runtime_supervisor",
                name="g1_fast_livo2_supervisor",
                output="screen",
                parameters=[
                    {
                        key: topics[key]
                        for key in (
                            "lidar_topic",
                            "imu_topic",
                            "rgb_topic",
                            "depth_topic",
                            "odom_topic",
                            "raw_odom_topic",
                            "raw_cloud_topic",
                            "diagnostics_topic",
                            "reset_topic",
                            "map_control_topic",
                            "map_control_status_topic",
                            "command_topic",
                            "status_topic",
                            "collection_status_topic",
                        )
                    }
                ],
            ),
        ]
    )
