from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="g1_fast_livo2",
                executable="frame_adapter",
                name="g1_fast_livo2_adapter",
                output="screen",
            ),
            Node(
                package="g1_fast_livo2",
                executable="runtime_supervisor",
                name="g1_fast_livo2_supervisor",
                output="screen",
            ),
        ]
    )
