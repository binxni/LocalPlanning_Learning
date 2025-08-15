from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='local_planner',
            executable='planner_node',
            parameters=['config/planner_params.yaml']
        )
    ])
