from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="robot_supervisor",
            executable="commander",
            output="screen"
        ),
        Node(
            package="robot_supervisor",
            executable="voice_chat",
            output="screen"
        ),
    ])

