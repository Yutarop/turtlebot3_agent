import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    gazebo_launch_dir = os.path.join(
        get_package_share_directory("turtlebot3_gazebo"), "launch"
    )
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_launch_dir, "turtlebot3_world.launch.py")
        )
    )

    ekf_config_file_path = os.path.join(
        get_package_share_directory("turtlebot3_agent"), "config", "ekf_turtlebot3.yaml"
    )

    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[{"use_sim_time": True}, ekf_config_file_path],
    )

    return LaunchDescription([gazebo_launch, ekf_node])
