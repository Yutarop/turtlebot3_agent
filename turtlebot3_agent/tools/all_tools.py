from turtlebot3_agent.tools.motion_tools import (
    make_move_linear_tool,
    make_rotate_tool,
    make_transform_odom_to_map_tool,
)
from turtlebot3_agent.tools.sensor_tools import make_get_lidar_scan_tool
from turtlebot3_agent.tools.status_tools import make_get_turtle_pose_tool


def make_all_tools(node) -> list:
    """
    Creates all the tools needed for the LangChain agent.

    Args:
        node: The TB3Agent node instance

    Returns:
        list: List of tools available to the agent
    """
    return [
        make_move_linear_tool(node),
        make_rotate_tool(node),
        make_transform_odom_to_map_tool(node),
        make_get_turtle_pose_tool(node),
        make_get_lidar_scan_tool(node),
    ]
