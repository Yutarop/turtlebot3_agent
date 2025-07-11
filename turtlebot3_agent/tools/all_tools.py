from langgraph.graph import MessagesState

from turtlebot3_agent.tools.motion_tools import (
    make_detect_traffic_cone_tool,
    make_navigate_to_pose_tool,
    make_transform_odom_to_map_tool,
)
from turtlebot3_agent.tools.sensor_tools import (
    make_get_lidar_scan_tool,
    make_start_camera_display_tool,
)
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
        make_detect_traffic_cone_tool(node),
        make_transform_odom_to_map_tool(node),
        make_start_camera_display_tool(node),
        make_navigate_to_pose_tool(node),
        make_get_turtle_pose_tool(node),
        make_get_lidar_scan_tool(node),
    ]
