#!/usr/bin/env python3

from langchain.tools import Tool


def make_get_turtle_pose_tool(node):
    """
    Create a tool that returns the current pose of the TurtleBot3.

    Args:
        node: The TB3Agent node instance

    Returns:
        Tool: A LangChain tool for getting the robot's pose
    """

    def inner(_input: str = "") -> str:
        """
        Returns the current pose of the TurtleBot3 by subscribing to the /odom topic.
        The input is unused but required by LangChain.
        """
        if not node.wait_for_pose(timeout=0.5):
            result = "Could not retrieve pose for the turtle bot."
        else:
            result = (
                f"Current Pose of the turtlebot: x={node.x:.3f}, y={node.y:.3f}, "
                f"theta={node.yaw:.3f} rad, "
            )
        return result

    return Tool.from_function(
        func=inner,
        name="get_turtle_pose",
        description="""
        Returns the current pose of the TurtleBot3 by subscribing to the /odom topic.

        Args:
        - _input: the input is unused but required by LangChain. Set an empty string "".

        Returns:
        - Position: x (meters), y (meters)
        - Orientation: yaw (in radians)
        """,
    )


# @tool
def get_battery_level():
    pass
