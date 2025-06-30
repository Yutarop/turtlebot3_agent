#!/usr/bin/env python3

import math

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel, Field
from transforms3d.euler import euler2quat


def make_navigate_to_pose_tool(node):
    """
    Create a tool that uses Nav2 NavigateToPose action to navigate to a goal position.

    Args:
        node: The TB3Agent node instance

    Returns:
        StructuredTool: A LangChain tool for navigation
    """

    class NavigateInput(BaseModel):
        x: float = Field(description="Goal X coordinate in meters (map frame)")
        y: float = Field(description="Goal Y coordinate in meters (map frame)")
        yaw: float = Field(description="Goal yaw angle in radians")

    def inner(inputs: NavigateInput) -> str:
        success = node.navigate_to_pose(x=inputs.x, y=inputs.y, yaw=inputs.yaw)

        if success:
            if not node.wait_for_pose(timeout=0.5):
                return "Navigation completed, but pose data not received yet."
            else:
                map_x, map_y, map_yaw = node.transform_odom_to_map(
                    odom_x=node.x, odom_y=node.y, odom_yaw=node.yaw
                )
                return (
                    f"Navigation completed successfully! "
                    f"You are now at position ({round(map_x, 2)}, {round(map_y, 2)}) in the map frame"
                    f"with a yaw angle of {round(map_yaw, 3)} radians."
                )
        else:
            return "Navigation failed. The goal may be unreachable or the action server is not available."

    return StructuredTool.from_function(
        func=inner,
        name="navigate_to_pose",
        description="""
        Navigate to a specific goal position using Nav2 path planning and obstacle avoidance.

        Args:
        - x (float): Goal X coordinate in meters (map frame)
        - y (float): Goal Y coordinate in meters (map frame)  
        - yaw (float): Goal yaw angle in radians

        This tool uses the Nav2 navigation stack to plan a path and navigate to the goal
        while avoiding obstacles. It's more robust than direct movement commands for
        navigating in complex environments with obstacles.
        """,
    )


def make_transform_odom_to_map_tool(node):
    """
    Create a tool that transforms coordinates from odom frame to map frame.

    Args:
        node: The TB3Agent node instance

    Returns:
        StructuredTool: A LangChain tool for coordinate transformation
    """

    class TransformInput(BaseModel):
        odom_x: float = Field(description="X coordinate in pose frame (meters)")
        odom_y: float = Field(description="Y coordinate in pose frame (meters)")
        odom_yaw: float = Field(description="Yaw angle in pose frame (radians)")

    def inner(inputs: TransformInput) -> str:
        try:
            map_x, map_y, map_yaw = node.transform_odom_to_map(
                odom_x=inputs.odom_x, odom_y=inputs.odom_y, odom_yaw=inputs.odom_yaw
            )
            return (
                f"Transformed coordinates from pose to map frame:\n"
                f"Odom: ({round(inputs.odom_x, 2)}, {round(inputs.odom_y, 2)}, {round(inputs.odom_yaw, 2)} rad)\n"
                f"Map:  ({round(map_x, 2)}, {round(map_y, 2)}, {round(map_yaw, 2)} rad)"
            )
        except Exception as e:
            return f"Failed to transform coordinates: {str(e)}"

    return StructuredTool.from_function(
        func=inner,
        name="transform_odom_to_map",
        description="""
        Transform coordinates from pose frame to map frame using TF2.

        Args:
        - odom_x (float): X coordinate in robot's current frame (meters)
        - odom_y (float): Y coordinate in robot's current frame (meters)
        - odom_yaw (float): Yaw angle in robot's current frame (radians)

        Returns the equivalent coordinates in the map frame. This is useful for
        converting robot's current position from odometry to map coordinates
        for navigation planning.
        """,
    )
