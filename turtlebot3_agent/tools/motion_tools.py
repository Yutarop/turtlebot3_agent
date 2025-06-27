#!/usr/bin/env python3

import math

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel, Field
from transforms3d.euler import euler2quat


def make_move_linear_tool(node):
    """
    Create a tool that publishes Twist commands to move the TurtleBot3.

    Args:
        node: The TB3Agent node instance

    Returns:
        StructuredTool: A LangChain tool for sending movement commands
    """

    class MoveInput(BaseModel):
        distance: float = Field(
            description="Linear distance in meters. Positive for forward, negative for backward."
        )

    def inner(inputs: MoveInput) -> str:
        if inputs.distance != 0.0:
            node.move_linear(distance_m=inputs.distance)

        if not node.wait_for_pose(timeout=0.5):
            return "Pose data not received yet."
        return (
            f"You are at position ({round(node.x, 2)}, {round(node.y, 2)}) "
            f"with a yaw angle of {round(node.yaw, 3)} degrees."
        )

    return StructuredTool.from_function(
        func=inner,
        name="move_linear",
        description="""
        Moves the TurtleBot3 forward or backward by the specified distance.

        Args:
        - distance (float): Linear distance in meters. Positive for forward, negative for backward.
        """,
    )


def make_rotate_tool(node):
    """
    Create a tool that publishes Twist commands to rotate the TurtleBot3.

    Args:
        node: The TB3Agent node instance

    Returns:
        StructuredTool: A LangChain tool for sending movement commands
    """

    class RotateInput(BaseModel):
        angle: float = Field(
            description="Relative rotation in radians. Positive to turn left (counterclockwise), negative to turn right (clockwise)."
        )

    def inner(inputs: RotateInput) -> str:
        if inputs.angle != 0.0:
            node.rotate_angle(angle_rad=inputs.angle)

        if not node.wait_for_pose(timeout=0.5):
            return "Pose data not received yet."
        return (
            f"You are at position ({round(node.x, 2)}, {round(node.y, 2)}) "
            f"with a yaw angle of {round(node.yaw, 3)} degrees."
        )

    return StructuredTool.from_function(
        func=inner,
        name="rotate_robot",
        description="""
        Rotates the TurtleBot3 by the specified angle (in radians).

        Args:
        - angle (float): Relative rotation in radians. Positive for left (counterclockwise), negative for right (clockwise).
        """,
    )


def make_move_non_linear_tool(node):
    """
    Create a LangChain tool for executing curved movement.

    Args:
        node: The TB3Agent node instance

    Returns:
        StructuredTool: LangChain-compatible tool
    """

    class MoveCurveInput(BaseModel):
        linear_velocity: float = Field(
            description="Forward/backward speed in m/s. Positive to move forward, negative to move backward."
        )
        angular_velocity: float = Field(
            description="Rotational speed in rad/s. Positive to turn left (counterclockwise), negative to turn right (clockwise)."
        )
        duration_sec: float = Field(description="Duration of movement in seconds.")

    def inner(inputs: MoveCurveInput) -> str:
        node.move_non_linear(
            linear_velocity=inputs.linear_velocity,
            angular_velocity=inputs.angular_velocity,
            duration_sec=inputs.duration_sec,
        )
        if not node.wait_for_pose(timeout=0.5):
            return "Pose data not received yet."
        return (
            f"You are at position ({round(node.x, 2)}, {round(node.y, 2)}) "
            f"with a yaw angle of {round(node.yaw, 3)} degrees."
        )

    return StructuredTool.from_function(
        func=inner,
        name="move_non_linear",
        description="""
        Moves the TurtleBot3 in a curved trajectory.

        Args:
        - linear_velocity (m/s): Forward/backward speed. Positive for forward motion. Should be less than 0.21m/s.
        - angular_velocity (rad/s): Rotational speed. Positive for left (CCW) turns. Should be less than 0.3rad/s.
        - duration_sec (s): How long the motion lasts.

        Use this to make the robot move in arcs or spirals.
        """,
    )


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
            return (
                f"Navigation completed successfully! "
                f"You are now at position ({round(node.x, 2)}, {round(node.y, 2)}) "
                f"with a yaw angle of {round(node.yaw, 3)} radians."
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
        odom_x: float = Field(description="X coordinate in odom frame (meters)")
        odom_y: float = Field(description="Y coordinate in odom frame (meters)")
        odom_yaw: float = Field(description="Yaw angle in odom frame (radians)")

    def inner(inputs: TransformInput) -> str:
        try:
            map_x, map_y, map_yaw = node.transform_odom_to_map(
                odom_x=inputs.odom_x, odom_y=inputs.odom_y, odom_yaw=inputs.odom_yaw
            )
            return (
                f"Transformed coordinates from odom to map frame:\n"
                f"Odom: ({round(inputs.odom_x, 3)}, {round(inputs.odom_y, 3)}, {round(inputs.odom_yaw, 3)} rad)\n"
                f"Map:  ({round(map_x, 3)}, {round(map_y, 3)}, {round(map_yaw, 3)} rad)"
            )
        except Exception as e:
            return f"Failed to transform coordinates: {str(e)}"

    return StructuredTool.from_function(
        func=inner,
        name="transform_odom_to_map",
        description="""
        Transform coordinates from odom frame to map frame using TF2.

        Args:
        - odom_x (float): X coordinate in robot's current frame (meters)
        - odom_y (float): Y coordinate in robot's current frame (meters)
        - odom_yaw (float): Yaw angle in robot's current frame (radians)

        Returns the equivalent coordinates in the map frame. This is useful for
        converting robot's current position from odometry to map coordinates
        for navigation planning.
        """,
    )
