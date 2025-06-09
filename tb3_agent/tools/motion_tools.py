#!/usr/bin/env python3

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel, Field


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
