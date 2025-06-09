#!/usr/bin/env python3
import math

from langchain.tools import tool

from tb3_agent.utils import normalize_angle


@tool
def degrees_to_radians(degrees: float) -> float:
    """
    Converts an angle in degrees to radians.

    Args:
        degrees (float): The angle in degrees.

    Returns:
        float: The angle in radians.
    """
    radians = round(math.radians(degrees), 3)
    return radians


@tool
def calculate_euclidean_distance(
    current_x: float,
    current_y: float,
    target_x: float,
    target_y: float,
) -> float:
    """
    Calculates the Euclidean distance between the current position and the target position.

    Args:
        current_x (float): X coordinate of the current position.
        current_y (float): Y coordinate of the current position.
        target_x (float): X coordinate of the target position.
        target_y (float): Y coordinate of the target position.

    Returns:
        float: Distance in meters.
    """
    dx = target_x - current_x
    dy = target_y - current_y
    return round(math.sqrt(dx**2 + dy**2), 3)


@tool
def calculate_absolute_angle(
    current_x: float,
    current_y: float,
    target_x: float,
    target_y: float,
) -> float:
    """
    Calculates the absolute angle (in radians) from the current position to the target position.

    Args:
        current_x (float): X coordinate of the current position.
        current_y (float): Y coordinate of the current position.
        target_x (float): X coordinate of the target position.
        target_y (float): Y coordinate of the target position.

    Returns:
        float: Absolute angle in radians from the current position to the target.
    """
    dx = target_x - current_x
    dy = target_y - current_y
    return round(math.atan2(dy, dx), 3)


@tool
def calculate_relative_angle_from_yaw(
    absolute_angle: float,
    current_yaw: float,
) -> float:
    """
    Converts an absolute angle to a relative angle based on the current yaw orientation.

    Args:
        absolute_angle (float): Absolute angle to the target (in radians).
        current_yaw (float): Current yaw angle of the robot (in radians).

    Returns:
        float: Relative angle in radians.
    """
    return round(normalize_angle(absolute_angle - current_yaw), 3)


@tool
def calculate_relative_angle(
    current_x: float,
    current_y: float,
    current_yaw: float,
    target_x: float,
    target_y: float,
) -> float:
    """
    Calculates the relative angle (in radians) from the current orientation
    to the direction of the target position, by combining absolute angle and yaw.

    Internally calls:
      - calculate_absolute_angle
      - calculate_relative_angle_from_yaw

    Args:
        current_x (float): X coordinate of the current position.
        current_y (float): Y coordinate of the current position.
        current_yaw (float): Current yaw angle in radians.
        target_x (float): X coordinate of the target position.
        target_y (float): Y coordinate of the target position.

    Returns:
        float: Relative angle in radians.
    """
    absolute_angle = calculate_absolute_angle.invoke(
        {
            "current_x": current_x,
            "current_y": current_y,
            "target_x": target_x,
            "target_y": target_y,
        }
    )
    relative_angle = calculate_relative_angle_from_yaw.invoke(
        {"absolute_angle": absolute_angle, "current_yaw": current_yaw}
    )
    return relative_angle


@tool
def calculate_distance_and_relative_angle(
    current_x: float,
    current_y: float,
    current_yaw: float,
    target_x: float,
    target_y: float,
) -> dict:
    """
    Calculates the straight-line distance and relative angle (in radians)
    from the robot's current position and orientation to a target coordinate.

    Internally calls:
      - calculate_euclidean_distance
      - calculate_absolute_angle
      - calculate_relative_angle_from_yaw

    Args:
        current_x (float): X coordinate of the robot's current position.
        current_y (float): Y coordinate of the robot's current position.
        current_yaw (float): yaw angle of the robot's current position.
        target_x (float): X coordinate of the target point.
        target_y (float): Y coordinate of the target point.

    Returns:
        dict: {
            "distance_m": float,
            "relative_angle_rad": float
        }
    """
    distance = calculate_euclidean_distance.invoke(
        {
            "current_x": current_x,
            "current_y": current_y,
            "target_x": target_x,
            "target_y": target_y,
        }
    )
    absolute_angle = calculate_absolute_angle.invoke(
        {
            "current_x": current_x,
            "current_y": current_y,
            "target_x": target_x,
            "target_y": target_y,
        }
    )
    relative_angle = calculate_relative_angle_from_yaw.invoke(
        {"absolute_angle": absolute_angle, "current_yaw": current_yaw}
    )

    return {
        "distance_m": round(distance, 3),
        "relative_angle_rad": round(relative_angle, 3),
    }
