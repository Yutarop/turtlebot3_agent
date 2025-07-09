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
        node.navigate_to_pose_async(x=inputs.x, y=inputs.y, yaw=inputs.yaw)

        node.wait_for_navigation_completion()

        if not node.wait_for_pose(timeout=0.5):
            return "Navigation completed, but pose data not received yet."
        else:
            map_x, map_y, map_yaw = node.transform_odom_to_map(
                odom_x=node.x, odom_y=node.y, odom_yaw=node.yaw
            )
            return (
                f"Navigation completed successfully! "
                f"You are now at position ({round(map_x, 2)}, {round(map_y, 2)}) in the map frame "
                f"with a yaw angle of {round(map_yaw, 3)} radians."
            )

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


import time


def make_change_goal_tool(node):
    """
    Create a tool that changes the navigation goal after a 5-second delay.
    Args:
        node: The TB3Agent node instance
    Returns:
        StructuredTool: A LangChain tool for changing navigation goal
    """

    class ChangeGoalInput(BaseModel):
        x: float = Field(description="New goal X coordinate in meters (map frame)")
        y: float = Field(description="New goal Y coordinate in meters (map frame)")
        yaw: float = Field(description="New goal yaw angle in radians")

    def inner(inputs: ChangeGoalInput) -> str:
        node.get_logger().info(
            f"Goal change scheduled in 5 seconds to: x={inputs.x}, y={inputs.y}, yaw={inputs.yaw}"
        )

        # 5秒待機
        time.sleep(5.0)

        # ゴールを変更
        node.change_goal(x=inputs.x, y=inputs.y, yaw=inputs.yaw)

        # ナビゲーションの完了を待機
        node.wait_for_navigation_completion()

        # ナビゲーション完了後の処理
        if not node.wait_for_pose(timeout=0.5):
            return f"Goal changed to ({inputs.x}, {inputs.y}, {inputs.yaw}) and navigation completed, but pose data not received yet."
        else:
            map_x, map_y, map_yaw = node.transform_odom_to_map(
                odom_x=node.x, odom_y=node.y, odom_yaw=node.yaw
            )
            return (
                f"Goal successfully changed after 5 seconds! "
                f"Navigation to new goal completed. "
                f"You are now at position ({round(map_x, 2)}, {round(map_y, 2)}) in the map frame "
                f"with a yaw angle of {round(map_yaw, 3)} radians."
            )

    return StructuredTool.from_function(
        func=inner,
        name="change_goal",
        description="""
        Change the current navigation goal after a 5-second delay.
        This tool cancels the current navigation goal and sets a new one.
        Args:
        - x (float): New goal X coordinate in meters (map frame)
        - y (float): New goal Y coordinate in meters (map frame)  
        - yaw (float): New goal yaw angle in radians
        
        Use this when you want to redirect the robot to a different destination
        while it's currently navigating. The tool waits 5 seconds before changing
        the goal to allow for any necessary preparations.
        """,
    )


def make_detect_traffic_cone_tool(node):
    """
    Create a tool for detecting traffic cones in camera images and changing goal when detected.

    Args:
        node: The TB3Agent node instance

    Returns:
        StructuredTool: A LangChain tool that detects traffic cones and changes goal
    """

    class DetectConeAndChangeGoalInput(BaseModel):
        goal_x: float = Field(description="New goal X coordinate in meters (map frame)")
        goal_y: float = Field(description="New goal Y coordinate in meters (map frame)")
        goal_yaw: float = Field(description="New goal yaw angle in radians")

    def inner(inputs: DetectConeAndChangeGoalInput) -> str:
        if not hasattr(node, "camera_image"):
            return "Camera image topic is not subscribed. Make sure the node subscribes to image_raw topic."

        if not hasattr(node, "cone_model") or node.cone_model is None:
            return (
                "YOLO cone detection model is not loaded. Please check the model path."
            )

        try:
            node.start_cone_detection()
            node.get_logger().info("🔍 Starting cone detection...")

            while True:
                if node.cone_detected is True:
                    node.is_cone_detection_active = False
                    node.cone_detected = False

                    node.get_logger().info(
                        f"✅ Traffic cone detected! Changing goal to ({inputs.goal_x}, {inputs.goal_y}, {inputs.goal_yaw})"
                    )

                    node.change_goal(
                        x=inputs.goal_x, y=inputs.goal_y, yaw=inputs.goal_yaw
                    )

                    node.wait_for_navigation_completion()

                    if not node.wait_for_pose(timeout=0.5):
                        return f"Traffic cone detected and goal changed to ({inputs.goal_x}, {inputs.goal_y}, {inputs.goal_yaw}), but pose data not received yet."
                    else:
                        map_x, map_y, map_yaw = node.transform_odom_to_map(
                            odom_x=node.x, odom_y=node.y, odom_yaw=node.yaw
                        )
                        return (
                            f"✅ Traffic cone detected!\n"
                            f"🚩 Goal changed to ({inputs.goal_x}, {inputs.goal_y}, {inputs.goal_yaw})\n"
                            f"🤖 Robot reached new goal at ({round(map_x, 2)}, {round(map_y, 2)}) with yaw {round(map_yaw, 3)} radians."
                        )

                time.sleep(0.1)

        except Exception as e:
            return f"❌ Error during cone detection: {str(e)}"

    return StructuredTool.from_function(
        func=inner,
        name="detect_traffic_cone_and_change_goal",
        description="""
        Detect traffic cones in the camera image. Once a cone is detected, automatically
        change the navigation goal to a specified new location.

        Args:
        - goal_x (float): New goal X coordinate in meters (map frame)
        - goal_y (float): New goal Y coordinate in meters (map frame)
        - goal_yaw (float): New goal yaw angle in radians

        Returns:
        - Message confirming cone detection and successful goal change
        """,
    )
