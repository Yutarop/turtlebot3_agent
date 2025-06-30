from turtlebot3_agent.tools.motion_tools import (
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
        make_transform_odom_to_map_tool(node),
        make_start_camera_display_tool(node),
        make_navigate_to_pose_tool(node),
        make_get_turtle_pose_tool(node),
    ]


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


#!/usr/bin/env python3

import threading

import cv2
from cv_bridge import CvBridge
from langchain.tools import StructuredTool


def make_start_camera_display_tool(node):
    """
    Create a tool for starting continuous camera display from TurtleBot3's image_raw topic.
    This version keeps displaying images until manually stopped.

    Args:
        node: The TB3Agent node instance

    Returns:
        StructuredTool: A LangChain tool that starts continuous camera display
    """

    def inner(_input: str = "") -> str:
        if not hasattr(node, "camera_image"):
            return "Camera image topic is not subscribed. Make sure the node subscribes to image_raw topic."

        try:

            def continuous_display():
                cv2.namedWindow("TurtleBot3 Camera - Continuous", cv2.WINDOW_AUTOSIZE)
                bridge = CvBridge()

                while True:
                    if hasattr(node, "camera_image") and node.camera_image is not None:
                        try:
                            cv_image = bridge.imgmsg_to_cv2(node.camera_image, "bgr8")
                            cv2.imshow("TurtleBot3 Camera - Continuous", cv_image)
                        except Exception as e:
                            print(f"Error converting image: {e}")

                    key = cv2.waitKey(30) & 0xFF
                    if (
                        key == ord("q")
                        or cv2.getWindowProperty(
                            "TurtleBot3 Camera - Continuous", cv2.WND_PROP_VISIBLE
                        )
                        < 1
                    ):
                        break

                cv2.destroyWindow("TurtleBot3 Camera - Continuous")

            # Start continuous display in a separate thread
            display_thread = threading.Thread(target=continuous_display)
            display_thread.daemon = True
            display_thread.start()

            return "Continuous camera display started. Press 'q' in the image window to stop."

        except Exception as e:
            return f"Error starting camera display: {str(e)}"

    return StructuredTool.from_function(
        func=inner,
        name="start_continuous_camera_display",
        description="""
        Starts continuous display of camera images from the TurtleBot3's image_raw topic.
        The display will continue until 'q' key is pressed in the image window.

        Args:
        - _input: the input is unused but required by LangChain. Set an empty string "".
        
        Returns:
        - Success message if continuous display is started.
        - Error message if camera data is not available or display fails.
        """,
    )
