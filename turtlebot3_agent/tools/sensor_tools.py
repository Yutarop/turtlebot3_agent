#!/usr/bin/env python3

import threading

import cv2
from cv_bridge import CvBridge
from langchain.tools import StructuredTool


def make_get_lidar_scan_tool(node):
    """
    Create a tool for checking whether there is an obstacle within 3 meters
    of the TurtleBot3 using LiDAR (LaserScan) data. If an obstacle is found,
    the function also describes the direction it is located (front, left, right, back).

    Args:
        node: The TB3Agent node instance

    Returns:
        StructuredTool: A LangChain tool that checks for nearby obstacles and their direction
    """

    def inner(_input: str = "") -> str:
        if node.scan is None or not hasattr(node.scan, "ranges"):
            return "LiDAR data is not available yet."

        ranges = node.scan.ranges
        len(ranges)

        direction_ranges = {
            "front": ranges[0:45] + ranges[-45:],
            "left": ranges[45:135],
            "right": ranges[225:315],
            "back": ranges[135:225],
        }

        detected_directions = []

        for direction, values in direction_ranges.items():
            if any(0.1 < dist < 3.0 for dist in values):
                detected_directions.append(direction)

        if detected_directions:
            direction_str = ", ".join(detected_directions)
            return f"Obstacles detected within 3 meters in the following direction(s): {direction_str}."
        else:
            return "No obstacles within 3 meters in any direction."

    return StructuredTool.from_function(
        func=inner,
        name="check_nearby_obstacles",
        description="""
        Checks whether there are any obstacles within 3 meters of the TurtleBot3 using LiDAR.

        Args:
        - _input: the input is unused but required by LangChain. Set an empty string "".

        Returns:
        - "Obstacles detected within 3 meters in the following direction(s): ..." if any are found.
        - "No obstacles within 3 meters in any direction." otherwise.
        """,
    )


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
