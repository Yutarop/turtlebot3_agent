#!/usr/bin/env python3
"""
TurtleBot3 agent node for ROS2.
This module defines the TB3Agent class which manages the robot's movement, navigation,
and sensor data processing.
"""
import math
import os
import threading
import time

import rclpy
import tf2_geometry_msgs
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped, TransformStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from tf2_ros import TransformException
from transforms3d.euler import euler2quat, quat2euler
from ultralytics import YOLO

from turtlebot3_agent.utils import normalize_angle

TOPIC_CMD_VEL = "/cmd_vel"
TOPIC_ODOMETRY = "/odom"
TOPIC_IMG = "/camera/image_raw"
TOPIC_SCAN = "/scan"
PUBLISH_RATE = 0.05


class TB3Agent(Node):
    def __init__(self):
        """Initialize the TB3Agent node with publishers and subscribers."""
        super().__init__("turtlebot3_agent")
        self.declare_parameter("interface", "cli")
        self.declare_parameter("agent_model", "gemini-2.0-flash")

        self.interface = (
            self.get_parameter("interface").get_parameter_value().string_value
        )
        self.agent_model = (
            self.get_parameter("agent_model").get_parameter_value().string_value
        )

        # Publishers and subscribers
        self.pub = self.create_publisher(Twist, TOPIC_CMD_VEL, 10)
        self.create_subscription(Odometry, TOPIC_ODOMETRY, self.odom_callback, 10)
        self.create_subscription(LaserScan, TOPIC_SCAN, self.scan_callback, 10)
        self.create_subscription(Image, TOPIC_IMG, self.image_callback, 10)

        # Nav2 Action Client
        self._nav_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")

        # TF2 Buffer and Listener for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Sensor data
        self.scan = None
        self.odom = None
        self.camera_image = None

        # variables
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.bridge = CvBridge()

        # Navigation parameters
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_is_active = False

        # Navigation function parameters
        self.min_distance = 0.12
        self.max_distance = 0.8
        self.k_angle = 0.7
        self.max_angular_speed = 0.3

        # Initialize YOLO model for cone detection
        try:
            package_share_dir = get_package_share_directory(
                "ros2_traffic_cone_detection"
            )
            model_path = os.path.join(package_share_dir, "models", "cone_detection.pt")
            self.cone_model = YOLO(model_path, verbose=False)
            # self.get_logger().info(f"YOLO model loaded from: {model_path}")
        except Exception as e:
            self.get_logger().warning(f"Could not load YOLO model: {e}")
            self.cone_model = None

        # State management for cone detection
        self.cone_detected = False
        self.is_cone_detection_active = False
        self.cone_detection_lock = threading.Lock()

        # Synchronization
        self.pose_ready = threading.Event()

    def scan_callback(self, msg):
        """Process incoming laser scan data."""
        self.scan = msg

    def odom_callback(self, msg):
        """Process incoming odometry data and extract pose information."""
        self.odom = msg
        self.x, self.y, self.yaw = self.get_pose(msg)
        self.pose_ready.set()

    def get_pose(self, msg):
        """
        Extract position and orientation from odometry message.

        Args:
            msg: Odometry message

        Returns:
            tuple: (x, y, yaw) position and orientation
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q_x = msg.pose.pose.orientation.x
        q_y = msg.pose.pose.orientation.y
        q_z = msg.pose.pose.orientation.z
        q_w = msg.pose.pose.orientation.w
        (_, _, yaw) = quat2euler((q_w, q_x, q_y, q_z))
        return x, y, yaw

    def wait_for_pose(self, timeout=1.0):
        """
        Wait for pose data to become available.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if pose data is available, False if timed out
        """
        return self.pose_ready.wait(timeout=timeout)

    def image_callback(self, msg):
        """Process incoming image data and detect cones if active"""
        self.camera_image = msg

        # Process only if cone detection is active and the model is available
        if self.is_cone_detection_active and self.cone_model is not None:
            self._detect_cones_in_image(msg)

    def start_cone_detection(self):
        self.is_cone_detection_active = True

    def _detect_cones_in_image(self, image_msg):
        """Detect cones in the image"""
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

            # Run YOLO inference
            results = self.cone_model(cv_image)

            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.cone_model.names[cls_id]

                if "cone" in cls_name.lower():
                    self.cone_detected = True
                    self.get_logger().info(f"detected")
                    self.is_cone_detection_active = False

        except Exception as e:
            self.get_logger().error(f"Error in cone detection: {e}")

    def get_latest_image(self) -> Image:
        if self.latest_image is None:
            raise RuntimeError("No image received yet.")
        return self.latest_image

    def navigate_to_pose_async(self, x, y, yaw, timeout_sec=30.0):
        """Start navigation asynchronously (from change_goal.py)"""
        self._navigation_thread = threading.Thread(
            target=self._navigate_to_pose, args=(x, y, yaw, timeout_sec)
        )
        self._navigation_thread.start()

    def _navigate_to_pose(self, x, y, yaw, timeout_sec=30.0):
        """Internal method: Execute asynchronous navigation (from change_goal.py)"""
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("NavigateToPose action server not available.")
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)

        quat = euler2quat(0, 0, yaw)  # (x, y, z, w)
        goal_msg.pose.pose.orientation.x = float(quat[0])
        goal_msg.pose.pose.orientation.y = float(quat[1])
        goal_msg.pose.pose.orientation.z = float(quat[2])
        goal_msg.pose.pose.orientation.w = float(quat[3])

        self.get_logger().info(f"Sending async goal: x={x}, y={y}, yaw={yaw}")
        send_goal_future = self._nav_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return False

        self._current_goal_handle = goal_handle
        self.get_logger().info("Goal accepted, waiting for result...")

        try:
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(
                self, result_future, timeout_sec=timeout_sec
            )
            if not result_future.done():
                self.get_logger().warning("Navigation timeout. Cancelling...")
                self._cancel_current_goal()
                return False
            result = result_future.result()
        except Exception as e:
            self.get_logger().error(f"Exception during navigation: {e}")
            self._cancel_current_goal()
            return False

        if result.status == 4:
            self.get_logger().info("Navigation succeeded!")
        else:
            self.get_logger().warning(f"Navigation failed with status: {result.status}")
        return True

    def wait_for_navigation_completion(self):
        """Wait for asynchronous navigation to complete"""
        if self._navigation_thread is not None:
            self._navigation_thread.join()

    def change_goal(self, x, y, yaw, timeout_sec=30.0):
        """Change the goal during navigation (from change_goal.py)"""
        self.get_logger().info("Changing goal...")
        self._cancel_current_goal()
        self.navigate_to_pose_async(x, y, yaw, timeout_sec)

    def _cancel_current_goal(self):
        """Cancel the current goal (from change_goal.py)"""
        if self._current_goal_handle is not None:
            self.get_logger().info("Cancelling current goal...")
            cancel_future = self._current_goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)

            cancel_response = cancel_future.result()
            if len(cancel_response.goals_canceling) > 0:
                self.get_logger().info(
                    "Goal successfully cancelled. Waiting a bit before sending new goal..."
                )
                time.sleep(1.0)  # Time for Nav2 to complete internal processing
            else:
                self.get_logger().warning(
                    "Goal cancellation failed or was already finished."
                )

            self._current_goal_handle = None

    def transform_odom_to_map(self, odom_x, odom_y, odom_yaw):
        """
        Transform coordinates from odom frame to map frame.

        Args:
            odom_x (float): X coordinate in odom frame (meters)
            odom_y (float): Y coordinate in odom frame (meters)
            odom_yaw (float): Yaw angle in odom frame (radians)

        Returns:
            tuple: (map_x, map_y, map_yaw) coordinates in map frame

        Raises:
            Exception: If transform lookup fails
        """
        try:
            # Wait for transform to be available
            self.tf_buffer.can_transform(
                "map",
                "odom",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )

            # Get the transform from odom to map
            transform = self.tf_buffer.lookup_transform(
                "map", "odom", rclpy.time.Time()
            )

            # Transform the position
            point_odom = PointStamped()
            point_odom.header.frame_id = "odom"
            point_odom.header.stamp = self.get_clock().now().to_msg()
            point_odom.point.x = odom_x
            point_odom.point.y = odom_y
            point_odom.point.z = 0.0

            point_map = tf2_geometry_msgs.do_transform_point(point_odom, transform)

            # Transform the orientation
            # Extract rotation from transform
            trans_quat = transform.transform.rotation
            trans_euler = quat2euler(
                [trans_quat.w, trans_quat.x, trans_quat.y, trans_quat.z]
            )
            trans_yaw = trans_euler[2]  # Z-axis rotation (yaw)

            # Add the yaw angles
            map_yaw = odom_yaw + trans_yaw

            # Normalize the angle to [-pi, pi]
            while map_yaw > math.pi:
                map_yaw -= 2 * math.pi
            while map_yaw < -math.pi:
                map_yaw += 2 * math.pi

            return point_map.point.x, point_map.point.y, map_yaw

        except TransformException as ex:
            self.get_logger().error(f"Could not transform from odom to map: {ex}")
            raise Exception(f"Transform failed: {ex}")
        except Exception as ex:
            self.get_logger().error(f"Unexpected error during transformation: {ex}")
            raise Exception(f"Transform failed: {ex}")
