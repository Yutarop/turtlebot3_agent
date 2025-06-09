#!/usr/bin/env python3
"""
TurtleBot3 agent node for ROS2.
This module defines the TB3Agent class which manages the robot's movement, navigation,
and sensor data processing.
"""

import math
import threading
import time

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from transforms3d.euler import quat2euler

from turtlebot3_agent.utils import normalize_angle

TWIST_ANGULAR = 0.30
TWIST_VELOCITY = 0.21
ROTATION_ERROR_THRESHOLD = 0.017  # radians
DISTANCE_ERROR_THRESHOLD = 0.02  # meters
TOPIC_CMD_VEL = "/cmd_vel"
TOPIC_ODOMETRY = "/odometry/filtered"
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
        """Process incoming img data"""
        self.camera_image = msg

    def get_latest_image(self) -> Image:
        if self.latest_image is None:
            raise RuntimeError("No image received yet.")
        return self.latest_image

    def _stop_robot(self):
        """Send stop commands to bring the robot to a complete halt."""
        stop_cmd = Twist()
        for _ in range(5):
            self.pub.publish(stop_cmd)
            time.sleep(PUBLISH_RATE)

    def rotate_angle(self, angle_rad):
        """
        Rotate the robot by a specified angle.

        Args:
            angle_rad: Angle to rotate in radians (positive=counterclockwise)

        Returns:
            bool: True if rotation completed successfully
        """
        if not self.wait_for_pose():
            self.get_logger().warning("Failed to get initial pose for rotation")
            return False

        start_yaw = self.yaw
        target_yaw = normalize_angle(start_yaw + angle_rad)
        self.get_logger().info(
            f"start_yaw: {round(start_yaw, 3)}, angle_rad: {angle_rad}, "
            f"target_yaw_raw: {round(start_yaw + angle_rad, 3)}, "
            f"target_yaw: {round(target_yaw, 3)}"
        )
        angular_velocity = TWIST_ANGULAR if angle_rad > 0 else -TWIST_ANGULAR

        while rclpy.ok():
            current = self.yaw
            diff = normalize_angle(target_yaw - current)

            if abs(diff) < ROTATION_ERROR_THRESHOLD:
                break

            twist = Twist()
            twist.angular.z = angular_velocity
            self.pub.publish(twist)
            time.sleep(PUBLISH_RATE)

        self._stop_robot()
        return True

    def move_linear(self, distance_m):
        """
        Move the robot by a specified distance.

        Args:
            distance_m: Distance to move in meters (positive=forward)

        Returns:
            bool: True if movement completed successfully
        """
        if not self.wait_for_pose():
            self.get_logger().warning("Failed to get initial pose for movement")
            return False

        start_x, start_y = self.x, self.y
        self.get_logger().info(
            f"Moving: distance={distance_m}m from ({start_x}, {start_y})"
        )

        linear_velocity = TWIST_VELOCITY if distance_m > 0 else -TWIST_VELOCITY

        while rclpy.ok():
            dx = self.x - start_x
            dy = self.y - start_y
            moved_distance = math.sqrt(dx**2 + dy**2)
            remaining = abs(distance_m) - moved_distance

            if remaining < DISTANCE_ERROR_THRESHOLD:
                break

            twist = Twist()
            twist.linear.x = linear_velocity
            self.pub.publish(twist)
            time.sleep(PUBLISH_RATE)

        self._stop_robot()
        return True

    def move_non_linear(
        self,
        duration_sec,
        linear_velocity=TWIST_VELOCITY,
        angular_velocity=TWIST_ANGULAR,
    ):
        """
        Move the robot with specified linear and angular velocities for a certain duration.

        Args:
            linear_velocity (float): Linear velocity in m/s. Positive for forward.
            angular_velocity (float): Angular velocity in rad/s. Positive for left turn.
            duration_sec (float): Time in seconds to apply the twist.

        Returns:
            bool: True if movement executed.
        """
        if not self.wait_for_pose():
            self.get_logger().warning("Failed to get initial pose for curve movement")
            return False

        self.get_logger().info(
            f"Moving in a curve: linear_velocity={linear_velocity} m/s, "
            f"angular_velocity={angular_velocity} rad/s, duration={duration_sec} sec"
        )

        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time < duration_sec):
            twist = Twist()
            twist.linear.x = linear_velocity
            twist.angular.z = angular_velocity
            self.pub.publish(twist)
            time.sleep(PUBLISH_RATE)

        self._stop_robot()
        return True
