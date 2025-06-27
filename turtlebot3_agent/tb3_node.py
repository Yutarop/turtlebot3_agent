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
import tf2_geometry_msgs
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped, TransformStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from tf2_ros import TransformException
from transforms3d.euler import euler2quat, quat2euler

from turtlebot3_agent.utils import normalize_angle

TWIST_ANGULAR = 0.30
TWIST_VELOCITY = 0.21
ROTATION_ERROR_THRESHOLD = 0.017  # radians
DISTANCE_ERROR_THRESHOLD = 0.02  # meters
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

        # TF2 Buffer and Listener for coordinate transformations - この2行を追加
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

    def navigate_to_pose(self, x, y, yaw, timeout_sec=30.0):
        """
        Navigate to a specific pose using Nav2 NavigateToPose action.

        Args:
            x (float): Goal X coordinate in meters (map frame)
            y (float): Goal Y coordinate in meters (map frame)
            yaw (float): Goal yaw angle in radians
            timeout_sec (float): Maximum time to wait for navigation completion

        Returns:
            bool: True if navigation completed successfully, False otherwise
        """
        # Wait for action server
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("NavigateToPose action server not available.")
            return False

        # Create goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Set position
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0

        # Convert yaw to quaternion
        quat = euler2quat(0, 0, yaw)  # (roll, pitch, yaw)
        goal_msg.pose.pose.orientation.w = float(quat[0])
        goal_msg.pose.pose.orientation.x = float(quat[1])
        goal_msg.pose.pose.orientation.y = float(quat[2])
        goal_msg.pose.pose.orientation.z = float(quat[3])

        self.get_logger().info(f"Sending navigation goal: x={x}, y={y}, yaw={yaw}")

        # Send goal
        send_goal_future = self._nav_client.send_goal_async(goal_msg)

        # Wait for goal acceptance
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=5.0)

        if not send_goal_future.done():
            self.get_logger().error("Failed to send goal within timeout")
            return False

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected by NavigateToPose server")
            return False

        self.get_logger().info("Goal accepted, waiting for result...")

        # Wait for result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_sec)

        if not result_future.done():
            self.get_logger().error("Navigation timed out")
            # Cancel the goal
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
            return False

        result = result_future.result()
        if result.status == 4:  # SUCCEEDED
            self.get_logger().info("Navigation completed successfully!")
            return True
        else:
            self.get_logger().error(f"Navigation failed with status: {result.status}")
            return False

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
