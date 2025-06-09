#!/usr/bin/env python3
import rclpy

from tb3_agent.tb3_node import TB3Agent


def main():
    rclpy.init()
    node = TB3Agent()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
