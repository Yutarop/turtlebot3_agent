import sys
import threading
import tkinter as tk

import rclpy
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage
from rclpy.node import Node

from turtlebot3_agent.interface.chat_gui import ChatUI
from turtlebot3_agent.interface.gui_interface import GUIAgentInterface

LAST = -1


def chat_invoke(
    interface: str,
    agent_executor: AgentExecutor,
    node: Node,
    spin_thread: threading.Thread,
):
    if interface == "cli":
        _run_cli(agent_executor, node, spin_thread)
    elif interface == "gui":
        _run_gui(agent_executor, node)
    else:
        raise ValueError(f"Unsupported interface: {interface}")


def _run_cli(agent_executor: AgentExecutor, node: Node, spin_thread: threading.Thread):
    try:
        while True:
            user_input = input("user: ")
            if user_input.lower() in {"quit", "exit"}:
                break
            result = agent_executor.invoke(
                {"messages": [HumanMessage(content=user_input)]}
            )
            print("turtlebot agent:", result["messages"][LAST].content)
    except KeyboardInterrupt:
        node.get_logger().info("User interrupted. Shutting down...")
    finally:
        _shutdown(node, spin_thread)


def _run_gui(agent_executor: AgentExecutor, node: Node):
    interface = GUIAgentInterface(agent_executor)

    def on_close():
        interface.shutdown()
        _shutdown(node)
        root.destroy()
        sys.exit(0)

    try:
        root = tk.Tk()
        ChatUI(root, interface)
        root.protocol("WM_DELETE_WINDOW", on_close)
        root.mainloop()
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Exiting.")
        on_close()


def _shutdown(node: Node, spin_thread: threading.Thread = None):
    node.destroy_node()
    rclpy.shutdown()
    if spin_thread and spin_thread.is_alive():
        spin_thread.join(timeout=1.0)
