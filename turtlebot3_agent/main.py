#!/usr/bin/env python3
"""
Main entry point for the TurtleBot3 agent with LangChain integration.
This module initializes the ROS2 node and connects it with a language-based agent
that can interpret natural language commands to control the robot.
"""

import threading

import rclpy
from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor

from turtlebot3_agent.chat_entrypoint import chat_invoke
from turtlebot3_agent.llms import create_agent
from turtlebot3_agent.nodes import run_agent_reasoning
from turtlebot3_agent.tb3_node import TB3Agent
from turtlebot3_agent.tools.all_tools import make_all_tools

load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1


def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT


def spin_node_thread(node):
    """
    Run the ROS2 node in a separate thread to allow concurrent execution.

    Args:
        node: The ROS2 node to spin
    """
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()


def main():
    """
    Main function that initializes and runs the TurtleBot3 agent with LangChain integration.
    """

    rclpy.init()
    node = TB3Agent()

    spin_thread = threading.Thread(target=spin_node_thread, args=(node,), daemon=True)
    spin_thread.start()

    if not node.wait_for_pose(timeout=3.0):
        node.get_logger().error("Initial pose not received")
        node.destroy_node()
        rclpy.shutdown()
        return

    tools = make_all_tools(node)
    tool_node = ToolNode(tools)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0).bind_tools(
        tools
    )

    flow = StateGraph(MessagesState)

    flow.add_node(AGENT_REASON, run_agent_reasoning(llm))
    flow.set_entry_point(AGENT_REASON)
    flow.add_node(ACT, tool_node)

    flow.add_conditional_edges(AGENT_REASON, should_continue, {END: END, ACT: ACT})

    flow.add_edge(ACT, AGENT_REASON)

    app = flow.compile()
    app.get_graph().draw_mermaid_png(output_file_path="flow.png")

    chat_invoke(
        interface=node.interface,
        agent_executor=app,
        node=node,
        spin_thread=spin_thread,
    )


if __name__ == "__main__":
    main()
