from turtlebot3_agent.interface.base_interface import BaseAgentInterface


class GUIAgentInterface(BaseAgentInterface):
    def __init__(self, agent_executor):
        self.agent_executor = agent_executor

    def send_user_input(self, user_input: str) -> str:
        try:
            result = self.agent_executor.invoke({"input": user_input})
            return result.get("output", "[No response]")
        except Exception as e:
            return f"[Error] {e}"

    def shutdown(self):
        pass  # Any specific shutdown procedures can go here
