from dotenv import load_dotenv
from langgraph.graph import MessagesState

from turtlebot3_agent.prompts import prompt

load_dotenv()


def run_agent_reasoning(llm):
    def _inner(state: MessagesState) -> MessagesState:
        response = llm.invoke(
            [{"role": "system", "content": prompt}, *state["messages"]]
        )
        return {"messages": [response]}

    return _inner
