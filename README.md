## Project Overview
`TurtleBot3 Agent` enables intuitive control of a TurtleBot3 robot using natural language. It interprets user instructions and uses tools to perform tasks such as moving, accessing sensor data, and navigating.

## TurtleBot3 Agent Demo in the Real World
##### Prompt used
> I want you to move in square. Each length should be 0.5 meters long.

https://github.com/user-attachments/assets/033b9089-e20c-4a29-9203-5779a9b7d532

#### Behind the Scenes
You can see which tools are provided and being used in the demo at the URL below:  
https://smith.langchain.com/public/246d31ee-a674-4f65-ba4f-fe2ae1e52d8b/r/ae995e3b-8363-4c8c-935c-fd65f6b43557

## Getting Started
#### Requirements
- ROS 2 Humble Hawksbill (This project has only been tested with ROS 2 Humble. Compatibility with other ROS 2 distributions is not guaranteed.)
- Python 3.10+
- Other dependencies as listed in `requirements.txt`
### 1. Clone and build in a ROS2 workspace 
```bash
$ cd ~/{ROS_WORKSPACE}/src
$ git clone https://github.com/Yutarop/turtlebot3_agent.git
$ python3 -m pip install -r turtlebot3_agent/requirements.txt
$ cd ~/{ROS_WORKSPACE} && colcon build
```
### 2. Set LLM Models and TurtleBot3 model
To make your API keys available in your development environment, add them to your shell configuration file (e.g., `~/.bashrc`), then reload the file using `source`.

```bash
# Your TurtleBot3 model
export TURTLEBOT3_MODEL=burger

# API keys for LLM providers (set only the one you plan to use)
export OPENAI_API_KEY=your_openai_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
export GOOGLE_API_KEY=your_google_api_key
export COHERE_API_KEY=your_cohere_api_key
export MISTRAL_API_KEY=your_mistral_api_key
```

To specify which Large Language Model (LLM) your agent should use, you need to configure the model name.

- **Python ([tb3_node.py](https://github.com/Yutarop/turtlebot3_agent/blob/main/turtlebot3_agent/tb3_node.py)):**

 ```python
 self.declare_parameter("agent_model", "gpt-4o-mini")
 ```
#### (Optional) Enable Tracing with LangSmith
To trace and debug agent behavior using LangSmith, set the following environment variables:

Basic Tracing Configuration:
```bash
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
export LANGSMITH_TRACING=false
```
Full Configuration with API Key and Project Name:
```bash
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=your_api_key_here
export LANGSMITH_PROJECT=your_project_name_here
```

#### Apply the changes
Once you have configured the variables, proceed to build and apply the changes to finalize the setup:
```bash
$ cd ~/{ROS_WORKSPACE} && colcon build
$ source ~/.bashrc
```

### 3. Run
```bash
$ ros2 run turtlebot3_agent main
```

## Provided Tools for the AI Agent
`TurtleBot3 Agent` utilizes the tools implemented in the `tools/` directory as callable functions that it can invoke during the reasoning process to accomplish tasks. Feel free to customize your tools however you like!
#### Tools
```
tools/
├── all_tools.py        # Consolidates and provides access to all available tools for the system
├── math_tools.py       # Performs arithmetic and geometric calculations
├── status_tools.py     # Retrieves the current status of the TurtleBot3, such as position and orientation
├── motion_tools.py     # Manages TurtleBot3 movement, such as forward motion and rotation
└── sensor_tools.py     # Handles TurtleBot3 camera and LiDAR operations, such as capturing images and scanning the environment
```
