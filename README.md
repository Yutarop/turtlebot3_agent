![logo4](https://github.com/user-attachments/assets/5075b307-f1cb-44af-8c01-1919d2b9397d)
![ROS2-humble Industrial CI](https://github.com/Yutarop/turtlebot3_agent/actions/workflows/ros2_ci.yml/badge.svg)
## Project Overview
`TurtleBot3 Agent` enables intuitive control of a TurtleBot3 robot using natural language. It interprets user instructions and uses tools to perform tasks such as moving, accessing sensor data, and navigating.

## TurtleBot3 Agent Demo
##### Prompt used
> Please move to (2.0, 2.0). Then, check for any obstacles around you there. If you find an obstacle, show me an image and come back to your staring point.



https://github.com/user-attachments/assets/5eb21ea0-ab1f-4d9c-b051-ef4c97e58262



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
$ vcs import . < turtlebot3_agent/tb3_agent.repos
$ sudo apt install -y ros-humble-cv-bridge ros-humble-robot-localization
$ rosdep install -r --from-paths . --ignore-src -y
$ cd ~/{ROS_WORKSPACE} && colcon build
```
### 2. Set LLM Models and TurtleBot3 model
To make your API keys available in your development environment, add them to your shell configuration file (e.g., `~/.bashrc`, `~/.zshrc`), then reload the file using `source`.

```bash
# TurtleBot3 model with camera
export TURTLEBOT3_MODEL=burger_cam

# API keys for LLM providers (set only the one you plan to use)
export OPENAI_API_KEY=your_openai_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
export GOOGLE_API_KEY=your_google_api_key
export COHERE_API_KEY=your_cohere_api_key
export MISTRAL_API_KEY=your_mistral_api_key
```

To specify which Large Language Model (LLM) your agent should use, you need to configure the model name.

- **Python ([turtleagent_node.py](https://github.com/Yutarop/turtlebot3_agent/blob/main/turtlebot3_agent/tb3_node.py)):**

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
$ ros2 launch turtlebot3_agent tb3_agent.launch.py
$ ros2 run turtlebot3_agent main
```
