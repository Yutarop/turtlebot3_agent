import os
from glob import glob

from setuptools import find_packages, setup

package_name = "tb3_agent"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.*")),
        (
            os.path.join("share", package_name, "config"),
            glob("config/ekf_turtlebot3.yaml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ubuntu",
    maintainer_email="yutarop.storm.7@gmail.com",
    description="A ROS2 package that integrates a TurtleBot3 with LangChain and language models to enable natural language control of the robot.",
    license="MIT License",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "tb3_node_entrypoint = tb3_agent.tb3_node_entrypoint:main",
            "main = tb3_agent.main:main",
        ],
    },
)
