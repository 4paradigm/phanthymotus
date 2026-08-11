from glob import glob
import os

from setuptools import find_packages, setup


package_name = "g1_nav2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Phanthy Motus",
    maintainer_email="devnull@example.com",
    description="G1 Nav2 adapters with persistent maps and semantic tags",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "canvas_pointcloud_bridge = g1_nav2.canvas_pointcloud_node:main",
            "loco_odom_bridge = g1_nav2.loco_odom_node:main",
            "navigation_command_bridge = g1_nav2.navigation_command_node:main",
            "runtime_supervisor = g1_nav2.runtime_supervisor:main",
        ],
    },
)
