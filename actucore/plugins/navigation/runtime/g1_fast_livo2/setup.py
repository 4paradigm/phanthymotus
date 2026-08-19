from glob import glob
import os

from setuptools import find_packages, setup


package_name = "g1_fast_livo2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Phanthy Motus",
    maintainer_email="devnull@example.com",
    description="FAST-LIVO2 runtime ownership and canonical G1 frame adapter",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "frame_adapter = g1_fast_livo2.adapter_node:main",
            "runtime_supervisor = g1_fast_livo2.runtime_supervisor:main",
        ]
    },
)
