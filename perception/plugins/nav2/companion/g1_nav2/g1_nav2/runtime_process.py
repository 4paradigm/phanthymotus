"""ROS-independent validation for supervised Nav2 child launches."""

from __future__ import annotations

import os
import math
from pathlib import Path
import re


VALID_MODES = {"mapping", "localization"}


def plain_map_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("map_name must be a string")
    name = value.strip()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name) is None:
        raise ValueError("map_name must be a plain name")
    return name


def build_launch_command(
    *, mode: str, map_name: str = "", maps_root: str = "/maps", environ=None
) -> list[str]:
    """Build the child launch command without accepting arbitrary ROS arguments."""

    env = os.environ if environ is None else environ
    if mode not in VALID_MODES:
        raise ValueError("mode must be mapping or localization")
    command = [
        "ros2",
        "launch",
        "g1_nav2",
        "g1_nav2.launch.py",
        f"mode:={mode}",
    ]
    for key, argument in (
        ("NAV2_LIDAR_X", "lidar_x"),
        ("NAV2_LIDAR_Y", "lidar_y"),
        ("NAV2_LIDAR_Z", "lidar_z"),
        ("NAV2_LIDAR_ROLL", "lidar_roll"),
        ("NAV2_LIDAR_PITCH", "lidar_pitch"),
        ("NAV2_LIDAR_YAW", "lidar_yaw"),
    ):
        value = str(env.get(key, "")).strip()
        if not value:
            raise ValueError(f"{key} is required")
        if not math.isfinite(float(value)):
            raise ValueError(f"{key} must be finite")
        command.append(f"{argument}:={value}")

    publish_static_tf = str(
        env.get("NAV2_PUBLISH_LIDAR_STATIC_TF", "true")
    ).strip().lower()
    if publish_static_tf not in {"true", "false"}:
        raise ValueError("NAV2_PUBLISH_LIDAR_STATIC_TF must be true or false")
    command.append(f"publish_lidar_static_tf:={publish_static_tf}")

    if mode == "localization":
        name = plain_map_name(map_name)
        map_yaml = Path(maps_root) / name / "map.yaml"
        if not map_yaml.is_file():
            raise ValueError(f"saved map is missing: {map_yaml}")
        command.extend((f"map_name:={name}", f"map:={map_yaml}"))
    return command
