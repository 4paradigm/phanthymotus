"""Shared card height/radius launch transforms for both Nav2 costmaps.

Pure configuration transformation, before node startup; no ROS services or I/O.
"""
from __future__ import annotations

import math


def apply_obstacle_height_band(config: dict, minimum: float, maximum: float) -> None:
    """Validate first, then overwrite all eight leaves, including absent defaults.

    RewrittenYaml in Humble only substitutes existing leaves. Layer-level height
    keys missing from an older YAML otherwise silently retain Nav2's 0..2 default.
    Do not invent a missing costmap or observation source: reject that input.
    """
    for value in (minimum, maximum):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not -3.0 <= value <= 3.0):
            raise ValueError("obstacle height limits must be finite numbers within [-3.0, 3.0]")
    if minimum >= maximum:
        raise ValueError("obstacle minimum height must be less than maximum")
    if not isinstance(config, dict):
        raise ValueError("Nav2 parameters must be a mapping")

    targets = []
    for name in ("local_costmap", "global_costmap"):
        try:
            params = config[name][name]["ros__parameters"]
            layer = params["obstacle_layer"]
            source = layer["registered_cloud"]
            plugins = params["plugins"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{name} must define obstacle_layer.registered_cloud") from exc
        sources = layer.get("observation_sources") if isinstance(layer, dict) else None
        if (not isinstance(layer, dict) or not isinstance(source, dict)
                or not isinstance(plugins, list) or "obstacle_layer" not in plugins
                or sources != "registered_cloud"
                or layer.get("plugin") != "nav2_costmap_2d::ObstacleLayer"):
            raise ValueError(f"{name} must use the registered_cloud ObstacleLayer/source")
        if params.get("global_frame") != "map":
            raise ValueError(f"{name} global_frame must be map for the shared map-z height band")
        targets.extend((layer, source))
    for target in targets:
        target["min_obstacle_height"] = float(minimum)
        target["max_obstacle_height"] = float(maximum)


def apply_inflation_radius(config: dict, radius: float) -> None:
    """Set only the two inflation layers, including missing legacy radius leaves.

    Validate both targets before mutation. The (0, 2] limit belongs to this card,
    not Nav2; no footprint or cost scaling is changed or inferred.
    """
    if (isinstance(radius, bool) or not isinstance(radius, (int, float))
            or not 0 < radius <= 2.0 or not math.isfinite(radius)):
        raise ValueError("inflation_radius_m must be a finite number within (0, 2.0]")
    targets = []
    for name in ("local_costmap", "global_costmap"):
        try:
            params = config[name][name]["ros__parameters"]
            layer = params["inflation_layer"]
            plugins = params["plugins"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{name} must define inflation_layer") from exc
        if (not isinstance(layer, dict) or not isinstance(plugins, list)
                or "inflation_layer" not in plugins
                or layer.get("plugin") != "nav2_costmap_2d::InflationLayer"):
            raise ValueError(f"{name} must use the standard InflationLayer")
        targets.append(layer)
    for layer in targets:
        layer["inflation_radius"] = float(radius)
