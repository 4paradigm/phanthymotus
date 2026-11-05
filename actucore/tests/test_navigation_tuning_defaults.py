"""Field-tuned navigation defaults (2026-11-05 G1 session).

These values were found on the robot while chasing "maps double up / goals covered
by obstacles / robot stalls". They are defaults, not per-card overrides:

* mapper cost per frame had to drop below the 100 ms LiDAR budget, otherwise the
  single-threaded loop backlogs up to ~10 s and the adapter rejects stale frames;
* the costmap footprint describes an ellipse (the robot is not a rectangle), so a
  rotation sweep no longer invents a 0.35 m corner radius out of a 0.15 x 0.32 body;
* the static layer clears the robot footprint, so a pose that lands on phantom
  points from an older map can still plan;
* a saved map can lag a session by up to a second under load, hence the 1.0 s
  input age limit instead of 0.5 s.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "plugins" / "controlled_semantic_spatial" / "runtime"


def load_lio() -> dict:
    return yaml.safe_load((RUNTIME / "navigation_lio.yaml").read_text())


def load_nav2() -> dict:
    return yaml.safe_load((RUNTIME / "nav2" / "config" / "nav2_params.yaml").read_text())


def footprint_points(raw: str):
    return [(float(x), float(y)) for x, y in re.findall(r"\[(-?[0-9.]+),\s*(-?[0-9.]+)\]", raw)]


def test_mapper_cost_defaults_leave_headroom_against_the_lidar_budget():
    params = load_lio()["/**"]["ros__parameters"]
    assert params["lio"]["max_iterations"] == 3
    assert params["preprocess"]["filter_size_surf"] == 0.2
    # The VIO branch is unused while img_en is 0 and stays untouched.
    assert params["vio"]["max_iterations"] == 5


def test_costmap_footprint_is_an_ellipse_not_a_rectangle():
    config = load_nav2()
    for name in ("local_costmap", "global_costmap"):
        params = config[name][name]["ros__parameters"]
        points = footprint_points(params["footprint"])
        assert len(points) >= 8, "an ellipse needs more than four vertices"
        radii = [math.hypot(x, y) for x, y in points]
        # Rotation sweep is bounded by the body half width, not by a rectangle corner.
        assert max(radii) < 0.33
        assert max(radii) > 0.31


def test_costmap_padding_and_static_footprint_clearing_are_pinned():
    config = load_nav2()
    for name in ("local_costmap", "global_costmap"):
        params = config[name][name]["ros__parameters"]
        assert params["footprint_padding"] == 0.01
    static_layer = config["global_costmap"]["global_costmap"]["ros__parameters"]["static_layer"]
    assert static_layer["footprint_clearing_enabled"] is True


def test_inflation_still_leaves_a_gradient_above_the_inscribed_radius():
    config = load_nav2()
    for name in ("local_costmap", "global_costmap"):
        params = config[name][name]["ros__parameters"]
        points = footprint_points(params["footprint"])
        # Inscribed radius: shortest distance from the origin to a footprint edge.
        best = min(
            _distance_to_segment((0.0, 0.0), points[i], points[(i + 1) % len(points)])
            for i in range(len(points))
        )
        effective = best + params["footprint_padding"]
        assert params["inflation_layer"]["inflation_radius"] > effective + 0.05


def test_adapter_accepts_a_one_second_old_frame_by_default():
    source = (RUNTIME / "fast_livo2" / "fast_livo2" / "adapter_node.py").read_text()
    assert 'self.declare_parameter("source_max_age_sec", 1.0)' in source


def _distance_to_segment(point, start, end):
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
