from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]


class Nav2CanvasVisualizationTest(unittest.TestCase):
    def test_costmap_payload_preserves_grid_geometry_and_costs(self) -> None:
        bridge_path = REPO_ROOT / "agent-core" / "src" / "ros2_bridge.py"
        spec = importlib.util.spec_from_file_location("nav2_core_ros2_bridge", bridge_path)
        bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bridge)

        zero_orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
        message = SimpleNamespace(
            header=SimpleNamespace(
                frame_id="map",
                stamp=SimpleNamespace(sec=12, nanosec=34),
            ),
            info=SimpleNamespace(
                resolution=0.05,
                width=2,
                height=2,
                origin=SimpleNamespace(
                    position=SimpleNamespace(x=-1.0, y=-2.0),
                    orientation=zero_orientation,
                ),
            ),
            data=[0, 25, 99, -1],
        )

        payload = bridge._occupancy_grid_payload(message)

        self.assertEqual(payload["schema"], "phanthy.navigation.costmap.v1")
        self.assertEqual(payload["frame_id"], "map")
        self.assertEqual(payload["stamp_ns"], 12_000_000_034)
        self.assertEqual(payload["origin"], {"x": -1.0, "y": -2.0, "yaw": 0.0})
        self.assertEqual(payload["data"], [0, 25, 99, -1])

    def test_agent_core_uses_native_navigation_messages(self) -> None:
        bridge = (REPO_ROOT / "agent-core" / "src" / "ros2_bridge.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("fmt == 'sensor/odometry'", bridge)
        self.assertIn("from nav_msgs.msg import Odometry", bridge)
        self.assertIn("fmt == 'sensor/path'", bridge)
        self.assertIn("from nav_msgs.msg import Path", bridge)
        self.assertIn("fmt == 'sensor/costmap'", bridge)
        self.assertIn("from nav_msgs.msg import OccupancyGrid", bridge)
        self.assertIn("_odometry_payload(msg)", bridge)
        self.assertIn("_path_payload(msg)", bridge)
        self.assertIn("_occupancy_grid_payload(msg)", bridge)
        dockerfile = (REPO_ROOT / "agent-core" / "Dockerfile").read_text(
            encoding="utf-8"
        )
        self.assertIn("ros-humble-nav-msgs", dockerfile)

    def test_dashboard_registers_odometry_and_path_renderers(self) -> None:
        renderer = (
            REPO_ROOT / "agent-core" / "web" / "js" / "renderers" / "navigation.js"
        ).read_text(encoding="utf-8")
        dashboard = (
            REPO_ROOT / "agent-core" / "web" / "js" / "monitor-dashboard.js"
        ).read_text(encoding="utf-8")
        detail = (
            REPO_ROOT / "agent-core" / "web" / "js" / "detail-panel.js"
        ).read_text(encoding="utf-8")

        self.assertIn("hint === 'sensor/odometry'", renderer)
        self.assertIn("hint === 'sensor/path'", renderer)
        self.assertIn("hint === 'sensor/costmap'", renderer)
        self.assertIn("/ws/bus/plan", renderer)
        self.assertIn("/ws/bus/ubuntu/navigation/odom", renderer)
        self.assertIn("Rejected (&gt;=99)", renderer)
        self.assertIn("value >= 99", renderer)
        self.assertNotIn("value > 0", renderer)
        self.assertNotIn("Inflated", renderer)
        self.assertIn("OdometryRenderer", dashboard)
        self.assertIn("PathRenderer", dashboard)
        self.assertIn("CostmapRenderer", dashboard)
        self.assertIn("OdometryRenderer", detail)
        self.assertIn("PathRenderer", detail)
        self.assertIn("CostmapRenderer", detail)

    def test_mapping_renderer_overlays_the_map_frame_plan(self) -> None:
        mapping = (
            REPO_ROOT / "agent-core" / "web" / "js" / "renderers" / "mapping.js"
        ).read_text(encoding="utf-8")

        self.assertIn("/ws/bus/plan", mapping)
        self.assertIn("new THREE.Line", mapping)
        self.assertIn("data.frame_id !== 'map'", mapping)
        self.assertIn("this._goalMesh.position.set", mapping)
        self.assertIn("PATH  ${poses.length} poses", mapping)
        self.assertIn("planWs?.close()", mapping)
        self.assertIn("this._setViewMode", mapping)
        self.assertIn("this._viewMode === '3d' ? '2d' : '3d'", mapping)
        self.assertIn("this._controls.enableRotate = false", mapping)
        self.assertIn("this._camera.up.set(0, 0, -1)", mapping)
        self.assertIn("this._fitTopDownView()", mapping)
        self.assertIn(
            "this._robotMesh.rotation.set(0, robotYaw, 0)", mapping
        )
        self.assertIn("MVFILT2", mapping)
        self.assertIn("filterMin", mapping)
        self.assertIn("Below the configured navigation band: blue", mapping)
        self.assertIn("Above the configured navigation band: magenta", mapping)
        self.assertIn("this._heightLegend.innerHTML", mapping)
        self.assertNotIn(
            "this._robotMesh.rotation.set(0, -robotYaw, 0)", mapping
        )


if __name__ == "__main__":
    unittest.main()
