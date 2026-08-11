from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class Nav2CanvasVisualizationTest(unittest.TestCase):
    def test_agent_core_uses_native_navigation_messages(self) -> None:
        bridge = (REPO_ROOT / "agent-core" / "src" / "ros2_bridge.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("fmt == 'sensor/odometry'", bridge)
        self.assertIn("from nav_msgs.msg import Odometry", bridge)
        self.assertIn("fmt == 'sensor/path'", bridge)
        self.assertIn("from nav_msgs.msg import Path", bridge)
        self.assertIn("_odometry_payload(msg)", bridge)
        self.assertIn("_path_payload(msg)", bridge)
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
        self.assertIn("OdometryRenderer", dashboard)
        self.assertIn("PathRenderer", dashboard)
        self.assertIn("OdometryRenderer", detail)
        self.assertIn("PathRenderer", detail)

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
        self.assertNotIn(
            "this._robotMesh.rotation.set(0, -robotYaw, 0)", mapping
        )


if __name__ == "__main__":
    unittest.main()
