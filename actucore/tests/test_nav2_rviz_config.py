from __future__ import annotations

import unittest
from pathlib import Path


ACTUCORE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = (
    ACTUCORE_ROOT / "plugins" / "navigation" / "runtime" / "nav2"
)


class Nav2RvizConfigTest(unittest.TestCase):
    def test_rviz_config_is_packaged_with_required_read_only_topics(self) -> None:
        rviz_path = PACKAGE_ROOT / "rviz" / "nav2.rviz"
        self.assertTrue(rviz_path.is_file())
        rviz = rviz_path.read_text(encoding="utf-8")

        self.assertIn("Fixed Frame: map", rviz)
        required_topics = (
            "/plan",
            "/global_costmap/costmap",
            "/local_costmap/costmap",
            "/local_costmap/published_footprint",
            "/ubuntu/navigation/cloud_registered",
            "/ubuntu/navigation/odom",
        )
        for topic in required_topics:
            self.assertIn(f"Value: {topic}", rviz)

        for state_changing_tool in ("SetGoal", "SetInitialPose", "GoalTool"):
            self.assertNotIn(state_changing_tool, rviz)

        setup = (PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('glob("rviz/*.rviz")', setup)


if __name__ == "__main__":
    unittest.main()
