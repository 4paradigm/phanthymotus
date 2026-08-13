from __future__ import annotations

import unittest
from pathlib import Path


CORE_ROOT = Path(__file__).resolve().parents[1]


class AgentCoreDockerfileMirrorTest(unittest.TestCase):
    def test_apt_and_ros_mirrors_are_consumed_before_update(self) -> None:
        dockerfile = (CORE_ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("ARG PYPI_MIRROR APT_MIRROR ROS_MIRROR", dockerfile)
        self.assertIn("mirrors\\.tencentyun\\.com", dockerfile)
        self.assertIn("ports\\.ubuntu\\.com", dockerfile)
        self.assertIn("packages\\.ros\\.org/ros2/ubuntu", dockerfile)
        self.assertIn("/^[[:space:]]*deb-src[[:space:]]/d", dockerfile)
        self.assertIn("^Types:[[:space:]]*deb", dockerfile)
        self.assertIn(
            "APT source-package entries remain after binary-only rewrite",
            dockerfile,
        )
        self.assertLess(dockerfile.index('if [ -n "${APT_MIRROR}" ]'),
                        dockerfile.index("apt-get -o Acquire::AllowInsecureRepositories=true update"))


if __name__ == "__main__":
    unittest.main()
