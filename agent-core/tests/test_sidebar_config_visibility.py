"""Run config-field visibility regressions without a browser or GPU."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_instance_config_visibility():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for frontend regressions")
    subprocess.run([node, str(Path(__file__).with_name("sidebar_config_visibility.cjs"))],
                   check=True, capture_output=True, text=True)
