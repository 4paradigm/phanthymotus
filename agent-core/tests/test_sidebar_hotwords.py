"""Run the tag-input DOM regression with Node 18+ (no npm packages required)."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_sidebar_hotwords_render_and_save():
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if not node:
        pytest.skip("DOM regression requires Node 18+; run sidebar_hotwords.cjs on a JS test host")
    version = subprocess.check_output([node, "--version"], text=True).strip()
    if int(version.lstrip("v").split(".")[0]) < 18:
        pytest.skip("DOM regression requires Node 18+; set NODE_BINARY to a compatible executable")
    subprocess.run([node, str(Path(__file__).with_name("sidebar_hotwords.cjs"))],
                   check=True, timeout=30)
