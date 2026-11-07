"""Run the odometry behavior contract through the browser ES modules.

The dedicated renderer reads standard motus.odom/1 directly; ordinary JSON
keeps its original route. Tests exercise both real view mount paths, pose
validity, snapshot replacement, raw copying and log history.
"""
import os
import pathlib
import shutil
import subprocess

import pytest

WEB = pathlib.Path(__file__).resolve().parents[1] / 'web' / 'js'


def test_odometry_and_ordinary_json_view_behavior():
    runtime = os.environ.get('NODE') or shutil.which('node')
    if runtime is None:
        pytest.skip('Node.js is required for the browser ES module behavior tests')
    result = subprocess.run(
        [runtime, '--test', str(WEB / 'odom-renderer.test.mjs'), str(WEB / 'json-util.test.mjs')],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
