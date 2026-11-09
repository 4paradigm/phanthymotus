"""Run the odometry behavior contract through the browser ES modules.

`state/odom` no longer has a dedicated renderer: it goes through the same generic
JSON renderers as `data/json`. The test file checks field display, invalid frames,
preview-buffer transport and that the removed odometry renderer modules are gone.
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
