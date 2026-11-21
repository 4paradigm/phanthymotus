"""Perception sets its BLAS thread limit before importing native modules."""
import os
from pathlib import Path
import subprocess
import sys


def test_perception_caps_openblas_before_ros_import():
    source = Path(__file__).resolve().parents[1]
    probe = """
from importlib.abc import MetaPathFinder
import os
import runpy
import sys

class BeforeROS(MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'rclpy':
            assert 'numpy' not in sys.modules
            assert os.environ['OPENBLAS_NUM_THREADS'] == '1'
            assert os.environ['OMP_NUM_THREADS'] == '7'
            raise SystemExit(0)

sys.meta_path.insert(0, BeforeROS())
runpy.run_path('main.py', run_name='__main__')
raise AssertionError('ROS import was not observed')
"""
    env = dict(os.environ, OPENBLAS_NUM_THREADS='8', OMP_NUM_THREADS='7')
    result = subprocess.run([sys.executable, '-c', probe], cwd=source, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
