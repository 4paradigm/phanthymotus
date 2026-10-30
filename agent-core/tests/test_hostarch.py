"""hostarch.acc_arch：从宿主内核版本反推 JetPack 线。

取值决定 resource-center 给这台机器返回哪些镜像；认不出的内核按 `none` 算，
于是只看得到架构无关的镜像 —— 这是安全方向，但也意味着漏一条线就整条线的
Jetson 镜像在那台机器上都消失，而且没有任何报错。

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest agent-core/tests/test_hostarch.py -q
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
import hostarch  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv('ACC_ARCH', raising=False)
    hostarch.acc_arch.cache_clear()
    yield
    hostarch.acc_arch.cache_clear()


@pytest.mark.parametrize('kernel, expected', [
    ('Linux version 5.10.216-tegra (buildbrain@mobile-u64-6336)', 'jetson-jp5'),
    ('Linux version 5.15.148-tegra (buildbrain@mobile-u64-6422)', 'jetson-jp6'),
    # Thor，2026-10-30 实测 /proc/version；JetPack 7.2.1 / L4T R39.2.1
    ('Linux version 6.8.12-1021-tegra (buildbrain@03d5624e-58e4-4d2e-841c-81)', 'jetson-jp7'),
    ('Linux version 6.11.0-26-generic (buildd@lcy02-amd64-074)', 'none'),
    ('Linux version 9.9.1-tegra', 'none'),
])
def test_kernel_to_jetpack_line(monkeypatch, kernel, expected):
    monkeypatch.setattr(hostarch, '_kernel_string', lambda: kernel)
    assert hostarch.acc_arch() == expected


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv('ACC_ARCH', 'jetson-jp6')
    monkeypatch.setattr(hostarch, '_kernel_string', lambda: 'Linux version 6.8.12-1021-tegra')
    assert hostarch.acc_arch() == 'jetson-jp6'
