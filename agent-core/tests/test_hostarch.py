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
def _fresh(monkeypatch, tmp_path):
    monkeypatch.delenv('ACC_ARCH', raising=False)
    # 默认没有 NVIDIA 驱动可读；JetPack 7 的用例自己写一个版本号进去
    monkeypatch.setattr(hostarch, '_NVIDIA_DRIVER_VERSION', str(tmp_path / 'absent'))
    hostarch.acc_arch.cache_clear()
    yield
    hostarch.acc_arch.cache_clear()


@pytest.mark.parametrize('kernel, expected', [
    ('Linux version 5.10.216-tegra (buildbrain@mobile-u64-6336)', 'jetson-jp5'),
    ('Linux version 5.15.148-tegra (buildbrain@mobile-u64-6422)', 'jetson-jp6'),
    # Thor，2026-10-30 实测 /proc/version；JetPack 7.2.1 / L4T R39.2.1。读不到驱动版本 → 按 7.1
    ('Linux version 6.8.12-1021-tegra (buildbrain@03d5624e-58e4-4d2e-841c-81)', 'jetson-jp7.1'),
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


@pytest.mark.parametrize('driver, expected', [
    ('580.00', 'jetson-jp7.1'),     # thor_pack 实测：星源智 N4，R38.4，CUDA 13.0
    ('580.65.06', 'jetson-jp7.1'),
    ('595.12', 'jetson-jp7.2'),     # R39 / CUDA 13.2 的驱动比 580 新
    ('garbage', 'jetson-jp7.1'),    # 认不出就按 7.1：只会少看到镜像，不会装错
])
def test_jetpack7_minor_comes_from_the_gpu_driver(monkeypatch, tmp_path, driver, expected):
    """7.1 和 7.2 内核都是 6.8，分它们的是驱动：给 7.2 构建的镜像在 7.1 的驱动上跑不起来。"""
    version = tmp_path / 'version'
    version.write_text(driver + '\n')
    monkeypatch.setattr(hostarch, '_NVIDIA_DRIVER_VERSION', str(version))
    monkeypatch.setattr(hostarch, '_kernel_string', lambda: 'Linux version 6.8.12-rt-tegra (root@VM)')
    assert hostarch.acc_arch() == expected


def test_jetpack_minor_does_not_touch_older_lines(monkeypatch, tmp_path):
    version = tmp_path / 'version'
    version.write_text('595.12\n')
    monkeypatch.setattr(hostarch, '_NVIDIA_DRIVER_VERSION', str(version))
    monkeypatch.setattr(hostarch, '_kernel_string', lambda: 'Linux version 5.15.148-tegra')
    assert hostarch.acc_arch() == 'jetson-jp6'
