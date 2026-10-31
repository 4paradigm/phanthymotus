"""hostarch.py — 在容器内探测宿主的加速器 / CPU 架构。

用途：向 resource-center 请求镜像目录时带上本机架构，让它只返回能在本机运行的镜像
（perception / actucore 是 Jetson 专用镜像，按 JetPack 大版本分别构建，不能混装；
ModelHub 在 JetPack 7 上还要分小版本，见 _jp7_line）。

为什么读 /proc/version：已在 JetPack 6 真机（L4T 36.4）验证，agent-core 容器内
/proc/device-tree 和 /etc/nv_tegra_release **都不存在**，但 /proc/version 仍显示
宿主内核（"Linux version 5.15.148-tegra …"）—— 内核不受 namespace 隔离。NVIDIA BSP
内核永远带 `-tegra` 后缀，所以从内核版本即可反推 L4T 大版本、进而推出 JetPack 线。
挂载宿主路径能拿到更精确的小版本，但那要改 install.sh 生成的 compose 并让所有机器
重装 core，不值得。

刻意不落库：这是宿主属性而非用户配置，持久化后在克隆 SD 卡、原地升级 JetPack 之后
就是过期值。模块级 memo 足够 —— 升级必然重启容器，进程寿命 == 部署寿命。

取值必须与 resource-center/lib/arch.ts 保持一致。
"""

import functools
import os
import platform
import re

# 本机无 NVIDIA 加速器。也兼任「无法识别的 tegra 内核」——失败方向安全：
# 只会看到 agnostic 镜像，而不是装上跑不了的镜像。
ACC_NONE = 'none'

# tegra 内核 major.minor -> JetPack 线
_TEGRA_KERNEL_TO_ACC = {
    '4.9': 'jetson-jp4',    # L4T 32.x（未实测，暂无 jp4 镜像）
    '5.10': 'jetson-jp5',   # L4T 35.x
    '5.15': 'jetson-jp6',   # L4T 36.x
    '6.8': 'jetson-jp7',    # L4T 38.x / 39.x，小版本见 _jp7_line
}

# JetPack 7 的两条线内核都是 6.8（Thor 实测：R39.2.1 是 6.8.12-1021-tegra，星源智 N4
# 的 R38.4 是 6.8.12-rt-tegra），从内核分不出。分得出、也真正要紧的是 GPU 驱动：
# 7.1（R38）的驱动只支持到 CUDA 13.0，7.2（R39）到 13.2，而 Jetson 的 CUDA 没有
# 前向兼容 —— 给 7.2 构建的镜像装到 7.1 上，vLLM 启动就报
# CUDA_ERROR_SYSTEM_DRIVER_MISMATCH。2026-10-31 thor_pack 就这么装上了 jp7.2 的
# ModelHub，因为两条线都只报 jetson-jp7。
#
# 驱动版本读 /sys/module/nvidia/version：和 /proc/version 一样不受容器隔离，agent-core
# 容器里读得到（thor_pack 实测 580.00 = CUDA 13.0）。/proc/driver/nvidia/version 在
# 厂商自编的 BSP 上是 "TempVersion"，靠不住。
#
# 读不到或不认识时按 7.1 算：新驱动能跑旧 CUDA 的镜像，所以把 7.2 的机器当成 7.1
# 只会少看到几个镜像，反过来才会装上跑不了的。
_JP71_MAX_DRIVER = 580      # R38.x 的驱动大版本；更新的驱动是 R39 / JetPack 7.2
_NVIDIA_DRIVER_VERSION = '/sys/module/nvidia/version'

_TEGRA_RE = re.compile(r'(\d+)\.(\d+)\.\d+\S*-tegra')


def _kernel_string() -> str:
    try:
        with open('/proc/version') as f:
            return f.read()
    except OSError:
        return platform.release()  # 容器内同样是宿主内核


def _nvidia_driver_major() -> int | None:
    try:
        with open(_NVIDIA_DRIVER_VERSION) as f:
            return int(f.read().strip().split('.')[0])
    except (OSError, ValueError):
        return None


def _jp7_line() -> str:
    major = _nvidia_driver_major()
    if major is not None and major > _JP71_MAX_DRIVER:
        return 'jetson-jp7.2'
    if major is None:
        print('[hostarch] cannot read the NVIDIA driver version; assuming JetPack 7.1')
    return 'jetson-jp7.1'


@functools.cache
def acc_arch() -> str:
    """加速器架构：jetson-jp5 / jetson-jp6 / jetson-jp7.1 / jetson-jp7.2 / none。

    JetPack 7 报到小版本（见 _jp7_line），其余报大版本。

    ACC_ARCH 环境变量可覆盖（误判时不必重新构建镜像；install.sh 生成的 compose
    已带 env_file: .env）。
    """
    override = os.environ.get('ACC_ARCH', '').strip()
    if override:
        print(f'[hostarch] acc_arch overridden by env: {override}')
        return override

    m = _TEGRA_RE.search(_kernel_string())
    if not m:
        return ACC_NONE  # 不是 Jetson BSP 内核

    line = f'{m.group(1)}.{m.group(2)}'
    acc = _TEGRA_KERNEL_TO_ACC.get(line)
    if not acc:
        print(f'[hostarch] unknown tegra kernel {line}; treating host as {ACC_NONE}')
        return ACC_NONE
    if acc == 'jetson-jp7':
        return _jp7_line()
    return acc


@functools.cache
def cpu_arch() -> str:
    """CPU 架构：arm64 / amd64。CPU_ARCH 环境变量可覆盖。"""
    override = os.environ.get('CPU_ARCH', '').strip()
    if override:
        print(f'[hostarch] cpu_arch overridden by env: {override}')
        return override

    m = platform.machine().lower()
    if m in ('aarch64', 'arm64'):
        return 'arm64'
    if m in ('x86_64', 'amd64'):
        return 'amd64'
    print(f'[hostarch] unrecognized machine {m!r}; passing through')
    return m


def facets() -> dict:
    return {'acc_arch': acc_arch(), 'cpu_arch': cpu_arch()}
