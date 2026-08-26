"""hostarch.py — 在容器内探测宿主的加速器 / CPU 架构。

用途：向 resource-center 请求镜像目录时带上本机架构，让它只返回能在本机运行的镜像
（perception / actucore 是 Jetson 专用镜像，按 JetPack 大版本分别构建，不能混装）。

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
}

_TEGRA_RE = re.compile(r'(\d+)\.(\d+)\.\d+\S*-tegra')


def _kernel_string() -> str:
    try:
        with open('/proc/version') as f:
            return f.read()
    except OSError:
        return platform.release()  # 容器内同样是宿主内核


@functools.cache
def acc_arch() -> str:
    """加速器架构：jetson-jp5 / jetson-jp6 / none。

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
