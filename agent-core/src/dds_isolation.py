"""
dds_isolation.py — 启动时确认 DDS 隔离真的生效了。

为什么需要这个自检：`FASTRTPS_DEFAULT_PROFILES_FILE` 指向的文件不存在、路径写错、
或容器忘了挂载时，**FastDDS 会静默回退到默认传输**——绑到所有网卡上，隔离完全
失效，而日志里没有任何异常。机器人照常工作，直到有一天另一台机器人替它回答了
指令，才发现问题。

这类"失败了但看起来正常"的模式，在这套 peer 代码里已经出现过两次（rclpy 线程
静默死亡、画布门只写在注释里没实现），所以这里宁可多打一行日志。

检查两件事，都不需要网络往返：
  1. profile 文件是否存在且可读
  2. 本进程的 UDP socket 是否只绑在 loopback

第 2 项是真正的判据。第 1 项通过而第 2 项失败，说明文件被读了但没生效
（例如 XML 语法错误，FastDDS 同样会静默忽略）。
"""

import glob
import os


def _profile_path() -> str:
    return os.environ.get('FASTRTPS_DEFAULT_PROFILES_FILE', '')


# 镜像里随代码一起打包的那份，作为落盘来源。
_BUNDLED_PROFILE = '/deploy/dds-local.xml'


def ensure_profile() -> str:
    """确保 profile 文件真实存在，必要时从镜像里补齐。返回一行说明。

    两个动机：

    1. **已经装好的机器不该被要求重跑 install.sh。** 升级 core 镜像时，这里会
       把新版 profile 自动落到 /opt/phanthy-motus/（该目录是可写挂载）。

    2. **bind mount 会凭空造出一个目录。** compose 里写了
       `- /opt/phanthy-motus/dds-local.xml:...:ro` 而宿主机上没有这个文件时，
       Docker **自动建一个同名目录**，FastDDS 读不到有效 profile 就静默回退到
       所有网卡 —— 隔离失效、外表毫无异常。R1 上真实发生过，而且当时那台机器
       上 9 个 DDS 端口全绑在 0.0.0.0。

    目录这种情况这里会就地清掉再写文件；但**已经把它当目录挂进去的容器必须重启**
    才能看到文件，所以返回值会明确说出来。
    """
    path = _profile_path()
    if not path:
        return ''

    try:
        if os.path.isdir(path):
            # 只删空目录：非空说明是别的东西，不该由我们处置。
            try:
                os.rmdir(path)
            except OSError as e:
                return (f'⚠ {path} 是目录且无法删除（{e}）—— 这是 bind mount 在宿主机'
                        f'缺文件时自动创建的，隔离不会生效。请手工删除后重启相关容器')
            _write_profile(path)
            return (
                f'{path} 原本是个目录（宿主机缺文件时 bind mount 自动创建），已替换为文件。\n'
                f'[dds] ⚠ 其它 DDS 容器现在会启动失败，报 '
                f'"not a directory: Are you trying to mount a directory onto a file"'
                f' —— 它们创建时把挂载类型固化成了目录，docker start 改不了。\n'
                f'[dds] ⚠ 必须 **重建**（docker rm + run / compose up -d），仅 restart 无效。')

        if not os.path.exists(path):
            _write_profile(path)
            return f'{path} 缺失，已从镜像补齐'

        # 已存在且内容一致时不动它，避免每次启动都写盘
        if os.path.isfile(_BUNDLED_PROFILE):
            with open(_BUNDLED_PROFILE, 'rb') as f:
                bundled = f.read()
            with open(path, 'rb') as f:
                current = f.read()
            if current != bundled:
                _write_profile(path)
                return f'{path} 与镜像内版本不一致，已更新'
    except OSError as e:
        return f'⚠ 无法确保 profile 文件（{type(e).__name__}: {e}）'
    return ''


def _write_profile(path: str) -> None:
    with open(_BUNDLED_PROFILE, 'rb') as src:
        data = src.read()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as dst:
        dst.write(data)
    os.chmod(path, 0o644)


# RTPS 端口分配：每个 domain 独占 250 个端口，块内布局为
#   base+0 发现多播、base+1 用户多播、base+10+2i / base+11+2i 第 i 个参与者单播。
# 这里的「参与者」是同一台机器上的 DDS 进程，与机器人数量无关
# —— 实测两个容器就占了 5 个端口（17900, 17910-17913）。
_DOMAIN_PORT_SPAN = 250


def _dds_port_range() -> tuple[int, int]:
    """本 domain 占用的端口区间（闭区间）。

    从 ROS_DOMAIN_ID 推导而非硬编码 17900，只是为了 domain 万一变了自检不会
    静默失效 —— 方案本身所有机器人一律用 42，不做任何分配。
    """
    try:
        domain = int(os.environ.get('ROS_DOMAIN_ID', '0'))
    except ValueError:
        domain = 0
    base = 7400 + _DOMAIN_PORT_SPAN * domain
    return base, base + _DOMAIN_PORT_SPAN - 1


def _external_dds_binds() -> list[str]:
    """本进程绑在非 loopback 地址上的 **DDS** UDP socket。

    只看本 domain 的端口段。第一版统计了进程全部非 loopback UDP socket，把
    mDNS 的 5353 也算成泄漏 —— 而 mDNS 本来就必须走网络（peer 发现依赖它），
    于是隔离明明生效却报 isolated=false。会误报的自检很快会被无视，
    那还不如没有。

    各机器人的 DDS 用的是同一组端口（同 domain 必然如此），靠多播互相发现，
    这正是串台的成因；隔离靠的是绑定网卡从 0.0.0.0 收到 127.0.0.1，端口不变。
    所以这里判断的是「绑在哪」，端口段只用来区分 DDS 与非 DDS。

    直接读 /proc，不依赖 ss/lsof（容器里未必有，也未必有权限看到 pid）。
    空列表 = 隔离生效。
    """
    lo_port, hi_port = _dds_port_range()
    inodes = set()
    for fd in glob.glob(f'/proc/{os.getpid()}/fd/*'):
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith('socket:['):
            inodes.add(target[8:-1])
    if not inodes:
        return []

    external = []
    try:
        with open('/proc/net/udp') as f:
            lines = f.readlines()[1:]
    except OSError:
        return []

    for line in lines:
        fields = line.split()
        if len(fields) < 10 or fields[9] not in inodes:
            continue
        hex_ip, hex_port = fields[1].split(':')
        # /proc 里是小端十六进制
        ip = '.'.join(str(int(hex_ip[i:i + 2], 16)) for i in (6, 4, 2, 0))
        port = int(hex_port, 16)
        # 只关心 DDS。mDNS(5353) 等其它服务本就该走网络，算进来会让隔离
        # 正常时误报 —— 这正是第一版的 bug。
        if not (lo_port <= port <= hi_port):
            continue
        # 127.0.0.0/8 是本机；224.0.0.0/4 是多播组，加入在哪个网卡由白名单决定，
        # 不能凭组地址判断，所以这里不算作外部绑定。
        first = int(ip.split('.')[0])
        if ip.startswith('127.') or 224 <= first <= 239:
            continue
        external.append(f'{ip}:{port}')
    return sorted(set(external))


def check_and_report() -> dict:
    """检查并打印结论。返回结构供 /api 暴露。

    只报告，不抛异常也不退出：隔离失效时机器人本身仍能工作，贸然让 Agent Core
    起不来会把一个配置问题升级成一次停机。
    """
    profile = _profile_path()
    result = {
        'profile_configured': bool(profile),
        'profile_path': profile,
        'profile_readable': bool(profile) and os.path.isfile(profile) and os.access(profile, os.R_OK),
        'profile_is_directory': False,
        'external_binds': [],
        'isolated': False,
        'detail': '',
    }

    if not profile:
        result['detail'] = ('未配置 FASTRTPS_DEFAULT_PROFILES_FILE —— DDS 未隔离，'
                            '同网段的其他机器人会收到本机的指令')
        print(f'[dds] ⚠ {result["detail"]}')
        return result

    # 目录要单独报。报成"不可读"会让人去查权限，而真正的原因是宿主机缺文件、
    # bind mount 自动建了同名目录 —— 这两件事的修法完全不同。
    if os.path.isdir(profile):
        result['profile_is_directory'] = True
        result['detail'] = (
            f'{profile} 是**目录**不是文件 —— 宿主机上缺这个文件时 bind mount 会自动'
            f'建一个同名目录，FastDDS 静默回退到所有网卡，隔离失效。'
            f'需在宿主机删除该目录、放上真正的 XML，然后重启所有 DDS 容器')
        print(f'[dds] ⚠ {result["detail"]}')
        return result

    if not result['profile_readable']:
        result['detail'] = (f'profile 文件不可读: {profile} —— FastDDS 会静默回退到'
                            f'默认传输，隔离失效。检查该文件是否已挂载进容器')
        print(f'[dds] ⚠ {result["detail"]}')
        return result

    external = _external_dds_binds()
    result['external_binds'] = external
    if external:
        result['detail'] = (f'profile 已加载但仍绑到外部地址 {", ".join(external[:4])} '
                            f'—— XML 可能有语法错误而被忽略')
        print(f'[dds] ⚠ {result["detail"]}')
        return result

    result['isolated'] = True
    result['detail'] = 'DDS 已隔离到本机（UDP socket 仅绑 loopback）'
    print(f'[dds] {result["detail"]}')
    return result
