ARG PYPI_MIRROR=https://pypi.org/simple/
ARG ROS_BASE_IMAGE=bj-warehouse.tencentcloudcr.com/phanthy-motus/ros-base:latest
FROM ${ROS_BASE_IMAGE}
ARG PYPI_MIRROR

# Install ffmpeg + docker CLI + compose plugin (static binaries, avoid apt docker.io QEMU segfault).
# --retry survives the transient TLS drops this network path has hit more than
# once (curl exit 56 / HTTP2 stream errors) — retrying the whole RUN restarts
# the ffmpeg apt install too, which is the expensive part.
# The docker CLI binary is mirrored to COS (same pattern unitree/g1 uses for
# cyclonedds) rather than Tsinghua's mirror, which was the more frequent
# offender. The compose plugin is proxied through ghfast.top (same GitHub
# mirror already used in perception/Dockerfile.jetson).
ARG DOCKER_CLI_URL=https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public/docker-27.5.1-aarch64.tgz
RUN apt-get -o Acquire::AllowInsecureRepositories=true update && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
        python3-pip python3-dbus ffmpeg curl && \
    curl -fsSL --retry 3 --retry-delay 2 --retry-connrefused --retry-max-time 60 \
        ${DOCKER_CLI_URL} \
        | tar -xz --strip-components=1 -C /usr/local/bin/ docker/docker && \
    mkdir -p /usr/local/lib/docker/cli-plugins && \
    curl -fsSL --retry 3 --retry-delay 2 --retry-connrefused --retry-max-time 60 \
        https://ghfast.top/https://github.com/docker/compose/releases/download/v2.37.0/docker-compose-linux-aarch64 \
        -o /usr/local/lib/docker/cli-plugins/docker-compose && \
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} uv

WORKDIR /work

COPY pyproject.toml /work/
ENV UV_INDEX_URL=${PYPI_MIRROR}
RUN uv lock && uv sync --no-dev --no-install-project

# ROS_BASE_IMAGE contract: the configured ROS Humble install supplies rclpy and
# standard sensor_msgs/nav_msgs packages. Link that install into the venv and
# fail the build if the bridge's concrete message types are missing; no extra
# package or image layer is installed here.
RUN echo "/opt/ros/humble/lib/python3.10/site-packages" >> .venv/lib/python3.10/site-packages/ros2.pth && \
    echo "/opt/ros/humble/local/lib/python3.10/dist-packages" >> .venv/lib/python3.10/site-packages/ros2.pth && \
    echo "/ros_ws/install/audio_msgs/local/lib/python3.10/dist-packages" >> .venv/lib/python3.10/site-packages/ros2.pth && \
    echo "/usr/lib/python3/dist-packages" >> .venv/lib/python3.10/site-packages/system.pth && \
    /bin/bash -c "source /opt/ros/humble/setup.bash && .venv/bin/python -c 'import rclpy; from sensor_msgs.msg import PointCloud2; from nav_msgs.msg import OccupancyGrid, Odometry, Path'"

COPY web/      /work/web/
COPY src/      /work/src/
COPY tools/    /work/tools/
COPY resource/ /work/resource/
COPY deploy/   /deploy/

# Bake defaults to a path immune to volume mounts (for cold-start init)
COPY resource/memory/defaults/ /opt/defaults/memory/

# `file` —— 模型拿它判断下载下来的东西到底是不是图片。Tianyi 实测：LLM 下了两张人物
# 照片后连着两次 `file /tmp/x.png` 都撞上 `/bin/sh: 1: file: not found`，白烧两轮才改用
# 别的办法。装一个 100KB 的包比让它每次现场摸索便宜。
#
# **单独一层、放在文件末尾**，不并进上面那个 apt RUN：并进去会让它之后的 uv sync（依赖
# 安装，这个镜像最贵的一层）整个失效重建。同 perception 那边记下的教训。
#
# 装的时候把 ldconfig 挡掉，装完恢复。
#
# x86 上交叉构建 arm64 走的是 qemu-user，而 `libc-bin` 的触发器会调 `ldconfig` ——
# 它在 qemu 下**偶发**段错误，dpkg 随之 exit 139，整个构建失败：
#
#     Processing triggers for libc-bin (2.35-0ubuntu3.13) ...
#     qemu: uncaught target signal 11 (Segmentation fault) - core dumped
#     dpkg: error processing package libc-bin (--configure): ... exit status 139
#
# 偶发比必现更麻烦：同一层、同一个包版本，在构建主机上连过 8 次，第 9 次挂掉，
# 看起来像是代码改坏了什么。挡掉之后，这一层不再取决于 qemu 能不能跑通 ldconfig。
#
# 不补跑 ldconfig 是有意的：libmagic 落在 /usr/lib/<triplet>，那是动态链接器的默认
# 搜索路径 —— 缓存里没有它，链接器会退回去扫目录，`file` 照常能起来。末尾那句
# `file --version` 就是当场证明这一点，而不是等模型在真机上撞（这个包本来就是那么
# 加进来的）。
# `.real` 这个名字**已经被占了**：基础镜像里 /usr/sbin/ldconfig 是个 387 字节的包装
# 脚本，真二进制在 /usr/sbin/ldconfig.real（/sbin 是 /usr/sbin 的符号链接，所以
# /sbin/ldconfig.real 是同一个文件）。往那个名字上 mv 会**直接覆盖掉真二进制**，
# 容器里从此再也跑不了 ldconfig —— 第一版就是这么干的，而当时加的 `file --version`
# 照样通过，因为它验的不是被破坏的那一处。所以用一个不可能撞名的临时名字，
# 并且在同一层里把 ldconfig 那两个文件也验一遍。
#
# **验它还在，但不执行它。** 第二版这里写的是 `ldconfig --version`，而在 qemu 下执行
# ldconfig 恰恰就是这一整层要绕开的那个随机段错误（实测基线 2/5 失败）。于是那句检查
# 本身开始随机崩掉构建 —— 一个用「跑一次」来证明「没坏」的检查，放在一个「跑一次就
# 可能崩」的环境里。改成看文件在不在、是不是原来那个 ELF：`cp -a` 保留大小与 inode
# 之外的一切，覆盖掉真二进制的那种坏法（第一版）在这里一眼就能看出来。
RUN cp -a /usr/sbin/ldconfig /tmp/ldconfig.masked-orig && \
    ln -sf /bin/true /usr/sbin/ldconfig && \
    apt-get -o Acquire::AllowInsecureRepositories=true update && \
    apt-get install -y --no-install-recommends --allow-unauthenticated file && \
    rm -rf /var/lib/apt/lists/* && \
    mv /tmp/ldconfig.masked-orig /usr/sbin/ldconfig && \
    file --version && \
    test -x /usr/sbin/ldconfig.real && \
    file /usr/sbin/ldconfig.real | grep -q ELF && \
    test "$(stat -c %s /usr/sbin/ldconfig.real)" -gt 100000

ARG IMAGE_TAG=unknown
ENV IMAGE_TAG=${IMAGE_TAG}
RUN echo "${IMAGE_TAG}" > /work/VERSION

# One write per log line instead of 8 KB block flushes that can tear a Docker
# log record mid-write.
ENV PYTHONUNBUFFERED=1
# ANSI colour codes are control bytes; in a non-tty container log they are
# exactly the "garbage characters" that make `docker logs` unreadable.
ENV RCUTILS_COLORIZED_OUTPUT=0

EXPOSE 15678
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash && uv run src/start.py"]
