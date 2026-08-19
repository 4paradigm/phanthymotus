ARG PYPI_MIRROR=https://pypi.org/simple/
ARG APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
ARG ROS_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu
ARG ROS_BASE_IMAGE=bj-warehouse.tencentcloudcr.com/phanthy-motus/ros-base:latest
FROM ${ROS_BASE_IMAGE}
ARG PYPI_MIRROR APT_MIRROR ROS_MIRROR

# Install ffmpeg + docker CLI + compose plugin (static binaries, avoid apt docker.io QEMU segfault).
# --retry survives the transient TLS drops this network path has hit more than
# once (curl exit 56 / HTTP2 stream errors) — retrying the whole RUN restarts
# the ffmpeg apt install too, which is the expensive part.
# The docker CLI binary is mirrored to COS (same pattern unitree/g1 uses for
# cyclonedds) rather than Tsinghua's mirror, which was the more frequent
# offender. The compose plugin is proxied through ghfast.top (same GitHub
# mirror already used in perception/Dockerfile.jetson).
# ros-humble-nav-msgs is needed by the navigation canvas renderers (odom / path
# / costmap), which decode nav_msgs on the Agent Core side.
ARG DOCKER_CLI_URL=https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public/docker-27.5.1-aarch64.tgz
RUN if [ -n "${APT_MIRROR}" ]; then \
        find /etc/apt -type f \( -name '*.list' -o -name '*.sources' \) \
            -exec sed -E -i \
            "s#https?://(mirrors\.tencentyun\.com|ports\.ubuntu\.com)/ubuntu-ports/?#https://${APT_MIRROR}/ubuntu-ports/#g" {} +; \
    fi && \
    if [ -n "${ROS_MIRROR}" ]; then \
        find /etc/apt -type f \( -name '*.list' -o -name '*.sources' \) \
            -exec sed -E -i \
            "s#https?://packages\.ros\.org/ros2/ubuntu#${ROS_MIRROR}#g" {} +; \
    fi && \
    apt-get -o Acquire::AllowInsecureRepositories=true update && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
        python3-pip python3-dbus ffmpeg curl ros-humble-nav-msgs && \
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

# Link system ROS2 packages into venv so sensor_msgs/nav_msgs/rclpy are importable
RUN echo "/opt/ros/humble/lib/python3.10/site-packages" >> .venv/lib/python3.10/site-packages/ros2.pth && \
    echo "/opt/ros/humble/local/lib/python3.10/dist-packages" >> .venv/lib/python3.10/site-packages/ros2.pth && \
    echo "/ros_ws/install/audio_msgs/local/lib/python3.10/dist-packages" >> .venv/lib/python3.10/site-packages/ros2.pth && \
    echo "/usr/lib/python3/dist-packages" >> .venv/lib/python3.10/site-packages/system.pth

COPY web/      /work/web/
COPY src/      /work/src/
COPY resource/ /work/resource/
COPY deploy/   /deploy/

# Bake defaults to a path immune to volume mounts (for cold-start init)
COPY resource/memory/defaults/ /opt/defaults/memory/

ARG IMAGE_TAG=unknown
ENV IMAGE_TAG=${IMAGE_TAG}
RUN echo "${IMAGE_TAG}" > /work/VERSION

EXPOSE 15678
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash && uv run src/start.py"]
