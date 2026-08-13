ARG FAST_LIVO2_BASE_IMAGE=phanthy-fast-livo2:g1-1fcd0d0-n3save1
FROM ${FAST_LIVO2_BASE_IMAGE}

ARG APT_UBUNTU_MIRROR=mirrors.tuna.tsinghua.edu.cn
ARG APT_ROS_MIRROR=mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu

RUN set -eu; \
    for source in \
      /etc/apt/sources.list \
      /etc/apt/sources.list.d/*.list \
      /etc/apt/sources.list.d/*.sources; do \
      [ -f "${source}" ] || continue; \
      sed -E -i \
        "s|https?://ports\\.ubuntu\\.com/ubuntu-ports|https://${APT_UBUNTU_MIRROR}/ubuntu-ports|g; \
         s|https?://mirrors\\.tencentyun\\.com/ubuntu-ports|https://${APT_UBUNTU_MIRROR}/ubuntu-ports|g; \
         s|https?://packages\\.ros\\.org/ros2/ubuntu|https://${APT_ROS_MIRROR}|g; \
         /^[[:space:]]*deb-src[[:space:]]/d; \
         s|^Types:[[:space:]]*deb[[:space:]]+deb-src[[:space:]]*$|Types: deb|; \
         s|^Types:[[:space:]]*deb-src([[:space:]]+deb)?[[:space:]]*$|Types: deb|" \
        "${source}"; \
    done; \
    if grep -R -E \
      'https?://(ports\.ubuntu\.com/ubuntu-ports|packages\.ros\.org/ros2/ubuntu)' \
      /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null; then \
      echo 'official Ubuntu or ROS APT source remains after mirror rewrite' >&2; \
      exit 1; \
    fi; \
    if grep -R -E \
      '^[[:space:]]*deb-src[[:space:]]|^Types:.*deb-src' \
      /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null; then \
      echo 'APT source-package entries remain after binary-only rewrite' >&2; \
      exit 1; \
    fi; \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      ros-humble-ros2bag \
      ros-humble-rosbag2-storage-mcap && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /fast_livo2_card_ws
COPY g1_fast_livo2/ /fast_livo2_card_ws/src/g1_fast_livo2/
RUN . /opt/ros/humble/setup.sh && \
    . /opt/fast_livo_ws/install/setup.sh && \
    colcon build --packages-select g1_fast_livo2 --symlink-install

COPY g1_lio.yaml /config/g1_lio.yaml

ENV ROS_DOMAIN_ID=42
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV FASTDDS_BUILTIN_TRANSPORTS=UDPv4
ENV RCUTILS_LOGGING_BUFFERED_STREAM=1

LABEL org.opencontainers.image.title="Phanthy Motus FAST-LIVO2 Perception companion"
LABEL org.opencontainers.image.description="Session mapping, automatic MCAP capture and canonical map/base_link adapter"
LABEL org.opencontainers.image.licenses="Apache-2.0 AND GPL-2.0-only AND GPL-3.0-only"
LABEL org.opencontainers.image.fast-livo2-revision="1fcd0d05cadaeb25ca59fd87cda95aaaee41e3ea"

CMD ["/bin/bash", "-lc", "source /opt/ros/humble/setup.bash && source /opt/fast_livo_ws/install/setup.bash && source /fast_livo2_card_ws/install/setup.bash && exec ros2 launch g1_fast_livo2 g1_fast_livo2.launch.py"]
