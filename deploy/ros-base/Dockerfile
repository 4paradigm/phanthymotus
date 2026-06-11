ARG APT_MIRROR=mirrors.tencentyun.com
ARG ROS_BASE_IMAGE=mirror.ccs.tencentyun.com/library/ros:humble-ros-base
FROM ${ROS_BASE_IMAGE}
ARG APT_MIRROR

# Switch apt mirror (conditional)
RUN if [ -n "${APT_MIRROR}" ]; then \
        sed -i "s/archive.ubuntu.com/${APT_MIRROR}/g; s/security.ubuntu.com/${APT_MIRROR}/g; s/ports.ubuntu.com/${APT_MIRROR}/g" /etc/apt/sources.list; \
    fi && \
    rm -f /etc/apt/sources.list.d/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Build audio_msgs ROS2 package
COPY audio_msgs/ /ros_ws/src/audio_msgs/
RUN cd /ros_ws && \
    . /opt/ros/humble/setup.sh && \
    colcon build --packages-select audio_msgs --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source audio_msgs in entrypoint so all downstream images inherit it
RUN sed -i 's|exec "$@"|source /ros_ws/install/setup.bash\nexec "$@"|' /ros_entrypoint.sh
