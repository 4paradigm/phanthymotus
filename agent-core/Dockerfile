ARG PYPI_MIRROR=https://pypi.org/simple/
ARG ROS_BASE_IMAGE=bj-warehouse.tencentcloudcr.com/phanthy-motus/ros-base:latest
FROM ${ROS_BASE_IMAGE}
ARG PYPI_MIRROR

RUN apt-get -o Acquire::AllowInsecureRepositories=true update && \
    apt-get install -y --no-install-recommends --allow-unauthenticated python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} uv

WORKDIR /work

COPY pyproject.toml /work/
ENV UV_INDEX_URL=${PYPI_MIRROR}
RUN uv lock && uv sync --no-dev --no-install-project

# Link system ROS2 packages into venv so sensor_msgs/rclpy are importable
RUN echo "/opt/ros/humble/lib/python3.10/site-packages" >> .venv/lib/python3.10/site-packages/ros2.pth && \
    echo "/opt/ros/humble/local/lib/python3.10/dist-packages" >> .venv/lib/python3.10/site-packages/ros2.pth && \
    echo "/ros_ws/install/audio_msgs/local/lib/python3.10/dist-packages" >> .venv/lib/python3.10/site-packages/ros2.pth

COPY web/      /work/web/
COPY src/      /work/src/
COPY resource/ /work/resource/

# Bake defaults to a path immune to volume mounts (for cold-start init)
COPY resource/memory/defaults/ /opt/defaults/memory/

ARG IMAGE_TAG=unknown
ENV IMAGE_TAG=${IMAGE_TAG}
RUN echo "${IMAGE_TAG}" > /work/VERSION

EXPOSE 15678
CMD ["/bin/bash", "-c", "rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null; source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash && uv run src/start.py"]
