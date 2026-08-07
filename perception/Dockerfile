ARG PYPI_MIRROR=https://pypi.org/simple/
ARG ROS_BASE_IMAGE=bj-warehouse.tencentcloudcr.com/phanthy-motus/ros-base:latest
FROM ${ROS_BASE_IMAGE}
ARG PYPI_MIRROR

RUN apt-get update && \
    mv /sbin/ldconfig /sbin/ldconfig.real && \
    ln -s /bin/true /sbin/ldconfig && \
    apt-get install -y --no-install-recommends python3-pip && \
    mv /sbin/ldconfig.real /sbin/ldconfig && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url ${PYPI_MIRROR} \
    torch
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} \
    pyyaml requests silero-vad websockets webrtcvad websocket-client dashscope
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} \
    kiss-icp scipy
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} \
    ultralytics
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} \
    sherpa-onnx
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} \
    phonemizer

# Override base image RMW (some ros-base tags ship cyclonedds by default)
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV ROS_DOMAIN_ID=42

WORKDIR /work
COPY main.py    /work/main.py
COPY plugins/   /work/plugins/
COPY utils/     /work/utils/
COPY config.yaml /work/config.yaml
COPY deploy/     /deploy/

EXPOSE 15720
EXPOSE 15721
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash && python3 /work/main.py"]
