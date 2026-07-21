ARG PYPI_MIRROR=https://pypi.org/simple/
ARG ROS_BASE_IMAGE=bj-warehouse.tencentcloudcr.com/phanthy-motus/ros-base:latest
ARG ONNXRUNTIME_PKG=onnxruntime
FROM ${ROS_BASE_IMAGE}
ARG PYPI_MIRROR
ARG ONNXRUNTIME_PKG

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
    pyyaml requests pillow silero-vad websockets webrtcvad websocket-client dashscope \
    rapidocr ${ONNXRUNTIME_PKG}
# NOTE:
# - Default ONNXRUNTIME_PKG=onnxruntime is CPU-only.
# - For x86 GPU: build with --build-arg ONNXRUNTIME_PKG=onnxruntime-gpu
# - For Jetson/ARM64: install the Jetson-specific onnxruntime wheel before this step
#   and set ONNXRUNTIME_PKG to the local wheel path / package name.
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} \
    kiss-icp scipy
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} \
    ultralytics
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} \
    sherpa-onnx

WORKDIR /work
COPY main.py    /work/main.py
COPY plugins/   /work/plugins/
COPY utils/     /work/utils/
COPY config.yaml /work/config.yaml
COPY config/    /work/config/
COPY deploy/     /deploy/

EXPOSE 15720
EXPOSE 15721
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash && python3 /work/main.py"]
