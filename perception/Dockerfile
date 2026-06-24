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

# Pre-download silero VAD model via torch.hub (avoids runtime network requirement)
RUN python3 -c "\
import torch; \
torch.hub.load('snakers4/silero-vad', 'silero_vad', source='github', trust_repo=True); \
print('silero hub model cached')"

WORKDIR /work
COPY main.py    /work/main.py
COPY plugins/   /work/plugins/
COPY config.yaml /work/config.yaml

EXPOSE 15720
EXPOSE 15721
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash && python3 /work/main.py"]
