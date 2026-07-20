ARG PYPI_MIRROR=https://pypi.org/simple/
ARG ROS_BASE_IMAGE=bj-warehouse.tencentcloudcr.com/phanthy-motus/ros-base:latest
# ARG OCR_MODEL_BASE_URL=http://172.28.4.81:34567/guohongjie/models/ppocr-v6-tiny
ARG OCR_MODEL_BASE_URL=http://172.28.4.81:8200/ppocr-v6-tiny
FROM ${ROS_BASE_IMAGE}
ARG PYPI_MIRROR OCR_MODEL_BASE_URL

ENV OCR_MODEL_BASE_URL=${OCR_MODEL_BASE_URL}

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
    pyyaml requests silero-vad websockets webrtcvad websocket-client dashscope \
    onnxruntime pyclipper "opencv-python-headless==4.7.0.72"
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} \
    kiss-icp scipy
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} \
    ultralytics
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} \
    sherpa-onnx

# Fail the build early if the OCR OpenCV dependency cannot be loaded.
RUN python3 -c "import cv2; print('OpenCV', cv2.__version__, cv2.__file__)"

WORKDIR /work
COPY main.py    /work/main.py
COPY plugins/   /work/plugins/
COPY utils/     /work/utils/
COPY config.yaml /work/config.yaml
COPY config/fastdds_large_message.xml /opt/phanthy-motus/config/fastdds_large_message.xml
COPY deploy/     /deploy/

EXPOSE 15720
EXPOSE 15721
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash && python3 /work/main.py"]
