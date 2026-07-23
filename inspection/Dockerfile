ARG ROS_BASE_IMAGE=bj-warehouse.tencentcloudcr.com/phanthy-motus/ros-base:latest
ARG PYPI_MIRROR=https://pypi.org/simple/
ARG UBUNTU_PORTS_MIRROR=https://mirrors.ustc.edu.cn/ubuntu-ports
FROM ${ROS_BASE_IMAGE}
ARG PYPI_MIRROR
ARG UBUNTU_PORTS_MIRROR

RUN sed -i "s|http://mirrors.tencentyun.com/ubuntu-ports|${UBUNTU_PORTS_MIRROR}|g" /etc/apt/sources.list && \
    ! grep -q "mirrors.tencentyun.com/ubuntu-ports" /etc/apt/sources.list && \
    apt-get update -o Acquire::Retries=3 && \
    mv /sbin/ldconfig /sbin/ldconfig.real && \
    ln -s /bin/true /sbin/ldconfig && \
    apt-get install -y --no-install-recommends \
        python3-gi \
        python3-gst-1.0 \
        gir1.2-gstreamer-1.0 \
        gir1.2-gst-plugins-base-1.0 \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base-apps \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly && \
    find /usr/lib/aarch64-linux-gnu/gstreamer-1.0 \
        -maxdepth 1 -type f -size 0 -delete && \
    test -z "$(find /usr/lib/aarch64-linux-gnu/gstreamer-1.0 \
        -maxdepth 1 -type f -size 0 -print -quit)" && \
    mv /sbin/ldconfig.real /sbin/ldconfig && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements.txt /work/requirements.txt
ENV INSPECTION_PYTHON_DIR=/opt/inspection-python
RUN python3 -m pip install --no-cache-dir --ignore-installed \
        --target ${INSPECTION_PYTHON_DIR} \
        -i ${PYPI_MIRROR} \
        -r /work/requirements.txt && \
    PYTHONPATH=${INSPECTION_PYTHON_DIR} python3 -c \
        'import pathlib, qcloud_cos, yaml; path = pathlib.Path(yaml.__file__).resolve(); assert str(path).startswith("/opt/inspection-python/"), path; assert yaml.__version__ == "6.0.2", yaml.__version__; print("isolated inspection Python dependencies PASS")'
ENV PYTHONPATH=/opt/inspection-python:${PYTHONPATH}

COPY . /work/
COPY deploy/ /deploy/

EXPOSE 15671
CMD ["/bin/bash", "-c", "if [ -f /opt/ros/humble/install/setup.bash ]; then source /opt/ros/humble/install/setup.bash; else source /opt/ros/humble/setup.bash; fi && source /ros_ws/install/setup.bash && python3 /work/main.py"]
