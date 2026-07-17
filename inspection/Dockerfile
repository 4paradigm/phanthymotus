ARG ROS_BASE_IMAGE=bj-warehouse.tencentcloudcr.com/phanthy-motus/ros-base:latest
ARG PYPI_MIRROR=https://pypi.org/simple/
FROM ${ROS_BASE_IMAGE}
ARG PYPI_MIRROR

WORKDIR /work
COPY requirements.txt /work/requirements.txt
RUN python3 -m pip install --no-cache-dir -i ${PYPI_MIRROR} -r /work/requirements.txt

COPY . /work/

EXPOSE 15671
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash && python3 /work/main.py"]
