ARG FAST_LIVO2_BASE_IMAGE=phanthy-fast-livo2:g1-1fcd0d0-n3save1
FROM ${FAST_LIVO2_BASE_IMAGE}

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
LABEL org.opencontainers.image.description="Session mapping plus canonical map/base_link adapter"
LABEL org.opencontainers.image.licenses="Apache-2.0 AND GPL-2.0-only AND GPL-3.0-only"
LABEL org.opencontainers.image.fast-livo2-revision="1fcd0d05cadaeb25ca59fd87cda95aaaee41e3ea"

CMD ["/bin/bash", "-lc", "source /opt/ros/humble/setup.bash && source /opt/fast_livo_ws/install/setup.bash && source /fast_livo2_card_ws/install/setup.bash && exec ros2 launch g1_fast_livo2 g1_fast_livo2.launch.py"]
