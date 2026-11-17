FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential libgl1 libegl1 libegl-mesa0 libosmesa6 libglfw3 libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/upstream
ARG SUPERMIMIC_REV=abc124c1bd5976341034668af3d12c087f2f1f55
ARG MJLAB_REV=dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd
RUN git clone https://github.com/HybridRobotics/mjlab_research.git && git -C mjlab_research checkout ${MJLAB_REV}
RUN git clone https://github.com/HybridRobotics/supermimic.git && git -C supermimic checkout ${SUPERMIMIC_REV}
RUN pip install --no-cache-dir ./mjlab_research ./supermimic 'onnxruntime>=1.23,<2' 'Pillow>=11,<13'
WORKDIR /app
COPY actucore/motion_service/ /app/motion_service/
ENV PYTHONUNBUFFERED=1 WANDB_MODE=offline MUJOCO_GL=egl
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
EXPOSE 15732
ENTRYPOINT ["python", "-m", "motion_service.server"]
CMD ["--host", "0.0.0.0", "--data-dir", "/data", "--profiles", "/profiles/profiles.json"]
