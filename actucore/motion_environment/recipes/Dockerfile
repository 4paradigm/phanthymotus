FROM python:3.13.9-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libegl1 libegl-mesa0 libosmesa6 libglfw3 libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/upstream
RUN git init mjlab_research && git -C mjlab_research remote add origin https://github.com/HybridRobotics/mjlab_research.git && git -C mjlab_research -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=120 fetch --depth=1 --progress origin dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd && git -C mjlab_research checkout --detach FETCH_HEAD
RUN git init supermimic && git -C supermimic remote add origin https://github.com/HybridRobotics/supermimic.git && git -C supermimic -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=120 fetch --depth=1 --progress origin abc124c1bd5976341034668af3d12c087f2f1f55 && git -C supermimic checkout --detach FETCH_HEAD
COPY constraints.txt /opt/constraints.txt
COPY wheels/ /opt/wheels/
COPY install_cached.py /opt/install_cached.py
RUN python /opt/install_cached.py
RUN pip install --no-cache-dir torch==2.14.1+cu126 --index-url https://download.pytorch.org/whl/cu126
RUN python -c "import importlib.metadata as m, subprocess, sys; installed=m.packages_distributions(); subprocess.check_call([sys.executable,'-m','pip','install','--no-cache-dir','https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl']) if 'warp' not in installed or m.version('warp-lang') != '1.18.0+cu12' else None"
RUN pip install --no-cache-dir -c /opt/constraints.txt ./mjlab_research ./supermimic onnxruntime Pillow
# Torch/Inductor and Triton compile native kernels when the official GPU task starts.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY motion_service/ /app/motion_service/
COPY profiles/ /profiles/
COPY runtime_probe.py entrypoint.py /app/
ENV PYTHONUNBUFFERED=1 WANDB_MODE=offline MUJOCO_GL=egl
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
EXPOSE 15732
ENTRYPOINT ["python", "/app/entrypoint.py"]
