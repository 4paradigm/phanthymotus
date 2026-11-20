"""Render the bound MuJoCo model without touching the live simulation.

The caller holds its session lock while constructing FrameRenderer and copies
qpos/qvel under that lock before calling render(). MuJoCo's GL context, private
model/data, and renderer are owned by one worker thread for their whole lifetime.
Select MUJOCO_GL (and MUJOCO_EGL_DEVICE_ID on multi-GPU hosts) before importing
MuJoCo. A graphics failure is reported; no replacement geometry is generated.
"""
from __future__ import annotations

import copy
from concurrent.futures import Future, TimeoutError as FutureTimeout
import io
import math
import os
import queue
import threading


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return value


class FrameRenderer:
    def __init__(self, model, max_width=960, max_height=640, queue_size=2):
        self.max_width = _integer(max_width, "max_width", 64, 1920)
        self.max_height = _integer(max_height, "max_height", 64, 1080)
        self._queue = queue.Queue(_integer(queue_size, "queue_size", 1, 8))
        self._lock = threading.Lock()
        self._closed = False
        self._close_error = None
        self._ready = Future()
        self._thread = threading.Thread(target=self._run, args=(model,),
                                        name="mujoco-render", daemon=True)
        self._thread.start()
        # Wait for the worker's independent model copy while the caller still
        # holds its session lock. No GL context is created until the first frame.
        self._nq, self._nv = self._ready.result()

    def render(self, snapshot, camera=None, width=960, height=640, timeout=5, jpeg_quality=88):
        import numpy as np

        width = _integer(width, "width", 64, self.max_width)
        height = _integer(height, "height", 64, self.max_height)
        jpeg_quality = _integer(jpeg_quality, "jpeg_quality", 40, 95)
        timeout = _number(timeout, "timeout")
        if not 0 < timeout <= 30:
            raise ValueError("timeout must be greater than zero and at most 30 seconds")
        if not isinstance(snapshot, dict):
            raise ValueError("snapshot must contain qpos and qvel")
        state = {}
        for name, size in (("qpos", self._nq), ("qvel", self._nv)):
            try:
                values = np.array(snapshot[name], dtype=float, copy=True)
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"snapshot {name} must contain {size} finite numbers") from exc
            if values.shape != (size,) or not np.isfinite(values).all():
                raise ValueError(f"snapshot {name} must contain {size} finite numbers")
            state[name] = values
        state["time"] = _number(snapshot.get("time", 0.0), "snapshot time")
        view = self._camera_parameters(camera)
        future = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("MuJoCo renderer is closed")
            try:
                self._queue.put_nowait((future, state, view, width, height, jpeg_quality))
            except queue.Full as exc:
                raise RuntimeError("MuJoCo renderer is busy; retry the frame request") from exc
        try:
            return future.result(timeout=timeout)
        except FutureTimeout as exc:
            future.cancel()  # Queued, obsolete frames must never build a backlog.
            raise TimeoutError("MuJoCo frame rendering timed out") from exc

    @staticmethod
    def _camera_parameters(camera):
        if camera is None:
            return {}
        if not isinstance(camera, dict) or set(camera) - {"lookat", "distance", "azimuth", "elevation"}:
            raise ValueError("camera accepts lookat, distance, azimuth and elevation")
        view = {}
        for name in ("distance", "azimuth", "elevation"):
            if name in camera:
                view[name] = _number(camera[name], f"camera {name}")
        if "distance" in view and not .1 <= view["distance"] <= 50:
            raise ValueError("camera distance must be between 0.1 and 50 metres")
        if "elevation" in view and not -89 <= view["elevation"] <= 89:
            raise ValueError("camera elevation must be between -89 and 89 degrees")
        if "lookat" in camera:
            point = camera["lookat"]
            if not isinstance(point, (list, tuple)) or len(point) != 3:
                raise ValueError("camera lookat must contain three finite numbers")
            view["lookat"] = [_number(x, "camera lookat") for x in point]
            if any(abs(x) > 1e6 for x in view["lookat"]):
                raise ValueError("camera lookat is outside the supported scene range")
        return view

    def _native_size(self, width, height):
        # Exact reduced ratios make 640x360 and 960x540 share one GL context.
        # Rasterize at the largest matching viewport, then resize only the CPU
        # image. Changing JPEG size/quality must not reload the robot meshes.
        divisor = math.gcd(width, height)
        x, y = width // divisor, height // divisor
        scale = min(self.max_width // x, self.max_height // y)
        return x * scale, y * scale

    def _run(self, source_model):
        renderer = None
        try:
            import mujoco
            import numpy as np
            from PIL import Image

            model = copy.copy(source_model)
            source_model = None
            # This is the private copy, so limiting framebuffer allocation cannot
            # change model fingerprints or settings in the physics session.
            model.vis.global_.offwidth = self.max_width
            model.vis.global_.offheight = self.max_height
            data = mujoco.MjData(model)
            options = mujoco.MjvOption()
            options.geomgroup[:] = 1
            # mjlab's official G1 uses group 2 for its 35 visual meshes and group
            # 3 for collision proxies. Keep any actual visual meshes in other
            # groups visible, and always retain group 0 (including the floor).
            meshes = model.geom_type == int(mujoco.mjtGeom.mjGEOM_MESH)
            if np.any(meshes & (model.geom_group == 2)) and not np.any(meshes & (model.geom_group == 3)):
                options.geomgroup[3] = 0
            self._ready.set_result((model.nq, model.nv))
            size = None
            while True:
                request = self._queue.get()
                if request is None:
                    break
                future, state, view, width, height, jpeg_quality = request
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    native_size = self._native_size(width, height)
                    if renderer is None or size != native_size:
                        if renderer is not None:
                            renderer.close()
                            renderer = None
                        renderer = mujoco.Renderer(model, width=native_size[0], height=native_size[1])
                        size = native_size
                    # Only these private data arrays are changed. mj_forward
                    # refreshes geometry and lighting; it never advances time.
                    data.qpos[:] = state["qpos"]
                    data.qvel[:] = state["qvel"]
                    data.time = state["time"]
                    mujoco.mj_forward(model, data)
                    cam = mujoco.MjvCamera()
                    bodies = data.xpos[1:]
                    if len(bodies):
                        low, high = bodies.min(axis=0), bodies.max(axis=0)
                        center = (low + high) / 2
                        distance = max(2.2, float(np.max(high - low)) * 2.4)
                    else:
                        center, distance = np.zeros(3), 2.7
                    cam.lookat[:] = view.get("lookat", center)
                    cam.distance = view.get("distance", distance)
                    cam.azimuth = view.get("azimuth", 130) % 360
                    cam.elevation = view.get("elevation", -15)
                    renderer.update_scene(data, camera=cam, scene_option=options)
                    pixels = renderer.render()
                    picture = Image.fromarray(pixels)
                    if native_size != (width, height):
                        picture = picture.resize((width, height), Image.Resampling.BILINEAR)
                    output = io.BytesIO()
                    picture.save(output, format="JPEG", quality=jpeg_quality)
                    future.set_result({"mime_type": "image/jpeg", "width": width,
                                       "height": height, "data": output.getvalue()})
                except Exception as exc:
                    future.set_exception(RuntimeError(
                        f"MuJoCo offscreen rendering failed (MUJOCO_GL={os.environ.get('MUJOCO_GL', 'default')}): "
                        f"{exc}. Check the EGL graphics driver/device; CPU rendering requires "
                        "libOSMesa and MUJOCO_GL=osmesa before service startup."))
        except Exception as exc:
            if not self._ready.done():
                self._ready.set_exception(RuntimeError(f"MuJoCo renderer initialization failed: {exc}"))
        finally:
            if renderer is not None:
                try:
                    renderer.close()
                except Exception as exc:
                    self._close_error = exc

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is not None and not item[0].done():
                        item[0].set_exception(RuntimeError("MuJoCo renderer is closed"))
                self._queue.put_nowait(None)
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("MuJoCo rendering thread did not stop")
        if self._close_error is not None:
            raise RuntimeError(f"MuJoCo graphics cleanup failed: {self._close_error}")
