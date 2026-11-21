"""Bounded optional RGB evidence; failures never hold the scan admission path."""
from collections import deque
from copy import deepcopy
from contextlib import contextmanager
import threading

import numpy as np

from .camera_geometry import CameraModel, camera_parameter, camera_transform, nearest_depth, project_depth
from .pointcloud_slam_transforms import rigid
from .pointcloud_slam_capture_mount import CaptureMounts, validate_mount_declaration


def validate_rgb_camera(declaration, *, topic):
    try:
        if (not isinstance(declaration, dict) or declaration.get('schema') not in (None, 'motus.camera/1') or
                not isinstance(topic, str) or not topic or
                declaration.get('format') != 'image/jpeg'):
            raise ValueError('RGB requires a connected JPEG topic')
        CameraModel(declaration)
        return dict(topic=topic, camera_info=deepcopy(declaration))
    except (TypeError, AttributeError, OverflowError) as error:
        raise ValueError('invalid optional RGB declaration') from error


def project_measured_depth(measured, camera_from_cloud, K, width, height):
    depth, source_model, minimum, maximum = measured
    if isinstance(source_model, CameraModel):
        points = source_model.backproject(depth, minimum=minimum, maximum=maximum)
        return project_depth(points, camera_from_cloud, K, width, height)
    fx, fy, cx, cy = source_model
    valid = np.isfinite(depth) & (depth > 0.) & (depth >= minimum) & (depth <= maximum)
    depth = np.where(valid, depth, 0.)
    source_K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    if (depth.shape == (height, width) and np.allclose(source_K, K, atol=1e-8, rtol=0.) and
            np.allclose(camera_from_cloud, np.eye(4), atol=1e-8, rtol=0.)):
        return depth
    # Keep separate grid coordinates to avoid packing and copying dense XYZ
    # point lists or competing with native tracking for BLAS worker threads.
    transform = rigid(camera_from_cloud)[:3]
    projection = np.asarray(K, dtype=float).reshape(3, 3) @ transform
    columns = np.arange(depth.shape[1], dtype=np.float32) - cx
    rows = (np.arange(depth.shape[0], dtype=np.float32) - cy)[:, None]
    # Match the float32 backprojection used by the metric cloud before warping.
    x = (columns*depth/fx).astype(np.float32, copy=False).astype(float)
    y = (rows*depth/fy).astype(np.float32, copy=False).astype(float)
    depth = depth.astype(float)

    def component(row):
        return row[0]*x + row[1]*y + row[2]*depth + row[3]

    z = component(transform[2])
    scale = z if np.array_equal(projection[2], transform[2]) else component(projection[2])
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        u = np.rint(component(projection[0])/scale)
        v = np.rint(component(projection[1])/scale)
    z[~valid] = 0.
    return nearest_depth(u, v, z, width, height)


class RgbLoopInput:
    def __init__(self, descriptor, settings, *, max_pending=10, on_decoded=None):
        self._on_decoded = on_decoded
        self.camera, self.mounts = None, None
        self.K, self.width, self.height, self.frame_id = None, None, None, None
        self._condition = threading.Condition()
        self._frames = deque(maxlen=max(1, max_pending))
        self._raw, self._thread, self._closed = None, None, False
        self._generation, self._last_stamp, self._used_stamp = 0, 0, 0
        self._counts = dict(received=0, decoded=0, decode_failures=0, dropped=0, matched=0)
        self._decoding_stamp = None
        self._state = 'disconnected' if descriptor is None else 'fallback'
        self._reason = 'RGB is disconnected' if descriptor is None else 'waiting for fresh RGB'
        try:
            if descriptor is None:
                return
            validated = validate_rgb_camera(descriptor['camera_info'], topic=descriptor['topic'])
            camera = validated['camera_info']; vendor = camera.get('vendor') or {}
            if settings.get('extrinsic_mode') == 'dynamic' and vendor.get('body_mount') is not None:
                mount = validate_mount_declaration(vendor['body_mount'], vendor.get('frame_id', ''))
                self.mounts = CaptureMounts(mount, vendor.get('frame_id', ''), vendor.get('clock_id'))
            self.camera = camera
            self.geometry = CameraModel(camera)
            static_mount = camera_parameter(camera, 'body_from_camera')
            self.static_body_from_color = (None if static_mount is None else
                                           camera_transform(static_mount, 'body_from_camera'))
            baseline = camera_parameter(camera, 'color_from_depth')
            self.depth_from_color = None if baseline is None else np.linalg.inv(camera_transform(baseline))
            self.K, self.width, self.height = self.geometry.rectified_K, self.geometry.width, self.geometry.height
            self.frame_id, self._clock_id = vendor.get('frame_id', ''), vendor.get('clock_id')
        except Exception as error:
            self.camera, self.mounts = None, None
            self._reason = str(error)

    def _fallback(self, reason, *, clear=False):
        self._state, self._reason = 'fallback', str(reason)
        if clear:
            self._generation += 1
            self._raw = None
            self._frames.clear()

    def start(self):
        with self._condition:
            if self.camera is not None and not self._closed and self._thread is None:
                try:
                    self._thread = threading.Thread(target=self._run, daemon=True, name='slam-optional-rgb')
                    self._thread.start()
                except Exception as error:
                    self._thread = None
                    self._fallback(error)

    def close(self):
        with self._condition:
            self._closed = True
            self._generation += 1
            self._raw = None
            self._frames.clear()
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=.05)

    def image(self, msg, context=None, *, now_ns):
        with self._condition:
            if self.camera is None or self._closed:
                return
            self._counts['received'] += 1
            try:
                header_stamp = getattr(getattr(msg, 'header', None), 'stamp', None)
                sec, nano = getattr(header_stamp, 'sec', None), getattr(header_stamp, 'nanosec', None)
                if sec is nano is None:
                    stamp = now_ns
                elif type(sec) is int and type(nano) is int and 0 <= nano < 10**9:
                    stamp = sec*10**9 + nano
                else:
                    raise ValueError('RGB timestamp must be integer nanoseconds')
                if (not isinstance(msg.format, str) or 'jpeg' not in msg.format.lower() or
                        not 0 < len(msg.data) <= 4*1024*1024):
                    raise ValueError('RGB requires JPEG data of at most 4 MiB')
                self._last_stamp = stamp
                if self._raw is not None:
                    self._counts['dropped'] += 1
                self._raw = (stamp, bytes(msg.data), self._generation)
                self._condition.notify()
            except Exception as error:
                self._fallback(error, clear=True)

    def _decode_jpeg(self, data):
        # Read SOF dimensions before OpenCV can allocate an oversized bitmap.
        offset = 2
        dimensions_valid = False
        if data[:2] != b'\xff\xd8':
            raise ValueError('RGB payload is not JPEG')
        while offset+4 <= len(data):
            if data[offset] != 255:
                break
            while offset < len(data) and data[offset] == 255:
                offset += 1
            if offset+3 > len(data):
                break
            marker = data[offset]; offset += 1
            length = int.from_bytes(data[offset:offset+2], 'big')
            if length < 2 or offset+length > len(data):
                break
            if marker in (192, 193, 194, 195, 197, 198, 199, 201, 202, 203, 205, 206, 207):
                if length < 8 or (int.from_bytes(data[offset+3:offset+5], 'big'),
                        int.from_bytes(data[offset+5:offset+7], 'big')) != (self.height, self.width):
                    raise ValueError('JPEG dimensions differ from calibration')
                dimensions_valid = True
                break
            if marker in (218, 217):
                break
            offset += length
        if not dimensions_valid:
            raise ValueError('JPEG dimensions unavailable')
        import cv2
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._raw is not None)
                if self._closed:
                    return
                stamp, data, generation = self._raw
                self._raw = None
                self._decoding_stamp = stamp
            try:
                image = self._decode_jpeg(data)
                if (not isinstance(image, np.ndarray) or image.shape != (self.height, self.width, 3) or
                        image.dtype != np.uint8):
                    raise ValueError('RGB decode failed or dimensions differ from calibration')
                image = self.geometry.rectify_color(image)
                with self._condition:
                    if self._closed or generation != self._generation:
                        continue
                    self._frames.append((stamp, image))
                    self._counts['decoded'] += 1
                    self._state, self._reason = 'ready', None
                    self._condition.notify_all()
                # Never call the owner while holding the decoder condition:
                # gateway capture/publication takes its lock before ours.
                if self._on_decoded is not None:
                    self._on_decoded(stamp)
            except Exception as error:
                with self._condition:
                    if generation == self._generation and not self._closed:
                        self._counts['decode_failures'] += 1
                        # Clear old decoded images, preserving newer queued input for recovery.
                        self._generation += 1
                        if self._raw is not None:
                            stamp, data, _ = self._raw
                            self._raw = (stamp, data, self._generation)
                        self._frames.clear()
                        self._fallback(error)
            finally:
                with self._condition:
                    self._decoding_stamp = None
                    self._condition.notify_all()

    def take(self, cloud_stamp, now_ns, *, exact=False, wait_s=0., max_skew_ns=None):
        with self._condition:
            if wait_s > 0:
                # Only wait for an already received matching JPEG. This is a
                # scheduling budget, never an input timestamp/skew constraint.
                self._condition.wait_for(lambda: self._closed or
                    any(item[0] == cloud_stamp for item in self._frames) or
                    (self._decoding_stamp != cloud_stamp and
                     (self._raw is None or self._raw[0] != cloud_stamp)), timeout=wait_s)
            if self.camera is None or self._closed:
                return None
            candidates = list(self._frames)
            if exact:
                candidates = [item for item in candidates if item[0] == cloud_stamp]
                if not candidates:
                    return None
            if not candidates:
                self._fallback('no decoded RGB available')
                return None
            selected = min(candidates, key=lambda item: abs(item[0]-cloud_stamp))
            if max_skew_ns is not None and abs(selected[0]-cloud_stamp)>max_skew_ns:
                return None
            self._used_stamp = selected[0]
            selected_index = min(range(len(self._frames)), key=lambda i: abs(self._frames[i][0]-cloud_stamp))
            del self._frames[selected_index]
            self._counts['matched'] += 1
            self._state, self._reason = 'ready', None
            return selected[0], selected[1], self._generation

    @contextmanager
    def publication(self, generation):
        """Serialize the final publication check with RGB invalidation."""
        with self._condition:
            yield not self._closed and generation == self._generation

    def mount(self, packet, now_ns):
        with self._condition:
            if self.mounts is not None:
                try:
                    self.mounts.observe(packet, now_ns)
                except Exception as error:
                    self._fallback(error, clear=True)

    def capture_mount(self, stamp_ns, fallback_static_matrix, now_ns):
        with self._condition:
            try:
                if self.camera is None:
                    return None
                if self.mounts is None:
                    if self.static_body_from_color is not None:
                        return self.static_body_from_color.copy()
                    cloud_mount = rigid(fallback_static_matrix)
                    return cloud_mount if self.depth_from_color is None else cloud_mount @ self.depth_from_color
                matrix = self.mounts.matrix_at(stamp_ns)
                if matrix is None:
                    raise ValueError('no numeric mount covers the RGB capture')
                return matrix
            except Exception as error:
                self._fallback(error)
                return None

    def info(self):
        with self._condition:
            return dict(self._counts, state=self._state, reason=self._reason,
                        buffered_frames=len(self._frames), pending_raw=int(self._raw is not None))
