"""Bounded optional RGB evidence; failures never hold the scan admission path."""
from collections import deque
from copy import deepcopy
from contextlib import contextmanager
import threading

import numpy as np

from .pointcloud_slam_transforms import rigid
from .pointcloud_slam_capture_mount import CaptureMounts, validate_mount_declaration


def validate_rgb_camera(declaration, *, topic):
    try:
        if (not isinstance(declaration, dict) or declaration.get('schema') != 'motus.camera/1' or
                not isinstance(topic, str) or not topic or declaration.get('topic') != topic or
                declaration.get('format') != 'image/jpeg' or
                any(type(declaration.get(k)) is not int or not 0 < declaration[k] <= 4096
                    for k in ('width', 'height')) or declaration.get('distortion_model') != 'pinhole'):
            raise ValueError('RGB requires the exact JPEG topic and declared pinhole dimensions')
        k = np.asarray(declaration.get('K'), dtype=float)
        d = np.asarray(declaration.get('D'), dtype=float)
        if (k.size != 9 or not np.isfinite(k).all() or k.flat[0] <= 0 or k.flat[4] <= 0 or
                not np.allclose(k.reshape(3, 3)[2], [0, 0, 1]) or k.flat[1] != 0 or k.flat[3] != 0 or
                not d.size or not np.isfinite(d).all() or np.any(d != 0)):
            raise ValueError('RGB requires finite pinhole intrinsics and zero distortion')
        vendor = declaration.get('vendor') or {}
        if (any(not isinstance(vendor.get(k), str) or not vendor[k].strip() or vendor[k] == 'unknown'
                for k in ('frame_id', 'clock_id')) or
                vendor.get('timestamp_semantics') != 'source_capture_mapped' or
                vendor.get('registration') != 'registered_rectified'):
            raise ValueError('RGB requires registered rectified geometry and source capture time')
        return dict(topic=topic, camera_info=deepcopy(declaration))
    except (TypeError, AttributeError, OverflowError) as error:
        raise ValueError('invalid optional RGB declaration') from error


def project_depth(points, camera_from_cloud, K, width, height):
    """Nearest optical Z of measured points; holes remain zero metres."""
    transform = rigid(camera_from_cloud)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    moved = points @ transform[:3, :3].T + transform[:3, 3]
    moved = moved[np.isfinite(moved).all(axis=1) & (moved[:, 2] > 0)]
    k = np.asarray(K, dtype=float).reshape(3, 3)
    pixels = moved @ k.T
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        uv = np.rint(pixels[:, :2]/pixels[:, 2:3])
    valid = (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < width) &
             (uv[:, 1] >= 0) & (uv[:, 1] < height) & (moved[:, 2] <= np.finfo(np.float32).max))
    uv = uv[valid].astype(np.int64)
    depth = np.full(width*height, np.inf, dtype=np.float32)
    np.minimum.at(depth, uv[:, 1]*width+uv[:, 0], moved[valid, 2].astype(np.float32))
    depth[~np.isfinite(depth)] = 0
    return depth.reshape(height, width)


class RgbLoopInput:
    def __init__(self, descriptor, settings, *, max_pending=10):
        self.camera, self.mounts = None, None
        self.K, self.width, self.height, self.frame_id = None, None, None, None
        self._condition = threading.Condition()
        self._frames = deque(maxlen=max(1, max_pending))
        self._raw, self._thread, self._closed = None, None, False
        self._generation, self._last_stamp, self._used_stamp = 0, 0, 0
        self._counts = dict(received=0, decoded=0, decode_failures=0, dropped=0, matched=0)
        self._state = 'disconnected' if descriptor is None else 'fallback'
        self._reason = 'RGB is disconnected' if descriptor is None else 'waiting for fresh RGB'
        try:
            if descriptor is None:
                return
            validated = validate_rgb_camera(descriptor['camera_info'], topic=descriptor['topic'])
            camera = validated['camera_info']; vendor = camera['vendor']
            if vendor['frame_id'] != settings['cloud_frame']:
                raise ValueError('RGB and cloud must share the declared registered optical frame')
            capture = settings.get('capture_mount')
            if vendor['clock_id'] != settings['clock_id']:
                reference = vendor.get('clock_reference')
                if (not capture or vendor['clock_id'] != capture['source_clock'] or
                        not isinstance(reference, dict) or
                        any(reference.get(k) != capture['clock_reference'].get(k) or
                            not isinstance(reference.get(k), str) or not reference[k].strip()
                            for k in ('clock_id', 'domain_id', 'timestamp_basis')) or
                        reference['clock_id'] != settings['clock_id'] or
                        reference['timestamp_basis'] != 'host_realtime'):
                    raise ValueError('RGB and odometry have no declared common capture clock')
            if settings.get('extrinsic_mode') == 'dynamic':
                mount = validate_mount_declaration(vendor.get('body_mount'), vendor['frame_id'])
                if not capture or mount != capture['declaration']:
                    raise ValueError('RGB requires the matching declared capture mount')
                self.mounts = CaptureMounts(mount, vendor['frame_id'], vendor['clock_id'])
            self.camera = camera
            self.K = np.asarray(camera['K'], dtype=float).reshape(3, 3)
            self.width, self.height = camera['width'], camera['height']
            self.frame_id, self._clock_id = vendor['frame_id'], vendor['clock_id']
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
                stamp = msg.header.stamp.sec*10**9 + msg.header.stamp.nanosec
                if (type(stamp) is not int or not 0 <= msg.header.stamp.nanosec < 10**9 or
                        msg.header.stamp.sec < 0 or stamp <= self._last_stamp or
                        not -50_000_000 <= now_ns-stamp <= 500_000_000):
                    raise ValueError('RGB capture is stale, repeated, reset or future')
                if msg.header.frame_id != self.frame_id:
                    raise ValueError('RGB optical frame changed')
                if context is not None and (context.time_basis != 'source_capture' or
                        context.normalized_clock_id != self._clock_id or context.observation_stamp_ns != stamp):
                    raise ValueError('RGB context must preserve the declared source capture clock and stamp')
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
            try:
                image = self._decode_jpeg(data)
                if (not isinstance(image, np.ndarray) or image.shape != (self.height, self.width, 3) or
                        image.dtype != np.uint8):
                    raise ValueError('RGB decode failed or dimensions differ from calibration')
                with self._condition:
                    if self._closed or generation != self._generation:
                        continue
                    self._frames.append((stamp, image))
                    self._counts['decoded'] += 1
                    self._state, self._reason = 'ready', None
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

    def take(self, cloud_stamp, now_ns):
        with self._condition:
            if self.camera is None or self._closed:
                return None
            self._frames = deque((item for item in self._frames if item[0] > self._used_stamp and
                -50_000_000 <= now_ns-item[0] <= 500_000_000), maxlen=self._frames.maxlen)
            candidates = [item for item in self._frames if abs(item[0]-cloud_stamp) <= 50_000_000]
            if not candidates:
                self._fallback('no fresh decoded RGB within 50 ms of cloud capture')
                return None
            selected = min(candidates, key=lambda item: abs(item[0]-cloud_stamp))
            self._used_stamp = selected[0]
            self._frames = deque((item for item in self._frames if item[0] > self._used_stamp),
                                 maxlen=self._frames.maxlen)
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
                    return rigid(fallback_static_matrix)
                evidence = self.mounts.packets.get(stamp_ns)
                if (self.mounts.state == 'lost' or not evidence or evidence['status'] != 'available' or
                        not -50_000_000 <= now_ns-stamp_ns <= 500_000_000):
                    raise ValueError('no fresh available mount for exact RGB capture timestamp')
                return rigid(evidence['matrix'])
            except Exception as error:
                self._fallback(error)
                return None

    def info(self):
        with self._condition:
            return dict(self._counts, state=self._state, reason=self._reason,
                        buffered_frames=len(self._frames), pending_raw=int(self._raw is not None))
