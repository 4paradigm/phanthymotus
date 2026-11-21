"""Numeric capture mounts for runtime-discovered pose/cloud inputs."""
from collections import OrderedDict
from copy import deepcopy

from .camera_geometry import camera_parameter, camera_transform
from .rgbd_body_mount import CaptureMounts, validate_mount_declaration
from .pointcloud_slam_odom import decode_slam_odom


def resolve_capture_mount(sample, camera, cloud_frame):
    pose = decode_slam_odom(sample)
    if not isinstance(camera, dict) or camera.get('schema') != 'motus.camera/1':
        raise ValueError('connected cloud camera declaration is required')
    vendor = camera.get('vendor', {})
    clock = vendor.get('clock_id')
    reference = vendor.get('clock_reference') or {}
    mount = validate_mount_declaration(vendor.get('body_mount'), cloud_frame)
    baseline = camera_parameter(camera, 'color_from_depth')
    settings = dict(cloud_frame=cloud_frame, child_frame=pose.child_frame_id,
                    clock_id=pose.clock_id, child_from_cloud=None, extrinsic_mode='dynamic',
                    capture_mount=dict(declaration=mount, source_clock=clock,
                                       clock_reference=deepcopy(reference),
                                       interpolate=pose.external_origin is not None,
                                       color_from_depth=deepcopy(baseline)))
    return settings, dict(resolution='declared_capture_mount',
        extrinsic_source=deepcopy(mount), clock_relation='producer_declared_common_reference',
        clock_reference=deepcopy(reference), clock_accuracy_validated=False)


def validate_cloud_clock(context, clock_id, stamp_ns, capture_mount=None):
    """Require a numeric cloud timestamp; clock labels remain descriptive."""
    if type(stamp_ns) is not int:
        raise ValueError('cloud timestamp must be integer nanoseconds')


class CaptureClouds:
    """Bounded raw-cloud waiting for the nearest supplied numeric mount."""
    def __init__(self, config, *, max_pending):
        self.reference = deepcopy(config['clock_reference'])
        self.interpolate = config.get('interpolate', False)
        self.color_from_depth = config.get('color_from_depth')
        mount = config['declaration']
        self.mounts = CaptureMounts(mount, mount['child_frame'], config['source_clock'])
        self.pending = OrderedDict()
        self.max_pending = max_pending

    def validate_pose(self, sample, stream):
        return  # Numeric pose validation belongs to the stream.

    def clear(self):
        self.pending.clear()
        self.mounts.packets.clear()

    def matrix_at(self, stamp, frame):
        matrix = self.mounts.matrix_at(stamp)
        if matrix is not None and self.color_from_depth is not None and frame != self.mounts.frame:
            matrix = matrix @ camera_transform(self.color_from_depth)
        return matrix

    def mount(self, packet, stream, *, now, measurement_now_ns):
        self.mounts.observe(packet, measurement_now_ns)
        self._drain(stream, now, measurement_now_ns)

    def cloud(self, stamp, frame, cloud, count, stream, *, now, measurement_now_ns, received=None):
        stream._check_measurement_time(stamp, measurement_now_ns)
        if type(stamp) is not int:
            raise ValueError('integer cloud capture timestamp required')
        if frame == stream._external_body_frame:
            stream.cloud(stamp, frame, cloud, point_count=count, now=now, received=received)
            return
        self.pending.setdefault(stamp, (frame, deepcopy(cloud), count, now if received is None else received))
        while len(self.pending) > self.max_pending:
            self.pending.popitem(last=False)
        self._drain(stream, now, measurement_now_ns)

    def _drain(self, stream, now, measurement_now_ns):
        for stamp in sorted(self.pending):
            frame, cloud, count, received = self.pending[stamp]
            packets = [p for p in self.mounts.packets.values() if p.get('matrix') is not None]
            if not packets:
                stream.last_error = 'waiting for numeric capture mount'
                continue
            evidence = min(packets, key=lambda p: abs(p['stamp_ns']-stamp))
            if self.interpolate:
                matrix = self.matrix_at(stamp, frame)
                if matrix is None:
                    stream.last_error = 'waiting for numeric capture mount at cloud timestamp'
                    continue
                evidence = dict(evidence, stamp_ns=stamp, matrix=matrix.tolist())
            del self.pending[stamp]
            context = dict(child_from_cloud=evidence['matrix'], extrinsic=deepcopy(evidence))
            stream.cloud(stamp, frame, cloud, point_count=count, now=now,
                         measurement_now_ns=measurement_now_ns, capture_context=context, received=received)
