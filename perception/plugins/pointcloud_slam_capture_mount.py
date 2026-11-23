"""Numeric capture mounts for runtime-discovered pose/cloud inputs."""
from collections import OrderedDict
from copy import deepcopy

from .camera_geometry import camera_parameter, camera_transform
from .rgbd_body_mount import (CaptureMounts, _awaiting_tf, _mount_matrix,
                              _same_mount_source, validate_mount_declaration)
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
                                       paired_sensor_odom=pose.external_origin is not None,
                                       color_from_depth=deepcopy(baseline)))
    return settings, dict(resolution='declared_capture_mount',
        extrinsic_source=deepcopy(mount), clock_relation='producer_declared_common_reference',
        clock_reference=deepcopy(reference), clock_accuracy_validated=False)


def validate_cloud_clock(context, clock_id, stamp_ns, capture_mount=None):
    """Require a numeric cloud timestamp; clock labels remain descriptive."""
    if type(stamp_ns) is not int:
        raise ValueError('cloud timestamp must be integer nanoseconds')


class MappingMounts(CaptureMounts):
    """Measured geometry available to mapping, including during a late TF lookup."""
    previous_measurement = None

    def observe(self, packet, now_ns):
        previous = self.previous_measurement
        self.previous_measurement = None
        if previous and isinstance(packet, dict) and not _same_mount_source((previous, packet)):
            self.packets.clear()
            previous = None
        stamp = super().observe(packet, now_ns)
        current = self.packets[stamp]
        if current.get('matrix') is not None:
            if previous is None or stamp >= previous['stamp_ns']:
                previous = current
            self.previous_measurement = previous
        elif _awaiting_tf(current):
            prior = current.get('previous_measurement')
            if (isinstance(prior, dict) and type(prior.get('stamp_ns')) is int and
                    0 < prior['stamp_ns'] <= stamp and
                    (previous is None or prior['stamp_ns'] > previous['stamp_ns'])):
                previous = dict(current, matrix=_mount_matrix(prior.get('matrix')).tolist(),
                                stamp_ns=prior['stamp_ns'], status='available', reason=None)
                del previous['previous_measurement']
            if previous:
                self.previous_measurement = dict(previous, association='previous_measurement')
        else:
            self.packets.clear()
        return stamp

    def nearest(self, stamp):
        numeric = [p for p in self.packets.values() if p.get('matrix') is not None]
        if self.previous_measurement is not None:
            numeric.append(self.previous_measurement)
        return min(numeric, key=lambda p: abs(p['stamp_ns'] - stamp), default=None)

    def delayed_measurement(self):
        prior = self.previous_measurement
        return prior if prior and prior.get('association') == 'previous_measurement' else None


class CaptureClouds:
    """Bounded raw-cloud waiting for the nearest supplied numeric mount."""
    def __init__(self, config, *, max_pending):
        self.reference = deepcopy(config['clock_reference'])
        self.color_from_depth = config.get('color_from_depth')
        mount = config['declaration']
        self.mounts = MappingMounts(mount, mount['child_frame'], config['source_clock'])
        self.pending = OrderedDict()
        self.max_pending = max_pending

    def validate_pose(self, sample, stream):
        return  # Numeric pose validation belongs to the stream.

    def clear(self):
        self.mounts.previous_measurement = None
        self.pending.clear()
        self.mounts.packets.clear()

    def _cloud_matrix(self, evidence, frame):
        matrix = evidence['matrix']
        if self.color_from_depth is not None and frame != self.mounts.frame:
            matrix = (matrix @ camera_transform(self.color_from_depth)).tolist()
        return matrix

    def matrix_at(self, stamp, frame):
        evidence = self.mounts.nearest(stamp)
        return self._cloud_matrix(evidence, frame) if evidence is not None else None

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
            evidence = self.mounts.nearest(stamp)
            if evidence is None:
                stream.last_error = 'waiting for numeric capture mount'
                continue
            del self.pending[stamp]
            context = dict(child_from_cloud=self._cloud_matrix(evidence, frame), extrinsic=deepcopy(evidence))
            stream.cloud(stamp, frame, cloud, point_count=count, now=now,
                         measurement_now_ns=measurement_now_ns, capture_context=context, received=received)
