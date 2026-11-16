"""Exact capture-time mounts for runtime-discovered pose/cloud inputs.

Clock reference declarations identify an existing normalized timeline; they do
not certify camera clock accuracy or manufacture a strict SLAM input contract.
"""
from collections import OrderedDict
from copy import deepcopy

from .rgbd_body_mount import CaptureMounts, validate_mount_declaration
from .pointcloud_slam_odom import decode_slam_odom


def resolve_capture_mount(sample, camera, cloud_frame):
    pose = decode_slam_odom(sample)
    if not isinstance(camera, dict) or camera.get('schema') != 'motus.camera/1':
        raise ValueError('connected cloud camera declaration is required')
    vendor = camera.get('vendor', {})
    if vendor.get('frame_id') != cloud_frame:
        raise ValueError('cloud frame differs from camera declaration')
    clock = vendor.get('clock_id')
    if not isinstance(clock, str) or not clock.strip() or clock == 'unknown':
        raise ValueError('capture mount source clock is required')
    reference = vendor.get('clock_reference')
    other = sample['vendor']['motus_slam'].get('clock_reference')
    keys = ('clock_id', 'domain_id', 'timestamp_basis')
    if (not isinstance(reference, dict) or not isinstance(other, dict) or
            any(not isinstance(reference.get(k), str) or not reference[k].strip()
                or reference[k] == 'unknown' or reference[k] != other.get(k) for k in keys) or
            reference['clock_id'] != pose.clock_id or
            reference['timestamp_basis'] != 'host_realtime'):
        raise ValueError('dynamic mount requires the same declared host clock reference')
    mount = validate_mount_declaration(vendor.get('body_mount'), cloud_frame)
    if mount['parent_frame'] != pose.child_frame_id:
        raise ValueError('capture mount parent differs from odometry child frame')
    settings = dict(cloud_frame=cloud_frame, child_frame=pose.child_frame_id,
                    clock_id=pose.clock_id, child_from_cloud=None, extrinsic_mode='dynamic',
                    capture_mount=dict(declaration=mount, source_clock=clock,
                                       clock_reference=deepcopy(reference)))
    return settings, dict(resolution='declared_capture_mount',
        extrinsic_source=deepcopy(mount), clock_relation='producer_declared_common_reference',
        clock_reference=deepcopy(reference), clock_accuracy_validated=False)


def validate_cloud_clock(context, clock_id, stamp_ns, capture_mount=None):
    """Use only a previously resolved common reference for different clock names."""
    if context is None:
        return
    if context.time_basis == 'source_capture' and context.observation_stamp_ns != stamp_ns:
        raise ValueError('cloud observation differs from capture timestamp')
    if context.normalized_clock_id == clock_id:
        return
    if (capture_mount is None or context.time_basis != 'source_capture' or
            context.normalized_clock_id != capture_mount['source_clock'] or
            capture_mount['clock_reference']['clock_id'] != clock_id):
        raise ValueError('cloud and pose have no comparable observation clock')


class CaptureClouds:
    """Bounded raw-cloud waiting; never substitutes latest/nearest/static TF."""
    def __init__(self, config, *, max_pending):
        self.reference = deepcopy(config['clock_reference'])
        mount = config['declaration']
        self.mounts = CaptureMounts(mount, mount['child_frame'], config['source_clock'])
        self.pending = OrderedDict()
        self.max_pending = max_pending

    def validate_pose(self, sample, stream):
        vendor = sample.get('vendor') if isinstance(sample, dict) else None
        quality = vendor.get('motus_slam') if isinstance(vendor, dict) else None
        if not isinstance(quality, dict) or quality.get('status') != 'tracking':
            return  # The stream handles unavailable/invalid tracking states.
        ref = quality.get('clock_reference')
        if not isinstance(ref, dict) or any(ref.get(k) != self.reference[k]
                for k in ('clock_id', 'domain_id', 'timestamp_basis')):
            self.mounts.fault('pose clock reference changed; restart inputs')
            self.clear()
            stream._lose()
            raise ValueError('pose clock reference changed; restart inputs')

    def clear(self):
        self.pending.clear()
        self.mounts.packets.clear()

    def mount(self, packet, stream, *, now, measurement_now_ns):
        try:
            self.mounts.observe(packet, measurement_now_ns)
        except ValueError:
            self.clear()
            stream._lose()
            raise
        self._drain(stream, now, measurement_now_ns)

    def cloud(self, stamp, frame, cloud, count, stream, *, now, measurement_now_ns, received=None):
        if frame != self.mounts.frame:
            raise ValueError('cloud frame differs from capture mount')
        stream._check_measurement_time(stamp, measurement_now_ns)
        if type(stamp) is not int or stamp <= 0:
            raise ValueError('positive cloud capture timestamp required')
        self.pending.setdefault(stamp, (frame, deepcopy(cloud), count, now if received is None else received))
        while len(self.pending) > self.max_pending:
            self.pending.popitem(last=False)
        self._drain(stream, now, measurement_now_ns)

    def _drain(self, stream, now, measurement_now_ns):
        if self.mounts.state == 'lost':
            self.clear()
            stream.last_error = 'capture mount identity invalid; restart inputs'
            return
        for stamp in sorted(self.pending):
            frame, cloud, count, received = self.pending[stamp]
            if (now-received > stream.max_age_s or
                    not -50_000_000 <= measurement_now_ns-stamp <= 500_000_000):
                del self.pending[stamp]
                self.mounts.packets.pop(stamp, None)
                stream.last_error = 'cloud expired while waiting for capture mount'
                continue
            evidence = self.mounts.packets.get(stamp)
            if evidence is None:
                stream.last_error = 'waiting for exact capture mount'
                continue
            del self.pending[stamp]
            self.mounts.packets.pop(stamp)
            if evidence['status'] != 'available':
                stream.last_error = 'capture mount unavailable for cloud'
                continue
            context = dict(child_from_cloud=evidence['matrix'], extrinsic=deepcopy(evidence))
            stream.cloud(stamp, frame, cloud, point_count=count, now=now,
                         measurement_now_ns=measurement_now_ns, capture_context=context, received=received)
