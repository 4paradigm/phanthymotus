"""Optional capture-time body-camera evidence; never relabel the main pose."""
from collections import OrderedDict
from copy import deepcopy

import numpy as np

from .pointcloud_slam_transforms import rigid
from .rgbd_fusion import pose_values, quaternion


_IDENTITY = ('schema', 'parent_frame', 'child_frame', 'source', 'version', 'epoch',
             'mode', 'length_unit', 'transform_direction')


def validate_mount_declaration(value, camera_frame, input_topics=()):
    if not isinstance(value, dict):
        raise ValueError('body_mount must be a source declaration')
    if (value.get('schema') != 'motus.camera_mount/1' or value.get('source') != 'tf' or
            value.get('mode') != 'dynamic' or value.get('length_unit') != 'm' or
            value.get('transform_direction') != 'body_from_camera' or
            value.get('child_frame') != camera_frame):
        raise ValueError('body_mount schema, source, direction, unit or optical frame mismatch')
    for name in ('topic', 'parent_frame', 'version', 'epoch'):
        if not isinstance(value.get(name), str) or not value[name].strip() or value[name] == 'unknown':
            raise ValueError('body_mount '+name+' is required')
    if value['parent_frame'] == camera_frame or value['topic'] in input_topics:
        raise ValueError('body_mount must name a distinct body frame and metadata topic')
    return deepcopy(value)


class CaptureMounts:
    """Bounded exact-nanosecond joins. A bad identity retires only mount evidence."""
    def __init__(self, declaration, camera_frame, clock):
        self.declaration = validate_mount_declaration(declaration, camera_frame)
        self.frame, self.clock = camera_frame, clock
        self.packets = OrderedDict()
        self.last_stamp = 0
        self.state, self.reason = 'unavailable', 'waiting for capture-time body-camera TF'

    def fault(self, reason):
        self.state, self.reason = 'lost', str(reason)
        self.packets.clear()

    def observe(self, packet, now_ns):
        if self.state == 'lost':
            return
        try:
            if not isinstance(packet, dict) or any(packet.get(k) != self.declaration[k] for k in _IDENTITY):
                raise ValueError('body mount identity, epoch or frame changed; restart card to adopt it')
            stamp = packet.get('stamp_ns')
            if (packet.get('clock_id') != self.clock or packet.get('stamp_source') != 'measurement' or
                    type(stamp) is not int or stamp <= self.last_stamp or
                    not -50_000_000 <= now_ns-stamp <= 500_000_000):
                raise ValueError('body mount capture clock/time is invalid, stale, reset or repeated')
            status = packet.get('status')
            if status == 'available':
                rigid(packet.get('matrix'))
            elif status != 'unavailable' or 'matrix' in packet:
                raise ValueError('unavailable body mount cannot carry a held matrix')
            self.last_stamp = stamp
            self.state, self.reason = status, packet.get('reason')
            self.packets[stamp] = deepcopy(packet)
            while len(self.packets) > 128:
                self.packets.popitem(last=False)
        except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as error:
            self.state, self.reason = 'lost', str(error)
            self.packets.clear()
            raise ValueError(str(error)) from error

    def info(self, now_ns):
        status, reason = self.state, self.reason
        if status != 'lost' and self.last_stamp and now_ns-self.last_stamp > 500_000_000:
            status, reason = 'unavailable', 'body mount capture is stale'
        return dict(status=status, reason=reason, topic=self.declaration['topic'],
                    parent_frame=self.declaration['parent_frame'], child_frame=self.frame,
                    last_measurement_ns=self.last_stamp, buffered_samples=len(self.packets),
                    accuracy_validated=False)

    def attach(self, sample, now_ns):
        vendor = sample['vendor']
        vendor['body_pose'] = None
        stamp = vendor.get('motus_slam', {}).get('stamp_ns')
        evidence = self.packets.get(stamp) if type(stamp) is int else None
        diagnostic = self.info(now_ns)
        if self.state == 'lost':
            evidence = None
        elif evidence is None:
            diagnostic.update(status='unavailable', reason='no body mount for exact image capture timestamp')
        elif not -50_000_000 <= now_ns-stamp <= 500_000_000:
            diagnostic.update(status='unavailable', reason='matching body mount is stale')
            evidence = None
        else:
            diagnostic.update(status=evidence['status'], reason=evidence.get('reason'), packet=deepcopy(evidence))
        vendor['camera_mount'] = diagnostic
        if evidence is None or diagnostic['status'] != 'available':
            return sample
        quality = vendor.get('motus_slam', {})
        pose = sample.get('pose')
        if quality.get('status') != 'tracking' or not isinstance(pose, dict):
            return sample
        if quality.get('clock_id') != self.clock or pose.get('child_frame_id') != self.frame:
            diagnostic.update(status='unavailable', reason='camera pose frame or clock differs from mount')
            return sample
        position, rotation = pose_values(pose)
        transform = rigid(evidence['matrix'])
        body_rotation = rotation@transform[:3, :3].T
        body_position = position-body_rotation@transform[:3, 3]
        vendor['body_pose'] = dict(position=body_position.tolist(), quaternion=quaternion(body_rotation),
                                  frame_id=pose['frame_id'], child_frame_id=self.declaration['parent_frame'],
                                  stamp_ns=stamp, pose_reference='body', covariance=None,
                                  accuracy_validated=False,
                                  covariance_reason='body-camera extrinsic uncertainty is not supplied')
        return sample
