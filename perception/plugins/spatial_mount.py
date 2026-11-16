"""Optional capture-time body-camera evidence; never relabel the main pose."""
from collections import OrderedDict
from copy import deepcopy

import numpy as np

from .pointcloud_slam_transforms import rigid


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
                    now_ns-stamp < -50_000_000):
                raise ValueError('body mount capture clock/time invalid: stamp={}, previous={}, age_ns={}'.format(
                    stamp, self.last_stamp, now_ns-stamp if type(stamp) is int else None))
            status = packet.get('status')
            if status == 'available':
                rigid(packet.get('matrix'))
            elif status != 'unavailable' or 'matrix' in packet:
                raise ValueError('unavailable body mount cannot carry a held matrix')
            self.last_stamp = stamp
            if now_ns-stamp > 500_000_000:
                # A delayed packet is unusable, but does not change source identity.
                # Keep the ordering watermark and allow the next fresh capture.
                self.state, self.reason = 'unavailable', 'body mount capture is stale'
                self.packets.clear()
                return
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
