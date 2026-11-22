"""Optional numeric body-camera transforms; source conventions belong to providers."""
from collections import OrderedDict
from copy import deepcopy

import numpy as np

from .pointcloud_slam_transforms import rigid
from .rgbd_fusion import quaternion, rotation_from_quaternion
from .rgbd_imu import interpolate_orientation


_MAX_PACKETS = 128
_INTERPOLATION_NS = 200_000_000


def _mount_matrix(value):
    matrix = rigid(value)
    rotation = matrix[:3, :3]
    # Allow ordinary float32/composition roundoff, independent of translation
    # scale or robot geometry. Reflections and non-rigid matrices are unusable
    # by the quaternion and SE(3) operations that consume capture mounts.
    tolerance = 1e-5
    with np.errstate(over='ignore', invalid='ignore'):
        usable = (np.allclose(matrix[3], [0., 0., 0., 1.], rtol=0., atol=tolerance) and
                  np.allclose(rotation.T @ rotation, np.eye(3), rtol=0., atol=tolerance) and
                  np.isclose(np.linalg.det(rotation), 1., rtol=0., atol=tolerance))
    if not usable:
        raise ValueError('body mount must be a homogeneous rigid transform with a proper rotation')
    # Keep the homogeneous row exact so accepted rounding error cannot couple
    # to a large translation and make the full matrix singular.
    matrix[3] = [0., 0., 0., 1.]
    return matrix


def validate_mount_declaration(value, camera_frame, input_topics=()):
    if not isinstance(value,dict) or not isinstance(value.get('topic'),str) or not value['topic']:
        raise ValueError('body_mount must declare a topic')
    result=deepcopy(value)
    result.setdefault('parent_frame','body')
    result.setdefault('child_frame',camera_frame)
    return result


def _awaiting_tf(packet):
    reason = packet.get('reason')
    return (packet.get('matrix') is None and packet.get('status') == 'unavailable' and
            isinstance(reason, str) and 'extrapolation into the future' in reason)


def _same_mount_source(packets):
    return all(len({packet[key] for packet in packets if isinstance(packet.get(key), str)}) <= 1
               for key in ('epoch', 'clock_id', 'parent_frame', 'child_frame', 'source', 'version', 'mode'))


class CaptureMounts:
    """Bounded numeric transforms associated by nearest supplied timestamp."""
    def __init__(self,declaration,camera_frame,clock):
        self.declaration=validate_mount_declaration(declaration,camera_frame)
        self.frame,self.clock=camera_frame,clock
        self.packets=OrderedDict()
        self._retained={}
        self._missing={}
        self.last_stamp=0
        self.state,self.reason='unavailable','waiting for a numeric body-camera transform'

    def fault(self,reason):
        self.state,self.reason='lost',str(reason)
        self.packets.clear()
        self._retained = {stamp: {} for stamp in self._retained}
        self._missing.clear()

    def retain(self, stamps, *, replace=True):
        """Pin capture dependencies until their owner's mutable history expires."""
        retained = {} if replace else self._retained
        for stamp in stamps:
            evidence = self._retained.get(stamp)
            if evidence is None:
                evidence = {value: item for value, item in self.packets.items()
                            if abs(value-stamp) <= _INTERPOLATION_NS}
            retained[stamp] = evidence
        self._retained = retained
        if replace:
            self._missing = {stamp: bounds for stamp, bounds in self._missing.items() if stamp in retained}

    def observe(self,packet,now_ns):
        try:
            if not isinstance(packet,dict) or type(packet.get('stamp_ns', now_ns)) is not int:
                raise ValueError('body mount must contain an integer timestamp')
            packet = dict(packet, stamp_ns=packet.get('stamp_ns', now_ns))
            stamp=packet['stamp_ns']
            if packet.get('matrix') is not None:
                packet['matrix'] = _mount_matrix(packet['matrix']).tolist()
            previous = self.packets.get(stamp)
            if previous is None:
                previous = next((evidence[stamp] for evidence in self._retained.values()
                                 if stamp in evidence), None)
            if (previous is not None and previous.get('matrix') is not None and
                    _awaiting_tf(packet) and _same_mount_source((previous, packet))):
                packet = previous  # A delayed lookup diagnostic cannot undo exact evidence.
            self.last_stamp=stamp
            self.state='available' if packet.get('matrix') is not None else 'unavailable'
            self.reason=packet.get('reason')
            self.packets[stamp]=deepcopy(packet)
            for capture, evidence in self._retained.items():
                if abs(capture-stamp) <= _INTERPOLATION_NS:
                    # Preserve new boundaries before a later event in this same
                    # batch can evict them from the ordinary arrival cache.
                    evidence[stamp] = self.packets[stamp]
                    self._bound_evidence(capture, evidence)
            self.packets.move_to_end(stamp)
            while len(self.packets)>_MAX_PACKETS:self.packets.popitem(last=False)
            return stamp
        except (ValueError,TypeError,KeyError,np.linalg.LinAlgError) as error:
            self.fault(error)
            raise ValueError(str(error)) from error

    def info(self,now_ns):
        return dict(status=self.state,reason=self.reason,topic=self.declaration['topic'],
                    parent_frame=self.declaration['parent_frame'],child_frame=self.frame,
                    last_measurement_ns=self.last_stamp,buffered_samples=len(self.packets),
                    accuracy_validated=False)

    def _bound_evidence(self, stamp, packets):
        if len(packets) <= _MAX_PACKETS:
            return
        before = max((value for value, item in packets.items()
                      if value < stamp and not _awaiting_tf(item)), default=None)
        after = min((value for value, item in packets.items()
                     if value > stamp and not _awaiting_tf(item)), default=None)
        essential = {value for value in (before, stamp, after) if value in packets}
        nearby = sorted(packets.keys()-essential, key=lambda value: (abs(value-stamp), value))
        discarded = nearby[_MAX_PACKETS-len(essential):]
        missing_before, missing_after = self._missing.get(stamp, (None, None))
        before = [value for value in discarded if value < stamp]
        after = [value for value in discarded if value > stamp]
        if missing_before is not None:
            before.append(missing_before)
        if missing_after is not None:
            after.append(missing_after)
        # Only the nearest lost timestamp on either side can bound a bracket
        # around this capture. A fully witnessed narrower bracket stays usable.
        self._missing[stamp] = max(before, default=None), min(after, default=None)
        for value in discarded:
            del packets[value]

    def matrix_at(self, stamp):
        """Return an exact transform or interpolate across a short numeric bracket."""
        packets = dict(self._retained.get(stamp, {}))
        packets.update((value, item) for value, item in self.packets.items()
                       if abs(value-stamp) <= _INTERPOLATION_NS)
        transform, dependencies = self._resolve(stamp, packets)
        if stamp in self._retained:
            # A nearest endpoint can later become a transient miss, widening
            # the bracket. Preserve evidence throughout the possible radius,
            # including boundaries outside today's exact/narrow resolution.
            self._bound_evidence(stamp, packets)
            missing = self._missing.get(stamp)
            if missing is not None:
                exact = dependencies.get(stamp)
                endpoints = [value for value, item in dependencies.items() if not _awaiting_tf(item)]
                complete = exact is not None and not _awaiting_tf(exact)
                witnessed = (len(endpoints) == 2 and
                             (missing[0] is None or missing[0] <= min(endpoints)) and
                             (missing[1] is None or missing[1] >= max(endpoints)))
                if not complete and not witnessed:
                    transform = None
                # Exact or narrower evidence can resolve the capture now, but
                # does not restore lost packets if its bracket expands later.
            self._retained[stamp] = packets
        return transform

    @staticmethod
    def _resolve(stamp, packets):
        packet = packets.get(stamp)
        if packet is not None and not _awaiting_tf(packet):
            transform = rigid(packet['matrix']) if packet.get('matrix') is not None else None
            return transform, {stamp: packet}
        # A premature TF query has no matrix yet. Only numeric endpoints within
        # the same short bracket can cover it. Other failures remain boundaries.
        window_ns = _INTERPOLATION_NS
        nearby = {value: item for value, item in packets.items() if abs(value-stamp) <= window_ns}
        before = max((value for value, item in nearby.items()
                      if value < stamp and not _awaiting_tf(item)), default=None)
        after = min((value for value, item in nearby.items()
                     if value > stamp and not _awaiting_tf(item)), default=None)
        left = stamp-window_ns if before is None else before
        right = stamp+window_ns if after is None else after
        dependencies = {value: item for value, item in nearby.items() if left <= value <= right}
        if before is None or after is None or after-before > window_ns:
            return None, dependencies
        endpoints = [nearby[value].get('matrix') for value in (before, after)]
        bracket = [item for value, item in nearby.items() if before <= value <= after]
        if any(item is None for item in endpoints) or not _same_mount_source(bracket):
            return None, dependencies
        a, b = map(rigid, endpoints)
        q = interpolate_orientation([(before, np.asarray(quaternion(a[:3, :3]))),
                                     (after, np.asarray(quaternion(b[:3, :3])))], stamp)
        transform = np.eye(4)
        transform[:3, :3] = rotation_from_quaternion(q)
        weight = (stamp-before)/(after-before)
        transform[:3, 3] = a[:3, 3]+weight*(b[:3, 3]-a[:3, 3])
        return transform, dependencies

    def attach(self,sample,now_ns):
        vendor=sample['vendor']
        vendor['body_pose']=None
        stamp=vendor.get('motus_slam',{}).get('stamp_ns')
        available=[(t,p) for t,p in self.packets.items() if p.get('matrix') is not None]
        evidence=(min(available,key=lambda item:abs(item[0]-stamp))[1]
                  if type(stamp) is int and available else None)
        diagnostic=self.info(now_ns)
        if evidence is None:
            diagnostic.update(status='unavailable',reason=self.reason or 'no numeric body-camera transform')
        else:
            diagnostic.update(status='available',reason=evidence.get('reason'),packet=deepcopy(evidence))
        vendor['camera_mount']=diagnostic
        pose=sample.get('pose')
        if evidence is None or not isinstance(pose,dict):return sample
        position=np.asarray(pose.get('position'),dtype=float)
        q=np.asarray(pose.get('quaternion'),dtype=float)
        if (position.shape!=(3,) or q.shape!=(4,) or not np.isfinite(position).all() or
                not np.isfinite(q).all() or np.linalg.norm(q)==0):
            raise ValueError('body pose requires finite position and nonzero quaternion')
        rotation=rotation_from_quaternion(q/np.linalg.norm(q))
        transform=rigid(evidence['matrix'])
        body_rotation=rotation@transform[:3,:3].T
        body_position=position-body_rotation@transform[:3,3]
        vendor['body_pose']=dict(position=body_position.tolist(),quaternion=quaternion(body_rotation),
            frame_id=pose.get('frame_id',''),child_frame_id=evidence.get('parent_frame',self.declaration['parent_frame']),
            stamp_ns=stamp,pose_reference='body',covariance=None,accuracy_validated=False,
            covariance_reason='body-camera extrinsic uncertainty is not supplied')
        return sample
