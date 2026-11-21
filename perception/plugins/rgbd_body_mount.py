"""Optional numeric body-camera transforms; source conventions belong to providers."""
from collections import OrderedDict
from copy import deepcopy

import numpy as np

from .pointcloud_slam_transforms import rigid
from .rgbd_fusion import quaternion, rotation_from_quaternion
from .rgbd_imu import interpolate_orientation


def validate_mount_declaration(value, camera_frame, input_topics=()):
    if not isinstance(value,dict) or not isinstance(value.get('topic'),str) or not value['topic']:
        raise ValueError('body_mount must declare a topic')
    result=deepcopy(value)
    result.setdefault('parent_frame','body')
    result.setdefault('child_frame',camera_frame)
    return result


class CaptureMounts:
    """Bounded numeric transforms associated by nearest supplied timestamp."""
    def __init__(self,declaration,camera_frame,clock):
        self.declaration=validate_mount_declaration(declaration,camera_frame)
        self.frame,self.clock=camera_frame,clock
        self.packets=OrderedDict()
        self.last_stamp=0
        self.state,self.reason='unavailable','waiting for a numeric body-camera transform'

    def fault(self,reason):
        self.state,self.reason='lost',str(reason)
        self.packets.clear()

    def observe(self,packet,now_ns):
        try:
            if not isinstance(packet,dict) or type(packet.get('stamp_ns', now_ns)) is not int:
                raise ValueError('body mount must contain an integer timestamp')
            packet = dict(packet, stamp_ns=packet.get('stamp_ns', now_ns))
            stamp=packet['stamp_ns']
            if packet.get('matrix') is not None:
                rigid(packet['matrix'])
            self.last_stamp=stamp
            self.state='available' if packet.get('matrix') is not None else 'unavailable'
            self.reason=packet.get('reason')
            self.packets[stamp]=deepcopy(packet)
            self.packets.move_to_end(stamp)
            while len(self.packets)>128:self.packets.popitem(last=False)
        except (ValueError,TypeError,KeyError,np.linalg.LinAlgError) as error:
            self.fault(error)
            raise ValueError(str(error)) from error

    def info(self,now_ns):
        return dict(status=self.state,reason=self.reason,topic=self.declaration['topic'],
                    parent_frame=self.declaration['parent_frame'],child_frame=self.frame,
                    last_measurement_ns=self.last_stamp,buffered_samples=len(self.packets),
                    accuracy_validated=False)

    def matrix_at(self, stamp):
        """Return an exact transform or interpolate across a short numeric bracket."""
        packet = self.packets.get(stamp)
        if packet is not None:
            return rigid(packet['matrix']) if packet.get('matrix') is not None else None
        before = max((value for value in self.packets if value < stamp), default=None)
        after = min((value for value in self.packets if value > stamp), default=None)
        if before is None or after is None or after-before > 200_000_000:
            return None
        endpoints = [self.packets[value].get('matrix') for value in (before, after)]
        if any(item is None for item in endpoints):
            return None
        a, b = map(rigid, endpoints)
        q = interpolate_orientation([(before, np.asarray(quaternion(a[:3, :3]))),
                                     (after, np.asarray(quaternion(b[:3, :3])))], stamp)
        transform = np.eye(4)
        transform[:3, :3] = rotation_from_quaternion(q)
        weight = (stamp-before)/(after-before)
        transform[:3, 3] = a[:3, 3]+weight*(b[:3, 3]-a[:3, 3])
        return transform

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
