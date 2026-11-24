from copy import deepcopy
import numpy as np


def declaration(**changes):
    d = dict(schema='motus.camera_mount/1', topic='/camera/rgbd/body_mount',
             parent_frame='base_link', child_frame='camera_optical', source='tf',
             version='capture_tf_v1', epoch='boot1', mode='dynamic', length_unit='m',
             transform_direction='body_from_camera', status='unavailable',
             reason='waiting for TF', accuracy_validated=False)
    d.update(changes)
    return d


def packet(stamp=1_000_000_123, matrix=None, **changes):
    p = {k: v for k, v in declaration().items() if k != 'topic'}
    p.update(stamp_ns=stamp, clock_id='camera-clock', stamp_source='measurement',
             status='available', reason=None,
             matrix=np.asarray(np.eye(4) if matrix is None else matrix).tolist())
    p.update(changes)
    return p


def pose():
    return dict(frame_id='run1_odom',child_frame_id='camera_optical',
                position=[1.,2.,3.],quaternion=[0.,0.,0.,1.],covariance=None)
