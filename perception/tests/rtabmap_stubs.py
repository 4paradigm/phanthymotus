"""Message-only stand-ins for native RTAB-Map adapter unit tests."""
import sys
import types
from types import SimpleNamespace as NS

import vision_stubs


def header():
    return NS(stamp=NS(sec=0, nanosec=0), frame_id='')


def vector():
    return NS(x=0., y=0., z=0.)


def quaternion():
    return NS(x=0., y=0., z=0., w=1.)


class Odometry:
    def __init__(self):
        self.header, self.child_frame_id, self.data = header(), '', None
        self.pose = NS(pose=NS(position=vector(), orientation=quaternion()), covariance=[0.] * 36)
        self.twist = NS(twist=NS(linear=vector(), angular=vector()), covariance=[0.] * 36)


class TransformStamped:
    def __init__(self):
        self.header, self.child_frame_id = header(), ''
        self.transform = NS(translation=vector(), rotation=quaternion())


class TFMessage:
    def __init__(self):
        self.transforms, self.data = [], None


class Info:
    pass


class MapGraph:
    pass


for package, classes in {
    'nav_msgs': {'Odometry': Odometry},
    'geometry_msgs': {'TransformStamped': TransformStamped},
    'tf2_msgs': {'TFMessage': TFMessage},
    'rtabmap_msgs': {'Info': Info, 'MapGraph': MapGraph},
}.items():
    module, messages = types.ModuleType(package), types.ModuleType(package + '.msg')
    messages.__dict__.update(classes)
    module.msg = messages
    sys.modules[package], sys.modules[package + '.msg'] = module, messages
