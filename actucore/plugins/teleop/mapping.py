"""Robot-independent OpenXR relative pose mapping. No model or robot SDK."""
import numpy as np

def finite(value, shape):
    raw=np.asarray(value)
    if raw.dtype.kind not in 'iuf':raise ValueError('invalid_numeric_type')
    x = np.asarray(value, dtype=float)
    if x.shape != shape or not np.isfinite(x).all():
        raise ValueError("invalid_finite_shape")
    return x


def transform(pose):
    p = finite(pose['position'], (3,))
    x,y,z,w = finite(pose['orientation'], (4,))
    if abs(x*x+y*y+z*z+w*w-1) > 0.002:
        raise ValueError('quaternion_not_unit')
    t = np.eye(4)
    t[:3,3] = p
    t[:3,:3] = [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
               [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
               [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    return t


class RelativeMapping:
    def __init__(self, scale=0.5):
        if not 0 < scale <= 1:
            raise ValueError('position_scale')
        self.scale = scale
        self.reference = None
        self.controller_offsets={s:np.eye(4) for s in ("left","right")}

    def reset(self, frame, measured_palms):
        # OpenXR +X right,+Y up,-Z forward -> robot +X forward,+Y left,+Z up.
        basis = np.array([[0,0,-1],[-1,0,0],[0,1,0]], dtype=float)
        head = basis @ transform(frame['head'])[:3,:3]
        forward = head @ np.array([0,0,-1.0])
        yaw = np.arctan2(forward[1], forward[0])
        c,s = np.cos(-yaw),np.sin(-yaw)
        self.rotation = np.array([[c,-s,0],[s,c,0],[0,0,1]]) @ basis
        self.reference = [(transform(frame[f'{side}_controller']) @ self.controller_offsets[side], finite(palm,(4,4)).copy())
                          for side,palm in zip(('left','right'),measured_palms)]

    def targets(self, frame):
        if self.reference is None:
            raise ValueError('mapping_not_calibrated')
        targets=[]
        for side,(origin,robot) in zip(('left','right'),self.reference):
            current=transform(frame[f'{side}_controller']) @ self.controller_offsets[side]
            target=robot.copy()
            target[:3,3] += self.scale * self.rotation @ (current[:3,3]-origin[:3,3])
            target[:3,:3] = self.rotation @ current[:3,:3] @ origin[:3,:3].T @ self.rotation.T @ robot[:3,:3]
            targets.append(target)
        return targets
