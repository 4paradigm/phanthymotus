"""Synthetic geometry and OpenXR input for the isolated ROS acceptance harness."""
import copy
import hashlib
import json
import time
from teleop.kinematics import ARM_NAMES
from teleop.dispatch import MotionIntent


def synthetic_profile(tmp_path):
    # Anatomically meaningless fixture: exercises real FK/IK and collision code only.
    xml=['<robot name="offline"><link name="torso_link"/>']
    axes=['0 1 0','1 0 0','0 0 1','0 1 0','0 0 1','0 1 0','1 0 0']
    for side,sign in [('left',1),('right',-1)]:
        parent='torso_link'
        for i,name in enumerate(n for n in ARM_NAMES if n.startswith(side)):
            child=name.replace('_joint','_link');xyz=f'0 {sign*.5} 0' if i==0 else '.08 0 0'
            xml.append(f'<link name="{child}"/><joint name="{name}" type="revolute"><parent link="{parent}"/><child link="{child}"/><origin xyz="{xyz}"/><axis xyz="{axes[i]}"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint>')
            parent=child
    xml.append('</robot>');urdf=tmp_path/'synthetic.urdf';urdf.write_text(''.join(xml))
    profile=dict(arm_joint_names=list(ARM_NAMES),torso_frame='torso_link',schema='motus.tianyi-calibration.v1',version='offline-only',urdf_path=str(urdf),
        urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),locked_joints={},
        palm_frames={s:dict(position=[.05,0.,0.],orientation=[0.,0.,0.,1.]) for s in ('left','right')},
        controller_to_palm={s:dict(position=[0.,0.,0.],orientation=[0.,0.,0.,1.]) for s in ('left','right')},
        workspace={'torso_box':[[-.2,-.1,-.2],[.1,.1,.2]],'left':[[0,.3,-1],[1,1,1]],'right':[[0,-1,-1],[1,-.3,1]],
            'capsules':[{'from':f'{s}_elbow_pitch_link','to':f'{s}_wrist_roll_link','radius_m':.02,'group':s} for s in ('left','right')]})
    path=tmp_path/'profile.json';path.write_text(json.dumps(profile));return path


def intent(seq=1, clutch=1, generation=1, forward=0.):
    pose={'position':[0.,1.,0.], 'orientation':[0.,0.,0.,1.]}
    frame={'head':copy.deepcopy(pose), 'left_controller':copy.deepcopy(pose),
           'right_controller':copy.deepcopy(pose),
           'controllers':{s:{'buttons':[0.,1.]} for s in ('left','right')}}
    frame['left_controller']['position'][2]-=forward
    frame['right_controller']['position'][2]-=forward/2
    now=time.monotonic()
    return MotionIntent(1,generation,seq,clutch,now,now+.1,frame,True,now)
