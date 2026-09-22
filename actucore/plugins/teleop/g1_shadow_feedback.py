"""Read-only G1 sensor feedback for Shadow IK without a teleop execution plugin."""
import json
import math
import time
from .adapter import DriverLink

ARM_INDICES = (15,16,17,18,19,22,23,24,25,26)


def parse_joints(payload, received_ns):
    rows = payload['joints']
    if not isinstance(rows,list) or len(rows)!=35:raise ValueError('g1_joint_count')
    motors={}
    for row in rows:
        idx=row['idx']
        if type(idx) is not int or idx in motors or not 0<=idx<35:raise ValueError('g1_joint_index')
        if any(type(row[k]) not in (int,float) or not math.isfinite(row[k]) for k in ('q','dq')):
            raise ValueError('g1_joint_nonfinite')
        motors[idx]=row
    return {'profile_id':'unitree_g1_23_dual_arm_relative_v1',
            'state':'idle','reason':None,'ownership_held':False,'output_active':False,
            'publisher_present':False,'monotonic_ns':received_ns,
            'feedback_source':'driver_joints_readonly','source_timestamp_available':False,
            'feedback':{'q':[motors[i]['q'] for i in ARM_INDICES],
                        'dq':[motors[i]['dq'] for i in ARM_INDICES], 'arm_ns':received_ns}}


class G1ShadowFeedbackLink(DriverLink):
    # The standard sensor publishes at 10 Hz. This is a display/Shadow-only
    # receive timeout, never a robot motion TTL or proof of source clock age.
    feedback_max_age_ns=250_000_000

    def __init__(self,cfg,executor):
        if cfg.get('mode','shadow')!='shadow' or cfg.get('robot_profile')!='g1_23':
            raise ValueError('joints_feedback_requires_g1_shadow')
        super().__init__(cfg,executor)
        from std_msgs.msg import String
        self.joint_sample=None
        self.node.create_subscription(String,f"/{cfg['namespace']}/state/joints",self._joints,self.qos)

    def _joints(self,msg):
        try:value=parse_joints(json.loads(msg.data),time.monotonic_ns())
        except (ValueError,KeyError,TypeError):return
        with self.condition:
            self.joint_sample=value
            self.condition.notify_all()

    def feedback(self):
        with self.condition:value=self.joint_sample
        if value is None:raise ValueError('driver_joints_missing')
        if not 0<=time.monotonic_ns()-value['monotonic_ns']<=self.feedback_max_age_ns:
            raise ValueError('driver_joints_stale')
        return value

    def call(self,*args,**kwargs):raise ValueError('shadow_feedback_has_no_execution_authority')
    def claim(self,*args,**kwargs):raise ValueError('shadow_feedback_has_no_execution_authority')
    def resume(self,*args,**kwargs):raise ValueError('shadow_feedback_has_no_execution_authority')
    def send(self,*args,**kwargs):raise ValueError('shadow_feedback_has_no_execution_authority')
