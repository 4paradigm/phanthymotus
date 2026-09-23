"""New end-effector producer, without ROS nodes or any hardware publisher."""
import copy
import hashlib
import hmac
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'plugins'))
from teleop.dispatch import MotionIntent
from teleop.motion_control import EefIntentAdapter, MotionControlLink
from teleop.plugin import TeleopPlugin
from teleop.project import validate_binding


class Link:
    def __init__(self):
        self.lease=None;self.preview_lease=None;self.management_request=None
        self.last_send=None;self.sent=[];self.calls=[];self.management_retry=False
        self.pending_resume=False
        self.metadata={'calibrated':True,'model_version':'model1','calibration_version':'cal1',
                       'frame':'chest','effector_ids':['left','right']}
        self.token={'boot_id':'boot1','session_id':'session1','secret':'ab'*32}
        self.poses=[[.2,.3,.4,0.,0.,0.,1.],[.2,-.3,.4,0.,0.,0.,1.]]
    def call(self, action, deadline, **kwargs):
        self.calls.append(action)
        assert deadline>time.monotonic()
        if action=='calibrate':return copy.deepcopy(self.metadata)
        return {}
    def feedback(self):
        return {'state':'ready','output_active':False,
                'timing_policy':{'recoverable_hold':True,'management_retry_ms':300},
                'eef_snapshot':{**self.metadata,'poses':copy.deepcopy(self.poses),
                    'state_seq':7,'monotonic_ns':time.monotonic_ns()}}
    def prepare_transport(self):pass
    def prepare_preview(self, deadline):
        self.calls.append('prepare_preview');self.preview_lease={**self.token,'preview':True}
    def claim(self, deadline):self.calls.append('claim');self.lease=dict(self.token)
    def resume(self, deadline):
        self.calls.append('resume')
        if self.pending_resume:
            self.pending_resume=False
            raise ValueError('driver_management_pending')
    def send_poses(self, values, snap, intent, epoch, deadline, *, live):
        assert deadline>time.monotonic()
        self.sent.append((values,copy.deepcopy(snap),intent.sequence,epoch,live))


def intent(seq=1, clutch=1):
    pose={'position':[0.,1.,0.], 'orientation':[0.,0.,0.,1.]}
    frame={'head':copy.deepcopy(pose),'left_controller':copy.deepcopy(pose),
           'right_controller':copy.deepcopy(pose),
           'controllers':{side:{'buttons':[0.,1.]} for side in ('left','right')}}
    now=time.monotonic()
    return MotionIntent(1,1,seq,clutch,now,now+.3,frame,received_monotonic=now)


def test_calibration_uses_driver_not_client_model_file():
    link=Link();adapter=EefIntentAdapter(link,'shadow')
    result=adapter.calibrate('/path/that/does/not/exist')
    assert result['calibrated'] and adapter.solver is None
    assert link.calls==['calibrate','prepare_preview'] and link.lease is None


def test_preview_maps_measured_pose_without_claim_or_joint_output():
    link=Link();adapter=EefIntentAdapter(link,'shadow');adapter.calibrate()
    assert adapter.apply(intent()).ok
    moved=intent(2);moved.frame['left_controller']['position'][1]+=.4
    assert adapter.apply(moved).ok
    values,snap,seq,epoch,live=link.sent[-1]
    np.testing.assert_allclose(values[:3],[.2,.3,.6])
    np.testing.assert_allclose(values[7:10],[.2,-.3,.4])
    assert seq==2 and epoch==1 and live is False and link.lease is None
    assert 'claim' not in link.calls


def test_regrip_uses_current_measured_snapshot_and_new_mapping_epoch():
    link=Link();adapter=EefIntentAdapter(link,'live');adapter.calibrate()
    adapter.apply(intent());adapter.apply(intent(2))
    link.poses[0][2]=.7
    adapter.apply(intent(3,2));adapter.apply(intent(4,2))
    assert link.sent[-1][0][2]==.7
    assert link.sent[-1][3]==2
    assert link.calls.count('claim')==1 and link.calls.count('resume')==1


def test_management_pending_retries_same_clutch_without_publishing_old_target():
    link=Link();adapter=EefIntentAdapter(link,'live');adapter.calibrate()
    adapter.apply(intent());adapter.apply(intent(2))
    link.pending_resume=True
    before=len(link.sent)
    assert adapter.apply(intent(3,2)).ok
    assert adapter.apply(intent(4,2)).ok
    assert len(link.sent)==before
    assert adapter.apply(intent(5,2)).ok
    assert len(link.sent)==before+1 and link.sent[-1][2]==5


def test_model_change_requires_new_calibration_before_any_publish():
    link=Link();adapter=EefIntentAdapter(link,'live');adapter.calibrate()
    link.metadata['model_version']='different'
    assert not adapter.apply(intent()).ok
    assert not link.sent and not link.lease


def test_wire_is_signed_eef_and_never_extends_original_input_deadline(monkeypatch):
    class String:pass
    monkeypatch.setitem(sys.modules,'std_msgs',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'std_msgs.msg',SimpleNamespace(String=String))
    link=MotionControlLink.__new__(MotionControlLink)
    link.lease={'boot_id':'boot','session_id':'session','secret':'ab'*32}
    link.preview_lease=None;link.seq=0
    messages=[];link.publisher=SimpleNamespace(publish=lambda m:messages.append(json.loads(m.data)))
    original=intent()
    deadline=original.expires_monotonic-.15
    snap={'model_version':'m','calibration_version':'c','frame':'chest'}
    values=[.1,.2,.3,0.,0.,0.,1.]*2
    link.send_poses(values,snap,original,9,deadline,live=True)
    message=messages[0]
    signature=message.pop('mac')
    assert signature==hmac.new(bytes.fromhex('ab'*32),json.dumps(message,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),hashlib.sha256).hexdigest()
    assert message['valid_until_ns']<=int(deadline*1e9)
    assert message['mode']=='eef_pose' and message['schema']=='motus.control/2'
    assert message['source_seq']==original.sequence and message['mapping_epoch']==9
    assert 'q' not in message and 'secret' not in message


def test_new_card_advertises_eef_and_feedback_only_not_robot_joints():
    card=TeleopPlugin({'control_backend':'motion_control'},None)
    tool=card.get_tools()[0]
    assert tool['topic_out'][0]['format']=='control/eef'
    assert tool['topic_in']==[{'id':'feedback','format':'data/json','role':'feedback'}]
    assert tool['inputSchema']['x-resource']==['arm_l','arm_r']
    assert not card._calibrated()


def test_v2_binding_requires_same_driver_execution_resource():
    value={'mcp_id':'driver','tool':'motion_control','url':'http://127.0.0.1:15707/mcp',
           'namespace':'robot','command_topic':'/robot/motion/control/command',
           'feedback_topic':'/robot/motion/teleop/feedback','robot_profile':'tianyi2','protocol_version':2}
    value['execution_binding']={**value,'tool':'arm','command_topic':'/robot/motion/arm/command','resources':['arm_l','arm_r']}
    assert validate_binding(value,'tianyi2')==value
    value['execution_binding']['mcp_id']='other'
    with pytest.raises(ValueError,match='execution_binding_mismatch'):validate_binding(value,'tianyi2')


def test_preview_release_failure_still_closes_local_transport():
    link=MotionControlLink.__new__(MotionControlLink)
    link.preview_lease={'session_id':'preview'}
    closed=[]
    link.executor=SimpleNamespace(remove_node=lambda node:closed.append('removed'))
    link.node=SimpleNamespace(destroy_node=lambda:closed.append('destroyed'))
    def unavailable(*_):raise OSError('driver offline')
    link.call=unavailable
    with pytest.raises(OSError,match='driver offline'):link.close()
    assert closed==['removed','destroyed'] and link.preview_lease is None


@pytest.mark.parametrize('invalid', [
    {'preview':True}, {'state':'hold'}, {'ownership_held':True},
    {'output_active':True}, {'boot_id':'different'}, {'monotonic_ns':0},
])
def test_preview_release_requires_new_matching_idle_feedback(invalid):
    link=MotionControlLink.__new__(MotionControlLink)
    link.lease=None
    token=link.preview_lease={'boot_id':'boot','session_id':'preview','secret':'ab'*32}
    link.condition=threading.Condition()
    calls=[]
    link.call=lambda action,deadline:calls.append(action) or {'state':'idle','preview':True}
    link.feedback=lambda:{'monotonic_ns':time.monotonic_ns(), 'boot_id':'boot',
        'preview':False, 'state':'idle','ownership_held':False,'output_active':False, **invalid}
    assert not link.release_preview(time.monotonic()+.025)
    assert calls==['release'] and link.preview_lease is token
