from types import SimpleNamespace as NS

import pytest

from plugins.rgbd_odometry import RgbdOdometryPlugin
from plugins.rgbd_odometry_runtime import EstimateGate
from plugins.pointcloud_slam_odom import decode_slam_odom
from test_rgbd_odometry_contract import declarations


def messages(stamp):
    header=NS(stamp=NS(sec=stamp//10**9,nanosec=stamp%10**9),frame_id='run_odom')
    msg=NS(header=header,child_frame_id='camera_optical',
           pose=NS(pose=NS(position=NS(x=0.,y=0.,z=0.),
                          orientation=NS(x=0.,y=0.,z=0.,w=1.)),
                   covariance=[.001 if i%7==0 else 0. for i in range(36)]))
    info=NS(header=header,lost=False,inliers=50)
    return msg,info


def tracking_gate():
    gate=EstimateGate('run','camera_optical','camera-clock')
    for i in range(3):
        stamp=1_000_000_000+i*100_000_000
        gate.input(stamp);sample=gate.accept(*messages(stamp),stamp+10_000_000)
    assert decode_slam_odom(sample).segment_id=='run'
    return gate


def test_lost_diagnostic_latches_until_new_estimator_segment():
    gate=tracking_gate();msg,info=messages(1_300_000_000);info.lost=True
    assert gate.accept(msg,info,1_310_000_000) is None
    assert gate.status(1_310_000_000)=='lost'
    assert gate.accept(*messages(1_400_000_000),1_410_000_000) is None
    assert gate.status(1_410_000_000)=='lost'
    assert tracking_gate().state=='tracking'


def test_stale_input_invalid_covariance_and_wrong_frames_stop_tracking():
    gate=tracking_gate()
    assert gate.status(2_400_000_000)=='tracking'
    assert gate.status(2_700_000_001)=='lost'
    for change in [lambda m:setattr(m.header,'frame_id','wrong'),
                   lambda m:m.pose.covariance.__setitem__(0,9999.),
                   lambda m:setattr(m.pose.pose.orientation,'w',0.)]:
        gate=tracking_gate();msg,info=messages(1_300_000_000);change(msg)
        assert gate.accept(msg,info,1_310_000_000) is None
        assert gate.state=='lost'


def test_diagnostics_must_match_actual_pose_capture_time():
    gate=tracking_gate();msg,_=messages(1_300_000_000);_,info=messages(1_400_000_000)
    assert gate.accept(msg,info,1_410_000_000) is None
    assert gate.last_stamp==1_200_000_000


class Session:
    created=[]
    def __init__(self, calibration, executor, topic):
        self.topic=topic;self.stopped=False;self.segment=str(len(self.created));self.created.append(self)
    def info(self):return {'state':'running','status':'initializing','segment_id':self.segment}
    def close(self):self.stopped=True


def start(card, instance):
    return card.dispatch('rgbd_odometry',dict(action='start',instance_id=instance,
        input_topics=['/rgb','/depth'],camera_info=declarations()))


def test_card_exposes_common_odom_and_isolates_instances(monkeypatch):
    from plugins import rgbd_odometry as module
    monkeypatch.setattr(module,'RgbdSession',Session)
    monkeypatch.setattr(module,'require_runtime',lambda:None)
    Session.created=[];card=RgbdOdometryPlugin({},object())
    tool=card.get_tools()[0]
    assert [p['format'] for p in tool['topic_in']]==['image/jpeg','image/depth-zlib']
    assert tool['odom_interface']['slam_pose']['profile']=='motus_slam/1'
    a=start(card,'one');b=start(card,'two');assert a['topic_out']!=b['topic_out']
    assert start(card,'one')['topic_out']==a['topic_out']
    assert len(Session.created)==2
    card.dispatch('rgbd_odometry',{'action':'stop','instance_id':'one'})
    assert Session.created[0].stopped and not Session.created[1].stopped
    start(card,'one');assert len(Session.created)==3
    card.shutdown();assert all(s.stopped for s in Session.created)


def test_invalid_geometry_never_launches_estimator(monkeypatch):
    from plugins import rgbd_odometry as module
    monkeypatch.setattr(module,'RgbdSession',Session)
    Session.created=[];card=RgbdOdometryPlugin({},object())
    result=card.dispatch('rgbd_odometry',{'action':'start','input_topics':['/rgb','/depth']})
    assert result['state']=='error' and not Session.created


def test_canvas_card_ids_produce_valid_distinct_ros_topics(monkeypatch):
    from plugins import rgbd_odometry as module
    monkeypatch.setattr(module,'RgbdSession',Session)
    monkeypatch.setattr(module,'require_runtime',lambda:None)
    card=RgbdOdometryPlugin({},object())
    a=start(card,'card-m123');b=start(card,'card_m123')
    assert a['state']=='running' and b['state']=='running'
    assert a['topic_out']!=b['topic_out']
    assert '-' not in a['topic_out'][0]['topic']
    assert card.get_tools()[0]['multiInstance'] is True
    card.shutdown()
