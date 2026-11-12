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
    assert [p['format'] for p in tool['topic_in']]==['image/jpeg','image/depth-zlib','sensor/imu']
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


def test_resume_between_watchdog_ticks_cannot_bypass_stale_segment():
    gate=tracking_gate()
    assert gate.status(2_600_000_000)=='tracking'
    gate.input(2_750_000_000,now_ns=2_760_000_000)
    assert gate.accept(*messages(2_750_000_000),2_760_000_000) is None
    assert gate.state=='lost'


def test_fresh_estimate_cannot_overwrite_an_expired_previous_estimate():
    gate=tracking_gate()
    assert gate.accept(*messages(2_750_000_000),2_760_000_000) is None
    assert gate.state=='lost'


def test_unpaired_lost_diagnostic_ends_segment_before_next_good_pair():
    import threading
    from collections import OrderedDict
    from plugins.rgbd_odometry_runtime import RgbdSession
    session=object.__new__(RgbdSession)
    session.lock=threading.RLock();session.stop_event=threading.Event()
    session.imu=None;session.gate=tracking_gate();session.results={'pose':OrderedDict(),'info':OrderedDict()}
    session.counts={'estimates':0};published=[];session._publish=published.append
    msg,info=messages(1_300_000_000);info.lost=True
    session._result('info',info)
    assert session.gate.state=='lost'
    msg,info=messages(1_400_000_000)
    session._result('pose',msg);session._result('info',info)
    assert not published


def test_process_exit_during_stop_still_releases_node_and_tempfiles(monkeypatch):
    import os,subprocess,sys,tempfile,threading
    from pathlib import Path
    from plugins.rgbd_odometry_runtime import RgbdSession
    session=object.__new__(RgbdSession)
    session.stop_event=threading.Event();session.worker=None;session.log=None
    session.directory=tempfile.TemporaryDirectory();path=Path(session.directory.name)
    session.segment='test';events=[]
    session.node=NS(destroy_node=lambda:events.append('destroy'))
    session.executor=NS(remove_node=lambda n:events.append('remove'))
    session.process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(.1)'],start_new_session=True)
    killpg=os.killpg
    def race(pid,sig):
        session.process.wait(timeout=2)
        killpg(pid,sig)
    monkeypatch.setattr(os,'killpg',race)
    session.close()
    assert events==['remove','destroy'] and not path.exists()


def test_shutdown_attempts_other_instances_when_one_cleanup_fails():
    card=RgbdOdometryPlugin({},object());other=Session({},None,'/two')
    def fail():raise RuntimeError('cannot stop first estimator')
    card.instances={'first':(NS(close=fail),'a'),'second':(other,'b')}
    with pytest.raises(RuntimeError):card.shutdown()
    assert other.stopped and 'second' not in card.instances


def test_imu_toggle_is_per_instance_and_requires_third_binding(monkeypatch):
    from plugins import rgbd_odometry as module
    class RecordingSession(Session):
        def __init__(self, calibration, executor, topic):
            super().__init__(calibration,executor,topic);self.calibration=calibration
    monkeypatch.setattr(module,'RgbdSession',RecordingSession)
    monkeypatch.setattr(module,'require_runtime',lambda:None)
    card=RgbdOdometryPlugin({},object())
    field=card.get_tools()[0]['configSchema']['properties']['imu_assisted']
    assert field['default'] is False and field['scope']=='instance'
    result=card.dispatch('rgbd_odometry',{'action':'config','instance_id':'one','imu_assisted':True})
    assert result['config']['imu_assisted'] is True
    assert start(card,'one')['state']=='error'
    assisted=card.dispatch('rgbd_odometry',dict(action='start',instance_id='one',
        input_topics=['/rgb','/depth','/motion'],camera_info=declarations()))
    assert assisted['state']=='running'
    assert card.instances['one'][0].calibration['imu_topic']=='/motion'
    assert {p['topic'] for p in assisted['topic_in']}=={'/rgb','/depth','/motion'}
    assert start(card,'two')['state']=='running'
    assert 'imu_topic' not in card.instances['two'][0].calibration
    assert card.dispatch('rgbd_odometry',{'action':'config','instance_id':'one','imu_assisted':False})['state']=='error'
    card.shutdown()


def test_invalid_boolean_toggle_does_not_silently_enable_imu():
    card=RgbdOdometryPlugin({},object())
    assert card.dispatch('rgbd_odometry',{'action':'config','imu_assisted':'false'})['state']=='error'


def test_visual_frontend_settings_are_instance_scoped_and_require_restart(monkeypatch):
    from plugins import rgbd_odometry as module

    class RecordingSession(Session):
        def __init__(self, calibration, executor, topic):
            super().__init__(calibration, executor, topic)
            self.calibration = calibration

    monkeypatch.setattr(module, 'RgbdSession', RecordingSession)
    monkeypatch.setattr(module, 'require_runtime', lambda: None)
    Session.created = []
    card = RgbdOdometryPlugin({}, object())
    schema = card.get_tools()[0]['configSchema']['properties']
    assert schema['visual_frontend']['enum'] == ['f2m_feature', 'f2f_optical_flow']
    assert schema['visual_frontend']['default'] == 'f2m_feature'
    assert schema['max_features']['default'] == 1000
    assert start(card, 'normal')['state'] == 'running'
    assert card.instances['normal'][0].calibration['visual_frontend'] == 'f2m_feature'

    configured = card.dispatch('rgbd_odometry', {
        'action': 'config', 'instance_id': 'flow',
        'visual_frontend': 'f2f_optical_flow', 'max_features': 1500})
    assert configured['state'] == 'configured'
    assert start(card, 'flow')['state'] == 'running'
    first = card.instances['flow'][0]
    assert first.calibration['visual_frontend'] == 'f2f_optical_flow'
    assert first.calibration['max_features'] == 1500
    assert len(Session.created) == 2
    assert start(card, 'flow')['state'] == 'running'  # idempotent
    assert len(Session.created) == 2
    assert card.dispatch('rgbd_odometry', {
        'action': 'config', 'instance_id': 'flow',
        'visual_frontend': 'f2m_feature'})['state'] == 'error'
    assert not first.stopped

    card.dispatch('rgbd_odometry', {'action': 'stop', 'instance_id': 'flow'})
    assert first.stopped
    assert card.dispatch('rgbd_odometry', {
        'action': 'config', 'instance_id': 'flow',
        'visual_frontend': 'f2m_feature'})['state'] == 'configured'
    assert start(card, 'flow')['state'] == 'running'
    assert card.instances['flow'][0] is not first
    assert card.instances['flow'][0].calibration['visual_frontend'] == 'f2m_feature'
    card.shutdown()


@pytest.mark.parametrize('options', [
    {'visual_frontend': 'optical_flow'},
    {'visual_frontend': 'f2m_optical_flow'},
    {'visual_frontend': None},
    {'max_features': 0},
    {'max_features': 5001},
    {'max_features': 1000.0},
    {'max_features': True},
    {'max_features': '1000'},
])
def test_invalid_frontend_settings_never_launch_estimator(monkeypatch, options):
    from plugins import rgbd_odometry as module
    monkeypatch.setattr(module, 'RgbdSession', Session)
    Session.created = []
    card = RgbdOdometryPlugin({}, object())
    result = card.dispatch('rgbd_odometry', dict(
        action='start', input_topics=['/rgb', '/depth'],
        camera_info=declarations(), **options))
    assert result['state'] == 'error'
    assert not Session.created
