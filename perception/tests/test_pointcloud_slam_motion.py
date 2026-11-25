"""Only measured native motion may compensate asynchronous SLAM observations."""
from types import SimpleNamespace as NS
import numpy as np
import pytest

from plugins.pointcloud_slam_motion import MotionTimeline


def result(stamp, x, *, seed=False, lost=False, complexity=.3):
    header=NS(stamp=NS(sec=stamp//10**9,nanosec=stamp%10**9))
    pose=NS(header=header,pose=NS(pose=NS(position=NS(x=x,y=0.,z=0.),
        orientation=NS(x=0.,y=0.,z=0.,w=1.)),covariance=(np.eye(6)*.01).ravel().tolist()))
    info=NS(header=header,lost=lost,covariance=(np.eye(6)*(9999 if seed else .01)).ravel().tolist(),
        icp_correspondences=0 if seed else 100,icp_inliers_ratio=0. if seed else .8,
        icp_structural_complexity=complexity)
    return pose,info


def add(t,stamp,x,**kw):
    p,i=result(stamp,x,**kw)
    t.observe('pose',p);t.observe('info',i)


def test_integer_epoch_join_and_arbitrary_local_origin_cancel():
    t=MotionTimeline(8)
    a=1_794_658_670_000_000_001
    p1,i1=result(a,100.);p2,i2=result(a+2000,102.)
    t.observe('info',i2);t.observe('pose',p1)
    with pytest.raises(ValueError,match='motion'):t.relative(a,a+2000)
    t.observe('info',i1);t.observe('pose',p2)
    np.testing.assert_allclose(t.relative(a+1000,a+2000)[:3,3],[-1,0,0])


def test_seed_cannot_claim_measured_motion_before_a_verified_registration():
    t=MotionTimeline(8);add(t,100,0.,seed=True)
    with pytest.raises(ValueError,match='motion'):t.relative(100,100)
    add(t,300,2.)
    np.testing.assert_allclose(t.relative(200,300)[:3,3],[-1,0,0])


def test_failure_barrier_cannot_be_bracketed_using_different_valid_runs():
    t=MotionTimeline(8);add(t,100,0.);add(t,200,0.,lost=True);add(t,300,5.);add(t,400,6.)
    for rgb,cloud in [(150,300),(100,400),(100,300)]:
        with pytest.raises(ValueError,match='motion'):t.relative(rgb,cloud)
    np.testing.assert_allclose(t.relative(350,400)[:3,3],[-.5,0,0])


@pytest.mark.parametrize('failure',['degenerate','covariance','quaternion'])
def test_unobservable_or_invalid_native_result_is_not_frozen_tracking(failure):
    t=MotionTimeline(8);add(t,100,0.)
    p,i=result(200,1.)
    if failure=='degenerate':i.icp_structural_complexity=0.
    elif failure=='covariance':i.covariance[0]=float('nan')
    else:p.pose.pose.orientation.w=0.
    t.observe('pose',p);t.observe('info',i);add(t,300,2.)
    with pytest.raises(ValueError,match='motion'):t.relative(150,300)


def test_capacity_and_incomplete_result_timeout_are_bounded():
    t=MotionTimeline(4)
    for stamp in range(30):
        p,_=result(stamp,float(stamp));t.observe('pose',p)
    assert len(t.pending)<=4
    for stamp in range(30,60):add(t,stamp,float(stamp))
    assert len(t.poses)<=4
    t.invalidate(60)
    p,i=result(60,60.);t.observe('info',i);t.observe('pose',p)
    add(t,61,61.)
    with pytest.raises(ValueError,match='motion'):t.relative(59,61)


def test_verified_motion_after_loss_can_cover_nonmonotonic_capture_times():
    t=MotionTimeline(8)
    add(t,1000,10.);add(t,200,0.,lost=True)
    add(t,300,3.);add(t,100,1.)
    np.testing.assert_allclose(t.relative(200,300)[:3,3],[-1,0,0])
    with pytest.raises(ValueError,match='motion'):t.relative(300,1000)


@pytest.mark.parametrize('barrier',['lost','seed'])
def test_new_native_run_cannot_borrow_old_motion_at_overlapping_capture_times(barrier):
    t=MotionTimeline(12);add(t,100,0.);add(t,300,2.)
    if barrier=='lost':t.invalidate(200)
    else:add(t,300,10.,seed=True)
    with pytest.raises(ValueError,match='motion'):t.relative(200,300)
    add(t,400,11.)
    with pytest.raises(ValueError,match='motion'):t.relative(200,300)


def test_scan_motion_reinitializes_its_local_map_after_failed_geometry():
    from plugins.pointcloud_slam_motion import motion_parameters
    parameters=motion_parameters({'voxel_size_m':.04},'local_body')
    assert parameters['Odom/ResetCountdown']=='1'
    assert parameters['qos']==1
    assert parameters['Odom/GuessMotion']=='true'
    assert parameters['publish_tf'] is False and parameters['guess_frame_id']==''


def test_reinitialization_seed_does_not_connect_two_native_gauges():
    t=MotionTimeline(8)
    add(t,100,10.);add(t,200,0.,seed=True);add(t,300,1.)
    with pytest.raises(ValueError,match='motion'):t.relative(100,300)
    np.testing.assert_allclose(t.relative(250,300)[:3,3],[-.5,0,0])


@pytest.mark.parametrize('field',['icp_inliers_ratio','icp_structural_complexity'])
def test_nonfinite_native_geometry_validation_cannot_certify_motion(field):
    t=MotionTimeline(8);add(t,100,0.)
    p,i=result(200,1.);setattr(i,field,float('nan'))
    t.observe('pose',p);t.observe('info',i)
    with pytest.raises(ValueError,match='motion'):t.relative(100,200)


def test_scan_motion_ignores_late_result_after_inflight_timeout(tmp_path):
    import threading
    from plugins.pointcloud_slam_motion import ScanMotion
    motion=ScanMotion(None,'/motion',{'max_pending':2},tmp_path,threading.Event())
    add(motion.timeline,100,0.)
    motion.timeline.invalidate(200)
    motion._inflight=300
    for kind,message in zip(('pose','info'),result(250,1.5)):
        motion._observe(kind,message)
    for kind,message in zip(('pose','info'),result(300,2.)):
        motion._observe(kind,message)
    assert motion._result.is_set()
    with pytest.raises(ValueError,match='motion'):motion.timeline.relative(250,300)
    motion.close()


def test_native_motion_log_is_drained_without_unbounded_disk_growth(tmp_path):
    import subprocess
    import sys
    import threading
    from plugins.pointcloud_slam_motion import ScanMotion
    motion=ScanMotion(None,'/motion',{'max_pending':2},tmp_path,threading.Event())
    path=tmp_path/'motion.log'
    motion._log=path.open('w+b',buffering=0)
    motion._process=subprocess.Popen([sys.executable,'-c',
        "import sys;sys.stdout.buffer.write(b'x'*(3*1024*1024)+b'finished\\n')"],stdout=subprocess.PIPE)
    motion._log_thread=threading.Thread(target=motion._drain_log,daemon=True)
    try:
        motion._log_thread.start()
        motion._process.wait(timeout=3.)
        motion._log_thread.join(timeout=3.)
        assert not motion._log_thread.is_alive()
        assert path.stat().st_size<=1024*1024
        assert path.read_bytes().endswith(b'finished\n')
    finally:
        motion._process.kill();motion._process.wait(timeout=3.)
        motion.close()


def test_close_releases_all_owned_ros_handles_and_is_idempotent(tmp_path):
    import threading
    from plugins.pointcloud_slam_motion import ScanMotion
    class Node:
        def __init__(self):self.handles={'publisher','pose','info'}
        def destroy_subscription(self,handle):self.handles.remove(handle)
        def destroy_publisher(self,handle):self.handles.remove(handle)
    node=Node()
    motion=ScanMotion(node,'/motion',{'max_pending':2},tmp_path,threading.Event())
    motion._publisher='publisher';motion._subscriptions=['pose','info']
    motion.close();motion.close()
    assert not node.handles


def test_verified_registration_can_measure_motion_backwards_from_seed_time():
    t=MotionTimeline(8)
    add(t,300,3.,seed=True);add(t,100,1.)
    np.testing.assert_allclose(t.relative(200,300)[:3,3],[-1,0,0])


def test_native_launch_uses_reliable_private_transport_without_external_ros_file_logs(tmp_path,monkeypatch):
    import json
    import subprocess
    import sys
    import threading
    import rtabmap_stubs
    from plugins.pointcloud_slam_motion import ScanMotion
    executable=tmp_path/'lib/rtabmap_odom/icp_odometry'
    executable.parent.mkdir(parents=True);executable.write_text('#!/bin/sh\nexit 0\n');executable.chmod(0o755)
    packages=NS(get_package_prefix=lambda name:str(tmp_path))
    monkeypatch.setitem(sys.modules,'ament_index_python',NS(packages=packages))
    monkeypatch.setitem(sys.modules,'ament_index_python.packages',packages)
    monkeypatch.setattr(sys.modules['rtabmap_msgs.msg'],'OdomInfo',object,raising=False)
    monkeypatch.setattr(sys.modules['rclpy.qos'],'qos_profile_sensor_data',object(),raising=False)
    qos_event = NS(PublisherEventCallbacks=lambda **kw: NS(**kw),
                   SubscriptionEventCallbacks=lambda **kw: NS(**kw))
    monkeypatch.setitem(sys.modules, 'rclpy.qos_event', qos_event)
    profiles, events = [], []
    def publisher(cls,topic,qos,**kw):
        events.append(kw.get('event_callbacks'));profiles.append(qos);return 'publisher'
    def subscription(cls,topic,callback,qos,**kw):
        events.append(kw.get('event_callbacks'));profiles.append(qos);return topic
    node=NS(create_publisher=publisher,create_subscription=subscription,
        destroy_publisher=lambda handle:None,destroy_subscription=lambda handle:None)
    commands=[]
    def inspect_launch(command,**kwargs):
        commands.append(command)
        raise RuntimeError('launch inspected without starting native runtime')
    monkeypatch.setattr(subprocess,'Popen',inspect_launch)
    motion=ScanMotion(node,'/motion',{'max_pending':2,'voxel_size_m':.08},tmp_path/'session',threading.Event())
    motion._run()
    assert len(commands)==1
    assert '--disable-external-lib-logs' in commands[0][commands[0].index('--ros-args')+1:]
    parameters=json.loads((tmp_path/'session/motion.params.json').read_text())['/**']['ros__parameters']
    assert parameters['qos']==1
    from rclpy.qos import ReliabilityPolicy
    assert len(profiles)==3
    assert all(p.depth==1 and p.reliability==ReliabilityPolicy.RELIABLE for p in profiles)
    assert all(item is not None and item.use_default_callbacks is False for item in events)
    assert parameters['Odom/GuessMotion']=='true'
    assert parameters['Icp/PointToPlaneLowComplexityStrategy']=='0'
    motion.close()


def test_status_audits_single_child_and_close_joins_owned_threads(tmp_path):
    import subprocess
    import sys
    import threading
    from plugins.pointcloud_slam_motion import ScanMotion
    motion=ScanMotion(None,'/motion',{'max_pending':2},tmp_path,threading.Event())
    motion._process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    motion._log=(tmp_path/'motion.log').open('w+b',buffering=0)
    motion._log_thread=threading.Thread(target=motion._drain_log,daemon=True)
    motion._thread=threading.Thread(target=motion._stop.wait,daemon=True)
    motion._log_thread.start();motion._thread.start()
    try:
        before=motion.status()
        assert before['process_pid']==motion._process.pid
        assert before['worker_alive'] and not before['stopped']
        motion.close()
        after=motion.status()
        assert after['stopped'] and not after['worker_alive']
        assert motion._process.poll() is not None
        assert not motion._log_thread.is_alive()
    finally:motion.close()


def test_lost_same_stamp_retry_ignores_the_old_null_pose(tmp_path):
    import threading
    from plugins.pointcloud_slam_motion import ScanMotion
    motion=ScanMotion(None,'/motion',{'max_pending':2},tmp_path,threading.Event())
    motion._inflight=200
    old_pose,lost_info=result(200,0.,lost=True)
    old_pose.pose.pose.orientation.w=0.
    motion._observe('info',lost_info)
    assert motion._result.is_set()
    motion.timeline.begin(200)
    motion._result.clear()
    motion._observe('pose',old_pose)
    pose,info=result(200,2.)
    motion._observe('info',info)
    assert not motion._result.is_set()
    motion._observe('pose',pose)
    assert motion._result.is_set()
    assert motion.timeline.accepted==1
    np.testing.assert_allclose(motion.timeline.relative(200,200),np.eye(4))
    motion.close()


def test_timeout_restarts_private_native_and_ignores_old_same_stamp_callbacks(tmp_path,monkeypatch):
    import sys
    import threading
    import time
    import rtabmap_stubs
    from plugins.pointcloud_slam_motion import ScanMotion
    from vision_stubs import _wait_until
    executable=tmp_path/'lib/rtabmap_odom/icp_odometry'
    executable.parent.mkdir(parents=True)
    executable.write_text('#!'+sys.executable+'\nimport time\ntime.sleep(60)\n')
    executable.chmod(0o755)
    packages=NS(get_package_prefix=lambda name:str(tmp_path))
    monkeypatch.setitem(sys.modules,'ament_index_python',NS(packages=packages))
    monkeypatch.setitem(sys.modules,'ament_index_python.packages',packages)
    monkeypatch.setattr(sys.modules['rtabmap_msgs.msg'],'OdomInfo',object,raising=False)
    monkeypatch.setattr(sys.modules['rclpy.qos'],'qos_profile_sensor_data',object(),raising=False)
    qos_event = NS(PublisherEventCallbacks=lambda **kw: NS(**kw),
                   SubscriptionEventCallbacks=lambda **kw: NS(**kw))
    monkeypatch.setitem(sys.modules, 'rclpy.qos_event', qos_event)
    monkeypatch.setattr(sys.modules['sensor_msgs.msg'],'PointCloud2',
        lambda:NS(header=NS(stamp=NS(sec=0,nanosec=0),frame_id='')))
    class Node:
        def __init__(self):
            self.callbacks={};self.publishers=[];self.handles=set();self.old_process=None
        def create_subscription(self,cls,topic,callback,qos,**kw):
            self.callbacks[topic]=callback;self.handles.add(topic);return topic
        def destroy_subscription(self,topic):self.handles.remove(topic)
        def destroy_publisher(self,publisher):self.handles.remove(publisher.topic)
        def create_publisher(self,cls,topic,qos,**kw):
            node=self
            class Publisher:
                def get_subscription_count(self):return 1
                def publish(self,message):
                    self.stamp=message.header.stamp
                    if len(node.publishers)==1:
                        node.old_process=motion._process
                        node.old_log_thread=motion._log_thread
                        node.old_callbacks=dict(node.callbacks)
                        return
                    assert node.old_process.poll() is not None
                    assert not node.old_log_thread.is_alive()
                    assert not any(t in node.handles for t in node.old_callbacks)
                    old_pose,old_info=result(200,999.)
                    for topic,callback in node.old_callbacks.items():
                        callback(old_info if topic.endswith('info_lite') else old_pose)
                    assert not motion._result.is_set()
                    pose,info=result(200,2.,seed=True)
                    base=self.topic.rsplit('/',1)[0]
                    node.callbacks[base+'/odom'](pose)
                    node.callbacks[base+'/odom_info_lite'](info)
            publisher=Publisher();publisher.topic=topic
            self.publishers.append(publisher);self.handles.add(topic)
            return publisher
    node=Node()
    motion=ScanMotion(node,'/motion',{'max_pending':2,'voxel_size_m':.08,'max_update_ms':1},
                      tmp_path/'session',threading.Event())
    try:
        points=np.array([[1.,2.,3.]],dtype=np.float32)
        motion.submit(200,points,np.eye(4))
        assert _wait_until(lambda:node.old_process is not None and node.old_process.poll() is not None,timeout=3.)
        time.sleep(.35)  # No queued frame while the timed-out helper is retired.
        assert motion.error is None
        motion.submit(200,points,np.eye(4))
        assert _wait_until(lambda:len(node.publishers)==2 and motion.timeline._seed is not None,timeout=4.), motion.error
        assert node.publishers[0].topic=='/motion/scan_cloud'
        assert node.publishers[0].topic!=node.publishers[1].topic
        assert motion.status()['namespace']==node.publishers[1].topic.rsplit('/',1)[0]
        assert node.publishers[1].stamp.sec==0 and node.publishers[1].stamp.nanosec==200
        assert motion.timeline.accepted==0
        with pytest.raises(ValueError,match='motion'):motion.timeline.relative(200,200)
    finally:motion.close()
    assert not node.handles
    assert not motion._log_thread.is_alive()
    assert motion._process.poll() is not None
    motion.close()
