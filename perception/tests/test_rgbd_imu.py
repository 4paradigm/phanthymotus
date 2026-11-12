from collections import OrderedDict

import numpy as np
import pytest

from plugins.rgbd_imu import ImuAttitude, rotation_from_quaternion


def packet(stamp=1_000_000_000, **changes):
    p=dict(schema='motus.imu/1',source_id='camera/motion',epoch='boot1',
           stamp_ns=stamp,accel_stamp_ns=stamp,stamp_source='measurement',
           clock_id='capture-clock',camera_frame_id='optical',
           gyro_frame_id='gyro',accel_frame_id='accel',
           camera_from_gyro=np.eye(3).tolist(),camera_from_accel=np.eye(3).tolist(),
           rotation_source='sdk_factory_extrinsics',angular_velocity=[0.,0.,0.],
           linear_acceleration=[0.,0.,9.80665],
           units={'angular_velocity':'rad/s','linear_acceleration':'m/s^2'})
    p.update(changes);return p


def aid():return ImuAttitude('optical','capture-clock')


def test_gravity_initialization_maps_camera_specific_force_to_world_up():
    a=aid();out=a.observe(packet(linear_acceleration=[0.,-9.80665,0.]),1_010_000_000)
    np.testing.assert_allclose(rotation_from_quaternion(out['quaternion']) @ [0.,-1.,0.], [0,0,1],atol=1e-9)
    assert a.info(1_010_000_000)['heading']=='relative'


def test_known_gyro_rotation_and_factory_axis_rotation_are_applied_once():
    a=aid();rotation=[[0,-1,0],[1,0,0],[0,0,1]]
    # Sensor X angular velocity becomes camera +Y.
    for i in range(201):
        t=1_000_000_000+i*5_000_000
        out=a.observe(packet(t,angular_velocity=[1.,0.,0.],camera_from_gyro=rotation,
                            linear_acceleration=[0.,0.,0.] if i else [0.,0.,9.80665]),t+1_000_000)
    np.testing.assert_allclose(rotation_from_quaternion(out['quaternion']) @ [0,0,1],
                               [np.sin(1.),0,np.cos(1.)],atol=1e-7)
    assert out['stamp_ns']==2_000_000_000


@pytest.mark.parametrize('change',[
    {'clock_id':'other-clock'}, {'camera_frame_id':'body'}, {'epoch':''},
    {'units':{'angular_velocity':'deg/s','linear_acceleration':'m/s^2'}},
    {'camera_from_gyro':[[1,0,0],[0,1,0],[0,0,-1]]},
    {'angular_velocity':[0,float('nan'),0]}, {'stamp_source':'received'},
    {'accel_stamp_ns':970_000_000}, {'stamp_ns':True},
])
def test_invalid_source_is_not_promoted_to_attitude(change):
    a=aid()
    with pytest.raises(ValueError):a.observe(packet(**change),1_010_000_000)
    assert a.state=='lost'


@pytest.mark.parametrize('change',[
    {'epoch':'boot2'}, {'source_id':'other-camera'},
    {'gyro_frame_id':'new-frame'},
    {'camera_from_gyro':[[0,-1,0],[1,0,0],[0,0,1]]},
])
def test_source_restart_or_changed_extrinsic_cannot_reuse_filter_segment(change):
    a=aid();a.observe(packet(),1_010_000_000)
    with pytest.raises(ValueError):a.observe(packet(1_010_000_000,**change),1_020_000_000)


def test_stale_gap_repeated_stamp_and_expired_source_are_latched():
    for stamp,now in [(1_000_000_000,1_010_000_000),
                      (1_110_000_000,1_120_000_000),
                      (1_700_000_000,1_710_000_000)]:
        a=aid();a.observe(packet(),1_001_000_000)
        with pytest.raises(ValueError):a.observe(packet(stamp),now)
        assert a.state=='lost'
    a=aid();a.observe(packet(),1_010_000_000)
    assert a.status(1_600_000_000)=='lost'
    with pytest.raises(ValueError):a.observe(packet(1_605_000_000),1_610_000_000)


def test_frame_requires_bracketing_imu_coverage_with_bounded_history():
    a=aid();a.observe(packet(),1_001_000_000)
    assert not a.covers(1_010_000_000,1_011_000_000)
    a.observe(packet(1_020_000_000),1_021_000_000)
    assert a.covers(1_010_000_000,1_021_000_000)
    assert not a.covers(999_000_000,1_021_000_000)
    for i in range(1,601):
        t=1_020_000_000+i*5_000_000;a.observe(packet(t),t+1_000_000)
    assert a.info(t+1_000_000)['buffered_samples']<=512
    assert not a.covers(1_010_000_000,t+1_000_000)


def test_invalid_initial_gravity_waits_without_inventing_orientation():
    a=aid();assert a.observe(packet(linear_acceleration=[0,0,0]),1_010_000_000) is None
    assert a.state=='initializing'
    out=a.observe(packet(1_010_000_000),1_020_000_000)
    assert out and a.state=='active'


def test_optional_binding_is_explicit_and_off_mode_rejects_unconsumed_imu():
    from plugins.rgbd_odometry_contract import validate_start_inputs
    from test_rgbd_odometry_contract import declarations
    values=declarations()
    assert 'imu_topic' not in validate_start_inputs(values,['/depth','/rgb'],False)
    config=validate_start_inputs(values,['/motion','/depth','/rgb'],True)
    assert config['imu_topic']=='/motion'
    for topics,enabled in [(['/rgb','/depth','/motion'],False),(['/rgb','/depth'],True),
                           (['/rgb','/depth','/motion','/extra'],True)]:
        with pytest.raises(ValueError):validate_start_inputs(values,topics,enabled)


def test_native_estimator_parameters_only_enable_imu_when_requested():
    from plugins.rgbd_odometry_runtime import estimator_parameters
    base={'frame_id':'optical'}
    plain=estimator_parameters(base,'run')
    assisted=estimator_parameters(dict(base,imu_topic='/motion'),'run')
    assert not plain.get('wait_imu_to_init',False)
    assert assisted['wait_imu_to_init'] is True
    assert assisted['Odom/GuessSmoothingDelay']=='0'
    assert assisted['frame_id']=='optical'


@pytest.mark.parametrize('spawn_error', [False, True])
def test_session_buffers_imu_bursts_without_expanding_image_queues(monkeypatch, spawn_error):
    import sys
    from types import SimpleNamespace as NS
    from plugins import rgbd_odometry_runtime as runtime
    from vision_stubs import _FakeExecutor, _FakeNode
    monkeypatch.setattr(sys.modules['std_msgs.msg'], 'String', lambda **kw: NS(**kw))
    owned = []
    def single_executor(*, context):
        executor = _FakeExecutor(); executor.context = context; executor.closed = False
        def shutdown(**kw): executor.closed = True
        executor.shutdown = shutdown
        owned.append(executor)
        return executor
    monkeypatch.setattr(_FakeNode, 'context', object(), raising=False)
    monkeypatch.setattr(sys.modules['rclpy.executors'], 'SingleThreadedExecutor', single_executor)
    sensor_qos = object()
    monkeypatch.setattr(sys.modules['rclpy.qos'], 'qos_profile_sensor_data', sensor_qos, raising=False)
    monkeypatch.setattr(sys.modules['sensor_msgs.msg'], 'Imu', object, raising=False)
    monkeypatch.setitem(sys.modules, 'rtabmap_msgs.msg', NS(RGBDImage=object, OdomInfo=object))
    monkeypatch.setitem(sys.modules, 'nav_msgs.msg', NS(Odometry=object))
    monkeypatch.setitem(sys.modules, 'rclpy.callback_groups', NS(MutuallyExclusiveCallbackGroup=object))
    original = _FakeNode.create_subscription
    def subscribe(node, kind, topic, callback, qos, *, callback_group=None):
        sub = original(node, kind, topic, callback, qos)
        sub.callback_group = callback_group
        return sub
    monkeypatch.setattr(_FakeNode, 'create_subscription', subscribe)
    monkeypatch.setattr(runtime, 'require_runtime', lambda: '/offline/estimator')
    def process(*a, **kw):
        if spawn_error: raise RuntimeError('spawn failed')
        return NS(poll=lambda: 0, wait=lambda **kw: 0)
    monkeypatch.setattr(runtime.subprocess, 'Popen', process)
    monkeypatch.setattr(runtime.threading, 'Thread', lambda **kw: NS(start=lambda: None, join=lambda **kw: None, is_alive=lambda: False))
    calibration = dict(frame_id='optical', clock_id='capture-clock', K=np.eye(3),
                       color_topic='/rgb', depth_topic='/depth', imu_topic='/motion')
    shared = _FakeExecutor()
    if spawn_error:
        with pytest.raises(RuntimeError, match='spawn failed'):
            runtime.RgbdSession(calibration, shared, '/out')
        assert len(owned) == 1 and owned[0].closed and not owned[0].nodes and not shared.nodes
        return
    obj = runtime.RgbdSession(calibration, shared, '/out')
    try:
        assert len(owned) == 1 and obj.executor is owned[0]
        assert obj.executor.context is obj.node.context and not shared.nodes
        qos = {s.topic: s.qos for s in obj.node.subscriptions}
        assert qos['/motion'].depth == 100
        assert qos['/motion'].reliability == 'best_effort'
        assert qos['/rgb'] is sensor_qos and qos['/depth'] is sensor_qos
        groups = {s.topic: s.callback_group for s in obj.node.subscriptions}
        assert groups['/motion'] is not None and groups['/rgb'] is None
    finally:
        obj.close()
    assert owned[0].closed and not owned[0].nodes


def test_private_callback_worker_stops_before_node_and_executor_cleanup():
    import tempfile, threading
    from types import SimpleNamespace as NS
    from plugins import rgbd_odometry_runtime as runtime
    obj = object.__new__(runtime.RgbdSession)
    obj.lock = threading.RLock(); obj.stop_event = threading.Event()
    obj.gate = runtime.EstimateGate('run', 'optical', 'capture-clock'); obj.fusion = None
    obj.worker = obj.process = obj.log = None; obj.owns_executor = True
    obj.status_publisher = None
    obj.directory = tempfile.TemporaryDirectory(); obj.segment = 'run'
    entered = threading.Event(); events = []
    def spin_once(timeout_sec): entered.set(); obj.stop_event.wait(.01)
    def destroy():
        assert not obj.spin_worker.is_alive()
        events.append('destroy')
    obj.node = NS(destroy_node=destroy)
    obj.executor = NS(spin_once=spin_once, remove_node=lambda n: events.append('remove'),
                      shutdown=lambda **kw: events.append('shutdown'))
    obj.spin_worker = threading.Thread(target=obj._spin)
    obj.spin_worker.start()
    assert entered.wait(1)
    obj.close()
    assert events == ['remove', 'destroy', 'shutdown']


def test_private_callback_failure_retires_estimator():
    import threading
    from types import SimpleNamespace as NS
    from plugins import rgbd_odometry_runtime as runtime
    obj = object.__new__(runtime.RgbdSession)
    obj.stop_event = threading.Event(); obj.lock = threading.RLock()
    obj.gate = runtime.EstimateGate('run', 'optical', 'capture-clock'); obj.fusion = None
    def fail(**kw): raise RuntimeError('callback failed')
    obj.executor = NS(spin_once=fail)
    obj._spin()
    assert obj.gate.state == 'lost' and 'callback failed' in obj.gate.reason


def session(monkeypatch):
    import json,threading
    from types import SimpleNamespace as NS
    from plugins import rgbd_odometry_runtime as runtime
    import sys
    class Imu:
        def __init__(self):
            self.header=NS(stamp=NS(sec=0,nanosec=0),frame_id='')
            self.orientation=NS(x=0.,y=0.,z=0.,w=0.)
            self.angular_velocity=NS(x=0.,y=0.,z=0.)
            self.linear_acceleration=NS(x=0.,y=0.,z=0.)
            self.orientation_covariance=[0.]*9
    monkeypatch.setattr(sys.modules['sensor_msgs.msg'],'Imu',Imu,raising=False)
    obj=object.__new__(runtime.RgbdSession)
    obj.lock=threading.RLock();obj.stop_event=threading.Event();obj.imu=aid()
    obj.calibration={'frame_id':'optical','clock_id':'capture-clock'}
    obj.fusion=None;obj.registration_by_stamp=OrderedDict()
    obj.status_publisher=NS(publish=lambda msg:None)
    obj.summary_origin=None;obj.last_summary_ns=obj.last_summary_state=None
    obj.segment='run';obj.gate=runtime.EstimateGate('run','optical','capture-clock')
    obj.process=None;obj.started=runtime.time.monotonic()
    published=[];obj.imu_pub=NS(publish=published.append);obj._publish=lambda sample:None
    return obj,published,runtime,NS,json


@pytest.mark.parametrize('delay_ns,active', [(101_000_000, True), (601_000_000, False)])
def test_imu_ingress_keeps_intervals_but_rejects_stale_backlog(monkeypatch, delay_ns, active):
    import threading
    from collections import deque
    obj, published, runtime, NS, json = session(monkeypatch)
    obj.imu_pending = deque();obj.imu_queue_lock = threading.Lock();obj.imu_overflow = False
    monkeypatch.setattr(runtime.time, 'time_ns', lambda: 1_000_000_000+delay_ns)
    def receive():
        for i in range(20):
            obj._enqueue_imu(NS(data=json.dumps(packet(1_000_000_000+i*5_000_000))))
    with obj.lock:
        receiver = threading.Thread(target=receive)
        receiver.start();receiver.join(timeout=1)
        assert not receiver.is_alive()  # ingress must not wait for visual processing
        assert len(obj.imu_pending) == 20 and not published
    with obj.lock:
        obj._drain_imu()
    if active:
        assert len(published) == 20 and obj.imu.state == 'active'
        assert list(obj.imu.history) == [1_000_000_000+i*5_000_000 for i in range(20)]
    else:
        assert not published and obj.imu.state == 'lost'


def test_imu_ingress_overflow_retires_segment_instead_of_hiding_dropped_captures(monkeypatch):
    import threading
    from collections import deque
    obj, published, runtime, NS, json = session(monkeypatch)
    obj.imu_pending = deque([NS(data=json.dumps(packet()))]*100)
    obj.imu_queue_lock = threading.Lock();obj.imu_overflow = False
    with obj.lock:
        receiver = threading.Thread(target=lambda: obj._enqueue_imu(NS(data=json.dumps(packet()))))
        receiver.start();receiver.join(timeout=1)
        assert not receiver.is_alive()
        obj._drain_imu()
    assert not published and obj.gate.state == obj.imu.state == 'lost'
    assert 'overflow' in obj.imu.reason


def test_imu_ingress_only_queues_and_draining_does_not_chase_new_arrivals(monkeypatch):
    import threading
    from collections import deque
    obj, published, runtime, NS, json = session(monkeypatch)
    obj.imu_pending = deque();obj.imu_queue_lock = threading.Lock();obj.imu_overflow = False
    obj._enqueue_imu(NS(data=json.dumps(packet())))
    assert len(obj.imu_pending) == 1 and not published
    consumed = []
    def consume(msg):
        consumed.append(msg)
        if len(consumed) == 1:
            obj._enqueue_imu(NS(data='next batch'))
    obj._imu = consume
    with obj.lock:
        obj._drain_imu()
    assert len(consumed) == 1 and len(obj.imu_pending) == 1


def test_imu_fault_during_publication_cannot_emit_a_prebuilt_tracking_pose(monkeypatch):
    import sys, threading
    from collections import deque
    obj, _, runtime, NS, json = session(monkeypatch)
    obj.imu_pending = deque();obj.imu_queue_lock = threading.Lock();obj.imu_overflow = True
    monkeypatch.setattr(runtime.time, 'time_ns', lambda: 1_001_000_000)
    monkeypatch.setattr(sys.modules['std_msgs.msg'], 'String', lambda **kw: NS(**kw))
    output = []
    obj.publisher = NS(publish=lambda msg: output.append(json.loads(msg.data)))
    pose = dict(position=[0., 0., 0.], quaternion=[0., 0., 0., 1.],
                covariance=(np.eye(6)*.0001).ravel().tolist(),
                frame_id='run_odom', child_frame_id='optical')
    sample = runtime.make_sample('tracking', 1_000_000_000, 'run', 'capture-clock', pose)
    runtime.RgbdSession._publish(obj, sample)
    assert obj.gate.state == 'lost'
    assert output[0]['vendor']['motus_slam']['status'] == 'lost'
    assert output[0]['pose'] is None


def test_private_imu_retains_source_stamp_and_outputs_camera_attitude(monkeypatch):
    obj,published,runtime,NS,json=session(monkeypatch)
    monkeypatch.setattr(runtime.time,'time_ns',lambda:1_010_000_000)
    obj._imu(NS(data=json.dumps(packet(linear_acceleration=[0,-9.80665,0]))))
    assert len(published)==1
    msg=published[0]
    assert msg.header.stamp.sec==1 and msg.header.stamp.nanosec==0
    assert msg.header.frame_id=='optical'
    assert msg.orientation_covariance==[0.]*9  # unknown, not fabricated confidence
    np.testing.assert_allclose([msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w],
                               [-np.sqrt(.5),0,0,np.sqrt(.5)],atol=1e-9)


def test_enabled_imu_cannot_be_silently_bypassed_by_images_or_watchdog(monkeypatch):
    obj,_,runtime,NS,json=session(monkeypatch)
    monkeypatch.setattr(runtime.time,'time_ns',lambda:1_010_000_000)
    obj._imu(NS(data=json.dumps(packet())))
    assert not obj._imu_frame_ready(1_010_000_000,1_011_000_000)
    obj.imu.observe(packet(1_020_000_000),1_021_000_000)
    assert obj._imu_frame_ready(1_010_000_000,1_021_000_000)
    obj.gate.input(1_020_000_000)
    monkeypatch.setattr(runtime.time,'time_ns',lambda:1_600_000_000)
    obj._watchdog()
    assert obj.gate.state=='lost'
    assert not obj._imu_frame_ready(1_610_000_000,1_611_000_000)


def test_bad_imu_packet_reports_loss_without_forwarding_a_prior(monkeypatch):
    obj,published,runtime,NS,json=session(monkeypatch)
    monkeypatch.setattr(runtime.time,'time_ns',lambda:1_010_000_000)
    obj._imu(NS(data=json.dumps(packet(clock_id='wrong'))))
    assert not published and obj.gate.state=='lost'


def test_gravity_correction_removes_tilt_without_integrating_translation():
    a=aid();a.observe(packet(),1_001_000_000)
    for i in range(1,41):
        t=1_000_000_000+i*5_000_000
        a.observe(packet(t,angular_velocity=[.2,0,0],linear_acceleration=[0,0,0]),t)
    before=rotation_from_quaternion(a.quaternion) @ [0,0,1]
    assert abs(before[1])>.03
    for i in range(41,441):
        t=1_000_000_000+i*5_000_000;a.observe(packet(t),t)
    after=rotation_from_quaternion(a.quaternion) @ [0,0,1]
    assert abs(after[1])<.002
    assert a.info(t)['translation_integrated'] is False


def test_nontracking_output_keeps_selected_imu_mode_and_health(monkeypatch):
    from plugins.rgbd_odometry_contract import make_sample
    import sys
    obj,_,runtime,NS,json=session(monkeypatch)
    messages=[];obj.publisher=NS(publish=messages.append)
    monkeypatch.setattr(sys.modules['std_msgs.msg'],'String',lambda **kw:NS(**kw),raising=False)
    monkeypatch.setattr(runtime.time,'time_ns',lambda:1_010_000_000)
    del obj._publish
    obj.imu.fault('source reset')
    obj._publish(make_sample('lost',1_010_000_000,'run','capture-clock',reason='IMU source reset'))
    state=json.loads(messages[0].data)['vendor']['imu_assistance']
    assert state['enabled'] is True and state['status']=='lost'
