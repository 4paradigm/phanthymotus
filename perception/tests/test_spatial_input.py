import sys
import threading
import time
from types import SimpleNamespace as NS
import pytest
from plugins.spatial_input import InputAdapterRegistry


class Backend:
    def __init__(self):self.callbacks=[];self.closed=[]
    def subscribe(self,topic,kind,callback):
        self.callbacks.append(callback)
        return lambda:self.closed.append(callback)


def message(stamp=0):
    return NS(header=NS(stamp=NS(sec=stamp//10**9,nanosec=stamp%10**9),frame_id='camera'),data=bytearray(b'abc'))


def declaration(epoch='one'):
    return dict(vendor=dict(frame_id='camera',source_id='camera',source_epoch=epoch),schema='motus.camera/1')


def wait_for(fn):
    end=time.monotonic()+1
    while not fn() and time.monotonic()<end:time.sleep(.002)
    assert fn()


def test_two_consumers_share_one_ingress_stamp_and_isolated_message():
    b=Backend();r=InputAdapterRegistry(subscribe=b.subscribe);a=[];c=[]
    def mutate(m,ctx):m.header.frame_id='changed';m.data[0]=0;a.append(ctx)
    x=r.acquire('/depth',object,declaration(),mutate)
    y=r.acquire('/depth',object,declaration(),lambda m,ctx:c.append((m,ctx)))
    try:
        assert len(b.callbacks)==1
        b.callbacks[0](message());wait_for(lambda:a and c)
        assert a[0] is c[0][1]
        assert c[0][0].header.frame_id=='camera' and c[0][0].data==b'abc'
        b.callbacks[0](message());wait_for(lambda:len(a)==2)
        assert a[0].observation_id != a[1].observation_id
    finally:x.close();y.close();r.close()


def test_stopping_one_consumer_keeps_other_alive_and_final_release_disposes():
    b=Backend();r=InputAdapterRegistry(subscribe=b.subscribe);seen=[]
    a=r.acquire('/depth',object,declaration(),lambda *v:None)
    c=r.acquire('/depth',object,declaration(),lambda *v:seen.append(v))
    a.close();a.close();assert not b.closed
    b.callbacks[0](message());wait_for(lambda:seen)
    c.close();assert len(b.closed)==1;r.close()


def test_slow_consumer_does_not_block_fast_consumer_and_capacity_is_bounded():
    b=Backend();r=InputAdapterRegistry(subscribe=b.subscribe,max_pending=2,max_bytes=6)
    entered=threading.Event();release=threading.Event();seen=[]
    def slow(*args):entered.set();release.wait(1)
    a=r.acquire('/depth',object,declaration(),slow)
    c=r.acquire('/depth',object,declaration(),lambda m,ctx:seen.append(ctx))
    try:
        b.callbacks[0](message());assert entered.wait(1)
        for _ in range(12):b.callbacks[0](message())
        wait_for(lambda:len(seen)>0)
        assert a.status()['pending']<=2 and a.status()['pending_bytes']<=6
        assert a.status()['dropped']>0
    finally:release.set();a.close();c.close();r.close()


def test_source_restart_uses_new_epoch_without_delivering_old_callbacks():
    b=Backend();r=InputAdapterRegistry(subscribe=b.subscribe);seen=[]
    a=r.acquire('/depth',object,declaration(),lambda m,c:seen.append(c));a.close()
    c=r.acquire('/depth',object,declaration('two'),lambda m,c:seen.append(c))
    try:
        b.callbacks[0](message());b.callbacks[1](message());wait_for(lambda:seen)
        assert len(seen)==1 and seen[0].source_epoch=='two'
    finally:c.close();r.close()


def test_different_live_declarations_share_one_subscription():
    b=Backend();r=InputAdapterRegistry(subscribe=b.subscribe)
    a=r.acquire('/depth',object,declaration(),lambda *v:None)
    second=r.acquire('/depth',object,declaration('two'),lambda *v:None)
    try:assert len(b.callbacks)==1
    finally:a.close();second.close();r.close()


def test_derived_cloud_keeps_first_ingress_and_parent_identity():
    b=Backend();r=InputAdapterRegistry(subscribe=b.subscribe);seen=[]
    a=r.acquire('/depth',object,declaration(),lambda m,c:seen.append(c))
    cloud=[];c=r.acquire('/cloud',object,{},lambda m,c:cloud.append(c))
    try:
        b.callbacks[0](message());wait_for(lambda:seen)
        m=message(seen[0].observation_stamp_ns)
        derived=r.register_derived('/cloud',m,seen[0],'pointcloud')
        b.callbacks[1](m);wait_for(lambda:cloud)
        assert cloud[0] is derived
        assert cloud[0].ingress_monotonic_ns==seen[0].ingress_monotonic_ns
        assert cloud[0].parent_observation_ids==(seen[0].observation_id,)
    finally:a.close();c.close();r.close()


def test_consumer_timing_separates_queue_wait_from_callback_work():
    b=Backend();r=InputAdapterRegistry(subscribe=b.subscribe);seen=[]
    lease=r.acquire('/depth',object,declaration(),lambda m,c:seen.append(c))
    try:
        b.callbacks[0](message());wait_for(lambda:seen and lease.status()['timing']['callback']['count']==1)
        timing=lease.status()['timing']
        assert timing['queue_wait']['count']==1
        assert timing['callback']['p95_ms']>=0
    finally:lease.close();r.close()


def test_ros_subscriptions_isolate_callback_groups_across_one_topic_restart(monkeypatch):
    from rclpy import node, qos

    default_group = object()
    created, removed = [], []

    def subscribe(kind, topic, callback, profile, *, callback_group=None):
        sub = NS(topic=topic, group=callback_group or default_group)
        created.append(sub)
        return sub

    ros_node = NS(create_subscription=subscribe, destroy_subscription=removed.append,
                  destroy_node=lambda: None)
    monkeypatch.setattr(node, 'Node', lambda *args, **kwargs: ros_node)
    monkeypatch.setattr(qos, 'qos_profile_sensor_data', object(), raising=False)
    monkeypatch.setitem(sys.modules, 'rclpy.callback_groups', NS(MutuallyExclusiveCallbackGroup=object))
    executor = NS(add_node=lambda node: None, remove_node=lambda node: None)
    registry = InputAdapterRegistry(executor)
    consume = lambda *args: None
    try:
        first = registry.acquire('/depth', object, {}, consume)
        second = registry.acquire('/depth', object, {}, consume)
        registry.acquire('/rgb', object, {}, consume)
        assert len(created) == 2
        first.close()
        assert not removed
        second.close()
        assert removed == created[:1]
        registry.acquire('/depth', object, {}, consume)
        assert [sub.topic for sub in created] == ['/depth', '/rgb', '/depth']
        assert len({sub.group for sub in created}) == 3
    finally:
        registry.close()
