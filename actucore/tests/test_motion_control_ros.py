"""Opt-in, isolated ROS 2 integration of the current three-stage control path.

RUN_MOTION_CONTROL_ROS=1 requires a network-none container with loopback only,
ROS Humble, numerical/test dependencies and TIANYI_DRIVER_SOURCE. Ordinary test
runs skip this module; explicitly requested runs fail on missing prerequisites.
The actual Driver bus subprocess, both DDS command edges, feedback edge,
MotionControlLink, IK, arm admission and MotionGate run unmodified. Only MCP
HTTP transport, vendor I/O and measured hardware are substitutes. The finite
plant and synthetic URDF are NOT hardware or real-model acceptance evidence.
"""
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

if os.environ.get('RUN_MOTION_CONTROL_ROS') != '1':
    pytest.skip('opt-in isolated ROS test: set RUN_MOTION_CONTROL_ROS=1', allow_module_level=True)

import fcntl
import struct

# Do not importorskip here: explicit execution must fail if ROS is unavailable.
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from test_motion_control_chain import apply, frame, motion_frame, wait_for
from test_tianyi_execution_chain import tianyi_driver_source
from teleop.motion_control import EefIntentAdapter, MotionControlLink


@pytest.fixture
def ros_chain(tmp_path, monkeypatch):
    assert os.environ.get('ROS_DOMAIN_ID') == '42'
    # Docker Desktop also exposes unconfigured tunnel devices in this namespace.
    # Only loopback may be UP and no IP route may exist.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        active = {name for _, name in socket.if_nameindex()
                  if struct.unpack('H', fcntl.ioctl(probe, 0x8913, struct.pack('256s', name.encode()))[16:18])[0] & 1}
    assert active == {'lo'}, 'requires Docker --network none'
    assert len(Path('/proc/net/route').read_text().splitlines()) == 1
    assert os.environ.get('TIANYI_DRIVER_SOURCE'), 'explicit ROS run requires TIANYI_DRIVER_SOURCE'
    root = tianyi_driver_source()
    assert Path('/deploy/dds-local.xml').read_bytes() == (root/'dds-local.xml').read_bytes()
    spec = importlib.util.spec_from_file_location('_ros_driver_fixture', root/'tests/test_motion_control.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = module.chain.__wrapped__(tmp_path, monkeypatch)
    driver = next(fixture)
    e, c, p = driver.e, driver.c, driver.p
    rclpy.init(domain_id=42)
    ros = SingleThreadedExecutor()
    link = MotionControlLink({'namespace':'offline', 'driver_mcp_url':'http://127.0.0.1:1/mcp'}, ros)
    monitor = Node('offline_motion_evidence')
    ros.add_node(monitor)
    messages = {kind: [] for kind in ('eef', 'joints', 'feedback')}
    topics = {'eef':'/offline/motion/control/command', 'joints':'/offline/motion/arm/command',
              'feedback':'/offline/motion/teleop/feedback'}
    for kind, topic in topics.items():
        monitor.create_subscription(String, topic,
            lambda msg, kind=kind: messages[kind].append(json.loads(msg.data)), link.qos)
    management = []
    class LocalMCP:
        def open(self, request, timeout):
            assert timeout > 0 and request.full_url == link.url
            body = json.loads(request.data)
            assert body['method'] == 'tools/call' and body['params']['name'] == 'motion_control'
            args = body['params']['arguments']
            management.append(args['action'])
            value = c.dispatch(args['action'], args)
            return io.BytesIO(json.dumps({'result':{'isError':bool(value.get('error')),
                'content':[{'type':'text','text':json.dumps(value)}]}}).encode())
    link.opener = LocalMCP()
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    parent.setblocking(False)
    e._bus_socket = parent
    bus_log = (tmp_path/'bus.log').open('w+')
    e._bus_process = subprocess.Popen([sys.executable, str(root/'teleop_executor.py'),
        '--bus', str(child.fileno()), 'offline', '--control-v2'], pass_fds=(child.fileno(),),
        env=dict(os.environ), stdout=bus_log, stderr=subprocess.STDOUT)
    child.close()
    # Restore the real socket->DDS publisher replaced by the numerical fixture.
    monkeypatch.setattr(e, 'publish_joint_command', module.TeleopExecutor.publish_joint_command.__get__(e))
    e._closed.clear()
    e._thread = threading.Thread(target=e._watchdog_loop, daemon=True)
    closed = threading.Event()
    def plant_loop():
        while not closed.wait(.02):
            p.tick()  # Hardware measurement advances independently of publication.
    plant = threading.Thread(target=plant_loop, daemon=True)
    spin = threading.Thread(target=ros.spin, daemon=True)
    vendor_writes = []
    send = e.arm._send_pos
    def vendor(poses, speed):
        with p.lock:
            before = p.q.copy()
            result = send(poses, speed)
            np.testing.assert_array_equal(p.q, before)
            vendor_writes.append({'poses':poses, 'speed':speed})
            return result
    monkeypatch.setattr(e.arm, '_send_pos', vendor)
    plant.start();spin.start();e._thread.start()
    # Use the actual numerical worker; executor.start remains a no-op to avoid
    # creating vendor subscriptions, while its actual watchdog is above.
    module.MotionControl.start(c)
    value = SimpleNamespace(**vars(driver), link=link, messages=messages, topics=topics,
                            monitor=monitor, management=management, vendor_writes=vendor_writes)
    try:
        link.prepare_transport()
        wait_for(lambda: all(monitor.count_publishers(t) >= 1 and monitor.count_subscribers(t) >= 1
                            for t in topics.values()), timeout=8.)
        wait_for(lambda: bool(messages['feedback']) and link.latest is not None)
        yield value
    finally:
        c.stop()
        closed.set();plant.join(1)
        e._closed.set();e._thread.join(1)
        e._bus_process.terminate()
        e._bus_process.wait(timeout=3)
        parent.close()
        ros.shutdown(timeout_sec=2)
        spin.join(2)
        link.node.destroy_node();monitor.destroy_node()
        rclpy.try_shutdown()
        fixture.close()
        bus_log.seek(0)
        errors = bus_log.read()
        bus_log.close()
        assert not spin.is_alive() and not plant.is_alive() and not e._thread.is_alive()
        assert 'Traceback' not in errors, errors


@pytest.mark.parametrize('mode', ['shadow', 'live'])
def test_real_dds_eef_joint_and_feedback(ros_chain, mode):
    chain = ros_chain
    adapter = EefIntentAdapter(chain.link, mode)
    assert adapter.calibrate()['calibrated']
    # Calibration/claim replies are not feedback: await the real topic snapshot.
    wait_for(lambda: chain.link.latest['eef_snapshot']['model_version'] == chain.c.versions['model_version'])
    now = chain.link.last_feedback_received_ns
    wait_for(lambda: chain.link.last_feedback_received_ns > now)
    applied = apply(chain_for_apply(chain), adapter, 1)
    assert not chain.messages['eef'] and not chain.vendor_writes
    target = motion_frame(chain, adapter)
    for seq in (2, 3, 4):
        applied = apply(chain_for_apply(chain), adapter, seq, value=target)
    wait_for(lambda: chain.messages['eef'] and chain.messages['eef'][-1]['source_seq'] == 4)
    wait_for(lambda: chain.messages['feedback'][-1].get('control_decision', {}).get('source_seq') == 4)
    feedback = chain.messages['feedback'][-1]
    assert feedback['control_decision']['state'] == ('preview' if mode == 'shadow' else 'target_published'), feedback['control_decision']
    eef = chain.messages['eef'][-1]
    assert eef['schema'] == 'motus.control/2' and eef['mode'] == 'eef_pose'
    assert eef['mapping_epoch'] == 1 and adapter.solver is None
    if mode == 'shadow':
        assert not chain.messages['joints'] and not chain.vendor_writes and not chain.p.writes
        assert not chain.e.gate.session_id and 'claim' not in chain.management
        assert feedback['visualization']['ik']
    else:
        wait_for(lambda: chain.messages['joints'] and chain.messages['joints'][-1]['source_seq'] == 4)
        joint = chain.messages['joints'][-1]
        assert joint['mode'] == 'joint_position' and len(joint['values']) == 14
        assert joint['session_id'] == eef['session_id'] and joint['mapping_epoch'] == 1
        assert joint['valid_until_ns'] <= eef['valid_until_ns']
        assert 0 < joint['valid_until_ns']-joint['generated_ns'] <= 100_000_000
        wait_for(lambda: bool(chain.vendor_writes) and np.max(np.abs(chain.p.q)) > 0)
        assert np.max(np.abs(chain.p.dq)) <= 1.
        assert any(np.allclose(np.deg2rad(w['poses']['left']+w['poses']['right']), joint['values'])
                   for w in chain.vendor_writes)
        assert chain.link.pause(time.monotonic()+1.)
        assert chain.e.gate.status()['hold_confirmed']
    # FastDDS discovery does not expose remote history/depth on Humble (UNKNOWN/0).
    # Validate depth from real local ROS entities and report remote discovery as-is.
    assert chain.link.publisher.qos_profile.depth == 1
    assert all(sub.qos_profile.depth == 1 for sub in chain.monitor.subscriptions)
    endpoints = {}
    for kind, topic in chain.topics.items():
        pubs = chain.monitor.get_publishers_info_by_topic(topic)
        subs = chain.monitor.get_subscriptions_info_by_topic(topic)
        endpoints[kind] = {'topic':topic, 'publishers':[p.node_name for p in pubs],
            'subscribers':[p.node_name for p in subs], 'messages':len(chain.messages[kind]),
            'discovered_depth':[int(p.qos_profile.depth) for p in pubs+subs]}
        assert pubs and subs
        assert all(p.qos_profile.reliability == chain.link.qos.reliability for p in pubs+subs)
    print('ROS_EVIDENCE '+json.dumps({'mode':mode, 'domain':42, 'qos':'BEST_EFFORT/KEEP_LAST(1)/VOLATILE',
        'endpoints':endpoints, 'source_seq':4, 'vendor_writes':len(chain.vendor_writes),
        'finite_plant_max_abs_q_rad':float(np.max(np.abs(chain.p.q))),
        'robot_acceptance':False}, sort_keys=True))


def chain_for_apply(chain):
    # Shared helper's feedback hook is intentionally inert: only the actual DDS
    # subscription may update link.latest in this test.
    return SimpleNamespace(link=chain.link, wire=SimpleNamespace(feedback=lambda: None))
