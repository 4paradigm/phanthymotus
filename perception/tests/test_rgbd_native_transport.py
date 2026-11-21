"""The native child's UDP profile must not change other DDS participants."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from plugins import rgbd_odometry_runtime as runtime
from rclpy.qos import ReliabilityPolicy
from test_rgbd_odometry_status import sessions  # noqa: F401
from vision_stubs import _FakeNode


def test_private_rgbd_uses_bounded_reliable_reader_and_writer(sessions, monkeypatch):
    create, _, _ = sessions
    profiles = {}
    create_publisher = _FakeNode.create_publisher

    def capture(node, message_type, topic, qos):
        profiles[topic] = qos
        return create_publisher(node, message_type, topic, qos)

    monkeypatch.setattr(_FakeNode, 'create_publisher', capture)
    session = create()
    params = runtime.estimator_parameters(session.calibration, session.segment)
    writer = profiles[session.rgbd.topic]
    assert params['qos'] == 1
    assert writer.reliability == ReliabilityPolicy.RELIABLE
    assert params['topic_queue_size'] == writer.depth == 2
    assert params['qos_imu'] == 2
    assert profiles[session.publisher.topic] is not writer


@pytest.mark.parametrize('existing,expected', [('', 1048576),
    ('<receiveBufferSize>4194304</receiveBufferSize>', 4194304)])
def test_native_child_uses_private_buffer_profile_and_preserves_isolation(sessions, monkeypatch, tmp_path,
                                                                         existing, expected):
    create, _, _ = sessions
    profile = tmp_path/'local.xml'
    original = ('<dds xmlns="http://www.eprosima.com"><profiles><transport_descriptors>'
        '<transport_descriptor><transport_id>lo_only</transport_id><type>UDPv4</type>'
        '<interfaceWhiteList><address>127.0.0.1</address></interfaceWhiteList>' + existing +
        '</transport_descriptor></transport_descriptors><participant is_default_profile="true">'
        '<rtps><userTransports><transport_id>lo_only</transport_id></userTransports>'
        '<useBuiltinTransports>false</useBuiltinTransports></rtps></participant></profiles></dds>')
    profile.write_text(original)
    monkeypatch.setenv('FASTRTPS_DEFAULT_PROFILES_FILE', str(profile))
    popen = runtime.subprocess.Popen
    children = []
    def capture(*args, **kwargs):
        children.append(kwargs['env'])
        return popen(*args, **kwargs)
    monkeypatch.setattr(runtime.subprocess, 'Popen', capture)
    session = create()
    child_profile = Path(children[0]['FASTRTPS_DEFAULT_PROFILES_FILE'])
    assert child_profile != profile and child_profile.parent == Path(session.directory.name)
    # Match the default-namespace XML form accepted in the native A/B trial.
    assert child_profile.read_text().startswith('<dds xmlns="http://www.eprosima.com"')
    tree = ET.parse(child_profile)
    assert int(tree.find('.//{*}receiveBufferSize').text) == expected
    assert tree.find('.//{*}interfaceWhiteList/{*}address').text == '127.0.0.1'
    assert tree.find('.//{*}useBuiltinTransports').text == 'false'
    assert tree.find('.//{*}userTransports/{*}transport_id').text == 'lo_only'
    assert profile.read_text() == original
    assert os.environ['FASTRTPS_DEFAULT_PROFILES_FILE'] == str(profile)
    session.process.returncode = 0
    session.close()
    assert not child_profile.exists() and profile.exists()


def test_absent_or_non_udp_profile_does_not_invent_transport_settings(monkeypatch, tmp_path):
    monkeypatch.delenv('FASTRTPS_DEFAULT_PROFILES_FILE', raising=False)
    assert 'FASTRTPS_DEFAULT_PROFILES_FILE' not in runtime.estimator_environment(tmp_path)
    profile = tmp_path/'custom.xml'
    profile.write_text('<profiles><transport_descriptors><transport_descriptor>'
                       '<type>SHM</type></transport_descriptor></transport_descriptors></profiles>')
    monkeypatch.setenv('FASTRTPS_DEFAULT_PROFILES_FILE', str(profile))
    assert runtime.estimator_environment(tmp_path)['FASTRTPS_DEFAULT_PROFILES_FILE'] == str(profile)
    assert not (tmp_path/'native-dds.xml').exists()


def test_local_fastdds_child_gets_loopback_udp_receive_buffer(monkeypatch, tmp_path):
    monkeypatch.delenv('FASTRTPS_DEFAULT_PROFILES_FILE', raising=False)
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp')
    env = runtime.estimator_environment(tmp_path)
    tree = ET.parse(env['FASTRTPS_DEFAULT_PROFILES_FILE'])
    assert tree.find('.//{*}type').text == 'UDPv4'
    assert tree.find('.//{*}receiveBufferSize').text == '1048576'
    assert [item.text for item in tree.findall('.//{*}interfaceWhiteList/{*}address')] == ['127.0.0.1']
    assert tree.find('.//{*}useBuiltinTransports').text == 'false'
    assert env['ROS_LOCALHOST_ONLY'] == '0' and os.environ['ROS_LOCALHOST_ONLY'] == '1'
    assert 'FASTRTPS_DEFAULT_PROFILES_FILE' not in os.environ


@pytest.mark.parametrize('explicit', [True, False])
def test_local_cyclonedds_child_keeps_existing_isolation(monkeypatch, tmp_path, explicit):
    monkeypatch.delenv('FASTRTPS_DEFAULT_PROFILES_FILE', raising=False)
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    if explicit:
        monkeypatch.setenv('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    else:
        monkeypatch.delenv('RMW_IMPLEMENTATION', raising=False)
        monkeypatch.setitem(sys.modules, 'rclpy.utilities', SimpleNamespace(
            get_rmw_implementation_identifier=lambda: 'rmw_cyclonedds_cpp'))
    env = runtime.estimator_environment(tmp_path)
    assert env['ROS_LOCALHOST_ONLY'] == '1'
    assert 'FASTRTPS_DEFAULT_PROFILES_FILE' not in env


def test_invalid_configured_profile_fails_instead_of_losing_network_isolation(monkeypatch, tmp_path):
    profile = tmp_path/'broken.xml'
    profile.write_text('<broken')
    monkeypatch.setenv('FASTRTPS_DEFAULT_PROFILES_FILE', str(profile))
    with pytest.raises(ET.ParseError):
        runtime.estimator_environment(tmp_path)
