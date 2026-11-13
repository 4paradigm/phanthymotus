"""The input participant must receive the private profile before ROS starts."""
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest


@pytest.mark.parametrize('existing,expected', [(131072, 1048576), (4194304, 4194304)])
def test_main_initializes_ros_with_buffered_profile_and_restores_on_failure(monkeypatch, tmp_path,
                                                                          existing, expected):
    import main
    profile = tmp_path/'local.xml'
    original = ('<dds xmlns="http://www.eprosima.com"><profiles><transport_descriptors>'
                '<transport_descriptor><transport_id>local</transport_id><type>UDPv4</type>'
                f'<receiveBufferSize>{existing}</receiveBufferSize>'
                '<interfaceWhiteList><address>127.0.0.1</address></interfaceWhiteList>'
                '</transport_descriptor></transport_descriptors><participant is_default_profile="true">'
                '<rtps><useBuiltinTransports>false</useBuiltinTransports>'
                '<userTransports><transport_id>local</transport_id></userTransports>'
                '</rtps></participant></profiles></dds>')
    profile.write_text(original)
    monkeypatch.setenv('FASTRTPS_DEFAULT_PROFILES_FILE', str(profile))
    monkeypatch.setenv('OMP_NUM_THREADS', '7')
    monkeypatch.setattr(main, '_load_config', lambda: {})
    observed = []
    def init():
        private = Path(os.environ['FASTRTPS_DEFAULT_PROFILES_FILE'])
        assert private != profile
        observed.append(private)
        tree = ET.parse(private)
        assert int(tree.find('.//{*}receiveBufferSize').text) == expected
        assert tree.find('.//{*}interfaceWhiteList/{*}address').text == '127.0.0.1'
        assert tree.find('.//{*}useBuiltinTransports').text == 'false'
        assert tree.find('.//{*}userTransports/{*}transport_id').text == 'local'
        assert os.environ['OMP_NUM_THREADS'] == '7'
        raise RuntimeError('ROS startup failed')
    monkeypatch.setattr(main.rclpy, 'init', init, raising=False)
    with pytest.raises(RuntimeError, match='ROS startup failed'):
        main.main()
    assert observed and not observed[0].exists()
    assert os.environ['FASTRTPS_DEFAULT_PROFILES_FILE'] == str(profile)
    assert profile.read_text() == original


def test_invalid_main_profile_fails_before_ros_init(monkeypatch, tmp_path):
    import main
    profile = tmp_path/'bad.xml'
    profile.write_text('<broken')
    monkeypatch.setenv('FASTRTPS_DEFAULT_PROFILES_FILE', str(profile))
    monkeypatch.setattr(main, '_load_config', lambda: {})
    def init():
        pytest.fail('ROS must not initialize with an invalid configured profile')
    monkeypatch.setattr(main.rclpy, 'init', init, raising=False)
    with pytest.raises(ET.ParseError):
        main.main()
    assert os.environ['FASTRTPS_DEFAULT_PROFILES_FILE'] == str(profile)


def test_main_keeps_profile_through_ros_shutdown(monkeypatch, tmp_path):
    import main
    from types import SimpleNamespace
    profile = tmp_path/'local.xml'
    profile.write_text('<profiles><transport_descriptors><transport_descriptor>'
                       '<type>UDPv4</type></transport_descriptor></transport_descriptors></profiles>')
    monkeypatch.setenv('FASTRTPS_DEFAULT_PROFILES_FILE', str(profile))
    monkeypatch.setattr(main, '_load_config', lambda: {})
    observed = []
    def checkpoint(stage):
        private = Path(os.environ['FASTRTPS_DEFAULT_PROFILES_FILE'])
        assert private != profile and private.exists()
        observed.append((stage, private))
    monkeypatch.setattr(main.rclpy, 'init', lambda: checkpoint('init'), raising=False)
    monkeypatch.setattr(main.rclpy, 'shutdown', lambda: checkpoint('ros_shutdown'), raising=False)
    executor = SimpleNamespace(shutdown=lambda: checkpoint('executor_shutdown'))
    monkeypatch.setattr(main.rclpy.executors, 'MultiThreadedExecutor', lambda: executor)
    monkeypatch.setattr(main, 'PerceptionBundle', lambda *_: SimpleNamespace(
        shutdown=lambda: checkpoint('bundle_shutdown')))
    monkeypatch.setattr(main, '_bundle', None)
    monkeypatch.setattr(main.threading, 'Thread', lambda **_: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(main, '_start_registration', lambda *_: None)
    monkeypatch.setattr(main.signal, 'signal', lambda *_: None)
    monkeypatch.setattr(main, 'ThreadingHTTPServer', lambda *_: SimpleNamespace(
        serve_forever=lambda: checkpoint('serve')))
    main.main()
    assert [stage for stage, _ in observed] == ['init', 'serve', 'bundle_shutdown',
                                               'executor_shutdown', 'ros_shutdown']
    assert len({path for _, path in observed}) == 1
    assert not observed[0][1].exists()
    assert os.environ['FASTRTPS_DEFAULT_PROFILES_FILE'] == str(profile)


@pytest.mark.parametrize('profile_text', [None, '<profiles/>'])
def test_main_without_udp_profile_preserves_environment(monkeypatch, tmp_path, profile_text):
    import main
    key = 'FASTRTPS_DEFAULT_PROFILES_FILE'
    monkeypatch.delenv(key, raising=False)
    original = None
    if profile_text is not None:
        profile = tmp_path/'custom.xml'
        profile.write_text(profile_text)
        original = str(profile)
        monkeypatch.setenv(key, original)
    monkeypatch.setattr(main, '_load_config', lambda: {})
    def init():
        assert os.environ.get(key) == original
        raise RuntimeError('stop at ROS init')
    monkeypatch.setattr(main.rclpy, 'init', init, raising=False)
    with pytest.raises(RuntimeError, match='stop at ROS init'):
        main.main()
    assert os.environ.get(key) == original
