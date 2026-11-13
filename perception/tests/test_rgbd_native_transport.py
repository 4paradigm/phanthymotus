"""The native child's UDP profile must not change other DDS participants."""
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from plugins import rgbd_odometry_runtime as runtime
from test_rgbd_odometry_status import sessions  # noqa: F401


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


def test_invalid_configured_profile_fails_instead_of_losing_network_isolation(monkeypatch, tmp_path):
    profile = tmp_path/'broken.xml'
    profile.write_text('<broken')
    monkeypatch.setenv('FASTRTPS_DEFAULT_PROFILES_FILE', str(profile))
    with pytest.raises(ET.ParseError):
        runtime.estimator_environment(tmp_path)
