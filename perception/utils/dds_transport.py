"""Process-private DDS receive buffers for fragmented camera messages."""
from contextlib import contextmanager
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET


def receive_environment(directory):
    """Copy an explicit UDP profile without changing transport isolation."""
    env = dict(os.environ)
    profile = env.get('FASTRTPS_DEFAULT_PROFILES_FILE')
    if not profile:
        return env
    tree = ET.parse(profile)
    changed = False
    for transport in tree.findall('.//{*}transport_descriptor'):
        kind = transport.find('{*}type')
        if kind is None or kind.text != 'UDPv4':
            continue
        size = transport.find('{*}receiveBufferSize')
        if size is None:
            prefix = transport.tag.rpartition('}')[0] + '}' if '}' in transport.tag else ''
            size = ET.SubElement(transport, prefix+'receiveBufferSize')
        size.text = str(max(1048576, int(size.text or '0')))
        changed = True
    if changed:
        path = Path(directory)/'native-dds.xml'
        ET.register_namespace('', 'http://www.eprosima.com')
        tree.write(path, encoding='unicode')
        env['FASTRTPS_DEFAULT_PROFILES_FILE'] = str(path)
    return env


@contextmanager
def receive_buffer_profile():
    """Keep the profile alive until every participant in this process stops."""
    key = 'FASTRTPS_DEFAULT_PROFILES_FILE'
    original = os.environ.get(key)
    with TemporaryDirectory(prefix='motus-dds-') as directory:
        env = receive_environment(directory)
        try:
            if key in env:
                os.environ[key] = env[key]
            yield
        finally:
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original
