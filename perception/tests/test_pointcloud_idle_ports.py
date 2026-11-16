"""Canvas ports remain stable across lifecycle and metric publishing settings."""
import struct
import types

import pytest
from test_pointcloud_plugin import _plugin, metric_z16_declaration
from vision_stubs import _FakeImage


@pytest.mark.parametrize('inputs', [{}, {'input_topic': ''}, {'input_topic': '/depth'},
                                    {'input_topics': ['/depth']}])
@pytest.mark.parametrize('metric', [False, True])
def test_idle_ports_resolve_only_from_this_cards_input(inputs, metric):
    plugin, executor = _plugin()
    plugin.dispatch('pointcloud', {'action': 'config', 'instance_id': 'idle', 'metric_cloud': metric})
    info = plugin.dispatch('pointcloud', {'action': 'info', 'instance_id': 'idle', **inputs})
    expected = [{'format': 'sensor/pointcloud'}, {'format': 'data/json'},
                {'format': 'sensor/pointcloud2'}]
    if inputs.get('input_topic') or inputs.get('input_topics'):
        for port, suffix in zip(expected, ('pointcloud', 'pointcloud_summary', 'pointcloud_metric')):
            port['topic'] = '/depth/' + suffix
    assert info['topic_out'] == expected
    assert info['state'] == 'idle' and info['instances'] == {}
    assert not executor.nodes and not plugin._nodes
    assert 'camera_info' not in info and 'input_metadata' not in info


def test_idle_card_never_inherits_running_cards_ports_or_calibration():
    plugin, _ = _plugin()
    try:
        plugin.dispatch('pointcloud', {'action': 'start', 'instance_id': 'other', 'input_topic': '/other'})
        info = plugin.dispatch('pointcloud', {'action': 'info', 'instance_id': 'idle', 'input_topic': '/depth'})
        assert info['state'] == 'idle' and info['instances'] == {}
        assert info['topic_out'] == [{'topic': '/depth/pointcloud', 'format': 'sensor/pointcloud'},
                                     {'topic': '/depth/pointcloud_summary', 'format': 'data/json'},
                                     {'topic': '/depth/pointcloud_metric', 'format': 'sensor/pointcloud2'}]
        assert not info['topic_in']
        assert 'camera_info' not in info and 'input_metadata' not in info
        running = plugin.dispatch('pointcloud', {'action': 'info', 'instance_id': 'other'})
        assert running['state'] == 'running'
        assert running['topic_out'][0]['topic'] == '/other/pointcloud'
    finally:
        plugin.shutdown()


@pytest.mark.parametrize('inputs', [{'input_topics': ['/a', '/b']},
                                    {'input_topic': '/a', 'input_topics': ['/b']},
                                    {'input_topic': '/rgb', 'camera_info': {'/rgb': {'format': 'image/jpeg'}}}])
def test_idle_invalid_input_does_not_advertise_guessed_topics(inputs):
    plugin, executor = _plugin()
    info = plugin.dispatch('pointcloud', {'action': 'info', 'instance_id': 'idle', **inputs})
    assert info['state'] == 'error'
    assert [p['format'] for p in info['topic_out']] == [
        'sensor/pointcloud', 'data/json', 'sensor/pointcloud2']
    assert all(not p.get('topic') for p in info['topic_out'])
    assert not executor.nodes


def test_stop_and_rewire_rederive_idle_ports_and_preserve_metric_config():
    plugin, _ = _plugin({'metric_cloud': True})
    plugin.dispatch('pointcloud', {'action': 'config', 'instance_id': 'card', 'metric_cloud': False})
    try:
        plugin.dispatch('pointcloud', {'action': 'start', 'instance_id': 'card', 'input_topic': '/old'})
        plugin.dispatch('pointcloud', {'action': 'stop', 'instance_id': 'card'})
        info = plugin.dispatch('pointcloud', {'action': 'info', 'instance_id': 'card', 'input_topic': '/new'})
        assert [p['topic'] for p in info['topic_out']] == [
            '/new/pointcloud', '/new/pointcloud_summary', '/new/pointcloud_metric']
        empty = plugin.dispatch('pointcloud', {'action': 'info', 'instance_id': 'card', 'input_topic': ''})
        assert len(empty['topic_out']) == 3 and all(not p.get('topic') for p in empty['topic_out'])
        # Visible ports no longer reveal the toggle; starting without metric
        # calibration must still respect this card's override of the global True.
        restarted = plugin.dispatch('pointcloud', {
            'action': 'start', 'instance_id': 'card', 'input_topic': '/new'})
        assert restarted['state'] == 'running'
        running = plugin.dispatch('pointcloud', {'action': 'info', 'instance_id': 'card'})
        assert running['instances']['card']['metric_cloud'] is None
    finally:
        plugin.shutdown()


@pytest.mark.parametrize('instance_id', ['', 'card'])
def test_disabled_metric_port_is_visible_without_publishing_or_metric_metadata(instance_id):
    plugin, executor = _plugin()
    try:
        result = plugin.dispatch('pointcloud', {
            'action': 'start', 'instance_id': 'card', 'input_topic': '/depth',
            'camera_info': [metric_z16_declaration(1)],
            'input_metadata': {'/depth': {
                'format': 'image/depth-z16', 'frame_metadata_topic': '/depth/evidence'}},
        })
        assert result['state'] == 'running'
        info = plugin.dispatch('pointcloud', {'action': 'info', 'instance_id': instance_id})
        assert info['topic_out'] == [
            {'topic': '/depth/pointcloud', 'format': 'sensor/pointcloud'},
            {'topic': '/depth/pointcloud_summary', 'format': 'data/json'},
            {'topic': '/depth/pointcloud_metric', 'format': 'sensor/pointcloud2'},
        ]
        assert info['instances']['card']['metric_cloud'] is None
        assert 'input_metadata' not in info
        assert all(c['format'] != 'sensor/pointcloud2' for c in info.get('camera_info', []))

        node = executor.nodes[0]
        msg = _FakeImage(data=struct.pack('<H', 2000), encoding='16UC1',
                         width=1, height=1, step=2)
        msg.header.frame_id = 'optical'
        msg.header.stamp = types.SimpleNamespace(sec=123, nanosec=456)
        node._emit_depth(msg)
        assert [p.topic for p in node.publishers] == [
            '/depth/pointcloud', '/depth/pointcloud_summary']
        assert all(p.raw_messages for p in node.publishers)
        assert not any(s.topic == '/depth/evidence' for s in node.subscriptions)
    finally:
        plugin.shutdown()


def test_metric_toggle_preserves_ports_across_config_restart():
    plugin, executor = _plugin()
    expected = [
        {'topic': '/depth/pointcloud', 'format': 'sensor/pointcloud'},
        {'topic': '/depth/pointcloud_summary', 'format': 'data/json'},
        {'topic': '/depth/pointcloud_metric', 'format': 'sensor/pointcloud2'},
    ]
    try:
        for enabled in (True, False, True):
            config = plugin.dispatch('pointcloud', {
                'action': 'config', 'instance_id': 'card', 'metric_cloud': enabled})
            assert config['status'] == 'configured'
            idle = plugin.dispatch('pointcloud', {
                'action': 'info', 'instance_id': 'card', 'input_topic': '/depth'})
            assert idle['state'] == 'idle' and idle['topic_out'] == expected
            assert not executor.nodes
            result = plugin.dispatch('pointcloud', {
                'action': 'start', 'instance_id': 'card', 'input_topic': '/depth',
                'camera_info': [metric_z16_declaration(1)],
            })
            assert result['state'] == 'running'
            running = plugin.dispatch('pointcloud', {'action': 'info', 'instance_id': 'card'})
            assert running['topic_out'] == expected
            metric_publishers = [p for p in executor.nodes[0].publishers
                                 if p.topic == '/depth/pointcloud_metric']
            assert len(metric_publishers) == int(enabled)
    finally:
        plugin.shutdown()
