"""Canvas ports remain stable across the card lifecycle."""
import struct
import types

import pytest
from test_pointcloud_plugin import _plugin, metric_f32_declaration
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


def test_stop_and_rewire_rederive_idle_ports_and_ignore_legacy_metric_config():
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
        restarted = plugin.dispatch('pointcloud', {
            'action': 'start', 'instance_id': 'card', 'input_topic': '/new'})
        assert restarted['state'] == 'running'
        running = plugin.dispatch('pointcloud', {'action': 'info', 'instance_id': 'card'})
        assert running['instances']['card']['metric_cloud']['published'] == 0
    finally:
        plugin.shutdown()


@pytest.mark.parametrize('instance_id', ['', 'card'])
def test_metric_cloud_and_metadata_are_enabled_without_opt_in(instance_id):
    plugin, executor = _plugin()
    try:
        result = plugin.dispatch('pointcloud', {
            'action': 'start', 'instance_id': 'card', 'input_topic': '/depth',
            'camera_info': [metric_f32_declaration(1)],
            'input_metadata': {'/depth': {
                'format': 'image/depth-f32', 'frame_metadata_topic': '/depth/evidence'}},
        })
        assert result['state'] == 'running'
        info = plugin.dispatch('pointcloud', {'action': 'info', 'instance_id': instance_id})
        assert info['topic_out'] == [
            {'topic': '/depth/pointcloud', 'format': 'sensor/pointcloud'},
            {'topic': '/depth/pointcloud_summary', 'format': 'data/json'},
            {'topic': '/depth/pointcloud_metric', 'format': 'sensor/pointcloud2'},
        ]
        assert info['instances']['card']['metric_cloud']['published'] == 0
        assert info['input_metadata'][0]['frame_metadata_topic'] == '/depth/pointcloud_metric/metadata'
        assert any(c['format'] == 'sensor/pointcloud2' for c in info['camera_info'])

        node = executor.nodes[0]
        msg = _FakeImage(data=struct.pack('<f', 2.), encoding='32FC1',
                         width=1, height=1, step=4)
        msg.header.frame_id = 'optical'
        msg.header.stamp = types.SimpleNamespace(sec=123, nanosec=456)
        node._emit_depth(msg)
        assert {p.topic for p in node.publishers if p.raw_messages} == {
            '/depth/pointcloud', '/depth/pointcloud_summary', '/depth/pointcloud_metric'}
        assert any(s.topic == '/depth/evidence' for s in node.subscriptions)
    finally:
        plugin.shutdown()


@pytest.mark.parametrize('new_fx', [3.0, None])
def test_running_metric_output_rejects_changed_calibration(new_fx):
    plugin, executor = _plugin()
    original = metric_f32_declaration(1)
    revised = {**original, 'K': [new_fx, *original['K'][1:]] if new_fx is not None else None}
    start = {'instance_id': 'card', 'input_topic': '/depth'}
    try:
        assert plugin.dispatch('start', {**start, 'camera_info': [original]})['state'] == 'running'
        node = executor.nodes[0]
        result = plugin.dispatch('start', {**start, 'camera_info': [revised]})
        assert result['state'] == 'error'
        assert executor.nodes == [node]
        info = plugin.dispatch('info', {'instance_id': 'card'})
        metric = next(c for c in info['camera_info'] if c['format'] == 'sensor/pointcloud2')
        assert metric['K'] == original['K']
    finally:
        plugin.shutdown()


def test_legacy_metric_toggle_cannot_disable_publishing_or_restart_card():
    plugin, executor = _plugin({'metric_cloud': False})
    expected = [
        {'topic': '/depth/pointcloud', 'format': 'sensor/pointcloud'},
        {'topic': '/depth/pointcloud_summary', 'format': 'data/json'},
        {'topic': '/depth/pointcloud_metric', 'format': 'sensor/pointcloud2'},
    ]
    try:
        assert 'metric_cloud' not in plugin.get_tools()[0]['configSchema']['properties']
        for enabled in (False, True):
            config = plugin.dispatch('pointcloud', {
                'action': 'config', 'instance_id': 'card', 'metric_cloud': enabled})
            assert config['status'] == 'configured'
            assert 'metric_cloud' not in config['config']
            idle = plugin.dispatch('pointcloud', {
                'action': 'info', 'instance_id': 'card', 'input_topic': '/depth'})
            assert idle['state'] == 'idle' and idle['topic_out'] == expected
            assert not executor.nodes
            result = plugin.dispatch('pointcloud', {
                'action': 'start', 'instance_id': 'card', 'input_topic': '/depth',
                'camera_info': [metric_f32_declaration(1)],
            })
            assert result['state'] == 'running'
            running = plugin.dispatch('pointcloud', {'action': 'info', 'instance_id': 'card'})
            assert running['topic_out'] == expected
            node = executor.nodes[0]
            metric_publishers = [p for p in node.publishers
                                 if p.topic == '/depth/pointcloud_metric']
            assert len(metric_publishers) == 1
            plugin.dispatch('pointcloud', {
                'action': 'config', 'instance_id': 'card', 'metric_cloud': not enabled})
            assert executor.nodes == [node]
            msg = _FakeImage(data=struct.pack('<f', 2.), encoding='32FC1',
                             width=1, height=1, step=4)
            msg.header.stamp = types.SimpleNamespace(sec=123, nanosec=456)
            node._emit_depth(msg)
            assert len(metric_publishers[0].raw_messages) == 1
            plugin.dispatch('pointcloud', {'action': 'stop', 'instance_id': 'card'})
    finally:
        plugin.shutdown()
