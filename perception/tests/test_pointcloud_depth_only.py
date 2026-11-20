"""The pointcloud card consumes exactly one depth stream, never RGB."""
import pytest

from test_pointcloud_plugin import _plugin
import plugins.pointcloud as pc


def test_card_only_advertises_depth_inputs():
    tool = pc.TOOLS[0]
    assert {port['format'] for port in tool['topic_in']} == {'image/depth-z16', 'image/depth-zlib'}
    assert 'mode' not in tool['inputSchema']['properties']
    assert tool['inputSchema']['properties']['input_topics']['maxItems'] == 1


@pytest.mark.parametrize('args', [
    {'input_topic': '/rgb', 'mode': 'mono'},
    {'input_topics': ['/left', '/right'], 'mode': 'stereo'},
    {'input_topics': ['/a', '/b']},
    {'input_topic': '/a', 'input_topics': ['/b']},
    {'input_topic': '/a', 'input_topics': ['/a', '/b']},
    {'input_topics': '/depth'},
    {'input_topics': [None]},
    {'input_topic': '/rgb', 'camera_info': {'/rgb': {'format': 'image/jpeg'}}},
    {'input_topic': '/rgb', 'input_metadata': {'/rgb': {'format': 'image/jpeg'}}},
])
def test_non_depth_or_multiple_inputs_fail_before_creating_a_node(args):
    plugin, executor = _plugin()
    try:
        result = plugin.dispatch('pointcloud', {'action': 'start', **args})
        assert result['state'] == 'error'
        assert executor.nodes == [] and plugin._nodes == {}
    finally:
        plugin.shutdown()


@pytest.mark.parametrize('fmt,callback', [
    ('image/depth-z16', '_depth_image_cb'), ('image/depth-zlib', '_depth_comp_cb'),
])
@pytest.mark.parametrize('mode', [None, 'depth', 'auto'])
def test_single_depth_input_starts_without_model_loading(fmt, callback, mode):
    plugin, executor = _plugin()
    args = {'action': 'start', 'input_topics': ['/depth'],
            'camera_info': {'/depth': {'format': fmt}}}
    if mode is not None:
        args['mode'] = mode
    try:
        result = plugin.dispatch('pointcloud', args)
        assert result['state'] == 'running' and result['mode'] == 'depth'
        assert [s.callback.__name__ for s in executor.nodes[0].subscriptions] == [callback]
        info = plugin.dispatch('info', {})
        assert info['topic_in'] == [{'topic': '/depth', 'format': fmt}]
    finally:
        plugin.shutdown()


def test_compressed_rgb_is_not_reinterpreted_as_depth():
    import zlib
    from vision_stubs import _FakeCompressedImage
    # Even a decodable, depth-sized buffer must not override its RGB encoding.
    msg = _FakeCompressedImage(zlib.compress(bytes(640 * 480 * 2)), fmt='jpeg')
    depth, _ = pc._decode_depth_message(msg)
    assert depth is None
