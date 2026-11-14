"""Depth projection preserves source evidence and binds sidecars to payloads."""
import copy
import hashlib
import json
import types

import numpy as np
import pytest

from vision_stubs import _FakeExecutor, _FakeImage
from plugins.pointcloud import PointCloudPerceptionPlugin


def _declaration(topic='/depth', **overrides):
    return {
        'schema': 'motus.slam_input/1', 'topic': topic, 'kind': 'depth',
        'format': 'image/depth-z16', 'source_id': 'depth-camera',
        'length_unit': 'mm', 'frame_id': 'optical', 'reference_point': None,
        'clock_id': 'ros', 'stamp_source': 'measurement',
        'observation_model': 'pinhole', 'cloud_semantics': 'depth_frame',
        'timing_model': 'instantaneous', 'frame_metadata_topic': '/capture/evidence',
        'clock': {'mapping': 'capture-offset', 'verified': False, 'uncertainty_ns': None},
        'extrinsic': {'source': 'axis_convention', 'calibrated': False,
                      'parent_frame': 'base', 'child_frame': 'optical'},
        'vendor': {'unknown_fact': None}, **overrides,
    }


def _camera(topic='/depth'):
    return {'schema': 'motus.camera/1', 'topic': topic, 'format': 'image/depth-z16',
            'id': 'optical', 'source': 'derived-from-K', 'distortion_model': 'pinhole',
            'width': 1, 'height': 1, 'K': [2., 0., 0., 0., 2., 0., 0., 0., 1.],
            'vendor': {'depth_unit': 'mm', 'depth_source': 'measured',
                       'timestamp_semantics': 'source_capture_mapped'}}


def test_input_metadata_is_not_a_manual_pointcloud_card_parameter():
    # Canvas forwards upstream declarations internally; users should not see
    # or edit the input_metadata field in the pointcloud card form.
    tool = PointCloudPerceptionPlugin({'metric_cloud': True}, 'testns', _FakeExecutor()).get_tools()[0]
    schema = tool['inputSchema']
    assert 'input_metadata' not in schema['properties']
    assert all('input_metadata' not in item['params']
               for item in schema['x-action-params'].values())


def test_metric_rejects_conflicting_camera_and_input_transport_declarations():
    executor = _FakeExecutor()
    plugin = PointCloudPerceptionPlugin({'metric_cloud': True}, 'testns', executor)
    result = plugin.dispatch('start', {'input_topic': '/depth', 'camera_info': [_camera()],
        'input_metadata': {'/depth': _declaration(format='image/depth-zlib')}})
    assert result['state'] == 'error'
    assert not executor.nodes


def _running(declaration=None, **args):
    plugin = PointCloudPerceptionPlugin({'metric_cloud': True},
        'testns', _FakeExecutor())
    result = plugin.dispatch('pointcloud', {
        'action': 'start', 'instance_id': 'depth-card', 'input_topic': '/depth', 'camera_info': [_camera()],
        'input_metadata': {'/depth': declaration or _declaration()}, **args})
    assert result['state'] == 'running'
    return plugin, plugin._executor.nodes[0]


def _depth(value=2., stamp_ns=123_000_000_456, frame='optical'):
    msg = _FakeImage(data=np.array([round(value * 1000)], dtype='<u2').tobytes(),
                     encoding='16UC1', width=1, height=1, step=2)
    msg.header.frame_id = frame
    msg.header.stamp = types.SimpleNamespace(
        sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)
    return msg


def _packet(msg, **overrides):
    return {
        'schema': 'motus.slam_frame/1',
        'stamp_ns': msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec,
        'frame_id': msg.header.frame_id, 'clock_id': 'ros',
        'clock_epoch': 'epoch-1', 'source_generation': 'generation-1',
        'source_stamp': 2468., 'source_unit': 'ms', 'source_clock': 'camera',
        'mapping': {'method': 'arrival_offset', 'version': 'capture/1',
                    'offset_ns': 900, 'verified': False, 'uncertainty_ns': None},
        'data_sha256': hashlib.sha256(msg.data).hexdigest(),
        'vendor': {'unverified_tf': None}, **overrides,
    }


def _sidecar(node):
    return next(sub.callback for sub in node.subscriptions
                if sub.topic == '/capture/evidence')


def _published(node, suffix):
    return next(pub for pub in node.publishers if pub.topic == '/depth/pointcloud_metric' + suffix)


def test_metric_declaration_inherits_unknown_evidence_and_changes_only_projection_fields():
    source = _declaration(length_unit='mm')
    original = copy.deepcopy(source)
    plugin, node = _running(source)
    try:
        info = plugin.dispatch('pointcloud', {'action': 'info', 'instance_id': 'depth-card'})
        assert info['input_metadata'] == [{
            **original, 'topic': '/depth/pointcloud_metric', 'kind': 'cloud',
            'format': 'sensor/pointcloud2', 'length_unit': 'm',
            'frame_metadata_topic': '/depth/pointcloud_metric/metadata',
        }]
        assert source == original
        info['input_metadata'][0]['clock']['verified'] = True
        assert plugin.dispatch('pointcloud', {'action': 'info', 'instance_id': 'depth-card'})[
            'input_metadata'][0]['clock']['verified'] is False
        assert any(sub.topic == '/capture/evidence' for sub in node.subscriptions)
    finally:
        plugin.shutdown()


@pytest.mark.parametrize('sidecar_first', [True, False])
def test_frame_metadata_pairs_by_exact_stamp_frame_and_payload_in_either_arrival_order(sidecar_first):
    plugin, node = _running()
    try:
        msg = _depth()
        packet = _packet(msg)
        callback = _sidecar(node)
        pub = _published(node, '/metadata')
        if sidecar_first:
            callback(types.SimpleNamespace(data=json.dumps(packet)))
            assert not pub.messages
        node._emit_depth(msg)
        if not sidecar_first:
            assert not pub.messages
            callback(types.SimpleNamespace(data=json.dumps(packet)))
        cloud = _published(node, '').raw_messages[0]
        assert [json.loads(item) for item in pub.messages] == [{
            **packet, 'data_sha256': hashlib.sha256(cloud.data).hexdigest()}]
        assert (cloud.header.stamp.sec, cloud.header.stamp.nanosec) == (123, 456)
    finally:
        plugin.shutdown()


@pytest.mark.parametrize('mismatch', [
    {'stamp_ns': 123_000_000_457}, {'frame_id': 'other'},
    {'data_sha256': '0' * 64}, {'data_sha256': None},
])
def test_wrong_frame_stamp_or_payload_has_no_sidecar(mismatch):
    plugin, node = _running()
    try:
        msg = _depth()
        _sidecar(node)(types.SimpleNamespace(data=json.dumps(_packet(msg, **mismatch))))
        node._emit_depth(msg)
        assert _published(node, '').messages
        assert not _published(node, '/metadata').messages
    finally:
        plugin.shutdown()


def test_same_stamp_old_payload_cannot_receive_new_generation_evidence():
    plugin, node = _running()
    try:
        old, new = _depth(2.), _depth(3.)
        old_packet = _packet(old)
        new_packet = _packet(new, clock_epoch='epoch-2', source_generation='generation-2')
        callback = _sidecar(node)
        callback(types.SimpleNamespace(data=json.dumps(new_packet)))
        node._emit_depth(old)
        assert not _published(node, '/metadata').messages
        node._emit_depth(new)
        callback(types.SimpleNamespace(data=json.dumps(old_packet)))
        clouds = _published(node, '').raw_messages
        assert [json.loads(item) for item in _published(node, '/metadata').messages] == [
            {**new_packet, 'data_sha256': hashlib.sha256(clouds[1].data).hexdigest()},
            {**old_packet, 'data_sha256': hashlib.sha256(clouds[0].data).hexdigest()},
        ]
    finally:
        plugin.shutdown()


@pytest.mark.parametrize('sidecar_first', [True, False])
def test_pending_metadata_and_clouds_evict_oldest_after_ten(sidecar_first):
    plugin, node = _running()
    try:
        messages = [_depth(stamp_ns=123_000_000_000 + i) for i in range(11)]
        callback = _sidecar(node)
        for msg in messages:
            if sidecar_first:
                callback(types.SimpleNamespace(data=json.dumps(_packet(msg))))
            else:
                node._emit_depth(msg)
        if sidecar_first:
            node._emit_depth(messages[0])
        else:
            callback(types.SimpleNamespace(data=json.dumps(_packet(messages[0]))))
        assert not _published(node, '/metadata').messages
        if sidecar_first:
            node._emit_depth(messages[-1])
        else:
            callback(types.SimpleNamespace(data=json.dumps(_packet(messages[-1]))))
        assert len(_published(node, '/metadata').messages) == 1
    finally:
        plugin.shutdown()


def test_sidecar_subscription_is_owned_by_metric_node_lifecycle():
    plugin, node = _running()
    callback = _sidecar(node)
    packet = _packet(_depth())
    plugin.dispatch('pointcloud', {'action': 'stop', 'instance_id': 'depth-card'})
    assert node.destroyed
    assert not plugin._executor.nodes
    callback(types.SimpleNamespace(data=json.dumps(packet)))
    assert not _published(node, '/metadata').messages
    assert 'input_metadata' not in plugin.dispatch('pointcloud', {'action': 'info'})


def test_running_instance_refuses_metadata_change_until_stopped():
    plugin, node = _running()
    try:
        changed = _declaration(frame_metadata_topic='/other/evidence', source_id='other-camera')
        result = plugin.dispatch('pointcloud', {
            'action': 'start', 'instance_id': 'depth-card', 'input_topic': '/depth', 'camera_info': [_camera()],
            'input_metadata': {'/depth': changed}})
        assert result['state'] == 'error'
        assert 'stop' in result['message']
        info = plugin.dispatch('pointcloud', {'action': 'info', 'instance_id': 'depth-card'})
        assert info['input_metadata'][0]['source_id'] == 'depth-camera'
        assert any(sub.topic == '/capture/evidence' for sub in node.subscriptions)
    finally:
        plugin.shutdown()


def test_multi_instance_info_selects_only_requested_output_metadata():
    plugin, first = _running()
    try:
        other = _declaration('/other', source_id='other-camera', frame_metadata_topic='/other/meta')
        result = plugin.dispatch('pointcloud', {
            'action': 'start', 'instance_id': 'other-card', 'input_topic': '/other', 'camera_info': [_camera('/other')],
            'input_metadata': {'/depth': _declaration(), '/other': other}})
        assert result['state'] == 'running'
        assert [entry['source_id'] for entry in plugin.dispatch('pointcloud', {
            'action': 'info', 'instance_id': 'other-card'})['input_metadata']] == ['other-camera']
        assert {entry['topic'] for entry in plugin.dispatch('pointcloud', {
            'action': 'info'})['input_metadata']} == {'/depth/pointcloud_metric', '/other/pointcloud_metric'}
    finally:
        plugin.shutdown()


@pytest.mark.parametrize('metadata', [None, [], 'bad-shape',
                                      {'/depth': None}, {'/depth': []}, {'/depth': 'bad-shape'},
                                      {'/depth': {}}])
def test_explicit_malformed_metadata_shape_is_rejected_without_starting(metadata):
    executor = _FakeExecutor()
    plugin = PointCloudPerceptionPlugin({'metric_cloud': True},
        'testns', executor)
    result = plugin.dispatch('pointcloud', {
        'action': 'start', 'input_topic': '/depth', 'camera_info': [_camera()], 'input_metadata': metadata})
    assert result['state'] == 'error'
    assert 'input_metadata' in result['message']
    assert not executor.nodes


@pytest.mark.parametrize('metadata', [{}, {'/sibling': {}}])
def test_missing_sidecar_declaration_keeps_driver_calibrated_metric_start(metadata):
    plugin, node = _running(input_metadata=metadata)
    try:
        assert 'input_metadata' not in plugin.dispatch('pointcloud', {'action': 'info'})
        assert all(sub.topic != '/capture/evidence' for sub in node.subscriptions)
    finally:
        plugin.shutdown()
