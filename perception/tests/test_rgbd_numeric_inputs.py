"""Image and XYZ consumers admit numerical data independently of provenance."""
from dataclasses import replace
from types import SimpleNamespace as NS
import zlib

import numpy as np
import pytest

from plugins.rgbd_odometry_contract import PairBuffer, validate_inputs
from plugins.rgbd_pairing import RgbdPairer
from plugins.pointcloud_normalization import decode_cloud, normalize_cloud
from plugins.spatial_time import TimePolicy
from plugins import pointcloud as pc
from test_rgbd_odometry_status import sessions  # noqa: F401
from test_rgbd_odometry_contract import declarations
from test_pointcloud_plugin import _plugin, metric_zlib_declaration
from test_pointcloud_slam_ros import cloud
from vision_stubs import _FakeCompressedImage, _FakeImage


def test_rgbd_needs_only_format_dimensions_and_finite_intrinsics():
    data = declarations()
    for entry in data.values():
        for key in ('schema', 'id', 'distortion_model', 'D', 'source'):
            entry.pop(key, None)
        entry['vendor'] = {}
    data['/depth'].update(width=3, height=2, K=[2, .1, 1, 0, 3, 1, 0, 0, 1])
    result = validate_inputs(data, ['/rgb', '/depth'])
    assert result['depth_size'] == (3, 2)
    assert result['image_geometry'].depth.K[0, 0] == 2


@pytest.mark.parametrize('key,bad', [('width', 0), ('height', True), ('K', [1]*8),
    ('K', [float('nan')]*9), ('K', [0]*9)])
def test_rgbd_rejects_uncomputable_numeric_geometry(key, bad):
    data = declarations(); data['/depth'][key] = bad
    with pytest.raises(ValueError):
        validate_inputs(data, ['/rgb', '/depth'])


def test_pairing_rejects_expired_contexts_and_preserves_mismatched_capture_evidence():
    policy = TimePolicy('ingress')
    first = policy.select({}, 1_000_000_000, 10)
    second = replace(first, time_basis='source_capture', observation_stamp_ns=1, normalized_clock_id='other',
                     source_epoch='new', source_capture_id='different', ingress_monotonic_ns=10**12)
    pairer = RgbdPairer(max_wait_ns=1, max_skew_ns=1)
    pairer.push('color', NS(data=b'a'), first)
    assert pairer.push('depth', NS(data=b'b'), second) is None
    assert pairer.dropped_by_side == {'color': 1, 'depth': 0}

    pairer = RgbdPairer(max_wait_ns=1, max_skew_ns=1)
    fresh = replace(second, observation_stamp_ns=1_000_000_001,
                    ingress_monotonic_ns=11, ingress_realtime_ns=1_000_000_002)
    pairer.push('color', NS(data=b'c'), first)
    pair = pairer.push('depth', NS(data=b'd'), fresh)
    assert pair.color_context is first and pair.depth_context is fresh
    assert pair.method == 'receipt_estimated'
    assert pair[0][0] == 1_000_000_000 and pair[1][0] == 1_000_000_001
    assert pair.depth_context.observation_stamp_ns == 1_000_000_001


def test_legacy_pair_buffer_preserves_numeric_timestamp_behavior():
    legacy = PairBuffer(max_skew_ns=1)
    legacy.push('color', 0, 'color', 10**12)
    assert legacy.push('depth', -10, 'too_far', 1) is None
    assert legacy.push('depth', -1, 'depth', 1) == ((0, 'color'), (-1, 'depth'))


def test_cloud_without_frame_or_positive_time_still_normalizes_xyz():
    source = cloud(); source.header.frame_id = ''; source.header.stamp.sec = 0
    source.header.stamp.nanosec = 0
    stamp, frame, xyz = decode_cloud(source, max_points=100)
    assert stamp == 0 and frame == ''
    np.testing.assert_array_equal(xyz, [[1, 2, 3]])
    result = normalize_cloud(source, None, '0'*64)
    assert not result.source_digest_verified
    assert result.message.header.frame_id == ''


def test_metric_projection_accepts_unlabelled_intrinsics_and_zero_header():
    plugin, executor = _plugin({'metric_cloud': True})
    declaration = dict(format='image/depth-f32', width=2, height=1,
                       K=[2, .2, 0, 0, 2, 0, 0, 0, 1])
    try:
        assert plugin.dispatch('start', {'input_topic': '/depth',
            'camera_info': {'/depth': declaration}})['state'] == 'running'
        node = executor.nodes[0]
        msg = _FakeImage(data=np.array([2., 2.], dtype='<f4').tobytes(),
                         encoding='32FC1', width=2, height=1, step=8)
        msg.header.stamp = NS(sec=0, nanosec=0)
        node._emit_depth(msg)
        pub = next(p for p in node.publishers if p.topic.endswith('/pointcloud_metric'))
        np.testing.assert_allclose(np.frombuffer(pub.raw_messages[0].data, '<f4').reshape(-1, 3),
                                   [[0, 0, 2], [1, 0, 2]])
        assert pub.raw_messages[0].header.frame_id == ''
    finally:
        plugin.shutdown()


def test_zlib_projection_uses_declared_dimensions_and_rejects_wrong_length():
    plugin, executor = _plugin({'metric_cloud': True})
    declaration = dict(metric_zlib_declaration(), width=3, height=2)
    try:
        assert plugin.dispatch('start', {'input_topic': '/depth',
            'camera_info': {'/depth': declaration}})['state'] == 'running'
        node = executor.nodes[0]
        msg = _FakeCompressedImage(zlib.compress(np.full((2, 3), 1000, dtype='<u2').tobytes()), fmt='depth-zlib')
        msg.header.stamp = NS(sec=0, nanosec=0)
        node._emit_depth(msg)
        pub = next(p for p in node.publishers if p.topic.endswith('/pointcloud_metric'))
        assert pub.raw_messages[0].width == 6
        msg.data = zlib.compress(bytes(10))
        node._emit_depth(msg)
        assert len(pub.raw_messages) == 1 and node._metric_status()['last_skip_reason'] == 'invalid_depth'
    finally:
        plugin.shutdown()


def test_image_callback_accepts_different_frame_and_old_source_time(sessions):
    create, now, _ = sessions
    session = create()
    for kind, stamp in (('color', 5), ('depth', 0)):
        msg = NS(header=NS(frame_id='producer-frame', stamp=NS(sec=0, nanosec=stamp)), data=b'buffer')
        session._image(kind, msg)
    assert session.pending.qsize() == 1
    pair = session.pending.get_nowait()
    assert pair[0][0] == 5 and pair[1][0] == 0
    assert session._initialization_info(0)['phase'] == 'waiting_visual'
    assert session.has_fresh_inputs(now[0])
    assert session.gate.state != 'lost'


@pytest.mark.parametrize('message,context', [(NS(), NS(observation_stamp_ns=1)),
    (NS(data='text'), NS(observation_stamp_ns=1)), (NS(data=b'x'), None)])
def test_fifo_rejects_malformed_message_and_context(message, context):
    pairer = RgbdPairer()
    with pytest.raises(ValueError):
        pairer.push('color', message, context)
