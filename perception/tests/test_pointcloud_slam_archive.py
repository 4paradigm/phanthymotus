"""Persist bounded SLAM snapshots without restoring tracking."""
import numpy as np
import pytest
from plugins.pointcloud_slam_archive import save_archive, load_archive
from test_pointcloud_slam_segment_merge import make_two_segments


def settings():
    return {'cloud_frame': 'camera', 'child_frame': 'body', 'clock_id': 'clock',
            'child_from_cloud': np.eye(4).tolist()}


def test_roundtrip_preserves_local_clouds_poses_and_constraints(tmp_path):
    stream = make_two_segments()
    segments = stream.store.segments
    link = {'target': segments[0].segment_id, 'source': segments[1].segment_id,
            'target_index': 0, 'source_index': 0, 'transform': np.eye(4).tolist()}
    path = tmp_path / 'map.json'
    save_archive(path, stream.store, [link], {}, settings())
    store, links, loops = load_archive(path, settings(), max_frames=10, max_points=4000)
    assert store.requires_new_segment
    assert links == [link] and loops == {}
    assert [s.segment_id for s in store.segments] == [s.segment_id for s in segments]
    for actual, expected in zip(store.segments, segments):
        for a, b in zip(actual.frames, expected.frames):
            assert a.pose == b.pose and a.cloud_stamp_ns == b.cloud_stamp_ns
            np.testing.assert_array_equal(a.points, b.points)


def test_restore_rejects_changed_calibration_and_capacity(tmp_path):
    stream = make_two_segments(); path = tmp_path / 'map.json'
    save_archive(path, stream.store, [], {}, settings())
    with pytest.raises(ValueError, match='configuration'):
        load_archive(path, {**settings(), 'cloud_frame': 'other'}, max_frames=10, max_points=4000)
    with pytest.raises(ValueError):
        load_archive(path, settings(), max_frames=2, max_points=4000)


def test_restore_rejects_malformed_archive(tmp_path):
    path = tmp_path / 'map.json'; path.write_text('{"schema":"bad"}')
    with pytest.raises(ValueError):
        load_archive(path, settings(), max_frames=10, max_points=4000)


def test_restore_rejects_wrong_child_frame_even_if_manifest_is_unchanged(tmp_path):
    import json
    stream = make_two_segments(); path = tmp_path / 'map.json'
    save_archive(path, stream.store, [], {}, settings())
    value = json.loads(path.read_text())
    for frame in value['frames']:
        frame['pose']['child_frame_id'] = 'other'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='frame'):
        load_archive(path, settings(), max_frames=10, max_points=4000)


def test_failed_save_preserves_existing_archive(tmp_path):
    stream = make_two_segments(); path = tmp_path / 'map.json'
    save_archive(path, stream.store, [], {}, settings())
    original = path.read_bytes()
    with pytest.raises(ValueError):
        save_archive(path, stream.store, [], {'a': [{'transform': float('nan')}]}, settings())
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('.slam-*.tmp'))


def test_restore_rejects_geometry_configuration_change(tmp_path):
    stream = make_two_segments(); path = tmp_path / 'map.json'
    save_archive(path, stream.store, [], {}, {**settings(), 'preview_geometry': {'voxel_m': .1}})
    with pytest.raises(ValueError, match='configuration'):
        load_archive(path, {**settings(), 'preview_geometry': {'voxel_m': .2}}, max_frames=10, max_points=4000)
