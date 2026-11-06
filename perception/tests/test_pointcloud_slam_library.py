import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from plugins import pointcloud_slam_library as library
from plugins.pointcloud_slam_paged_archive import load_paged_archive
from test_pointcloud_slam_paged_store import add, store


@pytest.fixture
def saved(tmp_path):
    frames = store(tmp_path / 'sessions')
    add(frames, 0)
    cfg = {'cloud_frame': 'camera', 'child_frame': 'body', 'clock_id': 'clock',
           'child_from_cloud': np.eye(4).tolist(),
           'persistent_history': {'max_disk_bytes': 10_000_000, 'reserve_bytes': 0}}

    def save(name='地图', bound=None, request='one'):
        return library.save_revision(tmp_path, frames.records, {}, cfg, name, bound, request)[0]

    return save(), save, cfg


def test_tags_bound_to_revision_and_segment(tmp_path, saved):
    map_id, save, _ = saved
    assert library.list_tags(tmp_path, map_id, 1) == []
    assert library.revision_segments(tmp_path, map_id, 1) == ['a']
    tag = library.create_tag(tmp_path, map_id, 1, ' 前台 ', 0, 2, segment_id='a')
    assert tag['name'] == '前台' and tag['yaw'] is None
    save(bound=map_id, request='two')
    copy = save('副本', bound=map_id, request='copy')
    assert library.list_tags(tmp_path, map_id, 2) == library.list_tags(tmp_path, copy, 1) == []
    for rev, seg, x, yaw in [(3, 'a', 1, 0), (1, 'b', 1, 0), (1, 'a', True, 0),
                             (1, 'a', float('nan'), 1), (1, 'a', 1, float('inf'))]:
        with pytest.raises(ValueError):
            library.create_tag(tmp_path, map_id, rev, 'invalid', x, 1, yaw, segment_id=seg)
    with pytest.raises(ValueError, match='already in use'):
        library.create_tag(tmp_path, map_id, 1, '前台', 1, 1, segment_id='a')
    updated = library.update_tag(tmp_path, map_id, 1, tag['id'], '出口', 1, 3, 0.5, segment_id='a')
    assert updated == {**tag, 'name': '出口', 'x': 1, 'y': 3, 'yaw': .5}
    assert library.list_tags(tmp_path, map_id, 1) == [updated]
    with pytest.raises(ValueError):
        library.delete_tag(tmp_path, map_id, 2, tag['id'])
    assert next(e for e in library.map_entries(tmp_path) if e['id'] == map_id)['tag_count'] == 1
    library.delete_tag(tmp_path, map_id, 1, tag['id'])
    assert library.list_tags(tmp_path, map_id, 1) == []


def test_parallel_tag_writes_and_save_preserve_edits(tmp_path, saved):
    map_id, save, _ = saved
    def create(i):
        return library.create_tag(tmp_path, map_id, 1, str(i), i, i, segment_id='a')
    with ThreadPoolExecutor(max_workers=5) as pool:
        writers = [pool.submit(create, i) for i in range(12)]
        revision = pool.submit(save, '地图', map_id, 'two')
        tags = [f.result() for f in writers]
        assert revision.result() == map_id
    assert {t['id'] for t in library.list_tags(tmp_path, map_id, 1)} == {t['id'] for t in tags}
    assert library.map_entries(tmp_path)[0]['revision'] == 2


def test_failed_changes_preserve_committed_map(tmp_path, saved, monkeypatch):
    map_id, _, _ = saved
    tag = library.create_tag(tmp_path, map_id, 1, '出口', 0, 0, segment_id='a')
    def fail(*args):
        raise OSError('simulated commit failure')
    monkeypatch.setattr(library.os, 'replace', fail)
    with pytest.raises(OSError):
        library.delete_tag(tmp_path, map_id, 1, tag['id'])
    with pytest.raises(OSError):
        library.delete_map(tmp_path, map_id)
    assert library.list_tags(tmp_path, map_id, 1) == [tag]


def test_delete_only_map_with_matching_manifest(tmp_path, saved):
    map_id, save, _ = saved
    other = save('另一个', request='another')
    maps = tmp_path / 'maps'
    stranger = maps / ('e' * 32)
    stranger.mkdir()
    (stranger / 'manifest.json').write_text(json.dumps({'id': map_id}))
    link = maps / ('f' * 32)
    link.symlink_to(maps / other, target_is_directory=True)
    for target in ('../sessions', stranger.name, link.name):
        with pytest.raises(ValueError):
            library.delete_map(tmp_path, target)
    library.delete_map(tmp_path, map_id)
    stranger.rename(maps / 'unrelated')
    with pytest.raises(ValueError, match='bound map no longer exists'):
        save(bound=map_id, request='late-save')
    assert not (maps / map_id).exists()
    assert (maps / other / '1.json').exists() and (maps / 'unrelated').exists() and link.is_symlink()
    assert list((tmp_path / 'sessions').glob('*/*.npy'))


def test_view_archived_frames_with_changed_live_configuration(tmp_path, saved):
    map_id, _, cfg = saved
    path = tmp_path / 'maps' / map_id / '1.json'
    live = {**cfg, 'clock_id': 'new-clock', 'child_frame': 'new-child', 'max_cloud_points': 1,
            'persistent_history': {'max_disk_bytes': 1, 'reserve_bytes': 0}}
    viewed, _ = load_paged_archive(path, live, max_frames=3, max_points=30, read_only=True)
    assert viewed.segments[0].frames[0].pose.clock_id == cfg['clock_id']
    assert viewed.directory is None
    with pytest.raises(ValueError, match='configuration mismatch'):
        load_paged_archive(path, live, max_frames=3, max_points=30)
