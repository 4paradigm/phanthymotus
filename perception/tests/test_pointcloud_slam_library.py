import hashlib
import json
import sqlite3
import struct

import numpy as np
import pytest

from plugins import pointcloud_slam_library as library


def database(path, *, data=True):
    with sqlite3.connect(str(path)) as connection:
        connection.executescript('CREATE TABLE Node(id INTEGER PRIMARY KEY); INSERT INTO Node VALUES(1);'
                                 'CREATE TABLE Data(id INTEGER PRIMARY KEY, scan BLOB);')
        if data:
            connection.execute('INSERT INTO Data VALUES(1, ?)', (b'cloud',))
    return path


def packet(x=2., points=((-1., 2., 3.), (4., 5., 6.))):
    xyz = np.asarray(points, dtype='<f4')
    return struct.pack('<fffBI', x, -3., .5, 3, len(xyz)) + xyz.tobytes()


@pytest.fixture
def saved(tmp_path):
    db = database(tmp_path / 'source.db')
    cfg = {'reserve_bytes': 0, 'map_plane': 'xy'}
    artifacts = {'segments': [{'segment_id': 'a', 'frame_id': 'map_a',
                              'database': str(db), 'snapshot': packet()}],
                 'statistics': {'nodes': 1}}

    def save(name='地图', bound=None, request='one'):
        return library.save_rtabmap_revision(tmp_path, artifacts, cfg, name, bound, request)[0]

    return save(), save, cfg


def test_native_save_is_self_contained_and_idempotent(tmp_path, saved):
    map_id, save, _ = saved
    source = tmp_path / 'source.db'
    source.unlink()
    assert save() == map_id
    assert library.map_entries(tmp_path)[0]['revision'] == 1
    revision = json.loads((tmp_path / 'maps' / map_id / '1.json').read_text())
    segment = revision['segments'][0]
    copied = tmp_path / 'maps' / map_id / segment['database']
    assert copied.is_file()
    with sqlite3.connect(str(copied)) as connection:
        assert connection.execute('SELECT count(*) FROM Node').fetchone() == (1,)
    assert segment['database_sha256'] == hashlib.sha256(copied.read_bytes()).hexdigest()
    snapshots = library.load_map_snapshots(tmp_path, map_id, max_points=1)
    assert len(snapshots) == 1
    assert snapshots[0]['segment_id'] == 'a' and snapshots[0]['frame_id'] == 'map_a'
    assert struct.unpack('<fffBI', snapshots[0]['packet'][:17]) == (2., -3., .5, 3, 1)
    assert snapshots[0]['metadata']['tracking'] is False
    assert snapshots[0]['metadata']['statistics'] == {'nodes': 1}


@pytest.fixture
def legacy_annotations(tmp_path, saved):
    map_id, _, _ = saved
    manifest = tmp_path / 'maps' / map_id / 'manifest.json'
    data = json.loads(manifest.read_text())
    tags = {'1': [{'id': 'old-place', 'name': '入口', 'x': 1, 'y': 2,
                   'yaw': 0, 'segment_id': 'a'}]}
    data['tags'] = tags
    manifest.write_text(json.dumps(data))
    return tags


def test_map_management_preserves_legacy_annotations(tmp_path, saved, legacy_annotations):
    map_id, save, _ = saved
    manifest = tmp_path / 'maps' / map_id / 'manifest.json'
    before = manifest.read_bytes()
    assert library.load_map_snapshots(tmp_path, map_id)
    assert manifest.read_bytes() == before
    assert library.rename_map(tmp_path, map_id, '新名字')['tags'] == legacy_annotations
    assert save('新名字', bound=map_id, request='two') == map_id
    entry = library.map_entries(tmp_path)[0]
    assert entry['revision'] == 2 and entry['tags'] == legacy_annotations
    copy = save('副本', bound=map_id, request='copy')
    assert json.loads(manifest.read_text())['tags'] == legacy_annotations
    copied = json.loads((tmp_path / 'maps' / copy / 'manifest.json').read_text())
    assert not copied.get('tags')


def test_failed_changes_preserve_committed_map(tmp_path, saved, legacy_annotations, monkeypatch):
    map_id, save, _ = saved
    def fail(*args):
        raise OSError('simulated commit failure')
    monkeypatch.setattr(library.os, 'replace', fail)
    with pytest.raises(OSError):
        save(bound=map_id, request='failed')
    with pytest.raises(OSError):
        library.rename_map(tmp_path, map_id, '新名字')
    with pytest.raises(OSError):
        library.delete_map(tmp_path, map_id)
    assert library.map_entries(tmp_path)[0]['tags'] == legacy_annotations
    assert library.map_entries(tmp_path)[0]['revision'] == 1


def test_delete_and_rename_validate_manifest_identity(tmp_path, saved):
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
    stranger.rename(maps / 'unrelated')
    assert library.rename_map(tmp_path, map_id, ' 新名字 ')['name'] == '新名字'
    library.delete_map(tmp_path, map_id)
    with pytest.raises(ValueError, match='bound map no longer exists'):
        save(bound=map_id, request='late-save')
    assert not (maps / map_id).exists()
    assert (maps / other / '1.json').exists() and (maps / 'unrelated').exists() and link.is_symlink()


@pytest.mark.parametrize('invalid', ['empty', 'sqlite', 'path', 'packet', 'budget'])
def test_reject_invalid_native_artifacts_without_committing(tmp_path, invalid):
    db = database(tmp_path / 'source.db', data=invalid != 'empty')
    snapshot = packet()
    cfg = {'reserve_bytes': 0}
    if invalid == 'sqlite':
        db.write_bytes(b'not sqlite')
    if invalid == 'path':
        db = 'relative.db'
    if invalid == 'packet':
        snapshot = b'incomplete'
    if invalid == 'budget':
        cfg['max_disk_bytes'] = 1
    artifacts = {'segments': [{'segment_id': 'a', 'frame_id': 'map', 'database': str(db),
                               'snapshot': snapshot}], 'statistics': {}}
    with pytest.raises((ValueError, OSError)):
        library.save_rtabmap_revision(tmp_path, artifacts, cfg, 'bad', None, 'one')
    assert library.map_entries(tmp_path) == []


def test_no_snapshot_keeps_native_database_without_inventing_marker_or_plane(tmp_path):
    db = database(tmp_path / 'source.db')
    artifacts = {'segments': [{'segment_id': 'a', 'frame_id': 'map',
                              'database': str(db), 'snapshot': None}], 'statistics': {}}
    map_id, revision, duplicate = library.save_rtabmap_revision(
        tmp_path, artifacts, {'reserve_bytes': 0}, 'database only', None, 'one')
    assert revision == 1 and duplicate is False
    assert library.load_map_snapshots(tmp_path, map_id) == []
    data = json.loads((tmp_path / 'maps' / map_id / '1.json').read_text())
    assert [segment['segment_id'] for segment in data['segments']] == ['a']
    assert data['segments'][0]['map_plane'] is None


def test_snapshot_checksum_prevents_viewing_modified_geometry(tmp_path, saved):
    map_id, _, _ = saved
    path = tmp_path / 'maps' / map_id
    revision = json.loads((path / '1.json').read_text())
    (path / revision['segments'][0]['snapshot']).write_bytes(packet(x=9.))
    with pytest.raises(ValueError, match='checksum'):
        library.load_map_snapshots(tmp_path, map_id)


def test_late_manifest_failure_preserves_revision_and_allows_retry(tmp_path, saved, monkeypatch):
    map_id, save, _ = saved
    original = library.os.replace
    def fail_manifest(source, target):
        if target == tmp_path / 'maps' / map_id / 'manifest.json':
            raise OSError('manifest commit failed')
        return original(source, target)
    monkeypatch.setattr(library.os, 'replace', fail_manifest)
    with pytest.raises(OSError, match='manifest commit failed'):
        save(bound=map_id, request='retry')
    assert library.map_entries(tmp_path)[0]['revision'] == 1
    assert not (tmp_path / 'maps' / map_id / '2.json').exists()
    assert not (tmp_path / 'maps' / map_id / 'revisions' / '2').exists()
    monkeypatch.setattr(library.os, 'replace', original)
    save(bound=map_id, request='retry')
    assert library.map_entries(tmp_path)[0]['revision'] == 2


def test_existing_map_copy_cannot_bypass_cumulative_storage_budget(tmp_path, saved):
    map_id, _, _ = saved
    db = tmp_path / 'source.db'
    artifacts = {'segments': [{'segment_id': 'b', 'frame_id': 'map_b',
                              'database': str(db), 'snapshot': packet()}], 'statistics': {}}
    with pytest.raises(OSError, match='budget'):
        library.save_rtabmap_revision(tmp_path, artifacts,
            {'reserve_bytes': 0, 'max_disk_bytes': db.stat().st_size + len(packet())},
            '地图', map_id, 'next')
    assert library.map_entries(tmp_path)[0]['revision'] == 1


def test_revision_metadata_symlink_cannot_view_external_map(tmp_path, saved):
    map_id, _, _ = saved
    revision = tmp_path / 'maps' / map_id / '1.json'
    external = tmp_path / 'outside.json'
    revision.replace(external)
    revision.symlink_to(external)
    with pytest.raises(ValueError, match='location'):
        library.load_map_snapshots(tmp_path, map_id)


@pytest.mark.parametrize('frame_id', ['', None])
def test_database_without_snapshot_preserves_unknown_map_frame_and_plane(tmp_path, frame_id):
    db = database(tmp_path / 'source.db')
    artifacts = {'segments': [{'segment_id': 'a', 'frame_id': frame_id,
                              'database': str(db), 'snapshot': None}], 'statistics': {}}
    map_id, revision, _ = library.save_rtabmap_revision(
        tmp_path, artifacts, {'reserve_bytes': 0, 'map_plane': 'xy'}, 'native only', None, 'one')
    data = json.loads((tmp_path / 'maps' / map_id / '1.json').read_text())
    assert data['segments'][0]['frame_id'] is frame_id
    assert data['segments'][0]['map_plane'] is None
    assert library.load_map_snapshots(tmp_path, map_id) == []


def test_snapshot_still_requires_known_native_map_frame(tmp_path):
    db = database(tmp_path / 'source.db')
    artifacts = {'segments': [{'segment_id': 'a', 'frame_id': '',
                              'database': str(db), 'snapshot': packet()}], 'statistics': {}}
    with pytest.raises(ValueError, match='frame'):
        library.save_rtabmap_revision(tmp_path, artifacts, {'reserve_bytes': 0},
                                     'invalid snapshot', None, 'one')


def test_manifest_directory_sync_failure_keeps_referenced_revision_and_retries_durably(
        tmp_path, saved, monkeypatch):
    map_id, save, settings = saved
    directory = tmp_path / 'maps' / map_id
    original = library._sync_dir
    def fail_after_manifest_replace(path):
        if path == directory and json.loads((directory / 'manifest.json').read_text())['revision'] == 2:
            raise OSError('manifest directory sync failed')
        return original(path)
    monkeypatch.setattr(library, '_sync_dir', fail_after_manifest_replace)
    with pytest.raises(OSError, match='manifest directory sync failed'):
        save(bound=map_id, request='durable-retry')
    assert library.map_entries(tmp_path)[0]['revision'] == 2
    revision = json.loads((directory / '2.json').read_text())
    assert (directory / revision['segments'][0]['database']).is_file()
    assert library.load_map_snapshots(tmp_path, map_id)[0]['segment_id'] == 'a'
    with pytest.raises(OSError, match='manifest directory sync failed'):
        save(bound=map_id, request='durable-retry')
    monkeypatch.setattr(library, '_sync_dir', original)
    artifacts = {'segments': [{'segment_id': 'a', 'frame_id': 'map_a',
        'database': str(tmp_path / 'source.db'), 'snapshot': packet()}], 'statistics': {'nodes': 1}}
    assert library.save_rtabmap_revision(tmp_path, artifacts, settings, '地图', map_id,
                                         'durable-retry') == (map_id, 2, True)
    assert library.load_map_snapshots(tmp_path, map_id)[0]['segment_id'] == 'a'


@pytest.mark.parametrize('broken', ['revision', 'database', 'database_checksum',
                                    'snapshot', 'snapshot_checksum'])
def test_idempotent_retry_rejects_missing_or_corrupted_committed_artifacts(tmp_path, saved, broken):
    map_id, save, _ = saved
    directory = tmp_path / 'maps' / map_id
    revision_path = directory / '1.json'
    revision = json.loads(revision_path.read_text())
    segment = revision['segments'][0]
    if broken == 'revision':
        revision_path.unlink()
    elif broken in ('database', 'snapshot'):
        (directory / segment[broken]).unlink()
    elif broken == 'database_checksum':
        with sqlite3.connect(str(directory / segment['database'])) as connection:
            connection.execute('INSERT INTO Node VALUES(2)')
    else:
        (directory / segment['snapshot']).write_bytes(packet(x=9.))
    with pytest.raises((ValueError, OSError)):
        save()


@pytest.mark.parametrize('flags', [15, 23, 39, 47])
def test_typed_heading_survives_save_load_and_point_limit(tmp_path, flags):
    source = packet()
    typed = source[:12] + bytes([flags]) + source[13:]
    artifacts = {'segments': [{'segment_id': 'a', 'frame_id': 'map_a',
        'database': str(database(tmp_path / 'source.db')), 'snapshot': typed}], 'statistics': {}}
    map_id = library.save_rtabmap_revision(tmp_path, artifacts,
        {'reserve_bytes': 0, 'map_plane': 'xy'}, 'Typed', None, 'one')[0]
    loaded = library.load_map_snapshots(tmp_path, map_id, max_points=1)[0]['packet']
    assert loaded[:13] == typed[:13]
    assert len(loaded) == 29
