"""Atomic named RTABMap revisions and read-only legacy map snapshots."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import struct
from copy import deepcopy

import numpy as np
import tempfile
import unicodedata
import uuid

from .pointcloud_slam_legacy import SCHEMAS, file_digest, load_legacy_snapshots

MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024
MAX_SNAPSHOT_BYTES = 17 + 80000 * 12


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


_MAP_ID = re.compile(r'[0-9a-f]{32}')


@contextmanager
def _maps_locked(root, exclusive=True):
    maps = Path(root) / 'maps'
    if maps.is_symlink():
        raise ValueError('invalid map library directory')
    maps.mkdir(parents=True, exist_ok=True)
    with (maps / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield maps


def _manifest(maps, map_id):
    if not isinstance(map_id, str) or not _MAP_ID.fullmatch(map_id):
        raise ValueError('invalid map id')
    directory = maps / map_id
    path = directory / 'manifest.json'
    if directory.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError('unknown map')
    with path.open() as src:
        entry = json.load(src)
    if entry['id'] != map_id:
        raise ValueError('map id does not match directory')
    return entry


def _entries(maps):
    return [_manifest(maps, path.name) for path in sorted(maps.iterdir())
            if _MAP_ID.fullmatch(path.name) and not path.is_symlink()
            and (path / 'manifest.json').is_file()]


def map_entries(root):
    if not (Path(root) / 'maps').is_dir():
        return []
    with _maps_locked(root, False) as maps:
        return _entries(maps)


def validate_map_name(name):
    if not isinstance(name, str):
        raise ValueError('map name required')
    name = unicodedata.normalize('NFC', name.strip())
    if not name or len(name) > 64 or any(unicodedata.category(c) == 'Cc' or c in '/\\' for c in name):
        raise ValueError('invalid map name')
    return name


def delete_map(root, map_id):
    with _maps_locked(root) as maps:
        _manifest(maps, map_id)
        with tempfile.TemporaryDirectory(dir=maps, prefix='.deleting-') as stage:
            os.replace(maps / map_id, Path(stage) / 'map')
            _sync_dir(maps)



def rename_map(root, map_id, name):
    name = validate_map_name(name)
    with _maps_locked(root) as maps:
        entry = _manifest(maps, map_id)
        if any(other['id'] != map_id and other['name'] == name for other in _entries(maps)):
            raise ValueError('map name already in use')
        entry['name'] = name
        with tempfile.TemporaryDirectory(dir=maps) as stage:
            _write_json(Path(stage) / 'manifest.tmp', maps / map_id / 'manifest.json', entry)
        return entry

def _write_json(temp, target, obj, limit=None):
    with temp.open('w') as out:
        json.dump(obj, out, allow_nan=False, separators=(',', ':'))
        out.flush()
        if limit and out.tell() > limit:
            raise ValueError('map snapshot metadata exceeds capacity')
        os.fsync(out.fileno())
    os.replace(temp, target)
    _sync_dir(target.parent)


def _revision(maps, map_id, revision):
    entry = _manifest(maps, map_id)
    if type(revision) is not int or not any(s['revision'] == revision for s in entry['saves']):
        raise ValueError('unknown map revision')
    return entry


def _read_revision(path):
    if path.is_symlink():
        raise ValueError('invalid map revision location')
    with path.open('rb') as source:
        raw = source.read(MAX_CHECKPOINT_BYTES + 1)
    if len(raw) > MAX_CHECKPOINT_BYTES:
        raise ValueError('map metadata exceeds capacity')
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get('schema') not in (*SCHEMAS, 'motus.rtabmap-map/1'):
        raise ValueError('unsupported map revision')
    return data


def _mapping_packet(packet, max_points=80000):
    if not isinstance(packet, bytes) or not 17 <= len(packet) <= MAX_SNAPSHOT_BYTES:
        raise ValueError('invalid saved mapping packet')
    x, y, yaw, flags, count = struct.unpack('<fffBI', packet[:17])
    if ((flags & 3) != 3 or flags & ~63 or (flags & 24) == 24 or
            len(packet) != 17 + 12 * count or not np.isfinite([x, y, yaw]).all()):
        raise ValueError('invalid saved mapping packet')
    points = np.frombuffer(packet[17:], dtype='<f4').reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError('invalid saved mapping points')
    if count > max_points:
        points = points[np.linspace(0, count - 1, max_points, dtype=int)]
        return struct.pack('<fffBI', x, y, yaw, flags, len(points)) + points.tobytes()
    return packet


def _check_database(path):
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError('absolute native RTABMap database file required')
    try:
        with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as connection:
            if connection.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                raise ValueError('native RTABMap database integrity check failed')
            for table in ('Node', 'Data'):
                if connection.execute('SELECT 1 FROM "' + table + '" LIMIT 1').fetchone() is None:
                    raise ValueError('native RTABMap database has no ' + table + ' records')
    except sqlite3.DatabaseError as error:
        raise ValueError('invalid native RTABMap database') from error


def _copy_file(source, target):
    digest = hashlib.sha256()
    with source.open('rb') as src, target.open('xb') as dst:
        for chunk in iter(lambda: src.read(1024 * 1024), b''):
            dst.write(chunk)
            digest.update(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    return digest.hexdigest()


def save_rtabmap_revision(root, artifacts, settings, name, bound_id, request_id):
    """Commit native databases plus bounded display caches; no legacy solver data."""
    name = validate_map_name(name)
    if not isinstance(request_id, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', request_id):
        raise ValueError('invalid save request id')
    segments = artifacts.get('segments')
    if not isinstance(segments, list) or not segments or not isinstance(artifacts.get('statistics'), dict):
        raise ValueError('native RTABMap segments and statistics required')
    source_id = [segment['segment_id'] for segment in segments]
    if (len(set(source_id)) != len(source_id) or any(not isinstance(s, str) or not s.strip() for s in source_id)):
        raise ValueError('distinct native map segment ids required')
    with _maps_locked(root) as maps:
        entries = _entries(maps)
        for entry in entries:
            for saved in entry['saves']:
                if saved['request_id'] == request_id:
                    if entry['name'] != name or saved.get('source') != source_id:
                        raise ValueError('save request id reused')
                    _verify_saved_revision(maps / entry['id'], saved['revision'], source_id)
                    return entry['id'], saved['revision'], True
        if bound_id is not None and not any(item['id'] == bound_id for item in entries):
            raise ValueError('bound map no longer exists')
        entry = next((item for item in entries if item['name'] == name), None)
        if entry and entry['id'] != bound_id:
            raise ValueError('map name already in use')
        if entry is None:
            entry = {'id': uuid.uuid4().hex, 'name': name, 'saves': []}
        max_bytes = settings.get('max_disk_bytes', 2147483648)
        reserve = settings.get('reserve_bytes', 1073741824)
        if type(max_bytes) is not int or max_bytes <= 0 or type(reserve) is not int or reserve < 0:
            raise ValueError('invalid map storage budget')
        size = 0
        for segment in segments:
            frame_id, snapshot = segment.get('frame_id'), segment.get('snapshot')
            if (snapshot is not None and (not isinstance(frame_id, str) or not frame_id.strip()) or
                    snapshot is None and frame_id is not None and not isinstance(frame_id, str)):
                raise ValueError('native map frame id required')
            source = Path(segment['database'])
            _check_database(source)
            if snapshot is not None:
                _mapping_packet(snapshot)
            size += source.stat().st_size + (len(snapshot) if snapshot is not None else 0)
        final = maps / entry['id']
        existing_bytes = sum(path.stat().st_size for path in final.rglob('*') if path.is_file())
        if existing_bytes + size > max_bytes or shutil.disk_usage(maps).free < size + reserve:
            raise OSError('map disk budget or free-space reserve reached')
        revision = len(entry['saves']) + 1
        with tempfile.TemporaryDirectory(dir=maps, prefix='.saving-') as tmp:
            stage = Path(tmp)
            dest = final if entry['saves'] else stage / 'map'
            dest.mkdir(exist_ok=True)
            assets = stage / 'assets'
            assets.mkdir()
            saved_segments = []
            for index, segment in enumerate(segments):
                database = assets / f'{index}.db'
                digest = _copy_file(Path(segment['database']), database)
                _check_database(database)
                if digest != file_digest(Path(segment['database'])):
                    raise ValueError('native RTABMap database changed while saving')
                rel = Path('revisions') / str(revision)
                metadata = deepcopy((settings.get('segment_metadata') or {}).get(segment['segment_id'], {}))
                plane = (metadata.get('map_plane', settings.get('map_plane'))
                         if segment.get('snapshot') is not None else None)
                if plane not in (None, 'xy'):
                    raise ValueError('unsupported map annotation plane')
                saved_segment = {'segment_id': segment['segment_id'], 'frame_id': segment['frame_id'],
                    'database': str(rel / database.name), 'database_sha256': digest,
                    'snapshot': None, 'map_plane': plane, 'metadata': metadata}
                snapshot = segment.get('snapshot')
                if snapshot is not None:
                    cache = assets / f'{index}.mapping'
                    with cache.open('xb') as out:
                        out.write(snapshot)
                        out.flush()
                        os.fsync(out.fileno())
                    saved_segment.update(snapshot=str(rel / cache.name),
                                          snapshot_sha256=hashlib.sha256(snapshot).hexdigest())
                saved_segments.append(saved_segment)
            data = {'schema': 'motus.rtabmap-map/1', 'segments': saved_segments,
                    'statistics': artifacts['statistics'], 'resolved': {key: deepcopy(settings.get(key))
                        for key in ('resolved_inputs', 'extrinsic_mode', 'map_plane',
                            'cloud_frame', 'child_frame', 'clock_id',
                            'child_from_cloud', 'input_provenance')}}
            entry['saves'].append({'request_id': request_id, 'revision': revision, 'source': source_id,
                                   'segments': len(saved_segments), 'backend': 'rtabmap'})
            entry.update(revision=revision, backend='rtabmap')
            manifest = stage / 'manifest.ready'
            _write_json(stage / 'manifest.tmp', manifest, entry, MAX_CHECKPOINT_BYTES)
            target = dest / 'revisions' / str(revision)
            target.parent.mkdir(exist_ok=True)
            committed = False
            try:
                _sync_dir(assets)
                os.replace(assets, target)
                _sync_dir(target.parent)
                _write_json(stage / 'meta.tmp', dest / f'{revision}.json', data, MAX_CHECKPOINT_BYTES)
                os.replace(manifest, dest / 'manifest.json')
                committed = True
                _sync_dir(dest)
                if dest != final:
                    os.replace(dest, final)
                _sync_dir(maps)
            finally:
                if not committed and dest == final:
                    shutil.rmtree(target, ignore_errors=True)
                    (dest / f'{revision}.json').unlink(missing_ok=True)
        return entry['id'], revision, False


def _artifact_path(directory, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('invalid map artifact reference')
    path = directory / relative
    if directory.resolve() not in path.resolve().parents or any(
            p.is_symlink() for p in (path, *path.parents) if p != directory.parent):
        raise ValueError('invalid map artifact location')
    return path


def _verify_saved_revision(directory, revision, source_id):
    """A manifest entry is an acknowledgement only after its artifacts and sync succeed."""
    data = _read_revision(directory / f'{revision}.json')
    if (data['schema'] != 'motus.rtabmap-map/1' or
            [segment['segment_id'] for segment in data['segments']] != source_id):
        raise ValueError('saved native map revision does not match request')
    for segment in data['segments']:
        database = _artifact_path(directory, segment['database'])
        _check_database(database)
        if file_digest(database) != segment['database_sha256']:
            raise ValueError('saved native database checksum mismatch')
        if segment['snapshot'] is not None:
            cache = _artifact_path(directory, segment['snapshot'])
            with cache.open('rb') as source:
                packet = source.read(MAX_SNAPSHOT_BYTES + 1)
            if hashlib.sha256(packet).hexdigest() != segment['snapshot_sha256']:
                raise ValueError('saved mapping snapshot checksum mismatch')
            _mapping_packet(packet)
    for path in (directory / 'revisions' / str(revision), directory / 'revisions',
                 directory, directory.parent):
        _sync_dir(path)


def load_map_snapshots(root, map_id, revision=None, *, max_points=40000):
    """Read bounded cached geometry; a missing cache never invents a pose."""
    if type(max_points) is not int or not 0 < max_points <= 80000:
        raise ValueError('preview point limit must be between 1 and 80000')
    with _maps_locked(root, False) as maps:
        entry = _manifest(maps, map_id)
        revision = entry['revision'] if revision is None else revision
        _revision(maps, map_id, revision)
        directory = maps / map_id
        path = directory / f'{revision}.json'
        data = _read_revision(path)
        if data['schema'] in SCHEMAS:
            return load_legacy_snapshots(path, data, max_points=max_points)
        snapshots = []
        for segment in data['segments']:
            if segment['snapshot'] is None:
                continue
            cache = _artifact_path(directory, segment['snapshot'])
            with cache.open('rb') as source:
                packet = source.read(MAX_SNAPSHOT_BYTES + 1)
            if hashlib.sha256(packet).hexdigest() != segment['snapshot_sha256']:
                raise ValueError('saved mapping snapshot checksum mismatch')
            metadata = {**deepcopy(data.get('resolved') or {}), **deepcopy(segment.get('metadata') or {}),
                        'backend': 'rtabmap', 'tracking': False, 'map_plane': segment.get('map_plane'),
                        'statistics': deepcopy(data['statistics'])}
            snapshots.append({'segment_id': segment['segment_id'], 'frame_id': segment['frame_id'],
                              'packet': _mapping_packet(packet, max_points), 'metadata': metadata})
        return snapshots
