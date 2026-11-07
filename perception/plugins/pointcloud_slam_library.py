"""Named, self-contained revisions of paged SLAM maps on one filesystem."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unicodedata
import uuid

from .pointcloud_slam_paged_archive import configuration, MAX_CHECKPOINT_BYTES
from .pointcloud_slam_archive import archive_metadata


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
        return [dict(entry, tag_count=sum(len(tags) for tags in entry.get('tags', {}).values()))
                for entry in _entries(maps)]


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


def _segments(maps, map_id, revision):
    with (maps / map_id / f'{revision}.json').open() as src:
        frames = json.load(src)['frames']
    return list(dict.fromkeys(frame['pose']['segment_id'] for frame in frames))


def revision_segments(root, map_id, revision):
    with _maps_locked(root, False) as maps:
        _revision(maps, map_id, revision)
        return _segments(maps, map_id, revision)


def revision_plane(root, map_id, revision, segment_id):
    from .pointcloud_slam_paged_archive import inspect_paged_archive
    with _maps_locked(root, False) as maps:
        _revision(maps, map_id, revision)
        _, resolved = inspect_paged_archive(maps / map_id / f'{revision}.json')
        return resolved.get('segment_metadata', {}).get(segment_id, {}).get('map_plane')


def list_tags(root, map_id, revision):
    with _maps_locked(root, False) as maps:
        entry = _revision(maps, map_id, revision)
        return entry.get('tags', {}).get(str(revision), [])


def _change_tag(root, map_id, revision, tag_id, tag):
    with _maps_locked(root) as maps:
        entry = _revision(maps, map_id, revision)
        tags = entry.setdefault('tags', {}).setdefault(str(revision), [])
        if tag_id is not None:
            previous = next((t for t in tags if t['id'] == tag_id), None)
            if previous is None:
                raise ValueError('unknown map tag')
            tags.remove(previous)
        if tag is not None:
            if tag['segment_id'] not in _segments(maps, map_id, revision):
                raise ValueError('unknown map segment')
            if any(t['name'] == tag['name'] for t in tags):
                raise ValueError('tag name already in use')
            tag['id'] = tag_id or uuid.uuid4().hex
            tags.append(tag)
        with tempfile.TemporaryDirectory(dir=maps) as tmp:
            _write_json(Path(tmp) / 'meta.tmp', maps / map_id / 'manifest.json', entry)
        return tag


def _tag(name, x, y, yaw, segment_id):
    name = validate_map_name(name)
    coords = (x, y) if yaw is None else (x, y, yaw)
    if any(type(v) not in (int, float) or not -sys.float_info.max <= v <= sys.float_info.max for v in coords):
        raise ValueError('tag coordinates must be finite numbers')
    return {'name': name, 'x': x, 'y': y, 'yaw': yaw, 'segment_id': segment_id}


def create_tag(root, map_id, revision, name, x, y, yaw=None, *, segment_id):
    return _change_tag(root, map_id, revision, None, _tag(name, x, y, yaw, segment_id))


def update_tag(root, map_id, revision, tag_id, name, x, y, yaw=None, *, segment_id):
    if not tag_id:
        raise ValueError('tag id required')
    return _change_tag(root, map_id, revision, tag_id, _tag(name, x, y, yaw, segment_id))


def delete_tag(root, map_id, revision, tag_id):
    if not tag_id:
        raise ValueError('tag id required')
    _change_tag(root, map_id, revision, tag_id, None)


def save_revision(root, records, history, settings, name, bound_id, request_id):
    name = validate_map_name(name)
    if not isinstance(request_id, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', request_id):
        raise ValueError('invalid save request id')
    if not records:
        raise ValueError('no keyframes to save')
    root = Path(root)
    with _maps_locked(root) as maps:
        entries = _entries(maps)
        for entry in entries:
            for saved in entry['saves']:
                if saved['request_id'] == request_id:
                    if entry['name'] != name or saved['source'] != records[0]['session']:
                        raise ValueError('save request id reused')
                    return entry['id'], saved['revision'], True
        if bound_id is not None and not any(item['id'] == bound_id for item in entries):
            raise ValueError('bound map no longer exists')
        entry = next((item for item in entries if item['name'] == name), None)
        if entry and entry['id'] != bound_id:
            raise ValueError('map name already in use')
        if entry is None:
            entry = {'id': uuid.uuid4().hex, 'name': name, 'saves': []}
        final = maps / entry['id']
        revision = len(entry['saves']) + 1
        with tempfile.TemporaryDirectory(dir=maps, prefix='.saving-') as tmp:
            stage = Path(tmp)
            dest = final if entry['saves'] else stage / 'map'
            dest.mkdir(exist_ok=True)
            for record in records:
                rel = Path('sessions') / record['session'] / record['file']
                src, target = root / rel, dest / rel
                if target.exists():
                    continue
                if shutil.disk_usage(dest).free < src.stat().st_size + settings['persistent_history']['reserve_bytes']:
                    raise OSError('insufficient space to save map')
                target.parent.mkdir(parents=True, exist_ok=True)
                copy = stage / 'point.tmp'
                shutil.copyfile(src, copy)
                with copy.open('rb') as f:
                    if hashlib.sha256(f.read()).hexdigest() != record['sha256']:
                        raise ValueError('point file checksum mismatch')
                    os.fsync(f.fileno())
                os.replace(copy, target)
                _sync_dir(target.parent)
            data = {'schema': 'motus.experimental-slam-paged/2', 'configuration': configuration(settings),
                    'resolved': archive_metadata(settings, segment_ids=list(dict.fromkeys(
                        record['pose']['segment_id'] for record in records))),
                    'frames': records, 'history': history}
            _sync_dir(dest / 'sessions')
            _write_json(stage / 'meta.tmp', dest / f'{revision}.json', data, MAX_CHECKPOINT_BYTES)
            entry['saves'].append({'request_id': request_id, 'revision': revision,
                                   'source': records[0]['session'], 'keyframes': len(records)})
            entry['revision'] = revision
            _write_json(stage / 'meta.tmp', dest / 'manifest.json', entry)
            if dest != final:
                os.replace(dest, final)
            _sync_dir(maps)
        return entry['id'], revision, False
