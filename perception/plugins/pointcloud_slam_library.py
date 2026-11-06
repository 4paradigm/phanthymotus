"""Named, self-contained revisions of paged SLAM maps on one filesystem."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata
import uuid

from .pointcloud_slam_paged_archive import configuration, MAX_CHECKPOINT_BYTES


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def map_entries(root):
    entries = []
    for path in sorted((Path(root) / 'maps').glob('*/manifest.json')):
        with path.open() as src:
            entries.append(json.load(src))
    return entries


def save_revision(root, records, history, settings, name, bound_id, request_id):
    if not isinstance(name, str):
        raise ValueError('map name required')
    name = unicodedata.normalize('NFC', name.strip())
    if not name or len(name) > 64 or any(ord(c) < 32 or c in '/\\' for c in name):
        raise ValueError('invalid map name')
    if not isinstance(request_id, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', request_id):
        raise ValueError('invalid save request id')
    if not records:
        raise ValueError('no keyframes to save')
    root = Path(root)
    maps = root / 'maps'
    maps.mkdir(exist_ok=True)
    with (maps / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        entries = map_entries(root)
        for entry in entries:
            for saved in entry['saves']:
                if saved['request_id'] == request_id:
                    if entry['name'] != name or saved['source'] != records[0]['session']:
                        raise ValueError('save request id reused')
                    return entry['id'], saved['revision'], True
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
            data = {'schema': 'motus.experimental-slam-paged/1', 'configuration': configuration(settings),
                    'frames': records, 'history': history}
            def write_json(target, obj, limit=None):
                temp = stage / 'meta.tmp'
                with temp.open('w') as out:
                    json.dump(obj, out, allow_nan=False, separators=(',', ':'))
                    out.flush()
                    if limit and out.tell() > limit:
                        raise ValueError('map snapshot metadata exceeds capacity')
                    os.fsync(out.fileno())
                os.replace(temp, target)
                _sync_dir(target.parent)
            _sync_dir(dest / 'sessions')
            write_json(dest / f'{revision}.json', data, MAX_CHECKPOINT_BYTES)
            entry['saves'].append({'request_id': request_id, 'revision': revision,
                                   'source': records[0]['session']})
            entry['revision'] = revision
            write_json(dest / 'manifest.json', entry)
            if dest != final:
                os.replace(dest, final)
            _sync_dir(maps)
        return entry['id'], revision, False
