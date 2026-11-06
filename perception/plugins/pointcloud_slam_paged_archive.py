"""Small atomic checkpoints referencing immutable local point files."""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

from .pointcloud_slam_archive import _configuration
from .pointcloud_slam_odom import decode_slam_odom
from .pointcloud_slam_paged_store import PagedKeyframeStore, DiskPoints
from .pointcloud_slam_stream import _rigid


MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024

def configuration(settings):
    return {**_configuration(settings), 'persistent_history':settings['persistent_history']}


def save_paged_archive(path, store, history, settings):
    # Call under the input lock; history snapshots are committed immutable lists.
    data={'schema':'motus.experimental-slam-paged/1', 'configuration':configuration(settings),
          'frames':store.records, 'history':history}
    fd,temp=tempfile.mkstemp(prefix='.slam-',dir=Path(path).parent)
    try:
        with os.fdopen(fd,'w') as out:
            json.dump(data,out,allow_nan=False,separators=(',',':'));out.flush();os.fsync(out.fileno())
        if os.path.getsize(temp)>MAX_CHECKPOINT_BYTES:
            raise ValueError('paged checkpoint metadata capacity reached; prior save preserved')
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def load_paged_archive(path, settings, *, max_frames, max_points, read_only=False):
    path=Path(path);root=path.parent/'sessions'
    try:
        with path.open('rb') as src: raw=src.read(MAX_CHECKPOINT_BYTES+1)
        if len(raw)>MAX_CHECKPOINT_BYTES: raise ValueError('paged checkpoint too large')
        data=json.loads(raw)
        if read_only:
            # A saved map has its own coordinate frames, calibration and storage policy.
            settings = data['configuration']
        expected = configuration(settings)
        if (data.get('schema')!='motus.experimental-slam-paged/1' or
                data.get('configuration')!=expected):
            raise ValueError('paged archive configuration mismatch')
        frames,history=data['frames'],data['history']
        if not isinstance(frames,list) or not isinstance(history,dict):
            raise ValueError('invalid paged checkpoint')
        store=PagedKeyframeStore(None if read_only else root/uuid.uuid4().hex,max_frames=max_frames,max_points=max_points,
                                 **settings['persistent_history'])
        for record in frames:
            if (not isinstance(record['session'],str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',record['session']) or
                    not re.fullmatch(r'[0-9]{8}\.npy',record['file']) or
                    type(record['count']) is not int or not 0<record['count']<=settings.get('max_cloud_points',max_points)):
                raise ValueError('invalid point file reference')
            file=root/record['session']/record['file']
            if file.resolve().parent.parent != root.resolve(): raise ValueError('invalid point file location')
            size=file.stat().st_size
            if size>record['count']*24+4096 or store.disk_bytes+size>store.max_disk_bytes:
                raise ValueError('point file exceeds disk budget')
            if hashlib.sha256(file.read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError('point file checksum mismatch')
            points=DiskPoints(file,record['count'])
            p=record['pose']
            pose=decode_slam_odom({'schema':'motus.odom/1','units':{'length':'m'},'stamp_ms':p['stamp_ms'],
                'pose':{k:p[k] for k in ('frame_id','child_frame_id','position','quaternion','covariance')},
                'vendor':{'motus_slam':{'status':'tracking','stamp_source':'measurement',
                                      'segment_id':p['segment_id'],'clock_id':p['clock_id']}}})
            if (pose.clock_id!=settings['clock_id'] or
                    pose.child_frame_id!=settings['child_frame'].replace('{segment_id}',pose.segment_id)):
                raise ValueError('archived pose frame or clock mismatch')
            store._restore_ref=points
            store.add(pose,points,cloud_stamp_ns=record['stamp_ns'])
            store._restore_ref=None
            journal_size=0 if read_only else store._append_record(record)
            store.records.append(record);store.disk_bytes+=size+journal_size
        segments={s.segment_id:s for s in store.segments}
        for key,snapshot in history.items():
            if key not in segments or not snapshot['poses'] or len(snapshot['poses'])!=len(snapshot['stamps']):
                raise ValueError('invalid history poses')
            expected=[f.cloud_stamp_ns for f in segments[key].frames[:len(snapshot['stamps'])]]
            if snapshot['stamps']!=expected: raise ValueError('history capture stamps mismatch')
            for pose in snapshot['poses']: _rigid(pose)
        store.mark_lost()
        return store,history
    except (OSError,KeyError,TypeError,UnicodeError,json.JSONDecodeError) as error:
        raise ValueError('invalid paged map archive') from error
