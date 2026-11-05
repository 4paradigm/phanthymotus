import numpy as np
import pytest
from plugins.pointcloud_slam_paged_store import PagedKeyframeStore, StorageError
from plugins.pointcloud_slam_odom import decode_slam_odom
from test_pointcloud_slam_stream import odom


def store(path, **kw):
    return PagedKeyframeStore(path / "session", max_frames=3, max_points=30,
        max_disk_bytes=10_000_000, reserve_bytes=0, **kw)


def add(s, i, x=None, segment='a'):
    p = decode_slam_odom(odom(1000+i*100, i if x is None else x, segment))
    return s.add(p, np.array([[1., 2., 3.]]*4), cloud_stamp_ns=p.stamp_ms*1_000_000)


def test_pages_beyond_active_capacity_without_retaining_arrays(tmp_path):
    s = store(tmp_path)
    for i in range(605):
        add(s, i)
    assert s._frame_count == 605
    assert not isinstance(s.segments[0].frames[0].points, np.ndarray)
    np.testing.assert_array_equal(s.segments[0].frames[0].points[[0, 3]], [[1,2,3]]*2)
    assert len(list((tmp_path / 'session').glob('*.npy'))) == 605
    assert s.disk_bytes > 0


def test_motion_filter_uses_last_committed_frame_and_preserves_new_segment(tmp_path):
    s = store(tmp_path, min_translation_m=.1, min_rotation_deg=10)
    assert add(s, 0, 0)
    assert not add(s, 1, .02)
    assert add(s, 2, .11)
    p = odom(1300, .11); p['pose']['quaternion'] = [0, 0, np.sin(.2), np.cos(.2)]
    assert s.add(decode_slam_odom(p), [[1,2,3]], cloud_stamp_ns=1300000000)
    assert add(s, 4, .11, 'b')
    assert s._frame_count == 4


def test_failed_commit_preserves_history_and_stops_future_writes(tmp_path, monkeypatch):
    s = store(tmp_path); add(s, 0)
    def fail(*a): raise OSError('disk full')
    monkeypatch.setattr(s, '_append_record', fail)
    with pytest.raises(StorageError): add(s, 1)
    assert s._frame_count == 1
    assert len(list((tmp_path / 'session').glob('*.npy'))) == 1
    with pytest.raises(StorageError): add(s, 2)


def test_paged_archive_restores_history_without_loading_all_points(tmp_path):
    from plugins.pointcloud_slam_paged_archive import save_paged_archive, load_paged_archive
    from test_pointcloud_slam_archive import settings as config
    root=tmp_path/'sessions';s=store(root)
    for i in range(7): add(s,i)
    settings=config();settings['persistent_history']={'max_disk_bytes':10_000_000,'reserve_bytes':0}
    path=tmp_path/'saved.json'
    history={'a': {'poses':[np.eye(4).tolist() for _ in range(7)],
                   'stamps':[f.cloud_stamp_ns for f in s.segments[0].frames]}}
    save_paged_archive(path,s,history,settings)
    restored,snapshots=load_paged_archive(path,settings,max_frames=3,max_points=30)
    assert restored._frame_count==7 and restored.requires_new_segment
    assert snapshots==history
    assert not isinstance(restored.segments[0].frames[0].points,np.ndarray)
    with pytest.raises(ValueError): add(restored,8,segment='a')
    assert add(restored,8,segment='b')
    saved=restored._frame_count
    s.segments[0].frames[0].points.path.write_bytes(b'corrupt')
    with pytest.raises(ValueError): load_paged_archive(path,settings,max_frames=3,max_points=30)
    assert restored._frame_count==saved


@pytest.mark.parametrize('operation', ['save', 'rename', 'fsync', 'reserve'])
def test_disk_failures_do_not_advance_committed_state(tmp_path, monkeypatch, operation):
    import plugins.pointcloud_slam_paged_store as module
    from types import SimpleNamespace
    s=store(tmp_path);add(s,0)
    def fail(*a,**kw): raise OSError('injected disk failure')
    if operation=='save': monkeypatch.setattr(module.np,'save',fail)
    elif operation=='rename': monkeypatch.setattr(module.os,'replace',fail)
    elif operation=='fsync': monkeypatch.setattr(module.os,'fsync',fail)
    else: monkeypatch.setattr(module.shutil,'disk_usage',lambda p:SimpleNamespace(free=0))
    with pytest.raises(StorageError): add(s,1)
    assert s._frame_count==1 and len(s.records)==1
    assert len(list(s.directory.glob('*.npy')))==1


def test_stationary_clouds_remain_healthy_and_storage_fault_is_distinct(tmp_path, monkeypatch):
    from test_pointcloud_slam_stream import stream
    s=stream(max_cloud_points=4,max_points=30)
    s.store=store(tmp_path,min_translation_m=.1,min_rotation_deg=10)
    for i in range(10):
        t=1000+i*100
        s.odom(odom(t,0),now=t/1000)
        s.cloud(t*1_000_000,'camera',[[1,2,3]],now=t/1000)
    assert s.status(now=1.9)['cloud']=='active'
    assert s.status(now=1.9)['keyframes']==1
    monkeypatch.setattr(s.store,'_append_record',lambda *a:(_ for _ in ()).throw(OSError('disk full')))
    s.odom(odom(2000,1),now=2)
    s.cloud(2000000000,'camera',[[1,2,3]],now=2)
    assert s.status(now=2)['integration']=='storage_fault'
    assert s.status(now=2)['odometry']=='tracking'
    assert s.store._frame_count==1


def test_oversized_checkpoint_preserves_previous_file(tmp_path,monkeypatch):
    import plugins.pointcloud_slam_paged_archive as archive
    from test_pointcloud_slam_archive import settings
    s=store(tmp_path/'sessions');add(s,0)
    cfg=settings();cfg['persistent_history']={'max_disk_bytes':10000000,'reserve_bytes':0}
    path=tmp_path/'saved.json';archive.save_paged_archive(path,s,{},cfg);old=path.read_bytes()
    monkeypatch.setattr(archive,'MAX_CHECKPOINT_BYTES',1)
    with pytest.raises(ValueError): archive.save_paged_archive(path,s,{},cfg)
    assert path.read_bytes()==old
