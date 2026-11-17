"""Offline recording checks; generated payloads here test integrity, not hardware."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

TOOL = Path(__file__).resolve().parents[1] / 'tools/slam_record_replay.py'

def module():
    assert TOOL.is_file(), 'recording replay tool is missing'
    spec=importlib.util.spec_from_file_location('slam_record_replay', TOOL)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result

def record(tmp_path, *, stamp=1794898925701466892):
    data=b'actual bytes are opaque to the integrity reader'
    (tmp_path/'one.cdr').write_bytes(data)
    row=dict(topic='/color',type='sensor_msgs/msg/Image',file='one.cdr',stamp_ns=stamp,
             receipt_ns=stamp+123,sha256=hashlib.sha256(data).hexdigest())
    (tmp_path/'records.jsonl').write_text(json.dumps(row)+'\n')
    return row

def test_recording_retains_integer_nanoseconds_and_verifies_payload(tmp_path):
    row=record(tmp_path)
    assert module().read_records(tmp_path)==[row]
    (tmp_path/'one.cdr').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):module().read_records(tmp_path)

def test_recording_rejects_float_timestamp_and_path_escape(tmp_path):
    record(tmp_path,stamp=1794898925701466892.)
    with pytest.raises(ValueError,match='integer'):module().read_records(tmp_path)
    row=record(tmp_path);row['file']='../one.cdr'
    (tmp_path/'records.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='payload path'):module().read_records(tmp_path)

def test_pairing_is_one_to_one_and_does_not_shift_capture_time():
    tool=module();base=1794898925701466892
    rgb=[dict(stamp_ns=base+i*50_000_000) for i in range(3)]
    depth=[dict(stamp_ns=base-33_000_000+i*50_000_000) for i in range(3)]
    pairs=tool.pair_records(rgb,depth,40_000_000)
    assert [(a['stamp_ns'],b['stamp_ns']) for a,b in pairs]==[(a['stamp_ns'],b['stamp_ns']) for a,b in zip(rgb,depth)]
    assert all(type(a['stamp_ns']) is int for a,b in pairs)
    assert tool.pair_records(rgb,depth,10_000_000)==[]

def test_recovery_requires_visual_inliers_and_persisted_global_edge():
    tool=module()
    frames=[dict(index=3,input_pose_source=None,native=dict(ref=9,loop=2,stats={
        'Motus/RecoveryState/':2.,'Loop/Visual_inliers/':35.})),
        dict(index=4,input_pose_source=None,native=dict(ref=10,loop=0,stats={'Motus/RecoveryState/':3.}))]
    absent=tool.recovery_evidence(frames,[],{2,9,10})
    assert absent['verified_recoveries']==[]
    present=tool.recovery_evidence(frames,[(9,2,1)],{2,9,10})
    assert present['verified_recoveries']==[dict(index=3,ref=9,anchor=2,inliers=35.)]
    assert present['persisted_visual_tracking_nodes']==[10]
    assert not tool.recovery_evidence(frames,[(9,2,0)],{2,9,10})['verified_recoveries']
    frames[0]['native']['stats']['Loop/Visual_inliers/']=0.
    assert not tool.recovery_evidence(frames,[(9,2,1)],{2,9,10})['verified_recoveries']

def test_publication_cycle_pairing_excludes_previous_frameset_depth():
    tool=module();base=1794898925701466892
    colors=[dict(stamp_ns=base+i*50_000_000) for i in range(3)]
    depths=[dict(stamp_ns=base-35_000_000+i*50_000_000) for i in range(4)]
    pairs=tool.pair_publication_cycles(colors,depths)
    assert [(c['stamp_ns'],d['stamp_ns']) for c,d in pairs]==[(c['stamp_ns'],d['stamp_ns']) for c,d in zip(colors,depths[1:])]
    # Missing intermediate depth cannot borrow the next complete cycle's depth.
    pairs=tool.pair_publication_cycles(colors,[depths[0],depths[1],depths[3]])
    assert [c['stamp_ns'] for c,d in pairs]==[colors[0]['stamp_ns'],colors[2]['stamp_ns']]

def test_actual_native_headers_are_compared_to_original_integer_sources():
    tool=module();t=1794898925701466892
    source=[dict(rgb_stamp_ns=t,cloud_stamp_ns=t+700001)]
    attached={t+700001:dict(rgb=True,sensor_stamp_ns=t+700001,laser_scan_stamp_ns=t+700001,rgb_stamp_ns=t)}
    assert tool.header_evidence(source,attached,[dict(stamp_ns=t+700001)])['preserved'] is True
    attached[t+700001]['rgb_stamp_ns']=t+1
    assert tool.header_evidence(source,attached,[dict(stamp_ns=t+700001)])['preserved'] is False
    assert tool.header_evidence(source,{},[])['preserved'] is None

def test_preparation_keeps_numeric_data_with_arbitrary_clock_and_image_frame_labels(tmp_path,monkeypatch):
    """A numerical normalization control, explicitly not hardware evidence."""
    import cv2
    if not all(hasattr(cv2,name) for name in ('COLOR_RGB2BGR','remap','imencode')):
        pytest.skip('numerical image normalization requires actual OpenCV, not the repository test stub')
    from types import SimpleNamespace as Obj
    import numpy as np
    tool=module();stamp=1794898925701466892
    topics={'/rgb':'sensor_msgs/msg/Image','/depth':'sensor_msgs/msg/Image',
            '/rgb-info':'sensor_msgs/msg/CameraInfo','/depth-info':'sensor_msgs/msg/CameraInfo'}
    recording=tmp_path/'recording';recording.mkdir();rows=[]
    for index,(topic,kind) in enumerate(topics.items()):
        payload=topic.encode();filename=str(index)+'.cdr';(recording/filename).write_bytes(payload)
        rows.append(dict(topic=topic,type=kind,file=filename,stamp_ns=stamp+index,receipt_ns=stamp+100,
                         sha256=hashlib.sha256(payload).hexdigest()))
    (recording/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    image_frame=['calibrated_optical']
    def decode(directory,row):
        header=Obj(frame_id='calibrated_optical' if row['topic'].endswith('info') else image_frame[0])
        if row['topic'].endswith('info'):return Obj(width=1,height=1,header=header)
        return Obj(width=1,height=1,header=header,step=3 if row['topic']=='/rgb' else 2,
                   encoding='rgb8' if row['topic']=='/rgb' else '16UC1',is_bigendian=False,
                   data=b'\x10\x20\x30' if row['topic']=='/rgb' else b'\xb0\x04')
    geometry=dict(frame_id='calibrated_optical',color_x=np.zeros((1,1),np.float32),
                  color_y=np.zeros((1,1),np.float32),depth_x=np.zeros((1,1),np.float32),depth_y=np.zeros((1,1),np.float32))
    producer=tmp_path/'producer.py';producer.write_text('# numerical unit control only\n')
    monkeypatch.setattr(tool,'decode',decode)
    monkeypatch.setattr(tool,'load_driver',lambda path:Obj(make_geometry=lambda a,b,registered:geometry,
        camera_declarations=lambda *a:[dict(schema='motus.camera/1',format='image/jpeg',width=1,height=1,
            K=[1.,0.,0.,0.,1.,0.,0.,0.,1.],D=[],vendor={})]))
    evidence=tmp_path/'parameters.json'
    def run(label,sync,frame,output):
        evidence.write_text(json.dumps(dict(parameters={
            'depth_registration':dict(type=1,bool_value=True),'time_domain':dict(type=4,string_value=label),
            'enable_sync_host_time':dict(type=1,bool_value=sync)})))
        image_frame[0]=frame
        tool.prepare_registered(Obj(recording=recording,output=output,registration_evidence=evidence,driver=producer,
            color_topic='/rgb',depth_topic='/depth',color_info_topic='/rgb-info',depth_info_topic='/depth-info'))
    a,b=tmp_path/'a',tmp_path/'b'
    run('global',True,'calibrated_optical',a);run('arbitrary-source-label',False,'different_image_label',b)
    assert (a/'frames.jsonl').read_bytes()==(b/'frames.jsonl').read_bytes()
    assert np.array_equal(np.load(a/'0.npy'),np.load(b/'0.npy'))
    assert (a/'0.jpg').read_bytes()==(b/'0.jpg').read_bytes()

@pytest.mark.parametrize('global_inliers',[20.,42.])
def test_promoted_recovery_event_verifies_the_persisted_global_edge_after_loop_stats_clear(global_inliers):
    tool=module()
    frames=[dict(index=3,input_pose_source=None,native=dict(ref=9,loop=0,stats={
        'Motus/RecoveryState/':2.,'Loop/Visual_inliers/':0.,'Motus/VisualInliers/':0.,
        'Motus/RecoveryNode/':9.,'Motus/RecoveryAnchor/':2.,'Motus/RecoveryInliers/':global_inliers}))]
    result=tool.recovery_evidence(frames,[(2,9,1)],{2,9})
    assert result['verified_recoveries']==[dict(index=3,ref=9,anchor=2.,inliers=global_inliers)]
    assert result['unverified_localized_frames']==[]


@pytest.mark.parametrize('change,remove,links,persisted',[
    ({'Motus/RecoveryNode/':8.},None,[(9,2,1)],{2,9}),
    ({},'Motus/RecoveryNode/',[(9,2,1)],{2,9}),
    ({},'Motus/RecoveryAnchor/',[(9,2,1)],{2,9}),
    ({},'Motus/RecoveryInliers/',[(9,2,1)],{2,9}),
    ({'Motus/RecoveryAnchor/':0.},None,[(9,2,1)],{2,9}),
    ({'Motus/RecoveryAnchor/':3.},None,[(9,2,1)],{2,9}),
    ({'Motus/RecoveryInliers/':19.},None,[(9,2,1)],{2,9}),
    ({},None,[(9,2,2)],{2,9}),
    ({},None,[],{2,9}),
    ({},None,[(9,2,1)],{2}),
])
def test_invalid_or_partial_recovery_event_cannot_fall_back_to_legacy_success(change,remove,links,persisted):
    tool=module()
    stats={'Motus/RecoveryState/':2.,'Loop/Visual_inliers/':35.,'Motus/VisualInliers/':999.,
           'Motus/RecoveryNode/':9.,'Motus/RecoveryAnchor/':2.,'Motus/RecoveryInliers/':42.}
    stats.update(change)
    if remove:stats.pop(remove)
    frames=[dict(index=3,input_pose_source=None,native=dict(ref=9,loop=2,stats=stats))]
    result=tool.recovery_evidence(frames,links,persisted)
    assert result['verified_recoveries']==[]
    assert result['unverified_localized_frames']==[3]


def test_f2m_inliers_and_relocalization_counter_do_not_prove_global_recovery():
    tool=module()
    frames=[dict(index=3,input_pose_source=None,native=dict(ref=9,loop=0,stats={
        'Motus/RecoveryState/':2.,'Motus/VisualInliers/':999.,'Motus/Relocalizations/':100.}))]
    result=tool.recovery_evidence(frames,[(9,2,1)],{2,9})
    assert result['verified_recoveries']==[]
    assert result['unverified_localized_frames']==[3]
