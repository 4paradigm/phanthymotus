"""Synthetic plan checks are not native SLAM acceptance."""
import importlib.util
from pathlib import Path

import pytest

TOOL=Path(__file__).resolve().parents[1]/'tools/rtabmap_async_recovery_smoke.py'
IDENTITY=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]


def module():
    spec=importlib.util.spec_from_file_location('rtabmap_async_recovery_smoke',TOOL)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_async_plan_keeps_integer_capture_times_and_distinct_rendered_origins():
    tool=module();base=1_794_658_670_000_000_001
    frames=tool.plan_frames([['0','1','0','0','outbound'],['1','1','.192','.18','outbound']],
        'async',base_ns=base)
    first,last=frames
    assert first['rgb_stamp_ns']==first['cloud_stamp_ns']==base
    assert last['cloud_stamp_ns']==base+1_500_000_000
    assert last['rgb_stamp_ns']==base+1_400_000_000
    assert type(last['rgb_stamp_ns']) is int
    assert last['rgb_x']==.18
    assert last['cloud_x']==pytest.approx(.19285714285714287)
    # The independently rendered image lies on the continuous GT trajectory.
    assert last['cloud_x']*(14/15)+first['cloud_x']*(1/15)==pytest.approx(.18)
    assert last['external_x']==pytest.approx(.20485714285714288)


def test_positive_plan_excludes_discontinuous_negative_places_without_inventing_poses():
    tool=module()
    rows=[['0','1','0','0','outbound'],['1','0','0','15','novel_negative'],
          ['2','0','0','0','relocalize'],['3','0','0','.2','visual_only']]
    frames=tool.plan_frames(rows,'async')
    assert [f['phase'] for f in frames]==['outbound','relocalize','visual_only']
    assert [f['external_pose'] for f in frames]==[True,False,False]
    assert all(f['external_x'] is None for f in frames[1:])
    negative=tool.plan_frames(rows,'novel-negative')
    assert len(negative)==1 and negative[0]['phase']=='novel_negative'
    assert negative[0]['cloud_x']==negative[0]['rgb_x']==15.


def test_measurement_evidence_requires_observed_motion_and_true_async_headers():
    tool=module()
    records=[dict(external_pose=False,rgb_stamp_ns=100,cloud_stamp_ns=200,
        images=dict(rgb=True,depth=True,local_translation=[0.,0.,0.],rgb_stamp_ns=100,cloud_stamp_ns=200),
        motion=dict(matrix=[[1,0,0,-.01],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                    translation_error_m=.001,verified=2))]
    records[0]['images']['local_translation']=[-.01,0.,0.]
    assert tool.async_evidence(records)['measured_async_frames']==1
    records[0]['images']['local_translation']=[-.01,0.,0.]
    records[0]['motion']['verified']=0
    with pytest.raises(AssertionError,match='measured'):tool.async_evidence(records)
    records[0]['motion']['verified']=2
    records[0]['images']['rgb_stamp_ns']=200
    with pytest.raises(AssertionError,match='header'):tool.async_evidence(records)


def test_native_acceptance_needs_persisted_visual_geometry_and_continued_growth():
    tool=module()
    records=[dict(index=0,external_pose=False,rgb_stamp_ns=100,cloud_stamp_ns=200,
        phase='relocalize',selected_rgb_stamp_ns=100,selected_rgb_frame=0,selected_rgb_x=0.,images=dict(rgb=True,depth=True,local_translation=[0.,0.,0.],rgb_stamp_ns=100,cloud_stamp_ns=200),
        motion=dict(matrix=IDENTITY,verified=2,translation_error_m=.001),
        native=dict(ref=9,loop=2,stats={'Motus/RecoveryState/':2.,'Loop/Visual_inliers/':35.}))]
    records += [dict(index=i,external_pose=False,rgb_stamp_ns=100+i,cloud_stamp_ns=200+i,
        phase='visual_only',images=None,motion=None,
        native=dict(ref=9+i,loop=0,stats={'Motus/RecoveryState/':3.})) for i in (1,2,3)]
    with pytest.raises(AssertionError,match='geometric'):tool.validate_native(records,{2,9,10,11,12},[],'async')
    evidence=tool.validate_native(records,{2,9,10,11,12},[(9,2,1)],'async')
    assert evidence['verified_recoveries'][0]['inliers']==35.
    assert evidence['persisted_visual_tracking_nodes']==[10,11,12]
    records[-1]['native']=None
    with pytest.raises(AssertionError,match='continued'):tool.validate_native(records,{2,9,10,11,12},[(9,2,1)],'async')


def test_negative_validation_rejects_unpersisted_localization_claims():
    tool=module()
    row=dict(index=0,external_pose=False,rgb_stamp_ns=100,cloud_stamp_ns=200,
        images=dict(rgb=True,depth=True,local_translation=[0.,0.,0.],rgb_stamp_ns=100,cloud_stamp_ns=200),
        motion=dict(matrix=IDENTITY,verified=2,translation_error_m=0.),
        native=dict(ref=9,loop=2,stats={'Motus/RecoveryState/':2.,'Loop/Visual_inliers/':35.}))
    with pytest.raises(AssertionError,match='localized'):tool.validate_native([row],{2},[],'novel-negative')


def test_rgb_comparison_keeps_existing_endpoint_and_surface_improvement_thresholds(tmp_path):
    import json
    tool=module();rgb=tmp_path/'rgb';off=tmp_path/'off';rgb.mkdir();off.mkdir()
    (rgb/'evaluation.json').write_text(json.dumps(dict(revisit_endpoint_xyz_error_m=.1,surface_rms_m=.03)))
    (off/'evaluation.json').write_text(json.dumps(dict(revisit_endpoint_xyz_error_m=.6,surface_rms_m=.18)))
    assert tool.compare_runs(rgb,off)['endpoint_improvement_ratio']==pytest.approx(1/6)
    (rgb/'evaluation.json').write_text(json.dumps(dict(revisit_endpoint_xyz_error_m=.2,surface_rms_m=.03)))
    with pytest.raises(AssertionError,match='endpoint'):tool.compare_runs(rgb,off)


def test_origin_change_phase_cannot_pass_by_pausing_all_mapping():
    tool=module()
    records=[dict(phase='external_reset',native=dict(ref=10)),
             dict(phase='external_reset',native=dict(ref=11)),
             dict(phase='continuous_reset',native=dict(ref=12)),
             dict(phase='continuous_reset',native=dict(ref=13))]
    with pytest.raises(AssertionError,match='external_reset'):tool.reset_continuity(records,{})
    with pytest.raises(AssertionError,match='continuous_reset'):tool.reset_continuity(records,{10:records[0],11:records[1],12:records[2]})
    assert tool.reset_continuity(records,{10:records[0],11:records[1],12:records[2],13:records[3]})=={'external_reset':[10,11],'continuous_reset':[12,13]}


def test_positive_plan_retains_rgb_interruption_and_second_recovery_route():
    tool=module()
    rows=[['0','1','0','0','outbound'],['1','0','0','7.5','rgb_gap'],
          ['2','0','0','0','second_relocalize'],['3','0','0','.2','second_visual_only'],
          ['4','0','0','15','novel_negative']]
    frames=tool.plan_frames(rows,'async')
    assert [frame['phase'] for frame in frames]==['outbound','rgb_gap','second_relocalize','second_visual_only']
    assert frames[2]['rgb_x']==0.
    assert frames[2]['cloud_x']<0.  # Preserve the large reverse displacement in continuous GT.


def test_full_positive_requires_two_geometric_recoveries_and_growth_after_each():
    tool=module()
    def row(index,ref,phase,state,anchor=0):
        return dict(index=index,phase=phase,external_pose=False,selected_rgb_stamp_ns=index*100+1,selected_rgb_frame=index,selected_rgb_x=0.,rgb_stamp_ns=index*100+1,
            cloud_stamp_ns=index*100+2,images=dict(rgb=True,depth=True,local_translation=[0.,0.,0.],rgb_stamp_ns=index*100+1,
            cloud_stamp_ns=index*100+2),motion=dict(matrix=IDENTITY,verified=2,translation_error_m=.001),
            native=dict(ref=ref,loop=anchor,stats={'Motus/RecoveryState/':state,'Loop/Visual_inliers/':35.}))
    rows=[row(0,9,'relocalize',2.,2)]+[row(i,9+i,'visual_only',3.) for i in (1,2,3)]
    rows += [row(4,13,'second_relocalize',1.)]+[row(i,9+i,'second_visual_only',3.) for i in (5,6,7)]
    persisted={2,*range(9,17)};links=[(9,2,1)]
    with pytest.raises(AssertionError,match='two'):tool.validate_native(rows,persisted,links,'async')
    rows[4]=row(4,13,'second_relocalize',2.,2);links.append((13,2,1))
    evidence=tool.validate_native(rows,persisted,links,'async')
    assert len(evidence['verified_recoveries'])==2
    rows[-1]['native']=None
    with pytest.raises(AssertionError,match='second_visual_only'):tool.validate_native(rows,persisted,links,'async')


def test_bad_motion_candidate_can_be_reported_as_rejected_without_claiming_mapped_accuracy():
    tool=module()
    row=dict(index=7,external_pose=False,rgb_stamp_ns=100,cloud_stamp_ns=200,
        images=dict(rgb=True,depth=True,local_translation=[0.,0.,0.],rgb_stamp_ns=100,cloud_stamp_ns=200),
        native=dict(ref=99),motion=dict(matrix=IDENTITY,verified=2,translation_error_m=3.))
    evidence=tool.async_evidence([row],persisted_nodes={2})
    assert evidence['unretained_relative_failures']==[dict(index=7,ref=99,translation_error_m=3.)]
    with pytest.raises(AssertionError,match='retained'):tool.async_evidence([row],persisted_nodes={99})


def test_actual_selected_rgb_capture_can_differ_from_current_sent_capture():
    tool=module()
    row=dict(index=7,external_pose=False,rgb_stamp_ns=100,selected_rgb_stamp_ns=50,cloud_stamp_ns=200,
        images=dict(rgb=True,depth=True,local_translation=[0.,0.,0.],rgb_stamp_ns=50,cloud_stamp_ns=200),
        native=dict(ref=9),motion=dict(matrix=IDENTITY,verified=2,translation_error_m=.001))
    assert tool.async_evidence([row],persisted_nodes={9})['retained_async_frames']==1
    assert row['rgb_stamp_ns']==100
    row['images']['rgb_stamp_ns']=100
    with pytest.raises(AssertionError,match='header'):tool.async_evidence([row],persisted_nodes={9})


def test_external_rgb_gap_keeps_the_same_declared_external_origin_as_reset():
    tool=module()
    rows=[['0','1','0','0','outbound'],['1','1','100','0','external_reset'],
          ['2','1','100','0','external_rgb_gap'],['3','1','-100','0','continuous_reset']]
    frames=tool.plan_frames(rows,'async',origin=(7.,-3.,.5))
    assert [frame['external_position'] for frame in frames]==[
        [7.,-3.,.5],[100.,0.,0.],[100.,0.,0.],[-100.,0.,0.]]


def test_final_info_backfills_late_native_results_without_changing_captures():
    tool=module();capture=1_794_658_670_000_000_001
    records=[dict(cloud_stamp_ns=capture,rgb_stamp_ns=capture-100,native=None)]
    native=dict(ref=9,loop=2,stats={'Loop/Visual_inliers/':35.})
    completed=tool.finalize_records(records,{str(capture):native})
    assert completed[0]['native']==native
    assert completed[0]['cloud_stamp_ns']==capture and completed[0]['rgb_stamp_ns']==capture-100


def test_every_new_optimized_node_gets_unique_gt_even_when_info_was_absent():
    tool=module();base=1_794_658_670_000_000_001
    records=[dict(index=0,cloud_stamp_ns=base,native=None),
             dict(index=1,cloud_stamp_ns=base+1_500_000_000,native=None)]
    nodes={1:(1794658670.,None),2:(1794658671.5,None)}
    matched=tool.retained_ground_truth(records,nodes,{1,2})
    assert {node:row['index'] for node,row in matched.items()}=={1:0,2:1}
    with pytest.raises(AssertionError,match='capture GT'):tool.retained_ground_truth(records[:1],nodes,{1,2})
    assert set(tool.retained_ground_truth(records[1:],nodes,{1,2},historical_ids={1}))=={2}


def test_ambiguous_or_duplicate_mapped_capture_cannot_be_guessed():
    tool=module();base=1_794_658_670_000_000_001
    one=dict(index=0,cloud_stamp_ns=base,native=None)
    collision=dict(index=1,cloud_stamp_ns=base+1,native=None)
    with pytest.raises(AssertionError,match='ambiguous'):
        tool.retained_ground_truth([one,collision],{1:(1794658670.,None)},{1})
    with pytest.raises(AssertionError,match='duplicate mapped'):
        tool.retained_ground_truth([one],{1:(1794658670.,None),2:(1794658670.,None)},{1,2})


def test_unretained_motion_cannot_substitute_for_missing_recovery_attachments():
    tool=module()
    rows=[dict(index=0,phase='relocalize',external_pose=False,rgb_stamp_ns=100,cloud_stamp_ns=200,
        images=None,motion=None,native=dict(ref=9,loop=2,stats={
        'Motus/RecoveryState/':2.,'Loop/Visual_inliers/':35.}))]
    rows += [dict(index=i,phase='visual_only',external_pose=False,rgb_stamp_ns=100+i,cloud_stamp_ns=200+i,
        images=None,motion=None,native=dict(ref=9+i,loop=0,stats={'Motus/RecoveryState/':3.})) for i in (1,2,3)]
    rows.append(dict(index=4,phase='visual_only',external_pose=False,rgb_stamp_ns=104,cloud_stamp_ns=204,
        images=dict(rgb=True,depth=True,local_translation=[0.,0.,0.],rgb_stamp_ns=104,cloud_stamp_ns=204),
        motion=dict(matrix=IDENTITY,verified=2,translation_error_m=3.),native=None))
    with pytest.raises(AssertionError,match='recovery.*attachment'):
        tool.validate_native(rows,{2,9,10,11,12},[(9,2,1)],'async')


def test_cold_run_requires_mapping_after_external_poses_return():
    tool=module()
    matched={10:dict(external_pose=False),11:dict(external_pose=True)}
    with pytest.raises(AssertionError,match='external'):tool.cold_external_continuity(matched)
    matched[12]=dict(external_pose=True)
    assert tool.cold_external_continuity(matched)==[11,12]


def test_empty_snapshot_lists_and_zero_points_are_not_archive_acceptance():
    import struct
    tool=module()
    with pytest.raises(AssertionError,match='nonempty'):tool.archive_roundtrip([],[])
    empty=struct.pack('<fffBI',0.,0.,0.,3,0)
    with pytest.raises(AssertionError,match='points'):tool.archive_roundtrip([empty],[empty])
    packet=struct.pack('<fffBIfff',0.,0.,0.,3,1,1.,2.,3.)
    assert tool.archive_roundtrip([packet],[packet])==dict(packets=1,points=1,identical=True)


def test_live_wire_preview_uses_same_capture_optimized_graph_and_cloud(tmp_path):
    import hashlib
    import json
    import struct
    import numpy as np
    tool=module()
    np.save(tmp_path/'points.npy',np.array([[1.,2.,3.]],dtype='<f4'))
    graphs=[dict(stamp_ns=123,frame_id='/map',poses={'2':[0,0,0,0,0,0,1],
        '9':[4,5,6,0,0,0,1]},links=[dict(from_id=9,to_id=2,type=1)])]
    (tmp_path/'graph.json').write_text(json.dumps(graphs))
    (tmp_path/'preview-clouds.json').write_text(json.dumps([dict(stamp_ns=123,frame_id='/map',file='points.npy')]))
    # The stock wire format stores optimized X/Y and heading flags, then XYZ.
    packet=struct.pack('<fffBIfff',4.,5.,0.,39,1,1.,2.,3.)
    (tmp_path/'wire.bin').write_bytes(packet)
    (tmp_path/'preview-packets.json').write_text(json.dumps([dict(file='wire.bin',sha256=hashlib.sha256(packet).hexdigest())]))
    loop=[dict(stamp_ns=123,ref=9,anchor=2,inliers=35)]
    evidence=tool.live_preview_evidence(tmp_path,loop)
    assert evidence['verified'] and evidence['stamp_ns']==123 and evidence['points']==1
    graphs[0]['stamp_ns']=124
    (tmp_path/'graph.json').write_text(json.dumps(graphs))
    with pytest.raises(AssertionError,match='live preview'):tool.live_preview_evidence(tmp_path,loop)
    graphs[0]['stamp_ns']=123;graphs[0]['poses']['9'][0]=99.
    (tmp_path/'graph.json').write_text(json.dumps(graphs))
    with pytest.raises(AssertionError,match='live preview'):tool.live_preview_evidence(tmp_path,loop)


def test_scene_continuation_preserves_original_prefix_and_renders_fixed_new_positions(tmp_path):
    import hashlib
    import cv2
    if not hasattr(cv2,'GaussianBlur'):pytest.skip('actual OpenCV scene rendering required')
    path=TOOL.with_name('slam_visual_scene.py')
    spec=importlib.util.spec_from_file_location('async_smoke_scene',path)
    scene=importlib.util.module_from_spec(spec);spec.loader.exec_module(scene)
    base,extended=tmp_path/'base',tmp_path/'extended'
    assert scene.generate(base,origin=(7.,-3.,.5))==136
    assert scene.generate(extended,origin=(7.,-3.,.5),continuation_frames=8)==144
    prefix=(base/'frames.txt').read_text().splitlines()
    rows=(extended/'frames.txt').read_text().splitlines()
    assert rows[:136]==prefix
    assert (base/'origin.txt').read_bytes()==(extended/'origin.txt').read_bytes()
    for i in range(136):
        for suffix in ('.jpg','.png'):
            assert hashlib.sha256((base/f'{i}{suffix}').read_bytes()).digest()==hashlib.sha256((extended/f'{i}{suffix}').read_bytes()).digest()
    appended=[row.split() for row in rows[136:]]
    assert [float(row[3]) for row in appended]==[1.64,1.82,2.,2.18,2.36,2.54,2.72,2.9]
    assert all(row[1]=='0' and row[4]=='second_visual_only' for row in appended)
    assert all((extended/f'{i}.jpg').is_file() and (extended/f'{i}.png').is_file() for i in range(136,144))


def test_measuring_motion_without_applying_it_to_the_attachment_is_rejected():
    tool=module()
    measured=[[1,0,0,-.012],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    row=dict(index=1,external_pose=False,rgb_stamp_ns=100,cloud_stamp_ns=200,
        images=dict(rgb=True,depth=True,rgb_stamp_ns=100,cloud_stamp_ns=200,local_translation=[0.,0.,0.]),
        native=dict(ref=9),motion=dict(matrix=measured,verified=2,translation_error_m=.001))
    with pytest.raises(AssertionError,match='actual attachment.*measured'):tool.async_evidence([row],persisted_nodes={9})
    row['images']['local_translation']=[-.012,0.,0.]
    assert tool.async_evidence([row],persisted_nodes={9})['applied_transform_components']=='translation_only'
    row['images']['local_matrix']=IDENTITY
    with pytest.raises(AssertionError,match='actual attachment.*measured'):tool.async_evidence([row],persisted_nodes={9})
    row['images']['local_matrix']=measured
    assert tool.async_evidence([row],persisted_nodes={9})['applied_transform_components']=='full_matrix'
    row['images']['local_matrix']=[[0,-1,0,-.012],[1,0,0,0],[0,0,1,0],[0,0,0,1]]
    with pytest.raises(AssertionError,match='actual attachment.*measured'):tool.async_evidence([row],persisted_nodes={9})


def test_synchronous_cold_warmup_is_reported_separately_from_async_recovery():
    tool=module()
    rows=[dict(index=0,phase='reopened',external_pose=False,rgb_stamp_ns=100,cloud_stamp_ns=100,
        selected_rgb_stamp_ns=100,selected_rgb_frame=0,selected_rgb_x=0.,
        images=dict(rgb=True,depth=True,rgb_stamp_ns=100,cloud_stamp_ns=100,
                    local_translation=[0.,0.,0.],local_matrix=IDENTITY),motion=None,
        native=dict(ref=9,loop=2,stats={'Motus/RecoveryState/':2.,'Loop/Visual_inliers/':35.}))]
    rows += [dict(index=i,phase='reopened',external_pose=False,rgb_stamp_ns=100+i*100-10,
        cloud_stamp_ns=100+i*100,selected_rgb_stamp_ns=100+i*100-10,selected_rgb_frame=i,selected_rgb_x=0.,
        images=dict(rgb=True,depth=True,rgb_stamp_ns=100+i*100-10,cloud_stamp_ns=100+i*100,
                    local_translation=[0.,0.,0.],local_matrix=IDENTITY),
        motion=dict(matrix=IDENTITY,verified=2,translation_error_m=.001),
        native=dict(ref=9+i,loop=0,stats={'Motus/RecoveryState/':3.})) for i in (1,2,3)]
    with pytest.raises(AssertionError,match='asynchronous'):tool.validate_native(rows,{2,9,10,11,12},[(9,2,1)],'async')
    evidence=tool.validate_native(rows,{2,9,10,11,12},[(9,2,1)],'async',cold_reopen=True)
    assert len(evidence['verified_synchronous_cold_recoveries'])==1
    assert evidence['verified_async_recoveries']==[]
    assert evidence['retained_async_frames']==3


@pytest.mark.parametrize('promoted',[False,True])
def test_verified_visual_edges_preserve_legacy_and_promoted_event_geometry(promoted):
    tool=module()
    stats={'Motus/RecoveryState/':2.,'Loop/Visual_inliers/':35.}
    if promoted:
        stats.update({'Loop/Visual_inliers/':0.,'Motus/VisualInliers/':999.,
            'Motus/RecoveryNode/':9.,'Motus/RecoveryAnchor/':2.,'Motus/RecoveryInliers/':35.})
    rows=[dict(index=0,phase='relocalize',external_pose=False,rgb_stamp_ns=100,cloud_stamp_ns=200,
        selected_rgb_stamp_ns=100,selected_rgb_frame=0,selected_rgb_x=0.,
        images=dict(rgb=True,depth=True,rgb_stamp_ns=100,cloud_stamp_ns=200,
                    local_translation=[0.,0.,0.],local_matrix=IDENTITY),
        motion=dict(matrix=IDENTITY,verified=2,translation_error_m=.001),
        native=dict(ref=9,loop=0 if promoted else 2,stats=stats))]
    rows += [dict(index=i,phase='visual_only',external_pose=False,rgb_stamp_ns=100+i,
        cloud_stamp_ns=200+i,images=None,motion=None,
        native=dict(ref=9+i,loop=0,stats={'Motus/RecoveryState/':3.})) for i in (1,2,3)]
    evidence=tool.validate_native(rows,{2,9,10,11,12},[(9,2,1)],'async')
    assert evidence.get('verified_visual_edges')==[
        dict(index=0,stamp_ns=200,ref=9,anchor=2,inliers=35.)]
    if promoted:
        stats['Motus/RecoveryInliers/']=0.
        stats['Loop/Visual_inliers/']=999.;rows[0]['native']['loop']=2
        with pytest.raises(AssertionError,match='geometric'):
            tool.validate_native(rows,{2,9,10,11,12},[(9,2,1)],'async')


def test_saved_poses_preserve_the_complete_native_live_optimized_graph():
    import numpy as np
    tool=module()
    graphs=[dict(stamp_ns=123,poses={'2':[0,0,0,0,0,0,1],
        '9':[4,5,6,0,0,2**-.5,2**-.5]})]
    saved={2:np.eye(4)[:3],9:np.array([[0,-1,0,4],[1,0,0,5],[0,0,1,6]],dtype=float)}
    evidence=tool.saved_native_graph_evidence(graphs,saved)
    assert evidence['comparable'] and evidence['verified'] and evidence['nodes']==2
    saved[9][0,3]+=.0322
    evidence=tool.saved_native_graph_evidence(graphs,saved)
    assert evidence['comparable'] and not evidence['verified']
    assert evidence['maximum_translation_change_m']==pytest.approx(.0322)
    saved[9][0,3]-=.0322-5e-7
    assert tool.saved_native_graph_evidence(graphs,saved)['verified']
    saved[9][:,:3]=np.eye(3)
    assert not tool.saved_native_graph_evidence(graphs,saved)['verified']
    saved[9][0,3]=np.nan
    assert not tool.saved_native_graph_evidence(graphs,saved)['verified']


def test_partial_live_graph_cannot_claim_the_saved_global_graph_was_preserved():
    import numpy as np
    tool=module()
    graphs=[dict(stamp_ns=123,poses={'9':[4,5,6,0,0,0,1]})]
    saved={2:np.eye(4)[:3],9:np.eye(4)[:3]}
    evidence=tool.saved_native_graph_evidence(graphs,saved)
    assert not evidence['comparable'] and evidence['verified'] is None
