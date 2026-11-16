#!/usr/bin/env python3
"""Check native evidence using geometry and graph edges, never counts alone."""
import argparse
import json
from pathlib import Path
import sqlite3
import struct
import zlib
import numpy as np


def database_evidence(path, maximum_frame=None):
    with sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro',uri=True) as db:
        nodes={i:(stamp,pose) for i,stamp,pose in db.execute('SELECT id,stamp,pose FROM Node WHERE id>0')}
        links=[tuple(row) for row in db.execute('SELECT from_id,to_id,type FROM Link')]
        ids_blob,poses_blob=db.execute('SELECT opt_ids,opt_poses FROM Admin').fetchone()
        ids=np.frombuffer(zlib.decompress(ids_blob[:-12]),'<i4')
        matrices=np.frombuffer(zlib.decompress(poses_blob[:-12]),'<f4').reshape(-1,3,4)
        optimized=dict(zip(map(int,ids),matrices))
        maximum=dict(ratio=0.,absolute_m=0.,edge=None)
        for a,b,kind,transform,information in db.execute('SELECT from_id,to_id,type,transform,information_matrix FROM Link'):
            if a==b or a not in optimized or b not in optimized: continue
            measured=np.frombuffer(transform,'<f4').reshape(3,4)
            predicted=optimized[a][:,:3].T@(optimized[b][:,3]-optimized[a][:,3])
            error=float(np.max(np.abs(measured[:,3]-predicted)))
            diagonal=np.diag(np.frombuffer(information,'<f8').reshape(6,6))[:3]
            valid=diagonal[diagonal>.0001]
            ratio=error*np.sqrt(float(np.min(valid)) if len(valid) else .0001)
            if ratio>maximum['ratio']: maximum=dict(ratio=float(ratio),absolute_m=error,edge=[a,b,kind])
        surface_errors=[]
        linked={i for a,b,_ in links for i in (a,b) if i>0}
        for i,scan,scan_info in db.execute('SELECT id,scan,scan_info FROM Data'):
            if i not in linked or i not in optimized or not scan or not scan_info: continue
            if maximum_frame is not None and nodes[i][0]>maximum_frame+1: continue
            rows,cols,kind=struct.unpack('<iii',scan[-12:])
            if kind!=21: continue
            points=np.frombuffer(zlib.decompress(scan[:-12]),'<f4').reshape(rows*cols,3)[::64]
            local=np.frombuffer(scan_info,dtype='<f4',offset=28,count=12).reshape(3,4)
            points=points@local[:,:3].T+local[:,3]
            pose=optimized[i]; points=points@pose[:,:3].T+pose[:,3]
            x,y,z=points.T
            expected=3.+.24*np.sin(x*1.7)+.18*np.cos(y*2.3)+.14*np.sin(x*3+y)
            surface_errors.extend(np.abs(z-expected))
    return dict(nodes=nodes,links=links,optimized=optimized,surface_errors=surface_errors,
                maximum_saved_linear_constraint_error=maximum)


def core_metrics(prefix):
    prefix=Path(prefix)
    frames=[json.loads(line) for line in prefix.with_suffix('.jsonl').read_text().splitlines()]
    db=database_evidence(prefix.with_suffix('.db'),maximum_frame=61)
    edges={tuple(sorted((a,b))) for a,b,kind in db['links'] if kind==1}
    visual=[]
    for frame in frames:
        pair=tuple(sorted((frame['ref'],frame['loop'])))
        if frame['phase']=='revisit' and frame.get('loop_inliers',0)>=20 and pair in edges:
            old=db['nodes'].get(frame['loop'])
            if old and frame['frame']-(old[0]-1)>=10: visual.append(frame)
    revisit=[frame for frame in frames if frame['phase']=='revisit']
    residuals=np.asarray(db['surface_errors'])
    trajectory=np.asarray([np.linalg.norm(db['optimized'][f['ref']][:,3]-[f['truth_x'],0,0])
        for f in frames if f['phase'] in ('outbound','revisit') and f['ref'] in db['optimized']])
    endpoint=db['optimized'][revisit[-1]['ref']][:,3]
    result=dict(frames=len(frames),native_visual_global_edges=len(visual),
        visual_examples=visual[-3:],live_endpoint_x_error_m=abs(revisit[-1]['x']-revisit[-1]['truth_x']),
        saved_endpoint_xyz_error_m=float(np.linalg.norm(endpoint-[revisit[-1]['truth_x'],0,0])),
        saved_trajectory_xyz_rms_m=float(np.sqrt(np.mean(trajectory**2))),
        saved_trajectory_xyz_max_m=float(np.max(trajectory)),
        surface_rms_m=float(np.sqrt(np.mean(residuals**2))),surface_p95_m=float(np.percentile(residuals,95)),
        database_nodes=len(db['nodes']),saved_optimized_nodes=len(db['optimized']),
        maximum_saved_linear_constraint_error=db['maximum_saved_linear_constraint_error'],
        link_types={str(t):sum(kind==t for _,_,kind in db['links']) for t in (0,1,2,3)})
    return result,frames,db


def evaluate(rgb, no_rgb, recovery):
    enabled,_,_=core_metrics(rgb); disabled,_,_=core_metrics(no_rgb)
    restored,frames,db=core_metrics(recovery)
    assert enabled['native_visual_global_edges']>0, 'no visual-inlier-verified global revisit edge'
    assert enabled['saved_endpoint_xyz_error_m']<.15, 'RGB endpoint does not close'
    assert enabled['saved_endpoint_xyz_error_m']<disabled['saved_endpoint_xyz_error_m']*.3, 'no measured trajectory improvement'
    assert enabled['surface_rms_m']<disabled['surface_rms_m']*.7, 'no measured map alignment improvement'
    negative=[f for f in frames if f['phase'] in ('similar_negative','novel_negative')]
    anchor_nodes=next(f['nodes'] for f in reversed(frames) if f['phase']=='revisit')
    assert len(negative)>=8 and all(f['relocations']==0 and f['state']==1 and f['nodes']==anchor_nodes for f in negative), 'negative relocated or grew the map'
    localized=[f for f in frames if f['state']==2]
    assert len(localized)>=2 and all(f['loop_inliers']>=20 for f in localized), 'missing geometric recovery evidence'
    for phase in ('visual_only','second_visual_only'):
        tracked=[f for f in frames if f['phase']==phase and f['state']==3]
        assert len(tracked)>=3 and tracked[-1]['nodes']>tracked[0]['nodes']+2, 'localized without continued mapping'
        assert max(abs(f['x']-f['truth_x']) for f in tracked)<.2, 'visual trajectory misaligned'
    for phase in ('external_reset','continuous_reset'):
        samples=[f for f in frames if f['phase']==phase]
        assert samples and max(abs(f['x']-f['truth_x']) for f in samples)<.2, 'external origin reset broke alignment'
    gap=[f for f in frames if f['phase']=='rgb_gap']
    assert gap and len({f['nodes'] for f in gap})==1 and all(f['state']==1 and f['nodes']>0 for f in gap), 'missing-image frames changed or cleared the map'
    result=dict(rgb=enabled,no_rgb=disabled,recovery=restored,
        recoveries=[dict(frame=f['frame'],x=f['x'],truth_x=f['truth_x'],inliers=f['loop_inliers'],nodes=f['nodes']) for f in localized],
        negative_frames=len(negative),visual_only_frames=sum(f['state']==3 for f in frames),
        max_live_reset_x_error_m=max(abs(f['x']-f['truth_x']) for f in frames if 'reset' in f['phase']),
        max_saved_reset_xyz_error_m=max(float(np.linalg.norm(db['optimized'][f['ref']][:,3]-[f['truth_x'],0,0]))
            for f in frames if 'reset' in f['phase'] and f['ref'] in db['optimized']),
        classification='real pinned native core on deterministic synthetic inputs; no robot motion')
    return result


def evaluate_ros(directory, previous=None):
    """Validate actual gateway/native output; absent Info never means tracking."""
    directory=Path(directory)
    result=json.loads((directory/'result.json').read_text())
    frames=[json.loads(line) for line in (directory/'info.jsonl').read_text().splitlines()]
    databases=list((directory/'session').rglob('*.db'))
    db=database_evidence(databases[0] if databases else directory/'restored.db')
    old=database_evidence(next((Path(previous)/'session').rglob('*.db'))) if previous else None
    historical=max(old['nodes']) if old else 0
    edges={tuple(sorted((a,b))) for a,b,kind in db['links'] if kind==1}
    origin=np.asarray(result.get('scene_origin',[0,0,0]))
    assert result['segments']==1 and result['archive_reopened_identically'], 'process/map archive continuity failed'
    root=min(db['optimized'])
    assert np.linalg.norm(db['optimized'][root][:,3]-origin)<1e-4, 'original map origin changed'
    if old:
        assert root==min(old['optimized']) and len(db['optimized'])>len(old['optimized'])+3, 'reopen did not continue original graph'
    phases={}
    recoveries=[]
    visual_edges=[]
    for frame in frames:
        phase=phases.setdefault(frame['phase'],dict(observations=0,retained_nodes=[],errors=[],tracking_nodes=[]))
        phase['observations']+=1
        native=frame['native']
        if not native: continue
        node=native['ref'];stats=native['stats'];state=stats.get('Motus/RecoveryState/',-1)
        retained=node in db['optimized'] and node>historical
        if retained:
            truth=frame.get('truth_position',[frame['truth_x'],0,0])
            phase['retained_nodes'].append(node)
            phase['errors'].append(float(np.linalg.norm(db['optimized'][node][:,3]-truth)))
            if state==3: phase['tracking_nodes'].append(node)
        verified=native['loop']>0 and stats.get('Loop/Visual_inliers/',0)>=20 and tuple(sorted((node,native['loop']))) in edges
        if verified:
            visual_edges.append(dict(frame=frame['frame'],phase=frame['phase'],ref=node,anchor=native['loop'],
                inliers=stats['Loop/Visual_inliers/'],native_ms=stats.get('RtabmapROS/TimeRtabmap/ms')))
        if state==2:
            assert verified and retained, 'recovery lacks native geometry/persisted global edge'
            visual_edges[-1]['capture_elapsed_since_phase_start_s']=(frame['stamp_ns']-
                next(f['stamp_ns'] for f in frames if f['phase']==frame['phase']))/1e9
            if 'sent_monotonic' in frame:
                visual_edges[-1]['observation_to_info_wall_ms']=(native['received_monotonic']-frame['sent_monotonic'])*1000
                start=next(f for f in frames if f['phase']==frame['phase'])
                visual_edges[-1]['phase_start_to_recovery_wall_s']=native['received_monotonic']-start['sent_monotonic']
            recoveries.append(visual_edges[-1])
        if frame['phase'] in ('similar_negative','novel_negative','rgb_gap'):
            assert state==1 and not retained and native['loop']==0, 'unlocalized observation grew map'
    if old:
        assert recoveries and any(f['anchor']<=historical for f in recoveries), 'no geometric match to reopened history'
        assert len(phases['reopened']['tracking_nodes'])>=3, 'reopen localized without continued VO mapping'
        for f in frames:
            if f['phase']=='depth_gap':
                assert f['native'] is None, 'missing depth generated a synthetic native update'
                assert not any(i>historical and abs(stamp-f['stamp_ns']/1e9)<1e-6
                    for i,(stamp,_) in db['nodes'].items()), 'missing depth grew the map'
    else:
        assert any(f['phase']=='revisit' and (f['frame']+10)*1.5-db['nodes'][f['anchor']][0]>=15.
            for f in visual_edges), 'no actual historical revisit global edge'
        assert sum(phases[p]['observations'] for p in ('similar_negative','novel_negative'))==8
        assert len(recoveries)>=2, 'repeated loss recovery missing'
        for name in ('visual_only','second_visual_only'):
            assert len(phases[name]['tracking_nodes'])>=3, 'relocalized without continued VO mapping'
    for phase in phases.values():
        phase['max_saved_xyz_error_m']=max(phase.pop('errors'),default=None)
        if phase['max_saved_xyz_error_m'] is not None:
            assert phase['max_saved_xyz_error_m']<.35, 'saved map does not align to original world'
    graphs=json.loads((directory/'graph.json').read_text())
    assert graphs and any(str(root) in g['poses'] for g in graphs), 'no optimized graph display evidence'
    live_root=next(g['poses'][str(root)][:3] for g in reversed(graphs) if str(root) in g['poses'])
    assert np.linalg.norm(np.asarray(live_root)-origin)<1e-4, 'live map origin changed'
    archive=directory/'archive'/'maps'/result['map_id']
    saved=json.loads((archive/f"{result['revision']}.json").read_text())
    packet=(archive/saved['segments'][0]['snapshot']).read_bytes()
    marker_x,marker_y,_,_,count=struct.unpack_from('<fffBI',packet)
    marker_error=float(np.linalg.norm(np.asarray([marker_x,marker_y])-db['optimized'][max(db['optimized'])][:2,3]))
    assert marker_error<1e-4, 'saved display marker is not the optimized pose'
    points=np.frombuffer(packet,dtype='<f4',offset=17).reshape(count,3)-origin
    x,y,z=points.T
    surface=np.abs(z-(3.+.24*np.sin(x*1.7)+.18*np.cos(y*2.3)+.14*np.sin(x*3+y)))
    assert np.isfinite(points).all() and float(np.median(surface))<.15, 'saved display geometry is misaligned'
    return dict(classification='real pinned ROS gateway/native replay, deterministic synthetic scene; no robot',
        original_origin=origin.tolist(),historical_max_node=historical,saved_optimized_nodes=len(db['optimized']),
        maximum_saved_linear_constraint_error=db['maximum_saved_linear_constraint_error'],
        saved_preview=dict(points=count,marker_xy_error_m=marker_error,surface_median_m=float(np.median(surface))),
        recoveries=recoveries,visual_edges=visual_edges,phases=phases,
        rgb_preparation=result['statistics']['rgb_preparation'],segments=result['segments'],
        archive_reopened_identically=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb');parser.add_argument('--no-rgb');parser.add_argument('--recovery')
    parser.add_argument('--ros');parser.add_argument('--previous');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.ros: result=evaluate_ros(args.ros,args.previous)
    else:
        if not all((args.rgb,args.no_rgb,args.recovery)): parser.error('provide --ros or all core inputs')
        result=evaluate(args.rgb,args.no_rgb,args.recovery)
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
