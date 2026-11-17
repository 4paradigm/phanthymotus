#!/usr/bin/env python3
"""Replay the existing metric scene with genuinely separate RGB/scan captures.

Use slam_visual_scene.py to generate inputs first. This runner renders each scan
at a second point on the continuous synthetic trajectory, keeps both ground
truths, and uses the production ScanMotion/native recovery without pose injection.
Native execution requires an isolated ROS domain and the pinned patched overlay.
"""
import argparse
from array import array
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import uuid
import zlib


def finalize_records(records, infos):
    complete={int(stamp):value for stamp,value in dict(infos).items()}
    return [dict(row,native=deepcopy(complete.get(row['cloud_stamp_ns'],row['native']))) for row in records]


def retained_ground_truth(records, nodes, optimized_ids, historical_ids=()):
    captures={}
    for row in records:
        sec,nsec=divmod(row['cloud_stamp_ns'],10**9)
        captures.setdefault(sec+nsec*1e-9,[]).append(row)
    matched={};used=set()
    for node in sorted(set(optimized_ids)-set(historical_ids)):
        assert node in nodes, 'optimized node has no saved Node timestamp'
        candidates=captures.get(nodes[node][0],[])
        assert candidates, 'new optimized node lacks capture GT: '+str(node)
        assert len(candidates)==1, 'ambiguous capture GT for optimized node: '+str(node)
        row=candidates[0]
        assert row['cloud_stamp_ns'] not in used, 'duplicate mapped capture: '+str(row['cloud_stamp_ns'])
        used.add(row['cloud_stamp_ns']);matched[node]=row
    return matched


def cold_external_continuity(matched):
    external=sorted(node for node,row in matched.items() if row['external_pose'])
    assert len(external)>=2, 'cold reopen did not continue mapping after external poses returned'
    return external


def archive_roundtrip(expected, reopened):
    import numpy as np
    assert expected and reopened, 'archive acceptance requires nonempty snapshot packets'
    points=0
    for packet in expected:
        assert len(packet)>=17, 'snapshot packet is truncated'
        count=struct.unpack_from('<I',packet,13)[0]
        assert count>0 and len(packet)==17+count*12, 'snapshot must contain actual nonempty points'
        assert np.isfinite(np.frombuffer(packet,dtype='<f4',offset=17)).all(), 'snapshot points are not finite'
        points+=count
    assert reopened==expected, 'saved optimized snapshots changed on reopening'
    return dict(packets=len(expected),points=points,identical=True)


def saved_native_graph_evidence(graphs, optimized):
    import numpy as np
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from plugins.pointcloud_slam_transforms import pose_matrix
    ids=set(optimized)
    for graph in reversed(graphs):
        if not ids or not ids.issubset({int(node) for node in graph['poses']}):continue
        changes=[]
        for node in sorted(ids):
            values=graph['poses'][str(node)]
            native=pose_matrix(SimpleNamespace(position=values[:3],quaternion=values[3:]))[:3]
            change=np.asarray(optimized[node])-native
            changes.append((node,float(np.max(np.abs(change))),float(np.linalg.norm(change[:,3]))))
        worst=max(changes,key=lambda item:item[1])
        return dict(comparable=True,verified=all(item[1]<=1e-6 for item in changes),stamp_ns=graph['stamp_ns'],
            nodes=len(ids),serialization_tolerance=1e-6,worst_node=worst[0],
            maximum_matrix_component_change=worst[1],
            maximum_translation_change_m=max(item[2] for item in changes))
    return dict(comparable=False,verified=None,
        reason='no native live MapGraph covered all final saved optimized nodes')


def validate_selected_sources(records, frames):
    import numpy as np
    sources={frame['rgb_stamp_ns']:frame for frame in frames if frame.get('rgb_input_kind',
        'omitted' if frame['phase']=='rgb_gap' else 'invalid' if frame['phase']=='external_rgb_gap' else 'fixture')
        in ('fixture','black')}
    for row in records:
        image=row.get('images')
        if not image or not image['rgb']:continue
        source=sources.get(image['rgb_stamp_ns'])
        assert source and row.get('selected_rgb_stamp_ns')==source['rgb_stamp_ns'], 'RGB attachment has no original selected capture'
        assert (row.get('selected_rgb_frame')==source['frame'] and row.get('selected_rgb_x')==source['rgb_x']), 'RGB attachment selected source geometry differs from capture GT'
        assert row.get('selected_rgb_kind','fixture')==source.get('rgb_input_kind','fixture'), 'RGB attachment selected source kind differs from sent capture'
        motion=row.get('motion')
        if motion:
            matrix=np.asarray(motion['matrix'],dtype=float)
            assert matrix.shape==(4,4) and np.isfinite(matrix).all(), 'measured motion matrix is invalid'
            motion['translation_error_m']=float(np.linalg.norm(matrix[:3,3]-[source['rgb_x']-row['cloud_x'],0.,0.]))


def live_preview_evidence(directory, visual_edges, *, pose_reference=None, require_visual_loop=True):
    import numpy as np
    from types import SimpleNamespace
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from plugins.pointcloud_slam_preview import encode_mapping
    from plugins.pointcloud_slam_transforms import pose_matrix
    directory=Path(directory)
    graphs=json.loads((directory/'graph.json').read_text())
    clouds=json.loads((directory/'preview-clouds.json').read_text())
    packets=json.loads((directory/'preview-packets.json').read_text())
    if (directory/'preview-external-packets.json').is_file():
        packets+=json.loads((directory/'preview-external-packets.json').read_text())
    assert visual_edges or not require_visual_loop, 'live preview lacks a verified visual loop'
    first_loop=min((edge['stamp_ns'] for edge in visual_edges),default=0)
    received={}
    for packet in packets:
        raw=(directory/packet['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==packet['sha256'], 'live preview packet hash changed'
        received[packet['sha256']]=raw
    for graph in reversed(graphs):
        if graph['stamp_ns']<first_loop:continue
        ids=[int(node) for node in graph['poses'] if int(node)>0]
        if not ids:continue
        node=max(ids);raw_pose=graph['poses'][str(node)]
        pose=pose_matrix(SimpleNamespace(position=raw_pose[:3],quaternion=raw_pose[3:]))
        for cloud in clouds:
            if (cloud['stamp_ns'],cloud['frame_id'])!=(graph['stamp_ns'],graph.get('frame_id')):continue
            points=np.load(directory/cloud['file'])
            if not len(points):continue
            expected=encode_mapping(points,pose,max_points=10000,pose_reference=pose_reference)
            digest=hashlib.sha256(expected).hexdigest()
            if received.get(digest)!=expected:continue
            marker_x,marker_y=struct.unpack_from('<ff',expected)
            marker_error=float(np.linalg.norm([marker_x-pose[0,3],marker_y-pose[1,3]]))
            assert marker_error<1e-4, 'live preview marker does not use the optimized graph pose'
            return dict(verified=True,stamp_ns=graph['stamp_ns'],frame_id=cloud['frame_id'],
                optimized_node=node,points=len(points),packet_sha256=digest,
                marker_xy_error_m=marker_error,after_verified_visual_loop=bool(visual_edges))
    raise AssertionError('no actual live preview matched a same-capture optimized graph and cloud')


def plan_frames(rows, mode, *, base_ns=1_794_658_670_000_000_001, origin=(0.,0.,0.)):
    phases={'outbound','revisit','relocalize','visual_only','external_reset',
            'external_rgb_gap','continuous_reset','rgb_gap','second_relocalize',
            'second_visual_only','reopened','depth_gap'}
    if mode=='no-rgb':phases={'outbound','revisit'}
    elif mode=='vo-continuation':phases={'outbound'}
    elif mode=='similar-negative':phases={'similar_negative'}
    elif mode=='novel-negative':phases={'novel_negative'}
    frames=[];seen=set()
    last_gap_frame=next((row[0] for row in reversed(rows) if row[4]=='rgb_gap'),None)
    period,offset=1_500_000_000,100_000_000
    for number,external,external_x,truth_x,phase in rows:
        if phase not in phases:continue
        x=float(truth_x);stamp=base_ns+len(frames)*period
        # Existing JPEGs were rendered at x. Solve the next scan position so
        # piecewise-linear GT passes through that exact image capture position.
        cloud_x=x if not frames else (x-offset/period*frames[-1]['cloud_x'])/(1-offset/period)
        has_external=bool(int(external)) and not (mode=='vo-continuation' and len(frames)>=8)
        supplied_x=float(external_x)+cloud_x-x if has_external else None
        supplied_position=[supplied_x,0.,0.] if supplied_x is not None else None
        if supplied_position is not None and phase in ('outbound','revisit'):
            supplied_position=[a+b for a,b in zip(supplied_position,origin)]
        kind='omitted' if mode=='no-rgb' or phase=='rgb_gap' else 'invalid' if phase=='external_rgb_gap' else 'fixture'
        # The first gap scan can lose ICP and the second seeds its new run;
        # only the last gap capture has measured coverage for both exposures.
        if mode=='async' and ((phase=='relocalize' and phase not in seen) or
                (phase=='rgb_gap' and number==last_gap_frame)):kind='black'
        seen.add(phase)
        frames.append(dict(frame=int(number),index=len(frames),phase=phase,rgb_input_kind=kind,
            rgb_stamp_ns=stamp-offset if frames else stamp,cloud_stamp_ns=stamp,
            rgb_x=x,cloud_x=cloud_x,external_pose=has_external,
            external_x=supplied_x,external_position=supplied_position))
        if mode=='vo-continuation':frames[-1]['external_pose_withheld']=not has_external
    if not frames:raise ValueError('scene has no frames for '+mode)
    return frames


def rgb_payload(kind, source):
    if kind=='omitted':return None
    if kind=='invalid':return b'invalid JPEG'
    if kind=='black':
        import cv2
        import numpy as np
        ok,data=cv2.imencode('.jpg',np.zeros((240,320,3),dtype=np.uint8))
        assert ok, 'black JPEG encoding failed'
        return data.tobytes()
    return source


def async_evidence(records, persisted_nodes=None):
    import numpy as np
    measured=[];unretained_failures=[];retained=[];components=[]
    for row in records:
        rgb_stamp=row.get('selected_rgb_stamp_ns',row['rgb_stamp_ns'])
        if row['external_pose'] or rgb_stamp==row['cloud_stamp_ns']:continue
        image=row.get('images')
        if not image or not image['rgb'] or not image['depth']:continue
        assert image['rgb_stamp_ns']==rgb_stamp, 'RGB capture header was changed'
        assert image['cloud_stamp_ns']==row['cloud_stamp_ns'], 'cloud capture header was changed'
        motion=row.get('motion')
        assert motion and motion['verified']>0, 'no measured local scan motion for asynchronous RGB'
        native=row.get('native')
        accepted=persisted_nodes is None or native is not None and native['ref'] in persisted_nodes
        if accepted:
            assert motion['translation_error_m']<.05, 'retained relative motion disagrees with synthetic GT'
            matrix=np.asarray(motion['matrix'],dtype=float)
            assert matrix.shape==(4,4) and np.isfinite(matrix).all(), 'measured attachment matrix is invalid'
            if image.get('local_matrix') is not None:
                actual=np.asarray(image['local_matrix'],dtype=float)
                assert actual.shape==(4,4) and np.allclose(actual,matrix,atol=1e-9,rtol=0), 'actual attachment did not apply the measured full transform'
                components.append('full_matrix')
            else:
                actual=np.asarray(image.get('local_translation'),dtype=float)
                assert actual.shape==(3,) and np.allclose(actual,matrix[:3,3],atol=1e-9,rtol=0), 'actual attachment did not apply the measured translation'
                components.append('translation_only')
            retained.append(row)
        elif motion['translation_error_m']>=.05:
            unretained_failures.append(dict(index=row['index'],ref=native['ref'] if native else None,
                                           translation_error_m=motion['translation_error_m']))
        measured.append(row)
    assert measured, 'no measured asynchronous pose-null frame reached native mapping'
    return dict(measured_async_frames=len(measured),
        retained_async_frames=len(retained),unretained_relative_failures=unretained_failures,
        applied_transform_components='full_matrix' if components and all(c=='full_matrix' for c in components) else 'translation_only' if components else 'not_retained',
        maximum_relative_translation_error_m=max(row['motion']['translation_error_m'] for row in measured),
        maximum_retained_relative_translation_error_m=max((row['motion']['translation_error_m'] for row in retained),default=None))


def validate_native(records, persisted_nodes, links, mode, *, cold_reopen=False):
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from slam_record_replay import recovery_evidence
    rows=[dict(row,input_pose_source='supplied external pose' if row['external_pose'] else None)
          for row in records]
    evidence=recovery_evidence(rows,links,persisted_nodes)
    if mode=='no-rgb':
        assert not any(row.get('images',{}).get('rgb') for row in records if row.get('images')), 'RGB-off input contained imagery'
        assert not evidence['verified_recoveries'], 'RGB-off recovered visually'
    elif mode.endswith('-negative'):
        assert any(row['native'] for row in records), 'negative query has no native Info evidence'
        assert all(row['native'] is None or (row['native']['stats'].get('Motus/RecoveryState/')==1 and
                   row['native']['loop']==0) for row in records), 'negative query claimed localized tracking'
        assert not any(row['native'] and row['native']['ref'] in persisted_nodes for row in records), 'negative query grew connected map'
        assert not evidence['verified_recoveries'], 'negative query recovered'
        evidence.update(async_evidence(records,persisted_nodes))
    elif mode=='vo-continuation':
        assert not evidence['verified_recoveries'] and not evidence['unverified_localized_frames'], 'VO continuation incorrectly relocalized'
        null_rows=[row for row in records if not row['external_pose']]
        assert null_rows, 'VO continuation has no external-null observations'
        assert any(row['external_pose'] and row['index']<null_rows[0]['index'] and row['native'] and
            row['native']['stats'].get('Motus/VisualInliers/',0)>=20 and
            row.get('images') and row['images']['rgb'] for row in records), 'VO continuation lacks a measured paired external anchor'
        assert len(evidence['persisted_visual_tracking_nodes'])>=3, 'VO continuation did not continue persisted mapping'
        assert all(not row['native'] or (row['native']['stats'].get('Motus/RecoveryState/')!=2 and
            row['native']['stats'].get('Motus/RecoveryNode/',0)==0) for row in null_rows), 'VO continuation emitted a recovery event'
        continued=[row for row in null_rows if row['native'] and
            row['native']['stats'].get('Motus/RecoveryState/')==3 and row['native']['ref'] in persisted_nodes]
        assert all(row.get('images') and row['images']['rgb'] and row['images']['depth'] and
            row.get('motion') and row.get('selected_rgb_kind')=='fixture' and
            row['native']['stats'].get('Motus/VisualInliers/',0)>=20 for row in continued), 'VO continuation lacks actual measured attachment evidence'
        evidence['continued_without_relocalization_nodes']=evidence['persisted_visual_tracking_nodes']
        evidence.update(async_evidence(continued,persisted_nodes))
        assert evidence['retained_async_frames']==len(continued), 'VO continuation attachment was not retained'
    else:
        import numpy as np
        assert evidence['verified_recoveries'] and not evidence['unverified_localized_frames'], 'no verified geometric recovery'
        assert len(evidence['persisted_visual_tracking_nodes'])>=3, 'no continued visual-only map growth'
        recovered={item['index'] for item in evidence['verified_recoveries']}
        recovery_rows=[row for row in records if row['index'] in recovered]
        async_recovery_rows=[];sync_cold=set()
        for row in recovery_rows:
            assert row.get('selected_rgb_kind','fixture')!='black', 'black capture cannot verify geometric recovery'
            image,motion=row.get('images'),row.get('motion')
            assert (image and image['rgb'] and image['depth'] and
                    row.get('selected_rgb_stamp_ns') is not None and
                    row.get('selected_rgb_frame') is not None and row.get('selected_rgb_x') is not None), 'verified recovery lacks raw asynchronous attachment/source evidence'
            if (cold_reopen and row['index']==records[0]['index'] and
                    row['selected_rgb_stamp_ns']==row['cloud_stamp_ns']):
                assert image['rgb_stamp_ns']==image['cloud_stamp_ns']==row['cloud_stamp_ns'], 'cold warmup header changed'
                actual=np.asarray(image.get('local_matrix'),dtype=float)
                assert actual.shape==(4,4) and np.allclose(actual,np.eye(4),atol=1e-9,rtol=0), 'synchronous cold warmup did not apply the supplied identity mount'
                sync_cold.add(row['index']);continue
            assert motion, 'verified recovery lacks measured asynchronous motion'
            matrix=np.asarray(motion.get('matrix'),dtype=float)
            assert matrix.shape==(4,4) and np.isfinite(matrix).all(), 'verified recovery attachment lacks a measured matrix'
            async_recovery_rows.append(row)
        if async_recovery_rows:
            attachment=async_evidence(async_recovery_rows,persisted_nodes)
            assert attachment['retained_async_frames']==len(async_recovery_rows), 'verified recovery lacks retained asynchronous attachment evidence'
        evidence['verified_recovery_async_attachments']=len(async_recovery_rows)
        evidence['verified_synchronous_cold_recoveries']=[item for item in evidence['verified_recoveries'] if item['index'] in sync_cold]
        evidence['verified_async_recoveries']=[item for item in evidence['verified_recoveries'] if item['index'] not in sync_cold]
        phases={row['phase'] for row in records}
        if 'second_relocalize' in phases:
            assert len(evidence['verified_recoveries'])>=2, 'full route requires two persisted geometric recoveries'
        tracking={}
        for phase in ('visual_only','second_visual_only'):
            if phase not in phases:continue
            tracking[phase]=sorted({row['native']['ref'] for row in records if row['phase']==phase and
                row['native'] and row['native']['stats'].get('Motus/RecoveryState/')==3 and
                row['native']['ref'] in persisted_nodes})
            assert len(tracking[phase])>=3, phase+' lacks continued visual-only mapping after recovery'
        evidence['tracking_by_phase']=tracking
        evidence.update(async_evidence(records,persisted_nodes))
        assert evidence['retained_async_frames']>=1, 'no retained asynchronous motion attachment was verified'
    blackouts=[row for row in records if row.get('rgb_input_kind')=='black']
    losses=[]
    for i,black in enumerate(blackouts):
        observed=[row for row in records if row.get('selected_rgb_kind')=='black' and
            row.get('selected_rgb_stamp_ns')==black['rgb_stamp_ns'] and row.get('images') and
            row['images']['rgb'] and row['images']['depth'] and row['images'].get('rgb_all_zero') is True]
        assert observed, 'black injection has no actual featureless native receipt'
        lost=[row for row in observed if row['native'] and
            row['native']['stats'].get('Motus/RecoveryState/')==1 and
            0<=row['native']['stats'].get('Motus/VisualInliers/',20)<20]
        assert lost, 'black receipt did not demonstrate native VO loss'
        end=blackouts[i+1]['index'] if i+1<len(blackouts) else float('inf')
        assert any(lost[0]['index']<item['index']<end for item in evidence['verified_recoveries']), 'black loss lacks subsequent normal-image geometric recovery'
        losses.append(dict(source_index=black['index'],observation_index=lost[0]['index'],
            rgb_stamp_ns=black['rgb_stamp_ns'],native_ref=lost[0]['native']['ref']))
    evidence['verified_blackout_losses']=losses
    recovered={item['index']:item for item in evidence['verified_recoveries']}
    global_edges={tuple(sorted((a,b))) for a,b,kind in links if kind==1}
    visual_edges=[]
    for row in records:
        native=row['native']
        if not native or native['ref'] not in persisted_nodes:continue
        event=recovered.get(row['index'])
        anchor=event['anchor'] if event else native['loop']
        inliers=event['inliers'] if event else native['stats'].get('Loop/Visual_inliers/',0)
        if anchor>0 and inliers>=20 and tuple(sorted((native['ref'],anchor))) in global_edges:
            visual_edges.append(dict(index=row['index'],stamp_ns=row['cloud_stamp_ns'],
                ref=native['ref'],anchor=int(anchor),inliers=inliers))
    evidence['verified_visual_edges']=visual_edges
    return evidence


def reset_continuity(records, matched):
    retained={}
    for phase in ('external_reset','continuous_reset'):
        selected=[row for row in records if row['phase']==phase]
        if not selected:continue
        retained[phase]=sorted(node for node,row in matched.items() if row['phase']==phase)
        assert len(retained[phase])>=2, phase+' did not continue mapping after numerical origin change'
    return retained


def scan_depth(x, phase):
    """Same fixed surface and rays as slam_visual_scene.generate; no SLAM here."""
    import numpy as np
    v,u=np.indices((240,320),dtype=np.float32)
    rx,ry=(u-160.)/260.,(v-120.)/260.
    z=np.full_like(rx,3.)
    for _ in range(16):
        wx,wy=x+rx*z,ry*z
        z=3.+.24*np.sin(wx*1.7)+.18*np.cos(wy*2.3)+.14*np.sin(wx*3+wy)
    if phase=='similar_negative':z=12.+2.*np.sin(u*.17)*np.cos(v*.13)
    # Match the source scene's 1 mm depth quantization.
    z=np.rint(z*1000).astype('float32')/1000.
    return np.stack((rx*z,ry*z,z),axis=-1).reshape(-1,3)


def run(scene, output, mode='async', resume_run=None):
    domain=os.environ.get('ROS_DOMAIN_ID','')
    if not domain.isdecimal() or not 1<=int(domain)<=232:
        raise ValueError('set an isolated nonzero ROS_DOMAIN_ID')
    if mode.endswith('-negative') and resume_run is None:
        raise ValueError('negative query requires a saved reference map via --resume-run')
    os.environ['ROS_LOCALHOST_ONLY']='1'
    import numpy as np
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data as qos,QoSProfile,ReliabilityPolicy
    from rtabmap_msgs.msg import Info,MapGraph,OdomInfo,SensorData
    from sensor_msgs.msg import CompressedImage,PointCloud2,PointField
    from std_msgs.msg import String,UInt8MultiArray
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from plugins.pointcloud_slam_rgb import validate_rgb_camera
    from plugins.pointcloud_slam_ros import RtabmapNode
    from plugins.pointcloud_slam_library import save_rtabmap_revision,load_map_snapshots,copy_native_map
    from plugins.pointcloud_normalization import decode_cloud
    from plugins.pointcloud_slam_transforms import pose_matrix
    scene,output=Path(scene),Path(output)
    origin=np.loadtxt(scene/'origin.txt')
    frames=plan_frames([line.split() for line in (scene/'frames.txt').read_text().splitlines()],mode,origin=origin)
    output.mkdir(parents=True,exist_ok=False)
    (output/'preview-clouds').mkdir();(output/'preview-packets').mkdir()
    (output/'ground-truth.json').write_text(json.dumps(dict(origin=origin.tolist(),frames=frames,
        coordinate_definition='synthetic camera optical frame is the body; supplied mount is identity',
        motion='continuous piecewise-linear translation between actual scan captures',
        independent_hardware_ground_truth=False),indent=2))
    restored=None
    if resume_run:
        previous=Path(resume_run);saved=json.loads((previous/'result.json').read_text())
        restored=copy_native_map(previous/'archive',saved['map_id'],saved['revision'],output/'restored.db')
    prefix='/async_visual_'+uuid.uuid4().hex
    camera=dict(schema='motus.camera/1',format='image/jpeg',width=320,height=240,
        K=[260.,0.,160.,0.,260.,120.,0.,0.,1.],D=[])
    settings=dict(cloud_frame='',child_frame='',clock_id='',child_from_cloud=np.eye(4).tolist(),
        max_gap_ms=250,max_age_s=5.,max_pending=6,max_cloud_points=320*240,
        max_measurement_age_ms=None,max_future_skew_ms=None,voxel_size_m=.04,
        max_working_nodes=300,max_update_ms=3000,reserve_bytes=0,max_disk_bytes=512*1024**2,
        preview_interval_s=.1,preview_max_points=10000,
        rgb_input=validate_rgb_camera(camera,topic=prefix+'/rgb'))
    rclpy.init();producer=rclpy.create_node('async_visual_producer')
    gateway=RtabmapNode(prefix+'/odom',prefix+'/cloud',settings,session_dir=output/'session',restored_map=restored)
    executor=MultiThreadedExecutor(num_threads=3);executor.add_node(producer);executor.add_node(gateway)
    spin=threading.Thread(target=executor.spin,daemon=True);spin.start()
    pubs={topic:producer.create_publisher(cls,prefix+'/'+topic,qos)
        for topic,cls in [('odom',String),('cloud',PointCloud2),('rgb',CompressedImage)]}
    infos,inputs,motions,graphs,subscriptions,records={},{},{},[],[],[]
    sent_images={}
    preview_clouds=[];preview_packets=[];preview_rejections=[];rgb_publications=[]
    reliable=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE)
    pid=None;motion_pids=set();motion_subscriptions=[];motion_namespace=None
    def stamp(msg):return msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
    def info(msg):infos[stamp(msg)]=dict(ref=msg.ref_id,loop=msg.loop_closure_id,
        proximity=msg.proximity_detection_id,received_monotonic=time.monotonic(),
        stats=dict(zip(msg.stats_keys,msg.stats_values)))
    def sensor(msg):
        mount=msg.local_transform[0] if msg.local_transform else None
        translation=[getattr(mount.translation,k) for k in ('x','y','z')] if mount else None
        quaternion=[getattr(mount.rotation,k) for k in ('x','y','z','w')] if mount else None
        selected=sent_images.get(stamp(msg.left),{})
        black=selected.get('rgb_input_kind')=='black'
        inputs[stamp(msg)]=dict(rgb=bool(msg.left.data),depth=bool(msg.right.data),
            rgb_all_zero=bool(msg.left.data) and not bool(np.frombuffer(msg.left.data,dtype=np.uint8).any()) if black else None,
            cloud_stamp_ns=stamp(msg),rgb_stamp_ns=stamp(msg.left),
            local_translation=translation,local_quaternion=quaternion,
            local_matrix=pose_matrix(SimpleNamespace(position=translation,quaternion=quaternion)).tolist() if mount else None)
    def graph(msg):graphs.append(dict(stamp_ns=stamp(msg),frame_id=msg.header.frame_id,
        poses={str(i):[p.position.x,p.position.y,p.position.z,p.orientation.x,p.orientation.y,
                      p.orientation.z,p.orientation.w] for i,p in zip(msg.poses_id,msg.poses)},
        links=[dict(from_id=l.from_id,to_id=l.to_id,type=l.type) for l in msg.links]))
    def preview_cloud(msg):
        try:
            capture,frame,points=decode_cloud(msg,max_points=10000,sample=True)
        except ValueError as error:
            preview_rejections.append(dict(stamp_ns=stamp(msg),reason=str(error)));return
        filename='preview-clouds/'+str(len(preview_clouds))+'.npy'
        np.save(output/filename,points)
        preview_clouds.append(dict(stamp_ns=capture,frame_id=frame,file=filename,
            points=len(points),received_monotonic=time.monotonic()))
    def preview(msg):
        raw=bytes(msg.data);digest=hashlib.sha256(raw).hexdigest()
        filename='preview-packets/'+digest+'.bin'
        if not (output/filename).exists():(output/filename).write_bytes(raw)
        preview_packets.append(dict(file=filename,sha256=digest,bytes=len(raw),
            received_monotonic=time.monotonic()))
    def motion_info(msg,namespace):
        motions.setdefault(namespace,{}).setdefault(stamp(msg),{})['info']=dict(lost=msg.lost,
            correspondences=msg.icp_correspondences,inliers_ratio=msg.icp_inliers_ratio,
            complexity=msg.icp_structural_complexity,covariance=list(msg.covariance))
    def motion_pose(msg,namespace):
        p=msg.pose.pose
        motions.setdefault(namespace,{}).setdefault(stamp(msg),{})['pose']=dict(position=[p.position.x,p.position.y,p.position.z],
            quaternion=[p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w])
    def subscribe_motion(status):
        nonlocal motion_namespace,motion_subscriptions
        namespace=status['temporal_motion']['namespace']
        if namespace==motion_namespace:return
        for subscription in motion_subscriptions:producer.destroy_subscription(subscription)
        motion_subscriptions=[producer.create_subscription(cls,namespace+'/'+topic,
            lambda msg,callback=callback,namespace=namespace:callback(msg,namespace),qos)
            for cls,topic,callback in [(Odometry,'odom',motion_pose),(OdomInfo,'odom_info_lite',motion_info)]]
        motion_namespace=namespace
    def private_process_pids():
        children=subprocess.check_output(['ps','-o','pid=,args=','--ppid',str(os.getpid())],text=True)
        return [int(row.split()[0]) for row in children.splitlines()
                if len(row.split())>=2 and Path(row.split()[1]).name=='icp_odometry']
    def persist_native_evidence():
        for name,value in [('graph.json',graphs),('native-info.json',infos),('motion-info.json',motions),
                           ('preview-clouds.json',preview_clouds),('preview-packets.json',preview_packets),
                           ('preview-rejections.json',preview_rejections),('rgb-publications.json',rgb_publications)]:
            (output/name).write_text(json.dumps(value))
        completed=finalize_records(records,infos)
        (output/'info.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in completed))
    def wait(check,seconds):
        until=time.monotonic()+seconds
        while not check():
            status=gateway.status()
            subscribe_motion(status)
            if status['state'] in ('error','storage_limit'):raise RuntimeError(json.dumps(status))
            if time.monotonic()>until:return False
            time.sleep(.005)
        return True
    try:
        subscribe_motion(gateway.status())
        subscriptions.append(producer.create_subscription(UInt8MultiArray,gateway.outputs()[0]['topic'],preview,
            QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)))
        gateway.start()
        if not wait(lambda:all(p.get_subscription_count() for p in pubs.values()),30.):
            raise TimeoutError('gateway input discovery')
        next_send=time.monotonic()
        for frame in frames:
            while time.monotonic()<next_send:time.sleep(.005)
            next_send=time.monotonic()+1.3
            capture=frame['cloud_stamp_ns'];phase=frame['phase'];sent=time.monotonic()
            xyz=frame['external_position']
            packet=dict(schema='motus.odom/1',stamp_ms=capture//10**6,
                pose=dict(position=xyz,quaternion=[0.,0.,0.,1.]) if xyz is not None else None,
                vendor=dict(motus_slam=dict(stamp_ns=capture,status='tracking' if xyz is not None else 'lost')))
            message=String();message.data=json.dumps(packet);pubs['odom'].publish(message)
            image=CompressedImage();image.header.stamp.sec,image.header.stamp.nanosec=divmod(frame['rgb_stamp_ns'],10**9)
            source=(scene/f"{frame['frame']}.jpg").read_bytes()
            payload=rgb_payload(frame['rgb_input_kind'],source)
            publication=dict(index=frame['index'],frame=frame['frame'],kind=frame['rgb_input_kind'],
                stamp_ns=frame['rgb_stamp_ns'],source_file=f"{frame['frame']}.jpg",format='jpeg',
                header=dict(sec=image.header.stamp.sec,nanosec=image.header.stamp.nanosec,frame_id=image.header.frame_id),
                source_sha256=hashlib.sha256(source).hexdigest(),
                sent_sha256=hashlib.sha256(payload).hexdigest() if payload is not None else None,
                bytes=len(payload) if payload is not None else 0)
            if frame['rgb_input_kind'] in ('black','invalid'):
                (output/'rgb-injections').mkdir(exist_ok=True)
                publication['file']=f"rgb-injections/{frame['index']}.jpg"
                (output/publication['file']).write_bytes(payload)
            rgb_publications.append(publication)
            image.format='jpeg'
            if payload is not None:
                image.data=payload
                sent_images[frame['rgb_stamp_ns']]=frame
                pubs['rgb'].publish(image)
            points=scan_depth(frame['cloud_x'],phase)
            cloud=PointCloud2();cloud.header.stamp.sec,cloud.header.stamp.nanosec=divmod(capture,10**9)
            cloud.height,cloud.width=1,len(points);cloud.point_step,cloud.row_step=12,len(points)*12
            cloud.fields=[PointField(name=name,offset=j*4,datatype=7,count=1) for j,name in enumerate(('x','y','z'))]
            cloud.is_dense=True;cloud.data=array('B',np.asarray(points,dtype='<f4').tobytes())
            if phase!='depth_gap':pubs['cloud'].publish(cloud)
            if not wait(lambda:gateway._active is not None and gateway._active['process']._process is not None,40.):
                raise TimeoutError('native mapping process startup')
            native=gateway._active['process']
            if pid is None:
                pid=native._process.pid
                subscriptions += [producer.create_subscription(cls,native.namespace+'/'+topic,callback,qos)
                    for cls,topic,callback in [(Info,'info',info),(MapGraph,'mapGraph',graph),(SensorData,'sensor_data',sensor)]]
                subscriptions.append(producer.create_subscription(PointCloud2,native.namespace+'/cloud_map',preview_cloud,reliable))
            wait(lambda:capture in infos,8. if frame['index']==0 else 3.)
            if native._process.pid!=pid or len(gateway._segments)!=1:
                raise RuntimeError('native map process or segment changed')
            status=gateway.status();motion_status=status['temporal_motion']
            subscribe_motion(status)
            live_motion_pids=private_process_pids()
            if len(live_motion_pids)>1:raise RuntimeError('multiple local scan native processes were alive concurrently')
            if motion_status['process_pid'] is not None:motion_pids.add(motion_status['process_pid'])
            actual_image=inputs.get(capture)
            selected=None
            if actual_image and actual_image['rgb']:
                selected=sent_images.get(actual_image['rgb_stamp_ns'])
                if selected is None or selected['phase']=='external_rgb_gap':
                    raise RuntimeError('native RGB header has no original valid image capture')
            rgb_capture=selected['rgb_stamp_ns'] if selected else frame['rgb_stamp_ns']
            rgb_x=selected['rgb_x'] if selected else frame['rgb_x']
            measured=None
            try:
                relative=gateway._motion.timeline.relative(rgb_capture,capture)
                expected=np.array([rgb_x-frame['cloud_x'],0.,0.])
                measured=dict(matrix=relative.tolist(),verified=motion_status['verified'],
                    translation_error_m=float(np.linalg.norm(relative[:3,3]-expected)))
            except ValueError:pass
            record=dict(frame,stamp_ns=capture,truth_x=frame['cloud_x'],
                truth_position=(origin+np.array([frame['cloud_x'],0.,0.])).tolist(),
                rgb_truth_position=(origin+np.array([frame['rgb_x'],0.,0.])).tolist(),
                selected_rgb_stamp_ns=rgb_capture if selected else None,
                selected_rgb_frame=selected['frame'] if selected else None,
                selected_rgb_x=rgb_x if selected else None,
                selected_rgb_kind=selected['rgb_input_kind'] if selected else None,
                rgb_publication=publication,
                sent_monotonic=sent,native=deepcopy(infos.get(capture)),images=actual_image,
                motion=measured,live_motion_pids=live_motion_pids,status=status)
            records.append(record)
            with (output/'info.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            print(json.dumps(dict(index=frame['index'],phase=phase,native=record['native'],
                temporal_motion=motion_status)),flush=True)
        gateway.close();artifacts=gateway.artifacts();closed_motion=gateway._motion.status()
        if closed_motion['worker_alive'] or not closed_motion['stopped']:
            raise RuntimeError('local scan motion worker remained alive after close')
        if gateway._motion._process is not None and gateway._motion._process.poll() is None:
            raise RuntimeError('local scan motion native process remained alive after close')
        if not motion_pids or private_process_pids():raise RuntimeError('local scan native process lifecycle audit failed')
        persist_native_evidence()
        identity,revision,_=save_rtabmap_revision(output/'archive',artifacts,settings,'asynchronous synthetic recovery',None,'replay')
        reopened=load_map_snapshots(output/'archive',identity,revision,max_points=settings['preview_max_points'])
        archived=archive_roundtrip([s['snapshot'] for s in artifacts['segments'] if s['snapshot'] is not None],
                                  [s['packet'] for s in reopened])
        database=artifacts['segments'][0]['database']
        with sqlite3.connect(database) as db:
            if db.execute('PRAGMA integrity_check').fetchone()!=('ok',):raise RuntimeError('native database integrity failed')
            links=db.execute('SELECT from_id,to_id,type FROM Link').fetchall()
        result=dict(classification='actual pinned ROS/native on synthetic metric inputs; no hardware motion GT',
            mode=mode,native_pid=pid,motion_pids=sorted(motion_pids),closed_motion=closed_motion,
            motion_exit_code=gateway._motion._process.poll() if gateway._motion._process is not None else None,
            segments=len(artifacts['segments']),database=str(database),links=links,map_id=identity,revision=revision,
            scene_origin=origin.tolist(),archive_reopened_identically=archived['identical'],
            archive_snapshot=archived,preview_pose_reference=gateway._active.get('pose_reference'),
            live_preview_verified=False,statistics=artifacts['statistics'])
        (output/'result.json').write_text(json.dumps(result,indent=2))
        return result
    except Exception as error:
        (output/'failure.json').write_text(json.dumps(dict(error=str(error),status=gateway.status(),frames=len(records)),indent=2))
        raise
    finally:
        try:gateway.close()
        finally:
            executor.shutdown(timeout_sec=5.);spin.join(timeout=5.)
            persist_native_evidence()
            gateway.destroy_node();producer.destroy_node();rclpy.shutdown()


def evaluate(directory, previous=None):
    import numpy as np
    from evaluate_slam_visual_replay import database_evidence
    directory=Path(directory)
    result=json.loads((directory/'result.json').read_text())
    records=[json.loads(line) for line in (directory/'info.jsonl').read_text().splitlines()]
    records=finalize_records(records,json.loads((directory/'native-info.json').read_text()))
    frames=json.loads((directory/'ground-truth.json').read_text())['frames']
    validate_selected_sources(records,frames)
    if (directory/'rgb-publications.json').is_file():
        planned={frame['index']:frame for frame in frames}
        for publication in json.loads((directory/'rgb-publications.json').read_text()):
            frame=planned[publication['index']]
            assert (publication['kind'],publication['stamp_ns'])==(frame['rgb_input_kind'],frame['rgb_stamp_ns']), 'RGB publication header/kind differs from planned capture'
            if publication.get('file'):
                payload=(directory/publication['file']).read_bytes()
                assert len(payload)==publication['bytes'] and hashlib.sha256(payload).hexdigest()==publication['sent_sha256'], 'RGB injection bytes differ from actual sent payload'
    database=next((directory/'session').rglob('rtabmap.db'),directory/'restored.db')
    db=database_evidence(database)
    old=None
    if previous:
        old_db=next((Path(previous)/'session').rglob('rtabmap.db'),Path(previous)/'restored.db')
        old=database_evidence(old_db)
    matched=retained_ground_truth(records,db['nodes'],db['optimized'],old['nodes'] if old else ())
    ignored_info=[];qualified=[]
    for row in records:
        native=row['native']
        if (native and native['ref'] in matched and
                matched[native['ref']]['cloud_stamp_ns']!=row['cloud_stamp_ns']):
            ignored_info.append(dict(index=row['index'],ref=native['ref'],reason='Info refers to a different saved capture'))
            qualified.append(dict(row,native=None))
        else:qualified.append(row)
    evidence=validate_native(qualified,set(matched),db['links'],result['mode'],cold_reopen=bool(old))
    evidence['ignored_info_reference_mismatches']=ignored_info
    by_index={row['index']:row for row in records}
    recovery_starts=[next(row for row in records if row['phase']==phase)
        for phase in ('relocalize','second_relocalize') if any(row['phase']==phase for row in records)]
    for recovered in evidence['verified_recoveries']:
        row=by_index[recovered['index']]
        start=max((candidate for candidate in recovery_starts if candidate['index']<=row['index']),
                  key=lambda candidate:candidate['index'],default=records[0])
        recovered['source_elapsed_since_recovery_start_s']=(row['cloud_stamp_ns']-start['cloud_stamp_ns'])/1e9
        recovered['wall_elapsed_since_recovery_start_s']=row['native']['received_monotonic']-start['sent_monotonic']
        recovered['cloud_send_to_native_info_ms']=(row['native']['received_monotonic']-row['sent_monotonic'])*1000.
    evidence['origin_reset_nodes']=reset_continuity(records,matched)
    if old and result['mode']=='async':evidence['cold_external_return_nodes']=cold_external_continuity(matched)
    origin=np.asarray(result['scene_origin']);root=min(db['optimized'])
    assert result['segments']==1 and result['archive_reopened_identically'], 'map/archive continuity failed'
    assert np.linalg.norm(db['optimized'][root][:,3]-origin)<1e-4, 'original map origin changed'
    if old:
        assert root==min(old['optimized']), 'cold reopening replaced original map root'
        if result['mode'].endswith('-negative'):
            assert set(db['optimized'])==set(old['optimized']), 'negative query added connected optimized nodes'
        else:
            assert len(db['optimized'])>len(old['optimized'])+3, 'cold reopen did not continue original map'
            assert any(v['anchor'] in old['optimized'] for v in evidence['verified_recoveries']), 'reopen did not geometrically match saved history'
    phases={};external_errors=[];comparison_nodes=set()
    for node,row in matched.items():
        error=float(np.linalg.norm(db['optimized'][node][:,3]-row['truth_position']))
        phases.setdefault(row['phase'],[]).append(error)
        if row['phase'] in ('outbound','revisit'):
            external_errors.append(error);comparison_nodes.add(node)
    if result['mode']!='no-rgb':
        assert max((max(v) for v in phases.values()),default=0.)<.35, 'saved map misaligned with metric ground truth'
    visual_edges=evidence['verified_visual_edges'];visual_revisits=[]
    for edge in visual_edges:
        row=by_index[edge['index']]
        if (row['phase']=='revisit' and edge['anchor'] in db['nodes'] and
                row['cloud_stamp_ns']/1e9-db['nodes'][edge['anchor']][0]>=15.):
            visual_revisits.append({key:edge[key] for key in ('ref','anchor','inliers')})
    if 'revisit' in phases and result['mode']=='async':
        assert visual_revisits, 'no historical visual-inlier-verified persisted revisit edge'
    if result['mode']=='no-rgb':assert not visual_revisits, 'RGB-off acquired a visual revisit edge'
    preview_errors=[];retained_preview_errors=[]
    for row in records:
        image=row.get('images')
        if row['external_pose'] or not image or image['local_translation'] is None:continue
        rgb_x=row.get('selected_rgb_x',row['rgb_x'])
        expected=np.array([rgb_x-row['cloud_x'],0.,0.])
        error=float(np.linalg.norm(np.asarray(image['local_translation'])-expected))
        preview_errors.append(error)
        if row['native'] and row['native']['ref'] in db['optimized']:retained_preview_errors.append(error)
    if retained_preview_errors:assert max(retained_preview_errors)<.05, 'retained RGB mount disagrees with measured cross-time geometry'
    graphs=json.loads((directory/'graph.json').read_text())
    saved_graph=saved_native_graph_evidence(graphs,db['optimized'])
    assert saved_graph['verified'] is not False, 'saved optimized poses differ from the complete native live MapGraph'
    if not result['mode'].endswith('-negative'):
        assert any(str(root) in graph['poses'] for graph in graphs), 'optimized graph display lacks original root'
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from plugins.pointcloud_slam_library import load_map_snapshots
    archive=directory/'archive'/'maps'/result['map_id']
    saved=json.loads((archive/f"{result['revision']}.json").read_text())
    expected=[(archive/segment['snapshot']).read_bytes() for segment in saved['segments'] if segment.get('snapshot')]
    reopened=load_map_snapshots(directory/'archive',result['map_id'],result['revision'],max_points=10000)
    archived=archive_roundtrip(expected,[snapshot['packet'] for snapshot in reopened])
    marker_x,marker_y=struct.unpack_from('<ff',expected[0])
    marker_error=float(np.linalg.norm(np.array([marker_x,marker_y])-db['optimized'][max(db['optimized'])][:2,3]))
    assert marker_error<1e-4, 'saved preview marker differs from final optimized node'
    live=dict(verified=False,reason='no new preview required for a rejected negative query')
    if not result['mode'].endswith('-negative'):
        assert (directory/'preview-packets.json').is_file(), 'raw live preview evidence was not captured'
        live=live_preview_evidence(directory,visual_edges,pose_reference=result.get('preview_pose_reference'),
            require_visual_loop=result['mode']=='async')
    surface=[]
    with sqlite3.connect(f'file:{database.resolve()}?mode=ro',uri=True) as connection:
        for node,scan,scan_info in connection.execute('SELECT id,scan,scan_info FROM Data'):
            if node not in comparison_nodes or not scan or not scan_info:continue
            rows,cols,kind=struct.unpack('<iii',scan[-12:])
            if kind!=21:continue
            points=np.frombuffer(zlib.decompress(scan[:-12]),'<f4').reshape(rows*cols,3)[::64]
            local=np.frombuffer(scan_info,dtype='<f4',offset=28,count=12).reshape(3,4)
            points=points@local[:,:3].T+local[:,3]
            pose=db['optimized'][node];points=points@pose[:,:3].T+pose[:,3]-origin
            x,y,z=points.T
            surface.extend(np.abs(z-(3.+.24*np.sin(x*1.7)+.18*np.cos(y*2.3)+.14*np.sin(x*3+y))))
    summary=dict(evidence,mode=result['mode'],optimized_nodes=len(db['optimized']),
        new_optimized_nodes_checked=len(matched),
        historical_visual_revisit_edges=visual_revisits,
        maximum_saved_linear_constraint_error=db['maximum_saved_linear_constraint_error'],
        phases={phase:dict(retained_frames=len(values),max_xyz_error_m=max(values)) for phase,values in phases.items()},
        revisit_endpoint_xyz_error_m=phases.get('revisit',[None])[-1],
        external_trajectory_xyz_rms_m=float(np.sqrt(np.mean(np.square(external_errors)))) if external_errors else None,
        surface_rms_m=float(np.sqrt(np.mean(np.square(surface)))) if surface else None,
        maximum_published_camera_translation_error_m=max(preview_errors,default=None),
        maximum_retained_camera_translation_error_m=max(retained_preview_errors,default=None),
        archive_reopened_identically=True,archive_snapshot=archived,saved_marker_xy_error_m=marker_error,
        saved_native_graph=saved_graph,
        live_preview_verified=live['verified'],live_preview=live,
        closed_motion=result['closed_motion'],motion_exit_code=result['motion_exit_code'],
        classification=result['classification'])
    (directory/'evaluation.json').write_text(json.dumps(summary,indent=2))
    return summary


def compare_runs(rgb, no_rgb):
    enabled=json.loads((Path(rgb)/'evaluation.json').read_text())
    disabled=json.loads((Path(no_rgb)/'evaluation.json').read_text())
    endpoint,baseline=enabled['revisit_endpoint_xyz_error_m'],disabled['revisit_endpoint_xyz_error_m']
    assert endpoint is not None and baseline is not None, 'missing revisit endpoint evidence'
    assert endpoint<.15 and endpoint<baseline*.3, 'RGB endpoint failed existing E improvement thresholds'
    assert enabled['surface_rms_m'] is not None and disabled['surface_rms_m'] is not None, 'missing metric surface evidence'
    assert enabled['surface_rms_m']<disabled['surface_rms_m']*.7, 'RGB map surface failed existing E improvement threshold'
    return dict(endpoint_improvement_ratio=endpoint/baseline,
        surface_improvement_ratio=enabled['surface_rms_m']/disabled['surface_rms_m'],
        rgb=enabled,no_rgb=disabled)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene');parser.add_argument('output')
    parser.add_argument('--mode',choices=('async','vo-continuation','no-rgb','similar-negative','novel-negative'),default='async')
    parser.add_argument('--resume-run')
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--compare-no-rgb')
    args=parser.parse_args()
    if not args.evaluate_only:run(args.scene,args.output,args.mode,args.resume_run)
    summary=evaluate(args.output,args.resume_run)
    if args.compare_no_rgb:
        comparison=compare_runs(args.output,args.compare_no_rgb)
        (Path(args.output)/'rgb-comparison.json').write_text(json.dumps(comparison,indent=2))
        summary['comparison']=comparison
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
