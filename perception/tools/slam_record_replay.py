#!/usr/bin/env python3
"""Audit passive CDR captures, normalize measured RGB-D, and replay the gateway.

No robot connection. All source header stamps remain their original integer nanoseconds.
Registered normalization requires explicit source evidence and the actual
producer module. Legacy aligned depth is not equivalent to corrected color-Z:
missing SDK calibration is an error, not an identity-extrinsic assumption.
"""
import argparse
from array import array
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def read_records(directory):
    directory=Path(directory).resolve()
    rows=[json.loads(line) for line in (directory/'records.jsonl').read_text().splitlines()]
    for row in rows:
        for key in ('stamp_ns','receipt_ns'):
            if row[key] is not None and type(row[key]) is not int:
                raise ValueError(key+' must be integer nanoseconds')
        payload=(directory/row['file']).resolve()
        if payload.parent != directory or not payload.is_file():
            raise ValueError('invalid payload path: '+row['file'])
        if hashlib.sha256(payload.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('hash mismatch: '+row['file'])
    return rows


def pair_records(color,depth,max_skew_ns):
    """Source-order bounded pairing, matching the Driver's two-queue policy."""
    color=sorted(color,key=lambda row:row['stamp_ns'])
    depth=sorted(depth,key=lambda row:row['stamp_ns'])
    i=j=0;pairs=[]
    while i<len(color) and j<len(depth):
        delta=color[i]['stamp_ns']-depth[j]['stamp_ns']
        if abs(delta)<=max_skew_ns:
            pairs.append((color[i],depth[j]));i+=1;j+=1
        elif delta<0:i+=1
        else:j+=1
    return pairs


def pair_publication_cycles(color,depth):
    """For evidenced color-then-depth producer loops, never borrow prior depth."""
    color=sorted(color,key=lambda row:row['stamp_ns']);depth=sorted(depth,key=lambda row:row['stamp_ns'])
    pairs=[];j=0
    for i,row in enumerate(color):
        while j<len(depth) and depth[j]['stamp_ns']<row['stamp_ns']:j+=1
        if j>=len(depth):break
        if i+1<len(color) and depth[j]['stamp_ns']>=color[i+1]['stamp_ns']:continue
        pairs.append((row,depth[j]));j+=1
    return pairs


def audit(directory):
    rows=read_records(directory);topics={}
    for topic in sorted({row['topic'] for row in rows}):
        selected=[row for row in rows if row['topic']==topic]
        stamps=[row['stamp_ns'] for row in selected if row['stamp_ns'] is not None]
        topics[topic]=dict(count=len(selected),type=selected[0]['type'],
            first_stamp_ns=min(stamps) if stamps else None,last_stamp_ns=max(stamps) if stamps else None,
            regressions=sum(b<a for a,b in zip(stamps,stamps[1:])),
            duplicate_stamps=len(stamps)-len(set(stamps)))
    return dict(recording=str(Path(directory).resolve()),verified_payloads=len(rows),topics=topics,
                timestamp_policy='original integer source header stamp; never rebase',
                hardware_motion_ground_truth=False)


def decode(directory,row):
    from rclpy.serialization import deserialize_message
    import importlib
    package,kind,name=row['type'].split('/')
    message_type=getattr(importlib.import_module(package+'.'+kind),name)
    message=deserialize_message((Path(directory)/row['file']).read_bytes(),message_type)
    if hasattr(message,'header'):
        stamp=message.header.stamp.sec*10**9+message.header.stamp.nanosec
        if stamp!=row['stamp_ns']:raise ValueError('CDR/header timestamp differs from manifest')
    return message


def load_driver(path):
    import importlib.util
    # These are actual source modules, whose common imports require Driver root.
    path=Path(path).resolve();sys.path.insert(0,str(path.parents[2]));sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('recorded_rgbd_driver',path)
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    return driver


def prepare_registered(args):
    import cv2
    import numpy as np
    rows=read_records(args.recording)
    evidence=json.loads(Path(args.registration_evidence).read_text())
    values=evidence['parameters']
    if values['depth_registration'].get('bool_value') is not True:
        raise ValueError('measured registered-depth geometry evidence is required')
    def messages(topic):return [r for r in rows if r['topic']==topic]
    color_infos=messages(args.color_info_topic);depth_infos=messages(args.depth_info_topic)
    if not color_infos or not depth_infos:raise ValueError('both measured CameraInfo streams are required')
    color_info=decode(args.recording,color_infos[0]);depth_info=decode(args.recording,depth_infos[0])
    driver=load_driver(args.driver)
    geometry=driver.make_geometry(color_info,depth_info,registered=True)
    pairs=pair_records(messages(args.color_topic),messages(args.depth_topic),20_000_000)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    declaration=driver.camera_declarations('/recorded/camera',geometry,'recorded-source-capture')[0]
    # The replay's body reference is explicitly the measured common camera frame.
    declaration['vendor'].update(pose_reference='camera_optical')
    (output/'camera.json').write_text(json.dumps(declaration,indent=2))
    with (output/'frames.jsonl').open('w') as stream:
        for i,(cr,dr) in enumerate(pairs):
            color,depth=decode(args.recording,cr),decode(args.recording,dr)
            for image,info in ((color,color_info),(depth,depth_info)):
                if (image.width,image.height)!=(info.width,info.height):
                    raise ValueError('source image geometry differs from measured calibration')
            if color.encoding not in ('rgb8','bgr8') or depth.encoding!='16UC1' or depth.is_bigendian:
                raise ValueError('unsupported recorded RGB-D encoding')
            rgb=np.frombuffer(color.data,np.uint8).reshape(color.height,color.step)[:,:color.width*3].reshape(color.height,color.width,3)
            if color.encoding=='rgb8':rgb=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            z=np.frombuffer(depth.data,np.uint8).reshape(depth.height,depth.step)[:,:depth.width*2].copy().view('<u2').reshape(depth.height,depth.width)
            rect=cv2.remap(rgb,geometry['color_x'],geometry['color_y'],cv2.INTER_LINEAR)
            metric=cv2.remap(z,geometry['depth_x'],geometry['depth_y'],cv2.INTER_NEAREST)
            ok,jpeg=cv2.imencode('.jpg',rect,[cv2.IMWRITE_JPEG_QUALITY,90])
            if not ok:raise ValueError('JPEG encoding failed')
            (output/f'{i}.jpg').write_bytes(jpeg.tobytes());np.save(output/f'{i}.npy',metric)
            stream.write(json.dumps(dict(frame=i,rgb_stamp_ns=cr['stamp_ns'],cloud_stamp_ns=dr['stamp_ns'],
                rgb=f'{i}.jpg',depth=f'{i}.npy',source_color=cr,source_depth=dr))+'\n')
    report=audit(args.recording)
    report.update(pairs=len(pairs),normalization='offline equivalent of recorded producer make_geometry/remap',
        driver=str(Path(args.driver).resolve()),driver_sha256=hashlib.sha256(Path(args.driver).read_bytes()).hexdigest(),
        registration_evidence=evidence,body_reference=geometry['frame_id'],
        producer_helper_constraints='the external recorded-producer helper may require common optical frame labels; this reproduces its conversion, not a PR371 admission policy',
        clock_parameters='provenance only; no clock label or synchronization flag veto',
        body_from_cloud='identity by explicit common camera-coordinate definition; no physical body mount claimed')
    (output/'provenance.json').write_text(json.dumps(report,indent=2));return report


def prepare_aligned(args):
    """Correct recorded SDK-aligned depth using measured factory calibration."""
    import cv2
    import numpy as np
    import zlib
    from types import SimpleNamespace
    rows=read_records(args.recording);calibration=json.loads(Path(args.calibration).read_text())
    color=calibration['color'];extrinsic=calibration['depth_to_color']
    intr=SimpleNamespace(width=color['width'],height=color['height'],fx=color['fx'],fy=color['fy'],
        ppx=color['cx'],ppy=color['cy'],coeffs=color['D'],model=color['model'])
    extr=SimpleNamespace(rotation=extrinsic['rotation_column_major'],translation=extrinsic['translation_m'])
    driver=load_driver(args.driver);geometry=driver.color_depth_geometry(intr,extr)
    source_evidence=Path(args.publication_cycle_evidence).read_text()
    if not source_evidence.strip():raise ValueError('recorded producer color-before-depth loop evidence required')
    pairs=pair_publication_cycles([r for r in rows if r['topic']==args.color_topic],
        [r for r in rows if r['topic']==args.depth_topic])
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    declaration=dict(schema='motus.camera/1',format='image/jpeg',width=intr.width,height=intr.height,
        K=geometry['K'],D=[0.]*5,distortion_model='pinhole',vendor=dict(pose_reference='camera_optical'))
    (output/'camera.json').write_text(json.dumps(declaration,indent=2))
    with (output/'frames.jsonl').open('w') as stream:
        for i,(cr,dr) in enumerate(pairs):
            rgb,depth=decode(args.recording,cr),decode(args.recording,dr)
            if rgb.format!='jpeg' or 'zlib' not in depth.format:raise ValueError('unsupported aligned recording encoding')
            pixels=cv2.imdecode(np.frombuffer(bytes(rgb.data),np.uint8),cv2.IMREAD_COLOR)
            measured=np.frombuffer(zlib.decompress(bytes(depth.data)),dtype='<u2').reshape(intr.height,intr.width)
            # The input-unit choice is explicit evidence, never inferred from a label.
            unit=calibration['depth_scale'] if args.depth_values=='raw-sdk' else .001
            measured=cv2.remap(measured,geometry['map_x'],geometry['map_y'],cv2.INTER_NEAREST)
            metric=np.rint((measured.astype(np.float64)*unit*geometry['depth_z_scale']+geometry['depth_z_offset'])*1000.)
            metric[(measured==0)|(metric<1)|(metric>65535)]=0
            rect=cv2.remap(pixels,geometry['map_x'],geometry['map_y'],cv2.INTER_LINEAR)
            ok,jpeg=cv2.imencode('.jpg',rect,[cv2.IMWRITE_JPEG_QUALITY,90])
            if not ok:raise ValueError('JPEG encoding failed')
            (output/f'{i}.jpg').write_bytes(jpeg.tobytes());np.save(output/f'{i}.npy',metric.astype('<u2'))
            stream.write(json.dumps(dict(frame=i,rgb_stamp_ns=cr['stamp_ns'],cloud_stamp_ns=dr['stamp_ns'],
                rgb=f'{i}.jpg',depth=f'{i}.npy',source_color=cr,source_depth=dr))+'\n')
    report=audit(args.recording)
    report.update(pairs=len(pairs),normalization='offline factory-calibrated depth-Z to color-Z correction of legacy aligned data',
        calibration=calibration,depth_input_values=args.depth_values,
        pairing='evidenced color-before-depth publication cycle',
        publication_cycle_evidence_sha256=hashlib.sha256(source_evidence.encode()).hexdigest(),
        original_stamp_semantics=args.stamp_semantics,driver=str(Path(args.driver).resolve()),
        driver_sha256=hashlib.sha256(Path(args.driver).read_bytes()).hexdigest(),
        body_reference='measured color camera optical coordinates',
        limitations=['legacy compressed color is decoded and re-encoded; not bit-identical to current producer output',
                     'not a claim that the current producer is deployed on hardware'])
    (output/'provenance.json').write_text(json.dumps(report,indent=2));return report


def bootstrap(args):
    """Native RGB-D odometry establishes a camera-coordinate gauge from data."""
    import cv2
    import numpy as np
    import rclpy
    import subprocess
    import signal
    from ament_index_python.packages import get_package_prefix
    from nav_msgs.msg import Odometry
    from rtabmap_msgs.msg import OdomInfo,RGBDImage
    from sensor_msgs.msg import Image,CameraInfo
    from rclpy.qos import QoSProfile,ReliabilityPolicy
    directory=Path(args.prepared);output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    frames=[json.loads(s) for s in (directory/'frames.jsonl').read_text().splitlines()]
    if args.end_frame is not None:frames=[row for row in frames if row['frame']<args.end_frame]
    camera=json.loads((directory/'camera.json').read_text())
    domain=os.environ.get('ROS_DOMAIN_ID','')
    if not domain.isdecimal() or not 1<=int(domain)<=232:raise ValueError('isolated nonzero ROS_DOMAIN_ID required')
    os.environ['ROS_LOCALHOST_ONLY']='1'
    rclpy.init();node=rclpy.create_node('recorded_rgbd_bootstrap')
    namespace='/recorded_bootstrap_'+str(os.getpid());frame='recorded_camera_optical'
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE)
    poses={};infos={}
    def stamp(msg):return msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
    def pose(msg):
        p=msg.pose.pose
        poses[stamp(msg)]=dict(position=[p.position.x,p.position.y,p.position.z],
            quaternion=[p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w],
            covariance=list(msg.pose.covariance))
    def info(msg):
        infos[stamp(msg)]=dict(lost=msg.lost,inliers=msg.inliers,matches=msg.matches,
            features=msg.features,local_map_size=msg.local_map_size)
    node.create_subscription(Odometry,namespace+'/odom',pose,qos)
    node.create_subscription(OdomInfo,namespace+'/odom_info_lite',info,qos)
    publisher=node.create_publisher(RGBDImage,namespace+'/rgbd_image',qos)
    executable=Path(get_package_prefix('rtabmap_odom'))/'lib/rtabmap_odom/rgbd_odometry'
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from plugins.rgbd_odometry_runtime import estimator_parameters
    native_parameters=estimator_parameters(dict(frame_id=frame),namespace.lstrip('/'))
    native_parameters.update(qos=1,topic_queue_size=1,sync_queue_size=1,publish_null_when_lost=True,
        **{'Odom/ResetCountdown':'1','Odom/GuessMotion':'true'})
    parameters={'/**':{'ros__parameters':native_parameters}}
    (output/'params.json').write_text(json.dumps(parameters))
    log=(output/'native.log').open('wb')
    process=subprocess.Popen([str(executable),'--ros-args','--params-file',str(output/'params.json'),
        '-r','__ns:='+namespace],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    records=[]
    def pump_until(check,seconds):
        deadline=time.monotonic()+seconds
        while not check():
            if process.poll() is not None:raise RuntimeError('native RGB-D odometry exited')
            if time.monotonic()>deadline:return False
            rclpy.spin_once(node,timeout_sec=.02)
        return True
    try:
        if not pump_until(lambda:publisher.get_subscription_count()>0,30):
            raise TimeoutError('native RGB-D odometry discovery')
        endpoints={}
        for topic in ('rgbd_image','odom','odom_info_lite'):
            endpoints[topic]={}
            for role,method in [('publishers',node.get_publishers_info_by_topic),('subscriptions',node.get_subscriptions_info_by_topic)]:
                endpoints[topic][role]=[dict(node=e.node_name,namespace=e.node_namespace,type=e.topic_type,
                    reliability=e.qos_profile.reliability.name,history=e.qos_profile.history.name,
                    depth=e.qos_profile.depth,durability=e.qos_profile.durability.name) for e in method(namespace+'/'+topic)]
        (output/'endpoints.json').write_text(json.dumps(endpoints,indent=2))
        for index,row in enumerate(frames):
            before=set(infos);envelope=RGBDImage()
            for kind in ('rgb','depth'):
                image=Image();image.header.frame_id=frame
                capture=row['rgb_stamp_ns' if kind=='rgb' else 'cloud_stamp_ns']
                image.header.stamp.sec,image.header.stamp.nanosec=divmod(capture,10**9)
                pixels=cv2.imread(str(directory/row['rgb'])) if kind=='rgb' else np.load(directory/row['depth'])
                image.height,image.width=pixels.shape[:2]
                image.encoding='bgr8' if kind=='rgb' else '16UC1';image.is_bigendian=False
                image.step=image.width*(3 if kind=='rgb' else 2);image.data=array('B',pixels.tobytes())
                if kind=='rgb':
                    calibration=CameraInfo();calibration.header=deepcopy(image.header)
                    calibration.width,calibration.height=camera['width'],camera['height']
                    calibration.k=camera['K'];calibration.d=camera['D'];calibration.distortion_model='plumb_bob'
                    calibration.r=np.eye(3).ravel().tolist()
                    calibration.p=np.column_stack((np.asarray(camera['K']).reshape(3,3),np.zeros(3))).ravel().tolist()
                    envelope.rgb_camera_info=calibration
                    envelope.rgb=image
                else:
                    envelope.depth=image
                    envelope.depth_camera_info=deepcopy(calibration)
                    envelope.depth_camera_info.header=deepcopy(image.header)
            envelope.header=deepcopy(envelope.rgb.header if float(row['rgb_stamp_ns'])/1e9>float(row['cloud_stamp_ns'])/1e9 else envelope.depth.header)
            publisher.publish(envelope)
            if not pump_until(lambda:stamp(envelope) in infos,5.):
                raise TimeoutError('native RGB-D produced no result for original pair '+str(row['frame']))
            # Let the matching native Odometry callback arrive after OdomInfo.
            pump_until(lambda:all(s in poses or infos[s]['lost'] for s in set(infos)-before),1.)
            received=[dict(stamp_ns=s,pose=poses.get(s),info=infos[s]) for s in sorted(set(infos)-before)]
            records.append(dict(frame=row['frame'],rgb_stamp_ns=row['rgb_stamp_ns'],
                cloud_stamp_ns=row['cloud_stamp_ns'],native=received))
            print(json.dumps(records[-1]),flush=True)
            if index>0 and (infos[stamp(envelope)]['lost'] or infos[stamp(envelope)]['inliers']<=0):
                raise RuntimeError('native RGB-D registration lost; bootstrap gauge is not continuous')
        verified=[dict(stamp_ns=s,pose=poses[s],info=infos[s]) for s in sorted(infos)
                  if s in poses and not infos[s]['lost'] and infos[s]['inliers']>0
                  and np.isfinite(poses[s]['position']+poses[s]['quaternion']+poses[s]['covariance']).all()
                  and np.isclose(np.linalg.norm(poses[s]['quaternion']),1.,atol=1e-3)
                  and np.all(np.diag(np.asarray(poses[s]['covariance']).reshape(6,6))>0)]
        (output/'poses.json').write_text(json.dumps(verified,indent=2))
        (output/'frames.json').write_text(json.dumps(records,indent=2))
        summary=dict(verified_poses=len(verified),coordinate_reference='native RGB-D camera-start FLU world; camera optical child',
            initial_pose=native_parameters.get('initial_pose'),
            basis_interpretation='shared RGB-D numeric world-axis convention, not measured gravity or a physical body mount',
            independent_motion_ground_truth=False,pose_source='native measured feature registration; identity startup excluded',
            native_executable=str(executable),native_executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest())
        (output/'result.json').write_text(json.dumps(summary,indent=2))
        if len(verified)<2:raise RuntimeError('fewer than two registration-verified native poses')
        return summary
    except Exception as error:
        (output/'failure.json').write_text(json.dumps(dict(error=str(error),frames=records,stage='native RGB-D bootstrap'),indent=2))
        raise
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGINT)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=2)
        log.close();node.destroy_node();rclpy.shutdown()


def recovery_evidence(records,links,persisted_nodes,*,graph_nodes=None,original_root=None,
                      live_graphs=None,node_stamps=None):
    """Require replay evidence to belong to the original persisted pose graph."""
    perception_root=str(Path(__file__).resolve().parents[1])
    if perception_root not in sys.path:sys.path.insert(0,perception_root)
    from plugins.pointcloud_slam_graph import connected_nodes
    persisted_nodes=set(persisted_nodes)
    links=tuple(links)
    graph_nodes={node for node in (persisted_nodes if graph_nodes is None else graph_nodes) if node>0}
    root=min(graph_nodes,default=0) if original_root is None else original_root
    connected=connected_nodes(graph_nodes,links,(root,))
    retained=connected.intersection(persisted_nodes)
    live_by_stamp={}
    if live_graphs is not None:
        known={root}
        for graph in sorted(live_graphs,key=lambda value:value['stamp_ns']):
            ids=set(graph['ids'])
            edges=((edge['from_id'],edge['to_id'],edge['type']) for edge in graph['links'])
            reachable=connected_nodes(ids,edges,{root} if root in ids else known)
            if reachable:known=reachable
            live_by_stamp[graph['stamp_ns']]=(reachable,ids)
    global_edges={tuple(sorted((a,b))) for a,b,kind in links if kind==1}
    verified=[];short_recoveries=[];tracking=[];unverified=[]
    tracking_after_recovery=[];continuation=None;continuation_ref=None;seen=set()
    event_fields=('Motus/RecoveryNode/','Motus/RecoveryAnchor/','Motus/RecoveryInliers/')
    for row in records:
        native=row['native']
        if native is None:
            if row.get('images') is not None:continuation=None
            continue
        ref,anchor=native['ref'],native['loop'];stats=native['stats']
        state=stats.get('Motus/RecoveryState/');inliers=stats.get('Loop/Visual_inliers/',0)
        capture=row.get('cloud_stamp_ns',0)
        live,live_ids=(connected,graph_nodes) if live_graphs is None else live_by_stamp.get(capture,(set(),set()))
        sec,nsec=divmod(capture,10**9)
        same_capture=node_stamps is None or node_stamps.get(ref)==sec+nsec*1e-9
        new_node=ref in retained and ref in live and same_capture and ref not in seen
        if new_node:seen.add(ref)
        visual_only=row['input_pose_source'] is None and not row.get('baseline_only')
        if state!=3 or not visual_only:continuation=None
        if not visual_only:continue
        kind=None
        if state==2:
            event_valid=True
            # Promotion clears standard loop stats; never mix a partial event with them.
            if any(key in stats for key in event_fields):
                node,anchor,inliers=(stats.get(key,0) for key in event_fields)
                event_valid=all(key in stats for key in event_fields) and node==ref
            if event_valid and ref in retained and anchor in connected and math.isfinite(inliers) and inliers>=20 and tuple(sorted((ref,anchor))) in global_edges:
                verified.append(dict(index=row['index'],ref=ref,anchor=anchor,inliers=inliers))
                # The promotion edge can first appear in the next MapGraph.
                if anchor in live and ref in live_ids and same_capture:kind='global'
            else:unverified.append(row['index'])
        if state==3:
            if ref in retained:tracking.append(ref)
            old_inliers=stats.get('Motus/VisualReacquiredInliers/',0)
            all_inliers=stats.get('Motus/VisualInliers/',0)
            # A validated small-motion query can be omitted from the optimized graph.
            eligible_ref=ref>0 and ((ref in retained and ref in live and same_capture)
                                   or (ref not in graph_nodes and ref not in live_ids))
            if live and eligible_ref and (
                    math.isfinite(old_inliers) and math.isfinite(all_inliers) and 20<=old_inliers<=all_inliers):
                short_recoveries.append(dict(index=row['index'],ref=ref,inliers=old_inliers))
                kind='short_visual'
            elif new_node and continuation is not None and (continuation_ref is None or continuation_ref in live):
                continuation['nodes'].append(ref)
        if kind is not None:
            # Count subsequent new nodes once, stopping at the next loss or recovery.
            continuation=dict(kind=kind,index=row['index'],nodes=[])
            continuation_ref=ref if kind=='global' else None
            tracking_after_recovery.append(continuation)
    return dict(verified_recoveries=verified,unverified_localized_frames=unverified,
        verified_short_recoveries=short_recoveries,tracking_after_recovery=tracking_after_recovery,
        persisted_visual_tracking_nodes=sorted(set(tracking)))


def header_evidence(source,inputs,motion):
    """Compare actual emitted native headers with original integer sources."""
    cloud_stamps={row['cloud_stamp_ns'] for row in source};rgb_stamps={row['rgb_stamp_ns'] for row in source}
    attached=[row for row in inputs.values() if row['rgb']]
    bad_inputs=[row for row in attached if row['sensor_stamp_ns'] not in cloud_stamps or
        row['laser_scan_stamp_ns']!=row['sensor_stamp_ns'] or row['rgb_stamp_ns'] not in rgb_stamps]
    bad_motion=[row['stamp_ns'] for row in motion if row['stamp_ns'] not in cloud_stamps]
    return dict(attached_inputs=len(attached),motion_messages=len(motion),
        preserved=not bad_inputs and not bad_motion if attached else None,
        unexpected_inputs=bad_inputs,unexpected_motion_stamps_ns=bad_motion)


def replay_sources(args):
    """Select original frames; only injected poses require a measured timeline."""
    directory=Path(args.prepared)
    frames=[json.loads(s) for s in (directory/'frames.jsonl').read_text().splitlines()]
    samples=[];native_bootstrap=None
    if args.pose_frames>0 or args.all_poses:
        if not args.bootstrap_poses:raise ValueError('bootstrap poses required when injecting external poses')
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from plugins.pointcloud_slam_odom import PoseSample
        path=Path(args.bootstrap_poses);verified=json.loads(path.read_text())
        samples=[PoseSample(v['stamp_ns']//10**6,'recorded_native','recorded_camera','recorded',
            tuple(v['pose']['position']),tuple(v['pose']['quaternion']),tuple(v['pose']['covariance']),
            'recorded',None,None,v['stamp_ns']) for v in verified if not v['info']['lost'] and v['info']['inliers']>0]
        if len(samples)<2:raise ValueError('at least two native registration-verified bootstrap poses required')
        # Exclude startup gauge and endpoints outside the measured native timeline.
        frames=[r for r in frames if samples[0].stamp_ns<=r['cloud_stamp_ns']<=samples[-1].stamp_ns]
        native_bootstrap=dict(file=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            summary=json.loads(path.with_name('result.json').read_text()))
    frames=[r for r in frames if r['frame']>=args.start_frame and (args.end_frame is None or r['frame']<args.end_frame)]
    if len(frames)<3:raise ValueError('insufficient measured frames after native bootstrap/subrange selection')
    return frames,samples,native_bootstrap


def cold_replay(args):
    return bool(args.resume_run and args.pose_frames==0 and not args.all_poses)


def replay(args):
    """Feed genuine captures through the production gateway with explicit faults."""
    frames,samples,native_bootstrap=replay_sources(args)
    import cv2
    import numpy as np
    import rclpy
    import sqlite3
    import threading
    import uuid
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data,QoSProfile,ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage,PointCloud2,PointField
    from std_msgs.msg import String
    from rtabmap_msgs.msg import Info,MapGraph,SensorData,OdomInfo
    from nav_msgs.msg import Odometry
    from rclpy.serialization import serialize_message
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from plugins.pointcloud_slam_rgb import validate_rgb_camera
    from plugins.pointcloud_slam_ros import RtabmapNode
    from plugins.pointcloud_slam_library import save_rtabmap_revision,load_map_snapshots,copy_native_map
    from plugins.pointcloud_slam_snapshot import _optimized_poses
    domain=os.environ.get('ROS_DOMAIN_ID','')
    if not domain.isdecimal() or not 1<=int(domain)<=232:raise ValueError('isolated nonzero ROS_DOMAIN_ID required')
    os.environ['ROS_LOCALHOST_ONLY']='1'
    directory=Path(args.prepared);output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    camera=json.loads((directory/'camera.json').read_text());prefix='/recorded_replay_'+uuid.uuid4().hex
    settings=dict(cloud_frame='',child_frame='',clock_id='',child_from_cloud=np.eye(4).tolist(),
        max_gap_ms=250,max_age_s=5.,max_pending=12,max_cloud_points=camera['width']*camera['height'],
        max_measurement_age_ms=None,max_future_skew_ms=None,voxel_size_m=.04,max_working_nodes=300,
        max_update_ms=3000,reserve_bytes=0,max_disk_bytes=512*1024**2,preview_interval_s=.1,
        preview_max_points=10000,rgb_input=validate_rgb_camera(camera,topic=prefix+'/rgb'),
        native_parameters={'RGBD/LinearUpdate':0.,'RGBD/AngularUpdate':0.} if args.insert_every_observation else {})
    if args.offline_all_captures:settings['native_parameters']['Rtabmap/DetectionRate']=0.
    restored=None;historical_nodes=set();original_root=None
    if args.resume_run:
        previous=Path(args.resume_run);saved=json.loads((previous/'result.json').read_text())
        restored=copy_native_map(previous/'archive',saved['map_id'],saved['revision'],output/'restored.db')
        with sqlite3.connect(restored['database']) as db:
            historical_nodes={node for node, in db.execute('SELECT id FROM Node WHERE id>0')}
            try:original_root=min(_optimized_poses(db) or historical_nodes,default=None)
            except ValueError:original_root=min(historical_nodes,default=None)
    rclpy.init();producer=rclpy.create_node('recorded_replay_producer')
    gateway=RtabmapNode(prefix+'/odom',prefix+'/cloud',settings,session_dir=output/'session',restored_map=restored)
    executor=MultiThreadedExecutor(num_threads=3);executor.add_node(producer);executor.add_node(gateway)
    spin=threading.Thread(target=executor.spin,daemon=True);spin.start()
    odom=producer.create_publisher(String,prefix+'/odom',qos_profile_sensor_data)
    clouds=producer.create_publisher(PointCloud2,prefix+'/cloud',qos_profile_sensor_data)
    images=producer.create_publisher(CompressedImage,prefix+'/rgb',qos_profile_sensor_data)
    private_qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE)
    infos={};graphs=[];inputs={};preview_clouds=[];subscriptions=[];records=[];pid=None
    def stamp(msg):return msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
    def info(msg):infos[stamp(msg)]=dict(ref=msg.ref_id,loop=msg.loop_closure_id,proximity=msg.proximity_detection_id,received_monotonic=time.monotonic(),stats=dict(zip(msg.stats_keys,msg.stats_values)))
    def graph(msg):graphs.append(dict(stamp_ns=stamp(msg),ids=list(msg.poses_id),links=[dict(from_id=l.from_id,to_id=l.to_id,type=l.type) for l in msg.links]))
    def sensor(msg):
        depth_quality=dict(width=msg.right.width,height=msg.right.height,encoding=msg.right.encoding)
        if msg.right.data and msg.right.encoding=='32FC1':
            values=np.frombuffer(msg.right.data,dtype='>f4' if msg.right.is_bigendian else '<f4')
            depth_quality['positive_finite_pixels']=int(np.sum(np.isfinite(values)&(values>0)))
            depth_quality['stored_pixels']=int(values.size)
        inputs[stamp(msg)]=dict(rgb=bool(msg.left.data),depth=bool(msg.right.data),depth_quality=depth_quality,
            sensor_stamp_ns=stamp(msg),rgb_stamp_ns=stamp(msg.left),depth_stamp_ns=stamp(msg.right),
            sensor_frame_id=msg.header.frame_id,rgb_frame_id=msg.left.header.frame_id,
            laser_scan_frame_id=msg.laser_scan.header.frame_id,
            laser_scan_stamp_ns=stamp(msg.laser_scan),
            camera_info_stamps_ns=[stamp(i) for i in msg.left_camera_info],
            camera_intrinsics=[list(i.k) for i in msg.left_camera_info],
            local_transforms=[dict(translation=[t.translation.x,t.translation.y,t.translation.z],
                quaternion=[t.rotation.x,t.rotation.y,t.rotation.z,t.rotation.w]) for t in msg.local_transform])
    def preview(msg):
        preview_clouds.append(dict(stamp_ns=stamp(msg),width=msg.width,height=msg.height,
            point_step=msg.point_step,row_step=msg.row_step,data_bytes=len(msg.data),
            fields=[f.name for f in msg.fields],empty=msg.width*msg.height==0))
    def observe(native):
        nonlocal pid,subscriptions
        if not subscriptions:
            pid=native._process.pid
            subscriptions=[producer.create_subscription(cls,native.namespace+'/'+topic,callback,private_qos)
                for cls,topic,callback in [(Info,'info',info),(MapGraph,'mapGraph',graph),(SensorData,'sensor_data',sensor),
                    (PointCloud2,'cloud_map',preview)]]
    motion_directory=output/'native-motion';motion_directory.mkdir()
    motion_count=[0]
    def motion(kind,msg):
        sequence=motion_count[0];motion_count[0]+=1
        payload=serialize_message(msg);filename=str(sequence)+'.cdr'
        (motion_directory/filename).write_bytes(payload)
        record=dict(kind=kind,stamp_ns=stamp(msg),frame_id=msg.header.frame_id,file=filename,sha256=hashlib.sha256(payload).hexdigest())
        if kind=='odom':
            p=msg.pose.pose
            record.update(type='nav_msgs/msg/Odometry',child_frame_id=msg.child_frame_id,position=[p.position.x,p.position.y,p.position.z],
                quaternion=[p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w],
                covariance=list(msg.pose.covariance))
        else:
            record.update(type='rtabmap_msgs/msg/OdomInfo',lost=msg.lost,matches=msg.matches,
                inliers=msg.inliers,icp_inliers_ratio=msg.icp_inliers_ratio,
                icp_correspondences=msg.icp_correspondences,icp_structural_complexity=msg.icp_structural_complexity,
                icp_structural_distribution=msg.icp_structural_distribution,covariance=list(msg.covariance))
        numeric=[v for k,v in record.items() if isinstance(v,(int,float)) and not isinstance(v,bool)]
        record['finite_scalar_quality']=bool(np.isfinite(numeric).all())
        record['finite_covariance']=bool(np.isfinite(record['covariance']).all())
        record['positive_covariance_diagonal']=bool(np.all(np.diag(np.asarray(record['covariance']).reshape(6,6))>0))
        if kind=='odom':record['finite_pose']=bool(np.isfinite(record['position']+record['quaternion']).all())
        with (motion_directory/'records.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
    if gateway._motion is not None:
        for kind,cls,topic in [('odom',Odometry,'odom'),('info',OdomInfo,'odom_info_lite')]:
            producer.create_subscription(cls,gateway._motion.namespace+'/'+topic,
                lambda msg,k=kind:motion(k,msg),private_qos)
    def wait(check,seconds):
        deadline=time.monotonic()+seconds
        while not check():
            status=gateway.status()
            if status['state'] in ('error','storage_limit'):raise RuntimeError(json.dumps(status))
            if time.monotonic()>deadline:return False
            time.sleep(.01)
        return True
    try:
        gateway.start()
        if not wait(lambda:odom.get_subscription_count() and clouds.get_subscription_count() and images.get_subscription_count(),30):
            raise TimeoutError('gateway discovery')
        if cold_replay(args):
            gateway._activate(None)
            observe(gateway._active['process'])
            if not wait(lambda:all(producer.count_publishers(s.topic_name) for s in subscriptions),30):
                raise TimeoutError('cold native observer discovery')
        next_send=time.monotonic()
        first_end=max(frames[0]['cloud_stamp_ns'],frames[0]['rgb_stamp_ns'])
        pose_cursor=max(0,sum(sample.stamp_ns<=first_end for sample in samples)-2)
        for index,row in enumerate(frames):
            while time.monotonic()<next_send:time.sleep(.01)
            next_send=time.monotonic()+args.interval
            capture=row['cloud_stamp_ns'];rgb_stamp=row['rgb_stamp_ns'];sent=time.monotonic()
            has_pose=index<args.pose_frames or args.all_poses
            external_headers=[]
            if has_pose:
                while pose_cursor<len(samples) and samples[pose_cursor].stamp_ns<=max(capture,rgb_stamp):
                    sample=samples[pose_cursor];pose_cursor+=1
                    packet=dict(schema='motus.odom/1',stamp_ms=sample.stamp_ms,
                        pose=dict(position=sample.position,quaternion=sample.quaternion,covariance=sample.covariance),
                        vendor=dict(motus_slam=dict(stamp_ns=sample.stamp_ns,status='tracking',segment_id='recorded')))
                    message=String();message.data=json.dumps(packet);odom.publish(message)
                    external_headers.append(sample.stamp_ns)
            else:
                packet=dict(schema='motus.odom/1',stamp_ms=capture//10**6,pose=None,
                    vendor=dict(motus_slam=dict(stamp_ns=capture,status='lost',segment_id='recorded')))
                message=String();message.data=json.dumps(packet);odom.publish(message)
            phase='external_native_bootstrap' if has_pose else 'pose_null'
            rgb=CompressedImage();rgb.header.stamp.sec,rgb.header.stamp.nanosec=divmod(rgb_stamp,10**9)
            rgb.format='jpeg';rgb.data=(directory/row['rgb']).read_bytes()
            fault=args.fault_start<=index<args.fault_end
            if fault and args.fault=='invalid-rgb':rgb.data=b'invalid JPEG';phase+=':invalid_rgb'
            if fault and args.fault=='rgb-gap':phase+=':rgb_gap'
            else:images.publish(rgb)
            depth=np.load(directory/row['depth']).astype('float32')/1000.
            v,u=np.indices(depth.shape,dtype=np.float32);k=camera['K']
            points=np.stack(((u-k[2])*depth/k[0],(v-k[5])*depth/k[4],depth),axis=-1).reshape(-1,3)
            points=points[points[:,2]>0]
            cloud=PointCloud2();cloud.header.stamp.sec,cloud.header.stamp.nanosec=divmod(capture,10**9)
            cloud.height,cloud.width=1,len(points);cloud.point_step,cloud.row_step=12,len(points)*12
            cloud.fields=[PointField(name=n,offset=i*4,datatype=7,count=1) for i,n in enumerate(('x','y','z'))]
            cloud.is_dense=True;cloud.data=array('B',np.asarray(points,dtype='<f4').tobytes())
            if fault and args.fault=='depth-gap':phase+=':depth_gap'
            else:clouds.publish(cloud)
            if not wait(lambda:gateway._active is not None and gateway._active['process']._process is not None,35):
                raise TimeoutError('native map process did not start')
            native=gateway._active['process']
            observe(native)
            if index==0:wait(lambda:capture in infos,8.)
            if native._process.pid!=pid or len(gateway._segments)!=1:raise RuntimeError('native map/process segment changed')
            record=dict(frame=row['frame'],index=index,baseline_only=index==0 and not cold_replay(args),phase=phase,cloud_stamp_ns=capture,rgb_stamp_ns=rgb_stamp,
                sent_monotonic=sent,
                external_pose_headers_ns=external_headers,
                input_pose_source='native RGB-D feature registration' if has_pose else None,
                native=deepcopy(infos.get(capture)),images=inputs.get(capture),status=gateway.status())
            records.append(record)
            with (output/'info.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            print(json.dumps(dict(index=index,phase=phase,native=record['native'],recovery=record['status']['native_recovery'])),flush=True)
        # Future measured clouds may complete earlier asynchronous projection.
        wait(lambda:not gateway._queue and not gateway._motion_waiting,8.)
        source_by_stamp={r['cloud_stamp_ns']:r for r in records}
        targets=[stamp for stamp,data in inputs.items() if data['rgb'] or
                 source_by_stamp.get(stamp,{}).get('input_pose_source') is not None]
        last_target=max(targets) if targets else None
        drained=wait(lambda:last_target is None or last_target in infos,15.)
        gateway.close();artifacts=gateway.artifacts()
        for record in records:
            record['native']=deepcopy(infos.get(record['cloud_stamp_ns']))
            record['images']=inputs.get(record['cloud_stamp_ns'])
        (output/'info.jsonl').write_text(''.join(json.dumps(record)+'\n' for record in records))
        identity,revision,_=save_rtabmap_revision(output/'archive',artifacts,settings,'real captured RGB-D replay',None,'replay')
        reopened=load_map_snapshots(output/'archive',identity,revision,max_points=settings['preview_max_points'])
        expected=[s['snapshot'] for s in artifacts['segments'] if s['snapshot'] is not None]
        identical=[s['packet'] for s in reopened]==expected
        with sqlite3.connect(artifacts['segments'][0]['database']) as db:
            links=db.execute('SELECT from_id,to_id,type FROM Link').fetchall()
            node_stamps=dict(db.execute('SELECT id,stamp FROM Node WHERE id>0'))
            try:optimized=_optimized_poses(db)
            except ValueError:optimized={}
        persisted_nodes=set(node_stamps);nodes=len(persisted_nodes)
        motion_rows=[json.loads(line) for line in (motion_directory/'records.jsonl').read_text().splitlines()] if motion_count[0] else []
        evidence=recovery_evidence(records,links,persisted_nodes-historical_nodes,
            graph_nodes=optimized or (),original_root=original_root,
            live_graphs=graphs,node_stamps=node_stamps)
        acceptance={kind: bool(reopened) and identical and drained and len(artifacts['segments'])==1 and
            any(run['kind']==kind and len(run['nodes'])>=3 for run in evidence['tracking_after_recovery'])
            for kind in ('short_visual','global')}
        result=dict(native_pid=pid,segments=len(artifacts['segments']),database_nodes=nodes,links=links,
            native_motion_messages=motion_count[0],first_frame_baseline_only=not cold_replay(args),
            native_drain=dict(last_actual_input_stamp_ns=last_target,info_observed=drained),
            original_header_verification=header_evidence(frames,inputs,motion_rows),
            effective_native_parameters=json.loads(gateway._active['process'].parameter_path.read_text())['/**']['ros__parameters'],
            persisted_node_ids=sorted(persisted_nodes),selected_source_frames=[r['frame'] for r in frames],
            map_id=identity,revision=revision,archive_reopened_identically=identical,
            evidence=evidence,
            offline_recovery_acceptance=acceptance['global'],
            offline_recovery_acceptance_by_type=acceptance,
            statistics=artifacts['statistics'],input_frames=len(records),pose_null_frames=sum(r['input_pose_source'] is None for r in records),
            provenance=dict(json.loads((directory/'provenance.json').read_text()),
                native_bootstrap=native_bootstrap,
                offline_replay=dict(
                detection_schedule='DetectionRate=0, offline ingestion of all original captures' if args.offline_all_captures else 'native default DetectionRate',
                keyframe_insertion='LinearUpdate=0/AngularUpdate=0 to retain static observations' if args.insert_every_observation else 'native default update thresholds',
                visual_geometric_matching_thresholds='production native values unchanged',
                interpretation='static replay node growth is not physical movement or independent accuracy evidence')),motion_error=None,
            motion_error_reason='passive recording has no independent motion ground truth')
        (output/'graph.json').write_text(json.dumps(graphs));(output/'native-info.json').write_text(json.dumps(infos))
        (output/'native-inputs.json').write_text(json.dumps(inputs,indent=2))
        (output/'native-preview-clouds.json').write_text(json.dumps(preview_clouds,indent=2))
        (output/'result.json').write_text(json.dumps(result,indent=2));return result
    except Exception as error:
        (output/'failure.json').write_text(json.dumps(dict(error=str(error),status=gateway.status(),frames=len(records)),indent=2))
        raise
    finally:
        gateway.close();executor.shutdown(timeout_sec=5.);spin.join(timeout=5.)
        gateway.destroy_node();producer.destroy_node();rclpy.shutdown()


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    check=commands.add_parser('audit');check.add_argument('recording')
    prepare=commands.add_parser('prepare-registered');prepare.add_argument('recording');prepare.add_argument('output')
    prepare.add_argument('--driver',required=True);prepare.add_argument('--registration-evidence',required=True)
    prepare.add_argument('--color-topic',required=True);prepare.add_argument('--depth-topic',required=True)
    prepare.add_argument('--color-info-topic',required=True);prepare.add_argument('--depth-info-topic',required=True)
    aligned=commands.add_parser('prepare-aligned');aligned.add_argument('recording');aligned.add_argument('output')
    aligned.add_argument('--driver',required=True);aligned.add_argument('--calibration',required=True)
    aligned.add_argument('--color-topic',required=True);aligned.add_argument('--depth-topic',required=True)
    aligned.add_argument('--depth-values',choices=('raw-sdk','millimetres'),required=True)
    aligned.add_argument('--stamp-semantics',required=True);aligned.add_argument('--publication-cycle-evidence',required=True)
    native=commands.add_parser('bootstrap');native.add_argument('prepared');native.add_argument('output');native.add_argument('--end-frame',type=int)
    play=commands.add_parser('replay');play.add_argument('prepared');play.add_argument('output')
    play.add_argument('--bootstrap-poses');play.add_argument('--pose-frames',type=int,default=8)
    play.add_argument('--all-poses',action='store_true');play.add_argument('--interval',type=float,default=1.3)
    play.add_argument('--insert-every-observation',action='store_true')
    play.add_argument('--offline-all-captures',action='store_true')
    play.add_argument('--fault',choices=('none','invalid-rgb','rgb-gap','depth-gap'),default='none')
    play.add_argument('--fault-start',type=int,default=12);play.add_argument('--fault-end',type=int,default=16)
    play.add_argument('--resume-run')
    play.add_argument('--start-frame',type=int,default=0);play.add_argument('--end-frame',type=int)
    args=parser.parse_args()
    result={'audit':audit,'prepare-registered':prepare_registered,'prepare-aligned':prepare_aligned,'bootstrap':bootstrap,'replay':replay}[args.command](args.recording if args.command=='audit' else args)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
