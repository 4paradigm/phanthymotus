"""Explicit diagnostic: reachable synthetic PICO frames, never a real recording."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import yaml
from plugins.teleop import acceptance as a
from plugins.teleop.kinematics import RelativeMapping, TianyiIK, transform
from plugins.teleop.replay import checked_segment, digest, write_json


def generate(solver, base, template):
    base=np.asarray(base,dtype=float)
    mapper=RelativeMapping(.5)
    mapper.controller_offsets={s:transform(solver.profile['controller_to_palm'][s]) for s in ('left','right')}
    palms=solver.palms(base)
    mapper.reset(template,palms)
    rows=[];previous=base.copy()
    for i in range(501):
        t=i*.02;u=t-.5
        gain=max(0.,min(1.,u/3.,(7.5-u)/3.))
        q=base.copy()
        q[[0,7]]+=.08*gain
        q[[3,10]]-=.08*gain
        q[[5,12]]+=.04*gain
        targets=solver.palms(q)
        frame=copy.deepcopy(template)
        held=.5<=t<9.5
        frame.update(sequence=i,clutch_sequence=1 if t>=.5 else 0,deadman=held)
        frame.pop('epoch',None)
        for side,target,robot,(origin,_) in zip(('left','right'),targets,palms,mapper.reference):
            current=np.eye(4)
            current[:3,3]=origin[:3,3]+mapper.rotation.T@(target[:3,3]-robot[:3,3])/.5
            current[:3,:3]=mapper.rotation.T@target[:3,:3]@robot[:3,:3].T@mapper.rotation@origin[:3,:3]
            controller=current@np.linalg.inv(mapper.controller_offsets[side])
            frame[side+'_controller']={'position':controller[:3,3].tolist(),
                'orientation':Rotation.from_matrix(controller[:3,:3]).as_quat().tolist()}
            frame['controllers'][side]['buttons']=[0.,float(held)]
        mapped=mapper.targets(frame)
        if any(not np.allclose(x,y,atol=1e-9) for x,y in zip(mapped,targets)):
            raise ValueError('synthetic_inverse_mapping_failed')
        # Joint path is an offline geometric reference, not fabricated feedback.
        checked_segment(solver,previous,q,step=.02)
        previous=q
        rows.append({'received_ns':i*20_000_000,'source':'synthetic_fk_reference',
                     'frame':frame,'reference_q':q.tolist()})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--template-recording',required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    cfg=yaml.safe_load(args.config.read_text())['plugins']['teleop']
    if cfg.get('robot_profile')!='tianyi2':raise ValueError('tianyi_only')
    root=Path(cfg['capture']['state_file']).parent/'recordings'
    if Path(args.template_recording).name!=args.template_recording:raise ValueError('recording_id')
    lock=open(root.parent/'replay.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    a.idle()
    link,ex,context,thread=a.ros_link(cfg)
    try:baseline=a.fresh_baseline(link)
    finally:link.close();ex.shutdown();context.shutdown();thread.join(timeout=2)
    base=np.asarray(baseline[-1]['q'])
    source=root/args.template_recording/'poses.jsonl'
    template=next(json.loads(s)['frame'] for s in source.read_text().splitlines()
                  if json.loads(s)['frame']['deadman'])
    solver=TianyiIK(cfg['calibration_path'])
    rows=generate(solver,base,template)
    directory=root/('synthetic-fk-'+str(time.time_ns()));directory.mkdir()
    (directory/'poses.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    write_json(directory/'baseline-observed.json',baseline)
    write_json(directory/'manifest.json',{'source':'synthetic_fk_reference','real_pico_recording':False,
        'purpose':'supplemental production-chain diagnosis; does not replace original replay',
        'profile_sha256':digest(cfg['calibration_path']),'template_sha256':digest(source),
        'duration_s':10,'joint_amplitude_rad':.08,'base_measured_q':base.tolist(),
        'input_sha256':digest(directory/'poses.jsonl')})
    print('SYNTHETIC GEOMETRY PASS',str(directory),flush=True)
    if not args.execute:return
    def verify_start(config,recording,evidence):
        a.idle()
        if digest(config['calibration_path'])!=solver.profile_sha256:raise ValueError('profile_changed')
        link,ex,context,thread=a.ros_link(config)
        try:now=np.asarray(a.fresh_baseline(link)[-1]['q'])
        finally:link.close();ex.shutdown();context.shutdown();thread.join(timeout=2)
        if np.max(np.abs(now-base))>.005:raise ValueError('synthetic_start_moved')
        checked_segment(solver,now,base,step=.02)
        write_json(evidence/'origin.result.json',{'passed':True,'hardware_motion_requested':False,
            'source':'fresh_current_measured_pose','max_error_rad':float(np.max(np.abs(now-base)))})
    original=a.position_recorded_origin
    try:
        a.position_recorded_origin=verify_start
        a.combined(cfg,directory,directory)
    finally:a.position_recorded_origin=original
    print('SYNTHETIC PRODUCTION CHAIN PASS',str(directory),flush=True)


if __name__=='__main__':main()
