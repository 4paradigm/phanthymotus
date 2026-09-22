#!/usr/bin/env python3
"""Offline sampled operator pose replay; no robot/network or fabricated enable."""
import argparse,collections,hashlib,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'actucore/plugins'))
from teleop.kinematics import TianyiIK,RelativeMapping

def replay(profile,source):
    solver=TianyiIK(profile);mapper=RelativeMapping();seen=set();clutch=None
    counts=collections.Counter();details=[]
    for line in source.read_text().splitlines():
        row=json.loads(line);f=(row['card'].get('pose') or {}).get('latest')
        if not f:continue
        identity=(f['boot_id'],f['session_id'],f['epoch'],f['sequence'])
        if identity in seen:continue
        seen.add(identity)
        if not f['deadman']:
            clutch=None;continue
        q=np.asarray(row['driver']['feedback']['q'])
        key=(f['session_id'],f['epoch'],f['clutch_sequence'])
        if key!=clutch:mapper.reset(f,solver.palms(q));clutch=key
        targets=mapper.targets(f);start=time.monotonic()
        try:
            solver.solve(targets,q);code='ok'
            actual=solver.palms(solver.visualization_sample['ik_q'])
            residual=[float(np.linalg.norm(a[:3,3]-b[:3,3])) for a,b in zip(actual,targets)]
        except ValueError as exc:code=str(exc);residual=None
        counts[code]+=1
        details.append({'sequence':f['sequence'],'code':code,'tcp_error_m':residual,
                        'elapsed_ms':(time.monotonic()-start)*1000})
    return {'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'sampled_unique_frames':len(seen),'held_frames':len(details),'results':dict(counts),
        'hardware_output':False,'scope':'sampled raw poses, no network timing or full runtime replay',
        'details':details}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('profile',type=Path);p.add_argument('recording',type=Path)
    a=p.parse_args();print(json.dumps(replay(a.profile,a.recording),indent=2))
