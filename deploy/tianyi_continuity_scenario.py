"""Explicit supplemental input scenario; never modifies the original recording.

Run only through the occupancy-checked host runner. Extends input dwell times;
does not modify the production solver, controller, limits, or recovery methods.
"""
import copy
import fcntl
import json
from pathlib import Path
import signal
import time


def make_rows(source, waypoints):
    origin=waypoints[0]['frame'];target=waypoints[10]['frame']
    # This source pose failed IK near the end of the original physical replay.
    # Failure is observed again, never assumed or injected into the controller.
    failed=source[-1]['frame']
    phases=[('release',1,origin,False,0),('baseline',2,origin,True,1),
        ('sustained',8,target,True,1),('failure_input',2,failed,True,1),
        ('recovered',8,target,True,1),('release_again',1,target,False,1),
        ('regrip_baseline',2,origin,True,2),('regrip_motion',5,target,True,2)]
    rows=[];manifest=[]
    for name,seconds,pose,held,clutch in phases:
        first=len(rows)
        for _ in range(seconds*50):
            index=len(rows);frame=copy.deepcopy(pose)
            frame.update(sequence=index+1,clutch_sequence=clutch,deadman=held,
                         client_monotonic_ns=index*20_000_000+1)
            for controller in frame['controllers'].values():controller['buttons'][1]=1. if held else 0.
            rows.append({'received_ns':index*20_000_000+1,'frame':frame})
        manifest.append({'phase':name,'first_sequence':first+1,'last_sequence':len(rows),
                         'source_sequence':pose['sequence'],'duration_seconds':seconds})
    return rows,manifest


def assess(directory,phases):
    trace=[json.loads(s) for s in (directory/'actucore.jsonl').read_text().splitlines()]
    feedback=[json.loads(s) for s in (directory/'combined.jsonl').read_text().splitlines()]
    results=[]
    for phase in phases:
        lo,hi=phase['first_sequence'],phase['last_sequence']
        applies=[r for r in trace if lo<=r['input_sequence']<=hi]
        states=[r for r in feedback if lo<=r['sequence']<=hi and r.get('q')]
        publications=[r for r in applies if r.get('published')]
        errors={}
        for r in applies:
            code=(r.get('failure') or {}).get('code')
            if code:errors[code]=errors.get(code,0)+1
        excursion=[max(r['q'][j] for r in states)-min(r['q'][j] for r in states)
                   for j in range(14)] if states else None
        results.append({**phase,'applies':len(applies),'publications':len(publications),
            'first_publication_sequence':publications[0]['input_sequence'] if publications else None,
            'failures':errors,'actual_excursion_rad':excursion,
            'first_q':states[0]['q'] if states else None,'last_q':states[-1]['q'] if states else None})
    combined=json.loads((directory/'combined.result.json').read_text())
    expected={'sustained','recovered','regrip_motion'}
    resumed=all(r['publications']>0 for r in results if r['phase'] in expected)
    failure_seen=any(r['failures'] for r in results if r['phase']=='failure_input')
    return {'synthetic_supplement':True,'phases':results,'expected_failure_observed':failure_seen,
            'all_motion_phases_published':resumed,'stop_confirmed':combined['stop_confirmed'],
            'error':combined['error'],'tracking_error_policy':'report_only',
            'note':'Publication alone is not physical completion; inspect measured motion and independent ROS evidence.'}


def main():
    import yaml
    from plugins.teleop import acceptance as a
    cfg=yaml.safe_load(Path('/work/config.yaml').read_text())['plugins']['teleop']
    if cfg.get('robot_profile')!='tianyi2':raise ValueError('tianyi_only')
    recording=Path('/var/lib/motus-teleop/recordings/20260921T161243Z-b811f2d3')
    source=[json.loads(s) for s in (recording/'poses.jsonl').read_text().splitlines()]
    package=json.loads((recording/'analysis/trajectory.json').read_text())
    if a.digest(recording/'poses.jsonl')!=package['recording_sha256']:raise ValueError('recording_changed')
    for name,value in package['compiled_sources'].items():
        if a.digest(Path(a.__file__).with_name(name))!=value:raise ValueError('trajectory_code_changed')
    a.idle()
    directory=recording/('execution-'+str(time.time_ns()));directory.mkdir()
    rows,phases=make_rows(source,package['waypoints'])
    a.write_json(directory/'continuity-input.json',{'synthetic_supplement':True,
        'recording_sha256':package['recording_sha256'],'phases':phases,'rows':rows})
    original=a.paced_latest_inputs
    # Change only the test input iterator. All live runtime/IK/Driver paths remain real.
    a.paced_latest_inputs=lambda unused,start:original(rows,start)
    lock=open(Path(cfg['capture']['state_file']).parent/'replay.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def interrupted(*args):raise InterruptedError('operator_cancelled')
    for s in handlers:signal.signal(s,interrupted)
    try:a.combined(cfg,recording,directory)
    finally:
        for s,handler in handlers.items():signal.signal(s,handler)
        lock.close()
        a.paced_latest_inputs=original
        if (directory/'combined.result.json').exists():
            result=assess(directory,phases);a.write_json(directory/'continuity-result.json',result)
            print(json.dumps(result),flush=True)


if __name__=='__main__':main()
