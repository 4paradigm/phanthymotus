"""Offline recording analysis and bounded, authenticated trajectory execution."""
from __future__ import annotations
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from .kinematics import RelativeMapping, TianyiIK, finite, transform, MEASURED_LIMIT_TOLERANCE_RAD, POSITION_LEAD_SECONDS


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):Path(path).write_text(json.dumps(value,allow_nan=False,indent=2)+'\n')

def checked_segment(solver, a, b, step=.004):
    a=finite(a,(14,));b=finite(b,(14,))
    lo=solver.model.lowerPositionLimit[solver.indices];hi=solver.model.upperPositionLimit[solver.indices]
    if np.any(a<lo) or np.any(a>hi) or np.any(b<lo) or np.any(b>hi):raise ValueError('joint_limit')
    n=max(1,int(math.ceil(np.max(np.abs(b-a))/step)))
    for i in range(n):
        start=a+(b-a)*i/n;end=a+(b-a)*(i+1)/n
        if hasattr(solver,'_safe_transition'):solver._safe_transition(start,end)
        else:
            solver._safe_configuration(start,excursion=np.abs(end-start))
            solver._safe_configuration(end)


def analyze(recording, profile, destination):
    recording=Path(recording);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((recording/'manifest.json').read_text())
    complete=json.loads((recording/'result.json').read_text())
    if not complete.get('complete') or complete.get('dropped'):raise ValueError('recording_incomplete')
    if digest(profile)!=manifest['profile_sha256']:
        recorded=copy.deepcopy(manifest.get('profile'));current=json.loads(Path(profile).read_text())
        if not isinstance(recorded,dict):raise ValueError('recording_profile_changed')
        recorded.pop('joint_velocity_rad_s',None);current.pop('joint_velocity_rad_s',None)
        if recorded!=current:raise ValueError('recording_geometry_changed')
    rows=[json.loads(line) for line in (recording/'poses.jsonl').read_text().splitlines()]
    if len(rows)!=complete['frames']:raise ValueError('recording_frame_count')
    solver=TianyiIK(profile);scale=manifest['position_scale'];segments=[];diagnostics=[]
    for mode in ('recorded_feedback','continuous_solution'):
        solver.reset_target_state()
        mapper=RelativeMapping(scale)
        mapper.controller_offsets={s:transform(solver.profile['controller_to_palm'][s]) for s in ('left','right')}
        key=None;previous=None;seed=None;segment=[]
        for row in rows:
            f=row['frame'];stamp=row['received_ns'];obs=row['driver'].get('feedback',{})
            entry={'mode':mode,'input_sequence':f['sequence'],'received_ns':stamp}
            current_key=(f.get('epoch'),f['clutch_sequence'])
            gap=previous is not None and (stamp<=previous['received_ns'] or stamp-previous['received_ns']>100_000_000 or f['sequence']!=previous['frame']['sequence']+1)
            valid_input=f['deadman'] and all(f['tracking'].values()) and not gap
            try:
                if not valid_input:raise ValueError('input_gap' if gap else 'input_inactive')
                if not 0<=row.get('observed_ns',stamp)-obs.get('arm_ns',0)<=100_000_000:raise ValueError('recorded_feedback_stale')
                measured=finite(obs['q'],(14,))
                if key!=current_key:
                    if segment and mode=='continuous_solution':segments.append(segment)
                    segment=[];mapper.reset(f,solver.palms(measured));seed=measured;key=current_key
                    solver.reset_target_state()
                targets=mapper.targets(f);initial=measured if mode=='recorded_feedback' else seed
                begin=time.monotonic();result=solver.solve(targets,initial)
                entry.update(code='ok',elapsed_ms=(time.monotonic()-begin)*1000,
                             target_tcp=[x[:3,3].tolist() for x in targets],
                             ik_q=solver.visualization_sample['ik_q'].tolist(),step_q=result)
                # Use the full solution as an offline waypoint. Validate the entire
                # transition independently; display-only IK is never a motion permit.
                full=finite(entry['ik_q'],(14,))
                # Recorded-feedback mode is diagnostic only. Only the continuous
                # branch exports motion waypoints and needs full transit checks.
                entry['trajectory_checked']=mode=='continuous_solution'
                if mode=='continuous_solution':
                    checked_segment(solver,initial,full)
                    if not segment:segment.append({'q':initial.tolist(),'source_ns':stamp,'sequence':f['sequence'],'frame':copy.deepcopy(f)})
                    segment.append({'q':full.tolist(),'source_ns':stamp,'sequence':f['sequence'],'frame':copy.deepcopy(f)})
                    seed=full
            except (ValueError,KeyError) as exc:
                entry['code']=str(exc)
                if segment and mode=='continuous_solution':segments.append(segment)
                segment=[]
                # Collision/IK failures do not redefine the controller origin.
                # Only actual release/tracking loss starts a new mapping baseline.
                if not f['deadman'] or not all(f['tracking'].values()):key=None;seed=None
            diagnostics.append(entry);previous=row
        if segment and mode=='continuous_solution':segments.append(segment)
    (destination/'ik.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in diagnostics))
    useful=[s for s in segments if len(s)>2 and np.max(np.ptp(np.asarray([x['q'] for x in s]),axis=0))>=.03]
    if not useful:
        write_json(destination/'analysis.json',{'passed':False,'reason':'no_continuous_visible_segment','rows':len(rows)})
        raise ValueError('no_continuous_visible_segment')
    selected=max(useful,key=lambda s:(sum(np.max(np.ptp(np.asarray([x['q'] for x in s])[:,sl],axis=0))>=.03 for sl in (slice(0,7),slice(7,14))),len(s)))
    # At most six seconds of captured input per phase; never splice failed spans.
    selected=[x for x in selected if x['source_ns']-selected[0]['source_ns']<=6_000_000_000]
    package={'schema':'motus.teleop.replay.v1','recording_sha256':digest(recording/'poses.jsonl'),
             'profile_sha256':digest(profile),'recorded_profile_sha256':manifest['profile_sha256'],'sources':manifest['sources'],'position_scale':scale,
             'compiled_sources':{n:digest(Path(__file__).with_name(n)) for n in ('kinematics.py','workspace.py','reachability.py','trajectory.py','replay.py','acceptance.py','adapter.py','tianyi.py','runtime.py','dispatch.py')},
             'waypoints':selected,'hardware_output':False}
    write_json(destination/'trajectory.json',package)
    write_json(destination/'analysis.json',{'passed':True,'rows':len(rows),'valid_segments':len(useful),
               'selected_waypoints':len(selected),'recording_sha256':package['recording_sha256']})
    return package


def schedule(package, solver, measured):
    """Checked transitions and arm phases, <=45s, including stationary plateaus."""
    raw=finite(measured,(14,));lo=solver.model.lowerPositionLimit[solver.indices];hi=solver.model.upperPositionLimit[solver.indices]
    measured_tolerance=min(MEASURED_LIMIT_TOLERANCE_RAD,solver.velocity*.02)
    if np.any(raw<lo-measured_tolerance) or np.any(raw>hi+measured_tolerance):raise ValueError('starting_joint_limit')
    q=np.clip(raw,lo,hi)  # Bounded measured settling offset; commands remain strictly in limits.
    points=[{'t':0.,'q':q.tolist(),'phase':'baseline','initial_limit_adjustment_rad':(q-raw).tolist()}];t=1.
    points.append({'t':t,'q':q.tolist(),'phase':'baseline'})
    velocity=min(1.,solver.velocity)
    if velocity<=0:raise ValueError('velocity')
    for phase in ('left','right','both'):
        fixed=q.copy()  # The inactive arm stays at this phase's entry target.
        last_source=package['waypoints'][0]['source_ns']
        for w in package['waypoints']:
            target=finite(w['q'],(14,)).copy()
            if phase=='left':target[7:]=fixed[7:]
            if phase=='right':target[:7]=fixed[:7]
            checked_segment(solver,q,target,step=velocity*.02)
            dt=max(.02,float(np.max(np.abs(target-q)))/velocity,(w['source_ns']-last_source)/1e9)
            t+=dt;points.append({'t':t,'q':target.tolist(),'phase':phase});q=target;last_source=w['source_ns']
        t+=1.;points.append({'t':t,'q':q.tolist(),'phase':phase+'_plateau'})
    if getattr(getattr(solver, 'trajectory', None), 'enabled', False):
        from .trajectory import smooth_schedule
        points = smooth_schedule(points, solver.trajectory, solver)
    if points[-1]['t']>40:raise ValueError('trajectory_exceeds_round_budget')  # 5s reserved for pause/recovery.
    return points


def target_at(points, elapsed):
    for a,b in zip(points,points[1:]):
        if elapsed<=b['t']:
            if 'polynomial' in a:
                local = max(0., min(elapsed-a['t'], b['t']-a['t']))
                return np.polyval(np.asarray(a['polynomial']), local).tolist(), b['phase']
            u=max(0,min(1,(elapsed-a['t'])/(b['t']-a['t'])))
            return [x+(y-x)*u for x,y in zip(a['q'],b['q'])],b['phase']
    return points[-1]['q'],points[-1]['phase']


def execute(link, points, solver, sink, *, clock=time.monotonic, sleep=time.sleep):
    """Explicit motion caller only. No IK; time freezes during a bounded hold."""
    start=clock();elapsed=0.;last=clock();paused=False;failure=None;released=False;feedback_wait=None;hold_wait=None;recoveries=0;geometry_timeouts=0
    geometry_step_scale=1.
    link.management_retry=True;link.prepare_transport()
    def manage(method, seconds=2.5):
        end=min(clock()+seconds,start+45.)
        while clock()<end:
            try:return method(clock()+.09)
            except ValueError as exc:
                if str(exc)=='driver_lease_rebased':
                    if link.lease or link.management_request:raise ValueError('management_release_unconfirmed')
                    method=link.claim  # Old transaction was cancelled and physically released.
                elif str(exc) not in ('driver_management_pending','driver_feedback_missing','driver_feedback_stale_or_different_clock','robot_not_stopped'):raise
                sink({'event':'waiting_management','action':method.__name__,'reason':str(exc),'elapsed':elapsed,'hardware_target_sent':False})
                sleep(.01)
        raise ValueError('management_retry_exhausted')
    try:
        manage(link.claim)
        while elapsed<points[-1]['t']:
            now=clock()
            if now-start>=45:raise ValueError('round_deadline')
            try:state=link.feedback()
            except ValueError as exc:
                if str(exc) not in ('driver_feedback_missing','driver_feedback_stale_or_different_clock'):raise
                if feedback_wait is None:feedback_wait=now
                if now-feedback_wait>=.3:raise ValueError('driver_feedback_timeout') from exc
                sleep(.01);last=clock();continue
            # MCP resume rotates the lease before DDS publishes the next snapshot.
            # A pre-lease operator_pause frame cannot describe the new session.
            stamp=getattr(link,'lease_started_ns',0)
            if state.get('monotonic_ns',stamp)<stamp:
                state=link.feedback_after(stamp,clock()+.1)
            f=state['feedback']
            if state.get('state')=='fault':raise ValueError('driver_fault:'+str(state.get('reason')))
            age=time.monotonic_ns()-f['arm_ns']
            if age<0:raise ValueError('arm_feedback_clock')
            if age>100_000_000:
                if feedback_wait is None:feedback_wait=now
                if now-feedback_wait>=.3:raise ValueError('arm_feedback_stale')
                sleep(.01);last=clock();continue
            feedback_wait=None
            if not paused and elapsed>=points[-1]['t']/2:
                pause_end=clock()+2.5
                while clock()<pause_end:
                    try:
                        if link.pause(min(pause_end,clock()+.2)):break
                    except ValueError as exc:
                        if str(exc) not in ('driver_feedback_missing','driver_feedback_stale_or_different_clock','driver_feedback_ack_timeout'):raise
                    sleep(.01)
                else:raise ValueError('pause_unconfirmed')
                sleep(.2);manage(link.resume);paused=True;last=clock();continue
            if state.get('state')=='hold':
                if not state.get('continuation_allowed'):
                    reason=state.get('reason')
                    recoverable={'command_timeout','command_expired','arm_ns_stale','power_ns_stale','fixed_ns_stale'}
                    lease=getattr(link,'lease',None)
                    if reason not in recoverable or not isinstance(lease,dict) or any(state.get(k)!=lease.get(k) for k in ('boot_id','session_id')) or state.get('ownership_held') is not True:
                        raise ValueError('driver_hold:'+str(reason))
                    if hold_wait is None:hold_wait=now
                    if now-hold_wait>=2.5:raise ValueError('hold_recovery_exhausted')
                    fresh=all(type(f.get(k)) is int and 0<=time.monotonic_ns()-f[k]<=100_000_000 for k in ('arm_ns','power_ns','fixed_ns'))
                    if not state.get('hold_confirmed') or not fresh:
                        sleep(.01);last=clock();continue
                    manage(link.resume);recoveries+=1;hold_wait=None;last=clock()
                    sink({'event':'explicit_hold_recovery','elapsed':elapsed,'reason':reason,'recovery':recoveries,'hardware_target_sent':False})
                    continue
                if not state.get('continuation_ready'):sleep(.01);last=clock();continue
            hold_wait=None
            previous_elapsed=elapsed
            elapsed+=min(.02,max(0,now-last));last=now
            target,phase=target_at(points,elapsed)
            # Confirm real joint-box transit at execution time as well as offline.
            measured=finite(f['q'],(14,))
            previous=state.get('commanded_q')
            previous=measured if previous is None else finite(previous,(14,))
            step=previous+np.clip(np.asarray(target)-previous,-solver.velocity*.02,solver.velocity*.02)
            step=np.clip(step,measured-solver.velocity*POSITION_LEAD_SECONDS,measured+solver.velocity*POSITION_LEAD_SECONDS)
            # Retry a shorter advance from the last command when proving the
            # larger independent joint box was too expensive. The reference
            # target does not change. Tracking error is a reported measurement.
            step=previous+geometry_step_scale*(step-previous)
            step=np.clip(step,measured-solver.velocity*POSITION_LEAD_SECONDS,measured+solver.velocity*POSITION_LEAD_SECONDS)
            geometry_started=clock();geometry_cpu_started=time.thread_time()
            geometry_gc=[g['collections'] for g in gc.get_stats()]
            geometry_deadline=geometry_started+.06
            def geometry_metrics():
                return {'geometry_wall_ms':(clock()-geometry_started)*1000,
                        'geometry_thread_cpu_ms':(time.thread_time()-geometry_cpu_started)*1000,
                        'geometry_gc_collections':[g['collections']-before for g,before in zip(gc.get_stats(),geometry_gc)]}
            def geometry_budget():
                if clock()>=geometry_deadline:raise ValueError('collision_check_timeout')
            try:
                advance_scale=1.
                if getattr(getattr(solver, 'trajectory', None), 'enabled', False):
                    from .trajectory import execution_state
                    step = solver.command_step(target, measured, previous, geometry_budget, execution_state(state))
                elif hasattr(solver,'_safe_advance'):
                    step,advance_scale=solver._safe_advance(measured,previous,step,geometry_budget)
                elif hasattr(solver,'_safe_transition'):
                    solver._safe_transition(measured,step,geometry_budget)
                    solver._safe_transition(previous,step,geometry_budget)
                else:
                    solver._safe_configuration(measured,excursion=np.abs(step-measured));solver._safe_configuration(step)
            except ValueError as exc:
                sink({'event':'rejected_transition','elapsed':elapsed,'phase':phase,
                      'measured_q':measured.tolist(),'previous_q':previous.tolist(),
                      'candidate_q':step.tolist(),'desired_q':target,'arm_ns':f['arm_ns'],
                      'error':str(exc),'hardware_target_sent':False,**geometry_metrics()})
                if str(exc)=='collision_check_timeout' and not getattr(getattr(solver, 'trajectory', None), 'enabled', False):
                    geometry_timeouts+=1
                    if geometry_timeouts<3:
                        geometry_step_scale*=.5
                        elapsed=previous_elapsed;last=clock();sleep(.01);continue
                raise
            geometry_timeouts=0
            geometry_result=geometry_metrics()
            geometry_result['geometry_step_scale']=geometry_step_scale*advance_scale
            geometry_step_scale=min(1.,geometry_step_scale*1.25)
            # Send exactly the checked target; the Driver must not advance toward
            # an unchecked distant waypoint between producer updates.
            try:
                sent=link.send(step.tolist(),[0.,0.],clock()+.1,wait_for_execution=False,allow_continuation=True)
            except ValueError as exc:
                if str(exc) not in ('driver_feedback_missing','driver_feedback_stale_or_different_clock','driver_holding'):raise
                # The pre-send snapshot may age or enter hold during geometry.
                # Drop this candidate; the next cycle validates the actual hold
                # reason/owner and performs recovery, or rejects faults normally.
                elapsed=previous_elapsed;last=clock()
                sink({'event':'waiting_send_feedback','reason':str(exc),'elapsed':elapsed,'hardware_target_sent':False})
                sleep(.01);continue
            if sent is False:
                elapsed=previous_elapsed;last=clock()
                sink({'event':'waiting_driver_receipt','elapsed':elapsed,'hardware_target_sent':False})
                sleep(.01);continue
            sink({'monotonic_ns':time.monotonic_ns(),'elapsed':elapsed,'phase':phase,'target_q':target,
                  'trajectory_diagnostics':getattr(getattr(solver, 'trajectory', None), 'diagnostics', None),
                  'sent':sent,'sent_target_q':step.tolist(),'sequence':link.seq,
                  'publish':getattr(link,'last_send',None),
                  'session_id':link.lease.get('session_id') if link.lease else None,'driver':{k:state.get(k) for k in
                  ('state','reason','applied_sequence','commanded_q','output_active','stop_confirmed','diagnostics','last_vendor_command')},
                  'q':f['q'],'dq':f['dq'],'arm_ns':f['arm_ns'],**geometry_result})
            sleep(max(0,.02-(clock()-now)))
    except BaseException as exc:
        failure=type(exc).__name__+':'+str(exc)
    finally:
        try:
            end=clock()+3
            while clock()<end:
                try:
                    if link.stop(min(end,clock()+.25)):released=True;break
                except ValueError as exc:
                    if str(exc) not in ('driver_feedback_missing','driver_feedback_stale_or_different_clock','driver_feedback_ack_timeout','driver_management_pending'):raise
                sleep(.02)
        except (ValueError,OSError) as exc:failure=(failure or '')+';release:'+str(exc)
        sink({'event':'round_end','error':failure,'stop_confirmed':released,'paused_and_resumed':paused})
    if not released:raise ValueError('stop_unconfirmed')
    if failure:raise ValueError(failure)


def evaluate(rows):
    samples=[r for r in rows if 'q' in r];end=next((r for r in reversed(rows) if r.get('event')=='round_end'),{})
    if not samples:return {'passed':False,'reason':'no_actual_samples'}
    q=np.asarray([finite(r['q'],(14,)) for r in samples]);target=np.asarray([finite(r['target_q'],(14,)) for r in samples])
    baseline=q[[i for i,r in enumerate(samples) if r['phase']=='baseline']]
    if len(baseline)<5:return {'passed':False,'reason':'baseline_samples_missing'}
    noise=np.ptp(baseline,axis=0);actual=np.ptp(q,axis=0);command=np.ptp(target,axis=0)
    visible=(actual>=.03)&(actual>5*noise)&(command>=.03)
    signed=np.sum((q-q[0])*(target-target[0]),axis=0)>0
    arms=[bool(np.any(visible[sl]&signed[sl])) for sl in (slice(0,7),slice(7,14))]
    plateau=[i for i,r in enumerate(samples) if r['phase'].endswith('_plateau')]
    # Evaluate the latter half of each plateau, not transient approach frames.
    stable=[];missing=[]
    for phase in ('left_plateau','right_plateau','both_plateau'):
        ids=[i for i in plateau if samples[i]['phase']==phase]
        if len(ids)<5 or samples[ids[-1]]['elapsed']-samples[ids[0]]['elapsed']<.8:
            missing.append(phase)
        stable.extend(ids[len(ids)//2:])
    error=np.max(np.abs(q[stable]-target[stable]),axis=0) if stable else np.full(14,float('inf'))
    passed=all(arms) and not missing and bool(stable) and end.get('stop_confirmed') is True and end.get('paused_and_resumed') is True and not end.get('error')
    return {'passed':passed,'missing_plateaus':missing,'arms_visible':arms,'actual_excursion_rad':actual.tolist(),
        'noise_rad':noise.tolist(),'stable_max_error_rad':error.tolist() if stable else None,
        'tracking_error_policy':'report_only','tolerance_rad':None,'stop_confirmed':end.get('stop_confirmed'),
        'error':end.get('error'),'hardware_evidence':'measured_joint_feedback'}


def evaluate_combined(rows, baseline, *, stop_confirmed, error):
    """Compare measured movement to production targets, never output flags."""
    if len(baseline)<5 or len({r['arm_ns'] for r in baseline})<5:
        return {'passed':False,'reason':'baseline_samples_missing'}
    usable=[r for r in rows if r.get('target_q') is not None]
    if len(usable)<20:
        return {'passed':False,'reason':'target_samples_missing'}
    base=np.asarray([finite(r['q'],(14,)) for r in baseline])
    q=np.asarray([finite(r['q'],(14,)) for r in usable])
    commands=np.asarray([finite(r['target_q'],(14,)) for r in usable])
    # The orange model is the full IK goal, not the bounded next actuator step.
    # Legacy reports remain readable but are explicitly labeled as such.
    reference_present=any('ik_reference_q' in r for r in usable)
    if reference_present and any(r.get('ik_reference_q') is None for r in usable):
        return {'passed':False,'reason':'ik_reference_samples_missing'}
    targets=(np.asarray([finite(r['ik_reference_q'],(14,)) for r in usable])
             if reference_present else commands)
    noise=np.ptp(base,axis=0)
    actual=np.ptp(np.vstack([base[-1],q]),axis=0)
    command=np.ptp(np.vstack([base[-1],targets]),axis=0)
    signed=np.sum((q-base[-1])*(targets-base[-1]),axis=0)>0
    visible=(actual>=.03)&(actual>5*noise)&(command>=.03)&signed
    arms=[bool(np.any(visible[sl])) for sl in (slice(0,7),slice(7,14))]
    # Include every target-bearing sample, including lagging samples; no
    # cherry-picking only frames that already track well.
    tracking=np.max(np.abs(q-targets),axis=0)
    fresh=all(type(r.get('arm_ns')) is int and type(r.get('observed_ns')) is int
              and 0<=r['observed_ns']-r['arm_ns']<=100_000_000 for r in usable)
    distinct=len({r.get('arm_ns') for r in usable})>=20
    passed=bool(all(arms) and fresh and distinct
                and stop_confirmed is True and not error)
    return {'passed':passed,'arms_visible':arms,'excursion_rad':actual.tolist(),
            'comparison_basis':'full_ik_reference' if reference_present else 'bounded_command_legacy',
            'max_command_tracking_error_rad':np.max(np.abs(q-commands),axis=0).tolist(),
            'noise_rad':noise.tolist(),'max_tracking_error_rad':tracking.tolist(),
            'mean_tracking_error_rad':np.mean(np.abs(q-targets),axis=0).tolist(),
            'p95_tracking_error_rad':np.percentile(np.abs(q-targets),95,axis=0).tolist(),
            'tracking_error_policy':'report_only','tolerance_rad':None,'fresh_feedback':fresh,
            'stop_confirmed':stop_confirmed,'error':error}
