"""Tianyi operator-authorized recording/replay; no action is implicit on import."""
from __future__ import annotations
import argparse
import fcntl
import html
import json
import os
import queue
from pathlib import Path
import signal
import threading
import time
import urllib.request
import yaml
from .replay import analyze, digest, evaluate, evaluate_combined, execute, schedule, write_json, checked_segment
from .kinematics import TianyiIK, finite, MEASURED_LIMIT_TOLERANCE_RAD
import numpy as np

# Same MCP listener as every other ActuCore card. CLI reads the bundle config.
ACTUCORE_MCP_PORT = 15730


class _EvidenceWriter:
    """Ordered, bounded evidence I/O; never wait for disk in the input feeder."""
    def __init__(self, path, capacity=1024):
        self.file=path.open('x',buffering=1)
        self.queue=queue.Queue(maxsize=capacity)
        self.done=threading.Event();self.error=None
        self.thread=threading.Thread(target=self._run,daemon=True)
        self.thread.start()

    def __enter__(self):return self

    def __exit__(self,*args):
        # Caller must stop/release hardware before waiting for evidence flush.
        self.done.set()

    def write(self, text):
        if self.error is not None:raise ValueError('evidence_write_failed') from self.error
        if self.done.is_set():raise ValueError('evidence_writer_closed')
        try:self.queue.put_nowait(text)
        except queue.Full as exc:raise ValueError('evidence_backlog') from exc

    def _run(self):
        try:
            with self.file:
                while not (self.done.is_set() and self.queue.empty()):
                    try:text=self.queue.get(timeout=.02)
                    except queue.Empty:continue
                    self.file.write(text)
        except Exception as exc:self.error=exc

    def finish(self):
        self.done.set();self.thread.join(timeout=5)
        if self.thread.is_alive():raise ValueError('evidence_flush_timeout')
        if self.error is not None:raise ValueError('evidence_write_failed') from self.error


def call(port, tool, action, **args):
    headers={'Content-Type':'application/json'}
    if tool=='teleop' and action!='info':
        key=Path(os.environ.get('TELEOP_MANAGEMENT_KEY_FILE','/run/teleop-management.key')).read_text().strip()
        if len(key)<32:raise ValueError('management_key_missing')
        headers['X-Teleop-Management']=key
    req=urllib.request.Request(f'http://127.0.0.1:{port}/mcp',json.dumps({'jsonrpc':'2.0','id':1,
        'method':'tools/call','params':{'name':tool,'arguments':{'action':action,**args}}}).encode(),headers)
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=5) as r:result=json.load(r)
    if result.get('error'):raise ValueError('mcp_error:'+action)
    result=result['result'];value=json.loads(result['content'][0]['text'])
    if result.get('isError') or value.get('error'):raise ValueError(action+':'+str(value.get('code') or value.get('error')))
    return value


def idle():
    d=call(15707,'teleop_executor','info');v=call(ACTUCORE_MCP_PORT,'teleop','info')
    if d['state']!='idle' or d['ownership_held'] or d['output_active']:raise ValueError('driver_not_released')
    if v['authority_valid'] or v.get('mode')!='shadow':raise ValueError('card_not_isolated_shadow')
    if d.get('foreign_publishers') or d.get('hands_enabled') is not False or d.get('calibration_error'):raise ValueError('driver_preflight')
    return d


def record(cfg):
    idle()
    # Recreate only the card session if needed, always retaining Shadow isolation.
    call(ACTUCORE_MCP_PORT,'teleop','calibrate')
    status=call(ACTUCORE_MCP_PORT,'teleop','record_start')
    print('ARMED: waiting for both grips; then record 10 seconds. Hardware output disabled.',flush=True)
    try:
        call(ACTUCORE_MCP_PORT,'teleop','start')
        until=time.monotonic()+132
        while time.monotonic()<until:
            time.sleep(.5);status=call(ACTUCORE_MCP_PORT,'teleop','record_status')
            if status['state'] not in ('armed','recording'):break
    finally:
        status=call(ACTUCORE_MCP_PORT,'teleop','record_stop')
        call(ACTUCORE_MCP_PORT,'teleop','stop')
    print(json.dumps(status),flush=True)
    if not status.get('complete'):raise ValueError('recording_incomplete')
    return status['recording_id']


def journal_link(link, path):
    """Private recovery credential, separate from recording/evidence exports."""
    path=Path(path);original=link.call
    def save(value):
        temporary=path.with_suffix('.tmp')
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
        with os.fdopen(fd,'w') as f:
            json.dump(value,f);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
    def wrapped(action,deadline):
        save({'lease':link.lease,'management_request':link.management_request,
              'release_requested_ns':link.release_requested_ns})
        value=original(action,deadline)
        if action in ('claim','resume') and all(k in value for k in ('boot_id','session_id','secret')):
            save({'lease':value,'management_request':None,'release_requested_ns':0})
        return value
    link.call=wrapped


def recover(cfg, recording):
    v=call(ACTUCORE_MCP_PORT,'teleop','info')
    if v.get('authority_valid') or v.get('mode')!='shadow':raise ValueError('card_not_isolated_shadow')
    paths=sorted(recording.glob('execution-*/.lease.json'))
    if not paths:raise ValueError('replay_recovery_journal_missing')
    if len(paths)!=1:raise ValueError('multiple_unresolved_journals')
    path=paths[0];saved=json.loads(path.read_text())
    link,ex,context,thread=ros_link(cfg)
    try:
        link.lease=saved['lease'];link.management_request=saved['management_request']
        link.release_requested_ns=saved['release_requested_ns']
        if not link.lease and not link.management_request:raise ValueError('recovery_identity_missing')
        journal_link(link,path)
        end=time.monotonic()+2
        while time.monotonic()<end:
            if link.stop(time.monotonic()+.25):
                d=link.feedback()
                if d.get('ownership_held') or not d.get('stop_confirmed'):raise ValueError('recovery_stop_not_confirmed')
                path.unlink();print('REPLAY RELEASE CONFIRMED');return
            time.sleep(.02)
        raise ValueError('replay_release_unconfirmed')
    finally:link.close();ex.shutdown();context.shutdown();thread.join(timeout=2)


def ros_link(cfg):
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from .adapter import DriverLink
    context=rclpy.context.Context();context.init()
    ex=MultiThreadedExecutor(num_threads=2,context=context)
    link=DriverLink(cfg,ex);thread=threading.Thread(target=ex.spin,daemon=True);thread.start()
    end=time.monotonic()+5  # DDS discovery only; no lease or motion yet.
    while True:
        try:link.feedback();break
        except ValueError:
            if time.monotonic()>end:
                link.close();ex.shutdown();context.shutdown();thread.join(timeout=2);raise
            time.sleep(.02)
    return link,ex,context,thread


def report(directory):
    directory=Path(directory);reports=[]
    for path in sorted(directory.glob('round-*.jsonl')):
        rows=[json.loads(s) for s in path.read_text().splitlines()];result=evaluate(rows)
        write_json(path.with_suffix('.result.json'),result);reports.append({'file':path.name,**result})
    combined_result=json.loads((directory/'combined.result.json').read_text()) if (directory/'combined.result.json').exists() else None
    manifest=json.loads((directory/'run.json').read_text()) if (directory/'run.json').exists() else {}
    expected=manifest.get('execution_rounds',3)
    if type(expected) is not int or expected not in (1,3):raise ValueError('invalid_report_round_count')
    summary={'rounds':reports,'combined':combined_result,'execution_rounds':expected,
        'tracking_error_policy':'report_only',
        'passed':len(reports)==expected and all(r['passed'] for r in reports) and bool(combined_result and combined_result['passed'])}
    write_json(directory/'report.json',summary)
    charts=[]
    paths=sorted(directory.glob('round-*.jsonl'))
    if (directory/'combined.jsonl').exists():paths.append(directory/'combined.jsonl')
    for path in paths:
        rows=[json.loads(s) for s in path.read_text().splitlines()];rows=[r for r in rows if 'elapsed' in r]
        if not rows:continue
        combined=path.name=='combined.jsonl'
        reference='ik_reference_q' if combined and any('ik_reference_q' in r for r in rows) else 'target_q'
        label='完整IK目标' if reference=='ik_reference_q' else ('旧版短步目标，缺完整IK参考' if combined else '轨迹目标')
        duration=max(.001,rows[-1]['elapsed'])
        for j in range(14):
            values=[r[k][j] for r in rows for k in ('q',reference) if r.get(k) is not None]
            if not values:continue
            lo=min(values)-.01;hi=max(values)+.01
            lines=[]
            for key,color in [(reference,'#db8600'),('q','#138b58')]:
                segments=[];segment=[];previous=None
                for r in rows:
                    valid=r.get(key) is not None
                    if not valid or (previous is not None and r['elapsed']-previous>.1):
                        if segment:segments.append(segment);segment=[]
                    if valid:segment.append(f'{40+720*r["elapsed"]/duration:.2f},{180-150*(r[key][j]-lo)/(hi-lo):.2f}')
                    previous=r['elapsed']
                if segment:segments.append(segment)
                for points in segments:
                    lines.append(f'<polyline data-series="{key}" points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.5"/>')
            charts.append(f'<h3>{html.escape(path.name)} {label} / 实测 joint {j+1}: {lo:.3f}…{hi:.3f} rad</h3><svg viewBox="0 0 800 200">'+''.join(lines)+'</svg>')
    (directory/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Tianyi replay</title><h1>天轶回放：橙色目标 / 绿色实测</h1><p>缺失数据保留断线；组合使用完整IK目标。跟随误差只报告数值，不设通过门槛；passed表示执行与停止链路完成。</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'+''.join(charts))
    return reports


def run(cfg, recording, package, rounds=3):
    if rounds not in (1,3):raise ValueError('invalid_execution_rounds')
    profile=Path(cfg['calibration_path']);solver=TianyiIK(profile)
    if digest(profile)!=package['profile_sha256']:raise ValueError('trajectory_profile_changed')
    if digest(recording/'poses.jsonl')!=package['recording_sha256']:raise ValueError('recording_changed')
    # Both binaries and calibration must match the compilation, not just the old recording.
    for name,value in package['compiled_sources'].items():
        if digest(Path(__file__).with_name(name))!=value:raise ValueError('trajectory_code_changed')
    directory=recording/('execution-'+str(time.time_ns()));directory.mkdir()
    write_json(directory/'run.json',{'execution_rounds':rounds,'recording_sha256':digest(recording/'poses.jsonl'),
        'original_input_frames':len((recording/'poses.jsonl').read_text().splitlines()),
        'mode':'paired_trace' if rounds==1 else 'three_round_acceptance','started_ns':time.monotonic_ns()})
    lock=open(Path(cfg['capture']['state_file']).parent/'replay.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old_handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def interrupted(*args):raise InterruptedError('operator_cancelled')
    for s in old_handlers:signal.signal(s,interrupted)
    try:
        for index in range(rounds):
            d=idle();call(15707,'teleop_executor','start')
            link,ex,context,thread=ros_link(cfg)
            path=directory/f'round-{index+1}.jsonl'
            journal=directory/'.lease.json';journal_link(link,journal)
            evidence=None
            try:
                measured=link.feedback()['feedback']['q'];points=schedule(package,solver,measured)
                write_json(directory/f'schedule-{index+1}.json',points)
                # All geometric compilation precedes the limited hardware permit.
                arm=call(15707,'arm','info')
                if arm.get('state')!='ready':call(15707,'arm','start')
                prepared=call(15707,'teleop_executor','prepare_first_acceptance')
                if not prepared.get('first_acceptance_prepared'):raise ValueError('prepare_failed')
                if max(abs(a-b) for a,b in zip(measured,link.feedback()['feedback']['q']))>.02:raise ValueError('starting_pose_changed')
                with _EvidenceWriter(path) as f:
                    evidence=f
                    execute(link,points,solver,lambda row:f.write(json.dumps(row,allow_nan=False)+'\n'))
            finally:
                # Do not discard a lost/uncertain owner and pretend the service is safe.
                unresolved=bool(link.lease or link.management_request)
                link.close();ex.shutdown();context.shutdown();thread.join(timeout=2)
                if unresolved:raise ValueError('lease_unresolved_do_not_restart')
                if journal.exists():journal.unlink()
                if evidence:evidence.finish()
            results=report(directory)
            if not results[-1]['passed']:raise ValueError('execution_acceptance_failed:'+str(path))
            print('ROUND PASS',index+1,str(path),flush=True)
            time.sleep(1)
        # Combined stage follows the requested independent execution passes.
        combined(cfg,recording,directory)
    finally:
        report(directory)
        for s,h in old_handlers.items():signal.signal(s,h)
        lock.close()
    print('ACCEPTANCE PASS',str(directory),flush=True)


def fresh_baseline(link):
    """No authority: collect distinct fresh samples within one fixed second."""
    baseline=[];end=time.monotonic()+1.
    while len(baseline)<10 and time.monotonic()<end:
        try:fb=link.feedback()['feedback']
        except ValueError as exc:
            if str(exc) not in ('driver_feedback_missing','driver_feedback_stale_or_different_clock'):raise
            time.sleep(.02);continue
        if 0<=time.monotonic_ns()-fb['arm_ns']<=100_000_000 and (not baseline or fb['arm_ns']>baseline[-1]['arm_ns']):
            baseline.append({'q':fb['q'],'arm_ns':fb['arm_ns']})
        time.sleep(.02)
    if len(baseline)<5:raise ValueError('combined_baseline_missing')
    return baseline


def position_recorded_origin(cfg, recording, directory):
    """Bounded setup motion, with its own verified stop before the input replay."""
    package=json.loads((recording/'analysis/trajectory.json').read_text())
    if digest(recording/'poses.jsonl')!=package['recording_sha256']:raise ValueError('recording_changed')
    if digest(cfg['calibration_path'])!=package['profile_sha256']:raise ValueError('trajectory_profile_changed')
    rows=[json.loads(s) for s in (recording/'poses.jsonl').read_text().splitlines()]
    origin=next((r for r in rows if r['frame']['deadman'] and all(r['frame']['tracking'].values())),None)
    if origin is None:raise ValueError('recorded_origin_missing')
    observed=origin['driver']['feedback']
    if not 0<=origin.get('observed_ns',origin['received_ns'])-observed['arm_ns']<=100_000_000:
        raise ValueError('recorded_origin_stale')
    solver=TianyiIK(cfg['calibration_path']);velocity=min(1.,solver.velocity)
    lo=solver.model.lowerPositionLimit[solver.indices];hi=solver.model.upperPositionLimit[solver.indices]
    tolerance=min(MEASURED_LIMIT_TOLERANCE_RAD,velocity*.02)
    raw=finite(observed['q'],(14,))
    if np.any(raw<lo-tolerance) or np.any(raw>hi+tolerance):raise ValueError('recorded_origin_joint_limit')
    target=np.clip(raw,lo,hi)
    idle();link,ex,context,thread=ros_link(cfg)
    journal=directory/'.lease.json';journal_link(link,journal)
    result={'passed':False,'target_q':target.tolist(),'recorded_q':raw.tolist(),'hardware_motion_requested':False}
    try:
        measured=finite(fresh_baseline(link)[-1]['q'],(14,))
        if np.any(measured<lo-tolerance) or np.any(measured>hi+tolerance):raise ValueError('starting_joint_limit')
        initial=np.clip(measured,lo,hi)
        checked_segment(solver,initial,target,step=velocity*.02)
        if np.max(np.abs(measured-target))>.01:
            # Setup timing accommodates the measured slowest arm response (~0.16
            # rad/s); the execution velocity limit remains the configured 1 rad/s.
            duration=max(.02,float(np.max(np.abs(target-initial)))/min(velocity,.15))
            points=[{'t':0.,'q':initial.tolist(),'phase':'baseline'},
                    {'t':1.,'q':initial.tolist(),'phase':'baseline'},
                    {'t':1.+duration,'q':target.tolist(),'phase':'origin'},
                    {'t':4.+duration,'q':target.tolist(),'phase':'origin_plateau'}]
            if points[-1]['t']>40:raise ValueError('origin_positioning_budget')
            write_json(directory/'origin.schedule.json',points)
            call(15707,'teleop_executor','start')
            if call(15707,'arm','info').get('state')!='ready':call(15707,'arm','start')
            if not call(15707,'teleop_executor','prepare_first_acceptance').get('first_acceptance_prepared'):
                raise ValueError('prepare_failed')
            if np.max(np.abs(finite(link.feedback()['feedback']['q'],(14,))-measured))>.02:
                raise ValueError('starting_pose_changed')
            result['hardware_motion_requested']=True
            with (directory/'origin.jsonl').open('x',buffering=1) as f:
                execute(link,points,solver,lambda row:f.write(json.dumps(row,allow_nan=False)+'\n'))
        final=finite(fresh_baseline(link)[-1]['q'],(14,))
        result.update(final_q=final.tolist(),max_error_rad=float(np.max(np.abs(final-target))))
        if result['max_error_rad']>.02:raise ValueError('recorded_origin_not_reached')
        idle();result['passed']=True
    except BaseException as exc:
        result['error']=str(exc);raise
    finally:
        unresolved=bool(link.lease or link.management_request)
        result['released']=not unresolved
        write_json(directory/'origin.result.json',result)
        link.close();ex.shutdown();context.shutdown();thread.join(timeout=2)
        if not unresolved and journal.exists():journal.unlink()
        if unresolved:raise ValueError('lease_unresolved_do_not_restart')


def calibrate_fresh(adapter, path):
    """Zero-output calibration may wait for a fresh sample, never use old data."""
    end=time.monotonic()+1.
    while True:
        try:return adapter.calibrate(path)
        except ValueError as exc:
            if str(exc) not in ('driver_feedback_missing','driver_feedback_stale_or_different_clock'):
                raise
            if time.monotonic()>=end:raise
            time.sleep(.01)


def paced_latest_inputs(rows,start):
    """Preserve source wall timing, discarding superseded input rather than bursting it."""
    origin=rows[0]['received_ns'];index=0
    while index<len(rows):
        due=start+(rows[index]['received_ns']-origin)/1e9
        before=time.monotonic();requested=max(0.,due-before)
        if requested>0:time.sleep(requested)
        awake=time.monotonic();first=index
        if awake-start>40:raise ValueError('combined_deadline')
        while index+1<len(rows) and start+(rows[index+1]['received_ns']-origin)/1e9<=awake:index+=1
        due=start+(rows[index]['received_ns']-origin)/1e9
        timing={'input_lateness_ms':max(0.,(awake-due)*1000),
                'input_wait_requested_ms':requested*1000,'input_wait_wall_ms':(awake-before)*1000,
                'superseded_count':index-first,
                'superseded_first_sequence':rows[first]['frame']['sequence'] if index>first else None}
        if awake-due>.1:raise ValueError('combined_schedule_late_no_fresh_input')
        yield rows[index],timing
        index+=1


def combined(cfg,recording,directory):
    from .runtime import TeleopRuntime
    from .tianyi import TianyiIntentAdapter
    from .protocol import bind_rtc_frame_v1
    import copy
    idle();rows=[json.loads(s) for s in (recording/'poses.jsonl').read_text().splitlines()]
    if not rows or rows[-1]['received_ns']-rows[0]['received_ns']>35_000_000_000:raise ValueError('combined_recording_duration')
    position_recorded_origin(cfg,recording,directory)
    link,ex,context,thread=ros_link(cfg);runtime=None;error=None;released=False
    adapter=TianyiIntentAdapter(link,'live',cfg.get('position_scale',.5))
    journal=directory/'.lease.json';journal_link(link,journal)
    path=directory/'combined.jsonl';baseline=[];feedback_wait=None;max_feedback_gap_ms=0.;evidence=None;failure=None;ik_evidence=None
    latest_trace=[{}]
    try:
        ik_evidence=_EvidenceWriter(directory/'actucore.jsonl')
        def trace(row):
            latest_trace[0]=row  # Immutable replacement from the sole adapter writer.
            ik_evidence.write(json.dumps(row,allow_nan=False)+'\n')
        adapter.trace_sink=trace
        calibrate_fresh(adapter,cfg['calibration_path'])
        link.prepare_transport()
        baseline=fresh_baseline(link)
        write_json(directory/'combined.baseline.json',baseline)
        call(15707,'teleop_executor','prepare_first_acceptance')
        runtime=TeleopRuntime(mode='live',adapter=adapter,pose_timeout_ms=adapter.input_timeout_ms,
                             dispatch_io_timeout_ms=adapter.dispatch_io_timeout_ms,motion_interval_ms=20)
        runtime.prepare_local_session();binding,_=runtime.rtc_authority_snapshot()
        with _EvidenceWriter(path) as f:
            evidence=f
            start=time.monotonic();next_heartbeat=start
            for row,timing in paced_latest_inputs(rows,start):
                frame=copy.deepcopy(row['frame']);frame.pop('epoch',None)
                frame.update(schema_version=1,mode='live',client_monotonic_ns=time.monotonic_ns())
                mark=time.monotonic()
                if mark>=next_heartbeat:
                    runtime.heartbeat(binding,include_status=False)
                    next_heartbeat=time.monotonic()+.25
                timing['heartbeat_ms']=(time.monotonic()-mark)*1000
                mark=time.monotonic()
                runtime.submit_frame(bind_rtc_frame_v1(frame,authority=binding,expected_mode='live'),source='recorded_acceptance',include_status=False)
                timing['submit_ms']=(time.monotonic()-mark)*1000
                mark=time.monotonic()
                view=runtime.control_status()
                timing['status_ms']=(time.monotonic()-mark)*1000
                observed=latest_trace[0]
                out={'state':observed.get('output_state'),
                     'code':(observed.get('failure') or {}).get('code'),
                     'target_q':observed.get('ik_target_q') if observed.get('published') else None,
                     'ik_reference_q':observed.get('ik_reference_q') if observed.get('published') else None}
                mark=time.monotonic()
                try:
                    state=link.feedback();fb=state['feedback']
                    if not 0<=time.monotonic_ns()-fb['arm_ns']<=100_000_000:
                        raise ValueError('arm_feedback_stale')
                except ValueError as exc:
                    if str(exc) not in ('driver_feedback_missing','driver_feedback_stale_or_different_clock','arm_feedback_stale'):raise
                    now=time.monotonic()
                    if feedback_wait is None:feedback_wait=now
                    gap_ms=(now-feedback_wait)*1000;max_feedback_gap_ms=max(max_feedback_gap_ms,gap_ms)
                    f.write(json.dumps({'event':'feedback_unavailable','elapsed':now-start,'sequence':frame['sequence'],
                        'observed_ns':time.monotonic_ns(),'observer_timing_ms':timing,
                        'reason':str(exc),'target_q':None,'runtime_state':view['state'],'output_code':out.get('code')})+'\n')
                    if view['state']=='fault':raise ValueError('combined_dispatch_fault')
                    if gap_ms>=300:raise ValueError('combined_feedback_timeout')
                    continue
                timing['feedback_ms']=(time.monotonic()-mark)*1000
                if feedback_wait is not None:
                    max_feedback_gap_ms=max(max_feedback_gap_ms,(time.monotonic()-feedback_wait)*1000)
                feedback_wait=None
                f.write(json.dumps({'elapsed':time.monotonic()-start,'sequence':frame['sequence'],'deadman':frame['deadman'],
                    'q':fb['q'],'arm_ns':fb['arm_ns'],'observed_ns':time.monotonic_ns(),'target_q':out.get('target_q'),
                    'ik_reference_q':out.get('ik_reference_q'),'driver_sequence':state['applied_sequence'],
                    'state':state['state'],'reason':state.get('reason'),
                    'driver_session_id':state.get('session_id'),
                    'driver_continuations':state.get('diagnostics',{}).get('continuations'),
                    'continuation_ready':state.get('continuation_ready'),'dispatch':view['dispatch']['fault_code'],
                    'observer_timing_ms':timing,'runtime_state':view['state'],'runtime_reason':view.get('reason'),'output_state':out.get('state'),'output_code':out.get('code'),
                    'driver_command':state.get('diagnostics',{}).get('last_command'),
                    'adapter_diagnostics':{'last_failure':observed.get('failure'),
                        'last_apply':observed.get('last_apply'),'last_send':observed.get('publish')}})+'\n')
                if view['state']=='fault':raise ValueError('combined_dispatch_fault')
    except BaseException as exc:
        error=str(exc)
        if runtime:
            try:
                view=runtime.status()
                failure={'error':error,'state':view.get('state'),'reason':view.get('reason'),
                    'dispatch':{k:view['dispatch'].get(k) for k in ('state','fault_code','last_decision','last_io_stall','counters')},
                    'adapter':{k:view['dispatch'].get('adapter',{}).get('diagnostics',{}).get(k) for k in ('last_failure','last_apply','last_send')}}
            except Exception:pass  # Preserve the original failure even if diagnostics are unavailable.
    finally:
        if runtime:
            try:runtime.close()
            except Exception as exc:error=error or str(exc)
        end=time.monotonic()+2
        try:
            while time.monotonic()<end:
                if link.stop(time.monotonic()+.2):released=True;break
        except (ValueError,OSError) as exc:error=error or str(exc)
        link.close();ex.shutdown();context.shutdown();thread.join(timeout=2)
        if released and journal.exists():journal.unlink()
        if evidence:
            try:evidence.finish()
            except ValueError as exc:error=error or str(exc)
    if ik_evidence:
        try:ik_evidence.finish()
        except ValueError as exc:error=error or str(exc)
    if failure is not None:write_json(directory/'combined.failure.json',failure)
    data=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    result=evaluate_combined(data,baseline,stop_confirmed=released,error=error)
    result.update(error=error,stop_confirmed=released)
    result.update(feedback_gap_samples=sum(r.get('event')=='feedback_unavailable' for r in data),max_feedback_gap_ms=max_feedback_gap_ms)
    write_json(directory/'combined.result.json',result)
    if not result['passed']:raise ValueError('combined_acceptance_failed:'+str(result))



def main():
    global ACTUCORE_MCP_PORT
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path)
    p.add_argument('action',choices=['record','analyze','check','run','paired','report','recover']);p.add_argument('--recording')
    args=p.parse_args();bundle=yaml.safe_load(args.config.read_text())
    ACTUCORE_MCP_PORT=int(bundle.get('mcp_port',15730));cfg=bundle['plugins']['teleop']
    if cfg.get('robot_profile')!='tianyi2':raise ValueError('tianyi_only')
    root=Path(cfg['capture']['state_file']).parent/'recordings'
    if args.action=='record':record(cfg);return
    if args.action=='check':idle();print('CHECK PASS: isolated card, idle Driver; no hardware output');return
    if not args.recording or Path(args.recording).name!=args.recording or args.recording in ('.','..'):raise ValueError('recording_id_required')
    source=root/args.recording
    if source.resolve().parent!=root.resolve():raise ValueError('recording_path')
    if args.action=='recover':recover(cfg,source);return
    if args.action=='analyze':
        package=analyze(source,cfg['calibration_path'],source/'analysis')
        print('ANALYSIS PASS',len(package['waypoints']));return
    if args.action=='report':
        for d in source.glob('execution-*'):report(d)
        return
    package=json.loads((source/'analysis/trajectory.json').read_text());run(cfg,source,package,rounds=1 if args.action=='paired' else 3)

if __name__=='__main__':main()
