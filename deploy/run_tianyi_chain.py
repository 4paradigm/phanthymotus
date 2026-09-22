"""Explicit authorized paired replay on robot. Default is preflight only.

Uses existing containers; never changes Compose, images or Agent Core.
Evidence contains selected non-secret files only, not the private lease journal.
"""
import argparse
import json
import math
from pathlib import Path
import ssl
import subprocess
import time
import urllib.request
from summarize_tianyi_chain import summarize

AC='embodied-actucore-teleop'
DRIVER='embodied-x-humanoid-tianyi2.0'
RECORDING='20260921T161243Z-b811f2d3'
ROOT='/var/lib/motus-teleop/recordings/'+RECORDING

def command(*args):
    return subprocess.check_output(args,text=True,timeout=20).strip()

def rpc(action, tool='teleop_executor', port=15707):
    data={'jsonrpc':'2.0','id':1,'method':'tools/call',
          'params':{'name':tool,'arguments':{'action':action}}}
    request=urllib.request.Request(f'http://127.0.0.1:{port}/mcp',json.dumps(data).encode(),{'Content-Type':'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=5) as response:result=json.load(response)
    if result.get('error'):raise RuntimeError('driver_rpc_error')
    value=json.loads(result['result']['content'][0]['text'])
    if result['result'].get('isError') or value.get('error'):raise RuntimeError(action+':'+str(value.get('code')))
    return value


def core_status(path):
    if path not in ('/canvas/edit-status?session_id=teleop-replay','/config/project-running'):
        raise ValueError('core_read_only')
    key=next(line.split('=',1)[1].strip() for line in Path('/opt/phanthy-motus/.env').read_text().splitlines()
             if line.startswith('ACCESS_TOKEN='))
    request=urllib.request.Request('https://127.0.0.1:15678/api'+path,headers={'Authorization':'Bearer '+key})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))
    with opener.open(request,timeout=5) as response:return json.load(response)


def runtime_container(name):
    # Deliberately exclude Compose/baseline, environment and credentials.
    return json.loads(command('docker','inspect','--format',
        '{"id":{{json .Id}},"image":{{json .Image}},"state":{{json .State.Status}},'
        '"health":{{json (index .State "Health")}},"network":{{json .HostConfig.NetworkMode}}}',name))


def replay_preflight():
    """Read current execution conditions, independent of deployment drift review."""
    edit=core_status('/canvas/edit-status?session_id=teleop-replay')
    if edit.get('code')!=200 or 'editor' not in edit or edit['editor'] is not None:
        raise ValueError('canvas_occupied_or_unknown')
    if core_status('/config/project-running').get('running') is not False:
        raise ValueError('canvas_running_or_unknown')
    containers={name:runtime_container(name) for name in (AC,DRIVER)}
    for name,value in containers.items():
        health=value.get('health') or {}
        if value.get('state')!='running' or value.get('network')!='host' or health.get('Status','healthy')!='healthy':
            raise ValueError('container_not_ready:'+name)
    d=rpc('info');v=rpc('info','teleop',15740)
    if d.get('state')!='idle' or d.get('ownership_held') is not False or d.get('output_active') is not False:
        raise ValueError('driver_not_idle')
    if d.get('foreign_publishers') or d.get('hands_enabled') is not False or d.get('calibration_error') or d.get('feedback_executor_error'):
        raise ValueError('driver_conflict_or_error')
    if v.get('authority_valid') is not False or v.get('mode')!='shadow' or v.get('state') not in ('idle','fault'):
        raise ValueError('card_not_isolated')
    dispatch=v.get('dispatch') or {}
    if any(dispatch.get(k) for k in ('io_inflight','mailbox_depth','stop_queue_depth')):
        raise ValueError('dispatch_pending')
    f=d.get('feedback') or {};now=d.get('monotonic_ns',0)
    if not now or any(not 0<=now-f.get(k,0)<=100_000_000 for k in ('arm_ns','power_ns','fixed_ns')):
        raise ValueError('feedback_stale')
    if f.get('power_on') is not True or f.get('estop') is not False or f.get('fault') is not False:
        raise ValueError('robot_not_ready')
    q=f.get('q',[]);dq=f.get('dq',[])
    if len(q)!=14 or len(dq)!=14 or any(type(x) not in (int,float) or not math.isfinite(x) for x in q+dq):
        raise ValueError('joint_feedback_invalid')
    if any(abs(x)>.02 for x in dq):raise ValueError('arms_not_stationary')
    if rpc('info','arm').get('state') not in ('idle','ready') or rpc('info','servo').get('state')!='idle':
        raise ValueError('legacy_busy')
    config=json.loads(command('docker','exec',AC,'python3','-c',
        'import json,yaml; c=yaml.safe_load(open("/work/config.yaml"))["plugins"]["teleop"]; '
        'print(json.dumps({"mode":c.get("mode")}))'))
    if config.get('mode')!='shadow':raise ValueError('startup_config_not_shadow')
    return {name:{k:value[k] for k in ('id','image')} for name,value in containers.items()}

def executions():
    code='from pathlib import Path; import json; print(json.dumps(sorted(p.name for p in Path('+repr(ROOT)+').glob("execution-*"))))'
    return set(json.loads(command('docker','exec',AC,'python3','-c',code)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--continuity',action='store_true',help='Supplemental dwell/failure/regrip inputs; requires --execute for motion')
    args=parser.parse_args()
    args.output=args.output.resolve()
    args.output.relative_to('/data/hanzebei/pico-tianyi')
    current=replay_preflight()
    images=[current[n]['image'] for n in (AC,DRIVER)]
    print('REPLAY CHECK PASS: current occupancy, containers, authority and feedback; no deployment baseline required',flush=True)
    print(command('docker','exec','-w','/work',AC,'python3','-m','plugins.teleop.acceptance','--config','/work/config.yaml','check'),flush=True)
    if not args.execute:return
    args.output.mkdir(exist_ok=False)
    initial=executions();observers=[];files=[];error=None;trace_started=False
    (args.output/'run-host.json').write_text(json.dumps({'images':images,'containers':current,'recording':RECORDING,
        'started_ns':time.monotonic_ns(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()})+'\n')
    try:
        for domain,name in ((0,'ros-body'),(42,'ros-driver')):
            output=(args.output/(name+'.jsonl')).open('x');err=(args.output/(name+'.stderr')).open('x');files.extend([output,err])
            environment=['ROS_DOMAIN_ID='+str(domain),'ROS_LOCALHOST_ONLY='+('1' if domain==42 else '0'),
                'RMW_IMPLEMENTATION=rmw_fastrtps_cpp','FASTRTPS_DEFAULT_PROFILES_FILE='+('/work/dds-local.xml' if domain==42 else '')]
            argv=['docker','exec','-i']
            for value in environment:argv+=['-e',value]
            argv += [DRIVER,'bash','-c','source /opt/ros/humble/setup.bash && source /tianyi_ws/install/setup.bash && exec python3 - "$@"',
                     'observer','--domain',str(domain),'--namespace','nvidia_desktop','--seconds','180']
            with Path(__file__).with_name('record_tianyi_chain.py').open() as script:
                process=subprocess.Popen(argv,stdin=script,stdout=output,stderr=err)
            observers.append(process)
        deadline=time.monotonic()+15
        while True:
            if any(p.poll() is not None for p in observers):raise RuntimeError('observer_exited_before_motion')
            text=[(args.output/(n+'.jsonl')).read_text() for n in ('ros-body','ros-driver')]
            if 'arm_status' in text[0] and 'driver_feedback' in text[1]:break
            if time.monotonic()>deadline:raise RuntimeError('observer_discovery_failed_no_motion')
            time.sleep(.2)
        if replay_preflight()!=current:raise RuntimeError('teleop_containers_changed_before_replay')
        trace=rpc('trace_start');trace_started=True
        (args.output/'trace-start.json').write_text(json.dumps(trace)+'\n')
        with (args.output/'acceptance.log').open('x') as log:
            if args.continuity:
                with Path(__file__).with_name('tianyi_continuity_scenario.py').open() as scenario:
                    result=subprocess.run(['docker','exec','-i','-w','/work',AC,'bash','-c',
                        'source /opt/ros/humble/setup.bash && exec python3 -'],stdin=scenario,stdout=log,stderr=subprocess.STDOUT)
            else:
                result=subprocess.run(['docker','exec','-w','/work',AC,'bash','-c',
                    'source /opt/ros/humble/setup.bash && exec python3 -m plugins.teleop.acceptance --config /work/config.yaml paired --recording '+RECORDING],
                    stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('paired_replay_failed_see_acceptance_log')
    except BaseException as exc:
        error=str(exc)
    finally:
        if trace_started:
            try:
                final=rpc('trace_stop')
                (args.output/'trace-stop.json').write_text(json.dumps(final)+'\n')
            except Exception as exc:error=error or 'trace_stop:'+str(exc)
        # Observers have their own 180-second lifetime, no business process is killed.
        for process in observers:
            try:
                if process.wait(timeout=185):error=error or 'observer_failed'
            except subprocess.TimeoutExpired:error=error or 'observer_exit_timeout'
        for file in files:file.close()
        new=executions()-initial
        if len(new)==1:
            source=ROOT+'/'+next(iter(new))
            code='from pathlib import Path; import json; print(json.dumps([p.name for p in Path('+repr(source)+').iterdir() if p.is_file() and not p.name.startswith(".") and p.suffix in (".json",".jsonl",".html")]))'
            for name in json.loads(command('docker','exec',AC,'python3','-c',code)):
                command('docker','cp',AC+':'+source+'/'+name,str(args.output/name))
        else:error=error or 'execution_directory_not_unique'
        (args.output/'host-result.json').write_text(json.dumps({'error':error,'execution_directories':sorted(new)})+'\n')
        summarize(args.output)
    print('CHAIN EVIDENCE: '+str(args.output),flush=True)
    if error:raise RuntimeError(error)

if __name__=='__main__':main()
