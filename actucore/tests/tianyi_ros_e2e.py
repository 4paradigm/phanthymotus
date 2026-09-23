"""Isolated ARM64 ROS integration. Synthetic feedback/output only; no vendor SDK.

Run only using deploy/test_tianyi_teleop_isolated.sh (network none).
Uses the real Driver bus subprocess, MCP dispatch, authenticated DriverLink,
Tianyi adapter and numerical IK, with a synthetic calibrated robot model.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

sys.path.insert(0, os.environ['TIANYI_DRIVER_SOURCE'])
sys.path.insert(0, os.environ.get('ACTUCORE_PLUGIN_ROOT', str(Path(__file__).parents[1]/'plugins')))
from motion_stream import MotionGate
from teleop_executor import TeleopExecutor
from teleop.adapter import DriverLink
from teleop.tianyi import TianyiIntentAdapter
from teleop.kinematics import TianyiIK
from teleop.dispatch import StopRequest
from teleop.protocol import bind_rtc_frame_v1
from teleop.runtime import TeleopRuntime
from tianyi_fixture import synthetic_profile, intent
import rclpy
from rclpy.executors import SingleThreadedExecutor


def await_true(predicate, timeout=3):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        try:
            if predicate():return
        except ValueError:
            pass
        time.sleep(.01)
    raise AssertionError('condition_timeout')


def main():
    rclpy.init(domain_id=42)
    ros=SingleThreadedExecutor()
    spin=threading.Thread(target=ros.spin, daemon=True)
    spin.start()
    q=[0.]*14; writes=[]; power_age_ns=[0];claim_delay=[0.];resume_delay=[0.];lost_replies=[]
    def snapshot():
        result={k:time.monotonic_ns() for k in ('arm_ns','power_ns','hand_ns','fixed_ns')}
        result['power_ns'] -= power_age_ns[0]
        result.update(q=list(q),dq=[0.]*14,power_on=True,estop=False,fault=False,fixed_body=True)
        return result
    def emit(target,hands):
        writes.append((list(target),hands))
        q[:]=[a+.5*(b-a) for a,b in zip(q,target)]
    # Use the real constructor so new lifecycle resources cannot silently be
    # omitted from this integration fixture. No ROS subscription starts here.
    driver=TeleopExecutor({}, 'isolated_tianyi', None,
        SimpleNamespace(_pos_publisher=True),
        SimpleNamespace(_left_pub=True,_right_pub=True), [])
    driver._output_ready=True;driver._foreign_publishers=[]
    driver.plugins=[];driver.profile_error=None
    driver.profile={'hands_enabled':True}
    driver.subscribe_feedback=lambda:None  # Synthetic plant, NEVER vendor topics.
    driver.foreign_publishers=lambda:[]
    driver.gate=MotionGate(snapshot,emit,[(-2,2)]*14,live_enabled=True,acceptance_check=lambda:True)
    assert driver.dispatch('trace_start',{})['enabled']
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            rpc=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            params=rpc['params'];args=params['arguments']
            assert self.client_address[0]=='127.0.0.1' and params['name']=='teleop_executor'
            if args['action']=='claim':time.sleep(claim_delay[0])
            if args['action']=='resume':time.sleep(resume_delay[0])
            value=driver.dispatch(args['action'],args)
            if lost_replies and args['action']==lost_replies[0]:
                lost_replies.pop(0);self.close_connection=True;return
            result={'content':[{'type':'text','text':json.dumps(value)}]}
            if value.get('error'):result['isError']=True
            body=json.dumps({'jsonrpc':'2.0','id':rpc['id'],'result':result}).encode()
            self.send_response(200);self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
    http=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    http_thread=threading.Thread(target=http.serve_forever,daemon=True);http_thread.start()
    link=None;adapter=None
    try:
        with tempfile.TemporaryDirectory() as directory:
            profile=synthetic_profile(Path(directory))
            solver=TianyiIK(profile)
            solver.self_test(q)
            driver.profile_sha256=solver.profile_sha256
            driver.start()
            link=DriverLink({'namespace':driver.ns,'driver_mcp_url':f'http://127.0.0.1:{http.server_port}/mcp'},ros)
            await_true(lambda: bool(link.feedback()))
            link.management_retry=True;power_age_ns[0]=200_000_000
            try:link.claim(time.monotonic()+.15)
            except ValueError as exc:assert str(exc)=='driver_management_pending'
            else:raise AssertionError('stale_power_claim_granted')
            assert link.stop(time.monotonic()+.5)
            assert not driver.gate.session_id and not writes and not driver.gate.status()['stop_confirmed']
            power_age_ns[0]=0
            # Exercise the real dispatcher deadline on initial acquisition.
            # Delays are injected, not claimed to reproduce field phase timing.
            adapter=TianyiIntentAdapter(link,'live');adapter.calibrate(profile)
            runtime=TeleopRuntime(mode='live',adapter=adapter,auto_watchdog=False,pose_timeout_ms=100)
            try:
                runtime.prepare_local_session();binding,_=runtime.rtc_authority_snapshot()
                def submit(seq,held,clutch,up=0.):
                    f=intent(seq).frame
                    f.update(schema_version=1,sequence=seq,client_monotonic_ns=seq+1,
                             mode='live',deadman=held,clutch_sequence=clutch,
                             tracking=dict(head=True,left_controller=True,right_controller=True))
                    for side in ('left','right'):
                        f['controllers'][side]={'axes':[0.,0.,0.,0.],'buttons':[0.,float(held)]}
                        f[side+'_controller']['position'][1]+=up
                    runtime.submit_frame(bind_rtc_frame_v1(f,authority=binding,expected_mode='live'),source='isolated')
                claim_delay[0]=.060
                solve=adapter.solver.solve;collision=[False]
                def delayed_solve(*args,**kwargs):
                    if collision[0]:collision[0]=False;raise ValueError('torso_collision')
                    time.sleep(.045)
                    return solve(*args,**kwargs)
                adapter.solver.solve=delayed_solve
                submit(0,False,0);submit(1,True,1)
                assert runtime._dispatcher.wait_dispatched(1),runtime.status()['dispatch']
                assert not writes and not adapter.ik_times
                startup=adapter.last_apply
                assert startup['elapsed_ms']<100 and startup['remaining_ms']>0
                submit(2,True,1,up=.002)
                assert runtime._dispatcher.wait_dispatched(2),runtime.status()['dispatch']
                await_true(lambda:driver.gate.applied_seq>=1)
                assert runtime.status()['dispatch']['fault_code'] is None
                # Reproduce the field path, including a lost successful resume
                # reply. Keep the same grips and verify every dispatch budget.
                collision[0]=True;submit(3,True,1,up=.002)
                await_true(lambda:runtime.status()['dispatch']['state']=='safe_reclutch_required'
                           and runtime.status()['dispatch']['stop_acknowledged'])
                before=len(writes);submit(4,True,1,up=.002)
                assert runtime._dispatcher.wait_dispatched(4)
                assert adapter.output['state']=='recovery_validated' and driver.gate.state=='hold'
                resume_delay[0]=.065;lost_replies.append('resume')
                submit(5,True,1,up=.002);assert runtime._dispatcher.wait_dispatched(5)
                assert adapter.output['state']=='waiting_driver_management' and link.management_request
                owner=driver.gate.session_id
                submit(6,True,1,up=.002);assert runtime._dispatcher.wait_dispatched(6)
                assert link.lease['session_id']==owner and driver.gate.session_id==owner
                assert not link.management_request and len(writes)==before
                recovery=adapter.last_apply
                submit(7,True,1,up=.003);assert runtime._dispatcher.wait_dispatched(7)
                await_true(lambda:driver.gate.applied_seq>=1)
                assert runtime.status()['dispatch']['fault_code'] is None
            finally:
                claim_delay[0]=resume_delay[0]=0.;runtime.close();assert adapter.close().ok
            # Start the existing continuation sequence with a fresh adapter.
            adapter=TianyiIntentAdapter(link,'live');adapter.calibrate(profile)
            assert adapter.apply(intent()).ok
            for seq in range(2,12):
                frame=intent(seq)
                frame.frame['left_controller']['position'][1]+=.003
                frame.frame['right_controller']['position'][1]+=.002
                assert adapter.apply(frame).ok,adapter.output
                time.sleep(.025)
            await_true(lambda:driver.gate.applied_seq>=5)
            assert writes and max(abs(x) for x in q)>1e-6
            # Actual ROS/DDS, expired target, confirmed hold, same held clutch.
            session=driver.gate.session_id
            await_true(lambda:driver.gate.status()['continuation_ready'], timeout=.25)
            await_true(lambda:link.feedback().get('continuation_ready'), timeout=.1)
            frame=intent(12)
            frame.frame['left_controller']['position'][1]+=.002
            assert adapter.apply(frame).ok
            await_true(lambda:driver.gate.applied_seq==link.seq)
            assert driver.gate.session_id==session and driver.gate.diagnostics['continuations']==1
            # A 200 ms old power sample pauses output without a hard fault.
            power_age_ns[0]=200_000_000
            await_true(lambda:driver.gate.state=='hold' and driver.gate.reason=='power_ns_stale', timeout=.15)
            assert driver.gate.diagnostics['first_fault'] is None
            power_age_ns[0]=0
            await_true(lambda:driver.gate.status()['continuation_ready'], timeout=.15)
            await_true(lambda:link.feedback().get('continuation_ready'), timeout=.1)
            frame=intent(13)
            frame.frame['right_controller']['position'][1]+=.002
            assert adapter.apply(frame).ok
            await_true(lambda:driver.gate.applied_seq==link.seq)
            assert driver.gate.session_id==session and driver.gate.diagnostics['continuations']==2
            now=time.monotonic()
            assert adapter.safe_stop(StopRequest(2,'deadman_released',now,now+.3)).ok
            assert driver.gate.status()['hold_confirmed']
            before=len(writes);old_session=driver.gate.session_id
            assert adapter.apply(intent(20,clutch=2,generation=2)).ok
            assert driver.gate.session_id!=old_session and len(writes)==before
            frame=intent(21,clutch=2,generation=2)
            frame.frame['left_controller']['position'][1]-=.002
            assert adapter.apply(frame).ok
            await_true(lambda:driver.gate.applied_seq>=1)
            # Real-time command silence must hold, not replay the last input.
            await_true(lambda:driver.gate.status()['hold_confirmed'])
            assert driver.gate.reason=='command_timeout'
            assert adapter.close().ok
            assert not driver.gate.session_id
            # Independent trajectory player over the same actual ROS/MCP wire.
            from teleop.replay import execute
            claim_delay[0]=resume_delay[0]=0
            initial=list(q);target=[x+.04 for x in initial]
            points=[{'t':0.,'q':initial,'phase':'baseline'},
                    {'t':.3,'q':initial,'phase':'baseline'},
                    {'t':1.3,'q':target,'phase':'both'},
                    {'t':2.3,'q':target,'phase':'both_plateau'}]
            replay_rows=[]
            execute(link,points,solver,replay_rows.append)
            assert replay_rows[-1]['stop_confirmed'] and replay_rows[-1]['paused_and_resumed']
            assert max(abs(a-b) for a,b in zip(initial,q))>.03
            assert not driver.gate.session_id
            # New recoverable-hold path over REAL HTTP serialization and DDS.
            # This catches missing lease fields that a direct-call mock misses.
            adapter=TianyiIntentAdapter(link,'live');adapter.calibrate(profile)
            assert adapter.apply(intent()).ok
            original_solve=adapter.solver.solve
            owner=driver.gate.session_id
            def injected(*args,**kwargs):raise ValueError('ik_timeout')
            adapter.solver.solve=injected
            assert not adapter.apply(intent(2)).ok
            now=time.monotonic()
            assert adapter.safe_stop(StopRequest(2,'ik_timeout',now,now+1.)).ok,adapter.output
            time.sleep(.35)  # Longer than transport-timeout continuation window.
            await_true(lambda:link.feedback().get('continuation_ready'))
            adapter.solver.solve=original_solve
            resumed=intent(3,generation=2)
            resumed.frame['left_controller']['position'][1]+=.002
            assert adapter.apply(resumed).ok,adapter.output
            assert adapter.output['state']=='submitted',adapter.output
            await_true(lambda:driver.gate.applied_seq>=1)
            assert driver.gate.session_id==owner
            assert adapter.close().ok and not driver.gate.session_id
            trace=driver.info()['trace']
            assert trace['event_count']>0 and len(json.dumps(driver.info()))<65536
            assert any(e['event']=='command_decision' for e in trace['events'])
            assert not driver.dispatch('trace_stop',{})['enabled']
            print(json.dumps({'result':'PASS','architecture':os.uname().machine,
                              'synthetic_vendor_writes':len(writes),'hardware_writes':0,'replay_rows':len(replay_rows),
                              'stop_confirmed':driver.gate.status()['stop_confirmed'],
                              'continuation_checks': ['target_expiry', '200ms_power_feedback'],
                              'startup_dispatch':startup,
                              'collision_recovery_with_lost_resume_reply':recovery,
                              'transport_prepare_ms':link.transport_prepare_ms,
                              'transport':'real ROS2 domain42 + bus subprocess + loopback MCP'}))
    finally:
        if adapter:adapter.close()
        if link:link.close()
        driver.stop()
        http.shutdown();http.server_close();http_thread.join(1)
        ros.shutdown();spin.join(1);rclpy.shutdown()


if __name__=='__main__':main()
