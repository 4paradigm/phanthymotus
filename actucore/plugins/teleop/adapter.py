"""ActuCore motion intent adapter. Hardware access remains in the Driver."""
from __future__ import annotations
import hashlib
import hmac
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from .dispatch import AdapterAck
from .kinematics import RelativeMapping, transform


def validate_driver_endpoint(cfg):
    """Validate editable connection fields before saving or allocating ROS nodes."""
    if 'namespace' in cfg and not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}',cfg['namespace']):
        raise ValueError('invalid_namespace')
    if 'driver_mcp_url' not in cfg:return
    try:
        url=urllib.parse.urlsplit(cfg['driver_mcp_url'])
        valid=(url.scheme=='http' and not url.username and not url.password
               and url.path=='/mcp' and not url.query and not url.fragment
               and ipaddress.ip_address(url.hostname).is_loopback
               and (url.port is None or 0<url.port<=65535))
    except (ValueError,TypeError):valid=False
    if not valid:raise ValueError('driver_mcp_must_be_loopback')


class DriverLink:
    def __init__(self, cfg, executor):
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        from std_msgs.msg import String
        if os.environ.get('ROS_DOMAIN_ID')!='42':raise ValueError('teleop_requires_domain_42')
        profile=Path(os.environ.get('FASTRTPS_DEFAULT_PROFILES_FILE',''))
        bundled=Path('/deploy/dds-local.xml')
        if not profile.is_file() or not bundled.is_file() or profile.read_bytes()!=bundled.read_bytes():
            raise ValueError('local_dds_profile_mismatch')
        validate_driver_endpoint(cfg)
        self.url=cfg['driver_mcp_url'];self.executor=executor
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.condition=threading.Condition();self.latest=None
        self.lease=None;self.seq=0;self.publisher=None
        self.pending=deque();self.lease_started_ns=0;self.release_requested_ns=0;self.execution_progress_ns=0
        self.last_send = None
        self.last_feedback_received_ns = None
        self.last_call = None
        self.transport_prepare_ms = None
        self.management_retry = False
        self.management_request = None
        self.node=Node('actucore_teleop',context=executor.context)
        executor.add_node(self.node)
        ns=cfg['namespace']
        self.topic=f'/{ns}/motion/teleop'
        self.qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,
                           history=HistoryPolicy.KEEP_LAST,durability=DurabilityPolicy.VOLATILE)
        self.node.create_subscription(String,self.topic+'/feedback',self._feedback,self.qos)
        self.preview=self.node.create_publisher(String,f'/{ns}/teleop/status',self.qos)

    def _feedback(self,msg):
        try:
            value=json.loads(msg.data)
            if not isinstance(value,dict):return
            with self.condition:
                self.latest=value;self.last_feedback_received_ns=time.monotonic_ns();self.condition.notify_all()
        except (ValueError,TypeError):pass

    def feedback(self):
        with self.condition:
            if self.latest is None:raise ValueError('driver_feedback_missing')
            value=dict(self.latest)
        age=time.monotonic_ns()-value.get('monotonic_ns',0)
        if not 0<=age<=100_000_000:raise ValueError('driver_feedback_stale_or_different_clock')
        return value

    def reconcile_release(self):
        operation = getattr(self, 'management_request', None)
        if not self.lease and not operation:return True
        requested=getattr(self,'release_requested_ns',0)
        if not requested:return False
        if operation and not operation.get('cancel_acknowledged'):
            requested=max(requested,operation['until_ns'])
        try:value=self.feedback()
        except ValueError:return False
        boot_id=self.lease['boot_id'] if self.lease else operation['boot_id']
        if (value.get('boot_id')==boot_id and value['monotonic_ns']>=requested
                and (not operation or value.get('state')=='idle' and value.get('output_active') is False)
                and value.get('ownership_held') is False
                and (value.get('stop_confirmed') is True or operation and operation.get('cancelled_without_output') is True)):
            self.lease=None;self.management_request=None;self.release_requested_ns=0
            return True
        return False

    def feedback_after(self,stamp_ns,deadline):
        # MCP replies can precede the next DDS feedback frame. Old snapshots
        # neither prove lease loss nor acknowledge a new command or stop.
        with self.condition:
            while time.monotonic()<deadline:
                state=self.feedback()
                if state['monotonic_ns']>=stamp_ns:return state
                self.condition.wait(min(.01,max(0,deadline-time.monotonic())))
        raise ValueError('driver_feedback_ack_timeout')

    def call(self,action,deadline):
        started=time.monotonic();ok=False
        try:
            args={'action':action}
            if action in ('pause','release','resume','recoverable_hold') and self.lease:
                args.update(session_id=self.lease['session_id'],secret=self.lease['secret'])
            operation=getattr(self,'management_request',None)
            if operation and action in (operation['action'],'release'):
                args.update(request_id=operation['id'],request_valid_until_ns=operation['until_ns'])
                args.pop('session_id',None);args.pop('secret',None)
                args.update(operation['credentials'])
            body={'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':'teleop_executor','arguments':args}}
            timeout=min(0.08,deadline-time.monotonic())
            if timeout<=0:raise ValueError('driver_call_deadline')
            request=urllib.request.Request(self.url,json.dumps(body).encode(),{'Content-Type':'application/json'})
            with self.opener.open(request,timeout=timeout) as response:reply=json.load(response)
            if reply.get('error'):raise ValueError('driver_mcp_error')
            result=reply['result']
            value=json.loads(result['content'][0]['text'])
            if result.get('isError') or value.get('error'):raise ValueError(value.get('code') or value.get('error') or 'driver_refused')
            ok=True
            return value
        finally:
            finished=time.monotonic()
            self.last_call={'action':action,'ok':ok,'elapsed_ms':(finished-started)*1000,
                            'remaining_ms':(deadline-finished)*1000}

    def prepare_transport(self):
        """Warm local DDS only: no lease, vendor publisher or motion message."""
        if self.publisher is not None:return
        from std_msgs.msg import String
        started=time.monotonic()
        self.publisher=self.node.create_publisher(String,self.topic+'/command',self.qos)
        self.transport_prepare_ms=(time.monotonic()-started)*1000

    def claim(self,deadline):
        if self.lease:return
        self.lease=self._manage('claim',deadline);self.seq=0
        self.pending.clear();self.lease_started_ns=time.monotonic_ns();self.release_requested_ns=0;self.execution_progress_ns=0
        if not all(k in self.lease for k in ('boot_id','session_id','secret')):
            raise ValueError('driver_invalid_lease')
        self.prepare_transport()

    def resume(self,deadline):
        self.lease=self._manage('resume',deadline);self.seq=0
        self.pending.clear();self.lease_started_ns=time.monotonic_ns();self.release_requested_ns=0;self.execution_progress_ns=0

    def _manage(self, action, deadline):
        if not getattr(self,'management_retry',False):return self.call(action,deadline)
        operation=getattr(self,'management_request',None)
        if operation is None:
            operation={'id':secrets.token_hex(16),'action':action,
                'until_ns':time.monotonic_ns()+300_000_000,
                'boot_id':self.feedback()['boot_id'],
                'credentials':({k:self.lease[k] for k in ('session_id','secret')} if self.lease else {})}
            self.management_request=operation
        if operation['action']!=action:raise ValueError('driver_management_conflict')
        if deadline-time.monotonic()<=.005:
            raise ValueError('driver_management_pending')
        if operation.get('cancel_requested') or time.monotonic_ns()>=operation['until_ns']:
            operation['cancel_requested']=True
            try:stopped=self.stop(deadline)
            except (OSError,ValueError) as exc:
                if isinstance(exc,ValueError) and str(exc) not in ('driver_feedback_ack_timeout',
                        'driver_feedback_missing','driver_feedback_stale_or_different_clock'):
                    raise
                stopped=False
            if stopped:raise ValueError('driver_lease_rebased')
            raise ValueError('driver_management_pending')
        try:
            result=self.call(action,deadline)
        except OSError:
            raise ValueError('driver_management_pending') from None
        except ValueError as exc:
            if str(exc) in ('arm_ns_stale','power_ns_stale','fixed_ns_stale','hand_ns_stale'):
                # Keep the same bounded transaction. Stale safety feedback never
                # grants ownership; a fresh sample may allow this request later.
                raise ValueError('driver_management_pending') from None
            if str(exc)=='hold_not_resumable' and action=='resume':
                # A hold acknowledgement can be revoked by newer settling data
                # before this RPC. Retry only the same owned recoverable HOLD;
                # the Driver still decides whether stopping is confirmed.
                state=self.feedback()
                allowed={'operator_pause','command_timeout','command_expired','ik_recoverable',
                         'arm_ns_stale','power_ns_stale','fixed_ns_stale','hand_ns_stale'}
                if (self.lease and state.get('state')=='hold'
                        and state.get('ownership_held') is True
                        and state.get('reason') in allowed
                        and all(state.get(k)==self.lease.get(k) for k in ('boot_id','session_id'))):
                    raise ValueError('driver_management_pending') from None
            if str(exc)=='robot_not_stopped':
                # The Driver may briefly re-observe residual settling after its
                # hold acknowledgement. Keep the SAME bounded management request;
                # do not turn a retryable refusal into a new grip requirement.
                raise ValueError('driver_management_pending') from None
            if str(exc) in ('management_request_expired','management_lease_released','management_cancelled'):
                operation['cancel_requested']=True
                raise ValueError('driver_management_pending') from None
            raise
        if not all(k in result for k in ('boot_id','session_id','secret')):
            raise ValueError('driver_invalid_lease')
        self.management_request=None
        return result

    def recoverable_hold(self,deadline):
        return self.pause(deadline, action="recoverable_hold")

    def pause(self,deadline, *, action="pause"):
        if getattr(self,'management_request',None):return self.stop(deadline)
        if not self.lease:return True
        stamp=time.monotonic_ns()
        try:self.call(action,deadline)
        except OSError:
            # Delivery is uncertain after a transport timeout. Only matching,
            # fresh post-request DDS evidence can still confirm the hold.
            pass
        with self.condition:
            while time.monotonic()<deadline:
                state=self.feedback_after(stamp,deadline)
                if state.get('boot_id')!=self.lease['boot_id'] or state.get('session_id')!=self.lease['session_id']:
                    raise ValueError('driver_session_changed')
                if state.get('state')=='hold' and state.get('hold_confirmed'):return True
                self.condition.wait(min(.01,max(0,deadline-time.monotonic())))
        return False

    def send(self,q,hands,deadline,*,wait_for_execution=True,allow_continuation=False,target_ttl_ms=None,**g1):
        from std_msgs.msg import String
        if not self.lease:raise ValueError('driver_not_owned')
        if not wait_for_execution:
            state=self.feedback()
            # A cached pre-claim snapshot is not evidence about the new lease.
            if state['monotonic_ns']>=self.lease_started_ns:
                if state.get('boot_id')!=self.lease['boot_id']:raise ValueError('driver_restarted')
                if state.get('session_id')!=self.lease['session_id']:raise ValueError('driver_lease_lost')
                if (allow_continuation and state.get('state') == 'hold'
                        and state.get('continuation_allowed') is True
                        and state.get('continuation_ready') is not True):
                    return False  # Hold began during IK. Drop this target, do not queue it.
                continuing = (allow_continuation and state.get('state') == 'hold'
                              and state.get('continuation_ready') is True)
                if state.get('state') in ('hold','fault') and not continuing:raise ValueError('driver_holding')
                if continuing:
                    # Discard acknowledgements for targets already stopped by the
                    # Driver. This does not acknowledge their physical execution.
                    self.pending.clear();self.execution_progress_ns=0
                while self.pending and self.pending[0][0]<=state.get('applied_sequence',-1):
                    self.pending.popleft()
                    self.execution_progress_ns=max(self.execution_progress_ns,state['monotonic_ns'])
            # Latest-target execution can skip superseded sequences. Only actual
            # applied-sequence progress extends this watchdog, never new input.
            if self.pending:
                ack_deadline=(self.execution_progress_ns/1e9+.1 if self.execution_progress_ns
                              else self.pending[0][1])
                if time.monotonic()>=ack_deadline:
                    if allow_continuation and time.monotonic()<ack_deadline+.2:
                        return False  # No publish; wait for progress within the 300 ms recovery window.
                    raise ValueError('driver_execution_ack_timeout')
            else:
                self.execution_progress_ns=0
        remaining_ms=int((deadline-time.monotonic())*1000)
        if remaining_ms<=0:raise ValueError('motion_deadline')
        if target_ttl_ms is not None and (type(target_ttl_ms) is not int or not 1<=target_ttl_ms<=100):
            raise ValueError('invalid_target_ttl')
        # A newly computed command has its own generation time. The input/IO
        # deadline still gates publication; it is not the downstream transport TTL.
        now=time.monotonic_ns();ttl=min(100,remaining_ms) if target_ttl_ms is None else target_ttl_ms
        self.seq+=1
        body={'protocol':'motus.motion-target.v1','boot_id':self.lease['boot_id'],
              'session_id':self.lease['session_id'],'seq':self.seq,'generated_ns':now,
              'valid_for_ms':ttl,'q':q,'hands':hands}
        if g1:
            body.pop('hands');body.update(g1);body['protocol']='motus.motion-target.v2'
        canonical=json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        mac=hmac.new(bytes.fromhex(self.lease['secret']),canonical,hashlib.sha256).hexdigest()
        msg=String();msg.data=json.dumps({**body,'mac':mac},allow_nan=False)
        self.publisher.publish(msg)
        self.last_send = {'sequence':self.seq,'generated_ns':now,'published_ns':time.monotonic_ns(),
                          'valid_for_ms':ttl,'deadline_ns':now+ttl*1_000_000,
                          'feedback_received_ns':getattr(self,'last_feedback_received_ns',None)}
        if not wait_for_execution:
            # Execution acknowledgement has its own 100 ms budget; target TTL stays above.
            self.pending.append((self.seq,now/1e9+.1))
            return True
        with self.condition:
            while time.monotonic()<deadline:
                state=self.feedback_after(now,deadline)
                if state.get('boot_id')!=self.lease['boot_id']:raise ValueError('driver_restarted')
                if state.get('session_id')!=self.lease['session_id']:raise ValueError('driver_lease_lost')
                if state.get('state') in ('hold','fault'):raise ValueError('driver_holding')
                if state.get('applied_sequence',-1)>=self.seq:return
                self.condition.wait(min(0.01,max(0,deadline-time.monotonic())))
        raise ValueError('driver_execution_ack_timeout')

    def stop(self,deadline):
        if self.reconcile_release():return True
        if not self.release_requested_ns:self.release_requested_ns=time.monotonic_ns()
        try:
            result=self.call('release',deadline)
            operation=getattr(self,'management_request',None)
            if operation:
                operation['cancel_acknowledged']=True
                if (not self.lease and operation['action']=='claim' and not operation.get('credentials')
                        and result.get('cancelled_request_id')==operation['id']
                        and result.get('cancelled_without_output') is True
                        and result.get('boot_id')==operation['boot_id']
                        and result.get('state')=='idle' and result.get('ownership_held') is False
                        and result.get('output_active') is False):
                    operation['cancelled_without_output']=True
        except (OSError,ValueError) as exc:
            if isinstance(exc,ValueError) and str(exc) not in ('invalid_lease','management_owner_changed'):
                raise
        stamp=self.release_requested_ns
        with self.condition:
            while time.monotonic()<deadline:
                if self.reconcile_release():return True
                status=self.feedback_after(stamp,deadline)
                operation=getattr(self,'management_request',None)
                boot_id=self.lease['boot_id'] if self.lease else operation['boot_id']
                if status.get('boot_id') != boot_id:raise ValueError('driver_restarted_stop_unconfirmed')
                self.condition.wait(min(0.01,max(0,deadline-time.monotonic())))
        return False

    def show(self,value):
        from std_msgs.msg import String
        msg=String();msg.data=json.dumps(value,allow_nan=False);self.preview.publish(msg)

    def close(self):
        self.executor.remove_node(self.node);self.node.destroy_node()


class IntentAdapter:
    def __init__(self,link,mode,scale=0.5):
        self.link=link;self.hardware_output=mode=='live'
        self.mapper=RelativeMapping(scale);self.solver=None
        self.lock=threading.RLock();self.clutch=None
        self.output={'state':'idle','output_active':False,'publisher_present':False}
        self.ik_times=deque(maxlen=256);self.chain_times=deque(maxlen=256);self._revoked_generation=-1

    def startup_safe(self,deadline_monotonic):
        return AdapterAck(self.link.lease is None,'no_execution_lease')

    def calibrate(self,path):
        from .kinematics import TianyiIK
        with self.lock:
            if self.link.lease:raise ValueError('calibration_while_owned')
            solver=TianyiIK(path)
            q=self.link.feedback()['feedback']['q']
            result=solver.self_test(q)
            offsets={side:transform(solver.profile['controller_to_palm'][side]) for side in ('left','right')}
            self.mapper.controller_offsets=offsets
            self.solver=solver;self.clutch=None
            return result

    def apply(self,intent):
        deadline=min(intent.expires_monotonic,(intent.received_monotonic or intent.admitted_monotonic)+0.1)
        with self.lock:
            try:
                if intent.dispatch_generation<=self._revoked_generation:raise ValueError('dispatch_revoked')
                if self.solver is None:raise ValueError('calibration_missing')
                frame=intent.frame
                triggers=[frame['controllers'][s]['buttons'][0] for s in ('left','right')]
                if any(len(frame['controllers'][s]['buttons'])!=2 for s in ('left','right')):
                    raise ValueError('trigger_binding_missing')
                feedback=self.link.feedback();q=feedback['feedback']['q']
                if self.hardware_output and feedback.get('calibration_sha256')!=self.solver.profile_sha256:
                    raise ValueError('driver_calibration_mismatch')
                stamp=feedback['feedback'].get('arm_ns',0)
                if not 0<=time.monotonic_ns()-stamp<=100_000_000:raise ValueError('arm_feedback_stale')
                clutch=(intent.session_generation,intent.clutch_sequence)
                if clutch!=self.clutch:
                    if max(triggers)>0.05:raise ValueError('triggers_not_neutral')
                    self.mapper.reset(frame,self.solver.palms(q));self.clutch=clutch
                targets=self.mapper.targets(frame)
                target=self.solver.solve(targets,q,commanded=feedback.get("commanded_q"));self.ik_times.append(self.solver.last_ms)
                if time.monotonic()>=deadline:raise ValueError('ik_motion_deadline')
                if intent.dispatch_generation<=self._revoked_generation:raise ValueError('dispatch_revoked')
                if self.hardware_output:
                    self.link.claim(deadline)
                    self.link.send(target,triggers,deadline)
                    self.output={'state':'submitted','target_q':target}
                else:
                    self.output={'state':'would_apply','target_q':target,'hands':triggers,
                                 'publisher_present':False,'output_active':False}
                self.chain_times.append((time.monotonic()-(intent.received_monotonic or intent.admitted_monotonic))*1000)
                return AdapterAck(True)
            except Exception as exc:
                self.output={'state':'error','code':str(exc),'output_active':False}
                return AdapterAck(False,'motion_rejected')

    def safe_stop(self,request):
        # Fence before waiting on any in-flight solver. The arbiter also prevents late apply.
        self._revoked_generation=max(self._revoked_generation,request.dispatch_generation-1)
        acquired=self.lock.acquire(timeout=max(0,request.deadline_monotonic-time.monotonic()))
        if not acquired:return AdapterAck(False,'stop_unconfirmed')
        try:
            ok=self.link.stop(request.deadline_monotonic) if self.hardware_output else True
            self.clutch=None
            self.output={'state':'stopped' if ok else 'stop_unconfirmed','output_active':False if ok else None}
            return AdapterAck(ok,'stop_confirmed' if ok else 'stop_unconfirmed')
        except Exception:
            return AdapterAck(False,'stop_unconfirmed')
        finally:
            self.lock.release()

    def snapshot(self):
        with self.lock:
            output=dict(self.output)
            if self.hardware_output:
                try:
                    state=self.link.feedback()
                    output.update(output_active=state.get('output_active'),publisher_present=state.get('publisher_present'),driver=state)
                except ValueError:
                    output.update(output_active=None,publisher_present=None,driver_feedback_fresh=False)
            times=sorted(self.chain_times)
            p95=times[min(len(times)-1,int(len(times)*.95))] if times else None
            output['receive_to_driver_p95_ms']=p95 if self.hardware_output else None
            output['receive_to_target_p95_ms']=p95 if not self.hardware_output else None
            return {'output':output,'diagnostics':{'ik_ms':list(self.ik_times),
                'calibrated':self.solver is not None,'calibration_version':self.solver.profile.get('version') if self.solver else None}}

    def close(self):
        from .dispatch import StopRequest
        now=time.monotonic()
        return self.safe_stop(StopRequest(self._revoked_generation+2,'close',now,now+0.2))
