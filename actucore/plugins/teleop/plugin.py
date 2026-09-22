"""One Canvas card hosting PICO input and robot motion intentions."""
from __future__ import annotations
import asyncio
import copy
import time
import secrets
import threading
import hashlib
import json
import os
import tempfile
from pathlib import Path
from .recording import PoseRecorder
from .adapter import DriverLink, IntentAdapter, validate_driver_endpoint
from .capture import CaptureManager
from .capture_server import CaptureWssServer, capture_certificate_base64
from .protocol import TicketCodec, TicketVerifier
from .rtc import RtcManager
from .runtime import TeleopRuntime

ACTIONS=['info','config','project_start','project_stop','start','stop','finish','pair_headset','revoke_headset','calibrate','pause','resume','self_test','open_pairing','approve_pairing','reject_pairing','disconnect_headset','record_start','record_stop','record_status','installation_info','create_invitation','revoke_invitation']


class TeleopPlugin:
    PREFIX='teleop'

    def __init__(self,plugin_cfg,executor):
        self.cfg=copy.deepcopy(plugin_cfg);self.executor=executor
        self._lock=threading.RLock();self._loop=None;self._thread=None
        self.runtime=None;self.link=None;self.adapter=None;self.capture=None;self.server=None
        self.recorder=None
        self.operator_commands=None
        self._operator_cancel=threading.Event()
        self._cancel_lock=threading.Lock();self._stop_generation=0
        self._operation_lock=threading.RLock()
        self._operation_action=None
        # Project permission is volatile and deliberately absent from the saved
        # configuration. Pairing and reconnecting never restore permission.
        self._project_armed=False;self._project_stopping=False
        self._project_binding=None;self._project_error=None
        self._return_required=False
        self._operator_session_prepared=False
        self.error=None;self._closing=False;self._instance=None;self._capture_status={};self._enrollment_status={}
        # Retain accepted Canvas settings even when Core misses a short restart.
        # Only configuration is restored: never a lease, session or start request.
        state=self.cfg.get('capture',{}).get('state_file')
        self._config_file=Path(state).with_suffix('.config.json') if state else None
        self._config_base=hashlib.sha256(json.dumps(self.cfg,sort_keys=True).encode()).hexdigest()
        self._config_error=None
        try:
            if self._config_file and self._config_file.exists():
                saved=json.loads(self._config_file.read_text())
                if saved.get('schema_version')!=1:raise ValueError('schema')
                if saved.get('base_sha256')==self._config_base:
                    self.cfg=self._validate_configuration(saved['values'])
        except (OSError,ValueError,KeyError,TypeError,AttributeError):
            self._config_error=self.error='saved_configuration_invalid'

    def _actions(self):
        unsupported={'project_start','project_stop','finish','record_start','record_stop','record_status'} if self.cfg.get('robot_profile')=='g1_23' else set()
        return [action for action in ACTIONS if action not in unsupported]

    def _remote_control(self):
        # Existing saved projects retain their legacy path until explicitly
        # wired to motion_control. New installation config selects it directly.
        return self.cfg.get('robot_profile','tianyi2') != 'g1_23' and self.cfg.get('control_backend','legacy') == 'motion_control'

    def _calibrated(self):
        return bool(self.adapter and (getattr(self.adapter,'calibrated',False) or self.adapter.solver is not None))

    def get_tools(self):
        actions=self._actions()
        tools=[{'name':'teleop','type':'processor','multiInstance':False,'x-connection-panel':'teleop-v1',
            'description':'PICO 遥操：G1 双臂或天轶双臂与手开合。双握把使能；默认 Shadow；Driver 确认执行。',
            'inputSchema':{'type':'object','properties':{'action':{'type':'string','enum':actions},
                'instance_id':{'type':'string'},'request_id':{'type':'string'},'fingerprint':{'type':'string'},
                'driver_binding':{'type':'object'}},'required':['action'],'additionalProperties':False,
                'x-action-params':{a:{'params':['request_id','fingerprint'] if a in ('approve_pairing','reject_pairing') else ['driver_binding'] if a=='project_start' else []} for a in actions},
                'x-resource':(['arm_l','arm_r'] if self.cfg.get('robot_profile')=='g1_23' else ['arm_l','arm_r','hand_l','hand_r'])},
            'configSchema':{'type':'object','properties':{
                'robot_profile':{'type':'string','enum':['tianyi2','g1_23'],'default':'tianyi2','scope':'shared'},
                'control_backend':{'type':'string','enum':['legacy','motion_control'],'default':'legacy','scope':'shared'},
                'controller_to_palm':{'type':'object','scope':'shared'},
                'mode':{'type':'string','enum':['shadow','live'],'default':'shadow','scope':'shared'},
                'shadow_feedback_source':{'type':'string','enum':['teleop_executor','driver_joints'],'default':'teleop_executor','scope':'shared'},
                'mapping_version':{'type':'string','enum':['relative_v1','pr152_head_yaw_v1','pr152_clutch_relative_v1'],'default':'relative_v1','scope':'shared'},
                'position_scale':{'type':'number','minimum':0.01,'maximum':1,'default':0.5,'scope':'shared'},
                'namespace':{'type':'string','scope':'shared','x-sensitive':True},
                'driver_mcp_url':{'type':'string','scope':'shared','x-sensitive':True},
                'calibration_path':{'type':'string','scope':'shared','x-sensitive':True}},'additionalProperties':False},
            'topic_out':[{'topic':f"/{self.cfg.get('namespace','robot')}/motion/teleop/command",'format':'control/teleop'},
                         {'topic':f"/{self.cfg.get('namespace','robot')}/teleop/status",'format':'data/json'}]}]
        if self._remote_control():
            tool=tools[0]
            tool['description']='通用 PICO 遥操输入：连接 Driver 运动控制卡，双握把使能；Driver 解算与执行。'
            tool['topic_out'][0]={'id':'targets','topic':f"/{self.cfg.get('namespace','robot')}/motion/control/command",'format':'control/eef'}
            tool['topic_in']=[{'id':'feedback','format':'data/json','role':'feedback'}]
            tool['inputSchema']['x-resource']=['arm_l','arm_r']
        return tools

    def _run(self,coro,timeout=2):
        future=asyncio.run_coroutine_threadsafe(coro,self._loop)
        try:return future.result(timeout)
        except Exception:
            future.cancel();raise

    def _ensure_host(self):
        if self._config_error:raise ValueError(self._config_error)
        if self.runtime:return
        self._closing=False
        try:self._open_host()
        except Exception:
            self.stop()
            raise

    def _open_host(self):
        mode=self.cfg.get('mode','shadow')
        if mode not in ('shadow','live'):raise ValueError('invalid_mode')
        source=self.cfg.get('shadow_feedback_source','teleop_executor')
        if source not in ('teleop_executor','driver_joints'):raise ValueError('invalid_shadow_feedback_source')
        if source=='driver_joints':
            from .g1_shadow_feedback import G1ShadowFeedbackLink
            self.link=G1ShadowFeedbackLink(self.cfg,self.executor)
        elif self._remote_control():
            from .motion_control import MotionControlLink
            self.link=MotionControlLink(self.cfg,self.executor)
        else:self.link=DriverLink(self.cfg,self.executor)
        profile=self.cfg.get('robot_profile','tianyi2')
        if profile not in ('tianyi2','g1_23'):raise ValueError('unknown_robot_profile')
        from .tianyi import TianyiIntentAdapter
        adapter=TianyiIntentAdapter;capabilities=None
        if self._remote_control():
            from .motion_control import EefIntentAdapter,CAPABILITIES_EEF
            adapter=EefIntentAdapter;capabilities=CAPABILITIES_EEF
        if profile=='g1_23':
            from .g1 import G1IntentAdapter,CAPABILITIES_G1
            adapter=G1IntentAdapter;capabilities=CAPABILITIES_G1
        mapping=self.cfg.get('mapping_version','relative_v1')
        if mapping not in ('relative_v1','pr152_head_yaw_v1','pr152_clutch_relative_v1') or (profile!='g1_23' and mapping!='relative_v1'):
            raise ValueError('mapping_profile_mismatch')
        options={'mapping_version':mapping} if profile=='g1_23' else {}
        self.adapter=adapter(self.link,mode,self.cfg.get('position_scale',1 if mapping!='relative_v1' else .5),**options)
        if self.cfg.get('controller_to_palm'):
            from .mapping import transform
            self.adapter.mapper.controller_offsets={s:transform(self.cfg['controller_to_palm'][s]) for s in ('left','right')}
        self.runtime=TeleopRuntime(mode=mode,adapter=self.adapter,
                                   pose_timeout_ms=getattr(self.adapter,'input_timeout_ms',100),
                                   dispatch_io_timeout_ms=getattr(self.adapter,'dispatch_io_timeout_ms',100),
                                   capabilities=capabilities,
                                   motion_interval_ms=20,
                                   driver_id=profile+'-actucore',driver_name=profile+' ActuCore Teleop')
        if self.recorder is None:
            self.recorder=PoseRecorder(Path(self.cfg['capture']['state_file']).parent/'recordings')
        def record_frame(frame, received_at):
            with self.link.condition:feedback=dict(self.link.latest or {})
            self.recorder.capture(frame,received_at,feedback)
        if profile=='tianyi2':self.runtime.pose_observer=record_frame
        self._loop=asyncio.new_event_loop()
        self._thread=threading.Thread(target=self._loop.run_forever,name='actucore-capture',daemon=True)
        self._thread.start()
        codec=TicketCodec(secrets.token_urlsafe(48))
        self.rtc=RtcManager(self.runtime,TicketVerifier(codec))
        config=self.cfg['capture']
        self.capture=CaptureManager(self.runtime,self.rtc,codec,
            state_file=config['state_file'],public_wss_url=config['public_wss_url'],
            ca_certificate_base64=capture_certificate_base64(config),presence_interval_ms=250,presence_timeout_ms=1000)
        if profile=='g1_23':
            from .g1_visualization import snapshot as visualization_snapshot
        elif self._remote_control():
            visualization_snapshot=lambda adapter:adapter.visualization()
        else:
            from .tianyi_visualization import snapshot as visualization_snapshot
        from .operator_session import OperatorCommands
        self.operator_commands=OperatorCommands(self.capture,self._operator_execute)
        if profile=='tianyi2':self.capture.operator_commands=self.operator_commands
        def visual():
            result=visualization_snapshot(self.adapter)
            if profile=='tianyi2':
                operation=dict(self.operator_commands.status)
                if operation.get('state') not in ('returning','starting','stopping','error'):
                    if operation.get('action')=='start':
                        raw=result.get('state','idle')
                        operation['state']={'submitted':'active','would_apply':'active','preview_submitted':'active',
                            'target_published':'active','preview':'active',
                            'held':'hold','hold':'hold','error':'hold','armed_waiting_input':'ready',
                            'waiting_driver_hold':'hold','waiting_driver_feedback':'hold'}.get(raw,operation.get('state','idle'))
                result['operator']={**operation,'enabled':True,'armed':self._project_armed,
                                    'mode':self.cfg.get('mode','shadow')}
            return result
        self.capture.visualization_provider=visual
        self.server=CaptureWssServer(self.capture,config)
        self._enrollment_status=self.server.enrollment.status()
        self._run(self.server.start(),timeout=5)
        async def publish_status():
            while not self._closing:
                try:
                    self._capture_status=await self.capture.status()
                    self._enrollment_status=self.server.enrollment.status()
                    self.link.show(self.info())
                except Exception as exc:self.error=str(exc)
                await asyncio.sleep(0.1)
        self._status_future=asyncio.run_coroutine_threadsafe(publish_status(),self._loop)

    def info(self):
        properties=self.get_tools()[0]['configSchema']['properties']
        configuration={k:copy.deepcopy(self.cfg.get(k,v.get('default'))) for k,v in properties.items()
                       if k in self.cfg or 'default' in v}
        project={'armed':self._project_armed,'stopping':self._project_stopping,
                 'error':self._project_error,'driver_binding':copy.deepcopy(self._project_binding)}
        if not self.runtime:
            return {'state':'fault' if self.error else 'idle','reason':self.error,
                    'mode':self.cfg.get('mode','shadow'),'output_active':False,
                    'topic_out':self.get_tools()[0]['topic_out'],'configuration':configuration,'project':project}
        result=self.runtime.status()
        result['configuration']=configuration
        result['project']=project
        result['session_state']=result['state']
        result['state']={'prepared_shadow':'ready','prepared_live':'ready',
                         'active_shadow':'active','active_live':'active','released':'idle','paused':'hold'}.get(result['state'],result['state'])
        result['topic_out']=self.get_tools()[0]['topic_out']
        result['capture']=dict(self._capture_status)
        result['host_error']=self.error
        result['enrollment']=copy.deepcopy(self._enrollment_status)
        result['calibrated']=self._calibrated()
        result['operator']={**(dict(self.operator_commands.status) if self.operator_commands else {}),
                            'armed':self._project_armed}
        result['recording']=self.recorder.status() if self.recorder else {'state':'idle'}
        try:
            result['driver_feedback']=self.link.feedback()
            feedback=result['driver_feedback']
            stamp=feedback.get('feedback',{}).get('arm_ns',0)
            result['driver_feedback_fresh']=0<=time.monotonic_ns()-stamp<=getattr(self.link,'feedback_max_age_ns',100_000_000)
        except ValueError as exc:
            result['driver_feedback']=None
            result['driver_feedback_fresh']=False
            result['driver_feedback_error']=str(exc)
        return result

    def _validate_configuration(self,values):
        properties=self.get_tools()[0]['configSchema']['properties']
        if not isinstance(values,dict) or set(values)-set(properties):raise ValueError('unknown_config')
        updated={**self.cfg,**values}
        for key,value in values.items():
            definition=properties[key]
            if definition['type']=='string' and not isinstance(value,str):raise ValueError('invalid_config:'+key)
            if 'enum' in definition and value not in definition['enum']:raise ValueError('invalid_config:'+key)
        if updated.get('mode','shadow') not in ('live','shadow'):raise ValueError('invalid_mode')
        if 'controller_to_palm' in values:
            from .mapping import transform
            offsets=values['controller_to_palm']
            if not isinstance(offsets,dict) or set(offsets)!={'left','right'}:raise ValueError('controller_offsets_required')
            for offset in offsets.values():transform(offset)
        if updated.get('robot_profile')=='g1_23' and updated.get('control_backend','legacy')!='legacy':
            raise ValueError('g1_motion_control_not_migrated')
        mapping=updated.get('mapping_version','relative_v1')
        if mapping not in ('relative_v1','pr152_head_yaw_v1','pr152_clutch_relative_v1') or (updated.get('robot_profile')!='g1_23' and mapping!='relative_v1'):
            raise ValueError('mapping_profile_mismatch')
        scale=updated.get('position_scale',1 if mapping!='relative_v1' else .5)
        if mapping!='relative_v1' and scale!=1:raise ValueError('g1_head_yaw_requires_unit_scale')
        if type(scale) not in (int,float) or not .01<=scale<=1:raise ValueError('position_scale')
        if updated.get('shadow_feedback_source')=='driver_joints' and (
                updated.get('robot_profile')!='g1_23' or updated.get('mode','shadow')!='shadow'):
            raise ValueError('joints_feedback_requires_g1_shadow')
        validate_driver_endpoint(updated)
        return updated

    def _save_configuration(self,updated):
        if not self._config_file:return
        values={k:updated[k] for k in self.get_tools()[0]['configSchema']['properties'] if k in updated}
        payload={'schema_version':1,'base_sha256':self._config_base,'values':values}
        self._config_file.parent.mkdir(parents=True,exist_ok=True)
        name=None
        try:
            with tempfile.NamedTemporaryFile(mode='w',dir=self._config_file.parent,delete=False) as f:
                name=f.name
                json.dump(payload,f);f.flush();os.fsync(f.fileno())
            os.replace(name,self._config_file)
        finally:
            if name and os.path.exists(name):os.unlink(name)

    def dispatch(self,name,args):
        action=args.get('action','info')
        if action=='stop' and self.cfg.get('robot_profile','tianyi2')!='tianyi2':self._cancel_operations()
        try:
            # Long return operations must not hold the configuration/status lock
            # or occupy the WSS loop. One operation lock serializes both origins.
            if name=='teleop' and action in ('project_start','project_stop'):
                if action not in self._actions():raise ValueError('unsupported_action')
                return self._project_start(args.get('driver_binding')) if action=='project_start' else self._project_stop()
            if (name=='teleop' and action in ('start','resume','stop')
                    and self.cfg.get('robot_profile','tianyi2')=='tianyi2'):
                return self._operator_execute(action,threading.Event(),self._headset_connected)
            with self._lock:
                if name!='teleop':return None
                if action not in self._actions():raise ValueError('unsupported_action')
                instance=args.get('instance_id')
                if self._instance and instance and self._instance!=instance:raise ValueError('teleop_single_instance')
                if instance:self._instance=instance
                if action=='info':
                    self._ensure_host()
                    return self.info()
                if action=='config':
                    values={k:v for k,v in args.items() if k not in ('action','instance_id')}
                    updated=self._validate_configuration(values)
                    if updated==self.cfg and not self._config_error:return self.info()
                    if self._project_armed or self._project_stopping or self._return_required:
                        raise ValueError('project_stop_before_config')
                    if self.runtime and (self.runtime.status()['authority_valid'] or self.link.lease):
                        raise ValueError('release_before_config')
                    self.stop();self._save_configuration(updated)
                    self.cfg=updated;self.error=None;self._config_error=None;return self.info()
                self._ensure_host()
                if action=='finish':
                    connection=self.capture._connection
                    if connection is None:raise ValueError('finish_requires_connected_headset')
                    return self._run(self.operator_commands.submit(connection,{
                        'action':'finish','connection_id':connection.connection_id,
                        'request_id':args.get('request_id') or secrets.token_hex(16)}))
                if action=='record_status':return self.recorder.status()
                if action=='record_stop':return self.recorder.stop()
                if action=='record_start':
                    if self.cfg.get('robot_profile')!='tianyi2' or self.cfg.get('mode')!='shadow' or self.link.lease:
                        raise ValueError('record_requires_tianyi_shadow')
                    if self._remote_control():
                        metadata={'control_interface':'motus.control/2',
                            'calibration':copy.deepcopy(self.adapter.calibration),
                            'position_scale':self.cfg.get('position_scale',.5)}
                    else:
                        profile=Path(self.cfg['calibration_path'])
                        modules=('runtime.py','adapter.py','tianyi.py','kinematics.py','workspace.py','recording.py')
                        metadata={'profile':json.loads(profile.read_text()),
                            'profile_sha256':hashlib.sha256(profile.read_bytes()).hexdigest(),
                            'position_scale':self.cfg.get('position_scale',.5),
                            'sources':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in modules}}
                    return self.recorder.start(metadata,duration=10,wait_for_deadman=True)
                if action=='installation_info':return self.server.installation_info()
                if action=='create_invitation':return self._run(self.server.enrollment.create_invitation())
                if action=='revoke_invitation':return self.server.enrollment.revoke_invitation()
                if action=='open_pairing':return self._run(self.server.enrollment.open())
                if action in ('approve_pairing','reject_pairing'):
                    async def decide():
                        return self.server.enrollment.decide(args.get('request_id'),args.get('fingerprint'),action=='approve_pairing')
                    return self._run(decide())
                if action=='disconnect_headset':
                    self.runtime.release_local()
                    async def disconnect():
                        connection=self.capture._connection
                        if connection:
                            self.capture.disconnect_immediate(connection)
                            await connection.events.put({'type':'capture_stale'})
                            await self.capture.disconnect(connection)
                    self._run(disconnect())
                    return self.info()
                if action=='pair_headset':return self._run(self.capture.create_pairing())
                if action=='revoke_headset':
                    self.runtime.release_local()
                    async def revoke():
                        self.server.enrollment.revoke_invitation()
                        self.server.enrollment.pending=None
                        self.server.enrollment.deadline=0
                        return await self.capture.revoke_headset()
                    return self._run(revoke())
                if action in ('calibrate','self_test'):
                    if self.runtime.status()['authority_valid']:raise ValueError('release_before_calibrating')
                    return self.adapter.calibrate(self.cfg.get('calibration_path'))
                if action in ('start','resume'):
                    if action=='resume' and self.cfg.get('robot_profile','tianyi2')=='tianyi2' and not self._calibrated():
                        self._calibrate_recovered_host()
                    if not self._calibrated():raise ValueError('calibrate_before_start')
                    if action=='resume':
                        if self.cfg.get('robot_profile','tianyi2')=='tianyi2' and self.runtime.status()['dispatch'].get('fault_code'):
                            self._recover_tianyi_host()
                        else:
                            self.runtime.release_local()
                            self._run(self.capture.revoke_assignment('operator_resume'))
                    if (self.cfg.get('operator_session_enabled') is True and self.adapter.hardware_output
                            and not self.runtime.status()['authority_valid'] and not self.link.lease):
                        self.link.call('prepare_operator_session',time.monotonic()+.5)
                    self.runtime.prepare_local_session()
                    self._run(self.capture.issue_assignment_if_connected())
                elif action=='pause':self.runtime.pause_local()
                elif action=='stop':
                    from .protocol import ProtocolError
                    try:self.runtime.release_local()
                    except ProtocolError as exc:
                        if exc.code!='dispatch_stop_unconfirmed':raise
                    # A past dispatch failure must not prevent a fresh release attempt.
                    self._run(self.capture.revoke_assignment('operator_stop'))
                    if not self._release_driver():raise ValueError('stop_unconfirmed')
                    if self.cfg.get('operator_session_enabled') is True:
                        self.link.call('end_operator_session',time.monotonic()+.25)
                    self._instance=None
                self.error=None
                return self.info()
        except Exception as exc:
            self.error=str(exc)
            return {'state':'error','error':str(exc),'code':getattr(exc,'code','teleop_not_ready')}

    def _headset_connected(self):
        connection=self.capture._connection if self.capture else None
        return bool(connection and not self.capture.presence_expired(connection))

    def _project_start(self,binding):
        from .project import validate_binding
        binding=validate_binding(binding,self.cfg.get('robot_profile','tianyi2'))
        with self._operation_lock,self._lock:
            if self._project_stopping or self._project_error:raise ValueError('project_stop_required')
            if self._project_armed:
                if binding!=self._project_binding:raise ValueError('project_binding_changed')
                return {'state':'ready','armed':True,'driver_binding':copy.deepcopy(binding)}
            if self._return_required:raise ValueError('project_stop_required')
            updated={**self.cfg,'namespace':binding['namespace'],'driver_mcp_url':binding['url']}
            if binding['protocol_version']==2:updated['control_backend']='motion_control'
            elif self.cfg.get('control_backend')=='motion_control':updated['control_backend']='legacy'
            if updated!=self.cfg:
                if self.runtime and (self.runtime.status()['authority_valid'] or self.link.lease):
                    raise ValueError('release_before_binding')
                self.stop();self.cfg=updated
            self._ensure_host()
            self._project_binding=binding
            self._project_armed=True
            self.error=None
            return {'state':'ready','armed':True,'driver_binding':copy.deepcopy(binding)}

    def _project_stop(self):
        deadline=time.monotonic()+50
        with self._cancel_lock:stop_generation=self._stop_generation
        # Fence a calibration/start already holding _lock before waiting for
        # it. That worker rechecks these flags immediately before prepare.
        self._project_armed=False;self._project_stopping=True
        if self._operation_action in ('start','resume'):self._operator_cancel.set()
        with self._lock:
            self._project_armed=False;self._project_stopping=True
            self._operation_status('returning','project_stop')
        try:
            # A concurrent PICO finish may already be returning. Join it, then
            # reuse its completion instead of sending a second return sequence.
            if not self._operation_lock.acquire(timeout=max(0.,deadline-time.monotonic())):raise ValueError('project_stop_busy')
            try:
                result=self._finish_session(threading.Event(),lambda:True,deadline=deadline,
                                            stop_generation=stop_generation)
            finally:self._operation_lock.release()
            self._project_error=None
            self._operation_status('idle','project_stop')
            return {**result,'armed':False}
        except Exception as exc:
            self._project_error=str(exc)
            self._operation_status('error','project_stop',str(exc))
            raise
        finally:self._project_stopping=False;self._operation_action=None

    def _finish_session(self,cancel,connected,*,deadline=None,stop_generation=None):
        from .operator_session import return_arms
        with self._lock:
            self._install_cancel(cancel,stop_generation);self._operation_action='finish'
            if self.runtime:
                self.runtime.release_local()
                self._run(self.capture.revoke_assignment('operator_finish'))
                dispatch=self.runtime.status()['dispatch']
                if dispatch.get('io_inflight') or dispatch.get('stop_queue_depth'):
                    raise ValueError('return_waiting_for_adapter')
            live=bool(self.adapter and self.adapter.hardware_output)
            if not self._return_required or not live:
                if self.adapter and not self._release_driver():raise ValueError('stop_unconfirmed')
                result={'state':'idle','return_completed':True,'return_required':False,
                        'authority_released':True,'mode':self.cfg.get('mode','shadow')}
            else:result=None
        if result is None:
            if cancel.is_set():raise ValueError('return_cancelled')
            remaining=45. if deadline is None else min(45.,deadline-time.monotonic()-2.)
            if remaining<=0:raise ValueError('project_stop_timeout')
            with self.adapter.lock:
                if getattr(self.link,'release_requested_ns',0):self.link.reconcile_release()
                if not self.link.lease and not getattr(self.link,'management_request',None):
                    # Explicit stop clears the Driver's preparation as well as
                    # its lease. A later explicit finish is a new bounded return
                    # operation, not permission to reuse that cleared session.
                    if self.cfg.get('operator_session_enabled') is not True and not self._remote_control():
                        raise ValueError('operator_session_disabled')
                    self.link.call('prepare_operator_session',time.monotonic()+.5)
                    self._operator_session_prepared=True
            if getattr(self.adapter,'remote_motion_control',False):
                result=self.adapter.finish(cancel,connected,timeout=remaining)
            else:result=return_arms(self.adapter,cancel,connected,timeout=remaining)
            if not self._release_driver():raise ValueError('stop_unconfirmed')
            result.update(authority_released=True,return_required=True)
        if live and self._operator_session_prepared:
            self.link.call('end_operator_session',time.monotonic()+.25)
            self._operator_session_prepared=False
        self._return_required=False;self._operation_action=None
        self.error=None
        return result

    def _operator_execute(self, action, cancel, connected):
        """Called on a worker; never occupies the WSS presence loop."""
        if action=='stop':self._cancel_operations()
        with self._cancel_lock:stop_generation=self._stop_generation
        with self._operation_lock:
            self._operation_action=action
            self._operation_status({'finish':'returning','stop':'stopping'}.get(action,'starting'),action)
            try:
                result=self._operator_execute_locked(action,cancel,connected,stop_generation)
                self.error=None
                self._operation_status(result.get('state','idle'),action)
                return result
            except Exception as exc:
                self._operation_status('error',action,str(exc))
                raise
            finally:self._operation_action=None

    def _operation_status(self,state,action,error=None):
        if self.operator_commands is not None:
            self.operator_commands.set_status({'state':state,'action':action,'error':error})

    def _cancel_operations(self):
        with self._cancel_lock:
            self._stop_generation+=1
            self._operator_cancel.set()

    def _install_cancel(self,cancel,generation):
        with self._cancel_lock:
            if generation is not None and generation!=self._stop_generation:cancel.set()
            self._operator_cancel=cancel

    def _operator_execute_locked(self, action, cancel, connected,stop_generation=None):
        if self.cfg.get('robot_profile','tianyi2')!='tianyi2':
            raise ValueError('operator_profile_unsupported')
        if action=='stop':
            self._operator_cancel.set()
            with self._lock:
                from .protocol import ProtocolError
                if self.runtime:
                    try:self.runtime.release_local()
                    except ProtocolError as exc:
                        if exc.code!='dispatch_stop_unconfirmed':raise
                    self._run(self.capture.revoke_assignment('operator_stop'))
                    if not self._release_driver():raise ValueError('stop_unconfirmed')
                if self._operator_session_prepared:
                    self.link.call('end_operator_session',time.monotonic()+.25)
                    self._operator_session_prepared=False
                self._instance=None
                result={'state':'idle','authority_released':True,'mode':self.cfg.get('mode','shadow')}
        elif action in ('start','resume'):
            if not self._project_armed or self._project_stopping:raise ValueError('project_not_armed')
            if not connected() or cancel.is_set():raise ValueError('operator_connection_changed')
            if action=='start' and self.runtime.status()['authority_valid']:
                return {'state':'ready','mode':self.cfg.get('mode','shadow')}
            self._install_cancel(cancel,stop_generation)
            if cancel.is_set():raise ValueError('operator_start_cancelled')
            with self._lock:
                if self.runtime.status()['dispatch'].get('fault_code'):
                    self._recover_tianyi_host()
                elif action=='resume':
                    self.runtime.release_local()
                    self._run(self.capture.revoke_assignment('operator_resume'))
                    if not self._release_driver():raise ValueError('stop_unconfirmed')
                if self.link.lease:raise ValueError('operator_requires_release')
                self._calibrate_recovered_host()
                if self.adapter.hardware_output:
                    if self.cfg.get('operator_session_enabled') is not True and not self._remote_control():
                        raise ValueError('operator_session_disabled')
                if not connected() or cancel.is_set():raise ValueError('operator_connection_changed')
                if not self._project_armed or self._project_stopping:raise ValueError('project_not_armed')
                if self.adapter.hardware_output:
                    self.link.call('prepare_operator_session',time.monotonic()+.5)
                    self._operator_session_prepared=True
                # project_stop can fence us while the management RPC is in
                # flight. A successful preparation is not a surviving start.
                if (cancel.is_set() or not connected() or not self._project_armed
                        or self._project_stopping):
                    if self._operator_session_prepared:
                        self.link.call('end_operator_session',time.monotonic()+.25)
                        self._operator_session_prepared=False
                    raise ValueError('operator_start_cancelled')
                self.runtime.prepare_local_session()
                self._return_required=True
                self._run(self.capture.issue_assignment_if_connected())
                result=self.info()
        elif action=='finish':
            result=self._finish_session(cancel,connected,stop_generation=stop_generation)
        else:raise ValueError('operator_action_invalid')
        if result.get('error'):raise ValueError(result['error'])
        return {k:result[k] for k in ('state','mode','return_completed','return_required','authority_released','max_error_rad') if k in result}

    def _release_driver(self):
        """Wait for actual release after the bounded stop request has returned."""
        if getattr(self.link,'preview_lease',None):
            if not self.link.release_preview(time.monotonic()+1.):return False
        if not self.link.lease and not getattr(self.link,'management_request',None):return True
        deadline=time.monotonic()+1.
        if self.adapter.close().ok:return True
        while time.monotonic()<deadline:
            if self.link.reconcile_release():return True
            time.sleep(min(.02,max(0.,deadline-time.monotonic())))
        return False

    def _recover_tianyi_host(self):
        """Explicit operator recovery; no container restart or acceptance bypass."""
        from .protocol import ProtocolError
        status=self.runtime.status()['dispatch']
        if status.get('io_inflight') or status.get('stop_queue_depth'):
            raise ValueError('recovery_waiting_for_adapter')
        try:self.runtime.release_local()
        except ProtocolError as exc:
            if exc.code!='dispatch_stop_unconfirmed':raise
        # A previous stop error alone is not evidence either way. Verify a
        # fresh, completed release before discarding any faulted component.
        if not self._release_driver():raise ValueError('stop_unconfirmed')
        state=self.link.feedback()
        preview_only=bool(getattr(self.adapter,'remote_motion_control',False) and not self.adapter.hardware_output)
        if (state.get('state')!='idle' or state.get('ownership_held') is not False
                or state.get('output_active') is not False
                or (preview_only and (state.get('preview') is not False or self.link.lease))
                or (not preview_only and state.get('stop_confirmed') is not True)):
            raise ValueError('recovery_requires_confirmed_idle')
        feedback=state.get('feedback',{})
        now=time.monotonic_ns()
        freshness=('arm_ns',) if preview_only else ('arm_ns','power_ns','fixed_ns')
        if (any(not 0<=now-feedback.get(k,0)<=100_000_000 for k in freshness)
                or (not preview_only and (feedback.get('power_on') is not True
                    or feedback.get('estop') is not False or feedback.get('fault') is not False))):
            raise ValueError('recovery_feedback_not_ready')
        self.stop()
        self._ensure_host()
        self._calibrate_recovered_host()

    def _calibrate_recovered_host(self):
        # A newly opened ROS subscription has no sample yet. Keep the new
        # session unarmed while waiting; a retry never requires a restart.
        deadline=time.monotonic()+1.
        while True:
            try:
                self.link.feedback()
                self.adapter.calibrate(self.cfg.get('calibration_path'))
                return
            except ValueError as exc:
                if str(exc) not in ('driver_feedback_missing','driver_feedback_stale_or_different_clock'):
                    raise
                if time.monotonic()>=deadline:raise ValueError('recovery_feedback_not_ready') from None
                time.sleep(.02)

    def stop(self):
        self._cancel_operations()
        with self._lock:
            self._closing=True
            errors=[]
            if self.recorder:self.recorder.stop()
            if self.runtime:
                try:self.runtime.close()
                except Exception as exc:errors.append(str(exc))
            if self.adapter and (self.link.lease or getattr(self.link,'management_request',None)):
                if not self._release_driver():
                    # Keep the link and lease available for a later stop retry.
                    self.error='stop_unconfirmed'
                    raise ValueError(self.error)
            if self._loop and self._loop.is_running():
                if self.server:
                    try:self._run(self.server.close())
                    except Exception as exc:errors.append(str(exc))
                if getattr(self,'rtc',None):
                    try:self._run(self.rtc.close_all())
                    except Exception as exc:errors.append(str(exc))
                async def cancel_tasks():
                    tasks=[t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
                    for task in tasks:task.cancel()
                    await asyncio.gather(*tasks,return_exceptions=True)
                self._run(cancel_tasks())
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join(timeout=2)
                self._loop.close()
            if self.link:self.link.close()
            self.runtime=self.adapter=self.link=self.capture=self.server=None
            self._loop=self._thread=None
            if errors:self.error=';'.join(errors)
