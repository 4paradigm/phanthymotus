"""G1 model/mapping adapter. No vendor SDK or hardware publisher in ActuCore."""
import copy
import hashlib
from itertools import product
import json
from pathlib import Path
import time
import numpy as np
from .adapter import IntentAdapter
from .descriptor import CAPABILITIES
from .dispatch import AdapterAck, IK_HOLD_CODES, COLLISION_HOLD_CODES
from .g1_ik import G123PinocchioIk, ARM_JOINT_NAMES
from .g1_collision import G1Collision
from .g1_mapping import G1ControllerPoseMapper, G1ClutchRelativeMapper
from .kinematics import finite, transform
from .workspace import ArmWorkspace

PROFILE_ID = 'unitree_g1_23_dual_arm_relative_v1'
CAPABILITIES_G1 = copy.deepcopy(CAPABILITIES)
CAPABILITIES_G1.update(profile_id=PROFILE_ID, effectors=['dual_arm'])
CAPABILITIES_G1['outputs']['dual_arm']['joint_count'] = 10
CAPABILITIES_G1['outputs']['hands']['enabled'] = False


class G1IK(ArmWorkspace):
    def __init__(self, path, *, pr152_objective=False):
        raw = Path(path).read_bytes()
        self.profile = json.loads(raw)
        self.profile_sha256 = hashlib.sha256(raw).hexdigest()
        p = self.profile
        if p.get('schema') != 'motus.g1-calibration.v1' or p.get('profile_id') != PROFILE_ID:
            raise ValueError('g1_calibration_schema')
        if p.get('arm_joint_names') != list(ARM_JOINT_NAMES):
            raise ValueError('g1_joint_order')
        model_path = Path(p['urdf_path'])
        if not model_path.is_absolute(): model_path = Path(path).parent / model_path
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != p['urdf_sha256']:
            raise ValueError('calibration_model_changed')
        self.ik = G123PinocchioIk(model_path, palm_frames=p['palm_frames'], locked_joints=p['locked_joints'], pr152_objective=pr152_objective)
        self.pin, self.model = self.ik._pin, self.ik._model
        self.data = self.model.createData()
        self.indices = np.arange(10)
        self.frames = [self.model.getFrameId(n) for n in ('L_ee','R_ee')]
        self.torso = self.model.getFrameId(p['torso_frame'])
        if self.torso >= self.model.nframes: raise ValueError('torso_frame_missing')
        self.velocity = p['joint_velocity_rad_s']
        if type(self.velocity) not in (int,float) or not 0 < self.velocity <= 5.:
            raise ValueError('joint_velocity_limit')
        if not p['workspace'].get('capsules') or not p['workspace'].get('torso_box'):
            raise ValueError('collision_calibration_missing')
        # Official G1_23 rubber-hand mesh bounds, rounded outwards (see VALIDATION.md).
        # Bind these dimensions to the verified model, not an arbitrary replacement URDF.
        if p['urdf_sha256'] not in (
            'b1af86fb023c0b6f8e52723d224be6cad70916eaff2778e7dbe09e6f91faa9b9',
            '2d61264aaae95eac2545497a34ef316f270b0cd3d856bf9a4941ebff2592aca2',
        ):
            raise ValueError('g1_hand_geometry_unverified')
        for side, ee in (('left','L_ee'), ('right','R_ee')):
            y = (-.042408,.029961) if side == 'left' else (-.029961,.042408)
            corners = np.array(list(product((.000097,.253316),y,(-.041527,.064957))))
            tip = finite(p['palm_frames'][side]['position'],(3,))
            length2 = float(tip @ tip)
            projection = np.zeros(8) if length2 == 0 else np.clip(corners @ tip / length2,0,1)
            radius = float(np.max(np.linalg.norm(corners-projection[:,None]*tip,axis=1)))
            # A capsule is convex: enclosing all box corners encloses the complete mesh.
            p['workspace']['capsules'].append({'from':side+'_wrist_roll_joint',
                'to':ee,'radius_m':radius,'group':side})
        self.configure_workspace()
        self.body_collision = G1Collision(self.pin,self.model,model_path)
        self.last_collision_rejection = None
        self._collision_context = None
        self.last_ms = None
        self.residual = None
        self.last_solve_at = None
        self.visualization_sample = None

    def _safe_configuration(self,q,excursion=None):
        try:
            super()._safe_configuration(q,excursion)
            self.body_collision.check(self.data,excursion)
        except ValueError as exc:
            self.last_collision_rejection = {
                'reason':str(exc), 'checked_q':np.asarray(q).tolist(),
                'excursion_rad':None if excursion is None else np.asarray(excursion).tolist(),
                'context':copy.deepcopy(self._collision_context),
                'body':copy.deepcopy(self.body_collision.last_rejection) if str(exc).startswith('g1_body_collision:') else None}
            raise

    def palms(self, q):
        return self.ik.current_targets(finite(q,(10,)))

    def self_test(self, q):
        q = finite(q,(10,))
        self._safe_configuration(q)
        result = self.ik.warm_up(q, np.zeros(10))
        self.ik.reset(q)
        return {'state':'ready','hardware_output':False, **result, 'profile_id':PROFILE_ID,
                'calibration_sha256':self.profile_sha256}

    def solve(self, targets, measured, commanded=None, dq=None):
        start = time.monotonic()
        q0 = finite(measured,(10,))
        # Keep the last complete frame visible while the next solve is running.
        # Publish matched targets + IK in one reference replacement.
        visual_stamp = time.monotonic_ns()
        q, _ = self.ik.solve(*targets, q0, finite(dq,(10,)))
        self.visualization_sample = {"monotonic_ns":visual_stamp,
                                     "targets":[np.asarray(t).tolist() for t in targets],
                                     "ik_q":np.asarray(q).tolist()}
        actual = self.palms(q)
        self.residual = [{'position_m':float(np.linalg.norm(a[:3,3]-b[:3,3])),
                          'orientation_rad':float(np.linalg.norm(self.pin.log3(b[:3,:3].T @ a[:3,:3])))}
                         for a,b in zip(actual,targets)]
        # Pose residual is diagnostic in PR152 mode; legacy relative mode
        # retains the strict raw-solution position check in self.ik.
        reference = q0 if commanded is None else finite(commanded,(10,))
        elapsed = .02 if self.last_solve_at is None else start-self.last_solve_at
        # Do not accumulate movement credit across a HOLD or stale input gap.
        elapsed = elapsed if 0 < elapsed <= .1 else .02
        delta = q-reference
        fraction = 1/max(1.,float(np.max(np.abs(delta)))/(self.velocity*elapsed))
        q = reference + fraction*delta
        self._collision_context = {'measured_q':q0.tolist(),'reference_q':reference.tolist(),'target_q':q.tolist()}
        for previous in ([q0] if commanded is None else [q0,finite(commanded,(10,))]):
            excursion = np.abs(q-previous)
            self._safe_configuration(previous,excursion)
        self._safe_configuration(q)
        # Compute the same PR152 gravity compensation at the bounded target.
        tau = self.pin.rnea(self.model,self.data,q,np.zeros(10),np.zeros(10))
        self.last_ms = (time.monotonic()-start)*1000
        if self.last_ms > 80: raise ValueError('ik_timeout')
        self.last_solve_at = start
        return q.tolist(),finite(tau,(10,)).tolist()


class G1IntentAdapter(IntentAdapter):
    auto_collision_recovery = True
    def __init__(self,*args,mapping_version="relative_v1",**kwargs):
        super().__init__(*args,**kwargs)
        if mapping_version not in ("relative_v1","pr152_head_yaw_v1","pr152_clutch_relative_v1"):
            raise ValueError("g1_mapping_version")
        if mapping_version != "relative_v1" and self.mapper.scale != 1:
            raise ValueError("g1_head_yaw_requires_unit_scale")
        self.mapping_version=mapping_version
        self.head_mapper=G1ControllerPoseMapper() if mapping_version == "pr152_head_yaw_v1" else None
        if mapping_version == "pr152_clutch_relative_v1": self.head_mapper=G1ClutchRelativeMapper()
        self._collision_resume = False
        self._feedback_signal={'generation':0,'reason':None,'acknowledged':False}

    def external_release_signal(self):
        if not self.hardware_output or not self.link.lease:return dict(self._feedback_signal)
        try:state=self.link.feedback()
        except ValueError:return {**self._feedback_signal,'acknowledged':False}
        if state.get('boot_id')!=self.link.lease.get('boot_id') or state.get('session_id')!=self.link.lease.get('session_id'):
            return {**self._feedback_signal,'acknowledged':False}
        if self._collision_resume and state.get('state')=='hold' and state.get('reason')=='operator_pause':
            return {'generation':0,'reason':None,'acknowledged':False}
        generation=state.get('feedback_hold_generation',0)
        if generation>self._feedback_signal['generation']:
            self._feedback_signal={'generation':generation,'reason':state.get('reason'),'acknowledged':False}
        if self._feedback_signal['generation']:
            self._feedback_signal['acknowledged']=state.get('state')=='hold' and state.get('hold_confirmed') is True
        return dict(self._feedback_signal)

    def external_fault_code(self):
        if not self.hardware_output or not self.link.lease:return None
        try:state=self.link.feedback()
        except ValueError:return None
        return 'driver_fault' if state.get('state')=='fault' else None

    def snapshot(self):
        with self.lock:
            result = super().snapshot()
            if self.hardware_output:
                result['output']['receive_to_enqueue_p95_ms'] = result['output']['receive_to_driver_p95_ms']
                result['output']['receive_to_driver_p95_ms'] = None
            result['diagnostics']['mapping_version'] = self.mapping_version
            result['diagnostics']['last_collision_rejection'] = copy.deepcopy(self.solver.last_collision_rejection) if self.solver else None
            result['diagnostics']['ik'] = self.solver.ik.snapshot() if self.solver else None
            return result

    def calibrate(self,path):
        with self.lock:
            reconcile=getattr(self.link,'reconcile_release',None)
            if reconcile:reconcile()
            if self.link.lease: raise ValueError('calibration_while_owned')
            profile=json.loads(Path(path).read_bytes())
            waist=profile['safety'].get('waist_reference','acquisition')
            if waist not in ('acquisition','torso') or (waist=='torso' and self.mapping_version!='pr152_clutch_relative_v1'):
                raise ValueError('waist_mapping_mismatch')
            solver = G1IK(path, pr152_objective=self.head_mapper is not None)
            state = self.link.feedback()
            if state.get('profile_id') != PROFILE_ID: raise ValueError('driver_profile_mismatch')
            result = solver.self_test(state['feedback']['q'])
            self.mapper.controller_offsets = {side:transform(solver.profile['controller_to_palm'][side]) for side in ('left','right')}
            self.solver = solver
            self.clutch = None
            return result

    def apply(self,intent):
        deadline = min(intent.expires_monotonic,(intent.received_monotonic or intent.admitted_monotonic)+.1)
        with self.lock:
            try:
                if intent.dispatch_generation <= self._revoked_generation: raise ValueError('dispatch_revoked')
                if self.solver is None: raise ValueError('calibration_missing')
                state = self.link.feedback()
                if state.get('profile_id') != PROFILE_ID: raise ValueError('driver_profile_mismatch')
                if (state.get('state')=='hold' and state.get('reason') in {'feedback_unavailable','base_not_stationary'}
                        and state.get('monotonic_ns',0)>=getattr(self.link,'lease_started_ns',0)):
                    if not state.get('hold_confirmed') or (intent.session_generation,intent.clutch_sequence)==self.clutch:
                        raise ValueError(state['reason'])
                feedback = state['feedback']; q = finite(feedback['q'],(10,))
                if not 0 <= time.monotonic_ns()-feedback['arm_ns'] <= (100_000_000 if self.hardware_output else getattr(self.link,'feedback_max_age_ns',100_000_000)): raise ValueError('arm_feedback_stale')
                if self.hardware_output and state.get('calibration_sha256') != self.solver.profile_sha256: raise ValueError('driver_calibration_mismatch')
                clutch = (intent.session_generation,intent.clutch_sequence)
                fresh_clutch = clutch != self.clutch
                if fresh_clutch:
                    self.solver.ik.reset(q)
                    self.solver.last_solve_at = None
                    self.mapper.reset(intent.frame,self.solver.palms(q))
                    if isinstance(self.head_mapper,G1ClutchRelativeMapper):
                        self.head_mapper.reset(intent.frame,self.solver.palms(q))
                if self.hardware_output and fresh_clutch:
                    # Acquisition is a management operation, not a motion frame.
                    # Do not spend this frame's TTL on IK plus publisher discovery.
                    self.solver._safe_configuration(q)
                    if time.monotonic() >= deadline: raise ValueError('ik_motion_deadline')
                    if self.link.lease:
                        try:
                            self.link.resume(deadline)
                        except ValueError as exc:
                            # A fresh resume may observe settling after an
                            # earlier HOLD acknowledgement. Keep HOLD and
                            # require regrip; do not turn this refusal into a
                            # terminal fault or send a target in this frame.
                            if str(exc) == 'arms_not_stationary':
                                self.output = {'state':'error','code':str(exc),'output_active':False,'publisher_present':True}
                                return AdapterAck(False,str(exc))
                            raise
                    try:
                        self.link.claim(deadline)
                    except ValueError as exc:
                        if str(exc) == 'arms_not_stationary' and self.link.lease is None:
                            self.output = {'state':'error','code':str(exc),'output_active':False,'publisher_present':False}
                            return AdapterAck(False,str(exc))
                        raise
                    if time.monotonic() >= deadline: raise ValueError('motion_deadline')
                    if intent.dispatch_generation <= self._revoked_generation: raise ValueError('dispatch_revoked')
                    self.clutch = clutch
                    self.output = {'state':'armed_waiting_input','output_active':False,'publisher_present':True}
                    return AdapterAck(True)
                targets = self.head_mapper.map_frame(intent.frame) if self.head_mapper else self.mapper.targets(intent.frame)
                target,tau = self.solver.solve(targets,q,commanded=state.get('commanded_q'),dq=feedback['dq'])
                self.ik_times.append(self.solver.last_ms)
                if time.monotonic() >= deadline: raise ValueError('ik_motion_deadline')
                if intent.dispatch_generation <= self._revoked_generation: raise ValueError('dispatch_revoked')
                if self._collision_resume:
                    if self.hardware_output and self.link.lease:
                        # Only resume after the same anchored target passes all checks.
                        self.link.resume(deadline)
                        self._collision_resume = False
                        self.output = {'state':'armed_waiting_input','output_active':False,'publisher_present':True}
                        return AdapterAck(True)
                    self._collision_resume = False
                if self.hardware_output:
                    self.link.send(target,None,deadline,wait_for_execution=False,profile_id=PROFILE_ID,joint_names=list(ARM_JOINT_NAMES),tau_ff=tau)
                self.clutch = clutch
                self.output = {'state':'submitted' if self.hardware_output else 'would_apply',
                               'target_q':target,'desired_q':list(self.solver.visualization_sample['ik_q']),
                               'tau_ff':tau,'residual':self.solver.residual,
                               'output_active':False if not self.hardware_output else None,'publisher_present':False if not self.hardware_output else None}
                self.chain_times.append((time.monotonic()-(intent.received_monotonic or intent.admitted_monotonic))*1000)
                return AdapterAck(True)
            except Exception as exc:
                self.output = {'state':'error','code':str(exc),'output_active':False}
                code=str(exc).split(':',1)[0]
                if code in COLLISION_HOLD_CODES:
                    self.clutch = (intent.session_generation,intent.clutch_sequence)
                if code in {'arm_feedback_stale','lowstate_stale','motion_feedback_stale','odometry_stale','odometry_motion_history_missing'}:
                    code='feedback_unavailable'
                if code=='driver_holding':
                    try:
                        current=self.link.feedback()
                        if current.get('state')=='hold' and current.get('reason') in {'feedback_unavailable','base_not_stationary','command_timeout'}:
                            code=current['reason']
                    except ValueError:pass
                if isinstance(exc,ValueError) and code in IK_HOLD_CODES | {'feedback_unavailable','base_not_stationary','driver_execution_ack_timeout','command_timeout'}:
                    return AdapterAck(False,code)
                return AdapterAck(False,'motion_rejected')

    def safe_stop(self,request):
        self._revoked_generation = max(self._revoked_generation,request.dispatch_generation-1)
        if not self.lock.acquire(timeout=max(0,request.deadline_monotonic-time.monotonic())):
            return AdapterAck(False,'stop_unconfirmed')
        try:
            # All input interruptions hold. Explicit card stop/close releases below.
            if self.hardware_output and self.link.lease:
                state=self.link.feedback()
                if request.reason in {'feedback_unavailable','base_not_stationary'} and state.get('state')!='hold':
                    state=self.link.call('pause',request.deadline_monotonic)
                if state.get('state')=='hold' and state.get('reason') in {'feedback_unavailable','base_not_stationary'}:
                    self.clutch=None
                    self._collision_resume=False
                    return AdapterAck(bool(state.get('hold_confirmed')),'feedback_wait' if not state.get('hold_confirmed') else 'stop_confirmed')
            requested_ns = time.monotonic_ns()
            ok = self.link.pause(request.deadline_monotonic) if self.hardware_output else True
            self._collision_resume = bool(ok and request.reason in COLLISION_HOLD_CODES)
            if not self._collision_resume:
                self.clutch = None
            if not ok and self.hardware_output and self.link.lease:
                # A matching post-request HOLD is an accepted stop awaiting
                # physical rest, not permission to move or a terminal fault.
                state = self.link.feedback()
                if (state.get('boot_id') == self.link.lease.get('boot_id')
                        and state.get('session_id') == self.link.lease.get('session_id')
                        and state.get('monotonic_ns',0) >= requested_ns
                        and state.get('state') == 'hold'
                        and state.get('reason') == 'operator_pause'
                        and state.get('feedback_hold_generation',0) > 0):
                    return AdapterAck(False,'feedback_wait')
            return AdapterAck(ok,'stop_confirmed' if ok else 'stop_unconfirmed')
        except Exception as exc:
            self.output = {'state':'stop_unconfirmed','code':str(exc),'output_active':None}
            return AdapterAck(False,'stop_unconfirmed')
        finally: self.lock.release()

    def close(self):
        try:
            return AdapterAck(self.link.stop(time.monotonic()+.2) if self.hardware_output else True,'stop_unconfirmed')
        except Exception:
            return AdapterAck(False,'stop_unconfirmed')
