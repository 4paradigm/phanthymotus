"""Calibrated Tianyi fourteen-joint IK. No DDS or hardware output.

The Pinocchio objective follows the approach in PR152 (Apache-2.0),
with the Tianyi model, measured seed, calibrated palm frames and bounded output.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET
import numpy as np

# Match the Driver's existing stopped-position tolerance, without allowing an
# initial inward correction larger than one configured command step.
MEASURED_LIMIT_TOLERANCE_RAD = .02
POSITION_LEAD_SECONDS = .2

ARM_NAMES = tuple(f"{side}_{joint}_joint" for side in ("left", "right") for joint in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch",
    "wrist_yaw", "wrist_pitch", "wrist_roll"))


def finite(value, shape):
    raw=np.asarray(value)
    if raw.dtype.kind not in 'iuf':raise ValueError('invalid_numeric_type')
    x = np.asarray(value, dtype=float)
    if x.shape != shape or not np.isfinite(x).all():
        raise ValueError("invalid_finite_shape")
    return x


def transform(pose):
    p = finite(pose['position'], (3,))
    x,y,z,w = finite(pose['orientation'], (4,))
    if abs(x*x+y*y+z*z+w*w-1) > 0.002:
        raise ValueError('quaternion_not_unit')
    t = np.eye(4)
    t[:3,3] = p
    t[:3,:3] = [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
               [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
               [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    return t


class RelativeMapping:
    def __init__(self, scale=0.5):
        if not 0 < scale <= 1:
            raise ValueError('position_scale')
        self.scale = scale
        self.reference = None
        self.controller_offsets={s:np.eye(4) for s in ("left","right")}

    def reset(self, frame, measured_palms):
        # OpenXR +X right,+Y up,-Z forward -> robot +X forward,+Y left,+Z up.
        basis = np.array([[0,0,-1],[-1,0,0],[0,1,0]], dtype=float)
        head = basis @ transform(frame['head'])[:3,:3]
        forward = head @ np.array([0,0,-1.0])
        yaw = np.arctan2(forward[1], forward[0])
        c,s = np.cos(-yaw),np.sin(-yaw)
        self.rotation = np.array([[c,-s,0],[s,c,0],[0,0,1]]) @ basis
        self.reference = [(transform(frame[f'{side}_controller']) @ self.controller_offsets[side], finite(palm,(4,4)).copy())
                          for side,palm in zip(('left','right'),measured_palms)]

    def targets(self, frame):
        if self.reference is None:
            raise ValueError('mapping_not_calibrated')
        targets=[]
        for side,(origin,robot) in zip(('left','right'),self.reference):
            current=transform(frame[f'{side}_controller']) @ self.controller_offsets[side]
            target=robot.copy()
            target[:3,3] += self.scale * self.rotation @ (current[:3,3]-origin[:3,3])
            target[:3,:3] = self.rotation @ current[:3,:3] @ origin[:3,:3].T @ self.rotation.T @ robot[:3,:3]
            targets.append(target)
        return targets


from .workspace import ArmWorkspace, WorkspaceViolation
from .reachability import ReachableTarget
from .trajectory import JointTrajectory


def arm_chain_xml(model_bytes, torso, names):
    """Root FK at the calibrated torso; omit unrelated movable branches.

    Preserve fixed descendants (TCP and collision frames). Do not synthesize
    leg/head/finger joint values or silently lock an extra joint in an arm chain.
    """
    root = ET.fromstring(model_bytes)
    links = {link.get('name') for link in root.findall('link')}
    joints = {joint.get('name'): joint for joint in root.findall('joint')}
    if torso not in links:
        raise ValueError('torso_frame_missing')
    parents = {joint.find('child').get('link'): joint for joint in joints.values()}
    keep_links = {torso}
    keep_joints = set()
    for name in names:
        if name not in joints:
            raise ValueError('arm_joint_mapping')
        child = joints[name].find('child').get('link')
        visited = set()
        while child != torso:
            if child in visited or child not in parents:
                raise ValueError('arm_chain_not_under_torso')
            visited.add(child)
            joint = parents[child]
            if joint.get('type') != 'fixed' and joint.get('name') not in names:
                raise ValueError('unexpected_movable_arm_joint')
            keep_links.add(child)
            keep_joints.add(joint.get('name'))
            child = joint.find('parent').get('link')
    changed = True
    while changed:
        changed = False
        for name, joint in joints.items():
            if (joint.get('type') == 'fixed' and name not in keep_joints
                    and joint.find('parent').get('link') in keep_links):
                keep_joints.add(name)
                keep_links.add(joint.find('child').get('link'))
                changed = True
    for element in list(root):
        if (element.tag == 'link' and element.get('name') in keep_links
                or element.tag == 'joint' and element.get('name') in keep_joints):
            continue
        root.remove(element)
    return ET.tostring(root, encoding='unicode')


class TianyiIK(ArmWorkspace):
    def __init__(self, calibration_path):
        import pinocchio as pin
        from scipy.optimize import least_squares
        self.least_squares=least_squares
        self.pin = pin
        raw=Path(calibration_path).read_bytes()
        self.profile_sha256=hashlib.sha256(raw).hexdigest()
        self.profile = json.loads(raw)
        if self.profile.get('schema') != 'motus.tianyi-calibration.v1':
            raise ValueError('calibration_schema')
        self.hands_enabled = self.profile.get('hands_enabled', True)
        if type(self.hands_enabled) is not bool:
            raise ValueError('hands_enabled_boolean_required')
        urdf=Path(self.profile['urdf_path'])
        model_bytes=urdf.read_bytes()
        if hashlib.sha256(model_bytes).hexdigest()!=self.profile['urdf_sha256']:
            raise ValueError('calibration_model_changed')
        names=self.profile['arm_joint_names']
        if len(names)!=14 or len(set(names))!=14:raise ValueError('arm_joint_mapping')
        self.model=pin.buildModelFromXML(arm_chain_xml(model_bytes,self.profile['torso_frame'],names))
        if self.model.nq!=14 or set(self.model.names[1:])!=set(names):
            raise ValueError('tianyi_joint_order_mismatch')
        self.indices=np.array([self.model.joints[self.model.getJointId(n)].idx_q for n in names])
        self.velocity=self.profile.get('joint_velocity_rad_s',0.2)
        if type(self.velocity) not in (int,float) or not 0<self.velocity<=min(1.5,float(min(self.model.velocityLimit))):
            raise ValueError('joint_velocity_limit')
        self.frames=[]
        for side in ('left','right'):
            palm=transform(self.profile['palm_frames'][side])
            self.model.addFrame(pin.Frame(f'{side}_teleop_palm',self.model.getJointId(names[6 if side=='left' else 13]),
                               pin.SE3(palm[:3,:3],palm[:3,3]),pin.FrameType.OP_FRAME))
            self.frames.append(self.model.getFrameId(f'{side}_teleop_palm'))
        self.torso=self.model.getFrameId(self.profile['torso_frame'])
        if self.torso>=self.model.nframes:raise ValueError('torso_frame_missing')
        self.data=self.model.createData()
        self.lock=threading.RLock()
        self.axis_aware_sweep = True
        self.analytic_capsule_distance = True
        self.transition_refinement_depth = 10
        self.configure_workspace()
        self.last_ms=None
        self.visualization_sample=None
        self.last_valid_visualization=None
        self.target_policy = ReachableTarget(self.profile.get('target_projection'), pin)
        self.target_diagnostics = None
        self.trajectory = JointTrajectory(self.profile.get('trajectory_smoothing'), self.velocity)

    def reset_target_state(self):
        with self.lock:
            self.target_policy.reset()
            self.target_diagnostics = None
            self.trajectory.reset()

    def command_step(self, goal, measured, previous, budget, command_state=None):
        """Shared output path for live IK, replay and explicit arm return."""
        with self.lock:
            goal, measured, previous = (finite(q, (14,)) for q in (goal, measured, previous))
            lower = self.model.lowerPositionLimit[self.indices]
            upper = self.model.upperPositionLimit[self.indices]
            if self.trajectory.enabled:
                try:
                    q, lo, hi, plan = self.trajectory.propose(
                        goal, measured, previous, command_state, lower, upper, budget)
                    self._safe_transition(lo, hi, budget)
                    self._safe_transition(measured, q, budget)
                    budget()
                except Exception as exc:
                    self.trajectory.diagnostics = {'state': 'rejected', 'reason': str(exc)}
                    raise
                # Never clip a generated curve after its derivative checks.
                self.trajectory.commit(plan)
                self.last_advance_scale = 1.
                return q
            q = previous+np.clip(goal-previous, -self.velocity*.02, self.velocity*.02)
            q = np.clip(q, measured-self.velocity*POSITION_LEAD_SECONDS, measured+self.velocity*POSITION_LEAD_SECONDS)
            q = np.clip(q, lower, upper)
            q, self.last_advance_scale = self._safe_advance(measured, previous, q, budget)
            return q

    def _safe_advance(self, measured, previous, candidate, budget):
        """Shorten a fresh advance only after proving both entire joint boxes.

        The full IK reference stays unchanged. Collision margins, deadline,
        joint limits and the outstanding-travel bound are never relaxed.
        """
        measured,previous,candidate=(finite(v,(14,)) for v in (measured,previous,candidate))
        lower=self.model.lowerPositionLimit[self.indices];upper=self.model.upperPositionLimit[self.indices]
        lead=self.velocity*POSITION_LEAD_SECONDS
        failure=None
        for scale in (1.,.5,.25,.125,.0625):
            budget()
            trial=candidate.copy() if scale==1. else previous+scale*(candidate-previous)
            if (np.any(trial<lower) or np.any(trial>upper)
                    or np.any(trial<measured-lead) or np.any(trial>measured+lead)):
                continue
            try:
                self._safe_transition(measured,trial,budget)
                self._safe_transition(previous,trial,budget)
                budget()
            except WorkspaceViolation as exc:
                failure=exc
                continue
            return trial,scale
        raise failure if failure is not None else ValueError('joint_limit')

    def palms(self,q):
        with self.lock:
            model_q=np.empty(14);model_q[self.indices]=finite(q,(14,))
            self.pin.framesForwardKinematics(self.model,self.data,model_q)
            torso=self.data.oMf[self.torso].inverse()
            return [(torso*self.data.oMf[f]).homogeneous.copy() for f in self.frames]

    def _residual_jacobian(self, q, targets, measured):
        """Same position/log-rotation objective, differentiated in arm wire order."""
        model_q=np.empty(14);model_q[self.indices]=q
        self.pin.computeJointJacobians(self.model,self.data,model_q)
        self.pin.updateFramePlacements(self.model,self.data)
        torso=self.data.oMf[self.torso].inverse()
        residual=np.empty(26);jacobian=np.zeros((26,14))
        for index,(frame,target) in enumerate(zip(self.frames,targets)):
            actual=torso*self.data.oMf[frame]
            error=target[:3,:3].T@actual.rotation
            row=6*index
            residual[row:row+3]=7*(actual.translation-target[:3,3])
            residual[row+3:row+6]=self.pin.log3(error)
            spatial=self.pin.getFrameJacobian(self.model,self.data,frame,
                                              self.pin.LOCAL_WORLD_ALIGNED)[:,self.indices]
            jacobian[row:row+3]=7*torso.rotation@spatial[:3]
            # Jlog3 differentiates a right/local SO(3) perturbation.
            jacobian[row+3:row+6]=self.pin.Jlog3(error)@actual.rotation.T@torso.rotation@spatial[3:]
        residual[12:]=.01*(q-measured)
        jacobian[12:]=.01*np.eye(14)
        return residual,jacobian

    def solve(self,targets,measured,commanded=None,*,deadline_monotonic=None,command_state=None):
        with self.lock:
            targets=[finite(t,(4,4)) for t in targets]
            try:
                return self._solve(targets,measured,commanded,deadline_monotonic=deadline_monotonic,command_state=command_state)
            except Exception as exc:
                # Keep history only for rendering. Failed results never become commands.
                self.visualization_sample={'monotonic_ns':time.monotonic_ns(),
                    'targets':[t.copy() for t in targets], 'ik_q':None, 'error':str(exc)}
                self.target_diagnostics = {**(self.target_diagnostics or {}),
                    'state': 'rejected', 'error': str(exc),
                    'raw_targets': [t.tolist() for t in targets]}
                raise

    def _solve(self,targets,measured,commanded=None,*,deadline_monotonic=None,command_state=None):
        with self.lock:
            begin=time.monotonic();measured=finite(measured,(14,))
            self.target_diagnostics = None
            lower=self.model.lowerPositionLimit[self.indices];upper=self.model.upperPositionLimit[self.indices]
            measured_tolerance=min(MEASURED_LIMIT_TOLERANCE_RAD,self.velocity*.02)
            if np.any(measured<lower-measured_tolerance) or np.any(measured>upper+measured_tolerance):
                raise ValueError('joint_feedback_out_of_bounds')
            initial=np.clip(measured,lower,upper)
            # Leave time for adapter acknowledgement before the input TTL.
            # Input freshness (300 ms) and the P95 latency objective (100 ms)
            # are different contracts. A solve still has a fixed 100 ms bound.
            end = begin + .100
            if deadline_monotonic is not None:
                end = min(end, float(deadline_monotonic))
            def check_budget():
                if time.monotonic() >= end:
                    raise ValueError("ik_timeout")
            check_budget()
            targets=[finite(t,(4,4)) for t in targets]
            if len(targets)!=2:raise ValueError('dual_targets_required')
            raw_targets = targets
            targets, clipped = self.target_policy.targets(raw_targets, self.workspace)
            solve_lower, solve_upper = self.target_policy.bounds(lower, upper, measured)
            initial = np.clip(initial, solve_lower, solve_upper)
            previous_solution=self.last_valid_visualization
            if (previous_solution and 0<=time.monotonic_ns()-previous_solution['monotonic_ns']<=200_000_000
                    and all(np.linalg.norm(a[:3,3]-b[:3,3])<=.03
                            and np.linalg.norm(a[:3,:3]-b[:3,:3])<=.15
                            for a,b in zip(targets,previous_solution.get('feasible_targets', previous_solution['targets'])))):
                # Continuous small target changes can reuse the numerical seed.
                # Residual regularization, rate limits and swept geometry still
                # use CURRENT measured/commanded joints, never this old solution.
                initial=np.clip(previous_solution['ik_q'],solve_lower,solve_upper)
            cached_q=None;cached_value=None
            def evaluate(q):
                nonlocal cached_q,cached_value
                check_budget()
                if time.monotonic()-begin>0.085:raise ValueError('ik_timeout')
                if cached_q is None or not np.array_equal(q,cached_q):
                    cached_value=self._residual_jacobian(q,targets,measured)
                    cached_q=q.copy()
                check_budget()
                return cached_value
            result=self.least_squares(lambda q:evaluate(q)[0],initial,
                jac=lambda q:evaluate(q)[1],
                bounds=(solve_lower,solve_upper),
                max_nfev=30,ftol=1e-5,xtol=1e-5,gtol=1e-5)
            # Budget exhaustion (status 0) is not itself geometric failure.
            # Accept only a finite result passing the same residual, bounds,
            # collision, rate and time checks below; never relax these gates.
            if not result.success and result.status != 0:
                raise ValueError('ik_not_converged')
            check_budget()
            q=finite(result.x,(14,))
            actual=self.palms(q)
            check_budget()
            self.target_diagnostics = {'residuals': self.target_policy.residuals(actual, raw_targets),
                                       'solver_status': int(result.status)}
            if (not self.target_policy.enabled and any(
                    np.linalg.norm(a[:3,3]-b[:3,3])>0.015 or np.linalg.norm(a[:3,:3]-b[:3,:3])>0.15
                    for a,b in zip(actual,targets))):
                raise ValueError('ik_target_unreachable')
            clipped_fit = any(clipped) and all(
                r['position_m'] <= self.target_policy.POSITION_TOLERANCE and
                r['orientation_rad'] <= self.target_policy.ANGLE_TOLERANCE
                for r in self.target_policy.residuals(actual, targets))
            q, feasible_targets, boundary, diagnostics = self.target_policy.select(
                raw_targets, q, actual, clipped, result.success or clipped_fit)
            self.target_diagnostics.update(diagnostics)
            if self.target_policy.enabled:
                # A pose projection is intentional, not permission to use an
                # unchecked IK result. Collisions never become boundary points.
                self._safe_configuration(q)
                check_budget()
            # Immutable display snapshot is the full IK result, before rate limiting.
            preview_q=q.copy()
            # Slew from the last command while bounding outstanding physical travel
            # to 200 ms of configured speed. This is travel, not command TTL.
            # Validate the entire independently interpolated joint box.
            previous=measured if commanded is None else finite(commanded,(14,))
            # Preserve the validated fixed 20 ms command increment.
            q = self.command_step(q, measured, previous, check_budget, command_state)
            self.last_ms=(time.monotonic()-begin)*1000
            if self.last_ms>100:raise ValueError('ik_timeout')
            self.target_policy.boundary = boundary
            self.visualization_sample={'monotonic_ns':time.monotonic_ns(),
                'targets':[t.copy() for t in raw_targets], 'ik_q':preview_q,
                'feasible_targets':[t.copy() for t in feasible_targets],
                'target_state':diagnostics['state']}
            self.last_valid_visualization=self.visualization_sample
            return q.tolist()

    def self_test(self,measured):
        targets=self.palms(measured);result=self.solve(targets,measured)
        return {'state':'ready','hardware_output':False,'ik_ms':self.last_ms,'max_error_rad':float(np.max(np.abs(np.asarray(result)-measured)))}
