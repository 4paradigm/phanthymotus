"""Generic end-effector producer. Robot models and solving live in the Driver.

The existing paired-input dispatcher still owns grip and tracking admission.
This adapter maps poses, never imports a robot model and never emits joints.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import time
from scipy.spatial.transform import Rotation
from .adapter import DriverLink
from .descriptor import CAPABILITIES
from .dispatch import AdapterAck
from .mapping import finite, transform
from .tianyi import TianyiIntentAdapter

SCHEMA = 'motus.control/2'
CAPABILITIES_EEF = copy.deepcopy(CAPABILITIES)
CAPABILITIES_EEF.update(profile_id='relative_end_effectors_v1',
    outputs={'end_effectors': {'enabled': True, 'mode': 'eef_pose'}},
    effectors=['end_effectors'])


def pose_matrix(values):
    values = finite(values, (7,))
    return transform({'position': values[:3], 'orientation': values[3:]})


def pose_values(matrix):
    matrix = finite(matrix, (4, 4))
    return [*matrix[:3, 3].tolist(), *Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist()]


class MotionControlLink(DriverLink):
    tool = 'motion_control'

    def __init__(self, cfg, executor):
        super().__init__(cfg, executor)
        self.command_topic = f"/{cfg.get('namespace', 'robot')}/motion/control/command"
        self.preview_lease = None

    def call(self, action, deadline, **arguments):
        credentials = self.lease or self.preview_lease
        if credentials and action in ('finish', 'finish_status', 'pause', 'release', 'resume'):
            arguments = {**{k: credentials[k] for k in ('session_id', 'secret')}, **arguments}
        return super().call(action, deadline, **arguments)

    def prepare_preview(self, deadline):
        if self.lease:raise ValueError('preview_while_owned')
        value = self.call('prepare_preview', deadline)
        if value.get('preview') is not True or not all(value.get(k) for k in ('boot_id', 'session_id', 'secret')):
            raise ValueError('driver_invalid_preview')
        self.preview_lease = value
        self.seq = 0
        self.prepare_transport()
        return value

    def release_preview(self, deadline):
        """Retire the authenticated preview; never infer a hardware stop."""
        token = self.preview_lease
        if token is None:return True
        if self.lease:raise ValueError('preview_while_owned')
        stamp = time.monotonic_ns()
        try:
            self.call('release', deadline)
        except (OSError, ValueError) as exc:
            # A lost reply or an already retired identity still needs fresh
            # Driver evidence. Other refusals must not erase the local token.
            if isinstance(exc, ValueError) and str(exc) != 'invalid_lease':raise
        with self.condition:
            while time.monotonic() < deadline:
                try:
                    state = self.feedback()
                except ValueError:
                    state = {}
                if (state.get('monotonic_ns', 0) >= stamp
                        and state.get('boot_id') == token['boot_id']
                        and state.get('preview') is False and state.get('state') == 'idle'
                        and state.get('ownership_held') is False
                        and state.get('output_active') is False):
                    self.preview_lease = None
                    return True
                self.condition.wait(min(.01, max(0., deadline-time.monotonic())))
        return False

    def send_poses(self, values, snapshot, intent, epoch, deadline, *, live):
        from std_msgs.msg import String
        token = self.lease if live else self.preview_lease
        if not token:raise ValueError('driver_not_owned' if live else 'preview_not_prepared')
        now = time.monotonic_ns()
        until = min(int(deadline * 1e9), now + 300_000_000)
        if until <= now:raise ValueError('motion_deadline')
        self.seq += 1
        body = {'schema': SCHEMA, 'boot_id': token['boot_id'], 'session_id': token['session_id'],
            'seq': self.seq, 'source_seq': intent.sequence, 'mapping_epoch': epoch,
            'generated_ns': now, 'valid_until_ns': until,
            'mode': 'eef_pose', 'dof': len(values), 'values': values,
            'model_version': snapshot['model_version'], 'calibration_version': snapshot['calibration_version'],
            'frame': snapshot['frame']}
        canonical = json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        body['mac'] = hmac.new(bytes.fromhex(token['secret']), canonical, hashlib.sha256).hexdigest()
        message = String()
        message.data = json.dumps(body, allow_nan=False)
        if time.monotonic_ns() >= until:raise ValueError('motion_deadline')
        self.publisher.publish(message)
        self.last_send = {'sequence': self.seq, 'source_seq': intent.sequence,
            'mapping_epoch': epoch, 'generated_ns': now, 'deadline_ns': until,
            'published_ns': time.monotonic_ns(), 'mode': 'eef_pose'}
        return True

    def close(self):
        try:
            if self.preview_lease:
                self.call('release', time.monotonic()+.2)
        finally:
            # Preview cannot hold a hardware lease. A failed management reply
            # must not leak the local ROS node or block closing the host.
            self.preview_lease = None
            super().close()


class EefIntentAdapter(TianyiIntentAdapter):
    """Keep grip lifecycle, but replace robot-specific solving with eef output."""
    remote_motion_control = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calibration = None
        self.mapping_snapshot = None
        self.mapping_epoch = 0
        self._clutch_pending = False

    @property
    def calibrated(self):
        return self.calibration is not None

    def calibrate(self, path=None):
        with self.lock:
            if self.link.lease:raise ValueError('calibration_while_owned')
            result = self.link.call('calibrate', time.monotonic()+3.)
            if result.get('calibrated') is not True:raise ValueError('driver_calibration_missing')
            for key in ('model_version', 'calibration_version', 'frame'):
                if not isinstance(result.get(key), str) or not result[key]:
                    raise ValueError('driver_calibration_invalid')
            if result.get('effector_ids') != ['left', 'right']:
                raise ValueError('input_effector_binding_mismatch')
            self.calibration = result
            self.clutch = None
            self.mapping_snapshot = None
            self.link.prepare_transport()
            if not self.hardware_output:self.link.prepare_preview(time.monotonic()+.5)
            return {'state': 'ready', 'calibrated': True,
                **{key: result[key] for key in ('model_version', 'calibration_version', 'frame', 'effector_ids')}}

    def _snapshot(self, state):
        snap = state.get('eef_snapshot')
        if not isinstance(snap, dict):raise ValueError('feedback_unavailable')
        if not 0 <= time.monotonic_ns()-snap.get('monotonic_ns', 0) <= 100_000_000:
            raise ValueError('feedback_unavailable')
        for key in ('model_version', 'calibration_version', 'frame'):
            if snap.get(key) != self.calibration[key]:raise ValueError('driver_calibration_mismatch')
        if not isinstance(snap.get('poses'), list) or len(snap['poses']) != 2:
            raise ValueError('driver_eef_snapshot_invalid')
        return snap, [pose_matrix(pose) for pose in snap['poses']]

    def apply(self, intent):
        started = time.monotonic()
        published_before = self.link.last_send
        deadline = min(intent.expires_monotonic,
            (intent.received_monotonic or intent.admitted_monotonic)+self.input_timeout_ms/1000.)
        with self.lock:
            try:
                if intent.dispatch_generation <= self._revoked_generation:raise ValueError('dispatch_revoked')
                if not self.calibrated:raise ValueError('calibration_missing')
                state = self.link.feedback()
                policy = state.get('timing_policy', {})
                self._recoverable_hold_supported = policy.get('recoverable_hold') is True
                self.link.management_retry = policy.get('management_retry_ms') == 300
                if state.get('state') == 'fault':raise ValueError('driver_fault')
                if (self.hardware_output and self.link.lease
                        and state.get('monotonic_ns', 0) >= getattr(self.link,'lease_started_ns',2**63)):
                    if state.get('boot_id') != self.link.lease['boot_id']:raise ValueError('driver_restarted')
                    if state.get('session_id') != self.link.lease['session_id']:raise ValueError('driver_lease_lost')
                snap, palms = self._snapshot(state)
                clutch = (intent.session_generation, intent.clutch_sequence)
                if clutch != self.clutch:
                    self.mapper.reset(intent.frame, palms)
                    self.mapping_snapshot = copy.deepcopy(snap)
                    self.mapping_epoch += 1
                    self.clutch = clutch
                    self._clutch_pending = True
                    self._automatic_resume = False
                    # Calibrate on actual measured state; never reuse the prior
                    # clutch's solution or assume a claim reply is execution.
                if self._clutch_pending:
                    if self.hardware_output:
                        (self.link.resume if self.link.lease else self.link.claim)(deadline)
                    else:
                        self.link.prepare_preview(deadline)
                    self._clutch_pending = False
                    self.output = {'state': 'armed_waiting_input', 'output_active': False}
                    return AdapterAck(True)
                if self._automatic_resume:
                    # Transient input/feedback stops preserve the mapping.
                    # Rotate only the execution/preview identity, consume no
                    # target during management, and revalidate the next frame.
                    if self.hardware_output:
                        (self.link.resume if self.link.lease else self.link.claim)(deadline)
                    else:
                        self.link.prepare_preview(deadline)
                    self._automatic_resume = False
                    self.output = {'state': 'armed_waiting_input', 'output_active': False}
                    return AdapterAck(True)
                if self.hardware_output and not self.link.lease:
                    self.link.claim(deadline)
                    return AdapterAck(True, 'waiting_driver_management')
                if self.hardware_output and state.get('state') == 'hold':
                    if not state.get('continuation_ready'):
                        self.output = {'state': 'waiting_driver_hold', 'code': state.get('reason'), 'output_active': False}
                        return AdapterAck(True, 'waiting_driver_hold')
                targets = self.mapper.targets(intent.frame)
                values = [v for target in targets for v in pose_values(target)]
                if intent.dispatch_generation <= self._revoked_generation:raise ValueError('dispatch_revoked')
                self.link.send_poses(values, self.mapping_snapshot, intent, self.mapping_epoch, deadline,
                    live=self.hardware_output)
                self.output = {'state': 'submitted' if self.hardware_output else 'preview_submitted',
                    'target_poses': values, 'mapping_epoch': self.mapping_epoch,
                    'source_seq': intent.sequence, 'output_active': state.get('output_active', False)}
                self.chain_times.append((time.monotonic()-started)*1000)
                return AdapterAck(True)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                code = str(exc)
                self.last_failure = {'sequence': intent.sequence, 'phase': 'eef_mapping', 'code': code,
                    'monotonic_ns': time.monotonic_ns()}
                self.output = {'state': 'error', 'code': code, 'output_active': False}
                if code in ('driver_management_pending', 'driver_lease_rebased'):
                    if code == 'driver_lease_rebased' and not self._automatic_resume:self.clutch = None
                    return AdapterAck(True, 'waiting_driver_management')
                if code in ('driver_feedback_missing', 'driver_feedback_stale_or_different_clock'):
                    code = 'feedback_unavailable'
                return AdapterAck(False, code if code == 'feedback_unavailable' else 'motion_rejected')
            finally:
                if self.trace_sink is not None:
                    self.trace_sink({'event':'actucore_eef','monotonic_ns':time.monotonic_ns(),
                        'input_sequence':intent.sequence,'mapping_epoch':self.mapping_epoch,
                        'input_received_ns':int((intent.received_monotonic or intent.admitted_monotonic)*1e9),
                        'elapsed_ms':(time.monotonic()-started)*1000,
                        'published':self.link.last_send is not published_before,
                        'publish':copy.deepcopy(self.link.last_send) if self.link.last_send is not published_before else None,
                        'target_poses':self.output.get('target_poses'),
                        'output_state':self.output.get('state'),'reason':self.output.get('code')})

    def snapshot(self):
        with self.lock:
            output = copy.deepcopy(self.output)
            try:
                state = self.link.feedback()
                output.update(driver=state, output_active=state.get('output_active', False),
                              publisher_present=state.get('publisher_present', False))
            except ValueError:
                output.update(driver_feedback_fresh=False, output_active=None if self.hardware_output else False)
            return {'output': output, 'diagnostics': {'calibrated': self.calibrated,
                'calibration_version': (self.calibration or {}).get('calibration_version'),
                'last_failure': self.last_failure, 'last_send': self.link.last_send}}

    def safe_stop(self, request):
        result = super().safe_stop(request)
        if result.ok and not self.hardware_output and self.link.preview_lease:
            try:
                self.link.call('pause', request.deadline_monotonic)
            except (ValueError, OSError):
                # Preview is incapable of hardware output. Fencing locally and
                # making a new preview token on recovery discards late results.
                # Keep this identity for a subsequent explicit release retry.
                pass
        return result

    def visualization(self):
        try:
            value = copy.deepcopy(self.link.feedback().get('visualization'))
            if not isinstance(value, dict):raise ValueError('visualization_unavailable')
            value['mode'] = 'live' if self.hardware_output else 'shadow'
            return value
        except ValueError:
            return {'schema': 'motus.tianyi-visualization.v1', 'available': False,
                    'reason': 'feedback_unavailable'}

    def finish(self, cancel, connected, timeout=45.):
        """Request a Driver-owned return; no local interpolation or model."""
        deadline = time.monotonic()+timeout
        if not self.hardware_output:
            if not self.link.release_preview(deadline):raise ValueError('preview_release_unconfirmed')
            return {'state': 'idle', 'return_completed': True, 'authority_released': True}
        request = self.link.call('finish', min(deadline, time.monotonic()+.5))
        operation_id = request.get('operation_id')
        while time.monotonic() < deadline:
            if cancel.is_set() or not connected():
                self._confirm_stop(self.link.stop, time.monotonic()+self.stop_confirmation_timeout_ms/1000.)
                raise ValueError('return_cancelled' if cancel.is_set() else 'return_connection_lost')
            state = self.link.call('finish_status', min(deadline, time.monotonic()+.5), operation_id=operation_id)
            if state.get('error') or state.get('state') in ('error', 'fault'):
                raise ValueError(state.get('error') or state.get('reason') or 'return_failed')
            if state.get('return_completed') and state.get('authority_released'):
                self.link.lease = None
                return state
            time.sleep(.05)
        self._confirm_stop(self.link.stop, time.monotonic()+self.stop_confirmation_timeout_ms/1000.)
        raise ValueError('return_timeout')
