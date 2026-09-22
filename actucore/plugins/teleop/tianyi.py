"""Tianyi clutch/hold lifecycle; robot-side execution stays in the Driver."""
import time
from .adapter import IntentAdapter
from .dispatch import AdapterAck, IK_HOLD_CODES, COLLISION_HOLD_CODES, IK_RETRY_CODES


class TianyiIntentAdapter(IntentAdapter):
    # The hold command is immediate; physical settling/receipt has its own
    # budget and must not inherit the 100ms motion-frame deadline.
    stop_confirmation_timeout_ms = 2500
    input_timeout_ms = 300
    dispatch_io_timeout_ms = 150
    auto_collision_recovery = True
    auto_ik_recovery = True
    auto_workspace_recovery = True
    auto_shadow_feedback_recovery = True
    auto_live_transient_recovery = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._automatic_resume = False
        self._resume_validated = False
        self._same_session_hold = False
        self._recoverable_hold_supported = False
        self.trace_sink = None
        self.result_observer = None
        self.recording_errors = 0
        self.last_failure = None
        self.last_apply = None
        self._feedback_wait_started = None
        self._continuation_supported = False
        self._feedback_wait_seconds = .3

    def snapshot(self):
        with self.lock:
            value = super().snapshot()
            value["diagnostics"]["last_failure"] = (dict(self.last_failure) if self.last_failure else None)
            value["diagnostics"]["last_send"] = getattr(self.link, "last_send", None)
            value["diagnostics"]["last_apply"] = self.last_apply
            value["diagnostics"]["recording_errors"] = self.recording_errors
            value["diagnostics"]["transport_prepare_ms"] = getattr(self.link, "transport_prepare_ms", None)
            return value

    def calibrate(self, path):
        with self.lock:
            result = super().calibrate(path)
            if self.hardware_output:
                # ROS entity discovery can exceed a pose frame's budget. Do it
                # at explicit zero-output calibration, before grips are enabled.
                self.link.prepare_transport()
            return result

    def _wait_for_feedback(self, code, started):
        if not (self.hardware_output and self.link.lease and self._continuation_supported):
            return False
        if self._feedback_wait_started is None:
            self._feedback_wait_started = started
        if started-self._feedback_wait_started >= self._feedback_wait_seconds:
            return False
        self.output = {'state':'waiting_driver_hold', 'code':code, 'output_active':False}
        return True

    def external_fault_code(self):
        if not self.hardware_output or not self.link.lease:
            return None
        try:
            return 'driver_fault' if self.link.feedback().get('state') == 'fault' else None
        except ValueError:
            return None

    def apply(self, intent):
        started = time.monotonic()
        started_cpu = time.thread_time()
        phase = "validation"
        ik_started_ns = None
        solved_target = None
        previous_send = getattr(self.link, 'last_send', None)
        deadline = min(intent.expires_monotonic,
                       (intent.received_monotonic or intent.admitted_monotonic) + self.input_timeout_ms/1000.)
        with self.lock:
            try:
                if intent.dispatch_generation <= self._revoked_generation:
                    raise ValueError('dispatch_revoked')
                if self.solver is None:
                    raise ValueError('calibration_missing')
                frame = intent.frame
                buttons = [frame['controllers'][s]['buttons'] for s in ('left', 'right')]
                if any(len(b) != 2 for b in buttons):
                    raise ValueError('trigger_binding_missing')
                triggers = [b[0] for b in buttons]
                hands_enabled = getattr(self.solver, 'hands_enabled', True)
                if not hands_enabled:
                    triggers = [0., 0.]
                phase = "feedback"
                try:
                    state = self.link.feedback()
                except ValueError as exc:
                    if str(exc) in ('driver_feedback_missing', 'driver_feedback_stale_or_different_clock'):
                        if self._wait_for_feedback(str(exc), started):
                            return AdapterAck(True, 'waiting_driver_feedback')
                    raise
                policy = state.get('timing_policy', {})
                self._recoverable_hold_supported = policy.get('recoverable_hold') is True
                self.link.management_retry = policy.get('management_retry_ms') == 300
                wait_ms = policy.get('feedback_fault_timeout_ms')
                self._continuation_supported = (
                    type(policy.get('continuation_timeout_ms')) is int
                    and 100 <= policy['continuation_timeout_ms'] <= 1000
                    and type(wait_ms) is int and 100 <= wait_ms <= 1000)
                if self._continuation_supported:
                    self._feedback_wait_seconds = wait_ms/1000
                if self.hardware_output and self.link.lease and state.get('state') == 'fault':
                    raise ValueError('driver_fault')
                if self.hardware_output and self.link.lease and state.get('state') == 'hold':
                    if state.get('continuation_allowed') is True and state.get('continuation_ready') is not True:
                        self.output = {'state':'waiting_driver_hold', 'code':state.get('reason'), 'output_active':False}
                        return AdapterAck(True, 'waiting_driver_hold')
                q = state['feedback']['q']
                if not 0 <= time.monotonic_ns()-state['feedback'].get('arm_ns', 0) <= 100_000_000:
                    if self._wait_for_feedback('arm_feedback_stale', started):
                        return AdapterAck(True, 'waiting_driver_feedback')
                    raise ValueError('arm_feedback_stale')
                if self.hardware_output and state.get('calibration_sha256') != self.solver.profile_sha256:
                    raise ValueError('driver_calibration_mismatch')
                phase = "mapping"
                clutch = (intent.session_generation, intent.clutch_sequence)
                fresh = clutch != self.clutch
                if fresh:
                    if max(triggers) > .05:
                        raise ValueError('triggers_not_neutral')
                    self.mapper.reset(frame, self.solver.palms(q))
                    if hasattr(self.solver, 'reset_target_state'):
                        self.solver.reset_target_state()
                if self.hardware_output and (fresh or self._resume_validated or getattr(self.link,'management_request',None)):
                    # A clutch establishes a measured reference and a lease.
                    # It never sends a target: solving here only throws the IK
                    # result away and steals time from the bounded claim RPC.
                    phase = 'driver_resume' if self.link.lease else 'driver_claim'
                    if intent.dispatch_generation <= self._revoked_generation:
                        raise ValueError('dispatch_revoked')
                    if self.link.lease:
                        self.link.resume(deadline-.015)
                    else:
                        self.link.claim(deadline-.015)
                    self.clutch = clutch
                    self._automatic_resume = False
                    self._resume_validated = False
                    self.output = {'state':'armed_waiting_input', 'output_active':False}
                    return AdapterAck(True)
                targets = self.mapper.targets(frame)
                phase = "ik"
                ik_started_ns = time.monotonic_ns()
                target = self.solver.solve(targets, q, commanded=state.get('commanded_q'),
                                           deadline_monotonic=deadline-.015)
                solved_target = list(target)
                self.ik_times.append(self.solver.last_ms)
                if time.monotonic() >= deadline:
                    raise ValueError('ik_motion_deadline')
                if intent.dispatch_generation <= self._revoked_generation:
                    raise ValueError('dispatch_revoked')
                phase = "driver"
                if self.hardware_output:
                    if self._automatic_resume and not (self._same_session_hold
                            and state.get('continuation_ready') is True):
                        # Validation and management each get a fresh input
                        # budget. This result is discarded; even after rearm a
                        # subsequent frame must solve/check collision again.
                        self._resume_validated = True
                        self.output = {'state': 'recovery_validated', 'output_active': False}
                        return AdapterAck(True)
                    sent = self.link.send(target, triggers, deadline, wait_for_execution=False, allow_continuation=True, target_ttl_ms=100)
                    if sent is False:
                        self.output = {'state':'waiting_driver_hold', 'output_active':False}
                        return AdapterAck(True, 'waiting_driver_hold')
                    self._automatic_resume = False
                    self._same_session_hold = False
                    self.output = {'state': 'submitted', 'target_q': target, 'hands': triggers,
                                   'hands_enabled': hands_enabled}
                else:
                    self.clutch = clutch
                    self._automatic_resume = False
                    self.output = {'state': 'would_apply', 'target_q': target, 'hands': triggers,
                                   'hands_enabled': hands_enabled,
                                   'publisher_present': False, 'output_active': False}
                visual=getattr(self.solver,'last_valid_visualization',None)
                self._feedback_wait_started = None
                diagnostics = getattr(self.solver, 'target_diagnostics', None)
                if diagnostics is not None:
                    self.output['target_diagnostics'] = diagnostics
                if visual is not None:
                    self.output['ik_reference_q']=[float(x) for x in visual['ik_q']]
                self.chain_times.append((time.monotonic()-(intent.received_monotonic or intent.admitted_monotonic))*1000)
                return AdapterAck(True)
            except Exception as exc:
                code = str(exc)
                failed_at = time.monotonic()
                self.last_failure = {
                    'sequence': intent.sequence, 'phase': phase, 'code': code,
                    'monotonic_ns': time.monotonic_ns(),
                    'elapsed_ms': (failed_at-started)*1000,
                    'budget_at_start_ms': (deadline-started)*1000,
                    'remaining_ms': (deadline-failed_at)*1000,
                }
                self.output = {'state': 'error', 'code': code, 'output_active': False}
                # send() rechecks feedback before publishing. If it aged during
                # IK, discard this target and wait under the SAME bounded
                # recovery budget as a stale pre-solve snapshot. Only an actual
                # successful output resets that budget, not a fresh first read.
                if (phase == 'driver' and code in ('driver_feedback_missing',
                        'driver_feedback_stale_or_different_clock', 'arm_feedback_stale')
                        and self._wait_for_feedback(code, failed_at)):
                    return AdapterAck(True, 'waiting_driver_feedback')
                if code in ('driver_management_pending','driver_lease_rebased'):
                    if code=='driver_lease_rebased':
                        # Transport recovery is not a new operator clutch.
                        # Keep its relative reference; require a new successful
                        # solve before reclaiming, then a later fresh output.
                        self._resume_validated=False
                    self.output={'state':'waiting_driver_management','code':code,'output_active':False}
                    return AdapterAck(True, 'waiting_driver_management')
                if code == 'driver_holding':
                    try:
                        code = self.link.feedback().get('reason', code)
                    except ValueError:
                        pass
                # The wire can reject a packet that expired in transit while
                # the watchdog uses command_timeout. Both use confirmed-hold recovery.
                if code in ('arm_feedback_stale', 'driver_feedback_stale_or_different_clock', 'driver_feedback_missing',
                            'arm_ns_stale', 'power_ns_stale', 'fixed_ns_stale'):
                    code = 'feedback_unavailable'
                if code == 'command_expired':
                    code = 'command_timeout'
                if code == 'robot_not_stopped':
                    code = 'arms_not_stationary'
                recoverable = IK_HOLD_CODES | {'feedback_unavailable', 'command_timeout', 'driver_execution_ack_timeout',
                                               'arms_not_stationary', 'triggers_not_neutral'}
                return AdapterAck(False, code if code in recoverable else 'motion_rejected')
            finally:
                finished = time.monotonic()
                self.last_apply = {'sequence':intent.sequence, 'phase':phase,
                                   'elapsed_ms':(finished-started)*1000,
                                   'thread_cpu_ms':(time.thread_time()-started_cpu)*1000,
                                   'budget_at_start_ms':(deadline-started)*1000,
                                   'remaining_ms':(deadline-finished)*1000,
                                   'driver_call':getattr(self.link, 'last_call', None)}
                if self.trace_sink is not None or self.result_observer is not None:
                    failed=self.last_failure if self.last_failure and self.last_failure['sequence']==intent.sequence else None
                    published=getattr(self.link,'last_send',None) is not previous_send
                    event = {'event':'actucore_apply','monotonic_ns':time.monotonic_ns(),
                        'input_sequence':intent.sequence,'clutch_sequence':intent.clutch_sequence,
                        'input_received_ns':int((intent.received_monotonic or intent.admitted_monotonic)*1e9),
                        'ik_started_ns':ik_started_ns,'ik_succeeded':solved_target is not None,
                        'ik_target_q':solved_target,'failure':dict(failed) if failed else None,
                        'ik_reference_q':self.output.get('ik_reference_q') if solved_target is not None else None,
                        'target_diagnostics':getattr(self.solver, 'target_diagnostics', None) if ik_started_ns is not None else None,
                        'last_apply':dict(self.last_apply),
                        'output_state':self.output.get('state'),'published':published,
                        'publish':dict(self.link.last_send) if published and getattr(self.link,'last_send',None) else None,
                        'session_id':self.link.lease.get('session_id') if self.link.lease else None}
                    if self.result_observer is not None:
                        try:self.result_observer(event)
                        except Exception:self.recording_errors += 1
                    if self.trace_sink is not None:self.trace_sink(event)

    def safe_stop(self, request):
        self._revoked_generation = max(self._revoked_generation, request.dispatch_generation-1)
        if not self.lock.acquire(timeout=max(0, request.deadline_monotonic-time.monotonic())):
            return AdapterAck(False, 'stop_unconfirmed')
        try:
            self._resume_validated = False
            self._feedback_wait_started = None
            self._same_session_hold = bool(self.hardware_output and self._recoverable_hold_supported
                and self.link.lease and not getattr(self.link,"management_request",None)
                and request.reason in IK_RETRY_CODES | {'workspace_limit'})
            operation = self.link.recoverable_hold if self._same_session_hold else self.link.pause
            try:
                ok = self._confirm_stop(operation,request.deadline_monotonic) if self.hardware_output else True
            except ValueError as exc:
                if not self._same_session_hold or str(exc) != 'hold_not_resumable':
                    raise
                # A concurrent safety rejection may have closed continuation.
                # Confirm a normal authenticated pause instead; this never
                # reopens continuation or assumes the previous stop succeeded.
                self._same_session_hold = False
                ok = self._confirm_stop(self.link.pause, request.deadline_monotonic)
            retry_codes = COLLISION_HOLD_CODES | IK_RETRY_CODES | {'workspace_limit'}
            if self.hardware_output:
                retry_codes |= {"command_timeout", "driver_execution_ack_timeout", "feedback_unavailable", "arms_not_stationary"}
            else:
                retry_codes |= {"feedback_unavailable"}
            self._automatic_resume = bool(ok and request.reason in retry_codes)
            if not self._automatic_resume:
                self.clutch = None
            self.output = {'state': 'held' if ok else 'stop_unconfirmed',
                           'output_active': False if ok else None}
            return AdapterAck(ok, 'stop_confirmed' if ok else 'stop_unconfirmed')
        except Exception as exc:
            self.output = {'state': 'stop_unconfirmed', 'code': str(exc), 'output_active': None}
            return AdapterAck(False, 'stop_unconfirmed')
        finally:
            self.lock.release()

    @staticmethod
    def _confirm_stop(operation,deadline):
        while time.monotonic()<deadline:
            try:
                if operation(min(deadline,time.monotonic()+.25)):return True
            except ValueError as exc:
                if str(exc) not in ('driver_feedback_missing','driver_feedback_stale_or_different_clock','driver_feedback_ack_timeout','driver_management_pending'):raise
            time.sleep(.01)
        return False

    def close(self):
        try:
            ok = self._confirm_stop(self.link.stop,time.monotonic()+self.stop_confirmation_timeout_ms/1000.) if self.hardware_output else True
            return AdapterAck(ok, 'stop_confirmed' if ok else 'stop_unconfirmed')
        except Exception:
            return AdapterAck(False, 'stop_unconfirmed')
