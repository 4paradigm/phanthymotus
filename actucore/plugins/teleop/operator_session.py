"""Paired-headset operations and explicit, cancellable arm return.

No command queues, implicit homing, or persistent motion permission.
"""
from __future__ import annotations
import asyncio
import threading
import time
from collections import OrderedDict
import numpy as np


class OperatorCommands:
    def __init__(self, manager, execute):
        self.manager, self.execute = manager, execute
        self.receipts = OrderedDict()
        self.task = None
        self.cancel = threading.Event()
        self.status = {'state': 'idle', 'action': None, 'error': None}

    def valid(self, connection):
        return (self.manager._connection is connection
                and not self.manager.presence_expired(connection))

    async def submit(self, connection, message):
        request_id = message.get('request_id')
        action = message.get('action')
        if (not self.valid(connection) or message.get('connection_id') != connection.connection_id):
            raise ValueError('operator_connection_changed')
        if (action not in ('start', 'finish', 'stop') or not isinstance(request_id,str)
                or not 1 <= len(request_id) <= 80):
            raise ValueError('operator_request_invalid')
        key = (connection.connection_id, request_id)
        for old in list(self.receipts):
            if old[0]!=connection.connection_id:del self.receipts[old]
        if key in self.receipts:
            receipt = self.receipts[key]
            if receipt['action'] != action:raise ValueError('operator_request_conflict')
            return dict(receipt)
        if self.task and not self.task.done():
            if action != 'stop':raise ValueError('operator_busy')
            # Stop cancellation never waits on the long return operation in
            # the websocket loop. Worker will join it before releasing ownership.
            self.cancel.set()
        if len(self.receipts)>=256 and action!='stop':raise ValueError('operator_request_history_full')
        previous = self.task
        if action != 'stop':self.cancel = threading.Event()
        receipt = {'type':'operator_result','request_id':request_id,'action':action,'state':'accepted'}
        self.receipts[key] = receipt
        async def run():
            try:
                if previous and not previous.done():await previous
                if action != 'stop' and not self.valid(connection):
                    raise ValueError('operator_connection_changed')
                self.status = {'state': {'finish':'returning','start':'starting','stop':'stopping'}[action],
                               'action':action,'error':None}
                result = await asyncio.to_thread(self.execute, action, self.cancel,
                    lambda:self.valid(connection))
                receipt.update(state='completed', result=result)
                self.status = {'state':result.get('state','idle'),'action':action,'error':None}
            except Exception as exc:
                receipt.update(state='failed', error=str(exc))
                self.status = {'state':'error','action':action,'error':str(exc)}
            if self.valid(connection):await connection.events.put(dict(receipt))
        self.task = asyncio.create_task(run())
        return dict(receipt)


def return_arms(adapter, cancel, connected, *, timeout=45., clock=time.monotonic, sleep=time.sleep):
    """Caller has frozen runtime input and revoked its assignment, retains lease.

    Return targets are recomputed from fresh actual state; each segment is
    validated by the same collision model. Stop is confirmed before error exits.
    """
    if not adapter.hardware_output:raise ValueError('return_requires_live')
    link, solver = adapter.link, adapter.solver
    if solver is None or solver.hands_enabled:raise ValueError('return_requires_calibrated_arms_only')
    deadline = clock()+timeout
    settled = None
    zero = np.zeros(14)
    lower=solver.model.lowerPositionLimit[solver.indices]
    upper=solver.model.upperPositionLimit[solver.indices]
    if np.any(zero<lower) or np.any(zero>upper):raise ValueError('neutral_outside_limits')
    def check():
        if cancel.is_set():raise ValueError('return_cancelled')
        if not connected():raise ValueError('return_connection_lost')
        if clock()>=deadline:raise ValueError('return_timeout')
    try:
        with adapter.lock:
            check()
            link.prepare_transport()
            link.management_retry = link.feedback().get('timing_policy', {}).get('management_retry_ms') == 300
        management_deadline=min(deadline,clock()+2.5)
        method=link.resume if link.lease else link.claim
        while True:
            check()
            if clock()>=management_deadline:raise ValueError('return_management_timeout')
            try:
                with adapter.lock:method(min(management_deadline,clock()+.09))
                break
            except ValueError as exc:
                if str(exc)=='driver_lease_rebased':
                    if link.lease or link.management_request:raise ValueError('management_release_unconfirmed')
                    method=link.claim
                elif str(exc) not in ('driver_management_pending','driver_feedback_missing','driver_feedback_stale_or_different_clock','robot_not_stopped'):
                    raise
                sleep(.01)
        with adapter.lock:
            link.feedback_after(link.lease_started_ns,min(deadline,clock()+.25))
        while True:
            check()
            with adapter.lock:
                state=link.feedback();feedback=state['feedback']
                q=np.asarray(feedback['q'],dtype=float);dq=np.asarray(feedback['dq'],dtype=float)
                age=time.monotonic_ns()-feedback.get('arm_ns',0)
                if q.shape!=(14,) or dq.shape!=(14,) or not np.isfinite(q).all() or not np.isfinite(dq).all() or not 0<=age<=100_000_000:
                    raise ValueError('return_feedback_invalid')
                if state.get('state')=='fault':raise ValueError('return_driver_fault')
                if np.max(np.abs(q))<=.02 and np.max(np.abs(dq))<=.02:
                    settled=clock() if settled is None else settled
                    if clock()-settled>=.1:
                        if not adapter.close().ok:raise ValueError('stop_unconfirmed')
                        return {'state':'idle','return_completed':True,'max_error_rad':float(np.max(np.abs(q)))}
                else:settled=None
                if state.get('state')=='hold':
                    if not state.get('hold_confirmed'):raise ValueError('return_hold_unconfirmed')
                    # No automatic retry of a failed return. The operator may
                    # retry explicitly after the reported cause is resolved.
                    raise ValueError('return_driver_hold:'+str(state.get('reason')))
                previous=np.asarray(state.get('commanded_q') or q,dtype=float)
                step=previous+np.clip(zero-previous,-solver.velocity*.02,solver.velocity*.02)
                step=np.clip(step,q-solver.velocity*.2,q+solver.velocity*.2)
                step=np.clip(step,lower,upper)
                cycle_deadline=min(deadline,clock()+.10)
                def budget():
                    check()
                    if clock()>=cycle_deadline:raise ValueError('return_geometry_timeout')
                with solver.lock:
                    if getattr(getattr(solver, 'trajectory', None), 'enabled', False):
                        from .trajectory import execution_state
                        step = solver.command_step(zero, q, previous, budget, execution_state(state))
                    else:
                        solver._safe_transition(q,step,budget)
                        solver._safe_transition(previous,step,budget)
                check()
                link.send(step.tolist(),[0.,0.],cycle_deadline,wait_for_execution=False,target_ttl_ms=100)
                adapter.output={'state':'returning','target_q':step.tolist(),'output_active':True}
            sleep(.02)
    except Exception:
        with adapter.lock:
            confirmed=adapter._confirm_stop(link.pause,clock()+adapter.stop_confirmation_timeout_ms/1000.)
            adapter.output={'state':'held' if confirmed else 'stop_unconfirmed','output_active':False if confirmed else None}
        raise
