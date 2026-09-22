"""Local jerk-limited command generation and checked offline quintic curves.

This module owns mathematical trajectories, never device authority. Ruckig is
loaded only when enabled; intermediate waypoints/cloud APIs are never used.
"""
from __future__ import annotations

import copy
import time

import numpy as np


def vector(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf' or raw.shape != (14,) or not np.isfinite(raw).all():
        raise ValueError('trajectory_invalid_' + name)
    return raw.astype(float)


def execution_state(state, *, required=True):
    """Extract a coherent, non-secret issued-state snapshot from Driver status."""
    command = state.get('command_state')
    if command is None:
        if required:raise ValueError('trajectory_command_state_missing')
        return None
    command = copy.deepcopy(command)
    if ((state.get('state') == 'hold' and state.get('hold_confirmed') is True)
            or (state.get('state') == 'ready' and command.get('kind') == 'stationary_seed')):
        feedback = state.get('feedback', {})
        stamp = feedback.get('arm_ns')
        dq = vector(feedback.get('dq'), 'feedback_velocity')
        if (type(stamp) is not int or not 0 <= time.monotonic_ns()-stamp <= 100_000_000
                or np.max(np.abs(dq)) > .02):
            raise ValueError('trajectory_stationary_feedback_missing')
        command.update(stationary_confirmed=True, sample_ns=stamp)
    return command


class JointTrajectory:
    PERIOD = .02

    def __init__(self, config, velocity):
        config = {} if config is None else config
        if not isinstance(config, dict) or set(config) - {
                'enabled', 'max_acceleration_rad_s2', 'max_jerk_rad_s3'}:
            raise ValueError('trajectory_config')
        self.enabled = config.get('enabled', False)
        if type(self.enabled) is not bool:
            raise ValueError('trajectory_enabled')
        self.velocity = np.full(14, velocity, dtype=float)
        self.reset()
        if not self.enabled:return
        for key, name in (('max_acceleration_rad_s2', 'acceleration'), ('max_jerk_rad_s3', 'jerk')):
            value = config.get(key)
            if type(value) in (int, float):value = [value]*14
            limits = vector(value, name)
            if np.any(limits <= 0) or np.any(limits > 1e4):
                raise ValueError('trajectory_invalid_' + name)
            setattr(self, name, limits)
        try:
            import ruckig
        except ImportError as exc:
            raise ValueError('trajectory_dependency_missing') from exc
        self.ruckig = ruckig
        self.generator = ruckig.Ruckig(14, self.PERIOD)

    def reset(self):
        self.pending = None
        self.diagnostics = None

    def propose(self, goal, measured, previous, command_state, lower, upper, budget):
        """Return a sample and full-trajectory joint box, without committing it."""
        budget()
        previous = vector(previous, 'commanded_q')
        source = 'shadow_stationary_reference'
        dq = ddq = np.zeros(14)
        stamp = sequence = None
        if command_state is not None:
            if not isinstance(command_state, dict) or command_state.get('schema') != 'motus.command-state.v1':
                raise ValueError('trajectory_command_state_missing')
            stamp = command_state.get('sample_ns')
            if type(stamp) is not int or not 0 <= time.monotonic_ns()-stamp <= 100_000_000:
                raise ValueError('trajectory_command_state_stale')
            if not np.allclose(vector(command_state.get('q'), 'state_q'), previous, atol=1e-9, rtol=0):
                raise ValueError('trajectory_command_state_mismatch')
            sequence = command_state.get('target_sequence')
            if command_state.get('kind') == 'stationary_seed' or command_state.get('stationary_confirmed') is True:
                source = 'driver_stationary_reference'
            elif (self.pending is not None and not command_state.get('limited', True)
                  and type(command_state.get('dt_s')) in (int, float)
                  and abs(command_state['dt_s']-self.PERIOD) <= 1e-6
                  and sequence != self.pending['input_sequence']
                  and np.allclose(previous, self.pending['q'], atol=1e-9, rtol=0)):
                # Continue derivatives only after the Driver actually issued
                # the proposed position. An overwritten proposal is not state.
                dq, ddq = self.pending['dq'], self.pending['ddq']
                source = 'confirmed_planner_sample'
            else:
                dq = vector(command_state.get('dq'), 'command_velocity')
                ddq = vector(command_state.get('ddq'), 'command_acceleration')
                source = 'driver_finite_difference'
        lead = self.velocity*.2
        # Keep half the existing travel allowance for braking/reversal overshoot.
        # This is a target choice before planning, never a clip of its output.
        bounded_goal = np.clip(vector(goal, 'goal'), np.maximum(lower, measured-lead*.5), np.minimum(upper, measured+lead*.5))
        inp = self.ruckig.InputParameter(14)
        inp.current_position = previous.tolist()
        inp.current_velocity = dq.tolist()
        inp.current_acceleration = ddq.tolist()
        inp.target_position = bounded_goal.tolist()
        inp.target_velocity = [0.]*14
        inp.target_acceleration = [0.]*14
        inp.max_velocity = self.velocity.tolist()
        inp.max_acceleration = self.acceleration.tolist()
        inp.max_jerk = self.jerk.tolist()
        curve = self.ruckig.Trajectory(14)
        try:
            self.generator.validate_input(inp, True, True)
            # Each candidate is a transaction from observed/confirmed state.
            # calculate() avoids update()'s cached output clock advancing when
            # a candidate is rejected, overwritten, or never sent.
            result = self.generator.calculate(inp, curve)
        except Exception as exc:
            raise ValueError('trajectory_state_invalid') from exc
        if result not in (self.ruckig.Result.Working, self.ruckig.Result.Finished):
            raise ValueError('trajectory_generation_failed')
        budget()
        next_q, next_dq, next_ddq = curve.at_time(self.PERIOD)
        q = vector(next_q, 'generated_q')
        lo = np.array([bound.min for bound in curve.position_extrema])
        hi = np.array([bound.max for bound in curve.position_extrema])
        # Include the complete braking/reversal path, not just a safe endpoint.
        if (not np.isfinite(lo).all() or not np.isfinite(hi).all()
                or np.any(lo < lower-1e-10) or np.any(hi > upper+1e-10)
                or np.any(lo < measured-lead-1e-10) or np.any(hi > measured+lead+1e-10)):
            raise ValueError('trajectory_braking_envelope')
        plan = {'q': q, 'dq': vector(next_dq, 'generated_velocity'),
                'ddq': vector(next_ddq, 'generated_acceleration'),
                'input_sequence': sequence, 'input_sample_ns': stamp,
                'diagnostics': {'state': 'planned', 'state_source': source,
                    'period_s': self.PERIOD, 'duration_s': curve.duration,
                    'velocity_rad_s': list(next_dq), 'acceleration_rad_s2': list(next_ddq),
                    'bounded_goal_q': bounded_goal.tolist(),
                    'driver_limited_previous': command_state.get('limited') if command_state else None}}
        return q, lo, hi, plan

    def commit(self, plan):
        self.pending = plan
        self.diagnostics = copy.deepcopy(plan['diagnostics'])


def smooth_schedule(points, trajectory, solver):
    """Quintic interpolation and analytic v/a/j retiming, separately per phase.

    Starts/ends of a phase have zero velocity AND acceleration. Internal knots
    remain continuous; there is no artificial stop at every recorded frame.
    Every polynomial span's complete joint box is collision checked, including
    extrema between recorded waypoints. Input records are never overwritten.
    """
    from scipy.interpolate import make_interp_spline, PPoly
    if len(points) < 2:raise ValueError('trajectory_waypoints_required')
    times = np.asarray([p['t'] for p in points], dtype=float)
    qs = np.asarray([vector(p['q'], 'waypoint') for p in points])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('trajectory_timestamps')
    lower = solver.model.lowerPositionLimit[solver.indices]
    upper = solver.model.upperPositionLimit[solver.indices]
    output = [copy.deepcopy(points[0])]
    end = 1
    while end < len(points):
        start = end-1
        phase = points[end]['phase']
        while end+1 < len(points) and points[end+1]['phase'] == phase:end += 1
        ts = times[start:end+1]-times[start]
        values = qs[start:end+1]
        zeros = np.zeros(14)
        spline = make_interp_spline(ts, values, k=5,
            bc_type=([(1, zeros), (2, zeros)], [(1, zeros), (2, zeros)]))
        polynomials = [PPoly.from_spline((spline.t, spline.c[:, i], spline.k)) for i in range(14)]
        breaks = np.unique(spline.t)
        factor = 1.
        for order, limits in ((1, trajectory.velocity), (2, trajectory.acceleration), (3, trajectory.jerk)):
            for poly, limit in zip(polynomials, limits):
                derivative = poly.derivative(order)
                roots = derivative.derivative().roots(extrapolate=False)
                roots = roots[np.isfinite(roots) & (roots >= ts[0]) & (roots <= ts[-1])]
                maximum = np.max(np.abs(derivative(np.r_[breaks, roots])))
                factor = max(factor, float(maximum/limit)**(1/order))
        factor *= 1.+1e-9
        offset = output[-1]['t']
        for a,b in zip(breaks, breaks[1:]):
            lo,hi = [],[]
            for poly in polynomials:
                roots = poly.derivative().roots(extrapolate=False)
                roots = roots[np.isfinite(roots) & (roots > a) & (roots < b)]
                values = poly(np.r_[a,b,roots])
                lo.append(values.min());hi.append(values.max())
            lo,hi = np.array(lo),np.array(hi)
            if np.any(lo < lower-1e-10) or np.any(hi > upper+1e-10):
                raise ValueError('trajectory_curve_joint_limit')
            if hasattr(solver, '_safe_transition'):
                solver._safe_transition(lo, hi)
            else:
                solver._safe_configuration((lo+hi)*.5, excursion=(hi-lo)*.5)
            index = np.searchsorted(polynomials[0].x, a, side='right')-1
            coefficients = np.column_stack([p.c[:, index] for p in polynomials])
            coefficients = coefficients/np.array([factor**power for power in range(5,-1,-1)])[:, None]
            output[-1]['polynomial'] = coefficients.tolist()
            output[-1]['curve'] = 'quintic_vaj_v1'
            output.append({'t':offset+b*factor, 'q':[float(p(b)) for p in polynomials],
                           'phase':phase, 'time_scale':factor})
        end += 1
    return output
