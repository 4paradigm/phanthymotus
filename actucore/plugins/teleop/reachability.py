"""Explicit, opt-in reachable-pose saturation for Tianyi's constrained IK.

The least-squares result is a local projection, not a global workspace oracle.
The caller must still check the projected configuration AND actual swept output.
State is committed only after those checks; the raw clutch reference never moves.
"""
from __future__ import annotations

import numpy as np


class ReachableTarget:
    POSITION_TOLERANCE = .003
    ANGLE_TOLERANCE = .03

    def __init__(self, config, pin):
        self.pin = pin
        config = {} if config is None else config
        if not isinstance(config, dict) or set(config) - {
                'enabled', 'workspace_margin_m', 'joint_margin_rad',
                'release_frames'}:
            raise ValueError('target_projection_config')
        self.enabled = config.get('enabled', False)
        if type(self.enabled) is not bool:
            raise ValueError('target_projection_enabled')
        for name, default, maximum in (
                ('workspace_margin_m', .01, .05), ('joint_margin_rad', .03, .1)):
            value = config.get(name, default)
            if type(value) not in (int, float) or not np.isfinite(value) or not 0 < value <= maximum:
                raise ValueError('target_projection_' + name)
            setattr(self, name, float(value))
        self.release_frames = config.get('release_frames', 3)
        if type(self.release_frames) is not int or not 1 <= self.release_frames <= 10:
            raise ValueError('target_projection_release_frames')
        self.reset()

    def reset(self):
        self.boundary = None

    def bounds(self, lower, upper, measured):
        if not self.enabled:
            return lower, upper
        margin = np.minimum(self.joint_margin_rad, (upper-lower)*.1)
        # A joint already in the margin may stay still or move inward. Never
        # manufacture a jump to the inset when enabling/calibrating at rest.
        return (np.minimum(lower+margin, np.clip(measured, lower, upper)),
                np.maximum(upper-margin, np.clip(measured, lower, upper)))

    def targets(self, raw, workspace):
        targets = [t.copy() for t in raw]
        clipped = [False, False]
        if self.enabled:
            for i, side in enumerate(('left', 'right')):
                box = np.asarray(workspace[side], dtype=float)
                margin = self.workspace_margin_m
                if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[1]-box[0] <= 2*margin):
                    raise ValueError('target_projection_workspace')
                point = np.clip(raw[i][:3, 3], box[0]+margin, box[1]-margin)
                clipped[i] = bool(np.any(np.abs(point-raw[i][:3, 3]) > 1e-10))
                targets[i][:3, 3] = point
        return targets, clipped

    def residuals(self, actual, raw):
        return [{'position_m': float(np.linalg.norm(a[:3, 3]-b[:3, 3])),
                 'orientation_rad': float(np.linalg.norm(self.pin.log3(a[:3, :3].T @ b[:3, :3])))}
                for a, b in zip(actual, raw)]

    def select(self, raw, q, actual, clipped, converged):
        residuals = self.residuals(actual, raw)
        status = {'enabled': self.enabled, 'state': 'tracking',
                  'raw_targets': [t.tolist() for t in raw], 'residuals': residuals,
                  'reasons': []}
        if not self.enabled:
            status['feasible_targets'] = [t.tolist() for t in actual]
            return q, actual, None, status
        limited = [clipped[i] or r['position_m'] > self.POSITION_TOLERANCE
                   or r['orientation_rad'] > self.ANGLE_TOLERANCE
                   for i, r in enumerate(residuals)]
        boundary = self.boundary
        if any(limited) and boundary is None:
            # Exhausting the numerical budget does not establish a boundary.
            if not converged:
                raise ValueError('ik_not_converged')
            reasons = set()
            for i, residual in enumerate(residuals):
                rot_limited = residual['orientation_rad'] > self.ANGLE_TOLERANCE
                if clipped[i]: reasons.add('workspace')
                if residual['position_m'] > self.POSITION_TOLERANCE: reasons.add('position_residual')
                if rot_limited: reasons.add('orientation_residual')
            boundary = {'q': q.copy(), 'targets': [t.copy() for t in actual],
                        'release_count': 0, 'reasons': sorted(reasons)}
        elif boundary is not None:
            # A reachable pose can re-enter from any direction. A fixed surface
            # normal would incorrectly trap a hand returning along a curved
            # workspace. Stricter residuals plus consecutive fresh solves avoid
            # chatter without changing the raw mapping or integrating error.
            inward = not any(clipped) and all(
                r['position_m'] <= self.POSITION_TOLERANCE*.5 and
                r['orientation_rad'] <= self.ANGLE_TOLERANCE*.5 for r in residuals)
            boundary = {**boundary, 'release_count': boundary['release_count']+1 if inward else 0}
            if boundary['release_count'] >= self.release_frames:
                boundary = None
        if boundary is not None:
            q, actual = boundary['q'].copy(), [t.copy() for t in boundary['targets']]
            status.update(state='saturated', reasons=list(boundary['reasons']))
            status['release_count'] = boundary['release_count']
        status['feasible_targets'] = [t.tolist() for t in actual]
        status['fit_residuals'] = residuals
        status['residuals'] = self.residuals(actual, raw)
        return q, actual, boundary, status
