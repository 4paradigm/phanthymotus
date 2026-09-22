"""Read-only Tianyi FK for the PICO overlay; never acquires execution authority."""
import time
import numpy as np

SCHEMA = 'motus.tianyi-visualization.v1'


def chains(solver, q):
    q = np.asarray(q, dtype=float)
    if q.shape != (14,) or not np.isfinite(q).all():
        raise ValueError('visualization_invalid_q')
    model_q = np.empty(14)
    model_q[solver.indices] = q
    data = solver.model.createData()
    solver.pin.framesForwardKinematics(solver.model, data, model_q)
    torso = data.oMf[solver.torso].inverse()
    result = []
    for side, palm in enumerate(solver.frames):
        names = solver.profile['arm_joint_names'][side*7:(side+1)*7]
        points = [(torso * data.oMi[solver.model.getJointId(n)]).translation.tolist()
                  for n in names]
        points.append((torso * data.oMf[palm]).translation.tolist())
        result.append(points)
    return result


def snapshot(adapter):
    try:
        solver = adapter.solver
        if solver is None:
            return {'schema': SCHEMA, 'available': False, 'reason': 'calibration_missing'}
        try:
            state = adapter.link.feedback()
        except ValueError:
            state = {'reason': 'feedback_unavailable'}
        feedback = state.get('feedback', {})
        now = time.monotonic_ns()
        age = (now-feedback.get('arm_ns', 0))/1e6
        fresh = 0 <= age <= 100
        preview = solver.visualization_sample
        last_valid = solver.last_valid_visualization
        output = adapter.output
        # Fixed body reference derives shoulder anchors from this robot's neutral
        # model. It is an orientation aid, not a body mesh or collision envelope.
        neutral = chains(solver, np.zeros(14))
        left, right = neutral[0][0], neutral[1][0]
        height = (left[2]+right[2])/2
        body = [[left, [0, left[1], 0], [0, right[1], 0], right, left],
                [[0, 0, 0], [0, 0, height+.06]],
                [[0, -.06, height+.06], [0, .06, height+.06],
                 [0, .06, height+.20], [0, -.06, height+.20], [0, -.06, height+.06]]]
        return {'schema': SCHEMA, 'available': True,
                'mode': 'live' if adapter.hardware_output else 'shadow',
                'state': output.get('state', 'idle'),
                'reason': output.get('code') or state.get('reason') or (preview or {}).get('error') or '',
                'feedback_fresh': fresh, 'feedback_age_ms': max(0., age),
                'intent_age_ms': max(0., (now-preview['monotonic_ns'])/1e6) if preview else None,
                'solve_ms': solver.last_ms,
                'measured': chains(solver, feedback['q']) if fresh else [],
                'ik': chains(solver, preview['ik_q']) if preview and preview['ik_q'] is not None else [],
                # Explicit history: never substitute for the current IK result.
                'held_ik': chains(solver, last_valid['ik_q']) if last_valid else [],
                'held_ik_age_ms': max(0., (now-last_valid['monotonic_ns'])/1e6) if last_valid else None,
                'command': [],
                'targets': [t[:3, 3].tolist() for t in preview['targets']] if preview else [],
                'feasible_targets': [t[:3, 3].tolist() for t in preview.get('feasible_targets', [])] if preview else [],
                'target_diagnostics': getattr(solver, 'target_diagnostics', None),
                # These calibrated boxes are safety bounds, NOT a claim that
                # every enclosed pose is reachable.
                'workspace_bounds': [solver.workspace[s] for s in ('left', 'right')],
                'body': body, 'shoulder_height': height}
    except (ValueError, KeyError, TypeError):
        return {'schema': SCHEMA, 'available': False, 'reason': 'feedback_unavailable'}
