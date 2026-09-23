"""Read-only kinematic telemetry. Never claims authority or sends motion commands."""
import time
import numpy as np


def chains(solver, q):
    q = np.asarray(q, dtype=float)
    if q.shape != (10,) or not np.isfinite(q).all():
        raise ValueError('visualization_invalid_q')
    data = solver.model.createData()
    solver.pin.framesForwardKinematics(solver.model, data, q)
    return [[data.oMf[solver.model.getFrameId(name)].translation.tolist() for name in
             [side+'_'+joint+'_joint' for joint in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll')] + [ee]]
            for side,ee in (('left','L_ee'),('right','R_ee'))]


def snapshot(adapter):
    # Published sample/output objects are replaced atomically, never edited.
    # Display must remain available while the execution owner holds its lock.
    # FK uses immutable model geometry and its own Data, not solver.data.
    try:
        solver = adapter.solver
        output = adapter.output
        if solver is None: return {'schema':'motus.g1-visualization.v1','available':False,'reason':'calibration_missing'}
        state = adapter.link.feedback()
        feedback = state.get('feedback', {})
        now = time.monotonic_ns()
        age = (now-feedback.get('arm_ns',0))/1e6
        fresh = 0 <= age <= getattr(adapter.link,'feedback_max_age_ns',100_000_000)/1e6
        preview = solver.visualization_sample
        intent_age = (now-preview['monotonic_ns'])/1e6 if preview else None
        targets = [np.asarray(t)[:3,3].tolist() for t in preview['targets']] if preview else []
        return {'schema':'motus.g1-visualization.v1','available':True,
                'mode':'live' if adapter.hardware_output else 'shadow',
                'state':output.get('state','idle'),
                'reason':output.get('code') or state.get('reason') or '',
                'feedback_fresh':fresh,'feedback_age_ms':max(0.,age),
                'feedback_source':state.get('feedback_source','teleop_executor'),
                'source_timestamp_available':state.get('source_timestamp_available',True),
                'intent_age_ms':intent_age,'solve_ms':solver.last_ms,
                'measured':chains(solver,feedback['q']) if fresh else [],
                'ik':chains(solver,preview['ik_q']) if preview and preview.get('ik_q') is not None else [],
                'command':chains(solver,output['target_q']) if output.get('target_q') is not None else [],
                'targets':targets,
                'collision_reason':(solver.last_collision_rejection or {}).get('reason','')}
    except (ValueError,KeyError,TypeError):
        return {'schema':'motus.g1-visualization.v1','available':False,'reason':'feedback_unavailable'}
