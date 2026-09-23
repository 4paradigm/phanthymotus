"""Read-only velocity feasibility bound for complete IK targets in replay logs.

This reports necessary conditions, not a motion plan or hardware acceptance.
It does not modify the recording, tolerance, production controller or verdict.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze(rows, velocity=1.0, baseline_q=None):
    if not np.isfinite(velocity) or velocity <= 0:
        raise ValueError('invalid_velocity')
    samples = [r for r in rows if r.get('ik_reference_q') is not None]
    if len(samples) < 2:
        raise ValueError('full_ik_samples_missing')
    times = np.asarray([r['elapsed'] for r in samples], dtype=float)
    targets = np.asarray([r['ik_reference_q'] for r in samples], dtype=float)
    if (targets.shape != (len(samples), 14) or not np.isfinite(targets).all()
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError('invalid_target_time_series')
    extent = targets
    if baseline_q is not None:
        baseline_q = np.asarray(baseline_q, dtype=float)
        if baseline_q.shape != (14,) or not np.isfinite(baseline_q).all():
            raise ValueError('invalid_baseline')
        extent = np.vstack([baseline_q, targets])
    tolerance = np.maximum(.03, .1*np.ptp(extent, axis=0))
    lower = np.zeros(14)
    required = np.zeros(14)
    witnesses = [None]*14
    for end in range(1, len(samples)):
        dt = times[end]-times[:end]
        distance = np.abs(targets[end]-targets[:end])
        # |goal_b-goal_a| <= |err_a| + v*dt + |err_b|.
        bounds = np.maximum(0., (distance-velocity*dt[:, None])/2.)
        speeds = np.maximum(0., (distance-2*tolerance)/dt[:, None])
        required = np.maximum(required, np.max(speeds, axis=0))
        for joint in range(14):
            begin = int(np.argmax(bounds[:, joint]))
            if bounds[begin, joint] > lower[joint]:
                lower[joint] = bounds[begin, joint]
                witnesses[joint] = {'first_row': begin, 'last_row': end,
                    'first_elapsed_s': float(times[begin]), 'last_elapsed_s': float(times[end]),
                    'target_distance_rad': float(distance[begin, joint])}
    return {'schema': 'motus.teleop.velocity-feasibility.v1',
        'scope': 'observed_full_ik_targets_only', 'hardware_acceptance': False,
        'input_rows': len(rows), 'full_ik_rows': len(samples),
        'baseline_included_in_tolerance': baseline_q is not None,
        'velocity_rad_s': velocity, 'tolerance_rad': tolerance.tolist(),
        'minimum_possible_max_error_rad': lower.tolist(),
        'minimum_required_velocity_rad_s': required.tolist(),
        'infeasible_joint_indices': np.flatnonzero(lower > tolerance+1e-12).tolist(),
        'witnesses': witnesses,
        'limitations': 'A satisfied necessary condition does not prove feasibility; '
            'collision, acceleration, plant response, missing targets and initial conditions are not solved.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--velocity', type=float, default=1.)
    parser.add_argument('--baseline', type=Path, help='combined.baseline.json for the acceptance tolerance')
    args = parser.parse_args()
    raw = args.recording.read_bytes()
    baseline = args.baseline.read_bytes() if args.baseline else None
    baseline_q = json.loads(baseline)[-1]['q'] if baseline else None
    result = analyze([json.loads(line) for line in raw.splitlines() if line.strip()], args.velocity, baseline_q)
    result['source_sha256'] = hashlib.sha256(raw).hexdigest()
    if baseline is not None:
        result['baseline_sha256'] = hashlib.sha256(baseline).hexdigest()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
