#!/usr/bin/env python3
"""Offline only: recorded Tianyi state + synthetic reachable OpenXR trajectory.

No transport/ROS/device initialization. Never treat synthesized inputs as a
recording of an operator trial or the synthetic plant as hardware acceptance.
"""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'actucore/plugins'))
from teleop.kinematics import RelativeMapping, TianyiIK
from teleop.tianyi_visualization import snapshot


def replay(profile, recording, destination):
    rows = [json.loads(line) for line in recording.read_text().splitlines()]
    observations = [r['driver']['feedback']['q'] for r in rows
                    if len(r.get('driver', {}).get('feedback', {}).get('q', [])) == 14]
    if not observations:
        raise ValueError('recording_has_no_14_joint_feedback')
    measured = np.asarray(observations[0])
    solver = TianyiIK(profile)
    pose = {'position': [0., 1., 0.], 'orientation': [0., 0., 0., 1.]}
    frame = {'head': copy.deepcopy(pose), 'left_controller': copy.deepcopy(pose),
             'right_controller': copy.deepcopy(pose)}
    mapper = RelativeMapping()
    mapper.reset(frame, solver.palms(measured))
    feedback = {'q': measured.tolist(), 'arm_ns': time.monotonic_ns()}
    adapter = SimpleNamespace(solver=solver, output={}, hardware_output=False,
                              link=SimpleNamespace(feedback=lambda: {'feedback': feedback}))
    samples, errors, durations = [], [], []
    for joint in (0, 1, 3, 7, 8, 10):
        for phase in np.linspace(0, 2*np.pi, 31):
            # Known reachable poses from vendor FK, inverted into controller
            # coordinates and passed back through the real relative mapper.
            expected_q = measured.copy()
            expected_q[joint] += .12*np.sin(phase)
            expected = solver.palms(expected_q)
            moved = copy.deepcopy(frame)
            for side, (origin, base), target in zip(('left', 'right'), mapper.reference, expected):
                position = origin[:3, 3] + mapper.rotation.T @ (target[:3, 3]-base[:3, 3])/mapper.scale
                rotation = mapper.rotation.T @ target[:3, :3] @ base[:3, :3].T @ mapper.rotation @ origin[:3, :3]
                moved[side+'_controller'] = {'position': position.tolist(),
                    'orientation': Rotation.from_matrix(rotation).as_quat().tolist()}
            targets = mapper.targets(moved)
            for actual, desired in zip(targets, expected):
                np.testing.assert_allclose(actual, desired, atol=1e-10)
            command = solver.solve(targets, measured)
            assert max(abs(np.asarray(command)-measured)) <= solver.velocity*.02+1e-12
            solved = solver.palms(solver.visualization_sample['ik_q'])
            errors.extend(float(np.linalg.norm(a[:3, 3]-b[:3, 3])) for a, b in zip(solved, expected))
            durations.append(solver.last_ms)
            feedback['arm_ns'] = time.monotonic_ns()
            adapter.output = {'state': 'would_apply'}
            view = snapshot(adapter)
            assert len(view['ik']) == len(view['held_ik']) == 2
            samples.append(view)
    # Real unreachable target: reject command, retain only labelled historical IK.
    invalid = solver.palms(measured)
    for target in invalid:
        target[0, 3] += 3.
    try:
        solver.solve(invalid, measured)
    except ValueError as exc:
        rejection = str(exc)
    else:
        raise AssertionError('unreachable_target_was_accepted')
    adapter.output = {'state': 'held', 'code': rejection}
    failed = snapshot(adapter)
    assert failed['ik'] == [] and len(failed['held_ik']) == 2
    samples.append(failed)
    # A valid next input restores live preview without using the failed solution.
    solver.solve(solver.palms(measured), measured)
    feedback['arm_ns'] = time.monotonic_ns()
    adapter.output = {'state': 'would_apply'}
    recovered = snapshot(adapter)
    assert len(recovered['ik']) == 2
    samples.append(recovered)
    report = {
        'hardware_output': False, 'input_kind': 'synthetic_reachable_openxr_from_vendor_fk',
        'recording_sha256': hashlib.sha256(recording.read_bytes()).hexdigest(),
        'recorded_rows': len(rows), 'recorded_joint_observations': len(observations),
        'recorded_max_joint_excursion_rad': float(np.max(np.ptp(observations, axis=0))),
        'recorded_driver_reasons': dict(collections.Counter(str(r.get('driver', {}).get('reason')) for r in rows)),
        'reachable_frames': len(durations), 'reachable_passed': len(durations),
        'max_tcp_error_m': max(errors), 'ik_p95_ms': float(np.percentile(durations, 95)),
        'failed_target_rejected': rejection, 'held_preview_preserved': True,
        'next_valid_preview_restored': True, 'raw_operator_pose_replay': False,
        'physical_acceptance': False,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (destination/'visualizations.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in samples))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('recording', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.profile, args.recording, args.destination), indent=2))
