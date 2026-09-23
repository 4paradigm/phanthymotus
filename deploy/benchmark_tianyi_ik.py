"""Deterministic offline IK benchmark. No ROS, sockets or hardware output."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'actucore/plugins'))
from teleop.kinematics import TianyiIK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('calibration')
    parser.add_argument('--q', required=True, help='JSON file containing recorded q[14]')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    solver = TianyiIK(args.calibration)
    measured = np.array(json.loads(Path(args.q).read_text()))
    origin = solver.palms(measured)
    cases = []
    for axis in range(3):
        for sign in (-1, 1):
            target = [t.copy() for t in origin]
            for t in target:
                t[axis, 3] += sign * .02
            cases.append((f'translation_{axis}_{sign}', target))
            target = [t.copy() for t in origin]
            delta = np.zeros(3); delta[axis] = sign * .08
            for t in target:
                t[:3, :3] = solver.pin.exp3(delta) @ t[:3, :3]
            cases.append((f'rotation_{axis}_{sign}', target))
    for index in range(8):
        q = measured + .12 * np.sin(np.arange(14) + index)
        q = np.clip(q, solver.model.lowerPositionLimit[solver.indices] + 1e-6,
                    solver.model.upperPositionLimit[solver.indices] - 1e-6)
        cases.append((f'reachable_fk_{index}', solver.palms(q)))
    solver.self_test(measured)
    rows = []
    for repeat in range(args.repeats):
        for name, targets in cases:
            start = time.monotonic()
            row = {'case': name, 'repeat': repeat}
            try:
                row['q'] = solver.solve(targets, measured)
                row['result'] = 'ok'
            except ValueError as exc:
                row['result'] = str(exc)
            row['elapsed_ms'] = (time.monotonic() - start) * 1000
            rows.append(row)
    counts = {code: sum(r['result'] == code for r in rows) for code in sorted({r['result'] for r in rows})}
    print(json.dumps({'hardware_output': False, 'counts': counts,
                      'p95_ms': float(np.percentile([r['elapsed_ms'] for r in rows], 95)),
                      'rows': rows}, allow_nan=False))


if __name__ == '__main__':
    main()
