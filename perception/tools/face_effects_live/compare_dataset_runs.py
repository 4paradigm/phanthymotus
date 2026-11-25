"""Re-score immutable prediction files and compare identical source cohorts."""
import argparse
import json
from pathlib import Path

import numpy as np

from dataset_adapters import sha256
from score_dataset import distribution, score


def paired(before, after):
    key = lambda r: (r['clip'], r['frame_id'], r['gt_id'])
    a = {key(r): r for r in before['per_face']}; b = {key(r): r for r in after['per_face']}
    if set(a) != set(b): raise ValueError('different source face cohorts')
    result = {}
    for metric in ('raw_face_mae_px', 'raw_nme_interocular', 'effect_anchor_mae_px', 'glasses_eye_mae_px'):
        keys = [k for k in a if a[k][metric] is not None and b[k][metric] is not None]
        old = [a[k][metric] for k in keys]; new = [b[k][metric] for k in keys]
        delta = [y-x for x, y in zip(old, new)]
        result[metric] = {'common_instances': len(keys), 'before': distribution(old),
                          'after': distribution(new), 'delta_after_minus_before': distribution(delta),
                          'after_better': sum(d < -1e-8 for d in delta),
                          'after_worse': sum(d > 1e-8 for d in delta)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', required=True)
    p.add_argument('--runs', nargs='+', required=True, help='NAME=/prediction/directory, first is paired baseline')
    p.add_argument('--output', required=True); a = p.parse_args()
    labels = json.loads(Path(a.labels).read_text()); scores = {}; summaries = {}; expected = None
    for value in a.runs:
        name, path = value.split('=', 1); path = Path(path)
        summary = json.loads((path/'summary.json').read_text())
        inputs = summary['source_input_sha256']
        if expected is not None and inputs != expected: raise ValueError('different input hashes: '+name)
        expected = inputs
        quality = score(path, labels); scores[name] = quality; summaries[name] = summary
        # Never replace a previous score artifact silently.
        target = path/'quality-final.json'
        with target.open('x') as f: json.dump(quality, f, indent=2, allow_nan=False)
    baseline = next(iter(scores))
    report = {'schema': 'face_effects_independent_comparison/1', 'labels_sha256': sha256(a.labels),
              'identical_input_sha256': expected, 'dataset': labels['dataset'],
              'baseline': baseline,
              'runs': {name: {'summary': summaries[name], 'quality': {k: v for k, v in value.items() if k != 'per_face'}} for name, value in scores.items()},
              'paired_to_baseline': {name: paired(scores[baseline], value) for name, value in scores.items() if name != baseline},
              'limits': ['common-match comparisons exclude missing outcomes; always inspect full-cohort coverage alongside them',
                         'this evaluation informed one optional resolution profile; not a fresh blind holdout',
                         'different topology definitions and partial GT prevent claiming standard dense face benchmark scores']}
    with Path(a.output).open('x') as f: json.dump(report, f, indent=2, allow_nan=False)


if __name__ == '__main__': main()
