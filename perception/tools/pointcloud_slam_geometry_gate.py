"""Offline threshold experiments only; passing is not a validated loop edge."""
import math


def _number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def classify_registration(diagnostic, thresholds):
    """Apply explicit reciprocal-ICP gates; no threshold is a default or calibrated."""
    if not isinstance(thresholds, dict):
        raise ValueError('explicit thresholds required')
    fitness = thresholds.get('min_fitness')
    count = thresholds.get('min_correspondences')
    bounds = [thresholds.get(k) for k in
              ('max_rmse_m', 'max_cycle_translation_m', 'max_cycle_angle_deg')]
    if (not _number(fitness) or not 0 <= fitness <= 1 or
            type(count) is not int or count <= 0 or
            any(not _number(v) or v <= 0 for v in bounds)):
        raise ValueError('invalid geometry thresholds')
    if not isinstance(diagnostic, dict):
        raise ValueError('diagnostic must be an object')
    failed = []
    for direction in ('forward', 'reverse'):
        data = diagnostic.get(direction)
        data = data if isinstance(data, dict) else {}
        value = data.get('fitness')
        if not _number(value) or not fitness <= value <= 1:
            failed.append(direction + '_fitness')
        value = data.get('rmse_m')
        if not _number(value) or not 0 <= value <= bounds[0]:
            failed.append(direction + '_rmse')
        value = data.get('correspondences')
        if type(value) is not int or value < count:
            failed.append(direction + '_correspondences')
    for key, limit in (('cycle_translation_m', bounds[1]), ('cycle_angle_deg', bounds[2])):
        value = diagnostic.get(key)
        if not _number(value) or not 0 <= value <= limit:
            failed.append(key)
    return {'passes': not failed, 'failed': failed}


def summarize(items):
    """Summarize separately supplied labels; missing classes have unknown rates."""
    counts = {'tp': 0, 'fn': 0, 'fp': 0, 'tn': 0}
    for item in items:
        if (not isinstance(item, dict) or type(item.get('match')) is not bool or
                type(item.get('passes')) is not bool):
            raise ValueError('each labelled outcome needs boolean match and passes')
        key = ('tp' if item['passes'] else 'fn') if item['match'] else (
              'fp' if item['passes'] else 'tn')
        counts[key] += 1
    positives, negatives = counts['tp'] + counts['fn'], counts['fp'] + counts['tn']
    return {**counts, 'recall': counts['tp'] / positives if positives else None,
            'false_positive_rate': counts['fp'] / negatives if negatives else None}
