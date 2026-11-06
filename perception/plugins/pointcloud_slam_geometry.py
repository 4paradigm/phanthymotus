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
    stratified = 'max_forward_depth_score_m' in thresholds
    if stratified and 'max_rmse_m' in thresholds:
        raise ValueError('select exactly one registration error metric')
    score_limit = 'max_forward_depth_score_m' if stratified else 'max_rmse_m'
    bounds = [thresholds.get(k) for k in
              (score_limit, 'max_cycle_translation_m', 'max_cycle_angle_deg')]
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
        value = data.get('forward_depth_score_m' if stratified else 'rmse_m')
        if not _number(value) or not 0 <= value <= bounds[0]:
            failed.append(direction + ('_depth_score' if stratified else '_rmse'))
        value = data.get('correspondences')
        if type(value) is not int or value < count:
            failed.append(direction + '_correspondences')
    for key, limit in (('cycle_translation_m', bounds[1]), ('cycle_angle_deg', bounds[2])):
        value = diagnostic.get(key)
        if not _number(value) or not 0 <= value <= limit:
            failed.append(key)
    return {'passes': not failed, 'failed': failed}


