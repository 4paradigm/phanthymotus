"""Experimental gates must be explicit and reject missing or invalid evidence."""
import pytest

from tools.pointcloud_slam_geometry_gate import classify_registration, summarize


def diagnostic():
    return {'forward': {'fitness': .8, 'rmse_m': .02, 'correspondences': 100},
            'reverse': {'fitness': .75, 'rmse_m': .025, 'correspondences': 90},
            'cycle_translation_m': .01, 'cycle_angle_deg': .5}


def thresholds():
    return {'min_fitness': .6, 'max_rmse_m': .05, 'min_correspondences': 30,
            'max_cycle_translation_m': .03, 'max_cycle_angle_deg': 2.}


def test_explicit_experimental_gates_use_both_registration_directions():
    assert classify_registration(diagnostic(), thresholds())['passes'] is True
    bad = diagnostic()
    bad['reverse']['fitness'] = .1
    result = classify_registration(bad, thresholds())
    assert result['passes'] is False and 'reverse_fitness' in result['failed']


@pytest.mark.parametrize('change', [
    lambda d: d['forward'].update(rmse_m=None),
    lambda d: d['reverse'].update(fitness=float('nan')),
    lambda d: d['forward'].update(correspondences=0),
    lambda d: d.update(cycle_translation_m=None),
])
def test_missing_or_invalid_geometry_cannot_pass(change):
    d = diagnostic()
    change(d)
    assert classify_registration(d, thresholds())['passes'] is False


def test_invalid_thresholds_are_rejected_instead_of_silently_disabled():
    bad = thresholds()
    bad['min_fitness'] = float('nan')
    with pytest.raises(ValueError):
        classify_registration(diagnostic(), bad)


def test_confusion_counts_require_both_classes_and_do_not_claim_accuracy_without_them():
    items = [{'match': True, 'passes': True}, {'match': True, 'passes': False},
             {'match': False, 'passes': True}, {'match': False, 'passes': False}]
    assert summarize(items) == {'tp': 1, 'fn': 1, 'fp': 1, 'tn': 1,
                                'recall': .5, 'false_positive_rate': .5}
    result = summarize(items[:2])
    assert result['false_positive_rate'] is None
    with pytest.raises(ValueError):
        summarize([{'match': 'unknown', 'passes': True}])


def test_forward_depth_score_counts_missing_matches_and_requires_each_stratum():
    import numpy as np
    from plugins.pointcloud_slam_registration import _forward_depth_score

    points = np.array([[1., 0, 0], [3., 0, 0], [5., 0, 0], [6., 0, 0]])
    pairs = np.array([[0, 0], [1, 1], [2, 2]])
    assert _forward_depth_score(points, points, pairs, np.eye(4), .3) == pytest.approx(.3 / np.sqrt(6))
    assert _forward_depth_score(points[:2], points, pairs[:2], np.eye(4), .3) is None


def test_profile_score_does_not_bypass_cycle_or_global_default():
    gate = thresholds()
    del gate['max_rmse_m']
    gate['max_forward_depth_score_m'] = .09
    d = diagnostic()
    for key in ['forward', 'reverse']:
        d[key].update(rmse_m=.10, forward_depth_score_m=.082)
    assert classify_registration(d, gate)['passes']
    assert not classify_registration(d, thresholds())['passes']
    d['cycle_translation_m'] = .08
    assert classify_registration(d, gate)['failed'] == ['cycle_translation_m']
    del d['reverse']['forward_depth_score_m']
    assert 'reverse_depth_score' in classify_registration(d, gate)['failed']
