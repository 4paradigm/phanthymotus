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
