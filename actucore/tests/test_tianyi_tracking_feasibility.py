import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).parents[2]/'deploy/analyze_tianyi_tracking_feasibility.py'
spec = importlib.util.spec_from_file_location('tracking_feasibility', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rows(times, values):
    return [{'elapsed': t, 'ik_reference_q': [v]+[0.]*13} for t, v in zip(times, values)]


def test_fast_reference_has_analytic_impossibility_witness():
    result = module.analyze(rows([0., .2], [0., 1.2]))
    assert result['minimum_possible_max_error_rad'][0] == pytest.approx(.5)
    assert result['minimum_required_velocity_rad_s'][0] == pytest.approx(4.8)
    assert result['infeasible_joint_indices'] == [0]
    assert result['hardware_acceptance'] is False


def test_nonadjacent_samples_are_checked_and_zeros_are_not_success():
    result = module.analyze(rows([0., .1, .2], [0., .2, .4]))
    assert result['minimum_possible_max_error_rad'][0] == pytest.approx(.1)
    assert result['witnesses'][0]['first_row'] == 0
    assert result['witnesses'][0]['last_row'] == 2
    slow = module.analyze(rows([0., 2.], [0., 1.]))
    assert slow['infeasible_joint_indices'] == [] and slow['hardware_acceptance'] is False


@pytest.mark.parametrize('bad', [rows([0., 0.], [0., 1.]), rows([1., 0.], [0., 1.]),
                               rows([0., 1.], [0., float('nan')]), [{'elapsed': 0.}]])
def test_invalid_or_missing_evidence_cannot_pass(bad):
    with pytest.raises(ValueError):
        module.analyze(bad)


def test_tolerance_matches_acceptance_including_measured_baseline():
    result = module.analyze(rows([0., .2], [0., 1.2]), baseline_q=[-1.]+[0.]*13)
    assert result['baseline_included_in_tolerance'] is True
    assert result['tolerance_rad'][0] == pytest.approx(.22)
    assert result['minimum_required_velocity_rad_s'][0] == pytest.approx(3.8)
