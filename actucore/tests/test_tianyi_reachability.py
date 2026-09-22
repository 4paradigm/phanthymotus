"""Vendor-model saturation and runtime recovery; no robot I/O."""
import copy
import json
import time

import numpy as np
import pytest

from test_tianyi_arm_root import profile
from test_teleop import frame
from teleop.dispatch import AdapterAck, RecordingAdapter
from teleop.kinematics import TianyiIK
from teleop.protocol import bind_rtc_frame_v1
from teleop.runtime import TeleopRuntime
from teleop.workspace import WorkspaceViolation


def solver_at(tmp_path, **settings):
    path, p, _ = profile(tmp_path)
    p['target_projection'] = {'enabled': True, **settings}
    path.write_text(json.dumps(p))
    return TianyiIK(path)


def forward(solver, distance):
    targets = solver.palms(np.zeros(14))
    targets[0][0, 3] += distance
    return targets


def test_far_hand_saturates_without_drift_then_resumes_after_stable_reentry(tmp_path):
    solver = solver_at(tmp_path)
    measured = np.zeros(14)
    raw = forward(solver, .2)
    before = copy.deepcopy(raw)
    command = np.asarray(solver.solve(raw, measured))
    assert solver.target_diagnostics['state'] == 'saturated'
    assert solver.target_diagnostics['residuals'][0]['position_m'] > .015
    assert np.max(np.abs(command-measured)) <= solver.velocity*.02+1e-12
    boundary = solver.target_policy.boundary['q'].copy()
    np.testing.assert_array_equal(raw, before)
    for distance in (.3, .4, .25):
        solver.solve(forward(solver, distance), measured)
        np.testing.assert_array_equal(solver.target_policy.boundary['q'], boundary)
    for index in range(3):
        solver.solve(forward(solver, 0), measured)
        assert solver.target_diagnostics['state'] == ('tracking' if index == 2 else 'saturated')
    assert solver.target_policy.boundary is None
    actual = solver.palms(solver.last_valid_visualization['ik_q'])
    assert all(r['position_m'] < .0015 for r in solver.target_policy.residuals(actual, forward(solver, 0)))


def test_reentry_chatter_does_not_release_and_reclutch_clears_saturation(tmp_path):
    solver = solver_at(tmp_path)
    solver.solve(forward(solver, .2), np.zeros(14))
    for distance in (0, 0, .3, 0, 0):
        solver.solve(forward(solver, distance), np.zeros(14))
        assert solver.target_diagnostics['state'] == 'saturated'
    solver.reset_target_state()
    solver.solve(forward(solver, 0), np.zeros(14))
    assert solver.target_diagnostics['state'] == 'tracking'


def test_workspace_projection_has_inset_and_preserves_original_goal(tmp_path):
    solver = solver_at(tmp_path)
    solver.workspace['left'][1][0] = .08
    raw = forward(solver, .2)
    solver.solve(raw, np.zeros(14))
    state = solver.target_diagnostics
    assert state['state'] == 'saturated' and 'workspace' in state['reasons']
    assert state['raw_targets'][0][0][3] > .2
    assert state['feasible_targets'][0][0][3] < .08
    solver._safe_configuration(solver.target_policy.boundary['q'])


def test_projection_cannot_bypass_full_configuration_collision(tmp_path, monkeypatch):
    solver = solver_at(tmp_path)
    def collision(q, **kwargs):
        raise WorkspaceViolation('torso_collision', np.ones(14))
    monkeypatch.setattr(solver, '_safe_configuration', collision)
    with pytest.raises(ValueError, match='torso_collision'):
        solver.solve(forward(solver, .2), np.zeros(14))
    assert solver.target_policy.boundary is None
    assert solver.last_valid_visualization is None
    assert solver.target_diagnostics['state'] == 'rejected'


def test_expired_budget_and_failed_optimization_never_create_boundary(tmp_path, monkeypatch):
    solver = solver_at(tmp_path)
    raw = forward(solver, .2)
    with pytest.raises(ValueError, match='ik_timeout'):
        solver.solve(raw, np.zeros(14), deadline_monotonic=time.monotonic()-1)
    original = solver.least_squares
    def exhausted(*args, **kwargs):
        result = original(*args, **kwargs)
        result.success = False
        result.status = 0
        return result
    monkeypatch.setattr(solver, 'least_squares', exhausted)
    with pytest.raises(ValueError, match='ik_not_converged'):
        solver.solve(raw, np.zeros(14))
    assert solver.target_policy.boundary is None


@pytest.mark.parametrize('settings', [dict(enabled='true'), dict(joint_margin_rad=float('nan')),
    dict(workspace_margin_m=0), dict(release_frames=True), dict(unknown=1)])
def test_invalid_projection_policy_is_rejected(tmp_path, settings):
    with pytest.raises(ValueError, match='target_projection'):
        solver_at(tmp_path, **settings)


@pytest.mark.parametrize('stop_ok', [True, False])
def test_workspace_reentry_requires_confirmed_stop_and_keeps_same_clutch(stop_ok):
    class Adapter(RecordingAdapter):
        auto_workspace_recovery = True
        blocked = True
        def apply(self, intent):
            return AdapterAck(False, 'workspace_limit') if self.blocked else super().apply(intent)
        def safe_stop(self, request):
            if request.reason == 'workspace_limit' and not stop_ok:
                return AdapterAck(False, 'stop_unconfirmed')
            return super().safe_stop(request)
    adapter = Adapter()
    runtime = TeleopRuntime(mode='shadow', adapter=adapter, auto_watchdog=False)
    try:
        runtime.prepare_local_session()
        binding, _ = runtime.rtc_authority_snapshot()
        def send(seq, held=True):
            return runtime.submit_frame(bind_rtc_frame_v1(
                frame(seq, held, 0 if seq == 0 else 1), authority=binding,
                expected_mode='shadow'), source='test')
        send(0, False)
        send(1)
        deadline = time.monotonic()+.5
        while time.monotonic() < deadline:
            status = runtime.status()
            if status['dispatch'].get('stop_acknowledged') or status['dispatch'].get('fault_code'):
                break
            time.sleep(.002)
        adapter.blocked = False
        send(2)
        if stop_ok:
            assert runtime._dispatcher.wait_dispatched(2)
            assert runtime.status()['state'] == 'active_shadow'
        else:
            assert runtime.status()['dispatch']['last_would_apply_sequence'] is None
    finally:
        runtime.close()


def test_recording_keeps_raw_and_solved_sequences_separate(tmp_path):
    from teleop.recording import PoseRecorder
    recorder = PoseRecorder(tmp_path)
    recorder.start({}, duration=1)
    raw = frame(3)
    recorder.capture(raw, time.monotonic(), {'secret': 'do-not-record'})
    recorder.capture(frame(4), time.monotonic(), {})
    event = {'input_sequence': 3, 'ik_target_q': [0.]*14, 'published': False,
             'target_diagnostics': {'state': 'saturated'}, 'secret': 'do-not-record',
             'publish': {'mac': 'do-not-record'}}
    recorder.capture_solution(event)
    event['target_diagnostics']['state'] = 'changed-after-observation'
    result = recorder.stop()
    poses = [json.loads(x) for x in (recorder.path/'poses.jsonl').read_text().splitlines()]
    solutions = [json.loads(x) for x in (recorder.path/'solutions.jsonl').read_text().splitlines()]
    assert result['complete'] and result['frames'] == 2 and result['solutions'] == 1
    assert poses[0]['frame'] == raw
    assert solutions[0]['input_sequence'] == 3
    assert solutions[0]['target_diagnostics']['state'] == 'saturated'
    assert 'do-not-record' not in json.dumps([poses, solutions])


def test_solution_queue_loss_is_reported_without_blocking():
    import queue
    from teleop.recording import PoseRecorder
    recorder = PoseRecorder('/unused', capacity=1)
    recorder.result = {'state': 'recording', 'solutions_dropped': 0}
    recorder.until = time.monotonic()+1
    recorder.queue = queue.Queue(1)
    recorder.capture_solution({'input_sequence': 1})
    recorder.capture_solution({'input_sequence': 2})
    assert recorder.result['solutions_dropped'] == 1
