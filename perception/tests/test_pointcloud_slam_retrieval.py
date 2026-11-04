"""Offline retrieval scores must not manufacture loop-closure labels."""
import numpy as np
import pytest

from tools.pointcloud_slam_retrieval import rank_candidates, evaluate_labels, scan_context_ring


def test_excludes_same_and_temporally_adjacent_frames_and_ranks_by_distance():
    descriptors = np.array([[0., 0.], [.1, 0.], [.2, 0.], [1., 0.]])
    ranked = rank_candidates(descriptors, [0., 1., 20., 30.], 0, 5., 2)
    assert [item['index'] for item in ranked] == [2, 3]
    assert ranked[0]['distance'] == pytest.approx(.2)


def test_scores_only_manually_labelled_revisits_and_reports_unlabelled():
    rankings = {0: [{'index': 4}, {'index': 5}], 1: [{'index': 6}, {'index': 7}]}
    metrics = evaluate_labels(rankings, [{'query': 0, 'positive': [5]},
                                         {'query': 1, 'positive': [9]}], top_k=2)
    assert metrics == {'labelled_queries': 2, 'hits': 1, 'top_k': 2, 'recall': .5}
    assert evaluate_labels(rankings, [], top_k=2)['recall'] is None


def test_rejects_duplicate_labels_and_unavailable_query():
    with pytest.raises(ValueError):
        evaluate_labels({0: []}, [{'query': 0, 'positive': [1]}, {'query': 0, 'positive': [2]}], top_k=5)
    with pytest.raises(ValueError):
        evaluate_labels({}, [{'query': 0, 'positive': [1]}], top_k=5)


def test_optical_frame_ring_is_yaw_invariant_and_rejects_empty_cloud():
    # Rotate optical x,z around vertical y: these samples remain in the same rings.
    cloud = np.array([[.4, -.2, 1.], [.7, .1, 2.], [-.3, -.1, 3.]])
    rotated = np.column_stack((cloud[:, 2], cloud[:, 1], -cloud[:, 0]))
    assert scan_context_ring(cloud) == pytest.approx(scan_context_ring(rotated))
    with pytest.raises(ValueError):
        scan_context_ring(np.empty((0, 3)))


def test_submap_cli_requires_explicit_time_and_extrinsic_assumptions(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    tool = Path(__file__).resolve().parents[1] / 'tools' / 'pointcloud_slam_retrieval.py'
    proc = subprocess.run([sys.executable, str(tool), str(tmp_path), '--submap-frames', '5'],
                          capture_output=True, text=True)
    assert proc.returncode != 0
    assert 'extrinsic' in proc.stderr and 'max-gap-ms' in proc.stderr


def test_self_match_cannot_be_used_as_a_positive_label():
    with pytest.raises(ValueError):
        evaluate_labels({0: [{'index': 0}]}, [{'query': 0, 'positive': [0]}], top_k=5)


def test_nonfinite_capture_timestamp_is_not_a_loop_candidate():
    with pytest.raises(ValueError):
        rank_candidates(np.array([[0.], [1.]]), [0., float('nan')], 0, 1., 5)
