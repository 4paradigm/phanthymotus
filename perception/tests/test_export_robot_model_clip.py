"""Artifact metadata cannot drift away from candidate IDs or file bytes."""

import hashlib

import numpy as np
import pytest

from plugins.robot_model_catalog import CANDIDATES
from plugins.robot_model_service import load_candidate_vectors
from tools.export_robot_model_clip import file_pin, save_candidate_vectors


def test_saved_text_vectors_follow_catalogue_order(tmp_path):
    ids = [model_id for model_id, _ in CANDIDATES]
    raw = np.zeros((len(ids), 512), dtype=np.float32)
    raw[:, 0] = 3
    raw[:, 1] = 4
    path = tmp_path / "candidates.npz"

    save_candidate_vectors(str(path), ids, raw)

    loaded = load_candidate_vectors(str(path))
    assert loaded[0, :2] == pytest.approx([0.6, 0.8])


def test_export_rejects_missing_candidate_vector(tmp_path):
    ids = [model_id for model_id, _ in CANDIDATES][:-1]
    raw = np.ones((len(ids), 512), dtype=np.float32)
    with pytest.raises(ValueError, match="catalogue"):
        save_candidate_vectors(str(tmp_path / "candidates.npz"), ids, raw)


def test_file_pin_is_hash_of_artifact_bytes(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"fixed artifact bytes")
    assert file_pin(str(path)) == {
        "size": 20,
        "sha256": hashlib.sha256(b"fixed artifact bytes").hexdigest(),
    }
