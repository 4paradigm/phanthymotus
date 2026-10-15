"""CLIP image preprocessing and candidate bundle validation."""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from plugins.robot_model_catalog import CANDIDATES
from plugins.robot_model_service import load_candidate_vectors, preprocess_image


def test_preprocess_uses_clip_rgb_normalization():
    image = Image.new("RGB", (300, 240), (255, 0, 0))
    source = BytesIO()
    image.save(source, format="PNG")

    blob = preprocess_image(source.getvalue())

    assert blob.shape == (1, 3, 224, 224)
    assert blob.dtype == np.float32
    assert blob[0, :, 112, 112] == pytest.approx([
        (1 - 0.48145466) / 0.26862954,
        (0 - 0.4578275) / 0.26130258,
        (0 - 0.40821073) / 0.27577711,
    ], abs=1e-5)


def test_preprocess_uses_clip_floor_for_non_square_resize():
    pixels = np.random.default_rng(4).integers(0, 256, (240, 321, 3),
                                            dtype=np.uint8)
    image = Image.fromarray(pixels)
    source = BytesIO()
    image.save(source, format="PNG")
    # OpenAI CLIP's torchvision Resize(224) makes this 299x224, then
    # CenterCrop rounds the 37.5-pixel offset to 38 (columns 38..261).
    expected = np.asarray(
        image.resize((299, 224), Image.Resampling.BICUBIC)
             .crop((38, 0, 262, 224)), dtype=np.float32) / 255.0
    mean = np.array((0.48145466, 0.4578275, 0.40821073))
    std = np.array((0.26862954, 0.26130258, 0.27577711))
    expected = ((expected - mean) / std).transpose(2, 0, 1)[None]

    assert preprocess_image(source.getvalue()) == pytest.approx(expected, abs=1e-5)


def test_unreadable_crop_is_rejected():
    with pytest.raises(ValueError, match="decode"):
        preprocess_image(b"not an image")


def test_candidate_vectors_reject_catalogue_order_drift(tmp_path):
    path = tmp_path / "candidates.npz"
    ids = [model_id for model_id, _ in CANDIDATES]
    vectors = np.zeros((len(ids), 512), dtype=np.float32)
    vectors[:, 0] = 1
    np.savez(path, model_ids=ids[::-1], vectors=vectors)

    with pytest.raises(ValueError, match="catalogue"):
        load_candidate_vectors(str(path))


def test_candidate_vectors_require_unit_length(tmp_path):
    path = tmp_path / "candidates.npz"
    ids = [model_id for model_id, _ in CANDIDATES]
    vectors = np.zeros((len(ids), 512), dtype=np.float32)
    vectors[:, 0] = 2
    np.savez(path, model_ids=ids, vectors=vectors)

    with pytest.raises(ValueError, match="normalized"):
        load_candidate_vectors(str(path))
