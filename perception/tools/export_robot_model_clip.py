"""Build the image-only CLIP bundle for local robot model classification.

Build-time dependencies: openai/CLIP, torch, onnx, onnxruntime and Pillow.
The robot image needs only Pillow, NumPy and its existing ONNX Runtime worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.robot_model_catalog import CANDIDATES  # noqa: E402
from plugins.robot_model_service import load_candidate_vectors  # noqa: E402


OFFICIAL_CHECKPOINT_SHA256 = (
    "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
)


def file_pin(path: str) -> dict:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"size": size, "sha256": digest.hexdigest()}


def save_candidate_vectors(path: str, ids: list[str], vectors: np.ndarray) -> None:
    if ids != [model_id for model_id, _ in CANDIDATES]:
        raise ValueError("candidate IDs differ from the catalogue")
    values = np.asarray(vectors, dtype=np.float32)
    if values.shape != (len(ids), 512) or not np.isfinite(values).all():
        raise ValueError("candidate vectors have invalid shape or values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("candidate vectors include a zero vector")
    np.savez(path, model_ids=np.asarray(ids), vectors=values / norms)
    load_candidate_vectors(path)


def export_bundle(checkpoint_path: str, output_dir: str) -> dict:
    """Export image encoder and fixed text vectors from one verified checkpoint."""
    import clip
    import onnxruntime as ort
    import torch

    checkpoint_pin = file_pin(checkpoint_path)
    if checkpoint_pin["sha256"] != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA256 does not match official CLIP ViT-B/32")

    model, _ = clip.load(checkpoint_path, device="cpu", jit=False)
    model.eval()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ids = [model_id for model_id, _ in CANDIDATES]
    prompts = [prompt for _, prompt in CANDIDATES]
    with torch.no_grad():
        text_vectors = model.encode_text(clip.tokenize(prompts)).float().numpy()
    vector_path = output / "candidates.npz"
    save_candidate_vectors(str(vector_path), ids, text_vectors)

    # Export the registered visual submodule. A wrapper that captures `model`
    # in a closure makes ONNX treat its parameters as constants and refuse
    # those requiring gradients.
    encoder = model.visual.eval()
    onnx_path = output / "clip_image.onnx"
    sample = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    torch.onnx.export(encoder, sample, str(onnx_path), opset_version=17,
                      input_names=["image"], output_names=["embedding"],
                      dynamic_axes={"image": {0: "batch"},
                                    "embedding": {0: "batch"}})

    rng = np.random.default_rng(20261015)
    probe = rng.standard_normal((2, 3, 224, 224)).astype(np.float32)
    session = ort.InferenceSession(str(onnx_path),
                                   providers=["CPUExecutionProvider"])
    actual = session.run(None, {"image": probe})[0]
    with torch.no_grad():
        expected = model.encode_image(torch.from_numpy(probe)).float().numpy()
    error = float(np.max(np.abs(actual - expected)))
    if error > 1e-3:
        raise RuntimeError(f"ONNX differs from source CLIP: max abs error {error}")

    manifest = {
        "source": "openai/CLIP ViT-B/32",
        "source_checkpoint_sha256": checkpoint_pin["sha256"],
        "candidate_ids": ids,
        "onnx_max_abs_error": error,
        "files": {
            "clip_image.onnx": file_pin(str(onnx_path)),
            "candidates.npz": file_pin(str(vector_path)),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="official OpenAI ViT-B-32.pt")
    parser.add_argument("output_dir", help="directory for the exported bundle")
    args = parser.parse_args()
    print(json.dumps(export_bundle(args.checkpoint, args.output_dir), indent=2))
