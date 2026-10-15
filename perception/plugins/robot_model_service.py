"""CLIP image encoder hosted by the isolated ONNX Runtime worker."""

from __future__ import annotations

from io import BytesIO
import os

import numpy as np
from PIL import Image, UnidentifiedImageError

from plugins.robot_model_catalog import CANDIDATES


_MEAN = np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
_STD = np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Apply OpenAI CLIP's 224-pixel resize, centre crop and RGB scaling."""
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            image = source.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError("could not decode robot crop") from error
    width, height = image.size
    if not width or not height:
        raise ValueError("could not decode robot crop")
    scale = 224 / min(width, height)
    resized = image.resize((int(width * scale), int(height * scale)),
                           getattr(Image, "Resampling", Image).BICUBIC)
    left = round((resized.width - 224) / 2)
    top = round((resized.height - 224) / 2)
    rgb = np.asarray(resized.crop((left, top, left + 224, top + 224)),
                     dtype=np.float32) / 255.0
    return ((rgb - _MEAN) / _STD).transpose(2, 0, 1)[None].copy()


def load_candidate_vectors(path: str) -> np.ndarray:
    """Read embeddings only when their IDs and geometry match this build."""
    with np.load(path, allow_pickle=False) as bundle:
        ids = [str(value) for value in bundle["model_ids"]]
        vectors = np.asarray(bundle["vectors"], dtype=np.float32)
    if ids != [model_id for model_id, _ in CANDIDATES]:
        raise ValueError("candidate vector order differs from the catalogue")
    if vectors.shape != (len(CANDIDATES), 512) or not np.isfinite(vectors).all():
        raise ValueError("candidate vectors have invalid shape or values")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
        raise ValueError("candidate vectors are not normalized")
    return vectors


class RobotModelService:
    """One image encoder session in the ORT child, never in Perception parent."""

    def __init__(self, model_dir: str, providers: list[str]):
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            os.path.join(model_dir, "clip_image.onnx"), providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    def describe(self) -> dict:
        return {"providers": self._session.get_providers()}

    def embed(self, image_bytes: bytes) -> list[float]:
        blob = preprocess_image(image_bytes)
        value = np.asarray(
            self._session.run(None, {self._input_name: blob})[0],
            dtype=np.float32).reshape(-1)
        if value.shape != (512,) or not np.isfinite(value).all():
            raise ValueError("CLIP encoder returned an invalid embedding")
        norm = float(np.linalg.norm(value))
        if norm <= 0:
            raise ValueError("CLIP encoder returned a zero embedding")
        return (value / norm).tolist()
