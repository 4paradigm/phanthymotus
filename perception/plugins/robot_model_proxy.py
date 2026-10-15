"""Parent-process handle for the isolated robot image encoder."""

from __future__ import annotations

import os

from plugins.robot_model_service import load_candidate_vectors

SERVICE_KEY = "robot_model.service"


class RobotModelProxy:
    def __init__(self, model_dir: str, on_status=None):
        from plugins import ort_worker
        from utils.model_downloader import ensure_robot_model_bundle
        from utils.onnx_provider import ort_providers_for_device

        paths = ensure_robot_model_bundle(model_dir, progress_cb=on_status)
        self.vectors = load_candidate_vectors(paths["candidates.npz"])
        self._worker = ort_worker.get_worker()
        self._worker.service(
            SERVICE_KEY, "plugins.robot_model_service", "RobotModelService",
            model_dir=os.path.dirname(paths["clip_image.onnx"]),
            providers=ort_providers_for_device("auto"),
        )

    def embed(self, image_bytes: bytes) -> list[float]:
        return self._worker.call(SERVICE_KEY, "embed", {"image_bytes": image_bytes})

    def close(self) -> None:
        self._worker.drop(SERVICE_KEY)
