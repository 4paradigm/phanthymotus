#!/usr/bin/env python3
"""
plugins/face_proxy.py — the parent's handle on face recognition running elsewhere.

`plugins/face_service.py` runs the whole per-frame pipeline inside
`plugins/ort_worker.py`'s child, because the standalone ONNX Runtime cannot share a
process with sherpa-onnx's (see that module for the mechanism, and for the SIGSEGV it
causes on jp5.11). This is what `plugins/face.py` holds instead of a `FaceAnalyzer`.

It reconstructs `DetectedFace` objects from the child's reply, so everything downstream
— `bbox_xywh()`, `min_side`, `blur`, the subject selection, the payload shape — keeps
working unchanged. The quality gate ran in the child, and its verdict arrives as
`embedding is None` rather than as a separate flag: an unusable face is precisely one
that was not worth embedding.

One round trip per frame, carrying the JPEG the parent already had. The superseded
version of this split at `InferenceSession.run` instead and cost +27 ms per detection
and +5.1 ms per face; the numbers and the reasoning are in `face_service.py`.
"""

from __future__ import annotations

import logging

import numpy as np

from plugins.face_runtime import DetectedFace

log = logging.getLogger(__name__)

SERVICE_KEY = "face.service"


class FaceServiceProxy:
    """Stands in for `FaceAnalyzer`, but with one method instead of four.

    Deliberately *not* a drop-in for the old four-call surface. Offering
    `decode_image`/`detect`/`prepare`/`embed` here would mean shipping a 6 MB decoded
    frame back to the parent so it could hand pieces of it forward again, which is the
    mistake this replaces. The call is the pipeline.
    """

    def __init__(self, model_dir: str, device: str = "auto", num_threads: int = 2,
                 det_size=(640, 640), det_thresh: float = 0.5,
                 nms_thresh: float = 0.4, max_image_side: int = 2048,
                 max_image_pixels: float = 60e6, warmup: bool = True):
        from plugins import ort_worker
        from utils.onnx_provider import ort_providers_for_device, warn_on_parked_cores

        # Resolved here, in the parent, and passed down as a plain list. The child
        # should not have to ask the runtime what it offers — that question maps the
        # provider bridge, and the parent needs the answer anyway to report `device`.
        providers = ort_providers_for_device(device)
        self._device = "gpu" if providers[0] != "CPUExecutionProvider" else "cpu"
        # Session creation is where a parked-core Jetson abort()s under a bad ORT
        # version, and an abort prints no Python traceback. Say the precondition before
        # the child goes near it, so the last line before a silent death names it.
        warn_on_parked_cores("face")

        self._worker = ort_worker.get_worker()
        described = self._worker.service(
            SERVICE_KEY, "plugins.face_service", "build",
            model_dir=model_dir, providers=providers, num_threads=num_threads,
            det_size=tuple(det_size), det_thresh=det_thresh, nms_thresh=nms_thresh,
            max_image_side=max_image_side, max_image_pixels=max_image_pixels,
            warmup=warmup,
        ).describe()
        self._providers = list(described.get("providers") or providers)
        self._device = described.get("device", self._device)
        log.info("[face] service ready in the ORT worker: device=%s providers=%s",
                 self._device, self._providers)

    # ── the surface plugins/face.py reads ────────────────────────────────────

    @property
    def device(self) -> str:
        return self._device

    @property
    def providers(self) -> list:
        return list(self._providers)

    def recognise(self, image_bytes: bytes, max_faces: int = 0,
                  det_thresh: float = 0.5, min_face_px: int = 64,
                  blur_min: float = 60.0, want_aligned: bool = False,
                  embed_all: bool = False):
        """JPEG bytes -> `(shape, faces)`.

        `shape` is `(height, width)` of the decoded frame, or `None` if it could not be
        decoded — the caller reports "undecodable frame" for that and does not guess.
        `faces` are `DetectedFace`s with `embedding` set for the ones that passed the
        gate and `None` for the rest.
        """
        reply = self._worker.call(SERVICE_KEY, "recognise", {
            "image_bytes": image_bytes, "max_faces": max_faces,
            "det_thresh": det_thresh, "min_face_px": min_face_px,
            "blur_min": blur_min, "want_aligned": want_aligned,
            "embed_all": embed_all,
        })
        if not reply.get("decoded"):
            return None, []

        faces = []
        for entry in reply["faces"]:
            faces.append(DetectedFace(
                bbox=tuple(entry["bbox"]),
                det_score=entry["det_score"],
                kps=np.asarray(entry["kps"], dtype=np.float32),
                blur=entry["blur"],
                aligned=entry.get("aligned"),
                embedding=entry.get("embedding"),
            ))
        return (reply["height"], reply["width"]), faces

    def close(self) -> None:
        """Drop the service, releasing its models and its share of the GPU pool.

        The child stays: Kokoro's Japanese session may be in it, and a face card
        stopping must not take Japanese down with it.
        """
        try:
            self._worker.drop(SERVICE_KEY)
        except Exception as exc:                                  # noqa: BLE001
            log.warning("[face] dropping the face service failed: %s", exc)
