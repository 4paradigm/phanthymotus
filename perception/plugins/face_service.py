#!/usr/bin/env python3
"""
plugins/face_service.py — face recognition, entirely inside the ORT worker child.

## Why the boundary is here and not further in

An earlier version of this change put only the two `InferenceSession.run` calls in the
child and left decoding, letterboxing, alignment and the quality gate in the parent.
That works, and it was measured, and it was the wrong place to cut:

    detect p50   in-process 38.3 ms   ->   split at run()  65.3 ms   (+27 ms)
    embed  p50   in-process  7.83 ms  ->   split at run()  12.93 ms  (+5.1 ms)

because the parent was sending a 2.93 MiB normalised blob and getting 0.96 MiB of
pre-threshold candidates back, then sending a 147 kB crop per face and getting a 2 kB
embedding back — one round trip for detection plus one per face, with the largest
possible payload in both directions.

What the parent actually holds is the **`CompressedImage` JPEG**, 50-300 kB
(`plugins/face.py:920-932`), and what it actually wants is a few bounding boxes and a
512-float embedding each. So the frame crosses once, compressed, and the results come
back small:

    in   the JPEG bytes as received                        50-300 kB
    out  per face: bbox, score, blur, min_side, 512 f32    ~2 kB

The quality gate comes with it, deliberately. It sits between alignment and embedding
(`face.py:1017-1021`) and decides which faces are worth embedding at all; leaving it in
the parent would mean either embedding faces the parent then discards, or a second round
trip to ask.

## What stays in the parent

The FaceDB — `matrix @ embedding` on a 2 kB vector, plus the identity store, its lock
and its files (`plugins/face_db.py`). It needs nothing from ONNX, it is mutated from MCP
threads as well as the recognition worker, and it owns files on disk. Also the subject
selection, the payload shape, ROS publishing and the enrolment window: all cheap, all
pure, all already there.

## What this must never import

`sherpa_onnx`. This process exists to be the only ONNX Runtime in its address space —
see `plugins/ort_worker.py` for what happens otherwise. `plugins/face_runtime.py` is
safe to import: its only path to sherpa is `utils.onnx_provider.cuda_available`, whose
import is inside the function, and nothing here calls it. Providers arrive from the
parent as a plain list, so no provider helper is needed at all.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("face.service")


class FaceService:
    """One `FaceAnalyzer` and the whole per-frame pipeline, in the worker child.

    Created by `plugins/ort_worker.py` on request; the parent talks to it through
    `FaceServiceProxy` in `plugins/face_proxy.py` and never sees a session.
    """

    def __init__(self, model_dir: str, providers, num_threads: int = 2,
                 det_size=(640, 640), det_thresh: float = 0.5,
                 nms_thresh: float = 0.4, max_image_side: int = 2048,
                 max_image_pixels: float = 60e6, warmup: bool = True,
                 model: str = "buffalo_sc"):
        from plugins.face_runtime import FaceAnalyzer

        # The decode limits are per-call on decode_image, not analyzer state, so they
        # are held here and passed through. They are not cosmetic: max_side bounds the
        # letterbox downscale and max_pixels is the decompression-bomb guard.
        self._max_side = int(max_image_side)
        self._max_pixels = int(max_image_pixels)

        # in_process=True is correct *here*: this is the isolated process, so a session
        # created in it is exactly what the design wants. The flag exists so the same
        # class refuses to create one in the perception process by accident.
        self._analyzer = FaceAnalyzer(
            model_dir=model_dir, model=model, providers=providers,
            num_threads=num_threads, det_size=det_size, det_thresh=det_thresh,
            nms_thresh=nms_thresh, warmup=warmup, in_process=True,
        )
        log.info("face service ready: device=%s providers=%s",
                 self._analyzer.device, self._analyzer.providers)

    # ── metadata the parent reports in `info` ────────────────────────────────

    def describe(self) -> dict:
        return {"device": self._analyzer.device,
                "providers": list(self._analyzer.providers)}

    # ── the one call per frame ───────────────────────────────────────────────

    def recognise(self, image_bytes: bytes, max_faces: int = 0,
                  det_thresh: float = 0.5, min_face_px: int = 64,
                  blur_min: float = 60.0, want_aligned: bool = False,
                  embed_all: bool = False) -> dict:
        """JPEG bytes in, one dict per face out. No arrays cross except embeddings.

        Returns `{"width", "height", "faces": [...]}` where each face carries
        `bbox` (xyxy floats), `det_score`, `blur`, `min_side`, `usable`, and
        `embedding` (512 floats) for the usable ones — `None` otherwise, because an
        unusable face is reported with `person_id: null` and never embedded
        (`face.py:1028-1042`).

        `want_aligned` adds the 112x112x3 uint8 crop, 37 kB each. Only the registration
        paths need it, to store as a sample, so it is opt-in rather than always paid.

        `embed_all` overrides the gate for the identify/register paths that want an
        embedding even for a marginal face.
        """
        image = self._analyzer.decode_image(
            image_bytes, max_side=self._max_side,
            max_pixels=self._max_pixels)
        if image is None:
            return {"width": 0, "height": 0, "faces": [], "decoded": False}

        height, width = image.shape[:2]
        faces = self._analyzer.detect(image, max_faces=max_faces)

        out = []
        for face in faces:
            self._analyzer.prepare(image, face)
            min_side = min(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1])
            usable = (face.det_score >= det_thresh
                      and min_side >= min_face_px
                      and face.blur >= blur_min)
            entry = {
                "bbox": [float(v) for v in face.bbox],
                "det_score": float(face.det_score),
                "kps": np.asarray(face.kps, dtype=np.float32),
                "blur": float(face.blur),
                "min_side": float(min_side),
                "usable": bool(usable),
                "embedding": None,
            }
            if usable or embed_all:
                entry["embedding"] = np.asarray(
                    self._analyzer.embed(face.aligned), dtype=np.float32)
            if want_aligned:
                entry["aligned"] = np.ascontiguousarray(face.aligned)
            out.append(entry)

        return {"width": int(width), "height": int(height), "faces": out,
                "decoded": True}

    def close(self) -> None:
        self._analyzer.close()


def build(**kwargs) -> FaceService:
    """Factory the worker resolves by name, so `ort_worker.py` stays face-agnostic."""
    return FaceService(**kwargs)
