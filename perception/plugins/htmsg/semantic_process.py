"""
plugins/htmsg/semantic_process.py — Semantic scene graph extraction (Phase 3).

GPU subprocess: SAM2 + MobileCLIP for object detection and embedding.
Stub for Phase 1 — will be implemented when semantic layer is ready.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class SemanticProcessor:
    """Placeholder for Phase 3 semantic processing.

    Will run as a subprocess with:
    - SAM2-Tiny for object segmentation
    - MobileCLIP for semantic embedding (512-d vectors)
    - Depth projection for 3D localization
    """

    def __init__(self, config: dict):
        self._config = config
        self._running = False
        log.info("[htmsg:semantic] processor created (Phase 3 stub)")

    def start(self):
        """Start semantic processing subprocess."""
        log.info("[htmsg:semantic] start() called — not implemented yet (Phase 3)")
        self._running = True

    def stop(self):
        """Stop semantic processing."""
        self._running = False

    def process_keyframe(self, rgb: bytes, depth: bytes, pose: dict) -> list:
        """Process a keyframe image for semantic objects.

        Returns list of detected objects with embeddings.
        Phase 3 implementation.
        """
        return []
