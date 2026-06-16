"""
plugins/htmsg/graph_db.py — Scene graph storage (Phase 3).

Vector DB (ChromaDB) for semantic embeddings + SQLite for spatial edges.
Stub for Phase 1 — will be implemented with semantic layer.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from typing import Optional

import numpy as np

from .types import SceneNode, SceneEdge

log = logging.getLogger(__name__)


class GraphDB:
    """Scene graph storage backend.

    Phase 3 implementation will add:
    - ChromaDB collection for CLIP embeddings (vector search)
    - SQLite tables for spatial edges
    - Object association and deduplication
    """

    def __init__(self, db_path: str, max_nodes: int = 500):
        self._db_path = db_path
        self._max_nodes = max_nodes
        self._nodes: dict[str, SceneNode] = {}
        self._edges: list[SceneEdge] = []
        self._lock = threading.Lock()

        os.makedirs(db_path, exist_ok=True)
        log.info(f"[htmsg:graph_db] initialized at {db_path}")

    def add_node(self, node: SceneNode) -> None:
        """Add or update a scene node."""
        with self._lock:
            self._nodes[node.id] = node

    def query_text(self, text: str, top_k: int = 5) -> list[SceneNode]:
        """Query nodes by text similarity (CLIP embedding).

        Phase 3: will use ChromaDB vector search.
        """
        log.debug(f"[htmsg:graph_db] query_text('{text}') — not yet implemented")
        return []

    def query_near(self, x: float, y: float, z: float,
                   radius: float, top_k: int = 5) -> list[SceneNode]:
        """Query nodes by spatial proximity."""
        with self._lock:
            results = []
            for node in self._nodes.values():
                dx = node.centroid[0] - x
                dy = node.centroid[1] - y
                dz = node.centroid[2] - z
                dist = (dx*dx + dy*dy + dz*dz) ** 0.5
                if dist <= radius:
                    results.append((dist, node))
            results.sort(key=lambda x: x[0])
            return [n for _, n in results[:top_k]]

    def get_all_nodes(self) -> list[SceneNode]:
        """Get all scene nodes."""
        with self._lock:
            return list(self._nodes.values())

    @property
    def node_count(self) -> int:
        with self._lock:
            return len(self._nodes)
