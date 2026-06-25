"""
plugins/htmsg/plugin.py — HTMSGPlugin: Hierarchical Topological-Metric Semantic Graph.

Online continuous-learning scene graph for semantic navigation.
Layer 1: KISS-ICP LiDAR odometry (subprocess)
Layer 2: Pose graph with keyframes + Scan Context loop closure
Layer 3: Semantic scene graph (CLIP + SAM, future phase)
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String

from .types import Pose6D, Keyframe

log = logging.getLogger(__name__)

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    durability=DurabilityPolicy.VOLATILE,
)

def _build_tools(namespace: str) -> list:
    return [
        {
            "name": "htmsg",
            "type": "processor",
            "description": "Scene graph — start/stop HTMSG pipeline, query objects and spatial relations",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["start", "stop", "info", "query", "query_near"],
                        "description": "Action to perform"
                    },
                    "text": {
                        "type": "string",
                        "description": "Natural language query for semantic search (CLIP)"
                    },
                    "radius": {
                        "type": "number",
                        "description": "Search radius in meters (for query_near)"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max results to return",
                        "default": 5
                    },
                },
                "required": ["action"]
            },
            "topic_in": [
                {"topic": f"/{namespace}/lidar/cloud", "format": "sensor/pointcloud"},
                {"topic": f"/{namespace}/camera/rgb", "format": "image/jpeg"},
                {"topic": f"/{namespace}/camera/depth", "format": "image/depth-z16"},
            ],
            "topic_out": [
                {"topic": f"/{namespace}/htmsg/odometry", "format": "data/json"},
                {"topic": f"/{namespace}/htmsg/graph", "format": "sensor/htmsg"},
                {"topic": f"/{namespace}/htmsg/status", "format": "data/json"},
            ],
        }
    ]


class _HTMSGNode(Node):
    """ROS2 node for HTMSG: publishes odometry and status topics."""

    def __init__(self, namespace: str):
        super().__init__("htmsg")
        self._namespace = namespace
        self._odom_topic = f"/{namespace}/htmsg/odometry"
        self._status_topic = f"/{namespace}/htmsg/status"
        self._graph_topic = f"/{namespace}/htmsg/graph"

        self._odom_pub = self.create_publisher(String, self._odom_topic, _LOW_LAT_QOS)
        self._status_pub = self.create_publisher(String, self._status_topic, _LOW_LAT_QOS)
        self._graph_pub = self.create_publisher(String, self._graph_topic, _LOW_LAT_QOS)

        self._current_pose: Optional[Pose6D] = None
        self._lock = threading.Lock()

        # Status publishing timer (1Hz)
        self._status_timer = self.create_timer(1.0, self._publish_status)

    def publish_odometry(self, pose: Pose6D, timestamp: float):
        """Called from odometry callback to publish pose."""
        with self._lock:
            self._current_pose = pose
        msg = String()
        msg.data = json.dumps({
            "pose": pose.to_dict(),
            "timestamp": timestamp,
        })
        self._odom_pub.publish(msg)

    def publish_graph_delta(self, delta: dict):
        """Publish scene graph update."""
        msg = String()
        msg.data = json.dumps(delta)
        self._graph_pub.publish(msg)

    def _publish_status(self):
        """1Hz status heartbeat."""
        status = {
            "state": "running" if self._current_pose else "idle",
            "timestamp": time.time(),
        }
        with self._lock:
            if self._current_pose:
                status["current_pose"] = self._current_pose.to_dict()
        # Pose graph stats injected by plugin
        if hasattr(self, '_get_stats'):
            status.update(self._get_stats())
        msg = String()
        msg.data = json.dumps(status)
        self._status_pub.publish(msg)

        # Publish graph visualization data (sensor/htmsg format)
        if hasattr(self, '_get_graph_data'):
            graph_data = self._get_graph_data()
            if graph_data:
                gmsg = String()
                gmsg.data = json.dumps(graph_data)
                self._graph_pub.publish(gmsg)

    def get_pose(self) -> Optional[Pose6D]:
        with self._lock:
            return self._current_pose


class HTMSGPlugin:
    PREFIX = "htmsg"

    def __init__(self, plugin_cfg: dict, namespace: str, executor):
        self._cfg = plugin_cfg
        self._namespace = namespace
        self._executor = executor
        self._node: Optional[_HTMSGNode] = None
        self._odom_proxy = None  # OdometryProxy (subprocess)
        self._pose_graph = None  # PoseGraphManager
        self._running = False

        log.info(f"[htmsg] plugin init: namespace={namespace}")

    def get_tools(self) -> list:
        return _build_tools(self._namespace)

    def start(self) -> None:
        """Auto-start on plugin load."""
        if self._running:
            return
        self._start_pipeline()

    def stop(self) -> None:
        """Stop the HTMSG pipeline."""
        self._stop_pipeline()

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == "htmsg" else name

        if action == "info":
            odom_topic = f"/{self._namespace}/htmsg/odometry" if self._node else ""
            status_topic = f"/{self._namespace}/htmsg/status" if self._node else ""
            graph_topic = f"/{self._namespace}/htmsg/graph" if self._node else ""
            stats = self._get_stats()
            return {
                "name": "HTMSG",
                "state": "running" if self._running else "idle",
                "topic_out": [
                    {"topic": odom_topic, "format": "data/json", "desc": "10Hz odometry pose"},
                    {"topic": graph_topic, "format": "data/json", "desc": "Scene graph deltas"},
                    {"topic": status_topic, "format": "data/json", "desc": "1Hz system status"},
                ],
                "topic_in": [
                    {"topic": f"/{self._namespace}/lidar/cloud", "format": "sensor/pointcloud"},
                    {"topic": f"/{self._namespace}/camera/rgb", "format": "image/jpeg"},
                    {"topic": f"/{self._namespace}/camera/depth", "format": "image/depth-z16"},
                ],
                "stats": stats,
            }

        elif action == "start":
            if self._running:
                return {"status": "already_running"}
            self._start_pipeline()
            return {"status": "started"}

        elif action == "stop":
            if not self._running:
                return {"status": "already_stopped"}
            self._stop_pipeline()
            return {"status": "stopped"}

        elif action == "query":
            text = args.get("text", "")
            top_k = int(args.get("top_k", 5))
            if not text:
                return {"error": "text parameter required for query"}
            # Phase 3: semantic search via ChromaDB
            return {"error": "semantic query not yet implemented (Phase 3)"}

        elif action == "query_near":
            radius = float(args.get("radius", 5.0))
            top_k = int(args.get("top_k", 5))
            # Phase 3: spatial proximity search
            return {"error": "query_near not yet implemented (Phase 3)"}

        return None

    def _start_pipeline(self):
        """Start odometry subprocess + pose graph manager."""
        log.info("[htmsg] starting pipeline...")

        # Create ROS2 node
        self._node = _HTMSGNode(self._namespace)
        self._node._get_stats = self._get_stats
        self._node._get_graph_data = self._get_graph_data
        self._executor.add_node(self._node)

        # Start pose graph manager
        from .pose_graph import PoseGraphManager
        kf_cfg = self._cfg.get("keyframe", {})
        self._pose_graph = PoseGraphManager(
            dist_thresh=float(kf_cfg.get("dist_thresh", 1.0)),
            angle_thresh=float(kf_cfg.get("angle_thresh", 0.35)),
        )

        # Start odometry subprocess
        from .odometry_process import OdometryProxy
        self._odom_proxy = OdometryProxy(self._namespace, self._cfg)
        self._odom_proxy.set_callback(self._on_odometry)
        self._odom_proxy.start()

        self._running = True
        log.info("[htmsg] pipeline started")

    def _stop_pipeline(self):
        """Stop everything."""
        log.info("[htmsg] stopping pipeline...")
        self._running = False

        if self._odom_proxy:
            self._odom_proxy.stop()
            self._odom_proxy = None

        if self._node:
            self._executor.remove_node(self._node)
            self._node.destroy_node()
            self._node = None

        self._pose_graph = None
        log.info("[htmsg] pipeline stopped")

    def _on_odometry(self, pose: Pose6D, timestamp: float, cloud_xyz: Optional[np.ndarray]):
        """Callback from odometry subprocess with new pose estimate."""
        if not self._running or not self._node:
            return

        # Publish odometry
        self._node.publish_odometry(pose, timestamp)

        # Feed to pose graph for keyframe decision
        if self._pose_graph:
            kf = self._pose_graph.maybe_add_keyframe(pose, timestamp, cloud_xyz)
            if kf:
                log.info(f"[htmsg] new keyframe #{kf.id} at ({pose.x:.2f}, {pose.y:.2f}, {pose.z:.2f})")

    def _get_stats(self) -> dict:
        """Get current pipeline statistics."""
        stats = {
            "keyframe_count": 0,
            "odometry_running": self._odom_proxy is not None and self._odom_proxy.is_alive(),
        }
        if self._pose_graph:
            stats["keyframe_count"] = self._pose_graph.keyframe_count
        return stats

    def _get_graph_data(self) -> dict | None:
        """Build graph visualization payload for the renderer.

        Returns a JSON structure with keyframes, edges, robot pose, and scene nodes.
        Published at 1Hz to /{ns}/htmsg/graph with format sensor/htmsg.
        """
        if not self._pose_graph or not self._node:
            return None

        pose = self._node.get_pose()
        keyframes = self._pose_graph.get_keyframes(last_n=100)

        if not keyframes and not pose:
            return None

        # Build keyframe node list
        kf_nodes = []
        for kf in keyframes:
            kf_nodes.append({
                "id": kf.id,
                "x": round(kf.pose.x, 3),
                "y": round(kf.pose.y, 3),
                "z": round(kf.pose.z, 3),
                "ts": round(kf.timestamp, 1),
            })

        # Build edges (sequential connections between keyframes)
        edges = []
        for i in range(1, len(kf_nodes)):
            edges.append({"from": kf_nodes[i-1]["id"], "to": kf_nodes[i]["id"]})

        # TODO Phase 3: add loop closure edges and semantic nodes

        data = {
            "type": "htmsg_graph",
            "keyframes": kf_nodes,
            "edges": edges,
            "semantic_nodes": [],  # Phase 3
        }

        if pose:
            data["robot"] = pose.to_dict()

        return data
