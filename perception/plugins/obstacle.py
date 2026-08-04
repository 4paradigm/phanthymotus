#!/usr/bin/env python3
"""
plugins/obstacle.py — ObstacleDistancePlugin: obstacle distance estimation.

Subscribes to image/jpeg topics, estimates obstacle distance from camera,
publishes distance results to ROS2 topic.
Supports multi-instance (one instance per input topic).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

log = logging.getLogger(__name__)

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    durability=DurabilityPolicy.VOLATILE,
)

_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

TOOLS = [
    {
        "name": "obstacle",
        "type": "processor",
        "multiInstance": True,
        "description": "Obstacle Distance Estimation — estimate distance to obstacles from camera feed",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "config"],
                    "description": "Action to perform"
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 image topic to subscribe (e.g. /hostname/camera/rgb, required for action=start)"
                },
            },
            "required": ["action"]
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "provider": {"type": "string", "enum": ["openai", "qwen", "local"], "description": "Distance estimation provider", "scope": "shared"},
                "url":      {"type": "string", "description": "API URL (optional)", "scope": "shared"},
                "key":      {"type": "string", "description": "API Key", "format": "password", "scope": "shared"},
                "model":    {"type": "string", "description": "Model name", "scope": "instance"},
            },
            "required": ["provider"]
        },
        "topic_in":  [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [{"format": "data/json",  "desc": "obstacle distance estimation result"}],
    }
]


# ── Distance Estimation Adapters ──────────────────────────────────────────────

class DistanceAdapter(ABC):
    """障碍物距离估计适配器抽象基类"""

    @abstractmethod
    def estimate(self, image_bytes: bytes) -> dict:
        """估计图片中障碍物的距离，返回包含 pred_distance 的字典"""
        ...


class OpenAIVisionDistanceAdapter(DistanceAdapter):
    """OpenAI Vision API 距离估计"""

    _SYSTEM_PROMPT = (
        "You are an obstacle distance estimation system for a robot camera.\n\n"
        "Your task is to analyze the provided image and estimate the distance "
        "to the nearest obstacle in meters.\n\n"
        "Output format: Return a JSON object with:\n"
        '- "pred_distance": estimated distance in meters (float)\n'
        '- "confidence": confidence score 0-1 (float)\n'
        '- "reasoning": brief explanation of your estimation\n\n'
        "Rules:\n"
        "1. Distance should be in meters.\n"
        "2. If no obstacle is visible, return a large value (e.g., 10.0).\n"
        "3. Be precise — typical indoor distances range from 0.3m to 5m.\n"
        "4. Output ONLY the JSON object, nothing else.\n\n"
        'Example: {"pred_distance": 1.25, "confidence": 0.85, "reasoning": "clear wall visible"}'
    )

    def __init__(self, url: str, key: str, model: str):
        self.base_url = url.rstrip('/') if url else "https://api.openai.com/v1"
        self.key = key
        self.model = model or "gpt-4o-mini"

    def estimate(self, image_bytes: bytes) -> dict:
        import requests
        import base64

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        image_format = "jpeg"
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            image_format = "png"
        elif image_bytes[:2] == b'BM':
            image_format = "bmp"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            image_format = "webp"

        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{image_b64}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Estimate the distance to the nearest obstacle in this image."
                    }
                ]
            }
        ]

        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 512,
            },
            headers=headers,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return self._parse_result(content)

    @staticmethod
    def _parse_result(content: str) -> dict:
        """解析模型返回的 JSON 结果"""
        content = content.strip()
        if content.startswith("{"):
            try:
                parsed = json.loads(content)
                return {
                    "pred_distance": float(parsed.get("pred_distance", 10.0)),
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "reasoning": parsed.get("reasoning", ""),
                }
            except (json.JSONDecodeError, ValueError):
                pass
        # 尝试从 markdown 代码块中提取
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                return {
                    "pred_distance": float(parsed.get("pred_distance", 10.0)),
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "reasoning": parsed.get("reasoning", ""),
                }
            except (json.JSONDecodeError, ValueError):
                pass
        # 兜底：尝试提取数字
        import re
        numbers = re.findall(r"\d+\.?\d*", content)
        if numbers:
            try:
                return {"pred_distance": float(numbers[0]), "confidence": 0.5, "reasoning": content[:200]}
            except ValueError:
                pass
        log.warning(f"[obstacle] failed to parse distance result, returning default: {content[:200]!r}")
        return {"pred_distance": 10.0, "confidence": 0.0, "reasoning": "parse failed"}


class QwenVLDistanceAdapter(DistanceAdapter):
    """Qwen-VL 距离估计"""

    _SYSTEM_PROMPT = (
        "你是一个机器人摄像头障碍物距离估计系统。\n\n"
        "任务：分析提供的图片，估计最近障碍物的距离（单位：米）。\n\n"
        "输出格式：返回 JSON 对象，包含：\n"
        '- "pred_distance": 估计距离（米，浮点数）\n'
        '- "confidence": 置信度 0-1（浮点数）\n'
        '- "reasoning": 简要说明\n\n'
        "规则：\n"
        "1. 距离单位为米。\n"
        "2. 如果没有可见障碍物，返回较大值（如 10.0）。\n"
        "3. 室内典型距离范围：0.3m 到 5m。\n"
        "4. 只输出 JSON 对象，不要其他内容。\n\n"
        '示例：{"pred_distance": 1.25, "confidence": 0.85, "reasoning": "清晰可见的墙壁"}'
    )

    def __init__(self, url: str, key: str, model: str):
        self.base_url = url.rstrip('/') if url else "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.key = key
        self.model = model or "qwen-vl-max"

    def estimate(self, image_bytes: bytes) -> dict:
        import requests
        import base64

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        image_format = "jpeg"
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            image_format = "png"

        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/{image_format};base64,{image_b64}"
                    },
                    {
                        "type": "text",
                        "text": "估计这张图片中最近障碍物的距离。"
                    }
                ]
            }
        ]

        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 512,
            },
            headers=headers,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return OpenAIVisionDistanceAdapter._parse_result(content)


class LocalDistanceAdapter(DistanceAdapter):
    """本地距离估计（占位实现，可替换为实际模型）

    基于简单的图像特征进行粗略距离估计。
    实际部署时应替换为深度学习模型（如 MiDaS、DPT 等）。
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._model = None

    def estimate(self, image_bytes: bytes) -> dict:
        """返回随机距离值（占位实现，后续替换为真实深度模型）"""
        import random
        return {
            "pred_distance": round(random.uniform(0.0, 2.0), 2),
        }


def _build_distance_adapter(cfg: dict) -> Optional[DistanceAdapter]:
    """根据配置创建距离估计适配器"""
    provider = cfg.get('provider', 'local')

    if provider == 'openai':
        url, key = cfg.get('url', ''), cfg.get('key', '')
        if not key:
            return None
        return OpenAIVisionDistanceAdapter(url, key, cfg.get('model', ''))

    elif provider == 'qwen':
        url, key = cfg.get('url', ''), cfg.get('key', '')
        if not key:
            return None
        return QwenVLDistanceAdapter(url, key, cfg.get('model', ''))

    elif provider == 'local':
        return LocalDistanceAdapter(cfg.get('model_path'))

    return None


# ── ROS2 Node (one per instance/topic) ────────────────────────────────────────

class _ObstacleNode(Node):
    """Per-topic obstacle distance estimation node."""

    def __init__(self, input_topic: str, adapter: DistanceAdapter,
                 node_suffix: str):
        super().__init__(f"obstacle_{node_suffix}")
        self._input_topic = input_topic
        self._output_topic = f"{input_topic}/obstacle"
        self._adapter = adapter

        self._pub = self.create_publisher(String, self._output_topic, _PUB_QOS)
        self._sub: Optional[object] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=10)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._detect_count = 0
        self.state = "idle"

    def start(self) -> dict:
        if self._sub is not None:
            self.state = "running"
            return {"state": "running", "input": self._input_topic, "output": self._output_topic}
        self._stop_event.clear()
        self._sub = self.create_subscription(
            CompressedImage, self._input_topic, self._image_cb, _LOW_LAT_QOS
        )
        self._worker = threading.Thread(target=self._inference_worker, daemon=True,
                                         name=f"obstacle_worker_{self._input_topic}")
        self._worker.start()
        self.state = "running"
        log.info(f"[obstacle] started: {self._input_topic} -> {self._output_topic}")
        return {"state": "running", "input": self._input_topic, "output": self._output_topic}

    def stop(self) -> dict:
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        self._worker = None
        self.state = "idle"
        log.info(f"[obstacle] stopped: {self._input_topic}")
        return {"state": "idle", "input": self._input_topic}

    def _image_cb(self, msg: CompressedImage):
        log.info(
            f"[obstacle] received image frame: size={len(msg.data)} bytes, format={msg.format}, topic={self._input_topic}")
        # Drop old frame if queue full (no backpressure)
        try:
            self._frame_queue.put_nowait(msg.data)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(msg.data)
            except queue.Full:
                pass

    def _inference_worker(self):
        while not self._stop_event.is_set():
            try:
                jpeg_bytes = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                result = self._adapter.estimate(jpeg_bytes)
                self._publish_result(result)
            except Exception as e:
                log.error(f"[obstacle] inference error: {e}", exc_info=True)

    def _publish_result(self, result: dict):
        self._detect_count += 1
        msg = String()
        msg.data = json.dumps({
            "pred_distance": result.get("pred_distance", 10.0),
        }, ensure_ascii=False)
        self._pub.publish(msg)


# ── Plugin class ──────────────────────────────────────────────────────────────

class ObstacleDistancePlugin:
    PREFIX = "obstacle"

    def __init__(self, plugin_cfg: dict, executor):
        self._executor = executor
        self._provider = plugin_cfg.get("provider", "local")
        self._url = plugin_cfg.get("url", "")
        self._key = plugin_cfg.get("key", "")
        self._model = plugin_cfg.get("model", "")
        self._model_path = plugin_cfg.get("model_path")
        self._adapter = _build_distance_adapter(plugin_cfg)
        self._nodes: dict[str, _ObstacleNode] = {}
        self._instance_configs: dict[str, dict] = {}

        log.info(f"[obstacle] plugin init: provider={self._provider}, "
                 f"key={'set' if self._key else 'MISSING'}")

        if not self._adapter:
            log.warning("[obstacle] adapter not configured (missing key or invalid provider)")

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action", name)
        instance_id = args.get("instance_id", "")

        if action == "info":
            instances = {}
            for key, node in self._nodes.items():
                instances[key] = {
                    "input": node._input_topic,
                    "output": node._output_topic,
                    "detect_count": node._detect_count,
                }
            input_topic = args.get("input_topic", "")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                input_topic = node._input_topic
            elif not input_topic and self._nodes:
                first_node = next(iter(self._nodes.values()))
                input_topic = first_node._input_topic
            topics_in = [{"topic": input_topic, "format": "image/jpeg"}] if input_topic else []
            topics_out = [{"topic": f"{input_topic}/obstacle", "format": "data/json"}] if input_topic else []
            state = "running" if instances else "idle"
            return {
                "name": "ObstacleDistance", "manufacture": "Embodied", "model": "obstacle",
                "state": state,
                "instances": instances,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "Obstacle distance estimation from camera feed",
            }

        elif action == "start":
            input_topic = args.get("input_topic")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if not input_topic:
                raise ValueError("input_topic is required")
            node_key = instance_id or input_topic
            if node_key not in self._nodes:
                icfg = self._instance_configs.get(node_key, {})
                # Build adapter for this instance if config differs
                adapter = self._adapter
                if icfg:
                    adapter = _build_distance_adapter(icfg) or self._adapter
                suffix = node_key.replace("/", "_").replace("-", "_").lstrip("_")
                node = _ObstacleNode(input_topic, adapter, suffix)
                self._executor.add_node(node)
                self._nodes[node_key] = node
                node.start()
            return self._nodes[node_key].start()

        elif action == "stop":
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                result = node.stop()
                self._executor.remove_node(node)
                del self._nodes[instance_id]
                return result
            elif not instance_id and self._nodes:
                results = []
                for key in list(self._nodes.keys()):
                    node = self._nodes[key]
                    node.stop()
                    self._executor.remove_node(node)
                    del self._nodes[key]
                    results.append(key)
                return {"state": "idle", "stopped_instances": results}
            return {"state": "idle"}

        elif action == "config":
            cfg = {k: v for k, v in args.items() if k not in ('action', 'instance_id') and v is not None and v != ''}
            if instance_id:
                self._instance_configs[instance_id] = cfg
                if instance_id in self._nodes:
                    node = self._nodes[instance_id]
                    node.stop()
                    self._executor.remove_node(node)
                    del self._nodes[instance_id]
                return {"status": "configured", "instance_id": instance_id, "config": cfg}
            else:
                if "provider" in cfg:
                    self._provider = cfg["provider"]
                if "model" in cfg:
                    self._model = cfg["model"]
                if "key" in cfg:
                    self._key = cfg["key"]
                if "url" in cfg:
                    self._url = cfg["url"]
                # Rebuild global adapter
                self._adapter = _build_distance_adapter({
                    "provider": self._provider,
                    "url": self._url,
                    "key": self._key,
                    "model": self._model,
                    "model_path": self._model_path,
                })
                return {"status": "configured", "config": cfg}

        return None