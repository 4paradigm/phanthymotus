#!/usr/bin/env python3
"""
plugins/ocr.py — OCRPlugin: OCR 文字识别封装。

订阅 image/jpeg topic，持续进行 OCR 识别并发布结果到 ROS2 topic。
参考 asr.py 架构设计。
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from io import BytesIO


def _resource_snapshot() -> str:
    """轻量资源快照：RSS + 线程数（用于定位服务慢性死亡原因）"""
    rss_mb = -1.0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    rss_mb = int(line.split()[1]) / 1024.0
                    break
    except OSError:
        pass
    return f"rss={rss_mb:.0f}MB threads={threading.active_count()}"
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from plugins.ocr_runtime import (
    DEFAULT_DET_BOX_THRESH,
    DEFAULT_DET_THRESH,
    DEFAULT_DET_UNCLIP_RATIO,
    DEFAULT_MAX_SIDE_LEN,
    DEFAULT_REC_MIN_SCORE,
    RapidOCRAdapter,
    normalize_rapidocr_output,
    recognize_to_payload,
)

log = logging.getLogger(__name__)

_CAMERA_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

_RESULT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

TOOLS = [
    {
        "name": "ocr",
        "type": "processor",
        "multiInstance": True,
        "description": "OCR — recognize text in camera feed via image topic subscription",
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
                "backend": {"type": "string", "enum": ["mnn", "onnxruntime", "tensorrt"], "default": "mnn", "scope": "shared"},
                "fallback_backend": {"type": "string", "enum": ["", "mnn", "onnxruntime"], "default": "", "scope": "shared"},
                "model_dir": {"type": "string", "description": "OCR 模型或 TensorRT engine 目录", "scope": "shared"},
                "fallback_model_dir": {"type": "string", "description": "OCR 回退模型目录", "scope": "shared"},
                "device": {"type": "string", "enum": ["cpu", "cuda"], "default": "cpu", "scope": "shared"},
                "device_id": {"type": "integer", "minimum": 0, "default": 0, "scope": "shared"},
                "gpu_mem_mb": {"type": "integer", "minimum": 0, "default": 512, "scope": "shared"},
                "provider": {"type": "string", "enum": ["rapidocr", "openai", "qwen", "tesseract"], "description": "OCR 服务商", "scope": "shared"},
                "url":      {"type": "string", "description": "API URL (可选)", "scope": "shared"},
                "key":      {"type": "string", "description": "API Key", "format": "password", "scope": "shared"},
                "model":    {"type": "string", "description": "模型名称", "scope": "instance"},
                "language": {"type": "string", "description": "默认语言", "default": "zh", "scope": "instance"},
                "max_side_len": {"type": "integer", "minimum": 32, "default": DEFAULT_MAX_SIDE_LEN, "description": "检测输入最长边", "scope": "shared"},
                "det_thresh": {"type": "number", "minimum": 0, "maximum": 1, "default": DEFAULT_DET_THRESH, "description": "DB 文本像素阈值", "scope": "shared"},
                "det_box_thresh": {"type": "number", "minimum": 0, "maximum": 1, "default": DEFAULT_DET_BOX_THRESH, "description": "DB 文本框阈值", "scope": "shared"},
                "det_unclip_ratio": {"type": "number", "exclusiveMinimum": 0, "default": DEFAULT_DET_UNCLIP_RATIO, "description": "DB 文本框扩张比例", "scope": "shared"},
                "rec_min_score": {"type": "number", "minimum": 0, "maximum": 1, "default": DEFAULT_REC_MIN_SCORE, "description": "识别结果最低置信度", "scope": "shared"},
                "enable_preprocess": {"type": "boolean", "default": True, "description": "启用 OCR 图像预处理", "scope": "shared"},
                "min_interval_ms": {"type": "integer", "minimum": 0, "default": 0, "description": "帧处理最小间隔(ms)，限制 GPU 占用，0=不限", "scope": "shared"},
            },
            "required": ["provider"]
        },
        "topic_in":  [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [{"format": "data/json",  "desc": "OCR result with text boxes"}],
    }
]


# ── OCR Adapters ──────────────────────────────────────────────────────────────

class OCRAdapter(ABC):
    """OCR 适配器抽象基类"""

    @abstractmethod
    def recognize(self, image_bytes: bytes, language: str = "zh") -> list:
        """识别图片中的文字，返回文本列表（每项包含 text 和 bbox）"""
        ...

    @staticmethod
    def format_result(results: list) -> str:
        """将结果列表格式化为纯文本字符串（用于兼容旧逻辑）"""
        return " ".join(item.get("text", "") for item in results if item.get("text"))


class OpenAIVisionAdapter(OCRAdapter):
    """OpenAI Vision API (GPT-4o / GPT-4o-mini) OCR"""

    _SYSTEM_PROMPT_TEMPLATE = (
        "You are an OCR (Optical Character Recognition) system with bounding box detection.\n\n"
        "Your task is to extract ALL text from the provided image and return each text segment with its bounding box coordinates.\n\n"
        "Output format: Return a JSON array where each element contains:\n"
        "- \"text\": the extracted text string\n"
        "- \"bbox\": [x1, y1, x2, y2] coordinates of the bounding box (top-left x, top-left y, bottom-right x, bottom-right y)\n\n"
        "Rules:\n"
        "1. Extract text exactly as it appears in the image, preserving the original order.\n"
        "2. Each text segment should be a distinct line or logical text block.\n"
        "3. Bounding box coordinates should be integers representing pixel positions.\n"
        "4. Do NOT translate, summarize, or interpret the text.\n"
        "5. If there is no text in the image, return an empty array [].\n"
        "6. For multi-language text, transcribe each language as-is.\n\n"
        "Image dimensions: {width}x{height} (width x height).\n"
        "IMPORTANT: Return bounding box coordinates scaled to the original image dimensions ({width}x{height}).\n\n"
        "Output ONLY the JSON array, nothing else. Example:\n"
        '[{{"text": "Hello World", "bbox": [100, 50, 300, 80]}}, {{"text": "Price: $10", "bbox": [100, 100, 250, 130]}}]'
    )

    @staticmethod
    def _scale_results(results: list, orig_w: int, orig_h: int, model_w: int, model_h: int) -> list:
        """将模型返回的坐标缩放到原始图片尺寸"""
        if model_w <= 0 or model_h <= 0 or orig_w == model_w and orig_h == model_h:
            return results
        scale_x = orig_w / model_w
        scale_y = orig_h / model_h
        scaled = []
        for item in results:
            bbox = item.get("bbox", [])
            if bbox and len(bbox) == 4:
                scaled_item = {
                    **item,
                    "bbox": [
                        int(round(bbox[0] * scale_x)),
                        int(round(bbox[1] * scale_y)),
                        int(round(bbox[2] * scale_x)),
                        int(round(bbox[3] * scale_y)),
                    ],
                }
                scaled.append(scaled_item)
            else:
                scaled.append(item)
        return scaled

    @staticmethod
    def _convert_gemini_bbox(results: list, orig_w: int, orig_h: int) -> list:
        """Gemini 原生输出 [ymin, xmin, ymax, xmax] 归一化到 [0, 1000]，
        转换为像素坐标 [x1, y1, x2, y2]。"""
        converted = []
        for item in results:
            bbox = item.get("bbox", [])
            if len(bbox) == 4:
                ymin_n, xmin_n, ymax_n, xmax_n = bbox
                converted.append({
                    **item,
                    "bbox": [
                        int(xmin_n / 1000 * orig_w),
                        int(ymin_n / 1000 * orig_h),
                        int(xmax_n / 1000 * orig_w),
                        int(ymax_n / 1000 * orig_h),
                    ]
                })
            else:
                converted.append(item)
        return converted

    @staticmethod
    def _get_image_dimensions(image_bytes: bytes) -> tuple:
        """获取原始图片尺寸 (width, height)"""
        from PIL import Image
        with Image.open(BytesIO(image_bytes)) as img:
            return img.width, img.height

    def __init__(self, url: str, key: str, model: str):
        self.base_url = url.rstrip('/') if url else "https://api.openai.com/v1"
        self.key = key
        self.model = model or "gpt-4o-mini"

    def recognize(self, image_bytes: bytes, language: str = "zh") -> list:
        import requests

        # 获取原始图片尺寸
        orig_w, orig_h = self._get_image_dimensions(image_bytes)

        _MAX_SIDE = 3072
        scale = min(_MAX_SIDE / max(orig_w, orig_h), 1.0)

        system_prompt = self._SYSTEM_PROMPT_TEMPLATE.format(width=orig_w, height=orig_h)

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # 检测图片格式
        image_format = "jpeg"
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            image_format = "png"
        elif image_bytes[:2] == b'BM':
            image_format = "bmp"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            image_format = "webp"

        user_text = f"Extract all text from this image with bounding boxes. Language hint: {language}"

        messages = [
            {"role": "system", "content": system_prompt},
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
                        "text": user_text
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
                "max_tokens": 4096,
            },
            headers=headers,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed_results = self._parse_result(content)
        return self._convert_gemini_bbox(parsed_results, orig_w, orig_h)

    @staticmethod
    def _parse_result(content: str) -> list:
        """解析模型返回的 JSON 结果"""
        content = content.strip()
        # 尝试提取 JSON 数组
        if content.startswith("["):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass
        # 尝试从 markdown 代码块中提取
        import re
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # 尝试找到第一个 [ 到最后一个 ]
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                pass
        # 兜底：作为纯文本处理
        log.warning(f"[ocr] failed to parse JSON result, treating as plain text: {content[:200]!r}")
        return [{"text": content, "bbox": []}]


class QwenVLAdapter(OCRAdapter):
    """Qwen-VL (通义千问视觉模型) OCR

    通过 OpenAI 兼容接口调用 Qwen-VL 进行 OCR。
    """

    _SYSTEM_PROMPT_TEMPLATE = (
        "你是一个 OCR 文字识别系统，支持坐标检测。\n\n"
        "任务：从图片中提取所有文字，并返回每段文字的边界框坐标。\n\n"
        "输出格式：返回 JSON 数组，每个元素包含：\n"
        '- "text": 提取的文字\n'
        '- "bbox": [x1, y1, x2, y2] 边界框坐标（左上角x, 左上角y, 右下角x, 右下角y）\n\n'
        "规则：\n"
        "1. 准确提取图片中的所有文字，保持原有顺序。\n"
        "2. 每段文字应为独立的一行或逻辑文本块。\n"
        "3. 坐标为整数，表示像素位置。\n"
        "4. 不要翻译、总结或解释文字内容。\n"
        "5. 如果图片中没有文字，返回空数组 []。\n\n"
        "图片原始尺寸：{width}x{height}（宽 x 高）。\n"
        "重要：请返回基于原始图片尺寸的边界框坐标。\n\n"
        "只输出 JSON 数组，不要输出其他内容。示例：\n"
        '[{{"text": "你好世界", "bbox": [100, 50, 300, 80]}}, {{"text": "价格：10元", "bbox": [100, 100, 250, 130]}}]'
    )

    def __init__(self, url: str, key: str, model: str):
        self.base_url = url.rstrip('/') if url else "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.key = key
        self.model = model or "qwen-vl-max"

    def recognize(self, image_bytes: bytes, language: str = "zh") -> list:
        import requests

        # 获取原始图片尺寸
        orig_w, orig_h = OpenAIVisionAdapter._get_image_dimensions(image_bytes)

        # 使用模板填充尺寸信息
        system_prompt = self._SYSTEM_PROMPT_TEMPLATE.format(width=orig_w, height=orig_h)

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        image_format = "jpeg"
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            image_format = "png"

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/{image_format};base64,{image_b64}"
                    },
                    {
                        "type": "text",
                        "text": "请识别图片中的所有文字并返回坐标。"
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
                "max_tokens": 4096,
            },
            headers=headers,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return OpenAIVisionAdapter._parse_result(content)


class TesseractAdapter(OCRAdapter):
    """Tesseract 本地 OCR 引擎

    离线 OCR，无需网络，但精度较低。
    Tesseract 不支持坐标输出，返回无 bbox 的结果。
    """

    def __init__(self, language: str = "chi_sim+eng"):
        self._language = language

    def recognize(self, image_bytes: bytes, language: str = "zh") -> list:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise RuntimeError("pytesseract and PIL are required for Tesseract OCR")

        lang_map = {
            "zh": "chi_sim+eng",
            "ch": "chi_sim+eng",
            "zh-CN": "chi_sim+eng",
            "zh-TW": "chi_tra+eng",
            "en": "eng",
            "ja": "jpn+eng",
            "ko": "kor+eng",
        }
        tesseract_lang = lang_map.get(language, self._language)

        image = Image.open(BytesIO(image_bytes))
        # 使用 image_to_data 获取带坐标的结果
        data = pytesseract.image_to_data(image, lang=tesseract_lang, output_type=pytesseract.Output.DICT)
        results = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            if not text:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            results.append({
                "text": text,
                "bbox": [x, y, x + w, y + h],
            })
        return results


def _ocr_output_topic(input_topic: str) -> str:
    return f"{input_topic}/ocr"


def _freeze_config(value):
    if isinstance(value, dict):
        return tuple(
            sorted(
                (key, _freeze_config(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, list):
        return tuple(_freeze_config(item) for item in value)
    return value


def _adapter_signature(cfg: dict) -> tuple:
    provider = cfg.get('provider', 'rapidocr')
    common = (provider,)
    if provider == 'rapidocr':
        return common + (
            str(cfg.get('backend', 'onnxruntime')).strip().lower(),
            str(cfg.get('fallback_backend', '')).strip().lower(),
            cfg.get('model_dir', '/models/ocr/ppocrv6-tiny'),
            cfg.get('fallback_model_dir', ''),
            str(cfg.get('device', 'cpu')).strip().lower(),
            int(cfg.get('device_id', 0)),
            int(cfg.get('gpu_mem_mb', 512)),
            bool(cfg.get('use_angle_cls', True)),
            int(cfg.get('num_threads', 2)),
            int(cfg.get('max_side_len', DEFAULT_MAX_SIDE_LEN)),
            float(cfg.get('rec_min_score', DEFAULT_REC_MIN_SCORE)),
            bool(cfg.get('enable_preprocess', True)),
            float(cfg.get('det_thresh', DEFAULT_DET_THRESH)),
            float(cfg.get('det_box_thresh', DEFAULT_DET_BOX_THRESH)),
            float(cfg.get('det_unclip_ratio', DEFAULT_DET_UNCLIP_RATIO)),
            _freeze_config(cfg.get('large_image_strategy', {})),
        )
    if provider in ('openai', 'qwen'):
        return common + (
            cfg.get('url', ''),
            cfg.get('key', ''),
            cfg.get('model', ''),
        )
    if provider == 'tesseract':
        return common + (cfg.get('language', 'chi_sim+eng'),)
    return common


def _build_ocr_adapter(cfg: dict) -> Optional[OCRAdapter]:
    """根据配置创建 OCR 适配器"""
    provider = cfg.get('provider', 'rapidocr')

    if provider == 'rapidocr':
        return RapidOCRAdapter(
            cfg.get('model_dir', '/models/ocr/ppocrv6-tiny'),
            backend=str(cfg.get('backend', 'onnxruntime')).strip().lower(),
            fallback_backend=str(
                cfg.get('fallback_backend', '')
            ).strip().lower(),
            fallback_model_dir=cfg.get('fallback_model_dir', ''),
            device=str(cfg.get('device', 'cpu')).strip().lower(),
            device_id=int(cfg.get('device_id', 0)),
            gpu_mem_mb=int(cfg.get('gpu_mem_mb', 512)),
            use_angle_cls=bool(cfg.get('use_angle_cls', True)),
            num_threads=int(cfg.get('num_threads', 2)),
            max_side_len=int(cfg.get('max_side_len', DEFAULT_MAX_SIDE_LEN)),
            rec_min_score=float(
                cfg.get('rec_min_score', DEFAULT_REC_MIN_SCORE)
            ),
            enable_preprocess=bool(cfg.get('enable_preprocess', True)),
            det_thresh=float(cfg.get('det_thresh', DEFAULT_DET_THRESH)),
            det_box_thresh=float(
                cfg.get('det_box_thresh', DEFAULT_DET_BOX_THRESH)
            ),
            det_unclip_ratio=float(
                cfg.get('det_unclip_ratio', DEFAULT_DET_UNCLIP_RATIO)
            ),
            large_image_strategy=dict(
                cfg.get('large_image_strategy') or {}
            ),
        )

    elif provider == 'openai':
        url, key = cfg.get('url', ''), cfg.get('key', '')
        if not key:
            return None
        return OpenAIVisionAdapter(url, key, cfg.get('model', ''))

    elif provider == 'qwen':
        url, key = cfg.get('url', ''), cfg.get('key', '')
        if not key:
            return None
        return QwenVLAdapter(url, key, cfg.get('model', ''))

    elif provider == 'tesseract':
        return TesseractAdapter(cfg.get('language', 'chi_sim+eng'))

    return None


# ── ROS2 Node (订阅模式) ───────────────────────────────────────────────────────

class _OCRNode(Node):
    """订阅 image/jpeg topic，持续进行 OCR 识别"""

    def __init__(self, input_topic: str, adapter: OCRAdapter, language: str = "zh",
                 node_suffix: str = '', min_interval: float = 0.0):
        node_name = f"ocr_{node_suffix}" if node_suffix else "ocr"
        super().__init__(node_name)

        self._input_topic = input_topic
        self._output_topic = _ocr_output_topic(input_topic)
        self._adapter = adapter
        self._language = language
        # 帧处理最小间隔（秒）：限制 GPU 占用，0 = 不限
        self._min_interval = max(0.0, float(min_interval))
        self.state = "idle"

        self._sub = None
        self._pub = self.create_publisher(String, self._output_topic, _RESULT_QOS)

        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stop_event.set()
        self._generation = 0
        self._worker_threads: list[threading.Thread] = []
        self._frame_count = 0  # 收到的图片帧计数

        log.info(f"[ocr] node created: subscribing={self._input_topic}, publishing={self._output_topic}")

    def start(self) -> dict:
        if self.state == "running":
            return self._status_dict()

        if not self._adapter:
            raise RuntimeError("OCR adapter not configured")

        self._generation += 1
        generation = self._generation
        stop_event = threading.Event()
        frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = stop_event
        self._frame_queue = frame_queue
        if self._sub is None:
            self._sub = self.create_subscription(
                CompressedImage, self._input_topic, self._image_cb, _CAMERA_QOS
            )
        self.state = "running"
        self._worker_threads = [
            thread for thread in self._worker_threads if thread.is_alive()
        ]
        self._worker_thread = threading.Thread(
            target=self._worker,
            args=(generation, stop_event, frame_queue),
            daemon=True,
        )
        self._worker_threads.append(self._worker_thread)
        self._worker_thread.start()

        log.info(f"[ocr] started: {self._input_topic} → {self._output_topic} | {_resource_snapshot()}")
        return self._status_dict()

    def stop(self) -> dict:
        self.state = "idle"
        self._stop_event.set()
        while True:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        deadline = time.monotonic() + 3.0
        for thread in self._worker_threads:
            if thread.is_alive():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
        self._worker_threads = [
            thread for thread in self._worker_threads if thread.is_alive()
        ]
        if self._worker_threads:
            log.warning(
                "[ocr] %d worker(s) still stopping after timeout: %s",
                len(self._worker_threads),
                self._input_topic,
            )

        log.info(f"[ocr] stopped: {self._input_topic} | {_resource_snapshot()}")
        return {"state": "idle"}

    @property
    def worker_alive(self) -> bool:
        return any(thread.is_alive() for thread in self._worker_threads)

    def _image_cb(self, msg: CompressedImage):
        """接收图片帧，放入队列"""
        stop_event = self._stop_event
        frame_queue = self._frame_queue
        if self.state != "running" or stop_event.is_set():
            return
        self._frame_count += 1
        image_data = bytes(msg.data)
        log.info(f"[ocr] received image frame #{self._frame_count}: "
                 f"size={len(image_data)} bytes, format={msg.format}, "
                 f"topic={self._input_topic}")
        try:
            frame_queue.put_nowait((image_data, time.time()))
        except queue.Full:
            log.warning("[ocr] frame queue full, dropping old frame (queue_size=1)")
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                frame_queue.put_nowait((image_data, time.time()))
            except queue.Full:
                pass

    def _is_generation_active(
        self, generation: int, stop_event: threading.Event
    ) -> bool:
        return (
            self.state == "running"
            and self._generation == generation
            and self._stop_event is stop_event
            and not stop_event.is_set()
        )

    def _worker(
        self,
        generation: int,
        stop_event: threading.Event,
        frame_queue: queue.Queue,
    ):
        """后台工作线程：从队列取图片进行 OCR"""
        while not stop_event.is_set():
            try:
                image_bytes, ts = frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            t_start = time.time()
            payload = recognize_to_payload(
                self._adapter, image_bytes, self._language, ts
            )
            if not self._is_generation_active(generation, stop_event):
                continue
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self._pub.publish(msg)
            if "error" in payload:
                log.error("[ocr] recognition error: %s", payload["error"])
            else:
                log.info(
                    "[ocr] published result to %s: %d items",
                    self._output_topic,
                    len(payload["items"]),
                )

            # 限帧：距上一帧开始不足 min_interval 则等待（降低 GPU 占用）
            if self._min_interval > 0:
                remaining = self._min_interval - (time.time() - t_start)
                if remaining > 0:
                    stop_event.wait(remaining)

            # 资源监控：每 20 帧打一次快照（泄漏/线程堆积排查用）
            if self._frame_count % 20 == 0:
                log.info(f"[ocr] monitor | {_resource_snapshot()}")

    def _status_dict(self) -> dict:
        return {
            "state": self.state,
            "topic_in": [{"topic": self._input_topic, "format": "image/jpeg", "desc": "image input"}],
            "topic_out": [{"topic": self._output_topic, "format": "data/json", "desc": "OCR result"}],
        }


# ── Plugin ────────────────────────────────────────────────────────────────────

class OCRPlugin:
    PREFIX = "ocr"

    def __init__(self, plugin_cfg: dict, executor):
        self._plugin_cfg = dict(plugin_cfg)
        self._language = plugin_cfg.get('language', 'zh')
        self._nodes: dict[str, _OCRNode] = {}
        self._instance_configs: dict[str, dict] = {}
        self._instance_adapters: dict[str, tuple[tuple, OCRAdapter]] = {}
        self._retired_nodes: list[_OCRNode] = []
        self._executor = executor
        self._lifecycle_lock = threading.RLock()

        self._adapter = _build_ocr_adapter(plugin_cfg)
        log.info(
            f"[ocr] plugin init: provider={plugin_cfg.get('provider')}, "
            f"language={self._language}, "
            f"adapter_ok={self._adapter is not None}"
        )

    def get_tools(self) -> list:
        return TOOLS

    def _retire_node(self, node_key: str) -> dict:
        node = self._nodes.pop(node_key)
        result = node.stop()
        self._executor.remove_node(node)
        if node not in self._retired_nodes:
            self._retired_nodes.append(node)
        log.info(
            f"[ocr] node retired: {node_key} | nodes={len(self._nodes)} "
            f"retired={len(self._retired_nodes)} "
            f"instance_adapters={len(self._instance_adapters)} | {_resource_snapshot()}"
        )
        return result

    def _adapter_for_instance(self, instance_id: str) -> OCRAdapter | None:
        override = self._instance_configs.get(instance_id, {})
        cfg = {**self._plugin_cfg, **override}
        signature = _adapter_signature(cfg)
        if signature == _adapter_signature(self._plugin_cfg):
            return self._adapter

        cached = self._instance_adapters.get(instance_id)
        if cached and cached[0] == signature:
            return cached[1]

        adapter = _build_ocr_adapter(cfg)
        self._instance_adapters[instance_id] = (signature, adapter)
        log.warning(
            f"[ocr] NEW adapter built for instance={instance_id} "
            f"(cache size={len(self._instance_adapters)}) | {_resource_snapshot()}"
        )
        return adapter

    def _configure_node(self, node: _OCRNode, instance_id: str) -> None:
        cfg = {**self._plugin_cfg, **self._instance_configs.get(instance_id, {})}
        adapter = self._adapter_for_instance(instance_id) if instance_id else self._adapter
        node._adapter = adapter
        node._language = cfg.get("language", self._language)
        node._min_interval = max(
            0.0, float(cfg.get("min_interval_ms", 0)) / 1000.0
        )

    @staticmethod
    def _unique_nodes(nodes) -> list[_OCRNode]:
        unique = []
        seen = set()
        for node in nodes:
            identity = id(node)
            if identity not in seen:
                seen.add(identity)
                unique.append(node)
        return unique

    def prepare_shutdown(self) -> None:
        with self._lifecycle_lock:
            nodes = self._unique_nodes(
                [*self._nodes.values(), *self._retired_nodes]
            )
            for node in nodes:
                node.stop()

    def destroy_nodes(self) -> None:
        with self._lifecycle_lock:
            nodes = self._unique_nodes(
                [*self._nodes.values(), *self._retired_nodes]
            )
            for node in nodes:
                node.destroy_node()
            self._nodes.clear()
            self._retired_nodes.clear()

    def dispatch(self, name: str, args: dict) -> dict | None:
        with self._lifecycle_lock:
            return self._dispatch_locked(name, args)

    def _dispatch_locked(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == "ocr" else name
        instance_id = args.get("instance_id", "")

        if action == "info":
            input_topic = args.get("input_topic", "")

            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                return {
                    "name": "OCR", "manufacture": "Embodied", "model": "ocr",
                    "state": node.state,
                    "topic_in": [{"topic": node._input_topic, "format": "image/jpeg", "desc": ""}],
                    "topic_out": [{"topic": node._output_topic, "format": "data/json", "desc": ""}],
                    "desc": "OCR service — extracts text from images",
                }

            if instance_id:
                inferred_out = f"{input_topic}/ocr" if input_topic else ""
                return {
                    "name": "OCR", "manufacture": "Embodied", "model": "ocr",
                    "state": "idle",
                    "topic_in": [{"topic": input_topic, "format": "image/jpeg", "desc": ""}] if input_topic else [],
                    "topic_out": [{"topic": inferred_out, "format": "data/json", "desc": ""}] if inferred_out else [],
                    "desc": "OCR service — extracts text from images",
                }

            # 聚合所有实例信息
            if self._nodes:
                topics_in = [{"topic": n._input_topic, "format": "image/jpeg", "desc": ""} for n in self._nodes.values()]
                topics_out = [{"topic": n._output_topic, "format": "data/json", "desc": ""} for n in self._nodes.values()]
                states = list(set(n.state for n in self._nodes.values()))
                state = "running" if "running" in states else states[0] if states else "idle"
            else:
                inferred_out = f"{input_topic}/ocr" if input_topic else ""
                topics_in = [{"topic": input_topic, "format": "image/jpeg", "desc": ""}]
                topics_out = [{"topic": inferred_out, "format": "data/json", "desc": ""}]
                state = "idle"

            return {
                "name": "OCR", "manufacture": "Embodied", "model": "ocr",
                "state": state,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "OCR service — extracts text from images",
            }

        elif action == "start":
            input_topic = args.get("input_topic")
            if not input_topic:
                raise ValueError("input_topic is required for start action")

            node_key = instance_id or input_topic

            existing = self._nodes.get(node_key)
            if existing is not None and existing._input_topic != input_topic:
                self._retire_node(node_key)

            if node_key not in self._nodes:
                adapter = self._adapter
                language = self._language

                if instance_id and instance_id in self._instance_configs:
                    inst_adapter = self._adapter_for_instance(instance_id)
                    if inst_adapter:
                        adapter = inst_adapter
                    inst_lang = self._instance_configs[instance_id].get("language")
                    if inst_lang:
                        language = inst_lang

                node = _OCRNode(
                    input_topic, adapter, language,
                    node_suffix=node_key.replace('/', '_').replace('-', '_'),
                    min_interval=float(
                        {**self._plugin_cfg,
                         **self._instance_configs.get(instance_id, {})}
                        .get('min_interval_ms', 0)
                    ) / 1000.0,
                )
                self._executor.add_node(node)
                self._nodes[node_key] = node
            elif self._nodes[node_key].state != "running":
                self._configure_node(self._nodes[node_key], instance_id)

            return self._nodes[node_key].start()

        elif action == "stop":
            if instance_id and instance_id in self._nodes:
                return self._nodes[instance_id].stop()
            elif not instance_id and self._nodes:
                for node in self._nodes.values():
                    node.stop()
                return {"state": "idle"}
            return {"state": "idle"}

        elif action == "config":
            cfg = {k: v for k, v in args.items() if k not in ('action', 'instance_id') and v is not None and v != ''}

            if instance_id:
                previous = self._instance_configs.get(instance_id, {})
                self._instance_configs[instance_id] = {**previous, **cfg}
                self._instance_adapters.pop(instance_id, None)
                if instance_id in self._nodes:
                    node = self._nodes[instance_id]
                    node.stop()
                    self._configure_node(node, instance_id)
                return {"status": "configured", "instance_id": instance_id}
            else:
                updated_cfg = {**self._plugin_cfg, **cfg}
                rebuild = (
                    _adapter_signature(updated_cfg)
                    != _adapter_signature(self._plugin_cfg)
                )
                if rebuild:
                    self._adapter = _build_ocr_adapter(updated_cfg)
                    self._instance_adapters.clear()
                self._plugin_cfg = updated_cfg
                self._language = updated_cfg.get('language', self._language)
                for node_key, node in self._nodes.items():
                    node.stop()
                    self._configure_node(node, node_key)
                return {
                    "status": "configured",
                    "adapter_ok": self._adapter is not None,
                    "reused": not rebuild,
                }

        return None
