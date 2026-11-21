"""Native PerceptionBundle MCP processor; optional ROS, no actuator permissions."""
import base64
import hashlib
import threading
import time
from .backends import capabilities
from .effects import Effects
from .sources import create_source
from .service import Service


CONFIG_PROPERTIES = {
    "detector_backend": {"type": "string", "enum": ["yunet", "mediapipe"], "default": "yunet", "title": "检测流水线"},
    "landmarker_backend": {"type": "string", "enum": ["osf", "hybrid"], "default": "osf", "title": "关键点流水线"},
    "face_landmarker_model": {"type": "string", "title": "本地 MediaPipe Face Landmarker 模型（可选）"},
    "source_kind": {"type": "string", "enum": ["ros_compressed", "ros_image", "file", "usb", "rtsp"], "default": "ros_compressed", "title": "视频源类型"},
    "source_uri": {"type": "string", "title": "文件路径或 USB 设备", "description": "RTSP 凭据仅通过服务端配置提供"},
    "detector_model": {"type": "string", "title": "本地 YuNet ONNX"},
    "landmark_model": {"type": "string", "title": "本地 OpenSeeFace ONNX（可选）"},
    "single_face_landmark_model": {"type": "string", "title": "单人高精度关键点模型（可选，多人仍用基础模型）"},
    "flow_bridge": {"type": "boolean", "default": True, "title": "单人短时漏检像素跟踪（最多2帧/100ms）"},
    "provider": {"type": "string", "enum": ["cpu", "cuda", "tensorrt"], "default": "cpu"},
    "renderer": {"type": "string", "enum": ["cpu", "auto", "gl"], "default": "cpu"},
    "detect_interval": {"type": "integer", "minimum": 1, "maximum": 5, "default": 1},
    "detect_side": {"type": "integer", "minimum": 160, "maximum": 1920, "default": 640, "title": "检测输入最长边"},
    "detect_upscale": {"type": "boolean", "default": False, "title": "小脸检测上采样（更高耗时）"},
    "max_faces": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
    "effect": {"type": "string", "enum": ["all", "crown", "glasses", "orbit", "none"], "default": "all"},
}
for _field in CONFIG_PROPERTIES.values():
    _field["scope"] = "instance"

TOOLS = [{"name": "face_effects_live", "type": "processor", "multiInstance": True,
          "description": "通用实时人脸特效：贴纸、眼镜、动态装饰。输入摄像头或本地视频；不提供隐私脱敏保证。",
          "inputSchema": {"type": "object", "properties": {
              "action": {"type": "string", "enum": ["info", "start", "stop", "config", "list_effects", "select_effects", "snapshot"]},
              "instance_id": {"type": "string"}, "input_topic": {"type": "string"},
              "effects": {"type": "array", "items": {"type": "string"}},
              **CONFIG_PROPERTIES}, "required": ["action"],
              "x-action-params": {
                  "start": {"params": ["input_topic"], "description": "启动配置的视频源"},
                  "stop": {"params": [], "description": "停止并释放资源"},
                  "info": {"params": ["input_topic"], "description": "状态和输出端口"},
                  "config": {"params": [], "description": "停止后修改配置"},
                  "list_effects": {"params": [], "description": "列出已安装特效"},
                  "select_effects": {"params": ["effects"], "description": "切换或关闭特效"},
                  "snapshot": {"params": [], "description": "最近处理结果 JPEG 与元数据"}}},
          "configSchema": {"type": "object", "properties": CONFIG_PROPERTIES},
          "topic_in": [{"format": "image/jpeg", "desc": "camera"}],
          "topic_out": [{"format": "image/jpeg", "desc": "face effects"},
                        {"format": "data/json", "desc": "timing and tracking"}]}]


def validate_config(values):
    for key, schema in CONFIG_PROPERTIES.items():
        if key not in values:
            continue
        value = values[key]
        if schema["type"] == "integer":
            if isinstance(value, bool) or not isinstance(value, int) or not schema["minimum"] <= value <= schema["maximum"]:
                raise ValueError(f"invalid {key}")
        elif schema["type"] == "boolean":
            if type(value) is not bool:
                raise ValueError(f"invalid {key}")
        elif not isinstance(value, str) or len(value) > 4096:
            raise ValueError(f"invalid {key}")
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"invalid {key}")
    if "input_topic" in values and (not isinstance(values["input_topic"], str) or len(values["input_topic"]) > 512):
        raise ValueError("invalid input_topic")


class FaceEffectsLivePlugin:
    PREFIX = "face_effects_live"

    def __init__(self, plugin_cfg, executor=None):
        self.cfg, self.executor = dict(plugin_cfg), executor
        self.lock = threading.RLock()  # control calls serialized, workers never take it
        self.instances, self.configs, self.outputs = {}, {}, {}
        self.errors = {}

    def get_tools(self):
        import copy
        return copy.deepcopy(TOOLS)

    def _topic(self, key):
        # ROS2 tokens must start with a letter or underscore. An unprefixed
        # hexadecimal digest fails for instance IDs whose hash starts 0..9.
        return "/face_effects_live/i_"+hashlib.sha256(key.encode()).hexdigest()[:12]+"/image"

    def _config(self, key):
        stored = self.configs.get(key, {})
        result = {**self.cfg, **stored}
        if "effect" in stored and "effects" not in stored:
            result.pop("effects", None)
        return result

    def _info(self, key, input_topic=""):
        service = self.instances.get(key)
        cfg = self._config(key)
        topic = cfg.get("input_topic") or input_topic
        output = self.outputs.get(key)
        # Canvas resolves processor wires by info(instance_id,input_topic)
        # BEFORE start. Advertise deterministic per-instance ROS ports eagerly.
        publish_topic = output.topic if output else (self._topic(key) if self.executor is not None else None)
        return {"ok": True, "name": "Face Effects Live", "manufacture": "Embodied", "model": self.PREFIX,
                "desc": "贴纸 / 眼镜 / 动态装饰；短时几何与外观关联，无身份识别或隐私保护保证",
                **(service.status() if service else {"state": "error" if key in self.errors else "idle",
                                                    "error": self.errors.get(key, "")}),
                "topic_in": [{"topic": topic, "format": "image/raw" if cfg.get("source_kind")=="ros_image" else "image/jpeg", "desc": "camera"}] if topic else [],
                "topic_out": ([{"topic": publish_topic, "format": "image/jpeg", "desc": "effects"},
                               {"topic": publish_topic+"/metadata", "format": "data/json", "desc": "status"}] if publish_topic else []),
                "capabilities": capabilities(), "canvas_transport": "ros2" if publish_topic else "offline-snapshot",
                "permissions": ["image_input", "image_output"]}

    def dispatch(self, name, args):
        action = args.get("action") if name == self.PREFIX else name
        key = str(args.get("instance_id") or "default")
        if len(key) > 128:
            return {"ok": False, "error": "instance_id too long"}
        with self.lock:
            try:
                if action == "info":
                    if not args.get("instance_id") and self.instances:
                        result = self._info(key, args.get("input_topic", ""))
                        infos = {k: self._info(k) for k in self.instances}
                        result["instances"] = {k: {"state": i["state"]} for k, i in infos.items()}
                        result["state"] = "running" if any(i["state"] == "running" for i in infos.values()) else "idle"
                        result["topic_in"] = [p for i in infos.values() for p in i["topic_in"]]
                        result["topic_out"] = [p for i in infos.values() for p in i["topic_out"]]
                        return result
                    return self._info(key, args.get("input_topic", ""))
                if action == "list_effects":
                    effects = Effects(self._config(key).get("effects_manifest"))
                    try:
                        return {"ok": True, "effects": list(effects.plugins)}
                    finally:
                        effects.close()
                if action == "stop":
                    keys = [key] if args.get("instance_id") else list(set(self.instances) | set(self.errors))
                    results = {k: self._stop(k) for k in keys}
                    ok = all(r["state"] != "error" for r in results.values())
                    return {"ok": ok, "state": "idle" if ok else "error", "instances": results}
                if action == "config":
                    service = self.instances.get(key)
                    if service is not None:
                        raise ValueError("stop instance before changing configuration")
                    values = {k: v for k, v in args.items() if k in CONFIG_PROPERTIES or k == "input_topic"}
                    validate_config(values)
                    if "effect" in values:
                        self.configs.get(key, {}).pop("effects", None)
                    self.configs[key] = {**self.configs.get(key, {}), **values}
                    return {"ok": True, "status": "configured"}
                if action == "start":
                    return self._start(key, args)
                service = self.instances.get(key)
                if action == "select_effects":
                    names = args.get("effects")
                    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
                        raise ValueError("effects must be a string array")
                    effects = Effects(self._config(key).get("effects_manifest"))
                    try:
                        effects.select(names)
                    finally:
                        effects.close()
                    self.configs[key] = {**self.configs.get(key, {}), "effects": names}
                    if service:
                        service.select(names)
                    return {"ok": True, "effects": names}
                if action == "snapshot":
                    if service is None:
                        raise ValueError("instance not started")
                    with service.condition:
                        if service.jpeg is None or time.monotonic()-service.last_output_at > service.config.get("max_age_ms", 250)/1000:
                            return {"ok": False, "error": "no fresh output"}
                        return {"ok": True, "jpeg_base64": base64.b64encode(service.jpeg).decode(), "metadata": service.latest}
                return None
            except Exception as exc:
                # User errors are actionable; SDK exceptions are redacted.
                detail = str(exc) if isinstance(exc, (ValueError, KeyError)) else type(exc).__name__
                if action == "start":
                    detail = self.errors.get(key, detail)
                return {"ok": False, "state": "error", "error": detail}

    def _start(self, key, args):
        self.errors.pop(key, None)
        existing = self.instances.get(key)
        if existing is not None:
            if existing.state == "running":
                return self._info(key)
            status = self._stop(key)
            if status["state"] == "error":
                raise RuntimeError("previous worker still alive")
        if len(self.instances) >= 4:
            raise ValueError("maximum four instances")
        cfg = self._config(key)
        cfg.update({k: v for k, v in args.items() if k in CONFIG_PROPERTIES or k == "input_topic"})
        if "effect" in args:
            cfg.pop("effects", None)
        validate_config(cfg)
        kind = cfg.get("source_kind", "ros_compressed")
        uri = cfg.get("source_uri", "")
        if kind == "rtsp":
            # URI may be secret and must not pass through MCP argument logging.
            uri = self.cfg.get("rtsp_url", "")
        source_cfg = {"kind": kind, "path": uri, "url": uri, "topic": cfg.get("input_topic", ""),
                      "device": int(uri) if isinstance(uri, str) and uri.isdecimal() else uri}
        effect = cfg.get("effect", "all")
        if "effects" not in cfg:
            cfg["effects"] = ["crown", "glasses", "orbit"] if effect == "all" else ([] if effect == "none" else [effect])
        output = source = service = None
        try:
            source = create_source(source_cfg, self.executor)
            if self.executor is not None:
                from .outputs import ROSOutput
                output = ROSOutput(self._topic(key), self.executor)
            service = Service(cfg, source, output)
            self.instances[key] = service
            if output:
                self.outputs[key] = output
            self.configs[key] = {**self.configs.get(key, {}), **{k: v for k, v in cfg.items() if k != "rtsp_url"}}
            service.start()
            return self._info(key)
        except Exception as exc:
            self.errors[key] = (service.error + "; check local models and selected backend"
                                if service and service.error else
                                (str(exc) if isinstance(exc, ValueError) else "start: " + type(exc).__name__))
            if service is not None:
                service.stop()
                alive = any(t and t.is_alive() for t in (service.capture, service.processor))
                if alive:
                    raise RuntimeError("initialization failed; worker still alive")
            elif source is not None:
                source.close()
                finish = getattr(source, "finish", None)
                if finish:
                    finish()
            if output:
                output.close()
            self.outputs.pop(key, None)
            self.instances.pop(key, None)
            raise

    def _stop(self, key):
        service = self.instances.get(key)
        status = service.stop() if service else {"state": "stopped"}
        alive = service and any(t and t.is_alive() for t in (service.capture, service.processor))
        if not alive:
            self.errors.pop(key, None)
            self.instances.pop(key, None)
            output = self.outputs.pop(key, None)
            if output:
                output.close()
            return {"state": "stopped"}
        return status

    def shutdown(self):
        with self.lock:
            for key in list(self.instances):
                self._stop(key)
