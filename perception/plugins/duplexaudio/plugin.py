from __future__ import annotations

import copy
import hashlib
import re
from typing import Any, Optional


TOOL = {
    "name": "duplexaudio",
    "type": "processor",
    "multiInstance": True,
    "description": (
        "Duplex audio front-end: timestamped TTS reference, external-mic AEC, "
        "and clean PCM output for a standalone ASR card"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "info",
                    "config",
                    "start",
                    "stop",
                    "speak",
                    "aec_stats",
                    "aec_calibrate",
                ],
                "description": "Card lifecycle, speech, or AEC diagnostic action",
            },
            "instance_id": {"type": "string"},
            "input_topic": {
                "type": "string",
                "description": "External-mic ROS2 audio/pcm-16k topic",
            },
            "text": {
                "type": "string",
                "description": "Text to synthesize for action=speak",
            },
        },
        "required": ["action"],
        "x-action-params": {
            "info": {"params": ["instance_id", "input_topic"]},
            "config": {"params": ["instance_id"]},
            "start": {"params": ["instance_id", "input_topic"]},
            "stop": {"params": ["instance_id"]},
            "speak": {"params": ["instance_id", "text"]},
            "aec_stats": {"params": ["instance_id"]},
            "aec_calibrate": {"params": ["instance_id"]},
        },
    },
    "configSchema": {
        "type": "object",
        "properties": {
            "speaker_id": {
                "type": "integer",
                "default": 0,
                "scope": "shared",
            },
            "speed": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 2.0,
                "default": 1.0,
                "scope": "shared",
            },
            "aec_enabled": {
                "type": "boolean",
                "default": True,
                "scope": "instance",
            },
            "aec_backend": {
                "type": "string",
                "enum": ["auto", "livekit", "speexdsp"],
                "default": "auto",
                "scope": "shared",
            },
            "aec_delay_ms": {
                "type": "integer",
                "minimum": 0,
                "maximum": 2500,
                "default": 100,
                "scope": "instance",
            },
            "aec_filter_length_ms": {
                "type": "integer",
                "minimum": 50,
                "maximum": 1000,
                "default": 200,
                "scope": "shared",
            },
            "aec_failure_policy": {
                "type": "string",
                "enum": ["fail_closed", "passthrough"],
                "default": "fail_closed",
                "scope": "instance",
            },
        },
        "required": [],
    },
    "topic_in": [
        {"format": "audio/pcm-16k", "desc": "external microphone PCM"}
    ],
    "topic_out": [
        {"format": "audio/pcm-16k", "desc": "AEC-cleaned PCM for ASR"},
        {"format": "audio/pcm-16k", "desc": "TTS PCM for speaker playback"},
    ],
}


_SHARED_PATHS = {
    "speaker_id": ("tts", "speaker_id"),
    "speed": ("tts", "speed"),
    "aec_backend": ("aec", "backend"),
    "aec_filter_length_ms": ("aec", "filter_length_ms"),
}
_INSTANCE_PATHS = {
    "aec_enabled": ("aec", "enabled"),
    "aec_delay_ms": ("aec", "delay_ms"),
    "aec_failure_policy": ("aec", "failure_policy"),
}


class _Session:
    def __init__(
        self,
        *,
        input_topic: str,
        clean_topic: str,
        tts_topic: str,
        bridge,
        tts_node,
        aec,
        aec_init_error: Optional[str] = None,
    ):
        self.input_topic = input_topic
        self.clean_topic = clean_topic
        self.tts_topic = tts_topic
        self.bridge = bridge
        self.tts_node = tts_node
        self.aec = aec
        self.aec_init_error = aec_init_error


class DuplexAudioPlugin:
    PREFIX = "duplexaudio"

    def __init__(self, plugin_cfg: dict, executor):
        self._cfg = copy.deepcopy(plugin_cfg)
        self._executor = executor
        self._instances: dict[str, _Session] = {}
        self._instance_cfg: dict[str, dict] = {}
        self._tts_adapter = None
        self._load_error: Optional[str] = None

    def get_tools(self) -> list[dict]:
        return [TOOL]

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == self.PREFIX else name
        instance_id = args.get("instance_id") or "default"

        if action == "info":
            session = self._instances.get(instance_id)
            return self._status(
                "running" if session is not None else "idle",
                args.get("input_topic", ""),
                session,
            )

        if action == "config":
            return self._configure(instance_id, args)

        if action == "start":
            input_topic = args.get("input_topic")
            if not input_topic:
                topics = args.get("input_topics") or []
                if topics:
                    input_topic = topics[0]
            if not input_topic:
                raise ValueError("input_topic is required for start")
            input_topic = input_topic.rstrip("/")
            if not input_topic:
                raise ValueError("input_topic must name a ROS audio topic")
            existing = self._instances.get(instance_id)
            if existing is not None and existing.input_topic == input_topic:
                return self._status("running", input_topic, existing)
            for active_id, session in self._instances.items():
                if active_id != instance_id and session.input_topic == input_topic:
                    return {
                        "state": "error",
                        "error": "input_topic_in_use",
                        "message": (
                            f"input topic {input_topic!r} is already owned by "
                            f"instance {active_id!r}"
                        ),
                    }
            if existing is not None:
                self._stop_instance(instance_id)
            return self._start_instance(instance_id, input_topic)

        if action == "stop":
            input_topic = ""
            existing = self._instances.get(instance_id)
            if existing is not None:
                input_topic = existing.input_topic
            self._stop_instance(instance_id)
            return self._status("idle", input_topic, None)

        if action == "speak":
            text = str(args.get("text") or "").strip()
            if not text:
                raise ValueError("text is required for speak")
            session = self._select_session(args.get("instance_id"))
            session.tts_node.enqueue(text, trace_id=args.get("_trace_id", ""))
            return {
                "status": "queued",
                "instance_id": self._instance_id_for(session),
                "text": text,
                "topic_out": session.tts_topic,
            }

        if action == "aec_stats":
            session = self._select_session(args.get("instance_id"))
            result = session.bridge.stats()
            result["init_error"] = session.aec_init_error
            return result

        if action == "aec_calibrate":
            session = self._select_session(args.get("instance_id"))
            if session.aec is None:
                return {
                    "ok": False,
                    "reason": session.aec_init_error or "AEC is disabled",
                }
            return session.aec.calibrate()

        return None

    def _start_instance(self, instance_id: str, input_topic: str) -> dict:
        cfg = self._effective_config(instance_id)
        aec_cfg = cfg["aec"]
        failure_policy = aec_cfg["failure_policy"]
        aec = None
        aec_init_error = None
        if aec_cfg["enabled"]:
            from .aec import AECProcessor

            try:
                aec = AECProcessor(
                    delay_ms=aec_cfg["delay_ms"],
                    backend=aec_cfg["backend"],
                    filter_length_ms=aec_cfg["filter_length_ms"],
                )
            except Exception as exc:
                aec_init_error = str(exc)
                if failure_policy == "fail_closed":
                    return {
                        "state": "error",
                        "error": "aec_unavailable",
                        "message": aec_init_error,
                    }

        try:
            tts_adapter = self._ensure_tts_adapter(cfg)
        except Exception as exc:
            if aec is not None:
                aec.close()
            self._load_error = str(exc)
            return {
                "state": "error",
                "error": "model_load_failed",
                "message": str(exc),
            }

        from plugins.tts import _TTSNode
        from .node import DuplexAudioNode

        safe_stem = re.sub(r"[^a-zA-Z0-9_]", "_", instance_id)[:40]
        instance_digest = hashlib.blake2s(
            instance_id.encode("utf-8"), digest_size=4
        ).hexdigest()
        safe_id = f"{safe_stem}_{instance_digest}"
        clean_topic = f"{input_topic}/duplexaudio/clean"
        tts_topic = f"{input_topic}/duplexaudio/tts"
        bridge = DuplexAudioNode(
            input_topic,
            clean_topic,
            safe_id,
            aec,
            failure_policy=failure_policy,
        )
        tts_node = _TTSNode(
            None,
            tts_adapter,
            node_suffix=f"duplexaudio_{safe_id}",
            output_topic=tts_topic,
            frame_observer=(aec.push_reference if aec is not None else None),
        )
        nodes = (bridge, tts_node)
        try:
            for node in nodes:
                self._executor.add_node(node)
            tts_result = tts_node.start()
            if tts_result.get("state") != "running":
                raise RuntimeError(tts_result.get("message") or "TTS start failed")
        except Exception as exc:
            self._cleanup_nodes(nodes, aec)
            return {
                "state": "error",
                "error": "start_failed",
                "message": str(exc),
            }

        session = _Session(
            input_topic=input_topic,
            clean_topic=clean_topic,
            tts_topic=tts_topic,
            bridge=bridge,
            tts_node=tts_node,
            aec=aec,
            aec_init_error=aec_init_error,
        )
        self._instances[instance_id] = session
        return self._status("running", input_topic, session)

    def _stop_instance(self, instance_id: str) -> None:
        session = self._instances.pop(instance_id, None)
        if session is None:
            return
        self._cleanup_nodes((session.tts_node, session.bridge), session.aec)

    def _cleanup_nodes(self, nodes, aec) -> None:
        for node in nodes:
            try:
                stop = getattr(node, "stop", None)
                if stop is not None:
                    stop()
            except Exception:
                pass
            try:
                self._executor.remove_node(node)
            except Exception:
                pass
            try:
                node.destroy_node()
            except Exception:
                pass
        if aec is not None:
            try:
                aec.close()
            except Exception:
                pass

    def _ensure_tts_adapter(self, cfg: dict):
        if self._tts_adapter is None:
            from plugins.tts import _build_tts_adapter

            self._tts_adapter = _build_tts_adapter(cfg["tts"])
            if self._tts_adapter is None:
                raise RuntimeError("TTS adapter did not load")
        self._load_error = None
        return self._tts_adapter

    def _configure(self, instance_id: str, args: dict) -> dict:
        shared_changed = False
        instance_changed = False
        for key, path in _SHARED_PATHS.items():
            if key in args:
                self._validate_config_value(key, args[key])
                self._set_path(self._cfg, path, args[key])
                shared_changed = True
        if instance_id not in self._instance_cfg:
            self._instance_cfg[instance_id] = {}
        for key, path in _INSTANCE_PATHS.items():
            if key in args:
                self._validate_config_value(key, args[key])
                self._set_path(self._instance_cfg[instance_id], path, args[key])
                instance_changed = True

        stopped_instances = []
        if shared_changed:
            stopped_instances = list(self._instances)
            for active_id in list(self._instances):
                self._stop_instance(active_id)
            self._tts_adapter = None
            self._load_error = None
        elif instance_changed and instance_id in self._instances:
            stopped_instances = [instance_id]
            self._stop_instance(instance_id)
        return {
            "status": "configured",
            "restart_required": bool(stopped_instances),
            "stopped_instances": stopped_instances,
        }

    @staticmethod
    def _validate_config_value(key: str, value: Any) -> None:
        if key == "speed" and not 0.5 <= float(value) <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0")
        if key == "aec_delay_ms" and not 0 <= int(value) <= 2500:
            raise ValueError("aec_delay_ms must be between 0 and 2500")
        if key == "aec_filter_length_ms" and not 50 <= int(value) <= 1000:
            raise ValueError("aec_filter_length_ms must be between 50 and 1000")
        if key == "aec_backend" and value not in {"auto", "livekit", "speexdsp"}:
            raise ValueError("aec_backend must be auto, livekit, or speexdsp")
        if key == "aec_failure_policy" and value not in {
            "fail_closed",
            "passthrough",
        }:
            raise ValueError("aec_failure_policy must be fail_closed or passthrough")

    def _effective_config(self, instance_id: str) -> dict:
        defaults = {
            "tts": {
                "model_dir": "/models/sherpa-onnx/tts",
                "speaker_id": 0,
                "speed": 1.0,
            },
            "aec": {
                "enabled": True,
                "backend": "auto",
                "delay_ms": 100,
                "filter_length_ms": 200,
                "failure_policy": "fail_closed",
            },
        }
        self._deep_merge(defaults, self._cfg)
        self._deep_merge(defaults, self._instance_cfg.get(instance_id, {}))
        return defaults

    @staticmethod
    def _set_path(target: dict, path: tuple[str, ...], value: Any) -> None:
        current = target
        for part in path[:-1]:
            current = current.setdefault(part, {})
        current[path[-1]] = value

    @classmethod
    def _deep_merge(cls, target: dict, source: dict) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                cls._deep_merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    def _select_session(self, instance_id: Optional[str]) -> _Session:
        if instance_id:
            session = self._instances.get(instance_id)
            if session is None:
                raise ValueError(f"duplexaudio instance {instance_id!r} is not running")
            return session
        if len(self._instances) == 1:
            return next(iter(self._instances.values()))
        if not self._instances:
            raise ValueError("duplexaudio is not running; call start first")
        raise ValueError("instance_id is required when multiple duplexaudio instances run")

    def _instance_id_for(self, session: _Session) -> str:
        for instance_id, candidate in self._instances.items():
            if candidate is session:
                return instance_id
        return ""

    def _status(
        self, state: str, input_topic: str, session: Optional[_Session]
    ) -> dict:
        if session is not None:
            input_topic = session.input_topic
            topic_out = [
                {
                    "topic": session.clean_topic,
                    "format": "audio/pcm-16k",
                    "desc": "AEC-cleaned PCM for ASR",
                },
                {
                    "topic": session.tts_topic,
                    "format": "audio/pcm-16k",
                    "desc": "TTS PCM for speaker",
                },
            ]
            aec = session.bridge.stats()
            aec["init_error"] = session.aec_init_error
            if session.bridge.state == "error":
                state = "error"
        else:
            input_topic = input_topic.rstrip("/")
            topic_out = (
                [
                    {
                        "topic": f"{input_topic}/duplexaudio/clean",
                        "format": "audio/pcm-16k",
                        "desc": "AEC-cleaned PCM for ASR",
                    },
                    {
                        "topic": f"{input_topic}/duplexaudio/tts",
                        "format": "audio/pcm-16k",
                        "desc": "TTS PCM for speaker",
                    },
                ]
                if input_topic
                else []
            )
            aec = {"enabled": False, "configured": True}
        return {
            "name": "Duplex audio",
            "state": state,
            "topic_in": (
                [
                    {
                        "topic": input_topic,
                        "format": "audio/pcm-16k",
                        "desc": "external microphone PCM",
                    }
                ]
                if input_topic
                else []
            ),
            "topic_out": topic_out,
            "aec": aec,
            "load_error": self._load_error,
        }
