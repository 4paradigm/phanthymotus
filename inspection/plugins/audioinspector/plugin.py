from __future__ import annotations

from plugins.base import InspectorPlugin


class AudioInspectorPlugin(InspectorPlugin):
    def __init__(self) -> None:
        super().__init__(
            card_id="audioinspector",
            display_name="Audio Inspector",
            input_format="audio/pcm-16k",
            input_description="PCM_S16_LE, 16000 Hz, mono",
            instance_properties={
                "segment_seconds": {"type": "integer", "minimum": 5, "maximum": 600, "default": 60, "scope": "instance"},
                "local_retention_hours": {"type": "number", "minimum": 1, "maximum": 720, "default": 24, "scope": "instance"},
                "local_max_gb": {"type": "number", "minimum": 0.1, "default": 4, "scope": "instance"},
                "container": {"type": "string", "enum": ["wav"], "default": "wav", "scope": "instance"},
                "queue_seconds": {"type": "number", "minimum": 1, "default": 5, "scope": "instance"},
                "record_mode": {"type": "string", "enum": ["continuous"], "default": "continuous", "scope": "instance"},
                "auto_resume_after_reboot": {"type": "boolean", "default": False, "scope": "instance"},
            },
        )
