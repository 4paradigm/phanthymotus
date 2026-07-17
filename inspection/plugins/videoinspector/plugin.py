from __future__ import annotations

from plugins.base import InspectorPlugin


class VideoInspectorPlugin(InspectorPlugin):
    def __init__(self) -> None:
        super().__init__(
            card_id="videoinspector",
            display_name="Video Inspector",
            input_format="image/jpeg",
            input_description="sensor_msgs/CompressedImage JPEG frames",
            instance_properties={
                "segment_seconds": {"type": "integer", "minimum": 5, "maximum": 600, "default": 60, "scope": "instance"},
                "local_retention_hours": {"type": "number", "minimum": 1, "maximum": 168, "default": 6, "scope": "instance"},
                "local_max_gb": {"type": "number", "minimum": 0.1, "default": 20, "scope": "instance"},
                "encoder": {"type": "string", "enum": ["nvv4l2h264enc", "libx264"], "default": "nvv4l2h264enc", "scope": "instance"},
                "target_bitrate_kbps": {"type": "integer", "minimum": 256, "maximum": 20000, "default": 4000, "scope": "instance"},
                "max_fps": {"type": "number", "minimum": 1, "default": 30, "scope": "instance"},
                "queue_frames": {"type": "integer", "minimum": 1, "default": 8, "scope": "instance"},
                "auto_resume_after_reboot": {"type": "boolean", "default": False, "scope": "instance"},
            },
        )
