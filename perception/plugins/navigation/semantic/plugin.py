"""Phanthy Perception plugin wrapper for vision-and-language navigation."""

from __future__ import annotations

from .processor import Processor


class VisionAndLanguageNavigationPlugin:
    PREFIX = "vln"

    def __init__(
        self,
        plugin_cfg: dict,
        namespace: str,
        executor,
        **processor_dependencies,
    ):
        self._processor = Processor(
            plugin_cfg,
            namespace,
            executor,
            **processor_dependencies,
        )

    def get_tools(self) -> list:
        return [self._processor.manifest]

    def dispatch(self, name: str, args: dict) -> dict | None:
        if name != self.PREFIX:
            return None
        if not isinstance(args, dict):
            return self._processor.error(
                "invalid_argument",
                "arguments must be a JSON object",
            )

        action = args.get("action")
        if action == "start":
            return self._processor.start(args)
        if action == "stop":
            return self._processor.stop()
        if action == "info":
            return self._processor.info()
        if action == "config":
            return self._processor.configure(args)
        if action == "capture":
            return self._processor.capture()
        if action == "navigate":
            return self._processor.navigate(
                args.get("query"),
                control_nav_id=args.get("_control_nav_id"),
            )
        return self._processor.error(
            "unsupported_action",
            f"unknown vln action: {action!r}",
        )

    def stop(self) -> dict:
        """Compatibility hook used by PerceptionBundle shutdown."""

        return self._processor.stop()

    def close(self) -> None:
        self.stop()
