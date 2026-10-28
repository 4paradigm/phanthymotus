"""VITS2 TensorRT implementation for the standard ``tts`` plugin."""

from plugins.vits2_tts_trt.plugin import TTSPlugin as _Vits2TTSPlugin


class Vits2TTSPlugin(_Vits2TTSPlugin):
    """Expose VITS2 through the standard TTS MCP prefix."""

    PREFIX = "tts"
