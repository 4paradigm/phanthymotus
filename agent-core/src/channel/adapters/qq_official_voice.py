"""Optional QQ voice transcription over the official SDK STT pipeline.

The file was already downloaded through the bounded QQ CDN media bridge.
The SDK STT provider reads its credentials from QQ_STT_* environment variables.
No user-controlled text may supply a backend URL or API credential.
"""
from __future__ import annotations

from pathlib import Path

from channel.adapter import KIND_AUDIO


async def read_voice_text(http_client, event, attachments) -> str:
    """Use QQ's ASR hint or transcribe one already-verified local voice file."""
    originals = list(getattr(event, "attachments", None) or [])
    for element in getattr(event, "msg_elements", None) or []:
        originals.extend(getattr(element, "attachments", None) or [])
    for item in originals:
        hint = getattr(item, "asr_refer_text", "")
        if isinstance(hint, str) and hint.strip():
            return hint.strip()[:2000]

    audio_files = [a for a in attachments if a.kind == KIND_AUDIO]
    if not audio_files:
        return ""
    try:
        from qqbot_agent_sdk import STTPipeline, resolve_stt_config
        from qqbot_agent_sdk.audio import convert_audio_to_wav
        stt = resolve_stt_config({})  # Environment only, no inline secrets.
        if stt is None:
            return ""
        path = Path(audio_files[0].path)
        if not path.is_file() or not 0 < path.stat().st_size <= 10 * 1024 * 1024:
            return ""
        wav_file = await convert_audio_to_wav(
            path.read_bytes(), source_hint=path.name)
        if not wav_file:
            return ""
        try:
            pipeline = STTPipeline(
                http_client=http_client, stt_config_fn=lambda: stt,
                downloader=None)
            # The pinned SDK exposes this call on its STTPipeline. Use the
            # safe local WAV instead of letting the SDK re-fetch arbitrary URLs.
            answer = await pipeline._call_stt_api(wav_file)
            return answer.strip()[:2000] if isinstance(answer, str) else ""
        finally:
            Path(wav_file).unlink(missing_ok=True)
    except Exception:
        return ""  # Audio failures do not block regular message delivery.
