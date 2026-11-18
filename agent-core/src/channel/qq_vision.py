"""Attach validated QQ image uploads to an existing decision Core LLM turn.

No network download here. Only bounded files already received via the QQ
adapter's private media directory are eligible. Never trust a path from chat.
"""
from __future__ import annotations

from pathlib import Path
import stat

# Shared Feishu/QQ inbound storage; QQ is restricted to its own channel.
_QQ_MEDIA_ROOT = Path("/work/resource/channel_files")
_ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_IMAGES_PER_TURN = 2
_NO_VISION = (
    "[QQ image received but vision input is disabled. "
    "Do not claim to see or describe its contents; tell the user "
    "that image understanding is unavailable.]"
)
_NO_IMAGE = (
    "[QQ image unavailable or unreadable. Do not infer its contents "
    "from the filename or user text.]"
)


def _approved_image_paths(trigger: dict, media_root: Path = _QQ_MEDIA_ROOT) -> list[Path]:
    """Return only existing regular, private QQ image files from this event."""
    if not trigger.get("_qq_channel_turn"):
        return []
    files = trigger.get("_qq_image_files", [])
    if not isinstance(files, list):
        return []
    # Use the authenticated Collector identity, not an optional payload field.
    # The files still have to reside under this exact QQ Channel's directory.
    name = trigger.get("_qq_channel_id")
    if not isinstance(name, str) or not name:
        return []
    base = media_root.resolve(strict=False)
    if base == _QQ_MEDIA_ROOT.resolve(strict=False):
        from channel.store import _safe_name
        base = base / _safe_name(name)
    result = []
    for item in files[:_MAX_IMAGES_PER_TURN]:
        if not isinstance(item, dict) or item.get("kind") != "image":
            continue
        path = item.get("path", "")
        if not isinstance(path, str) or not Path(path).is_absolute():
            continue
        candidate = Path(path)
        if candidate.suffix.lower() not in _ALLOWED_IMAGE_EXT:
            continue
        try:
            # Symlink targets outside the dedicated QQ media root are rejected.
            if not candidate.resolve(strict=True).is_relative_to(base):
                continue
            metadata = candidate.stat()
            if (not stat.S_ISREG(metadata.st_mode)
                    or not 0 < metadata.st_size <= _MAX_IMAGE_BYTES):
                continue
        except (OSError, ValueError):
            continue
        result.append(candidate)
    return result


def redact_inline_images(messages: list[dict]) -> list[dict]:
    """Remove encoded images from diagnostic request logs, not LLM requests."""
    clean = []
    for message in messages:
        item = dict(message)
        if isinstance(item.get("content"), list):
            item["content"] = [
                {"type": "image_url", "image_url": "[image bytes omitted]"}
                if isinstance(part, dict) and part.get("type") == "image_url"
                else part
                for part in item["content"]
            ]
        clean.append(item)
    return clean


async def _gif_frames(path: Path) -> list[dict]:
    """Sample up to two bounded JPEG frames from a QQ GIF via existing ffmpeg.

    ChatCompletions vision endpoints are more interoperable with JPEG than
    animated GIF payloads. This samples frames, not a complete motion analysis.
    """
    import asyncio
    import base64

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-vf", "fps=1,scale='min(640,iw)':-2",
        "-frames:v", "2", "-q:v", "5",
        "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        data, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise ValueError("GIF frame decoding timed out") from None
    if proc.returncode != 0 or not data or len(data) > 3 * 1024 * 1024:
        raise ValueError("GIF frame decoding failed or exceeded limit")
    frames = []
    position = 0
    while len(frames) < 2:
        start = data.find(b"\xff\xd8", position)
        if start < 0:
            break
        end = data.find(b"\xff\xd9", start + 2)
        if end < 0:
            break
        blob = data[start:end + 2]
        if len(blob) > 0:
            frames.append({"type": "image_url", "image_url": "data:image/jpeg;base64,"
                           + base64.b64encode(blob).decode("ascii")})
        position = end + 2
    if not frames:
        raise ValueError("No decodable GIF frames")
    return [{"type": "text", "text":
             f"[当前 QQ 动图抽取 {len(frames)} 帧供视觉分析，不代表完整动画。]"}, *frames]


async def add_qq_images(messages: list[dict], trigger: dict,
                        *, media_root: Path = _QQ_MEDIA_ROOT,
                        reader=None, vision_enabled: bool | None = None) -> bool:
    """Replace only the triggering user text with a bounded multimodal list.

    Returns True if image bytes were attached. If vision is unavailable or the
    file was deleted, explicitly tell the LLM not to hallucinate image content.
    """
    if not messages or messages[-1].get("role") != "user":
        return False
    if not trigger.get("_qq_channel_turn"):
        return False
    declared = trigger.get("_qq_image_files")
    if not isinstance(declared, list) or not declared:
        return False
    from event.desktop import vision_input_enabled, _read_image
    enabled = vision_input_enabled() if vision_enabled is None else vision_enabled
    text = messages[-1].get("content", "")
    if not isinstance(text, str):
        return False
    if not enabled:
        messages[-1]["content"] = text + "\n" + _NO_VISION
        return False
    parts = [{"type": "text", "text": text}]
    for path in _approved_image_paths(trigger, media_root):
        try:
            contents = await (reader(path) if reader is not None else
                              _gif_frames(path) if path.suffix.lower() == ".gif"
                              else _read_image(path))
            if (not isinstance(contents, list)
                    or not any(isinstance(x, dict) and x.get("type") == "image_url"
                               for x in contents)):
                continue
            # Existing desktop reader returns an image_url string, whereas
            # DeepSeek Flash ChatCompletions expects {"url": "..."}.
            for part in contents:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part.get("image_url")
                    if isinstance(url, str):
                        part = {**part, "image_url": {"url": url}}
                parts.append(part)
        except (OSError, ValueError, RuntimeError):
            continue
    if len(parts) == 1:
        messages[-1]["content"] = text + "\n" + _NO_IMAGE
        return False
    messages[-1]["content"] = parts
    return True
