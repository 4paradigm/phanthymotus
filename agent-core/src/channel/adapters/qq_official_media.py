"""Bounded QQ Bot media bridge: one rich-media reply per verified trigger.

QQ provides SDK media upload for C2C and group; guild media is not supported
by the installed SDK. Untrusted inbound URLs are fetched only from QQ CDN
hosts, over HTTPS, without redirects, with a streaming size cap.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from channel.adapter import Attachment, KIND_AUDIO, KIND_FILE, KIND_IMAGE, KIND_VIDEO

_MAX_INBOUND = 10 * 1024 * 1024
_MAX_OUTBOUND = 20 * 1024 * 1024
_MAX_MEDIA_PER_EVENT = 3
_QQ_CDN_SUFFIXES = (
    "qq.com", "qq.com.cn", "qpic.cn", "gtimg.cn",
)
_FILE_TYPES = {
    KIND_IMAGE: "MEDIA_TYPE_IMAGE",
    KIND_VIDEO: "MEDIA_TYPE_VIDEO",
    KIND_AUDIO: "MEDIA_TYPE_VOICE",
    KIND_FILE: "MEDIA_TYPE_FILE",
}


def _allowed_qq_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
        return (parsed.scheme == "https" and parsed.username is None
                and parsed.password is None and parsed.port in (None, 443)
                and any(host == suffix or host.endswith("." + suffix)
                        for suffix in _QQ_CDN_SUFFIXES))
    except ValueError:
        return False


def _extension(mime: str, name: str) -> str:
    ext = (Path(name or "").suffix or mimetypes.guess_extension(mime) or ".bin").lower()
    if len(ext) > 10 or not ext[1:].isalnum():
        return ".bin"
    return ext


async def receive_media(client, event, cache_root: str) -> list[Attachment]:
    """Download up to 3 bounded QQ CDN attachments into persistent storage."""
    sources = list(getattr(event, "attachments", None) or [])
    for element in getattr(event, "msg_elements", None) or []:
        sources.extend(getattr(element, "attachments", None) or [])
    if not sources:
        return []
    cache = Path(cache_root)
    result: list[Attachment] = []
    for att in sources[:_MAX_MEDIA_PER_EVENT]:
        url = getattr(att, "url", "")
        if not isinstance(url, str) or not _allowed_qq_url(url):
            continue
        mime = str(getattr(att, "content_type", "") or "application/octet-stream").lower()
        name = str(getattr(att, "filename", "") or "")[:120]
        kind = (KIND_IMAGE if mime.startswith("image/") else
                KIND_VIDEO if mime.startswith("video/") else
                KIND_AUDIO if mime.startswith("audio/") or mime == "voice" else KIND_FILE)
        path = None
        try:
            async with client.stream("GET", url, follow_redirects=False, timeout=15.0) as response:
                if response.status_code != 200:
                    continue
                length = int(response.headers.get("content-length", "0") or 0)
                if length > _MAX_INBOUND:
                    continue
                cache.mkdir(mode=0o700, parents=True, exist_ok=True)
                path = cache / (uuid4().hex + _extension(mime, name))
                total = 0
                with path.open("xb") as output:
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > _MAX_INBOUND:
                            raise ValueError("QQ attachment exceeds limit")
                        output.write(chunk)
                if total:
                    result.append(Attachment(
                        kind=kind, path=str(path), name=name or path.name,
                        mime=mime, size=total))
                    path = None
        except Exception:
            # Media errors must never leak signed URLs or credentials into logs.
            pass
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
    return result


def _validate_outbound_file(attachment: Attachment) -> None:
    path = Path(attachment.path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("QQ attachment is not a regular file")
    if not path.is_relative_to(Path("/work/resource/qq_outbox")):
        raise ValueError("QQ media source must be in /work/resource/qq_outbox")
    if not 0 < path.stat().st_size <= _MAX_OUTBOUND:
        raise ValueError("QQ attachment size exceeds safe limit")
    if attachment.kind not in _FILE_TYPES:
        raise ValueError("QQ media type unsupported")


async def send_media(api, http_client, scope: str, chat_id: str, mid: str,
                     attachment: Attachment) -> None:
    """Upload one local file and submit one authenticated passive media reply."""
    if scope not in ("c2c", "group"):
        raise ValueError("QQ SDK only supports C2C/group rich media")
    _validate_outbound_file(attachment)
    from qqbot_agent_sdk import MediaUploader, MediaInfo, MessageToCreate, QQMessageType
    from qqbot_agent_sdk import constants

    uploader = MediaUploader(api_client=api, http_client=http_client)
    file_info = await uploader.upload(
        chat_type=scope, chat_id=chat_id, source=attachment.path,
        file_type=getattr(constants, _FILE_TYPES[attachment.kind]),
        file_name=attachment.name or Path(attachment.path).name,
    )
    outbound = MessageToCreate(
        content=" ",  # QQ OpenAPI v2 currently requires content for rich media.
        msg_type=QQMessageType.RICH_MEDIA, msg_id=mid,
        msg_seq=api.next_msg_seq(), media=MediaInfo(file_info=file_info),
    )
    if scope == "c2c":
        await api.post_c2c_message(chat_id, outbound)
    else:
        await api.post_group_message(chat_id, outbound)
