"""Bounded QQ Bot media bridge: one rich-media reply per verified trigger.

QQ provides SDK media upload for C2C and group; guild media is not supported
by the installed SDK. Untrusted inbound URLs are fetched only from QQ CDN
hosts, over HTTPS, without redirects, with a streaming size cap.
"""
from __future__ import annotations

from collections import Counter
import mimetypes
import os
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
_MEDIA_FAILURES: Counter[str] = Counter()


class QQMediaUploadError(RuntimeError):
    """The media upload failed before QQ's message POST was attempted."""


class QQMediaPostError(RuntimeError):
    """The media message POST failed; delivery may be ambiguous."""


def _note_media_failure(reason: str) -> None:
    """Feishu-style sampled diagnostics; never log signed CDN URLs."""
    _MEDIA_FAILURES[reason] += 1
    count = _MEDIA_FAILURES[reason]
    if count == 1 or count % 100 == 0:
        print(f"[qq-media] rejected reason={reason} count={count}", flush=True)


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
            _note_media_failure("untrusted_or_missing_cdn_url")
            continue
        mime = str(getattr(att, "content_type", "") or "application/octet-stream").split(";", 1)[0].strip().lower()
        name = str(getattr(att, "filename", "") or "")[:120]
        kind = (KIND_IMAGE if mime.startswith("image/") else
                KIND_VIDEO if mime.startswith("video/") else
                KIND_AUDIO if mime.startswith("audio/") or mime == "voice" else KIND_FILE)
        path = None
        try:
            async with client.stream("GET", url, follow_redirects=False, timeout=15.0) as response:
                if response.status_code != 200:
                    _note_media_failure("cdn_http_not_200")
                    continue
                length = int(response.headers.get("content-length", "0") or 0)
                if length > _MAX_INBOUND:
                    _note_media_failure("attachment_too_large")
                    continue
                cache.mkdir(mode=0o700, parents=True, exist_ok=True)
                path = cache / (uuid4().hex + _extension(mime, name))
                total = 0
                # QQ attachments are private conversations; never create
                # world-readable copies even under a protected cache directory.
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as output:
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
        except Exception as exc:
            # Report the error category only, never signed URL/body.
            _note_media_failure("download_" + type(exc).__name__)
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
    return result


def _validate_outbound_file(attachment: Attachment) -> None:
    # Match Core's established /work and /tmp output policy. The previous
    # qq_outbox-only directory was never populated by a production tool.
    from event.desktop import _check_path_allowed
    source = Path(attachment.path)
    if source.is_symlink():
        raise ValueError("QQ media source cannot be a symlink")
    path = source.resolve(strict=True)
    if _check_path_allowed(path):
        raise ValueError("QQ media source is outside allowed Core output directories")
    if not path.is_file():
        raise ValueError("QQ attachment is not a regular file")
    # Never transmit QQ's own saved AppSecret through the channel, even when
    # the deployment database lives under an otherwise allowed /work volume.
    from channel.qq_credential_store import directory as credential_directory
    if path.is_relative_to(credential_directory().resolve(strict=False)):
        raise ValueError("QQ managed credential files cannot be sent")
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
    try:
        file_info = await uploader.upload(
            chat_type=scope, chat_id=chat_id, source=attachment.path,
            file_type=getattr(constants, _FILE_TYPES[attachment.kind]),
            file_name=attachment.name or Path(attachment.path).name,
        )
        # Constructing the DTO and allocating a sequence are local operations;
        # no QQ message POST has happened yet, so failures remain retryable.
        outbound = MessageToCreate(
            content=" ",  # QQ OpenAPI v2 currently requires content for rich media.
            msg_type=QQMessageType.RICH_MEDIA, msg_id=mid,
            msg_seq=api.next_msg_seq(), media=MediaInfo(file_info=file_info),
        )
    except Exception as exc:
        raise QQMediaUploadError(
            "QQ media upload failed before message delivery") from exc
    try:
        if scope == "c2c":
            await api.post_c2c_message(chat_id, outbound)
        else:
            await api.post_group_message(chat_id, outbound)
    except Exception as exc:
        # The POST may have reached QQ even when the SDK reports an error.
        raise QQMediaPostError(
            "QQ media message delivery is unconfirmed") from exc
