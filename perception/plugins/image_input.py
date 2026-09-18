#!/usr/bin/env python3
"""
plugins/image_input.py — one way for a plugin to accept an image by reference.

Extracted from plugins/face.py when vop and ocr grew the same
`*_by_photo` / `*_by_url` actions. The rules below are security properties, not
conveniences, so the three plugins share one implementation rather than three
similar ones that can drift apart:

* **Paths are confined to configured roots.** The MCP server has no
  authentication and runs as root in the container, so an unrestricted path
  would let any LAN caller probe the filesystem by asking whether a file
  decodes as an image. Symlinks are resolved before the check — a link inside a
  root pointing outside it would otherwise pass.
* **Transfers are capped.** A guard on memory and download time, not a policy
  on image size: a merely *large* image is meant to be downscaled locally by
  the caller's decoder, because "your photo is 24 MB" is a limitation of ours
  rather than a property of their photo.
* **No base64 input.** It existed and an LLM failed on it twice in production:
  a 43 800-character string does not survive being carried through a model's
  context, and what arrived was truncated, so the decoder correctly refused it.
  Both remaining channels move a *reference* instead of the bytes.
* **`url` is its own action, never a parameter on the photo path.** It lets a
  caller make this container issue an outbound request, which is acceptable for
  a deliberate, named action and not for a generic input — and it keeps the
  capability visible on the card.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

__all__ = [
    "REASON_BAD_INPUT",
    "DEFAULT_URL_ACTION",
    "DEFAULT_MAX_IMAGE_BYTES",
    "DEFAULT_IMAGE_ROOTS",
    "BadInput",
    "load_image_bytes",
    "read_local",
    "fetch_url",
    "check_under_roots",
    "image_roots",
]

# Named in the "use X instead" half of every rejection. A caller told the
# exact action to switch to acts on it; one told "the _by_url action" has to
# go and find which one that is, so each plugin passes its own.
DEFAULT_URL_ACTION = "the _by_url action"

REASON_BAD_INPUT = "bad_input"

# Only has to be larger than any real photo; 64 MB covers a 60 MP original.
DEFAULT_MAX_IMAGE_BYTES = 64 * 1024 * 1024
DEFAULT_IMAGE_ROOTS = ("/models", "/tmp", "/work")


class BadInput(Exception):
    """An image could not be loaded. Carries the caller-facing detail."""

    def __init__(self, detail: str, source: str = ""):
        super().__init__(detail)
        self.detail = detail
        self.source = source

    def as_result(self) -> dict:
        result = {"ok": False, "reason": REASON_BAD_INPUT, "detail": self.detail}
        if self.source:
            result["source"] = self.source
        return result


def image_roots(cfg: dict) -> tuple[str, ...]:
    roots = cfg.get("image_roots") or DEFAULT_IMAGE_ROOTS
    return tuple(os.path.realpath(str(root)) for root in roots)


def check_under_roots(path: str, cfg: dict, url_action: str = DEFAULT_URL_ACTION) -> str:
    """Confine a caller-supplied path to the configured roots."""
    resolved = os.path.realpath(path)
    roots = image_roots(cfg)
    if any(resolved == root or resolved.startswith(root + os.sep) for root in roots):
        return resolved
    # A caller that names a plausible-but-invisible path is almost always
    # another container's filesystem — agent-core's /work and /tmp are its own,
    # which is exactly how the first LLM attempt failed. Say where to put it.
    raise BadInput(
        f"path must be under one of {', '.join(roots)}: got {path!r}. "
        "If you are calling from another container, upload the file through "
        "POST /api/mcp/<mcp_id>/file/upload — the reply carries a path this "
        "container can open — or use " + url_action + ".",
        path,
    )


def read_local(path: str, cfg: dict, max_bytes: int,
               url_action: str = DEFAULT_URL_ACTION) -> bytes:
    resolved = check_under_roots(path, cfg, url_action)
    try:
        if os.path.isdir(resolved):
            raise BadInput(f"{path!r} is a directory, not an image", path)
        size = os.path.getsize(resolved)
        if size > max_bytes:
            raise BadInput(
                f"file is {size} bytes, over the {max_bytes} byte transfer cap "
                "(raise max_image_bytes if this is a real photo)", path
            )
        with open(resolved, "rb") as handle:
            return handle.read()
    except OSError as error:
        raise BadInput(f"cannot read {path!r}: {error}", path) from error


class _Redirect308(urllib.request.HTTPRedirectHandler):
    """Follow 308, which this Python's redirect handler does not.

    `HTTPRedirectHandler` gained `http_error_308` in Python 3.11; jp5.11 runs
    3.8, so a permanent redirect surfaced to the caller as
    "cannot fetch ...: HTTP Error 308" — observed against
    https://ultralytics.com/images/bus.jpg, which is about as ordinary an image
    URL as exists. 308 differs from 301 only in preserving the method, and a
    GET redirected to a GET is the same request either way.
    """

    def http_error_308(self, req, fp, code, msg, headers):
        return self.http_error_301(req, fp, 301, msg, headers)


_OPENER = urllib.request.build_opener(_Redirect308())


def fetch_url(url: str, max_bytes: int) -> bytes:
    if not url.lower().startswith(("http://", "https://")):
        raise BadInput(f"only http(s) URLs are supported: {url!r}", url)
    try:
        with _OPENER.open(url, timeout=20) as response:
            data = response.read(max_bytes + 1)
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise BadInput(f"cannot fetch {url!r}: {error}", url) from error
    if len(data) > max_bytes:
        raise BadInput(f"download exceeds the {max_bytes} byte limit", url)
    if not data:
        raise BadInput(f"{url!r} returned no data", url)
    return data


def load_image_bytes(args: dict, cfg: dict,
                     url_action: str = DEFAULT_URL_ACTION) -> tuple[bytes, str]:
    """Read image bytes from `url` or `image_path`. Returns (bytes, source)."""
    max_bytes = int(cfg.get("max_image_bytes", DEFAULT_MAX_IMAGE_BYTES))

    if args.get("image_b64"):
        # Say what to do instead, rather than silently ignoring the argument:
        # the model that reaches for base64 has the file in hand already.
        raise BadInput(
            "image_b64 is no longer accepted — a long base64 string does not "
            "survive being carried through an LLM's context. Upload the file "
            "through POST /api/mcp/<mcp_id>/file/upload and pass the path it "
            "returns as image_path, or use " + url_action + ".",
            "image_b64",
        )

    url = args.get("url") or args.get("image_url")
    if url:
        return fetch_url(str(url), max_bytes), str(url)

    path = args.get("image_path")
    if path:
        return read_local(str(path), cfg, max_bytes, url_action), str(path)

    raise BadInput("one of image_path or url is required")
