"""
plugins/barcode_runtime.py — the zxing-cpp surface, and nothing else.

Everything version-sensitive about barcode decoding lives here: the import,
the keyword arguments `read_barcodes` accepts, the names of the format enum,
and the attributes a result object carries. `plugins/barcode.py` above it
deals only in plain dicts, which is what lets the card's lifecycle and
concurrency be tested on a laptop that has no zxing at all (the pattern
`actucore/tests/test_smolvla_provider.py` establishes).

Two version differences this has to absorb, because the two JetPack lines
cannot run the same zxing-cpp:

* **jp6.1 runs the PyPI wheel (3.1.1); jp5.11 runs an in-house 2.2.0 build.**
  zxing-cpp has never published a cp38 aarch64 wheel, and 2.3.0+ does not
  compile on Ubuntu 20.04's gcc 9 — `core/src/WriteBarcode.h` uses a C++20
  `requires` clause. 2.2.0 is the newest PyPI release that builds there.
* **2.2.0 has neither `try_invert` nor `max_number_of_symbols`**, and 3.1.1 has
  `try_invert` but *not* `max_number_of_symbols`. Read off the two real builds,
  not from a changelog. Rather than
  branching on a version string — which says nothing about what a given build
  actually accepts — the keywords are read out of `read_barcodes.__doc__`,
  which both pybind11 (2.x) and nanobind (3.x) generate from the real
  signature. See `_supported_kwargs`.

  Not from the TypeError: pybind11 answers an unexpected keyword with
  "Invoked with: ... kwargs: try_rotate=True, try_invert=True", listing
  *every* keyword passed rather than the one it rejected, so retry-on-error
  cannot tell them apart and would drop all of them.

Channel order is deliberately not converted: zxing reads three-channel arrays
as BGR, matching `cv2.imdecode`'s output, and a barcode is decoded from
luminance anyway, so a red/blue swap could not change the answer.
"""

from __future__ import annotations

import logging
import re
import threading
import time

log = logging.getLogger(__name__)

__all__ = [
    "DecodeError",
    "DecoderUnavailable",
    "REASON_DECODER_UNAVAILABLE",
    "REASON_DECODE_FAILED",
    "DEFAULT_MAX_TEXT_CHARS",
    "DEFAULT_MAX_SYMBOLS",
    "SCAN_MODES",
    "SCAN_MODE_ALL",
    "decode_codes",
    "decoder_status",
    "resolve_scan_mode",
    "supported_formats",
    "reader_options",
]

REASON_DECODER_UNAVAILABLE = "decoder_unavailable"
REASON_DECODE_FAILED = "decode_failed"

# A QR code's payload is attacker-controlled text that this card hands
# straight to an LLM. The cap is a bound on that, not a property of QR: the
# spec allows ~2900 bytes of binary and a code that long is almost always
# either a vCard or something trying to be a prompt.
DEFAULT_MAX_TEXT_CHARS = 2048

# More than this in one frame is a sheet of labels, not a thing being shown
# to a robot; the cap keeps a pathological frame from producing a result the
# LLM cannot read. Enforced in Python because *neither* shipped build exposes
# `max_number_of_symbols` to the reader — 2.2.0 and 3.1.1 both lack it.
DEFAULT_MAX_SYMBOLS = 16


class DecodeError(Exception):
    """An image could not be decoded. Carries the caller-facing detail."""

    reason = REASON_DECODE_FAILED

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail

    def as_result(self) -> dict:
        return {"ok": False, "reason": self.reason, "detail": self.detail}


class DecoderUnavailable(DecodeError):
    reason = REASON_DECODER_UNAVAILABLE


# ── the zxing module, imported once ──────────────────────────────────────────

_import_lock = threading.Lock()
_module = None
_import_error: str | None = None
_kwargs_cache: frozenset | None = None

# What to pass when the signature cannot be read at all. Present in both 2.3.0
# and 3.x, so a build whose docstring was stripped still decodes.
_FALLBACK_KWARGS = frozenset(
    {"formats", "try_rotate", "try_downscale", "is_pure"}
)


def _zxing():
    """`(module, None)` or `(None, detail)`. Imports at most once."""
    global _module, _import_error
    if _module is not None:
        return _module, None
    with _import_lock:
        if _module is not None:
            return _module, None
        if _import_error is not None:
            return None, _import_error
        try:
            import zxingcpp  # noqa: PLC0415 - optional dependency, imported late
        except Exception as error:  # noqa: BLE001 - reported, not raised
            _import_error = (
                f"zxing-cpp is not installed in this image ({error}). "
                "The barcode card cannot decode anything without it; "
                "rebuild the perception image for this JetPack line."
            )
            log.error("[barcode] zxing-cpp unavailable: %s", error)
            return None, _import_error
        _module = zxingcpp
        log.info("[barcode] zxing-cpp %s ready", _version(zxingcpp))
        return _module, None


def _version(module) -> str:
    """The installed version, for `info`.

    Asked of the package metadata first: zxing-cpp 3.1.1 exposes no
    `__version__` on the module at all (checked, not assumed), so reading only
    module attributes reports "unknown" for the build we actually ship.
    """
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return str(version("zxing-cpp"))
    except Exception:  # noqa: BLE001 - older/odd installs fall through
        pass
    for name in ("__version__", "zxing_cpp_version"):
        value = getattr(module, name, None)
        if callable(value):
            try:
                return str(value())
            except Exception:  # noqa: BLE001
                continue
        if value:
            return str(value)
    return "unknown"


def decoder_status() -> dict:
    """What `info` reports about this build's decoding capability."""
    module, detail = _zxing()
    if module is None:
        return {"available": False, "detail": detail}
    return {
        "available": True,
        "library": "zxing-cpp",
        "version": _version(module),
        "formats": supported_formats(),
    }


# Dropped from the reported list because they are not something to ask for:
# `NONE` is zxing's "unset" sentinel (spelled upper-case in 3.1.1, which a
# `"None"` comparison misses). The `All*` aggregates are kept — `AllMatrix`
# for "2D codes only" is one of the more useful values a caller can pass.
_NON_FORMAT_NAMES = frozenset({"none", "any", "invalid"})


def supported_formats() -> list[str]:
    """Format names this build accepts in `formats`, e.g. `["QRCode", ...]`."""
    module, _ = _zxing()
    if module is None:
        return []
    enum = getattr(module, "BarcodeFormat", None)
    if enum is None:
        return []
    names = [
        name for name in dir(enum)
        if name[:1].isupper() and name.lower() not in _NON_FORMAT_NAMES
    ]
    return sorted(names)


# ── scan modes ───────────────────────────────────────────────────────────────
#
# What a person configuring this card actually decides is *what kind of code
# they are pointing it at*, not which of zxing's 51 symbologies to enable. The
# format names are also not portable between the two JetPack lines — 3.1.1
# spells the aggregates `AllMatrix`/`AllLinear`, 2.2.0 spells them
# `MatrixCodes`/`LinearCodes` — so a card configured on one robot would be
# rejected on the other. Both problems go away by making the *mode* the
# setting and resolving it against the names this build actually has.
SCAN_MODE_ALL = "all"
SCAN_MODES = (SCAN_MODE_ALL, "qr", "2d", "1d")

# Tried in order; the first name this build has wins. The explicit lists are
# the fallback for a build with neither aggregate spelling.
_MODE_CANDIDATES = {
    "qr": (["QRCode", "MicroQRCode", "RMQRCode"],),
    "2d": (["AllMatrix"], ["MatrixCodes"],
           ["QRCode", "MicroQRCode", "RMQRCode", "DataMatrix", "Aztec",
            "PDF417", "MaxiCode"]),
    "1d": (["AllLinear"], ["LinearCodes"],
           ["EAN13", "EAN8", "UPCA", "UPCE", "Code128", "Code39", "Code93",
            "Codabar", "ITF", "DataBar", "DataBarExpanded"]),
}


def resolve_scan_mode(mode: str) -> list[str]:
    """`"2d"` → the format names *this* build calls 2D codes.

    `[]` means "do not narrow", which is both the `all` mode and the honest
    answer when no candidate name exists — narrowing to nothing would silently
    decode nothing at all.
    """
    name = str(mode or SCAN_MODE_ALL).strip().lower()
    if name in ("", SCAN_MODE_ALL):
        return []
    if name not in _MODE_CANDIDATES:
        raise DecodeError(
            f"unknown scan_mode {mode!r}. Use one of: " + ", ".join(SCAN_MODES)
        )
    available = set(supported_formats())
    if not available:
        return []
    for candidate in _MODE_CANDIDATES[name]:
        present = [item for item in candidate if item in available]
        if present:
            return present
    log.warning("[barcode] this zxing build has no format names for scan_mode "
                "%r; scanning every type instead", name)
    return []


def _resolve_formats(module, names) -> object | None:
    """`["qrcode", "ean13"]` → a combined BarcodeFormat, or None for "all".

    Case-insensitive, because the useful name for an LLM to produce is
    "qrcode" and the enum spells it "QRCode".
    """
    wanted = [str(name).strip() for name in (names or []) if str(name).strip()]
    if not wanted:
        return None
    enum = getattr(module, "BarcodeFormat", None)
    if enum is None:
        raise DecodeError("this zxing-cpp build exposes no BarcodeFormat enum")
    available = {name.lower(): name for name in dir(enum) if name[:1].isupper()}
    combined = None
    unknown = []
    for name in wanted:
        actual = available.get(name.lower().replace("-", "").replace("_", ""))
        if actual is None:
            unknown.append(name)
            continue
        value = getattr(enum, actual)
        combined = value if combined is None else combined | value
    if unknown:
        raise DecodeError(
            "unknown barcode format(s): " + ", ".join(sorted(unknown))
            + ". Supported: " + ", ".join(supported_formats())
        )
    return combined


def reader_options(cfg: dict) -> dict:
    """The decode options this card takes from config, as plain values."""
    return {
        "formats": list(cfg.get("formats") or []),
        "try_rotate": bool(cfg.get("try_rotate", True)),
        "try_downscale": bool(cfg.get("try_downscale", True)),
        "try_invert": bool(cfg.get("try_invert", True)),
        "is_pure": bool(cfg.get("is_pure", False)),
        "max_symbols": max(1, int(cfg.get("max_symbols", DEFAULT_MAX_SYMBOLS))),
        "max_text_chars": max(
            1, int(cfg.get("max_text_chars", DEFAULT_MAX_TEXT_CHARS))
        ),
    }


# ── decode ───────────────────────────────────────────────────────────────────

def _decode_frame(image_bytes: bytes):
    """JPEG/PNG bytes → a BGR ndarray, or DecodeError."""
    import cv2  # noqa: PLC0415 - matches the other vision plugins
    import numpy as np  # noqa: PLC0415

    if not image_bytes:
        raise DecodeError("image is empty (0 bytes)")
    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8),
                         cv2.IMREAD_COLOR)
    if frame is None:
        raise DecodeError(
            f"cannot decode {len(image_bytes)} bytes as an image — "
            "the file is not a picture, or it is truncated"
        )
    return frame


def _supported_kwargs(reader) -> frozenset:
    """The keyword names `read_barcodes` actually accepts, read once.

    Both binding generators put the real signature on the first line of
    `__doc__`; `try_invert` is in 3.x's and not in 2.3.0's, which is the whole
    reason this exists.
    """
    global _kwargs_cache
    if _kwargs_cache is not None:
        return _kwargs_cache
    doc = (getattr(reader, "__doc__", "") or "").strip()
    signature = doc.split("\n", 1)[0]
    names = set(re.findall(r"[(,]\s*([a-z_][a-z_0-9]*)\s*[:=]", signature))
    names.discard("image")
    _kwargs_cache = frozenset(names) if names else _FALLBACK_KWARGS
    if not names:
        log.warning("[barcode] cannot read read_barcodes' signature; passing "
                    "only %s", ", ".join(sorted(_FALLBACK_KWARGS)))
    return _kwargs_cache


def _read(module, frame, kwargs: dict) -> list:
    """`read_barcodes`, passing only the keywords this build declares."""
    reader = getattr(module, "read_barcodes", None)
    if reader is None:
        raise DecodeError(
            "this zxing-cpp build has no read_barcodes(); it is too old for "
            "this card"
        )
    supported = _supported_kwargs(reader)
    passed = {k: v for k, v in kwargs.items() if k in supported}
    try:
        return list(reader(frame, **passed))
    except TypeError as error:
        raise DecodeError(f"zxing-cpp rejected the decode call: {error}")


def _point(value) -> list[int]:
    return [int(getattr(value, "x", 0)), int(getattr(value, "y", 0))]


def _position(barcode) -> dict:
    position = getattr(barcode, "position", None)
    if position is None:
        return {}
    corners = {}
    for name in ("top_left", "top_right", "bottom_right", "bottom_left"):
        point = getattr(position, name, None)
        if point is not None:
            corners[name] = _point(point)
    return corners


def _enum_name(value) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name)
    text = str(value)
    return text.rsplit(".", 1)[-1]


def _as_dict(barcode, max_text_chars: int) -> dict:
    text = str(getattr(barcode, "text", "") or "")
    truncated = len(text) > max_text_chars
    raw = getattr(barcode, "bytes", None)
    corners = _position(barcode)
    entry = {
        "format": _enum_name(getattr(barcode, "format", "")),
        "text": text[:max_text_chars],
        "valid": bool(getattr(barcode, "valid", True)),
    }
    if truncated:
        # Stated rather than silent: an LLM acting on a URL that lost its tail
        # would act on a different URL.
        entry["truncated"] = True
        entry["text_length"] = len(text)
    if raw is not None:
        try:
            entry["bytes_len"] = len(raw)
        except TypeError:
            pass
    content_type = getattr(barcode, "content_type", None)
    if content_type is not None:
        entry["content_type"] = _enum_name(content_type)
    ec_level = getattr(barcode, "ec_level", "")
    if ec_level:
        entry["ec_level"] = str(ec_level)
    orientation = getattr(barcode, "orientation", None)
    if orientation is not None:
        entry["orientation_deg"] = int(orientation)
    if corners:
        entry["position"] = corners
        xs = [point[0] for point in corners.values()]
        ys = [point[1] for point in corners.values()]
        # The one derived number worth precomputing: "is it left or right of
        # me" is the question a robot asks next, and it should not need the
        # LLM to average four corners to find out.
        entry["center"] = [sum(xs) // len(xs), sum(ys) // len(ys)]
    error = getattr(barcode, "error", None)
    if error:
        entry["error"] = str(error)
    return entry


def decode_codes(image_bytes: bytes, opts: dict) -> dict:
    """Decode every barcode in one image.

    Returns `{"codes": [...], "image": {"width":…, "height":…},
    "elapsed_ms": …}`. Raises `DecodeError` when the image or the decoder is
    the problem — an image that simply holds no barcode is not an error and
    comes back as an empty `codes` list.
    """
    module, detail = _zxing()
    if module is None:
        raise DecoderUnavailable(detail or "zxing-cpp is unavailable")

    options = reader_options(opts)
    formats = _resolve_formats(module, options["formats"])

    started = time.monotonic()
    frame = _decode_frame(image_bytes)
    height, width = frame.shape[0], frame.shape[1]

    kwargs = {
        "try_rotate": options["try_rotate"],
        "try_downscale": options["try_downscale"],
        "try_invert": options["try_invert"],
        "is_pure": options["is_pure"],
        "max_number_of_symbols": options["max_symbols"],
    }
    if formats is not None:
        kwargs["formats"] = formats

    results = _read(module, frame, kwargs)
    codes = [
        _as_dict(barcode, options["max_text_chars"])
        for barcode in results[: options["max_symbols"]]
    ]
    return {
        "codes": codes,
        "image": {"width": int(width), "height": int(height)},
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }
