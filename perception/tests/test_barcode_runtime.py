"""
tests/test_barcode_runtime.py — the zxing-cpp surface, against a fake module.

The point of plugins/barcode_runtime.py is that exactly one file knows what
zxing-cpp's API looks like, so that file is where the version differences get
tested: 2.3.0 has no `try_invert`, and a build's real keyword list is read out
of `read_barcodes.__doc__` rather than guessed from a version string.

A real round-trip against the installed library runs too, when there is one —
see test_decodes_a_real_qr_code at the bottom.
"""

from __future__ import annotations

import sys
import types

import pytest

import plugins.barcode_runtime as runtime
from vision_stubs import frame_bytes  # noqa: F401  (installs cv2/ROS stubs)


# ── fakes ────────────────────────────────────────────────────────────────────

class _FakePoint:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _FakePosition:
    def __init__(self):
        self.top_left = _FakePoint(10, 20)
        self.top_right = _FakePoint(30, 20)
        self.bottom_right = _FakePoint(30, 40)
        self.bottom_left = _FakePoint(10, 40)


class _FakeFormat:
    def __init__(self, name):
        self.name = name

    def __or__(self, other):
        return _FakeFormat(f"{self.name}|{other.name}")


class _FakeBarcode:
    def __init__(self, text="hello", fmt="QRCode", valid=True):
        self.text = text
        self.bytes = text.encode()
        self.format = _FakeFormat(fmt)
        self.content_type = _FakeFormat("Text")
        self.position = _FakePosition()
        self.orientation = 0
        self.valid = valid
        self.ec_level = "M"
        self.error = ""


_SIGNATURE_3X = (
    "read_barcodes(image: object, formats: BarcodeFormat = <BarcodeFormat.NONE>, "
    "try_rotate: bool = True, try_downscale: bool = True, try_invert: bool = True, "
    "is_pure: bool = False, max_number_of_symbols: int = 255) -> list\n\n"
    "Read barcodes."
)
# 2.3.0: no try_invert. The whole reason _supported_kwargs exists.
_SIGNATURE_2X = (
    "read_barcodes(image: object, formats: zxingcpp.BarcodeFormat = ..., "
    "try_rotate: bool = True, try_downscale: bool = True, is_pure: bool = False, "
    "max_number_of_symbols: int = 255) -> List[zxingcpp.Result]"
)


def _install_fake_zxing(monkeypatch, *, results=None, signature=_SIGNATURE_3X,
                        strict=True):
    """A stand-in zxingcpp whose read_barcodes records how it was called."""
    calls = []

    accepted = set()
    if signature:
        import re
        accepted = set(re.findall(r"[(,]\s*([a-z_][a-z_0-9]*)\s*[:=]", signature))
        accepted.discard("image")

    def read_barcodes(image, **kwargs):
        if strict:
            unexpected = set(kwargs) - accepted
            if unexpected:
                # What pybind11 does: name every keyword passed, not the one
                # it rejected.
                raise TypeError(
                    "read_barcodes(): incompatible function arguments. "
                    "Invoked with: kwargs: "
                    + ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
                )
        calls.append({"shape": getattr(image, "shape", None), **kwargs})
        return list(results if results is not None else [_FakeBarcode()])

    read_barcodes.__doc__ = signature

    module = types.ModuleType("zxingcpp")
    module.__version__ = "9.9.9-fake"
    module.read_barcodes = read_barcodes

    class BarcodeFormat:
        QRCode = _FakeFormat("QRCode")
        EAN13 = _FakeFormat("EAN13")
        Code128 = _FakeFormat("Code128")

    module.BarcodeFormat = BarcodeFormat
    monkeypatch.setitem(sys.modules, "zxingcpp", module)
    _reset_runtime(monkeypatch)
    return module, calls


def _reset_runtime(monkeypatch):
    """Clear the module-level import/signature caches between tests."""
    monkeypatch.setattr(runtime, "_module", None, raising=False)
    monkeypatch.setattr(runtime, "_import_error", None, raising=False)
    monkeypatch.setattr(runtime, "_kwargs_cache", None, raising=False)


@pytest.fixture(autouse=True)
def _clean_runtime(monkeypatch):
    _reset_runtime(monkeypatch)
    yield
    _reset_runtime(monkeypatch)


# ── decoder availability ─────────────────────────────────────────────────────

def test_missing_library_is_reported_not_raised(monkeypatch):
    """An image built without the wheel must still answer `info`."""
    monkeypatch.setitem(sys.modules, "zxingcpp", None)  # import returns None
    monkeypatch.setattr(
        runtime, "_zxing", lambda: (None, "zxing-cpp is not installed"))
    status = runtime.decoder_status()
    assert status["available"] is False
    assert "not installed" in status["detail"]


def test_decoder_unavailable_raises_a_typed_failure(monkeypatch):
    monkeypatch.setattr(
        runtime, "_zxing", lambda: (None, "zxing-cpp is not installed"))
    with pytest.raises(runtime.DecoderUnavailable) as caught:
        runtime.decode_codes(frame_bytes(), {})
    assert caught.value.as_result()["reason"] == runtime.REASON_DECODER_UNAVAILABLE


def test_decoder_status_lists_formats(monkeypatch):
    _install_fake_zxing(monkeypatch)
    status = runtime.decoder_status()
    assert status["available"] is True
    assert "QRCode" in status["formats"]


# ── keyword compatibility across zxing versions ──────────────────────────────

def test_passes_try_invert_to_a_3x_build(monkeypatch):
    _, calls = _install_fake_zxing(monkeypatch, signature=_SIGNATURE_3X)
    runtime.decode_codes(frame_bytes(), {"try_invert": True})
    assert calls[0]["try_invert"] is True


def test_drops_try_invert_on_a_2x_build(monkeypatch):
    """The jp5.11 case: 2.3.0 has no try_invert and must not be sent one."""
    _, calls = _install_fake_zxing(monkeypatch, signature=_SIGNATURE_2X)
    result = runtime.decode_codes(frame_bytes(), {"try_invert": True})
    assert "try_invert" not in calls[0]
    assert calls[0]["try_rotate"] is True       # the rest still gets through
    assert result["codes"][0]["text"] == "hello"


def test_unreadable_signature_falls_back_to_the_common_keywords(monkeypatch):
    _, calls = _install_fake_zxing(monkeypatch, signature="", strict=False)
    runtime.decode_codes(frame_bytes(), {})
    assert set(calls[0]) - {"shape"} <= runtime._FALLBACK_KWARGS


def test_signature_is_read_once(monkeypatch):
    module, _ = _install_fake_zxing(monkeypatch)
    runtime.decode_codes(frame_bytes(), {})
    # Strip the docstring: a second read would now see nothing and fall back
    # to the smaller keyword set, which is observable.
    module.read_barcodes.__doc__ = ""
    assert "try_invert" in runtime._supported_kwargs(module.read_barcodes)


# ── formats ──────────────────────────────────────────────────────────────────

def test_format_names_are_case_insensitive(monkeypatch):
    _, calls = _install_fake_zxing(monkeypatch)
    runtime.decode_codes(frame_bytes(), {"formats": ["qrcode"]})
    assert calls[0]["formats"].name == "QRCode"


def test_several_formats_are_combined(monkeypatch):
    _, calls = _install_fake_zxing(monkeypatch)
    runtime.decode_codes(frame_bytes(), {"formats": ["EAN13", "Code128"]})
    assert calls[0]["formats"].name == "EAN13|Code128"


def test_empty_formats_means_all(monkeypatch):
    _, calls = _install_fake_zxing(monkeypatch)
    runtime.decode_codes(frame_bytes(), {"formats": []})
    assert "formats" not in calls[0]


def test_unknown_format_names_the_valid_ones(monkeypatch):
    _install_fake_zxing(monkeypatch)
    with pytest.raises(runtime.DecodeError) as caught:
        runtime.decode_codes(frame_bytes(), {"formats": ["QRCde"]})
    detail = caught.value.detail
    assert "QRCde" in detail and "QRCode" in detail


# ── results ──────────────────────────────────────────────────────────────────

def test_result_shape(monkeypatch):
    _install_fake_zxing(monkeypatch)
    result = runtime.decode_codes(frame_bytes(width=200, height=100), {})
    assert result["image"] == {"width": 200, "height": 100}
    code = result["codes"][0]
    assert code["format"] == "QRCode"
    assert code["text"] == "hello"
    assert code["bytes_len"] == 5
    assert code["valid"] is True
    assert code["position"]["top_left"] == [10, 20]
    # Averaged from the corners so the caller does not have to.
    assert code["center"] == [20, 30]


def test_long_text_is_truncated_and_says_so(monkeypatch):
    _install_fake_zxing(monkeypatch, results=[_FakeBarcode(text="x" * 50)])
    code = runtime.decode_codes(frame_bytes(), {"max_text_chars": 10})["codes"][0]
    assert code["text"] == "x" * 10
    assert code["truncated"] is True
    assert code["text_length"] == 50


def test_symbol_count_is_capped(monkeypatch):
    _install_fake_zxing(
        monkeypatch, results=[_FakeBarcode(text=str(n)) for n in range(10)])
    result = runtime.decode_codes(frame_bytes(), {"max_symbols": 3})
    assert len(result["codes"]) == 3


def test_no_codes_is_not_an_error(monkeypatch):
    _install_fake_zxing(monkeypatch, results=[])
    assert runtime.decode_codes(frame_bytes(), {})["codes"] == []


def test_undecodable_bytes_raise_decode_error(monkeypatch):
    _install_fake_zxing(monkeypatch)
    with pytest.raises(runtime.DecodeError) as caught:
        runtime.decode_codes(b"not a picture at all", {})
    assert "not a picture" in caught.value.detail or "truncated" in caught.value.detail


def test_empty_bytes_raise_decode_error(monkeypatch):
    _install_fake_zxing(monkeypatch)
    with pytest.raises(runtime.DecodeError):
        runtime.decode_codes(b"", {})


# ── one real round-trip, where the library exists ────────────────────────────

def _real_zxing():
    try:
        import zxingcpp
    except Exception:  # noqa: BLE001
        return None
    return zxingcpp


@pytest.mark.skipif(_real_zxing() is None, reason="zxing-cpp is not installed")
def test_decodes_a_real_qr_code(monkeypatch):
    """Encode and read back with the installed library.

    Deliberately not OpenCV's QRCodeDetector as the judge — it reports failure
    on structurally valid codes (it is weak at *locating*, not decoding), which
    is how you end up debugging an encoder that was right all along.

    Also no committed fixture image: the writer API is the one part that was
    renamed between 2.x (`write_barcode`) and 3.x (`create_barcode`), so the
    test finds whichever exists and skips if the build has neither.
    """
    import numpy as np

    zxingcpp = _real_zxing()
    _reset_runtime(monkeypatch)
    payload = "motus://barcode-roundtrip/42"

    image = None
    if hasattr(zxingcpp, "create_barcode"):
        barcode = zxingcpp.create_barcode(payload, zxingcpp.BarcodeFormat.QRCode)
        image = np.asarray(barcode.to_image(scale=6))
    elif hasattr(zxingcpp, "write_barcode"):
        image = np.asarray(
            zxingcpp.write_barcode(zxingcpp.BarcodeFormat.QRCode, payload,
                                   width=240, height=240))
    if image is None:
        pytest.skip("this zxing-cpp build has no writer API")

    import cv2
    if getattr(cv2, "__motus_fake__", False):
        pytest.skip("needs real OpenCV to encode the test image")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ok, buffer = cv2.imencode(".png", image)
    assert ok

    result = runtime.decode_codes(buffer.tobytes(), {})
    assert [code["text"] for code in result["codes"]] == [payload]
    assert result["codes"][0]["format"] == "QRCode"


# ── scan modes ───────────────────────────────────────────────────────────────

def test_scan_mode_prefers_the_aggregate_this_build_has(monkeypatch):
    """3.1.1 spells it AllMatrix; 2.2.0 spells it MatrixCodes. Same mode."""
    monkeypatch.setattr(runtime, "supported_formats",
                        lambda: ["AllMatrix", "QRCode", "DataMatrix"])
    assert runtime.resolve_scan_mode("2d") == ["AllMatrix"]
    monkeypatch.setattr(runtime, "supported_formats",
                        lambda: ["MatrixCodes", "QRCode", "DataMatrix"])
    assert runtime.resolve_scan_mode("2d") == ["MatrixCodes"]


def test_scan_mode_falls_back_to_explicit_names(monkeypatch):
    monkeypatch.setattr(runtime, "supported_formats",
                        lambda: ["QRCode", "DataMatrix", "Aztec"])
    assert runtime.resolve_scan_mode("2d") == ["QRCode", "DataMatrix", "Aztec"]


def test_qr_mode_takes_every_qr_variant_present(monkeypatch):
    monkeypatch.setattr(runtime, "supported_formats",
                        lambda: ["QRCode", "MicroQRCode", "EAN13"])
    assert runtime.resolve_scan_mode("qr") == ["QRCode", "MicroQRCode"]


def test_all_means_do_not_narrow(monkeypatch):
    monkeypatch.setattr(runtime, "supported_formats", lambda: ["QRCode"])
    assert runtime.resolve_scan_mode("all") == []
    assert runtime.resolve_scan_mode("") == []
    assert runtime.resolve_scan_mode(None) == []


def test_a_mode_with_no_usable_name_scans_everything(monkeypatch):
    """Narrowing to nothing would decode nothing at all — worse than not
    narrowing, and silent."""
    monkeypatch.setattr(runtime, "supported_formats", lambda: ["QRCode"])
    assert runtime.resolve_scan_mode("1d") == []


def test_unknown_mode_names_the_valid_ones(monkeypatch):
    monkeypatch.setattr(runtime, "supported_formats", lambda: ["QRCode"])
    with pytest.raises(runtime.DecodeError) as caught:
        runtime.resolve_scan_mode("qrcodes")
    assert "qr" in caught.value.detail and "scan_mode" in caught.value.detail
